"""LSD sound-production inference contract.

Bundles a trained model (prior + DiT + decoder + encoder + schedule) and
exposes the three producer-facing inference modes defined in
``docs/03.md``:

- ``generate_sound_bank`` — unconditional baseline sound bank (mode A).
- ``condition_on_audio`` — produce novelty/variants of a producer sound
  (mode B).
- ``synthesize_midi`` — render a MIDI note sequence using generated bank
  sounds (mode C).

This module owns **inference composition only**; it does not train or
define architectures (per AGENTS.md §11). ``seed=None`` everywhere means
a non-repeatable, time-sampled seed — a deliberate artistic feature.
"""

from __future__ import annotations

import time

import structlog

import torch
import torchaudio
from torch import Tensor, nn

from ald_sc._logging import configure_logging
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.audio_codec import BaselineAudioDecoder, EnCodecEncoder
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_ddim

configure_logging()

log = structlog.get_logger("ald_sc.inference")

__all__ = ["LSDModel", "MidiEvent"]

# A MIDI note event: (midi_note:int, start_seconds:float, duration_seconds:float).
MidiEvent = tuple[int, float, float]


def _resolve_seed(seed: int | None) -> int:
    """Return the seed, sampling a non-repeatable one when seed is None."""
    if seed is None:
        return int(time.time()) % (2**31)
    return int(seed)


def _apply_temperature(z: Tensor, temperature: float) -> Tensor:
    if temperature == 1.0:
        return z
    return z * float(temperature)


def _resample_1d(wave: Tensor, new_length: int) -> Tensor:
    """Resample a 1-D (T,) waveform to a target length (mono, no batch)."""
    wave = wave.unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(
        orig_freq=wave.shape[-1], new_freq=new_length
    )
    return resampler(wave).squeeze(0)


def _normalize(audio: Tensor) -> Tensor:
    """Peak-normalise a (1, T) waveform; no-op if silent."""
    peak = audio.abs().max()
    if peak > 0:
        return audio / peak
    return audio


class LSDModel:
    """Inference-time bundle of a trained LSD model.

    Parameters
    ----------
    prior : ArrowSpacePrior
        Frozen ArrowSpace prior (provides the spectral chart).
    dit : nn.Module
        Trained 1-D DiT denoiser (``latent_shape`` attribute required).
    decoder : nn.Module
        Trained graph or baseline audio decoder.
    encoder : EnCodecEncoder or compatible
        Frozen encoder used for audio-conditioned variants (mode B).
        Must expose ``encode(x, prior) -> (z, A, c_spec)``.
    schedule : CosineSchedule
        Noise schedule for DDIM sampling.
    sample_rate : int
        Output sample rate (default 24000 for EnCodec 24kHz).
    """

    def __init__(
        self,
        prior: ArrowSpacePrior,
        dit: nn.Module,
        decoder: nn.Module,
        encoder: EnCodecEncoder,
        schedule: CosineSchedule,
        sample_rate: int = 24000,
    ) -> None:
        self.prior = prior
        self.dit = dit
        self.decoder = decoder
        self.encoder = encoder
        self.schedule = schedule
        self.sample_rate = sample_rate
        self._is_baseline = isinstance(decoder, BaselineAudioDecoder)

    @torch.no_grad()
    def _sample_and_decode(
        self,
        seed: int,
        steps: int,
        temperature: float,
        z_init: Tensor | None = None,
    ) -> Tensor:
        """Sample (or use a provided) latent and decode it to a waveform."""
        device = next(self.dit.parameters()).device

        if z_init is None:
            z = sample_ddim(
                self.dit,
                self.schedule,
                batch_size=1,
                steps=steps,
                seed=seed,
                device=device,
            )
        else:
            z = z_init.to(device)

        z = _apply_temperature(z, temperature)
        a = z.mean(dim=2)
        c_spec = self.prior.chart_energy_descriptor(a)

        if self._is_baseline:
            audio = self.decoder(z)
        else:
            audio = self.decoder(z, c_spec)
        # Decoder returns (B, 1, T); collapse the mono channel to (1, T).
        audio = audio.squeeze(1)
        return _normalize(audio.clamp(-1, 1))

    @torch.no_grad()
    def generate_sound_bank(
        self,
        n: int = 8,
        steps: int = 50,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[Tensor]:
        """Mode A: generate a bank of ``n`` unconditional baseline sounds.

        Parameters
        ----------
        n : int
            Number of sounds to generate.
        steps : int
            DDIM sampling steps.
        temperature : float
            Noise scaling (lower = more conservative).
        seed : int or None
            Reproducible seed, or ``None`` for a non-repeatable run.

        Returns
        -------
        list[Tensor]
            ``n`` peak-normalised waveforms of shape ``(1, T)``.
        """
        s = _resolve_seed(seed)
        bank: list[Tensor] = []
        for i in range(n):
            clip = self._sample_and_decode(
                seed=s + i, steps=steps, temperature=temperature
            )
            bank.append(clip)
        log.info("sound_bank", n=n, steps=steps, temperature=temperature, seed=s)
        return bank

    @torch.no_grad()
    def condition_on_audio(
        self,
        audio: Tensor,
        n: int = 4,
        steps: int = 50,
        strength: float = 0.5,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> list[Tensor]:
        """Mode B: generate ``n`` novelty variants of a producer sound.

        The conditioning audio is encoded to its EnCodec latent; each
        variant is produced by interpolating from that latent toward a
        fresh noise draw (``strength`` controls how far from the source),
        then decoding with the graph decoder.

        Parameters
        ----------
        audio : Tensor (1, 1, T_audio) or (1, T_audio)
            The conditioning sound (must be encodable by ``self.encoder``).
        n : int
            Number of variants.
        steps : int
            DDIM sampling steps.
        strength : float
            0 < strength <= 1. Larger values push further from the source
            (more novelty); smaller values stay closer.
        temperature : float
            Noise scaling applied after interpolation.
        seed : int or None
            Reproducible seed, or ``None`` for a non-repeatable run.

        Returns
        -------
        list[Tensor]
            ``n`` peak-normalised variant waveforms of shape ``(1, T)``.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if audio.dim() != 3 or audio.shape[1] != 1:
            raise ValueError(
                f"audio must be (1, 1, T) or (1, T); got {tuple(audio.shape)}"
            )

        strength = float(max(0.0, min(1.0, strength)))
        s = _resolve_seed(seed)

        z_cond, _, _ = self.encoder.encode(audio, self.prior)

        variants: list[Tensor] = []
        for i in range(n):
            noise = torch.randn_like(
                z_cond, generator=torch.Generator().manual_seed(s + i)
            )
            z = (1.0 - strength) * z_cond + strength * noise
            clip = self._sample_and_decode(
                seed=s + i, steps=steps, temperature=temperature, z_init=z
            )
            variants.append(clip)
        log.info(
            "condition_variants",
            n=n,
            steps=steps,
            strength=strength,
            seed=s,
        )
        return variants

    @torch.no_grad()
    def synthesize_midi(
        self,
        events: list[MidiEvent],
        bank: list[Tensor],
        pitch_bank_root: int = 60,
        seed: int | None = None,
    ) -> Tensor:
        """Mode C: render a MIDI note sequence using generated bank sounds.

        Each MIDI note selects a bank sound (round-robin over ``bank``),
        pitch-shifts it from ``pitch_bank_root`` to the requested note via
        resampling, time-scales it to the requested duration, and places it
        at the requested start time in a mono output buffer.

        Parameters
        ----------
        events : list of (midi_note, start_seconds, duration_seconds)
            The MIDI sequence.
        bank : list[Tensor]
            Output sounds (e.g. from ``generate_sound_bank``). Must be
            non-empty; each element is ``(1, T)``.
        pitch_bank_root : int
            MIDI note that the bank sounds are assumed to be centred on
            (used as the pitch-shift reference).
        seed : int or None
            Seed for round-robin / jitter; ``None`` for non-repeatable.

        Returns
        -------
        Tensor (1, T_out)
            Mono render covering all events.
        """
        if not bank:
            raise ValueError("synthesize_midi requires a non-empty bank")

        sr = self.sample_rate
        events = sorted(events, key=lambda e: e[1])

        total_seconds = 0.0
        for _, start, dur in events:
            total_seconds = max(total_seconds, start + dur)
        total_seconds = max(total_seconds, 0.1)
        out = torch.zeros(1, int(total_seconds * sr) + 1)

        s = _resolve_seed(seed)

        for i, (note, start, dur) in enumerate(events):
            src = bank[i % len(bank)].squeeze(0)  # (T,)
            # Pitch shift via resample ratio (semitones from pitch_bank_root).
            semitones = note - pitch_bank_root
            ratio = 2.0 ** (-semitones / 12.0)
            new_len = max(1, int(src.shape[-1] * ratio))

            pitched = _resample_1d(src, new_len)

            # Time-scale to requested duration.
            target_len = max(1, int(dur * sr))
            if pitched.shape[-1] != target_len:
                pitched = _resample_1d(pitched, target_len)

            offset = int(start * sr)
            end = min(out.shape[-1], offset + pitched.shape[-1])
            out[0, offset:end] += pitched[: end - offset]

        log.info(
            "midi_render",
            n_events=len(events),
            bank_size=len(bank),
            pitch_bank_root=pitch_bank_root,
            seed=s,
        )
        return _normalize(out)
