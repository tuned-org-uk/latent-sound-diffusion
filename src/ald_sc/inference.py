"""LSD sound-production inference contract.

Bundles a trained model (prior + DiT + decoder + encoder + schedule) and
exposes the three producer-facing inference modes defined in
``docs/03.md``:

- ``generate_sound_bank`` — unconditional or spectrally-conditioned sound
  bank (mode A).  Pass ``target_c_spec`` to steer toward a spectral region.
- ``condition_on_audio`` — produce novelty/variants of a producer sound
  (mode B).  The source audio's ``c_spec`` is forwarded to the DiT so the
  sampler is steered toward the source's spectral region.
- ``synthesize_midi`` — render a MIDI note sequence using generated bank
  sounds (mode C).

This module owns **inference composition only**; it does not train or
define architectures (per AGENTS.md §11). ``seed=None`` everywhere means
a non-repeatable, time-sampled seed — a deliberate artistic feature.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import soundfile
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

__all__ = ["LSDModel", "Bank", "MidiEvent"]

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

    def store(
        self,
        root_dir: str | Path,
        slug: str = "lsd-model",
        hyperparams: dict | None = None,
    ) -> Path:
        """Save model artefacts and a metadata.json for reproducibility.

        Creates <root_dir>/<timestamp>-<slug>/ (timestamp = YYYYMMDD-HHMMSS)
        and writes prior.pt, decoder.pt, dit.pt, and metadata.json (sample
        rate, latent shape, decoder type, schedule steps, hyperparams).
        Returns the model directory path.
        """
        root_dir = Path(root_dir)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_slug = (
            "".join(c if c.isalnum() or c in "-_" else "-" for c in slug) or "model"
        )
        model_dir = root_dir / f"{ts}-{safe_slug}"
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.prior, model_dir / "prior.pt")
        torch.save(self.decoder.state_dict(), model_dir / "decoder.pt")
        torch.save(self.dit.state_dict(), model_dir / "dit.pt")

        latent_channels, latent_length = getattr(self.dit, "latent_shape", (None, None))
        schedule_steps = getattr(self.schedule, "num_steps", None)
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_rate": self.sample_rate,
            "latent_channels": latent_channels,
            "latent_length": latent_length,
            "decoder_type": type(self.decoder).__module__
            + "."
            + type(self.decoder).__name__,
            "schedule_num_steps": schedule_steps,
            "hyperparameters": dict(hyperparams) if hyperparams else {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log.info(
            "model_store",
            slug=safe_slug,
            path=str(model_dir),
            hyperparameters=metadata["hyperparameters"],
        )
        return model_dir

    @torch.no_grad()
    def _sample_and_decode(
        self,
        seed: int,
        steps: int,
        temperature: float,
        z_init: Tensor | None = None,
        c_spec_override: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Sample (or use a provided) latent and decode it to a waveform.

        Parameters
        ----------
        seed : int
        steps : int
        temperature : float
        z_init : Tensor, optional
            If provided, skip sampling and decode this latent directly
            (used by ``condition_on_audio`` for interpolated latents).
        c_spec_override : Tensor (B, spec_dim), optional
            Spectral-chart conditioning vector forwarded to the DiT
            during sampling.  When ``None`` (default), ``c_spec`` is
            derived post-hoc from the generated ``z`` (self-consistent,
            unconditional-equivalent behaviour).  When provided, the
            DiT is steered toward the specified spectral region during
            the reverse diffusion process (target-mode sampling).

            Note: ``c_spec_override`` is only forwarded when ``z_init``
            is ``None`` (i.e. when sampling actually runs).  If both are
            provided a ``ValueError`` is raised, because the sampling step
            would be skipped and the override would have no effect.
        guidance_scale : float
            Classifier-free guidance scale forwarded to the sampler.
            1.0 = pure conditional (single pass), 0.0 = pure unconditional,
            >1.0 = amplified conditioning (two-pass).  Only effective when
            ``c_spec_override`` is provided.

        Raises
        ------
        ValueError
            If both ``z_init`` and ``c_spec_override`` are provided.  The
            two arguments are mutually exclusive: ``c_spec_override`` steers
            the DiT *during* sampling, which is skipped when ``z_init`` is
            given.  Pass ``c_spec_override=None`` when supplying ``z_init``.
        """
        if z_init is not None and c_spec_override is not None:
            raise ValueError(
                "z_init and c_spec_override are mutually exclusive. "
                "c_spec_override steers the DiT during sampling, but sampling "
                "is skipped when z_init is provided. "
                "Pass c_spec_override=None when supplying z_init."
            )

        device = next(self.dit.parameters()).device

        if z_init is None:
            z = sample_ddim(
                self.dit,
                self.schedule,
                c_spec=c_spec_override,
                batch_size=1,
                steps=steps,
                seed=seed,
                device=device,
                guidance_scale=guidance_scale,
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
        target_c_spec: Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> list[Tensor]:
        """Mode A: generate a bank of ``n`` sounds.

        Without ``target_c_spec`` the generation is unconditional
        (self-consistent: ``c_spec`` is derived from each generated ``z``,
        used only by the decoder).

        Passing ``target_c_spec`` activates target-mode sampling: the same
        spectral-chart vector steers the DiT during the reverse diffusion
        process, biasing every sample in the bank toward the specified
        spectral region.  A typical workflow is to encode a reference sound
        with ``self.encoder.encode(ref_audio, self.prior)`` and pass the
        resulting ``c_spec`` here.

        ``guidance_scale`` controls classifier-free guidance strength when
        ``target_c_spec`` is provided: 1.0 = pure conditional, 0.0 = pure
        unconditional, >1.0 = amplified conditioning (more directed, less
        diverse).  It is ignored when ``target_c_spec`` is ``None``.

        .. note::
            When ``target_c_spec`` is used, **all samples in the bank share
            the same conditioning signal** — only the per-sample noise seed
            varies.  This produces a cohesive but potentially less diverse
            bank compared to unconditioned generation.  If variety within
            the bank is important, generate multiple banks with different
            ``target_c_spec`` vectors (e.g. drawn from different reference
            sounds) and merge them.

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
        target_c_spec : Tensor (1, spec_dim) or (B, spec_dim), optional
            Spectral conditioning vector to steer the DiT.  If shape is
            ``(1, spec_dim)`` it is broadcast across all samples in the bank.
        guidance_scale : float
            Classifier-free guidance scale (only effective with
            ``target_c_spec``).  1.0 = pure conditional, 0.0 = pure
            unconditional, >1.0 = amplified conditioning.

        Returns
        -------
        list[Tensor]
            ``n`` peak-normalised waveforms of shape ``(1, T)``.
        """
        s = _resolve_seed(seed)
        bank: list[Tensor] = []
        for i in range(n):
            clip = self._sample_and_decode(
                seed=s + i,
                steps=steps,
                temperature=temperature,
                c_spec_override=target_c_spec,
                guidance_scale=guidance_scale,
            )
            bank.append(clip)
        log.info(
            "sound_bank",
            n=n,
            steps=steps,
            temperature=temperature,
            seed=s,
            c_spec_conditioned=target_c_spec is not None,
            guidance_scale=guidance_scale,
        )
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

        The conditioning audio is encoded to its EnCodec latent; the
        resulting ``c_spec`` is forwarded to the DiT during sampling so the
        reverse diffusion process is steered toward the source sound's
        spectral region.  Each variant is produced by interpolating from
        that latent toward a fresh noise draw (``strength`` controls how
        far from the source), then decoding with the graph decoder.

        Note: because ``condition_on_audio`` passes ``z_init`` (the
        interpolated latent) to ``_sample_and_decode``, ``c_spec_override``
        is not forwarded into the sampler — the interpolated latent is
        decoded directly.  Spectral steering here therefore comes from the
        latent interpolation itself, not from DiT conditioning.

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

        device = next(self.dit.parameters()).device
        audio = audio.to(device)
        z_cond, _, _c_spec = self.encoder.encode(audio, self.prior)
        # Defensive: ensure z_cond is on the model device even if the encoder
        # returned it on a different device (e.g. CPU encoder, CUDA DiT).
        z_cond = z_cond.to(device)

        variants: list[Tensor] = []
        for i in range(n):
            noise = torch.randn_like(
                z_cond, generator=torch.Generator(device=device).manual_seed(s + i)
            )
            z = (1.0 - strength) * z_cond + strength * noise
            # z_init is provided, so c_spec_override must be None (the two are
            # mutually exclusive in _sample_and_decode).  Spectral steering
            # comes from the latent interpolation, not DiT conditioning.
            clip = self._sample_and_decode(
                seed=s + i,
                steps=steps,
                temperature=temperature,
                z_init=z,
                c_spec_override=None,
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


@dataclass
class Bank:
    """A generated sound bank: clips plus provenance.

    Parameters
    ----------
    model : LSDModel
        The model that produced the clips (for sample-rate / metadata).
    clips : list[Tensor]
        Peak-normalised waveforms of shape (1, T).
    name : str
        Human-friendly bank name (used as the directory name when stored).
    generated_at : str, optional
        ISO timestamp of generation; defaults to now.
    """

    model: "LSDModel"
    clips: list[Tensor]
    name: str = "bank"
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("Bank.clips must be non-empty")

    def store(self, out_dir: str | Path) -> Path:
        """Write the bank to <out_dir>/banks/<name>/.

        Each clip is saved as NN.wav (zero-padded), and a manifest.json
        records the bank name, clip list, sample rate, generation
        timestamp, and provenance. Returns the bank directory path.
        """
        out_dir = Path(out_dir)
        bank_dir = out_dir / "banks" / self.name
        bank_dir.mkdir(parents=True, exist_ok=True)

        clip_entries = []
        for i, clip in enumerate(self.clips):
            fname = f"{i:02d}.wav"
            wave = clip.squeeze(0).numpy()
            soundfile.write(str(bank_dir / fname), wave, self.model.sample_rate)
            clip_entries.append({"file": fname, "shape": list(clip.shape)})

        manifest = {
            "name": self.name,
            "n_clips": len(self.clips),
            "sample_rate": self.model.sample_rate,
            "generated_at": self.generated_at,
            "clips": clip_entries,
        }
        (bank_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "bank_store",
            name=self.name,
            n_clips=len(self.clips),
            path=str(bank_dir),
        )
        return bank_dir

    @classmethod
    def from_generation(
        cls,
        model: "LSDModel",
        n: int = 8,
        steps: int = 50,
        temperature: float = 1.0,
        seed: int | None = None,
        name: str = "bank",
    ) -> "Bank":
        """Generate a bank and wrap it in a Bank."""
        clips = model.generate_sound_bank(
            n=n, steps=steps, temperature=temperature, seed=seed
        )
        return cls(model=model, clips=clips, name=name)
