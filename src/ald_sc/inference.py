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

import json
import math
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

__all__ = ["LSDModel", "Bank", "MidiEvent", "BANK_MODES"]

# A MIDI note event: (midi_note:int, start_seconds:float, duration_seconds:float).
MidiEvent = tuple[int, float, float]

# Bank-generation variant modes (issue #53). Default "canonical" keeps the
# pre-#53 behaviour (n independent DDIM draws — near-identical clips while
# the unconditional DiT contracts, i.e. the "house voice"). The other three
# passed the pre-registered diversity gate on the current checkpoint
# (results/bank_variants.csv) and vary the latent around the canonical draw
# z_bar, exploiting the decoder's NOISE_INJECT-trained tolerance:
#   "jitter"   — z_bar + variety * std(z_bar) * eps_i      (fresh noise)
#   "residual" — z_bar amplified seed residual to relative std `variety`
#   "stopvar"  — same seed, step count swept 0.24..0.98 * steps
BANK_MODES = ("canonical", "jitter", "residual", "stopvar")


def _resolve_seed(seed: int | None) -> int:
    """Return the seed, sampling a non-repeatable one when seed is None."""
    if seed is None:
        return int(time.time()) % (2**31)
    return int(seed)


def _sanitize_slug(name: str, fallback: str) -> str:
    """Allowlist alphanumerics, '-', '_'; everything else collapses to '-'.

    Prevents path traversal via user-supplied directory names.
    """
    return (
        "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        or fallback
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_digests(directory: Path, names: list[str]) -> Path:
    """Write MANIFEST.sha256 over the given files; returns the manifest."""
    lines = [f"{_sha256_file(directory / n)}  {n}" for n in names]
    manifest = directory / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _git_commit_short() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _apply_temperature(z: Tensor, temperature: float) -> Tensor:
    if temperature == 1.0:
        return z
    return z * float(temperature)


def _resample_1d(wave: Tensor, new_length: int) -> Tensor:
    """Resample a 1-D (T,) waveform to a target length (mono, no batch)."""
    dev = wave.device
    wave = wave.unsqueeze(0).cpu()  # Resample on CPU (torchaudio MPS-safe)
    resampler = torchaudio.transforms.Resample(
        orig_freq=wave.shape[-1], new_freq=new_length
    )
    return resampler(wave).squeeze(0).to(dev)


_NOTE_FADE_SECONDS = 0.003


def _edge_fades(wave: Tensor, sr: int) -> Tensor:
    """Raised-cosine power fades (~3 ms) on head and tail of a placed note."""
    n = int(wave.shape[-1])
    fade = max(1, min(int(_NOTE_FADE_SECONDS * sr), n // 2))
    if n < 4:
        return wave
    w = wave.clone()
    ramp = torch.linspace(0.0, math.pi / 2.0, fade, device=wave.device)
    w[..., :fade] = w[..., :fade] * torch.sin(ramp)
    w[..., -fade:] = w[..., -fade:] * torch.cos(ramp)
    return w


def _fit_duration(wave: Tensor, target_len: int, sr: int) -> Tensor:
    """Fit a transposed clip to its time slot without re-pitching.

    Long clips are truncated (with tail fade); short ones are zero-
    padded. Deliberately NOT time-stretched: re-resampling after the
    transposition step is exactly what cancelled pitch shifts in v0.11.
    """
    n = int(wave.shape[-1])
    if n >= target_len:
        return _edge_fades(wave[..., :target_len], sr)
    out = torch.zeros(
        *wave.shape[:-1], int(target_len), dtype=wave.dtype, device=wave.device
    )
    out[..., :n] = _edge_fades(wave, sr)
    return out


def _normalize(audio: Tensor) -> Tensor:
    """DC-block then peak-normalise a (1, T) waveform; no-op if silent.

    The decoder can emit a large DC component (measured +0.4..+0.56 on
    generated banks vs |DC| < 0.002 on the corpus) — a constant offset
    that consumes headroom and renders as a thump/click on playback.
    Removing the mean before peak-normalisation is applied to every
    output path (Modes A/B/C).
    """
    audio = audio - audio.mean()
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
        safe_slug = _sanitize_slug(slug, fallback="model")
        model_dir = root_dir / f"{ts}-{safe_slug}"
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.prior.state_dict(), model_dir / "prior.pt")
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
            "prior_format": "state_dict",
            "torch_version": torch.__version__,
            "git_commit": _git_commit_short(),
            "hyperparameters": dict(hyperparams) if hyperparams else {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        _write_digests(
            model_dir, ["prior.pt", "decoder.pt", "dit.pt", "metadata.json"]
        )
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

    def _bank_latents(
        self,
        n: int,
        steps: int,
        seed: int,
        bank_mode: str,
        bank_variety: float,
    ) -> list[Tensor]:
        """Latents for one bank under the requested variant mode (issue #53).

        All non-canonical modes vary *around the canonical draw* z_bar
        (the DDIM endpoint for ``seed``), where the decoder's
        NOISE_INJECT-trained tolerance guarantees a latent neighbourhood
        that decodes to genuinely distinct waveforms.
        """
        device = next(self.dit.parameters()).device

        if bank_mode == "canonical":
            return [
                sample_ddim(
                    self.dit,
                    self.schedule,
                    batch_size=1,
                    steps=steps,
                    seed=seed + i,
                    device=device,
                )
                for i in range(n)
            ]

        z_bar = sample_ddim(
            self.dit,
            self.schedule,
            batch_size=1,
            steps=steps,
            seed=seed,
            device=device,
        )

        if bank_mode == "jitter":
            sig = z_bar.std()
            latents = []
            for i in range(n):
                gen = torch.Generator(device=device).manual_seed(seed + 100 + i)
                eps = torch.randn(z_bar.shape, device=device, generator=gen)
                latents.append(z_bar + bank_variety * sig * eps)
            return latents

        if bank_mode == "residual":
            draws = [
                sample_ddim(
                    self.dit,
                    self.schedule,
                    batch_size=1,
                    steps=steps,
                    seed=seed + i,
                    device=device,
                )
                for i in range(1, n)
            ]
            if not draws:
                return [z_bar]
            resid = torch.cat([(d - z_bar).flatten().unsqueeze(0) for d in draws])
            resid_std = float(resid.std())
            if resid_std < 1e-4:
                log.warning(
                    "residual_mode_roundoff",
                    resid_std=resid_std,
                    hint="seed residuals are numerical roundoff at this training "
                    "scale (contraction); amplified variants may be unplayable",
                )
            k = bank_variety * float(z_bar.std()) / max(resid_std, 1e-12)
            return [z_bar] + [z_bar + k * (d - z_bar) for d in draws]

        # bank_mode == "stopvar": bank_variety is the stop-time floor as a
        # fraction of `steps` (default 0.5). Floors below ~0.5 produced
        # under-denoised, unplayable clips on the v0.11 checkpoints
        # (perceptual feedback, PR #59) — kept controllable, not encouraged.
        lo = max(1, int(round(steps * bank_variety)))
        hi = max(lo + 1, int(round(steps * 0.98)))
        grid = [lo + (hi - lo) * i // max(n - 1, 1) for i in range(n)]
        return [
            sample_ddim(
                self.dit,
                self.schedule,
                batch_size=1,
                steps=s,
                seed=seed,
                device=device,
            )
            for s in grid
        ]

    @torch.no_grad()
    def generate_sound_bank(
        self,
        n: int = 8,
        steps: int = 50,
        temperature: float = 1.0,
        seed: int | None = None,
        bank_mode: str = "canonical",
        bank_variety: float = 0.5,
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
        bank_mode : str
            Variant strategy (issue #53), one of ``BANK_MODES``:

            - ``"canonical"`` (default) — n independent DDIM draws.
              Unchanged pre-#53 behaviour; at preliminary scale every draw
              contracts to the model's "house voice".
            - ``"jitter"`` — decode ``z_bar + a*std(z_bar)*eps_i``; the
              decoder's latent neighbourhood provides real diversity.
            - ``"residual"`` — amplify the (tiny) seed-to-seed residual
              to relative std ``bank_variety``. At preliminary scale the
              residual is numerical roundoff (contraction) and the
              amplified variants may be harsh; a warning is logged.
            - ``"stopvar"`` — same seed, step count swept from
              ``bank_variety * steps`` to ``0.98 * steps`` (floor
              default 0.5; lower floors trade playability for spread).

        bank_variety : float
            Diversity dial, interpreted per mode: jitter amplitude
            (alpha; stay <= ~0.15 to remain within the decoder's
            NOISE_INJECT noise tolerance), residual relative std, or
            the stop-time floor fraction for stopvar. Defaults to 0.5;
            0 disables variation (canonical output).

        Returns
        -------
        list[Tensor]
            ``n`` peak-normalised waveforms of shape ``(1, T)``.
        """
        if bank_mode not in BANK_MODES:
            raise ValueError(
                f"bank_mode must be one of {BANK_MODES}; got {bank_mode!r}"
            )
        if bank_variety < 0:
            raise ValueError(f"bank_variety must be >= 0; got {bank_variety}")

        s = _resolve_seed(seed)
        latents = self._bank_latents(
            n=n, steps=steps, seed=s, bank_mode=bank_mode, bank_variety=bank_variety
        )
        bank = [
            self._sample_and_decode(
                seed=s + i, steps=steps, temperature=temperature, z_init=z
            )
            for i, z in enumerate(latents)
        ]
        log.info(
            "sound_bank",
            n=n,
            steps=steps,
            temperature=temperature,
            seed=s,
            bank_mode=bank_mode,
            bank_variety=bank_variety,
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
        # Conditioning audio may arrive on CPU (e.g. bank clips staged on CPU
        # before MPS rendering) — move it to the model's device before encoding.
        audio = audio.to(self.prior.U_q.device)

        strength = float(max(0.0, min(1.0, strength)))
        s = _resolve_seed(seed)

        z_cond, _, _ = self.encoder.encode(audio, self.prior)

        variants: list[Tensor] = []
        for i in range(n):
            noise = torch.randn_like(
                z_cond,
                generator=torch.Generator(device=z_cond.device).manual_seed(s + i),
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

        Each MIDI note selects a bank sound **by composer list position**
        (round-robin over ``bank`` before chronological sorting, so timbre
        identity does not become an artifact of start times), pitch-shifts
        it from ``pitch_bank_root`` to the requested note via resampling,
        fits it to the requested duration by truncation or zero-padding
        (never by re-resampling, which would cancel the transposition),
        and places it at the requested start time with ~3 ms edge fades.

        Parameters
        ----------
        events : list of (midi_note, start_seconds, duration_seconds)
            The MIDI sequence. Events with negative/non-finite start or
            non-positive/non-finite duration are skipped with a warning.
        bank : list[Tensor]
            Output sounds (e.g. from ``generate_sound_bank``). Must be
            non-empty; each element is ``(1, T)``.
        pitch_bank_root : int
            MIDI note that the bank sounds are assumed to be centred on
            (used as the pitch-shift reference).
        seed : int or None
            Reserved for stochastic placement; currently unused.

        Returns
        -------
        Tensor (1, T_out)
            Mono render covering all events.

        Raises
        ------
        ValueError
            If no event survives validation.
        """
        if not bank:
            raise ValueError("synthesize_midi requires a non-empty bank")

        sr = self.sample_rate
        valid: list[tuple[int, float, float, int]] = []
        for orig_i, ev in enumerate(events):
            note, start, dur = ev
            if (
                not math.isfinite(start)
                or not math.isfinite(dur)
                or start < 0
                or dur <= 0
            ):
                log.warning(
                    "midi_event_invalid",
                    index=orig_i,
                    note=note,
                    start=start,
                    duration=dur,
                )
                continue
            valid.append((orig_i, float(start), float(dur), int(note)))
        if not valid:
            raise ValueError("no valid MIDI events; check start/duration values")
        valid.sort(key=lambda item: item[1])

        total_seconds = max(start + dur for _, start, dur, _ in valid)
        total_seconds = max(total_seconds, 0.1)
        out = torch.zeros(1, int(total_seconds * sr) + 1, device=bank[0].device)

        s = _resolve_seed(seed)

        for orig_i, start, dur, note in valid:
            src = bank[orig_i % len(bank)].reshape(-1)  # (T,)
            # Transposition only: playback-speed change via one resample.
            semitones = note - pitch_bank_root
            ratio = 2.0 ** (-semitones / 12.0)
            pitched_len = max(1, int(round(src.shape[-1] * ratio)))
            pitched = _resample_1d(src, pitched_len)

            fitted = _fit_duration(pitched, max(1, int(dur * sr)), sr)

            offset = max(0, int(start * sr))
            end = min(out.shape[-1], offset + fitted.shape[-1])
            if end <= offset:
                continue
            out[0, offset:end] += fitted[: end - offset]

        log.info(
            "midi_render",
            n_events=len(valid),
            n_skipped=len(events) - len(valid),
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
        bank_dir = out_dir / "banks" / _sanitize_slug(self.name, fallback="bank")
        bank_dir.mkdir(parents=True, exist_ok=True)

        clip_entries = []
        for i, clip in enumerate(self.clips):
            fname = f"{i:02d}.wav"
            wave = clip.squeeze(0).cpu().numpy()
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
        _write_digests(
            bank_dir, [e["file"] for e in clip_entries] + ["manifest.json"]
        )
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
        bank_mode: str = "canonical",
        bank_variety: float = 0.5,
    ) -> "Bank":
        """Generate a bank (see ``LSDModel.generate_sound_bank``) and wrap it."""
        clips = model.generate_sound_bank(
            n=n,
            steps=steps,
            temperature=temperature,
            seed=seed,
            bank_mode=bank_mode,
            bank_variety=bank_variety,
        )
        return cls(model=model, clips=clips, name=name)
