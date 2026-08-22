"""Audio data loading for ALD-SC sound generation.

Provides datasets for ESC-50 environmental sounds, generic audio folders,
synthetic test waveforms, and synthetic music-like clips. All audio is
loaded as 24 kHz mono and padded/cropped to a fixed length.

This module must not add model logic (per AGENTS.md §11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import soundfile
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ald_sc.stitching import equal_power_overlap_add

__all__ = [
    "ToyAudioDataset",
    "AudioFolderDataset",
    "Esc50Dataset",
    "MusicSynthDataset",
    "PairedSegmentDataset",
    "build_audio_dataloader",
    "load_audio_clip",
]

SAMPLE_RATE = 24000


def load_audio_clip(
    path: str | Path,
    target_sr: int = SAMPLE_RATE,
    target_length: int | None = None,
) -> Tensor:
    """Load an audio file, resample to target_sr, convert to mono, and
    pad/crop to target_length.

    Parameters
    ----------
    path : str or Path
        Path to audio file.
    target_sr : int
        Target sample rate (default 24000 for EnCodec).
    target_length : int, optional
        Target length in samples. If None, return full clip.
        If provided, pad with zeros or crop to exact length.

    Returns
    -------
    Tensor (1, target_length) or (1, T)
        Audio waveform in mono at target_sr.
    """
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        # Fallback for environments without torchcodec / ffmpeg backend
        data, sr = soundfile.read(str(path), dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            # soundfile returns (samples, channels); transpose to (channels, samples)
            waveform = waveform.T

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Pad or crop to target length
    if target_length is not None:
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]

    # Normalize to [-1, 1] (peak normalization)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform


class ToyAudioDataset(Dataset):
    """Synthetic audio dataset for testing and small experiments.

    Generates random waveforms (sine waves, noise, or mixtures) with
    optional structure. No external data required.

    Parameters
    ----------
    num_samples : int
        Number of clips to generate.
    audio_length : int
        Length of each clip in samples (default 24000 = 1s @ 24kHz).
    sample_rate : int
        Sample rate (default 24000).
    """

    def __init__(
        self,
        num_samples: int = 100,
        audio_length: int = 24000,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tensor:
        torch.manual_seed(3407 + index)
        t = torch.arange(self.audio_length, dtype=torch.float32) / self.sample_rate

        # Random frequency sine wave + noise
        freq = 100.0 + 400.0 * (index % 10) / 10.0
        sine = torch.sin(2 * torch.pi * freq * t)
        noise = 0.1 * torch.randn(self.audio_length)
        waveform = sine + noise

        # Normalize
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform.unsqueeze(0)  # (1, T)


class MusicSynthDataset(Dataset):
    """Synthetic music-like dataset for training and demos.

    Generates deterministic, harmonic clips that are richer than simple
    sine waves: each clip combines a fundamental tone, several harmonics,
    a simple amplitude envelope, rhythmic tremolo, and occasional noise
    bursts. No external data is required.

    This is intended as a drop-in replacement for real music corpora when
    demonstrating the full ALD-SC pipeline with the real EnCodec encoder.

    Parameters
    ----------
    num_samples : int
        Number of clips to generate.
    audio_length : int
        Length of each clip in samples (default 120000 = 5s @ 24kHz).
    sample_rate : int
        Sample rate (default 24000).
    seed : int
        Random seed for reproducibility.
    num_harmonics : int
        Number of harmonics above the fundamental.
    """

    def __init__(
        self,
        num_samples: int = 100,
        audio_length: int = 120000,
        sample_rate: int = SAMPLE_RATE,
        seed: int = 3407,
        num_harmonics: int = 4,
    ) -> None:
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.seed = seed
        self.num_harmonics = num_harmonics

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tensor:
        # Reproducible per-index generation
        rng = torch.Generator()
        rng.manual_seed(self.seed + index)
        t = torch.arange(self.audio_length, dtype=torch.float32) / self.sample_rate

        # Choose a base note from a pentatonic-ish scale (110-880 Hz)
        base_freqs = torch.tensor(
            [
                110.0,
                130.81,
                146.83,
                164.81,
                196.0,
                220.0,
                261.63,
                293.66,
                329.63,
                392.0,
                440.0,
                523.25,
            ],
        )
        base_idx = torch.randint(0, len(base_freqs), (1,), generator=rng).item()
        base_freq = base_freqs[int(base_idx)].item()

        # Fundamental + harmonics with decaying amplitudes
        waveform = torch.zeros(self.audio_length)
        harmonic_amplitudes = [1.0]
        for h in range(1, self.num_harmonics + 1):
            harmonic_amplitudes.append(0.6**h)

        for h, amp in enumerate(harmonic_amplitudes):
            freq = base_freq * (h + 1)
            phase = torch.rand(1, generator=rng).item() * 2 * torch.pi
            waveform += amp * torch.sin(2 * torch.pi * freq * t + phase)

        # Add a simple ADSR-like envelope (attack, decay, sustain, release)
        attack_samples = min(int(0.05 * self.sample_rate), self.audio_length // 4)
        release_samples = min(int(0.1 * self.sample_rate), self.audio_length // 4)
        envelope = torch.ones(self.audio_length)
        if attack_samples > 1:
            envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)
        if release_samples > 1:
            envelope[-release_samples:] = torch.linspace(1, 0, release_samples)

        waveform = waveform * envelope

        # Rhythmic tremolo synchronized to a beats-per-second rate
        bps = 2.0 + 2.0 * torch.rand(1, generator=rng).item()  # 2-4 beats per second
        tremolo = 0.85 + 0.15 * torch.sin(2 * torch.pi * bps * t)
        waveform = waveform * tremolo

        # Sparse noise bursts (percussive element)
        num_bursts = int(1 + torch.rand(1, generator=rng).item() * 3)
        for _ in range(num_bursts):
            burst_center = int(torch.rand(1, generator=rng).item() * self.audio_length)
            burst_width = int(0.02 * self.sample_rate)
            start = max(0, burst_center - burst_width // 2)
            end = min(self.audio_length, burst_center + burst_width // 2)
            if end > start:
                noise = torch.randn(end - start, generator=rng)
                waveform[start:end] += 0.2 * noise

        # Normalize to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform.unsqueeze(0)  # (1, T)


class AudioFolderDataset(Dataset):
    """Generic audio folder dataset.

    Loads all audio files from a directory (optionally filtered by
    extension), resampling and normalizing each to a fixed length.

    Parameters
    ----------
    root : str or Path
        Directory containing audio files (searched recursively).
    audio_length : int
        Target length in samples (default 120000 = 5s @ 24kHz).
    sample_rate : int
        Target sample rate (default 24000).
    extensions : tuple[str, ...]
        Audio file extensions to include.
    """

    def __init__(
        self,
        root: str | Path,
        audio_length: int = 120000,
        sample_rate: int = SAMPLE_RATE,
        extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
    ) -> None:
        self.root = Path(root)
        self.audio_length = audio_length
        self.sample_rate = sample_rate

        self.files: list[Path] = []
        for ext in extensions:
            self.files.extend(self.root.rglob(f"*{ext}"))
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        return load_audio_clip(
            self.files[index],
            target_sr=self.sample_rate,
            target_length=self.audio_length,
        )


class Esc50Dataset(Dataset):
    """ESC-50 environmental sound classification dataset.

    ESC-50 is a collection of 2000 5-second environmental audio clips
    (50 classes, 40 clips per class). This loader expects the standard
    ESC-50 directory layout:

        root/
        ├── audio/
        │   ├── 1-100032-A-0.wav
        │   ├── 1-100038-A-14.wav
        │   └── ...
        └── meta/
            └── esc50.csv

    Parameters
    ----------
    root : str or Path
        Root directory of ESC-50.
    audio_length : int
        Target length in samples (default 120000 = 5s @ 24kHz).
    sample_rate : int
        Target sample rate (default 24000; ESC-50 is 44.1 kHz).
    """

    def __init__(
        self,
        root: str | Path,
        audio_length: int = 120000,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.root = Path(root)
        self.audio_length = audio_length
        self.sample_rate = sample_rate

        audio_dir = self.root / "audio"
        self.files = sorted(audio_dir.glob("*.wav"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        return load_audio_clip(
            self.files[index],
            target_sr=self.sample_rate,
            target_length=self.audio_length,
        )


class _SizedDataset(Protocol):
    """A map-style dataset whose length is known."""

    def __getitem__(self, index: int) -> Tensor: ...

    def __len__(self) -> int: ...


class PairedSegmentDataset(Dataset):
    """Virtual k-times-longer segments from a base dataset of short clips.

    Item ``i`` joins base items ``[k·i, k·(i+1))`` with equal-power
    crossfades in the waveform domain (see ``ald_sc.stitching``), so an
    archive of short clips can feed long-form training without zero
    padding. Trailing clips are dropped when they do not fill a segment.

    Parameters
    ----------
    base : Dataset
        Sized dataset (implements ``__len__``) yielding (1, T) waveforms.
    crossfade_samples : int
        Overlap between consecutive joined clips, clamped to fit.
    clips_per_segment : int
        Number of base clips chained into one segment (2 = the original
        pairing; 5 × 2 s one-shots ≈ a 10 s stem).
    """

    def __init__(
        self,
        base: _SizedDataset,
        crossfade_samples: int = 480,
        clips_per_segment: int = 2,
    ) -> None:
        if crossfade_samples < 0:
            raise ValueError(f"crossfade_samples must be >= 0; got {crossfade_samples}")
        if clips_per_segment < 2:
            raise ValueError(f"clips_per_segment must be >= 2; got {clips_per_segment}")
        self.base = base
        self.crossfade_samples = int(crossfade_samples)
        self.clips_per_segment = int(clips_per_segment)

    def __len__(self) -> int:
        return len(self.base) // self.clips_per_segment

    def __getitem__(self, index: int) -> Tensor:
        start = index * self.clips_per_segment
        group = [self.base[start + i] for i in range(self.clips_per_segment)]
        overlap = min([self.crossfade_samples] + [int(g.shape[-1]) - 1 for g in group])
        overlap = max(overlap, 0)
        return equal_power_overlap_add(group, overlap=overlap)


def build_audio_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from an audio dataset.

    Parameters
    ----------
    dataset : Dataset
    batch_size : int
    shuffle : bool
    num_workers : int

    Returns
    -------
    DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
