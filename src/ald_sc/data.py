"""Audio data loading for ALD-SC sound generation.

Provides datasets for ESC-50 environmental sounds, generic audio folders,
and synthetic test waveforms. All audio is loaded as 24 kHz mono and
padded/cropped to a fixed length.

This module must not add model logic (per AGENTS.md §11).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "ToyAudioDataset",
    "AudioFolderDataset",
    "Esc50Dataset",
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
    waveform, sr = torchaudio.load(str(path))

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

    def __getitem__(self, idx: int) -> Tensor:
        torch.manual_seed(3407 + idx)
        t = torch.arange(self.audio_length, dtype=torch.float32) / self.sample_rate

        # Random frequency sine wave + noise
        freq = 100.0 + 400.0 * (idx % 10) / 10.0
        sine = torch.sin(2 * torch.pi * freq * t)
        noise = 0.1 * torch.randn(self.audio_length)
        waveform = sine + noise

        # Normalize
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

    def __getitem__(self, idx: int) -> Tensor:
        return load_audio_clip(
            self.files[idx],
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

    def __getitem__(self, idx: int) -> Tensor:
        return load_audio_clip(
            self.files[idx],
            target_sr=self.sample_rate,
            target_length=self.audio_length,
        )


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
