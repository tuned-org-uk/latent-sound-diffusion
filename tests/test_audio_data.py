"""Tests for audio data loading (no external data required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import soundfile
import torch
from ald_sc.data import (
    AudioFolderDataset,
    MusicSynthDataset,
    ToyAudioDataset,
    build_audio_dataloader,
    load_audio_clip,
)


class TestToyAudioDataset:
    def test_length(self) -> None:
        ds = ToyAudioDataset(num_samples=16, audio_length=24000)
        assert len(ds) == 16

    def test_item_shape(self) -> None:
        ds = ToyAudioDataset(num_samples=4, audio_length=48000)
        item = ds[0]
        assert item.shape == (1, 48000)

    def test_normalized(self) -> None:
        """Waveforms should be normalized to [-1, 1]."""
        ds = ToyAudioDataset(num_samples=4, audio_length=24000)
        for i in range(4):
            item = ds[i]
            assert item.abs().max() <= 1.0 + 1e-6

    def test_determinism(self) -> None:
        """Same index should produce the same waveform."""
        ds = ToyAudioDataset(num_samples=4, audio_length=24000)
        assert torch.allclose(ds[0], ds[0])

    def test_different_indices_different_audio(self) -> None:
        ds = ToyAudioDataset(num_samples=4, audio_length=24000)
        assert not torch.allclose(ds[0], ds[1])


class TestMusicSynthDataset:
    def test_length(self) -> None:
        ds = MusicSynthDataset(num_samples=8, audio_length=24000)
        assert len(ds) == 8

    def test_item_shape(self) -> None:
        ds = MusicSynthDataset(num_samples=2, audio_length=48000)
        item = ds[0]
        assert item.shape == (1, 48000)

    def test_normalized(self) -> None:
        ds = MusicSynthDataset(num_samples=2, audio_length=24000)
        for i in range(2):
            item = ds[i]
            assert item.abs().max() <= 1.0 + 1e-6

    def test_determinism(self) -> None:
        """Same index should produce the same waveform."""
        ds = MusicSynthDataset(num_samples=2, audio_length=24000)
        assert torch.allclose(ds[0], ds[0])

    def test_different_indices_different_audio(self) -> None:
        ds = MusicSynthDataset(num_samples=4, audio_length=24000)
        assert not torch.allclose(ds[0], ds[1])

    def test_has_multiple_harmonics(self) -> None:
        """Music-like dataset should contain richer spectral content than a single sine."""
        ds = MusicSynthDataset(num_samples=2, audio_length=24000, sample_rate=24000)
        item = ds[0].squeeze(0)
        # Compute magnitude spectrum
        spectrum = torch.fft.rfft(item).abs()
        # Count distinct peaks above a threshold
        threshold = spectrum.max() * 0.1
        peaks = (spectrum > threshold).sum().item()
        # A pure sine has ~1 peak; a musical tone should have several harmonics
        assert peaks >= 3

    def test_seed_reproducibility(self) -> None:
        """Two instances with the same seed should produce identical clips."""
        ds1 = MusicSynthDataset(num_samples=2, audio_length=24000, seed=123)
        ds2 = MusicSynthDataset(num_samples=2, audio_length=24000, seed=123)
        assert torch.allclose(ds1[0], ds2[0])
        assert torch.allclose(ds1[1], ds2[1])


class TestLoadAudioClip:
    def _write_wav(self, path: Path, sr: int, length: int, channels: int = 1) -> None:
        t = torch.arange(length, dtype=torch.float32) / sr
        freq = 440.0
        waveform = torch.sin(2 * torch.pi * freq * t)
        if channels > 1:
            waveform = waveform.unsqueeze(0).repeat(channels, 1)
        else:
            waveform = waveform.unsqueeze(0)
        soundfile.write(str(path), waveform.T.numpy(), sr)

    def test_load_mono_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=16000, length=16000)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        assert clip.shape == (1, 24000)

    def test_resample_to_target_sr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=16000, length=16000)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        # 1s at 16kHz -> 24000 samples after resampling
        assert clip.shape[1] == 24000

    def test_crop_to_target_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=24000, length=48000)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        assert clip.shape == (1, 24000)

    def test_pad_to_target_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=24000, length=12000)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        assert clip.shape == (1, 24000)

    def test_stereo_converts_to_mono(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=24000, length=24000, channels=2)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        assert clip.shape == (1, 24000)

    def test_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            self._write_wav(path, sr=24000, length=24000)
            clip = load_audio_clip(path, target_sr=24000, target_length=24000)
        assert clip.abs().max() <= 1.0 + 1e-6


class TestAudioFolderDataset:
    def test_loads_wav_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(3):
                t = torch.arange(24000, dtype=torch.float32) / 24000
                waveform = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
                soundfile.write(str(tmp_path / f"clip_{i}.wav"), waveform.T.numpy(), 24000)
            ds = AudioFolderDataset(tmp_path, audio_length=24000, sample_rate=24000)
            assert len(ds) == 3
            assert ds[0].shape == (1, 24000)

    def test_ignores_non_audio_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            t = torch.arange(24000, dtype=torch.float32) / 24000
            waveform = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
            soundfile.write(str(tmp_path / "clip.wav"), waveform.T.numpy(), 24000)
            (tmp_path / "readme.txt").write_text("not audio")
            ds = AudioFolderDataset(tmp_path, audio_length=24000, sample_rate=24000)
            assert len(ds) == 1


class TestBuildAudioDataloader:
    def test_dataloader(self) -> None:
        ds = ToyAudioDataset(num_samples=8, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.shape == (4, 1, 24000)

    def test_drop_last(self) -> None:
        """drop_last=True should drop incomplete final batch."""
        ds = ToyAudioDataset(num_samples=7, audio_length=24000)
        loader = build_audio_dataloader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 1  # only 1 full batch of 4
        assert batches[0].shape == (4, 1, 24000)
