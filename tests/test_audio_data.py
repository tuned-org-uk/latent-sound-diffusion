"""Tests for audio data loading (no external data required)."""

from __future__ import annotations

import torch

from ald_sc.data import MusicSynthDataset, ToyAudioDataset, build_audio_dataloader


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
