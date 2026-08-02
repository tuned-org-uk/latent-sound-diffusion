"""Tests for the LSD inference contract (sound-bank / condition / MIDI)."""

from __future__ import annotations


import torch
from torch import Tensor, nn

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule


class StubEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        return z, a, prior.chart_energy_descriptor(a)

    def extract_features(self, x: Tensor) -> Tensor:
        return self.proj(x).float()


def _make_model(latent_length: int = 16) -> LSDModel:
    torch.manual_seed(3407)
    embeddings = torch.randn(32, 128)
    prior = build_arrow_prior(embeddings, q=8, k=4)
    encoder = StubEncoder()
    decoder = GraphDecoder(128, 1, 128, 16, prior, (2, 4, 5, 8))
    dit = MinimalDiT(
        latent_channels=128,
        latent_length=latent_length,
        patch_size=4,
        dim=32,
        depth=1,
        num_heads=4,
        spec_dim=24,
    )
    sched = CosineSchedule(num_steps=100)
    return LSDModel(
        prior=prior,
        dit=dit,
        decoder=decoder,
        encoder=encoder,
        schedule=sched,
        sample_rate=24000,
    )


class TestGenerateSoundBank:
    def test_returns_n_clips(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=4, steps=4, seed=3407)
        assert len(bank) == 4
        for clip in bank:
            assert clip.shape[0] == 1
            assert clip.shape[1] == 16 * 320  # latent_length * stride

    def test_seed_reproducible(self) -> None:
        m = _make_model()
        a = m.generate_sound_bank(n=2, steps=3, seed=42)
        b = m.generate_sound_bank(n=2, steps=3, seed=42)
        for ca, cb in zip(a, b):
            assert torch.allclose(ca, cb)

    def test_seed_none_runs(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=2, steps=3, seed=None)
        assert len(bank) == 2

    def test_temperature_changes_output(self) -> None:
        m = _make_model()
        low = m.generate_sound_bank(n=1, steps=3, seed=1, temperature=0.1)[0]
        high = m.generate_sound_bank(n=1, steps=3, seed=1, temperature=2.0)[0]
        assert not torch.allclose(low, high)


class TestConditionOnAudio:
    def test_variants_shape(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        variants = m.condition_on_audio(audio, n=3, steps=3, strength=0.5, seed=3407)
        assert len(variants) == 3
        for v in variants:
            assert v.shape[0] == 1
            assert v.shape[1] == 16 * 320

    def test_seed_reproducible(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        a = m.condition_on_audio(audio, n=2, steps=2, strength=0.3, seed=7)
        b = m.condition_on_audio(audio, n=2, steps=2, strength=0.3, seed=7)
        for va, vb in zip(a, b):
            assert torch.allclose(va, vb)

    def test_strength_changes_output(self) -> None:
        m = _make_model()
        audio = torch.randn(1, 1, 16 * 320)
        weak = m.condition_on_audio(audio, n=1, steps=2, strength=0.1, seed=5)[0]
        strong = m.condition_on_audio(audio, n=1, steps=2, strength=1.0, seed=5)[0]
        assert not torch.allclose(weak, strong)


class TestSynthesizeMidi:
    def test_requires_non_empty_bank(self) -> None:
        m = _make_model()
        try:
            import pytest
        except ImportError:
            return
        with pytest.raises(ValueError, match="bank"):
            m.synthesize_midi([(60, 0.0, 0.5)], bank=[], seed=3407)

    def test_midi_to_audio_length_proportional(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=1, steps=3, seed=3407)
        short = m.synthesize_midi([(60, 0.0, 0.5)], bank=bank, seed=3407)
        long = m.synthesize_midi([(60, 0.0, 1.0)], bank=bank, seed=3407)
        assert short.shape[-1] < long.shape[-1]

    def test_pitched_bank_sound_is_retimed(self) -> None:
        m = _make_model()
        bank = m.generate_sound_bank(n=1, steps=3, seed=3407)
        out = m.synthesize_midi(
            [(69, 0.0, 0.25), (72, 0.25, 0.25)],
            bank=bank,
            pitch_bank_root=69,
            seed=3407,
        )
        assert out.shape[-1] > bank[0].shape[-1] // 2
