"""Tests for waveform-domain equal-power overlap-add stitching (Track B)."""

from __future__ import annotations

import math

import pytest
import torch

from ald_sc.stitching import equal_power_overlap_add


class TestEqualPowerOverlapAdd:
    def test_single_segment_passthrough(self) -> None:
        seg = torch.arange(5.0).unsqueeze(0)
        out = equal_power_overlap_add([seg], overlap=2)
        assert torch.equal(out, seg)

    def test_output_length_arithmetic(self) -> None:
        segs = [torch.ones(1, n) for n in (100, 80, 60)]
        out = equal_power_overlap_add(segs, overlap=20)
        assert out.shape == (1, 100 + 80 + 60 - 2 * 20)

    def test_seam_follows_complementary_power_law(self) -> None:
        """Overlap value must be a*cos(t) + b*sin(t), t sweeping pi/2 -> 0."""
        a = torch.full((1, 64), 1.0)
        b = torch.full((1, 64), 2.0)
        overlap = 16
        out = equal_power_overlap_add([a, b], overlap=overlap)
        assert torch.allclose(out[:, : 64 - overlap], torch.full((1, 48), 1.0))
        for i in range(overlap):
            t = (math.pi / 2) * (i / (overlap - 1))
            expected = 1.0 * math.cos(t) + 2.0 * math.sin(t)
            assert out[0, 48 + i] == pytest.approx(expected, abs=1e-6)

    def test_uncorrelated_noise_keeps_constant_power_through_seam(self) -> None:
        """Unit-variance noise on both sides -> overlap RMS stays ~1."""
        gen = torch.Generator().manual_seed(3407)
        a = torch.randn(1, 4000, generator=gen)
        b = torch.randn(1, 4000, generator=gen)
        overlap = 256
        out = equal_power_overlap_add([a, b], overlap=overlap)
        start = 4000 - overlap // 2 - 128
        mid_rms = out[0, start : start + 256].pow(2).mean().sqrt().item()
        assert 0.9 < mid_rms < 1.1, f"mid-seam RMS {mid_rms:.3f}"

    def test_zero_overlap_is_concatenation(self) -> None:
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0]])
        out = equal_power_overlap_add([a, b], overlap=0)
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_overlap_larger_than_shortest_segment_raises(self) -> None:
        a = torch.ones(1, 10)
        b = torch.ones(1, 4)
        with pytest.raises(ValueError, match="overlap"):
            equal_power_overlap_add([a, b], overlap=8)

    def test_batched_segments_supported(self) -> None:
        a = torch.ones(2, 32)
        b = torch.ones(2, 32)
        out = equal_power_overlap_add([a, b], overlap=8)
        assert out.shape == (2, 56)
