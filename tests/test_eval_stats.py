"""Tests for PROTOCOL_10S statistics helpers (per-frame diversity, TOST CI)."""

from __future__ import annotations

import pytest
import torch

from ald_sc.eval_stats import (
    bootstrap_fad_ci,
    cross_clip_frame_excess,
    equivalence_verdict,
)


class TestCrossClipFrameExcess:
    def _clouds(
        self, *centers: torch.Tensor, k: int = 16, dim: int = 8
    ) -> list[torch.Tensor]:
        clouds = []
        for c in centers:
            base = torch.nn.functional.normalize(c, dim=-1)
            spread = 0.01 * torch.randn(
                k, dim, generator=torch.Generator().manual_seed(3407)
            )
            clouds.append(torch.nn.functional.normalize(base + spread, dim=-1))
        return clouds

    def test_identical_clouds_have_zero_excess(self) -> None:
        c = torch.randn(1, 8)
        clouds = self._clouds(c, c)
        # Exactly equal clouds dip microscopically negative: the aligned
        # zero pairs in D survive the within-clip diagonal mask ((k−1)/k
        # artifact — see eval_stats docstring).
        excess = cross_clip_frame_excess(clouds)
        assert abs(excess) < 0.02, excess

    def test_disjoint_clouds_have_positive_excess(self) -> None:
        a = torch.zeros(1, 8)
        a[0, 0] = 1.0
        b = torch.zeros(1, 8)
        b[0, 1] = 1.0
        clouds = self._clouds(a, b)
        assert cross_clip_frame_excess(clouds) > 0.5

    def test_single_clip_undefined(self) -> None:
        c = torch.randn(1, 8)
        assert cross_clip_frame_excess(self._clouds(c)) == 0.0


class TestBootstrapFadCi:
    def _feats(
        self, n: int, dim: int = 16, shift: float = 0.0, seed: int = 7
    ) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        feats = torch.randn(n, dim, generator=g)
        return feats + shift

    def test_identical_sets_give_tight_low_ci(self) -> None:
        ref = self._feats(64)
        # Resampling-with-replacement shrinks covariance slightly, so the
        # FAD-proxy of bootstrapped identical sets sits near (not exactly
        # at) zero; what matters is that the whole CI stays small.
        low, high = bootstrap_fad_ci(
            ref.clone(), ref, n_boot=100, alpha=0.05, seed=3407
        )
        assert 0.0 <= low <= high < 25.0

    def test_shifted_set_ci_sits_above_identical_set_ci(self) -> None:
        ref = self._feats(64)
        low_same, _ = bootstrap_fad_ci(
            ref.clone(), ref, n_boot=100, alpha=0.05, seed=3407
        )
        arm = self._feats(64, shift=12.0, seed=99)
        low_shift, _ = bootstrap_fad_ci(arm, ref, n_boot=100, alpha=0.05, seed=3407)
        assert low_shift > low_same + 50

    def test_shifted_set_excludes_zero_above(self) -> None:
        ref = self._feats(64)
        arm = self._feats(64, shift=12.0, seed=99)
        low, high = bootstrap_fad_ci(arm, ref, n_boot=200, alpha=0.05, seed=3407)
        assert low > 0.0, (low, high)

    def test_deterministic_given_seed(self) -> None:
        ref = self._feats(48)
        arm = self._feats(48, shift=3.0, seed=5)
        ci1 = bootstrap_fad_ci(arm, ref, n_boot=50, seed=1234)
        ci2 = bootstrap_fad_ci(arm, ref, n_boot=50, seed=1234)
        assert ci1 == ci2


class TestEquivalenceVerdict:
    def test_within_margin_is_equivalent(self) -> None:
        assert equivalence_verdict(-3.0, 4.0, margin=10.0) == "equivalent"

    def test_high_side_breach_is_inferior(self) -> None:
        assert equivalence_verdict(12.0, 40.0, margin=10.0) == "inferior"
        assert equivalence_verdict(5.0, 40.0, margin=10.0) == "inconclusive"

    def test_low_side_breach_is_superior_not_equivalent(self) -> None:
        # Entirely BETTER than -margin: not 'equivalent', flag as superior.
        assert equivalence_verdict(-90.0, -60.0, margin=10.0) == "superior"

    def test_straddling_margin_is_inconclusive(self) -> None:
        assert equivalence_verdict(-20.0, 15.0, margin=10.0) == "inconclusive"

    def test_requires_two_sided_inputs(self) -> None:
        with pytest.raises(ValueError):
            equivalence_verdict(0.0, 10.0, margin=-1.0)
