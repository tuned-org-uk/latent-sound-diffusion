"""Tests for the per-mode entropic spectral schedule.

Tests the formulas from the paper §5 (Entropic time from the ArrowSpace
spectrum):
- τ_k(t) = ν_k · t
- ᾱ_k(τ_k) = cos²(π/2 · τ_k / τ_{k,max})
- τ_{k,max} = ν_k · T
- Heat death: Σ ν_k · mmse_k(t) < ε
"""

from __future__ import annotations

import torch
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.spectral_schedule import SpectralSchedule


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestSpectralSchedule:
    def test_construction_from_prior(self) -> None:
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        assert sched.q == 8
        assert sched.nu.shape == (8,)

    def test_tau_k_shape(self) -> None:
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        t = torch.tensor(0.5)
        tau_k = sched.tau_k(t)
        assert tau_k.shape == (8,)

    def test_tau_k_equals_nu_times_t(self) -> None:
        """τ_k(t) = ν_k · t"""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        t = torch.tensor(0.5)
        tau_k = sched.tau_k(t)
        assert torch.allclose(tau_k, sched.nu * t, atol=1e-6)

    def test_alpha_bar_k_shape(self) -> None:
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        t = torch.tensor(0.5)
        ab_k = sched.alpha_bar_k(t)
        assert ab_k.shape == (8,)

    def test_alpha_bar_k_at_t_zero_is_one(self) -> None:
        """At t=0, all modes should have ᾱ_k = 1 (no noise)."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        ab_k = sched.alpha_bar_k(torch.tensor(0.0))
        assert torch.allclose(ab_k, torch.ones(8), atol=1e-5)

    def test_alpha_bar_k_at_t_horizon_is_zero(self) -> None:
        """At t=T (horizon), all modes should have ᾱ_k ≈ 0 (full noise)."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        ab_k = sched.alpha_bar_k(torch.tensor(1.0))
        assert torch.allclose(ab_k, torch.zeros(8), atol=1e-5)

    def test_alpha_bar_k_monotone_decreasing(self) -> None:
        """ᾱ_k should be monotonically decreasing for each mode."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        ts = torch.linspace(0, 1, 50)
        ab_ks = torch.stack([sched.alpha_bar_k(t) for t in ts])
        for k in range(8):
            diffs = ab_ks[1:, k] - ab_ks[:-1, k]
            assert (diffs <= 1e-6).all(), f"Mode {k} not monotone"

    def test_all_modes_same_alpha_bar_in_external_time(self) -> None:
        """All modes have identical ᾱ_k in external time.

        This is because ν_k cancels in the ratio τ_k/τ_{k,max} = t/T.
        The paper notes: 'all modes formally reach equilibrium at the same
        external time T'. Per-mode differentiation comes through the heat-death
        metric Σ ν_k · mmse_k(t), not through ᾱ_k itself.
        """
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        t = torch.tensor(0.3)
        ab_k = sched.alpha_bar_k(t)
        # All modes should have nearly identical alpha_bar
        assert torch.allclose(ab_k, ab_k[0].expand_as(ab_k), atol=1e-5)

    def test_high_nu_accumulates_tau_faster(self) -> None:
        """High-ν modes accumulate entropic time τ_k faster than low-ν modes."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        nu = sched.nu
        k_low = nu.argmin().item()
        k_high = nu.argmax().item()
        t = torch.tensor(0.5)
        tau_k = sched.tau_k(t)
        assert tau_k[k_high] > tau_k[k_low], (
            f"High-ν mode (ν={nu[k_high]:.4f}) should accumulate more τ_k "
            f"than low-ν mode (ν={nu[k_low]:.4f}) at t=0.5"
        )

    def test_heat_death_metric_weighted_by_nu(self) -> None:
        """The heat-death metric Σ ν_k · ᾱ_k(t) is dominated by high-ν modes."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        nu = sched.nu
        k_high = nu.argmax().item()
        k_low = nu.argmin().item()
        t = torch.tensor(0.5)
        ab_k = sched.alpha_bar_k(t)
        contributions = nu * ab_k
        # The high-ν mode contributes more to the heat-death metric
        assert contributions[k_high] > contributions[k_low]

    def test_tau_k_max_equals_nu_times_horizon(self) -> None:
        """τ_{k,max} = ν_k · T"""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=2.5)
        assert torch.allclose(sched.tau_k_max, sched.nu * 2.5, atol=1e-6)

    def test_heat_death_criterion(self) -> None:
        """Heat death: Σ ν_k · mmse_k(t) < ε."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0, eps=1e-3)
        # At t=0, mmse should be high (lots to resolve)
        assert not sched.is_heat_death(torch.tensor(0.0))
        # At t=T (horizon), all modes should be near equilibrium
        assert sched.is_heat_death(torch.tensor(1.0))

    def test_heat_death_threshold(self) -> None:
        """The heat death metric should decrease over time."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        ts = torch.linspace(0.01, 0.99, 20)
        metrics = [sched.heat_death_metric(t).item() for t in ts]
        for i in range(1, len(metrics)):
            assert metrics[i] <= metrics[i - 1] + 1e-6, "Metric should be non-increasing"

    def test_entropy_rate(self) -> None:
        """dS_k/dt = -ν_k (the entropy exchange rate per mode)."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        rates = sched.entropy_rate()
        assert rates.shape == (8,)
        assert torch.allclose(rates, -sched.nu, atol=1e-6)

    def test_frozen_no_parameters(self) -> None:
        """The spectral schedule must have zero trainable parameters."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        params = list(sched.parameters())
        assert len(params) == 0, "SpectralSchedule must be fully frozen"

    def test_no_gradients_to_prior(self) -> None:
        """Computing alpha_bar_k should not produce gradients in the prior."""
        prior = _make_prior(f=32, q=8)
        sched = SpectralSchedule(prior, horizon=1.0)
        t = torch.tensor(0.5, requires_grad=True)
        ab_k = sched.alpha_bar_k(t)
        ab_k.sum().backward()
        # t should have a gradient
        assert t.grad is not None
        # prior buffers should not have gradients
        for buf in prior.buffers():
            assert buf.grad is None or buf.grad == 0
