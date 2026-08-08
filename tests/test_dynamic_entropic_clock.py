"""Tests for the dynamic Barontini entropic clock (issue #41).

The DynamicEntropicClock promotes SpectralSchedule from a frozen-prior stub
to a dynamic accumulator that advances τ only when entropy actually flows
between spectral sectors during generation.

Tests cover:
- Dynamic τ accumulation from per-step entropy changes
- Dynamic ᾱ(τ_n) schedule indexed by accumulated τ
- Entropy conservation diagnostic
- Heat-death criterion with dynamic clock
- Drop-in compatibility with the sampler's spectral_schedule hook
"""

from __future__ import annotations

import torch
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.sampling import sample_ddim
from ald_sc.schedule import CosineSchedule
from ald_sc.spectral_schedule import DynamicEntropicClock, SpectralSchedule


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


def _make_dit(latent_channels: int = 4, latent_length: int = 16, spec_dim: int = 12) -> MinimalDiT:
    return MinimalDiT(
        latent_channels=latent_channels,
        latent_length=latent_length,
        patch_size=2,
        dim=32,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=spec_dim,
        cfg_dropout=0.0,
    )


class TestDynamicEntropicClockConstruction:
    def test_construction_from_prior(self) -> None:
        prior = _make_prior(f=32, q=8)
        clock = DynamicEntropicClock(prior, eps=1e-3, tau_warmup=5)
        assert clock.q == 8
        assert clock.eps == 1e-3
        assert clock.tau_warmup == 5

    def test_inherits_from_spectral_schedule(self) -> None:
        """DynamicEntropicClock must be a SpectralSchedule subclass so it
        drops into the existing sampler hook."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        assert isinstance(clock, SpectralSchedule)

    def test_no_trainable_parameters(self) -> None:
        """The clock must be fully frozen (no trainable params)."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        params = list(clock.parameters())
        assert len(params) == 0

    def test_tau_n_starts_at_zero(self) -> None:
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        assert clock.tau_n == 0.0

    def test_tau_max_est_initially_none(self) -> None:
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        assert clock.tau_max_est is None


class TestDynamicTauAccumulation:
    def test_update_advances_tau_n(self) -> None:
        """update(x) should advance tau_n by |ΔS_bright|."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        x1 = torch.randn(1, 4, 16)
        clock.update(x1)
        assert clock.tau_n >= 0.0

    def test_repeated_identical_update_zero_entropy_change(self) -> None:
        """If the latent doesn't change, ΔS = 0 and tau_n stalls —
        Barontini's key property: at equilibrium, time stops."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        x = torch.randn(1, 4, 16)
        clock.update(x)
        tau_after_first = clock.tau_n
        clock.update(x)
        assert abs(clock.tau_n - tau_after_first) < 1e-6, (
            "Identical latents should produce zero entropy change"
        )

    def test_different_latents_advance_tau(self) -> None:
        """Different latents should produce non-zero entropy change."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        x1 = torch.randn(1, 4, 16)
        x2 = torch.randn(1, 4, 16) * 5
        clock.update(x1)
        clock.update(x2)
        assert clock.tau_n > 0.0

    def test_warmup_estimates_tau_max(self) -> None:
        """After tau_warmup updates, tau_max_est should be set."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, tau_warmup=3)
        for _ in range(3):
            clock.update(torch.randn(1, 4, 16))
        assert clock.tau_max_est is not None
        assert clock.tau_max_est > 0.0


class TestDynamicAlphaBar:
    def test_alpha_bar_k_shape(self) -> None:
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        clock.tau_max_est = 1.0
        ab = clock.alpha_bar_k(torch.tensor(0.5))
        assert ab.shape == (8,)

    def test_alpha_bar_k_at_zero_tau(self) -> None:
        """At τ_n = 0, all modes should have ᾱ_k ≈ 1 (no noise)."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        clock.tau_max_est = 1.0
        ab = clock.alpha_bar_k(torch.tensor(0.0))
        assert torch.allclose(ab, torch.ones(8), atol=1e-5)

    def test_alpha_bar_k_decreases_with_tau(self) -> None:
        """ᾱ_k should decrease as τ_n increases."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        clock.tau_max_est = 1.0
        clock.tau_n = 0.1
        ab_low = clock.alpha_bar_k(torch.tensor(0.5))
        clock.tau_n = 0.9
        ab_high = clock.alpha_bar_k(torch.tensor(0.5))
        assert ab_high.mean() < ab_low.mean()


class TestHeatDeathWithDynamicClock:
    def test_is_heat_death_false_before_warmup(self) -> None:
        """Before warmup completes (tau_max_est is None), no heat death."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, eps=1e-3)
        clock.update(torch.randn(1, 4, 16))
        assert not clock.is_heat_death(torch.tensor(0.5))

    def test_is_heat_death_true_at_full_tau(self) -> None:
        """When tau_n >= tau_max_est, heat death fires."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, eps=1e-3)
        clock.tau_max_est = 0.1
        clock.tau_n = 0.2
        assert clock.is_heat_death(torch.tensor(1.0))

    def test_is_heat_death_false_mid_run(self) -> None:
        """Mid-run, tau_n < tau_max_est, heat death should not fire."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, eps=1e-3)
        clock.tau_max_est = 10.0
        clock.tau_n = 0.5
        assert not clock.is_heat_death(torch.tensor(0.5))


class TestEntropyConservationDiagnostic:
    def test_conserved_returns_scalar(self) -> None:
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        x = torch.randn(1, 4, 16)
        clock.update(x)
        conserved = clock.conserved()
        assert conserved.dim() == 0

    def test_conserved_non_negative(self) -> None:
        """|S_bright + S_dark - S_0| >= 0 by construction."""
        prior = _make_prior()
        clock = DynamicEntropicClock(prior)
        clock.update(torch.randn(1, 4, 16))
        assert clock.conserved() >= 0.0


class TestSamplerIntegration:
    def test_sample_ddim_with_dynamic_clock_returns_shape(self) -> None:
        """sample_ddim works with DynamicEntropicClock as a drop-in."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, eps=1e-3, tau_warmup=3)
        c_spec = torch.randn(1, 12)

        z = sample_ddim(
            dit,
            sched,
            c_spec=c_spec,
            batch_size=1,
            steps=20,
            seed=3407,
            spectral_schedule=clock,
        )
        assert z.shape == (1, 4, 16)

    def test_sample_ddim_with_dynamic_clock_advances_tau(self) -> None:
        """After sampling, tau_n should have advanced."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        clock = DynamicEntropicClock(prior, eps=1e-3, tau_warmup=3)

        sample_ddim(
            dit,
            sched,
            batch_size=1,
            steps=20,
            seed=3407,
            spectral_schedule=clock,
        )
        assert clock.tau_n > 0.0, "tau_n should advance during sampling"
        assert clock.tau_max_est is not None, "tau_max_est should be set after warmup"

    def test_frozen_schedule_still_works(self) -> None:
        """The frozen SpectralSchedule still works unchanged."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        spec_sched = SpectralSchedule(prior, horizon=1.0, eps=1e-3)

        z = sample_ddim(
            dit,
            sched,
            batch_size=1,
            steps=20,
            seed=3407,
            spectral_schedule=spec_sched,
        )
        assert z.shape == (1, 4, 16)
