"""Tests for spectral (heat-death) stopping criterion in sampling (1-D)."""

from __future__ import annotations

import torch

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_ddim
from ald_sc.spectral_schedule import SpectralSchedule


def _make_dit(
    latent_channels: int = 4, latent_length: int = 16, spec_dim: int = 12
) -> MinimalDiT:
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


def _make_prior(f: int = 32, q: int = 8):
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestSpectralStopping:
    def test_sample_ddim_with_spectral_schedule_returns_shape(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        spec_sched = SpectralSchedule(prior, horizon=1.0, eps=1e-3)
        c_spec = torch.randn(2, 12)

        z = sample_ddim(
            dit,
            sched,
            c_spec=c_spec,
            batch_size=2,
            steps=50,
            seed=3407,
            spectral_schedule=spec_sched,
        )
        assert z.shape == (2, 4, 16)

    def test_spectral_stopping_fires_before_max_steps(self) -> None:
        """With a high epsilon, the spectral schedule should stop early."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        spec_sched = SpectralSchedule(prior, horizon=1.0, eps=100.0)

        c_spec = torch.randn(1, 12)
        z, steps_used = sample_ddim(
            dit,
            sched,
            c_spec=c_spec,
            batch_size=1,
            steps=50,
            seed=3407,
            spectral_schedule=spec_sched,
            return_steps=True,
        )
        assert steps_used < 50, f"Expected early stop, got {steps_used} steps"
        assert steps_used >= 0

    def test_no_spectral_schedule_uses_all_steps(self) -> None:
        """Without a spectral schedule, all steps are used."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)

        c_spec = torch.randn(1, 12)
        z, steps_used = sample_ddim(
            dit,
            sched,
            c_spec=c_spec,
            batch_size=1,
            steps=20,
            seed=3407,
            return_steps=True,
        )
        assert steps_used == 20

    def test_spectral_stopping_deterministic(self) -> None:
        """Same seed + same spectral schedule -> same result and same step count."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        prior = _make_prior()
        spec_sched = SpectralSchedule(prior, horizon=1.0, eps=1e-2)

        z1, s1 = sample_ddim(
            dit,
            sched,
            batch_size=1,
            steps=50,
            seed=3407,
            spectral_schedule=spec_sched,
            return_steps=True,
        )
        z2, s2 = sample_ddim(
            dit,
            sched,
            batch_size=1,
            steps=50,
            seed=3407,
            spectral_schedule=spec_sched,
            return_steps=True,
        )
        assert torch.allclose(z1, z2, atol=1e-6)
        assert s1 == s2

    def test_heat_death_metric_decreases_during_sampling(self) -> None:
        """The heat-death metric should decrease as sampling progresses."""
        prior = _make_prior()
        spec_sched = SpectralSchedule(prior, horizon=1.0)
        CosineSchedule(num_steps=100)

        metric_start = spec_sched.heat_death_metric(torch.tensor(0.99))
        metric_end = spec_sched.heat_death_metric(torch.tensor(0.01))
        assert metric_end > metric_start, (
            "Metric should be higher when modes are active (t near 0)"
        )
