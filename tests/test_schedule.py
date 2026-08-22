"""Tests for noise schedules and v-prediction targets (1-D audio)."""

from __future__ import annotations

import pytest
import torch

from ald_sc.schedule import CosineSchedule, LinearSchedule


class TestCosineSchedule:
    def test_alpha_bar_monotonically_decreasing(self) -> None:
        sched = CosineSchedule(num_steps=100)
        ab = sched.alpha_bar
        assert ab[0] > ab[-1]
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-6).all(), "alpha_bar must be monotonically non-increasing"

    def test_alpha_bar_range(self) -> None:
        sched = CosineSchedule(num_steps=100)
        ab = sched.alpha_bar
        assert ab[0] <= 1.0 + 1e-6
        assert ab[-1] >= 0.0 - 1e-6

    def test_add_noise_shape(self) -> None:
        torch.manual_seed(3407)
        sched = CosineSchedule(num_steps=100)
        z0 = torch.randn(2, 4, 16)
        t = torch.tensor([10, 50])
        noise = torch.randn_like(z0)
        z_t = sched.add_noise(z0, t, noise)
        assert z_t.shape == z0.shape

    def test_add_noise_at_t_zero_is_clean(self) -> None:
        torch.manual_seed(3407)
        sched = CosineSchedule(num_steps=100)
        z0 = torch.randn(2, 4, 16)
        t = torch.tensor([0, 0])
        noise = torch.randn_like(z0)
        z_t = sched.add_noise(z0, t, noise)
        assert torch.allclose(z_t, z0, atol=1e-5)

    def test_v_target_shape(self) -> None:
        torch.manual_seed(3407)
        sched = CosineSchedule(num_steps=100)
        z0 = torch.randn(2, 4, 16)
        t = torch.tensor([10, 50])
        noise = torch.randn_like(z0)
        v = sched.v_target(z0, t, noise)
        assert v.shape == z0.shape

    def test_v_target_add_noise_round_trip(self) -> None:
        """v_target and add_noise are consistent: z_t = sqrt(ab)*z0 + sqrt(1-ab)*noise
        and v = sqrt(ab)*noise - sqrt(1-ab)*z0, so z0 = sqrt(ab)*z_t - sqrt(1-ab)*v."""
        torch.manual_seed(3407)
        sched = CosineSchedule(num_steps=100)
        z0 = torch.randn(3, 4, 16)
        t = torch.tensor([10, 50, 90])
        noise = torch.randn_like(z0)
        z_t = sched.add_noise(z0, t, noise)
        v = sched.v_target(z0, t, noise)
        ab = sched.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, 1, 1)
        sqrt_1mab = (1 - ab).sqrt().view(-1, 1, 1)
        z0_recovered = sqrt_ab * z_t - sqrt_1mab * v
        assert torch.allclose(z0_recovered, z0, atol=1e-5)

    def test_sample_batch_shape(self) -> None:
        torch.manual_seed(3407)
        sched = CosineSchedule(num_steps=100)
        x0 = torch.randn(4, 4, 16)
        t = sched.sample_batch(x0)
        assert t.shape == (4,)
        assert (t >= 0).all() and (t < 100).all()

    def test_sample_sigmas_shape(self) -> None:
        sched = CosineSchedule(num_steps=100)
        sigmas = sched.sample_sigmas(steps=20)
        assert sigmas.shape == (21,)


class TestLinearSchedule:
    def test_alpha_bar_monotonically_decreasing(self) -> None:
        sched = LinearSchedule(num_steps=100)
        ab = sched.alpha_bar
        assert ab[0] > ab[-1]
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-6).all()

    def test_add_noise_shape(self) -> None:
        torch.manual_seed(3407)
        sched = LinearSchedule(num_steps=100)
        z0 = torch.randn(2, 4, 16)
        t = torch.tensor([10, 50])
        noise = torch.randn_like(z0)
        z_t = sched.add_noise(z0, t, noise)
        assert z_t.shape == z0.shape

    def test_v_target_add_noise_round_trip(self) -> None:
        torch.manual_seed(3407)
        sched = LinearSchedule(num_steps=100)
        z0 = torch.randn(3, 4, 16)
        t = torch.tensor([10, 50, 90])
        noise = torch.randn_like(z0)
        z_t = sched.add_noise(z0, t, noise)
        v = sched.v_target(z0, t, noise)
        ab = sched.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, 1, 1)
        sqrt_1mab = (1 - ab).sqrt().view(-1, 1, 1)
        z0_recovered = sqrt_ab * z_t - sqrt_1mab * v
        assert torch.allclose(z0_recovered, z0, atol=1e-5)


class TestScheduleInterface:
    def test_len(self) -> None:
        sched = CosineSchedule(num_steps=50)
        assert len(sched) == 50

    def test_getitem(self) -> None:
        sched = CosineSchedule(num_steps=50)
        assert sched[0].shape == ()

    def test_cosine_and_linear_different(self) -> None:
        cos = CosineSchedule(num_steps=100)
        lin = LinearSchedule(num_steps=100)
        assert not torch.allclose(cos.alpha_bar, lin.alpha_bar)


class TestSampleSigmasEndpoint:
    """The sampling ladder must terminate at sigma index 0 (alpha_bar == 1).

    Regression tests for the v0.11 bug where the ladder ended at
    round(T/steps)-1, leaving ~4% residual noise in every generated latent
    because the final pairwise update never reached t=0.
    """

    @pytest.mark.parametrize("steps", [1, 2, 10, 50, 1000])
    def test_cosine_ladder_terminates_at_zero(self, steps: int) -> None:
        sched = CosineSchedule(num_steps=1000)
        sigmas = sched.sample_sigmas(steps)
        assert sigmas.shape == (steps + 1,)
        assert sigmas[0] == 999
        assert sigmas[-1] == 0
        assert (sigmas[1:] <= sigmas[:-1]).all(), "ladder must be non-increasing"

    @pytest.mark.parametrize("steps", [1, 10, 50])
    def test_linear_ladder_terminates_at_zero(self, steps: int) -> None:
        sched = LinearSchedule(num_steps=1000)
        sigmas = sched.sample_sigmas(steps)
        assert sigmas[-1] == 0

    def test_ladder_endpoint_has_full_signal_energy(self) -> None:
        cos = CosineSchedule(num_steps=1000)
        lin = LinearSchedule(num_steps=1000)
        last = cos.sample_sigmas(50)[-1]
        assert last == 0
        assert cos.alpha_bar[last] == pytest.approx(1.0, abs=1e-6)
        last_lin = lin.sample_sigmas(50)[-1]
        assert last_lin == 0
        assert lin.alpha_bar[last_lin] == lin.alpha_bar.max()

    def test_no_residual_noise_floor_after_full_trajectory(self) -> None:
        """A latent denoised through the whole ladder must sit at alpha_bar≈1.

        With the v0.11 ladder the final update stopped at index 19 where
        sqrt(1-alpha_bar) ≈ 0.04; the corrected ladder ends at index 0.
        """
        torch.manual_seed(3407)
        from ald_sc.dit import MinimalDiT
        from ald_sc.sampling import sample_ddim

        dit = MinimalDiT(
            latent_channels=4,
            latent_length=16,
            patch_size=2,
            dim=32,
            depth=2,
            num_heads=4,
            spec_dim=12,
        )
        sched = CosineSchedule(num_steps=100)
        z, steps_used = sample_ddim(
            dit, sched, batch_size=2, steps=10, return_steps=True
        )
        # Every step of the ladder executes and terminates at alpha_bar==1.
        assert steps_used == 10
        final_sigma = sched.sample_sigmas(10)[-1]
        assert final_sigma == 0
        assert sched.alpha_bar[final_sigma] == pytest.approx(1.0, abs=1e-6)
