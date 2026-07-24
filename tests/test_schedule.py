"""Tests for noise schedules and v-prediction targets (1-D audio)."""

from __future__ import annotations

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
        assert sigmas.shape == (20,)


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
