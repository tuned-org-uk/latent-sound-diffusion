"""Tests for samplers (1-D audio latents)."""

from __future__ import annotations

import pytest
import torch

from ald_sc.dit import MinimalDiT
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_euler, sample_ddim


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


class TestSampleEuler:
    def test_output_shape(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)
        steps = 10

        z = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=steps)
        assert z.shape == (2, 4, 16)

    def test_determinism_with_seed(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z1 = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        z2 = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        assert torch.allclose(z1, z2, atol=1e-6)


class TestSampleDDIM:
    def test_output_shape(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10)
        assert z.shape == (2, 4, 16)

    def test_determinism_with_seed(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z1 = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        z2 = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        assert torch.allclose(z1, z2, atol=1e-6)


class TestEulerDDIMParity:
    def test_single_step_parity(self) -> None:
        """At one step both samplers reduce to the same z0 prediction."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, 12)

        z_euler = sample_euler(
            dit, sched, c_spec=c_spec, batch_size=1, steps=1, seed=3407
        )
        z_ddim = sample_ddim(
            dit, sched, c_spec=c_spec, batch_size=1, steps=1, seed=3407
        )
        assert torch.allclose(z_euler, z_ddim, atol=1e-4)

    def test_multi_step_parity(self) -> None:
        """Over a full trajectory both samplers must implement the same
        v-prediction DDIM update and agree.

        Regression test for the v0.11 bug where sample_ddim fed the raw
        velocity into the noise-direction slot while sample_euler derived
        the correct direction eps_hat = sqrt(ab)*v + sqrt(1-ab)*x; the two
        trajectories then diverged at every intermediate step.
        """
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, 12)

        z_euler = sample_euler(
            dit, sched, c_spec=c_spec, batch_size=1, steps=5, seed=3407
        )
        z_ddim = sample_ddim(
            dit, sched, c_spec=c_spec, batch_size=1, steps=5, seed=3407
        )
        max_diff = (z_euler - z_ddim).abs().max().item()
        assert torch.allclose(z_euler, z_ddim, atol=1e-5), (
            f"euler and ddim trajectories diverged (max diff {max_diff:.2e})"
        )


class TestSampleDDIMEarlyStop:
    """Issue #62: sample_ddim must support true early-stop truncation.

    Under the corrected v0.12 ladder every full run terminates at
    alpha_bar == 1, so re-integrating the same seed at different step
    counts converges to the identical endpoint (pure discretization
    error) — which collapsed the stopvar bank mode. Truncating the ladder
    at an intermediate sigma index instead returns a partially-denoised
    latent that retains sqrt(1 - ab[stop_sigma]) noise.
    """

    def _dit_and_sched(self):
        torch.manual_seed(3407)
        return _make_dit(), CosineSchedule(num_steps=100)

    def test_default_run_unchanged_when_stop_sigma_none(self) -> None:
        dit, sched = self._dit_and_sched()
        z_a = sample_ddim(dit, sched, batch_size=1, steps=10, seed=5)
        z_b = sample_ddim(dit, sched, batch_size=1, steps=10, seed=5, stop_sigma=None)
        assert torch.allclose(z_a, z_b)

    def test_stop_at_first_ladder_entry_returns_initial_noise(self) -> None:
        """Stopping at the top of the ladder performs zero updates."""
        dit, sched = self._dit_and_sched()
        ladder = sched.sample_sigmas(steps=10)
        z = sample_ddim(
            dit, sched, batch_size=1, steps=10, seed=5, stop_sigma=int(ladder[0])
        )
        expected = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(5))
        assert torch.allclose(z, expected, atol=1e-6)

    def test_stop_at_final_ladder_entry_equals_full_run(self) -> None:
        dit, sched = self._dit_and_sched()
        ladder = sched.sample_sigmas(steps=10)
        z_full = sample_ddim(dit, sched, batch_size=1, steps=10, seed=5)
        z_stop = sample_ddim(
            dit, sched, batch_size=1, steps=10, seed=5, stop_sigma=int(ladder[-1])
        )
        assert torch.allclose(z_stop, z_full, atol=1e-6)

    def test_mid_ladder_stop_retains_residual_noise(self) -> None:
        """A truncated latent differs from the endpoint and stops early.

        The full run terminates at ab == 1 (no residual); a mid-ladder
        stop must halt integration at its rung — reporting exactly that
        many executed updates — and leave the state measurably different
        from the fully denoised endpoint.
        """
        torch.manual_seed(3407)
        dit, sched = self._dit_and_sched()
        ladder = sched.sample_sigmas(steps=10)
        mid = int(ladder[len(ladder) // 2])
        full = sample_ddim(dit, sched, batch_size=1, steps=10, seed=5)
        stopped, used = sample_ddim(
            dit,
            sched,
            batch_size=1,
            steps=10,
            seed=5,
            stop_sigma=mid,
            return_steps=True,
        )
        assert not torch.allclose(stopped, full)
        assert used == len(ladder) // 2, (
            f"stop_sigma must halt integration at its rung; {used} updates run"
        )

    def test_earlier_stops_are_farther_from_endpoint(self) -> None:
        torch.manual_seed(3407)
        dit, sched = self._dit_and_sched()
        ladder = sched.sample_sigmas(steps=10)
        full = sample_ddim(dit, sched, batch_size=1, steps=10, seed=5)
        d_early = (
            (
                sample_ddim(dit, sched, steps=10, seed=5, stop_sigma=int(ladder[3]))
                - full
            )
            .abs()
            .mean()
        )
        d_late = (
            (
                sample_ddim(dit, sched, steps=10, seed=5, stop_sigma=int(ladder[8]))
                - full
            )
            .abs()
            .mean()
        )
        assert d_early > d_late

    @pytest.mark.parametrize("bad", [-1, 100])
    def test_invalid_stop_sigma_raises(self, bad: int) -> None:
        dit, sched = self._dit_and_sched()
        with pytest.raises(ValueError, match="stop_sigma"):
            sample_ddim(dit, sched, batch_size=1, steps=10, seed=5, stop_sigma=bad)

    def test_stopped_run_is_reproducible(self) -> None:
        dit, sched = self._dit_and_sched()
        ladder = sched.sample_sigmas(steps=10)
        a = sample_ddim(
            dit, sched, batch_size=1, steps=10, seed=9, stop_sigma=int(ladder[4])
        )
        b = sample_ddim(
            dit, sched, batch_size=1, steps=10, seed=9, stop_sigma=int(ladder[4])
        )
        assert torch.allclose(a, b)
