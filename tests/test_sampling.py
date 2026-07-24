"""Tests for samplers (1-D audio latents)."""

from __future__ import annotations

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
        """At a single step, Euler and DDIM should produce similar results."""
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
