"""Tests for classifier-free guidance (CFG) two-pass inference (issue #23).

Covers:
- ``cfg_forward`` formula correctness (two-pass blend matches the
  analytical ``v_uncond + s * (v_cond - v_uncond)``)
- ``guidance_scale=0.0`` produces unconditional output
- ``guidance_scale=1.0`` produces pure conditional output (single pass)
- ``guidance_scale>1.0`` produces amplified output (differs from both)
- ``guidance_scale`` ignored when ``c_spec is None``
- ``sample_ddim`` / ``sample_ddpm`` / ``sample_euler`` all support it
- ``LSDModel.generate_sound_bank`` forwards ``guidance_scale`` to the
  sampler

The AdaLN projection is zero-initialized by design on ``MinimalDiT``, so
an untrained model ignores ``c_spec``.  The ``perturb_adaln`` helper
unlocks the conditioning path (mirroring ``test_dit.py``) so the tests can
assert CFG behaviour without a trained model.
"""

from __future__ import annotations

import pytest
import torch

from ald_sc.sampling import (
    cfg_forward,
    sample_ddim,
    sample_ddim_steps,
    sample_ddpm,
    sample_euler,
)
from ald_sc.schedule import CosineSchedule

from tests.conftest import (
    LATENT_CH,
    LATENT_LEN,
    SPEC_DIM,
    make_dit,
    make_model,
    perturb_adaln,
)


# ---------------------------------------------------------------------------
# cfg_forward unit tests
# ---------------------------------------------------------------------------


class TestCfgForward:
    def test_formula_correctness(self) -> None:
        """Blended output must equal v_uncond + s * (v_cond - v_uncond)."""
        dit = make_dit()
        perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)
        scale = 3.0

        v_cond = dit(z, t, c_spec=c_spec)
        v_uncond = dit(z, t, c_spec=None)
        expected = v_uncond + scale * (v_cond - v_uncond)
        actual = cfg_forward(dit, z, t, c_spec, scale)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_scale_1_is_pure_conditional(self) -> None:
        dit = make_dit()
        perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        expected = dit(z, t, c_spec=c_spec)
        actual = cfg_forward(dit, z, t, c_spec, 1.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_scale_0_is_pure_unconditional(self) -> None:
        dit = make_dit()
        perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        expected = dit(z, t, c_spec=None)
        actual = cfg_forward(dit, z, t, c_spec, 0.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_no_c_spec_ignores_scale(self) -> None:
        dit = make_dit()
        perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])

        expected = dit(z, t, c_spec=None)
        for scale in [0.0, 1.0, 3.0, 10.0]:
            actual = cfg_forward(dit, z, t, None, scale)
            assert torch.allclose(actual, expected, atol=1e-6)

    def test_amplified_differs_from_both(self) -> None:
        """guidance_scale > 1 must differ from both pure cond and pure uncond."""
        dit = make_dit()
        perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        v_cond = cfg_forward(dit, z, t, c_spec, 1.0)
        v_uncond = cfg_forward(dit, z, t, c_spec, 0.0)
        v_amplified = cfg_forward(dit, z, t, c_spec, 3.0)

        assert not torch.allclose(v_amplified, v_cond, atol=1e-6)
        assert not torch.allclose(v_amplified, v_uncond, atol=1e-6)


# ---------------------------------------------------------------------------
# Sampler integration tests
# ---------------------------------------------------------------------------


class TestSamplerGuidanceScale:
    @pytest.mark.parametrize("sampler", [sample_ddim, sample_euler, sample_ddpm])
    def test_scale_0_matches_unconditional(self, sampler) -> None:
        """guidance_scale=0.0 + c_spec == c_spec=None (same seed)."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = sampler(dit, sched, c_spec=None, steps=10, seed=42)
        guided = sampler(
            dit, sched, c_spec=c_spec, steps=10, seed=42, guidance_scale=0.0
        )

        assert torch.allclose(uncond, guided, atol=1e-5)

    @pytest.mark.parametrize("sampler", [sample_ddim, sample_euler, sample_ddpm])
    def test_scale_1_matches_conditional(self, sampler) -> None:
        """guidance_scale=1.0 matches plain conditioned sampling (single pass)."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, SPEC_DIM)

        cond = sampler(dit, sched, c_spec=c_spec, steps=10, seed=42)
        guided = sampler(
            dit, sched, c_spec=c_spec, steps=10, seed=42, guidance_scale=1.0
        )

        assert torch.allclose(cond, guided, atol=1e-5)

    @pytest.mark.parametrize("sampler", [sample_ddim, sample_euler, sample_ddpm])
    def test_scale_3_differs_from_conditional(self, sampler) -> None:
        """guidance_scale=3.0 must diverge from pure conditional."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, SPEC_DIM)

        cond = sampler(dit, sched, c_spec=c_spec, steps=10, seed=42)
        amplified = sampler(
            dit, sched, c_spec=c_spec, steps=10, seed=42, guidance_scale=3.0
        )

        assert not torch.allclose(cond, amplified, atol=1e-5), (
            "guidance_scale=3.0 should amplify conditioning, diverging from "
            "the pure conditional trajectory."
        )

    def test_no_c_spec_ignores_scale(self) -> None:
        """Without c_spec, guidance_scale has no effect."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)

        base = sample_ddim(dit, sched, c_spec=None, steps=10, seed=42)
        for scale in [0.0, 3.0, 10.0]:
            out = sample_ddim(
                dit, sched, c_spec=None, steps=10, seed=42, guidance_scale=scale
            )
            assert torch.allclose(base, out, atol=1e-5)

    def test_ddim_steps_scale_0_matches_unconditional(self) -> None:
        """sample_ddim_steps honors guidance_scale=0.0 (unconditional)."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = list(sample_ddim_steps(dit, sched, c_spec=None, steps=10, seed=42))
        guided = list(
            sample_ddim_steps(
                dit, sched, c_spec=c_spec, steps=10, seed=42, guidance_scale=0.0
            )
        )

        assert len(uncond) == len(guided)
        for u, g in zip(uncond, guided):
            assert torch.allclose(u, g, atol=1e-5)

    def test_ddim_steps_scale_3_differs_from_conditional(self) -> None:
        """sample_ddim_steps with guidance_scale=3.0 diverges from conditional."""
        dit = make_dit()
        perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, SPEC_DIM)

        cond = list(sample_ddim_steps(dit, sched, c_spec=c_spec, steps=10, seed=42))
        amplified = list(
            sample_ddim_steps(
                dit, sched, c_spec=c_spec, steps=10, seed=42, guidance_scale=3.0
            )
        )

        assert len(cond) == len(amplified)
        assert not torch.allclose(cond[-1], amplified[-1], atol=1e-5), (
            "sample_ddim_steps final latent should diverge at guidance_scale=3.0"
        )


# ---------------------------------------------------------------------------
# LSDModel inference plumbing
# ---------------------------------------------------------------------------


class TestLSDModelGuidanceScale:
    def test_generate_sound_bank_forwards_guidance_scale(self) -> None:
        """generate_sound_bank(guidance_scale=0.0, target_c_spec) matches
        generate_sound_bank() without target_c_spec (same seed)."""
        m = make_model()
        perturb_adaln(m.dit)
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = m.generate_sound_bank(n=1, steps=4, seed=42)[0]
        guided = m.generate_sound_bank(
            n=1, steps=4, seed=42, target_c_spec=c_spec, guidance_scale=0.0
        )[0]

        assert torch.allclose(uncond, guided, atol=1e-4)
