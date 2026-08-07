"""Tests for classifier-free guidance (CFG) two-pass inference (issue #23).

Covers:
- ``_cfg_forward`` formula correctness (two-pass blend matches the
  analytical ``v_uncond + s * (v_cond - v_uncond)``)
- ``guidance_scale=0.0`` produces unconditional output
- ``guidance_scale=1.0`` produces pure conditional output (single pass)
- ``guidance_scale>1.0`` produces amplified output (differs from both)
- ``guidance_scale`` ignored when ``c_spec is None``
- ``sample_ddim`` / ``sample_ddpm`` / ``sample_euler`` all support it
- ``LSDModel.generate_sound_bank`` forwards ``guidance_scale`` to the
  sampler

The AdaLN projection is zero-initialized by design on ``MinimalDiT``, so
an untrained model ignores ``c_spec``.  The ``_perturb_adaln`` helper
unlocks the conditioning path (mirroring ``test_dit.py``) so the tests can
assert CFG behaviour without a trained model.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.sampling import _cfg_forward, sample_ddim, sample_ddpm, sample_euler
from ald_sc.schedule import CosineSchedule


SPEC_DIM = 24
LATENT_CH = 128
LATENT_LEN = 16


def _make_dit() -> MinimalDiT:
    torch.manual_seed(0)
    return MinimalDiT(
        latent_channels=LATENT_CH,
        latent_length=LATENT_LEN,
        patch_size=4,
        dim=32,
        depth=1,
        num_heads=4,
        spec_dim=SPEC_DIM,
        cfg_dropout=0.0,
    )


def _perturb_adaln(dit: MinimalDiT) -> None:
    """Make the conditioning path active on an untrained DiT."""
    for block in dit.blocks:
        nn.init.normal_(block.adaln.proj.weight, std=0.02)
        nn.init.normal_(block.adaln.proj.bias, std=0.02)


def _make_model() -> LSDModel:
    torch.manual_seed(0)
    embeddings = torch.randn(32, LATENT_CH)
    prior = build_arrow_prior(embeddings, q=8, k=4)

    class StubEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv1d(1, LATENT_CH, 320, stride=320)

        def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
            z = self.proj(x).float()
            a = z.mean(dim=2)
            return z, a, prior.chart_energy_descriptor(a)

    decoder = GraphDecoder(LATENT_CH, 1, LATENT_CH, 16, prior, (2, 4, 5, 8))
    dit = _make_dit()
    sched = CosineSchedule(num_steps=100)
    return LSDModel(
        prior=prior,
        dit=dit,
        decoder=decoder,
        encoder=StubEncoder(),
        schedule=sched,
        sample_rate=24000,
    )


# ---------------------------------------------------------------------------
# _cfg_forward unit tests
# ---------------------------------------------------------------------------


class TestCfgForward:
    def test_formula_correctness(self) -> None:
        """Blended output must equal v_uncond + s * (v_cond - v_uncond)."""
        dit = _make_dit()
        _perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)
        scale = 3.0

        v_cond = dit(z, t, c_spec=c_spec)
        v_uncond = dit(z, t, c_spec=None)
        expected = v_uncond + scale * (v_cond - v_uncond)
        actual = _cfg_forward(dit, z, t, c_spec, scale)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_scale_1_is_pure_conditional(self) -> None:
        dit = _make_dit()
        _perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        expected = dit(z, t, c_spec=c_spec)
        actual = _cfg_forward(dit, z, t, c_spec, 1.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_scale_0_is_pure_unconditional(self) -> None:
        dit = _make_dit()
        _perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        expected = dit(z, t, c_spec=None)
        actual = _cfg_forward(dit, z, t, c_spec, 0.0)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_no_c_spec_ignores_scale(self) -> None:
        dit = _make_dit()
        _perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])

        expected = dit(z, t, c_spec=None)
        for scale in [0.0, 1.0, 3.0, 10.0]:
            actual = _cfg_forward(dit, z, t, None, scale)
            assert torch.allclose(actual, expected, atol=1e-6)

    def test_amplified_differs_from_both(self) -> None:
        """guidance_scale > 1 must differ from both pure cond and pure uncond."""
        dit = _make_dit()
        _perturb_adaln(dit)
        dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        v_cond = _cfg_forward(dit, z, t, c_spec, 1.0)
        v_uncond = _cfg_forward(dit, z, t, c_spec, 0.0)
        v_amplified = _cfg_forward(dit, z, t, c_spec, 3.0)

        assert not torch.allclose(v_amplified, v_cond, atol=1e-6)
        assert not torch.allclose(v_amplified, v_uncond, atol=1e-6)


# ---------------------------------------------------------------------------
# Sampler integration tests
# ---------------------------------------------------------------------------


class TestSamplerGuidanceScale:
    @pytest.mark.parametrize("sampler", [sample_ddim, sample_euler, sample_ddpm])
    def test_scale_0_matches_unconditional(self, sampler) -> None:
        """guidance_scale=0.0 + c_spec == c_spec=None (same seed)."""
        dit = _make_dit()
        _perturb_adaln(dit)
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
        dit = _make_dit()
        _perturb_adaln(dit)
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
        dit = _make_dit()
        _perturb_adaln(dit)
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
        dit = _make_dit()
        _perturb_adaln(dit)
        sched = CosineSchedule(num_steps=100)

        base = sample_ddim(dit, sched, c_spec=None, steps=10, seed=42)
        for scale in [0.0, 3.0, 10.0]:
            out = sample_ddim(
                dit, sched, c_spec=None, steps=10, seed=42, guidance_scale=scale
            )
            assert torch.allclose(base, out, atol=1e-5)


# ---------------------------------------------------------------------------
# LSDModel inference plumbing
# ---------------------------------------------------------------------------


class TestLSDModelGuidanceScale:
    def test_generate_sound_bank_forwards_guidance_scale(self) -> None:
        """generate_sound_bank(guidance_scale=0.0, target_c_spec) matches
        generate_sound_bank() without target_c_spec (same seed)."""
        m = _make_model()
        _perturb_adaln(m.dit)
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = m.generate_sound_bank(n=1, steps=4, seed=42)[0]
        guided = m.generate_sound_bank(
            n=1, steps=4, seed=42, target_c_spec=c_spec, guidance_scale=0.0
        )[0]

        assert torch.allclose(uncond, guided, atol=1e-4)
