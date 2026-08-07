"""Tests for c_spec conditioning of the DiT (issue #22 acceptance criteria).

Covers:
- generate_sound_bank with target_c_spec diverges from unconditioned baseline
- two distinct c_spec vectors produce different outputs (same seed)
- _sample_and_decode raises ValueError when z_init + c_spec_override conflict
- train_audio_diffusion raises TypeError for a DiT without cfg_dropout
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_audio_diffusion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SPEC_DIM = 24
LATENT_CH = 128
LATENT_LEN = 16


class StubEncoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_CH, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        return z, a, prior.chart_energy_descriptor(a)


def _make_model() -> LSDModel:
    torch.manual_seed(0)
    embeddings = torch.randn(32, LATENT_CH)
    prior = build_arrow_prior(embeddings, q=8, k=4)
    encoder = StubEncoder()
    decoder = GraphDecoder(LATENT_CH, 1, LATENT_CH, 16, prior, (2, 4, 5, 8))
    dit = MinimalDiT(
        latent_channels=LATENT_CH,
        latent_length=LATENT_LEN,
        patch_size=4,
        dim=32,
        depth=1,
        num_heads=4,
        spec_dim=SPEC_DIM,
    )
    sched = CosineSchedule(num_steps=100)
    return LSDModel(
        prior=prior, dit=dit, decoder=decoder,
        encoder=encoder, schedule=sched, sample_rate=24000,
    )


# ---------------------------------------------------------------------------
# c_spec conditioning divergence tests
# ---------------------------------------------------------------------------

class TestCSpecConditioningDivergence:
    """Acceptance criterion: sampling with a specific c_spec produces
    different output than sampling with a different c_spec."""

    def test_conditioned_bank_differs_from_unconditioned(self) -> None:
        """generate_sound_bank with target_c_spec diverges from no-c_spec baseline."""
        m = _make_model()
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = m.generate_sound_bank(n=1, steps=4, seed=42)[0]
        cond = m.generate_sound_bank(n=1, steps=4, seed=42, target_c_spec=c_spec)[0]

        assert not torch.allclose(uncond, cond), (
            "Conditioned and unconditioned banks produced identical output on "
            "the same seed — c_spec is not being forwarded to the DiT."
        )

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_two_different_c_specs_produce_different_outputs(self, seed: int) -> None:
        """Two distinct c_spec vectors must yield different latents (same seed)."""
        m = _make_model()
        torch.manual_seed(seed + 1)
        c_spec_a = torch.randn(1, SPEC_DIM)
        c_spec_b = torch.randn(1, SPEC_DIM)

        out_a = m.generate_sound_bank(n=1, steps=4, seed=seed, target_c_spec=c_spec_a)[0]
        out_b = m.generate_sound_bank(n=1, steps=4, seed=seed, target_c_spec=c_spec_b)[0]

        assert not torch.allclose(out_a, out_b), (
            f"Two different c_spec vectors produced identical outputs (seed={seed}). "
            "The DiT spec_proj is not influencing the output."
        )


# ---------------------------------------------------------------------------
# _sample_and_decode conflict guard
# ---------------------------------------------------------------------------

class TestSampleAndDecodeConflict:
    def test_raises_when_z_init_and_c_spec_both_provided(self) -> None:
        """Providing both z_init and c_spec_override must raise ValueError."""
        m = _make_model()
        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        c_spec = torch.randn(1, SPEC_DIM)

        with pytest.raises(ValueError, match="mutually exclusive"):
            m._sample_and_decode(
                seed=0, steps=2, temperature=1.0,
                z_init=z, c_spec_override=c_spec,
            )


# ---------------------------------------------------------------------------
# cfg_dropout guard in train_audio_diffusion
# ---------------------------------------------------------------------------

class TestCfgDropoutGuard:
    def test_raises_type_error_for_non_conformant_dit(self) -> None:
        """train_audio_diffusion must raise TypeError if DiT lacks cfg_dropout."""
        torch.manual_seed(0)
        embeddings = torch.randn(32, LATENT_CH)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        sched = CosineSchedule(num_steps=100)

        class NoCfgDiT(nn.Module):
            latent_shape = (LATENT_CH, LATENT_LEN)

            def forward(self, z_t, t, c_spec=None):
                return torch.zeros_like(z_t)

        class StubVAE(nn.Module):
            class encoder(nn.Module):
                @staticmethod
                def encode(x, prior):
                    z = torch.zeros(x.shape[0], LATENT_CH, LATENT_LEN)
                    a = z.mean(dim=2)
                    return z, a, prior.chart_energy_descriptor(a)

            def parameters(self):
                return iter([])

        dummy_data = torch.randn(2, 1, LATENT_LEN * 320)
        loader = DataLoader(TensorDataset(dummy_data), batch_size=2)
        vae = StubVAE()

        with pytest.raises(TypeError, match="cfg_dropout"):
            # Exhaust the generator to trigger the error.
            list(train_audio_diffusion(
                loader=loader,
                audio_vae=vae,
                dit=NoCfgDiT(),
                prior=prior,
                schedule=sched,
                epochs=1,
            ))
