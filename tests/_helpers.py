"""Shared test fixtures and helpers for the ALD-SC test suite.

Provides common constants, a stub encoder, a DiT factory, an LSDModel
factory, and the ``perturb_adaln`` helper used by conditioning/CFG tests
to unlock the zero-initialized AdaLN path on untrained models.

Imported as ``from _helpers import ...`` (``tests`` is on ``pythonpath``
via ``pyproject.toml``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule

# Shared dimensions used across conditioning / CFG tests.
SPEC_DIM = 24
LATENT_CH = 128
LATENT_LEN = 16


class StubEncoder(nn.Module):
    """Minimal encoder stub for LSDModel tests (no real EnCodec)."""

    def __init__(self, latent_dim: int = LATENT_CH, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        return z, a, prior.chart_energy_descriptor(a)


def make_dit() -> MinimalDiT:
    """Create a small MinimalDiT for testing (cfg_dropout=0)."""
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


def perturb_adaln(dit: MinimalDiT) -> None:
    """Make the conditioning path active on an untrained DiT.

    ``AdaLN.proj`` is zero-initialized by design (so an untrained model's
    AdaLN produces zero scale/shift and ``c_spec`` has no effect).  Perturb
    the AdaLN weights so that the conditioning vector actually steers the
    predicted velocity, mirroring the approach in ``test_dit.py``.  This
    does not train the model — it only unlocks the conditioning path so we
    can assert it is wired and responsive without a converged trajectory.
    """
    for block in dit.blocks:
        nn.init.normal_(block.adaln.proj.weight, std=0.02)
        nn.init.normal_(block.adaln.proj.bias, std=0.02)


def make_model() -> LSDModel:
    """Create a small LSDModel for inference-plumbing tests."""
    torch.manual_seed(0)
    embeddings = torch.randn(32, LATENT_CH)
    prior = build_arrow_prior(embeddings, q=8, k=4)
    decoder = GraphDecoder(LATENT_CH, 1, LATENT_CH, 16, prior, (2, 4, 5, 8))
    dit = make_dit()
    sched = CosineSchedule(num_steps=100)
    return LSDModel(
        prior=prior,
        dit=dit,
        decoder=decoder,
        encoder=StubEncoder(),
        schedule=sched,
        sample_rate=24000,
    )
