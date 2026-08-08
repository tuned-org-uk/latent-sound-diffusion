"""Tests for the SpectralVAE encoder-decoder."""

from __future__ import annotations

import torch
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.vae import SpectralVAE


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestSpectralVAE:
    def test_forward_shapes(self) -> None:
        prior = _make_prior(f=32, q=8)
        vae = SpectralVAE(
            in_channels=3,
            latent_channels=4,
            feature_dim=32,
            base_channels=32,
        )
        x = torch.randn(2, 3, 32, 32)
        z, A, c_spec, x_hat = vae(x, prior)
        assert z.shape == (2, 4, 8, 8)
        assert A.shape == (2, 32)
        assert c_spec.shape == (2, 24)
        assert x_hat.shape == x.shape

    def test_gradient_flow(self) -> None:
        prior = _make_prior(f=32, q=8)
        vae = SpectralVAE(
            in_channels=3,
            latent_channels=4,
            feature_dim=32,
            base_channels=32,
        )
        x = torch.randn(2, 3, 32, 32)
        z, A, c_spec, x_hat = vae(x, prior)
        loss = x_hat.pow(2).mean() + z.pow(2).mean()
        loss.backward()
        for p in vae.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_c_spec_changes_with_input(self) -> None:
        prior = _make_prior(f=32, q=8)
        vae = SpectralVAE(
            in_channels=3,
            latent_channels=4,
            feature_dim=32,
            base_channels=32,
        )
        x1 = torch.randn(2, 3, 32, 32)
        x2 = torch.randn(2, 3, 32, 32)
        _, _, c1, _ = vae(x1, prior)
        _, _, c2, _ = vae(x2, prior)
        assert not torch.allclose(c1, c2, atol=1e-6), "c_spec must change with different inputs"

    def test_kl_loss(self) -> None:
        prior = _make_prior(f=32, q=8)
        vae = SpectralVAE(
            in_channels=3,
            latent_channels=4,
            feature_dim=32,
            base_channels=32,
        )
        x = torch.randn(2, 3, 32, 32)
        z, A, c_spec, x_hat = vae(x, prior)
        kl = vae.kl_loss()
        assert kl.dim() == 0
        assert kl >= 0
