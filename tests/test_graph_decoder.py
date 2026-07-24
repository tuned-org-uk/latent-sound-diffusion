"""Tests for the 1-D graph-structured decoder.

The graph decoder uses L_F (via U_q) to define reconstruction paths and
λ_ED to gate energy allocation. This is the constructive decoding operator
described in docs/00.md § "The research programme".

A wave-based reconstruction block propagates information along the graph's
smooth directions, weighted by the dispersion network — the graph-theoretic
analogue of the VAE reparameterization trick.
"""

from __future__ import annotations

import torch

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.graph_decoder import GraphDecoder, WaveReconstructionBlock


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


class TestWaveReconstructionBlock:
    def test_forward_shape(self) -> None:
        prior = _make_prior(f=32, q=8)
        block = WaveReconstructionBlock(
            channels=32,
            feature_dim=32,
            prior=prior,
        )
        h = torch.randn(2, 32, 16)
        c_spec = torch.randn(2, 24)
        out = block(h, c_spec)
        assert out.shape == h.shape

    def test_gradient_flow(self) -> None:
        prior = _make_prior(f=32, q=8)
        block = WaveReconstructionBlock(
            channels=32,
            feature_dim=32,
            prior=prior,
        )
        h = torch.randn(2, 32, 16, requires_grad=True)
        c_spec = torch.randn(2, 24)
        out = block(h, c_spec)
        out.sum().backward()
        assert h.grad is not None

    def test_c_spec_affects_output(self) -> None:
        prior = _make_prior(f=32, q=8)
        block = WaveReconstructionBlock(
            channels=32,
            feature_dim=32,
            prior=prior,
        )
        block.eval()
        h = torch.randn(2, 32, 16)
        c1 = torch.randn(2, 24)
        c2 = torch.randn(2, 24)
        out1 = block(h, c1)
        out2 = block(h, c2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_uses_laplacian_structure(self) -> None:
        """The block must use U_q from the prior (not just a conv)."""
        prior = _make_prior(f=32, q=8)
        block = WaveReconstructionBlock(
            channels=32,
            feature_dim=32,
            prior=prior,
        )
        assert hasattr(block, "U_q")
        assert block.U_q.shape == (32, 8)


class TestGraphDecoder:
    def test_forward_shape(self) -> None:
        prior = _make_prior(f=32, q=8)
        decoder = GraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        # 16 * (2*2) = 64
        assert x_hat.shape == (2, 1, 64)

    def test_gradient_flow(self) -> None:
        prior = _make_prior(f=32, q=8)
        decoder = GraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16, requires_grad=True)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        x_hat.sum().backward()
        assert z.grad is not None

    def test_c_spec_affects_output(self) -> None:
        prior = _make_prior(f=32, q=8)
        decoder = GraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 2),
        )
        decoder.eval()
        z = torch.randn(2, 4, 16)
        c1 = torch.randn(2, 24)
        c2 = torch.randn(2, 24)
        out1 = decoder(z, c1)
        out2 = decoder(z, c2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_uses_wave_blocks(self) -> None:
        """The decoder must contain WaveReconstructionBlock instances."""
        prior = _make_prior(f=32, q=8)
        decoder = GraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 2),
        )
        wave_blocks = [
            m for m in decoder.modules() if isinstance(m, WaveReconstructionBlock)
        ]
        assert len(wave_blocks) >= 1, "Decoder must use WaveReconstructionBlock"

    def test_prior_buffers_not_trained(self) -> None:
        """The prior's U_q and L_F must remain frozen in the decoder."""
        prior = _make_prior(f=32, q=8)
        decoder = GraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        x_hat.sum().backward()
        for buf in prior.buffers():
            assert buf.grad is None or (buf.grad == 0).all()
