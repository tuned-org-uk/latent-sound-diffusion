"""Tests for 1-D clock-gated decoding tempo.

The SpectralSchedule governs when each spectral mode resolves during
decoding. The decoder can also use the heat-death metric to decide when
reconstruction is complete.
"""

from __future__ import annotations

import torch
from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.graph_decoder import ClockGatedGraphDecoder, WaveReconstructionBlock
from ald_sc.spectral_schedule import SpectralSchedule


def _make_prior(f: int = 32, q: int = 8) -> ArrowSpacePrior:
    torch.manual_seed(3407)
    embeddings = torch.randn(64, f)
    return build_arrow_prior(embeddings, q=q, k=4)


def _make_schedule(f: int = 32, q: int = 8) -> tuple[ArrowSpacePrior, SpectralSchedule]:
    prior = _make_prior(f, q)
    sched = SpectralSchedule(prior, horizon=1.0, eps=1e-3)
    return prior, sched


class TestClockGatedGraphDecoder:
    def test_forward_shape(self) -> None:
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        assert x_hat.shape == (2, 1, 64)

    def test_gradient_flow(self) -> None:
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16, requires_grad=True)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        x_hat.sum().backward()
        assert z.grad is not None

    def test_c_spec_affects_output(self) -> None:
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        decoder.eval()
        z = torch.randn(2, 4, 16)
        c1 = torch.randn(2, 24)
        c2 = torch.randn(2, 24)
        out1 = decoder(z, c1)
        out2 = decoder(z, c2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_clock_gates_mode_resolution(self) -> None:
        """The clock should modulate gate strengths based on diffusion time."""
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        decoder.eval()
        z = torch.randn(1, 4, 16)
        c_spec = torch.randn(1, 24)

        x_early = decoder(z, c_spec, diffusion_time=torch.tensor(0.9))
        x_late = decoder(z, c_spec, diffusion_time=torch.tensor(0.1))
        assert not torch.allclose(x_early, x_late, atol=1e-6)

    def test_prior_and_schedule_frozen(self) -> None:
        """Both prior and spectral schedule must stay frozen."""
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        x_hat.sum().backward()
        for buf in prior.buffers():
            assert buf.grad is None or (buf.grad == 0).all()
        for buf in sched.buffers():
            assert buf.grad is None or (buf.grad == 0).all()

    def test_uses_wave_blocks(self) -> None:
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        wave_blocks = [m for m in decoder.modules() if isinstance(m, WaveReconstructionBlock)]
        assert len(wave_blocks) >= 2

    def test_diffusion_time_default_works(self) -> None:
        """Without explicit diffusion_time, decoder should still work."""
        prior, sched = _make_schedule()
        decoder = ClockGatedGraphDecoder(
            latent_channels=4,
            out_channels=1,
            feature_dim=32,
            base_channels=32,
            prior=prior,
            spectral_schedule=sched,
            upsample_strides=(2, 2),
        )
        z = torch.randn(2, 4, 16)
        c_spec = torch.randn(2, 24)
        x_hat = decoder(z, c_spec)
        assert x_hat.shape == (2, 1, 64)
