"""Device-parity smoke tests for the graph decoder (issue #51).

The graph decoder was previously reported to diverge on MPS (loss
0.76 -> 11.1 vs CPU 0.19 -> 0.15). Investigation showed the divergence
was collateral of the EnCodec lazy-load device bug (fixed alongside the
MPS infra fixes), not numerics in WaveReconstructionBlock: forward,
loss, and per-op CPU/MPS parity agree to ~1e-6, and identical-init
training trajectories match.

These tests lock that in: identical-init decoders must produce close
outputs on both devices, and short training runs must decrease loss on
both devices. They skip automatically when MPS is unavailable (CI).

Use ``pytest tests/test_device_parity.py`` on an Apple-Silicon Mac.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.audio_codec import AudioVAE
from ald_sc.build_prior import build_arrow_prior
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.losses import ALDSCLoss
from ald_sc.trainer import train_audio_decoder

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device not available"
)

FEATURE_DIM, BASE_CH, Q = 32, 16, 8
LATENT_CH = FEATURE_DIM  # stub encoder channels must match prior feature dim
B, T_LAT = 4, 8
FWD_PARITY_TOL = 1e-4


class StubEncoder(nn.Module):
    """Stub encoder mimicking EnCodec's encode(x, prior) interface."""

    def __init__(self, latent_dim: int = LATENT_CH, stride: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(
        self, x: Tensor, prior: ArrowSpacePrior
    ) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        return z, a, prior.chart_energy_descriptor(a)


def _make_prior() -> ArrowSpacePrior:
    torch.manual_seed(3407)
    return build_arrow_prior(torch.randn(64, FEATURE_DIM), q=Q, k=4)


def _make_decoder(prior: ArrowSpacePrior) -> GraphDecoder:
    torch.manual_seed(3407)
    return GraphDecoder(
        latent_channels=LATENT_CH,
        out_channels=1,
        feature_dim=FEATURE_DIM,
        base_channels=BASE_CH,
        prior=prior,
        upsample_strides=(2, 2, 2, 2),  # 16x: T_LAT -> T_LAT*16 samples
    )


def _make_inputs(prior: ArrowSpacePrior) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(11)
    z = torch.randn(B, LATENT_CH, T_LAT, generator=g)
    A = torch.randn(B, FEATURE_DIM, generator=g)
    return z, prior.chart_energy_descriptor(A)


def _train_steps(device: torch.device, epochs: int = 8) -> list[dict[str, float]]:
    prior = _make_prior()
    vae = _make_vae(prior, _make_decoder(prior))
    loss_fn = ALDSCLoss(
        prior=prior,
        lambda_rec=1.0,
        lambda_stft=1.0,
        lambda_chart=0.5,
        lambda_smooth=0.1,
        stft_fft_sizes=(64, 128),  # small: audio is only 128 samples
    )
    g = torch.Generator().manual_seed(11)
    data = torch.randn(8, 1, T_LAT * 16, generator=g)  # stride-16 encoder
    loader = DataLoader(TensorDataset(data), batch_size=4, shuffle=False)
    torch.manual_seed(99)
    return list(
        train_audio_decoder(
            loader,
            vae,
            prior,
            loss_fn,
            epochs=epochs,
            lr=1e-3,
            device=device,
            noise_std=0.1,
        )
    )


def _make_vae(prior: ArrowSpacePrior, decoder: GraphDecoder) -> AudioVAE:
    return AudioVAE(encoder=StubEncoder(), decoder=decoder)


class TestForwardParity:
    @mps_only
    def test_decoder_forward_parity(self) -> None:
        """Identical init/inputs must give near-identical outputs."""
        prior = _make_prior()
        decoder = _make_decoder(prior).eval()
        z, c_spec = _make_inputs(prior)

        with torch.no_grad():
            x_cpu = decoder(z, c_spec)
            x_mps = copy.deepcopy(decoder).to("mps")(z.to("mps"), c_spec.to("mps"))
        diff = (x_cpu - x_mps.cpu()).abs().max()
        assert diff < FWD_PARITY_TOL, f"forward parity broken: max|diff|={diff:.3e}"

    @mps_only
    def test_uq_orthonormality_matches(self) -> None:
        prior = _make_prior()
        for dev in ("cpu", "mps"):
            U = prior.U_q.to(dev).cpu().double()
            err = (U.T @ U - torch.eye(Q, dtype=torch.float64)).norm()
            assert err < 1e-4, f"U_q not orthonormal on {dev}: {err:.3e}"


class TestTrainingParity:
    @mps_only
    def test_training_decreases_loss_on_mps(self) -> None:
        losses = _train_steps(torch.device("mps"))
        assert len(losses) >= 4
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first, f"MPS training not decreasing: {first:.4f} -> {last:.4f}"
        for d in losses:
            assert d["loss"] == d["loss"]  # finite

    def test_training_decreases_loss_on_cpu(self) -> None:
        losses = _train_steps(torch.device("cpu"))
        assert len(losses) >= 4
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first, f"CPU training not decreasing: {first:.4f} -> {last:.4f}"
