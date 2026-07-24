"""Tests for audio training loops.

Uses a stub encoder to avoid downloading EnCodec weights in tests.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ald_sc.audio_codec import AudioVAE, BaselineAudioDecoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_audio_decoder, train_audio_diffusion


class StubEncoder(nn.Module):
    """Stub encoder mimicking EnCodec interface for testing."""

    def __init__(self, latent_dim: int = 128, stride: int = 320) -> None:
        super().__init__()
        self.proj = nn.Conv1d(1, latent_dim, stride, stride=stride)

    def encode(self, x: Tensor, prior) -> tuple[Tensor, Tensor, Tensor]:
        z = self.proj(x).float()
        a = z.mean(dim=2)
        c_spec = prior.chart_energy_descriptor(a)
        return z, a, c_spec

    def extract_features(self, x: Tensor) -> Tensor:
        return self.proj(x).float()


def _make_dataloader(n: int = 8, audio_length: int = 320 * 16) -> DataLoader:
    torch.manual_seed(3407)
    data = torch.randn(n, 1, audio_length)
    return DataLoader(TensorDataset(data), batch_size=4, shuffle=False)


def _make_vae(latent_dim: int = 128, base_channels: int = 16) -> AudioVAE:
    encoder = StubEncoder(latent_dim=latent_dim)
    decoder = BaselineAudioDecoder(
        latent_channels=latent_dim,
        out_channels=1,
        base_channels=base_channels,
        upsample_strides=(2, 4, 5, 8),  # 320x to match stub encoder stride
    )
    return AudioVAE(encoder=encoder, decoder=decoder)


class TestTrainAudioDecoder:
    def test_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loss_fn = ALDSCLoss(
            prior=prior, lambda_rec=1.0, lambda_stft=0.0,
            lambda_chart=0.5, lambda_smooth=0.1,
        )
        loader = _make_dataloader(n=8, audio_length=320 * 16)

        losses = list(train_audio_decoder(loader, vae, prior, loss_fn, epochs=3, lr=1e-3))
        assert len(losses) > 0
        assert "loss" in losses[0]
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first * 1.5

    def test_encoder_stays_frozen(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loss_fn = ALDSCLoss(
            prior=prior, lambda_rec=1.0, lambda_stft=0.0,
        )
        loader = _make_dataloader(n=8, audio_length=320 * 16)

        list(train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3))

        for p in vae.encoder.parameters():
            assert not p.requires_grad

    def test_yields_loss_dict_keys(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae()
        loss_fn = ALDSCLoss(
            prior=prior, lambda_rec=1.0, lambda_stft=0.0,
            lambda_chart=0.5, lambda_smooth=0.1,
        )
        loader = _make_dataloader(n=4, audio_length=320 * 16)

        losses = list(train_audio_decoder(loader, vae, prior, loss_fn, epochs=1, lr=1e-3))
        assert len(losses) > 0
        for d in losses:
            assert "epoch" in d
            assert "loss" in d
            assert "rec" in d
            assert "stft" in d
            assert "chart" in d
            assert "smooth" in d


class TestTrainAudioDiffusion:
    def test_diffusion_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae(latent_dim=128, base_channels=16)
        loader = _make_dataloader(n=8, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=64,
            depth=2,
            num_heads=4,
            text_dim=0,
            spec_dim=24,
        )

        losses = list(
            train_audio_diffusion(
                loader, vae, dit, prior, sched, epochs=3, lr=1e-3
            )
        )
        assert len(losses) > 0
        assert "loss" in losses[0]

    def test_vae_stays_frozen(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        vae = _make_vae()
        loader = _make_dataloader(n=4, audio_length=320 * 16)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=64,
            depth=2,
            num_heads=4,
            spec_dim=24,
        )

        list(train_audio_diffusion(loader, vae, dit, prior, sched, epochs=1, lr=1e-3))

        vae_grads_after = [p.grad for p in vae.parameters()]
        assert all(g is None for g in vae_grads_after), "VAE must stay frozen"
