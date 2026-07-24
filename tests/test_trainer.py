"""Tests for training loops.

The VAE training test uses the legacy image SpectralVAE (still 2-D).
The diffusion training tests are skipped until the audio trainer is
implemented (issue #6), since the 1-D DiT cannot consume 4-D image
VAE latents.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_vae, train_diffusion
from ald_sc.vae import SpectralVAE


def _make_dataloader(n: int = 8, image_size: int = 32) -> DataLoader:
    torch.manual_seed(3407)
    data = torch.randn(n, 3, image_size, image_size)
    return DataLoader(TensorDataset(data), batch_size=4, shuffle=False)


class TestTrainVAE:
    def test_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 16)
        prior = build_arrow_prior(embeddings, q=4, k=4)
        vae = SpectralVAE(
            in_channels=3, latent_channels=4, feature_dim=16, base_channels=16
        )
        loss_fn = ALDSCLoss(
            prior=prior, lambda_rec=1.0, lambda_chart=0.5, lambda_smooth=0.1
        )
        loader = _make_dataloader(n=8, image_size=32)

        losses = list(train_vae(loader, vae, prior, loss_fn, epochs=3, lr=1e-3))
        assert len(losses) > 0
        assert "loss" in losses[0]
        first = losses[0]["loss"]
        last = losses[-1]["loss"]
        assert last < first * 1.5


class TestTrainDiffusion:
    @pytest.mark.skip(
        reason="train_diffusion will be rewritten for 1-D audio in issue #6; "
        "the 1-D DiT cannot consume 4-D image VAE latents"
    )
    def test_diffusion_loss_decreases(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 16)
        prior = build_arrow_prior(embeddings, q=4, k=4)
        vae = SpectralVAE(
            in_channels=3, latent_channels=4, feature_dim=16, base_channels=16
        )
        ALDSCLoss(prior=prior)
        loader = _make_dataloader(n=8, image_size=32)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=4,
            latent_length=8,
            patch_size=2,
            dim=32,
            depth=2,
            num_heads=4,
            text_dim=0,
            spec_dim=12,
        )

        losses = list(
            train_diffusion(
                loader, vae, dit, prior, sched, epochs=3, lr=1e-3, cfg_dropout=0.0
            )
        )
        assert len(losses) > 0
        assert "loss" in losses[0]

    @pytest.mark.skip(
        reason="train_diffusion will be rewritten for 1-D audio in issue #6"
    )
    def test_vae_stays_frozen(self) -> None:
        torch.manual_seed(3407)
        embeddings = torch.randn(32, 16)
        prior = build_arrow_prior(embeddings, q=4, k=4)
        vae = SpectralVAE(
            in_channels=3, latent_channels=4, feature_dim=16, base_channels=16
        )
        loader = _make_dataloader(n=8, image_size=32)
        sched = CosineSchedule(num_steps=100)

        for p in vae.parameters():
            p.requires_grad_(False)

        dit = MinimalDiT(
            latent_channels=4,
            latent_length=8,
            patch_size=2,
            dim=32,
            depth=2,
            num_heads=4,
            text_dim=0,
            spec_dim=12,
        )

        [p.grad is not None for p in vae.parameters()]
        list(
            train_diffusion(
                loader, vae, dit, prior, sched, epochs=1, lr=1e-3, cfg_dropout=0.0
            )
        )
        vae_grads_after = [p.grad for p in vae.parameters()]
        assert all(g is None for g in vae_grads_after), "VAE must stay frozen"
