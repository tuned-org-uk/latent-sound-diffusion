"""Training loops for ALD-SC.

Provides ``train_vae()`` for Phase 1 spectral VAE training and
``train_diffusion()`` for Phase 2 latent diffusion training.

Training loops yield loss dicts (mirrors the from-scratch pattern) so
notebooks can plot live curves.

This module must not define model architectures (per AGENTS.md §11).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.losses import ALDSCLoss
from ald_sc.schedule import CosineSchedule
from ald_sc.vae import SpectralVAE

__all__ = ["train_vae", "train_diffusion"]


def train_vae(
    loader: DataLoader,
    vae: SpectralVAE,
    prior: ArrowSpacePrior,
    loss_fn: ALDSCLoss,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Iterator[dict[str, float]]:
    """Phase 1 spectral VAE training loop.

    Parameters
    ----------
    loader : DataLoader
        Image dataloader.
    vae : SpectralVAE
        The VAE to train.
    prior : ArrowSpacePrior
        Frozen ArrowSpace prior.
    loss_fn : ALDSCLoss
        Loss function instance.
    epochs : int
    lr : float
    device : torch.device

    Yields
    ------
    dict[str, float]
        Loss dict with 'epoch', 'loss', 'rec', 'chart', 'smooth'.
    """
    vae = vae.to(device)
    prior = prior.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
            optimizer.zero_grad()

            z, A, c_spec, x_hat = vae(x, prior)

            A_hat = A.detach()
            mu = vae._last_mu
            logvar = vae._last_logvar
            losses = loss_fn(x, x_hat, A, A_hat, mu=mu, logvar=logvar)

            losses["total"].backward()
            optimizer.step()

            yield {
                "epoch": epoch,
                "loss": float(losses["total"].item()),
                "rec": float(losses["rec"].item()),
                "chart": float(losses["chart"].item()),
                "smooth": float(losses["smooth"].item()),
            }


def train_diffusion(
    loader: DataLoader,
    vae: SpectralVAE,
    dit: nn.Module,
    prior: ArrowSpacePrior,
    schedule: CosineSchedule,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    cfg_dropout: float = 0.1,
) -> Iterator[dict[str, float]]:
    """Phase 2 latent diffusion training loop.

    The VAE and prior are frozen; only the DiT is trained.

    Parameters
    ----------
    loader : DataLoader
    vae : SpectralVAE
        Frozen VAE (encoder used for z0 and c_spec).
    dit : nn.Module
        DiT denoiser to train.
    prior : ArrowSpacePrior
        Frozen prior.
    schedule : CosineSchedule
        Noise schedule.
    epochs : int
    lr : float
    device : torch.device
    cfg_dropout : float
        Probability of dropping c_spec (classifier-free guidance).

    Yields
    ------
    dict[str, float]
    """
    vae = vae.to(device).eval()
    prior = prior.to(device)
    dit = dit.to(device)
    schedule = schedule

    for p in vae.parameters():
        p.requires_grad_(False)
    for p in prior.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(dit.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in loader:
            x = batch.to(device) if isinstance(batch, Tensor) else batch[0].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                z0, A, c_spec, _ = vae(x, prior)

            t = schedule.sample_batch(z0)
            noise = torch.randn_like(z0)
            z_t = schedule.add_noise(z0, t, noise)
            v_target = schedule.v_target(z0, t, noise)

            if cfg_dropout > 0:
                mask = torch.rand(z0.shape[0], device=device) < cfg_dropout
                c_spec = c_spec.clone()
                c_spec[mask] = 0.0

            v_pred = dit(z_t, t, c_spec=c_spec)
            loss = (v_pred - v_target).pow(2).mean()
            loss.backward()
            optimizer.step()

            yield {
                "epoch": epoch,
                "loss": float(loss.item()),
            }
