"""Inference samplers for ALD-SC latent diffusion.

Provides Euler and DDIM-style deterministic samplers that turn noise into
a latent, which can then be decoded through the frozen VAE to produce an image.

This module must not add training logic (per AGENTS.md §11).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from ald_sc.schedule import CosineSchedule

__all__ = ["sample_euler", "sample_ddim", "sample_ddim_steps"]


def _pairwise(iterable: list[Tensor]) -> Iterator[tuple[Tensor, Tensor]]:
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for curr in it:
        yield prev, curr
        prev = curr


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Euler sampler for latent diffusion.

    Starts from pure noise at t=T and steps backward to t=0.

    Parameters
    ----------
    model : nn.Module
        DiT denoiser predicting velocity v.
    schedule : CosineSchedule
        Noise schedule.
    c_spec : Tensor (B, spec_dim), optional
        Spectral conditioning vector.
    batch_size : int
    steps : int
        Number of sampling steps.
    seed : int
    device : torch.device

    Returns
    -------
    Tensor (B, latent_channels, latent_size, latent_size)
        Denoised latent z_0.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    # Infer latent shape from the model
    latent_channels = getattr(model, "latent_channels", 4)
    latent_size = getattr(model, "latent_size", 32)

    x = torch.randn(
        batch_size,
        latent_channels,
        latent_size,
        latent_size,
        device=device,
        generator=gen,
    )

    for sig, sig_prev in _pairwise(sigmas.tolist()):
        t = torch.full((batch_size,), int(sig), device=device, dtype=torch.long)
        v = model(x, t, c_spec=c_spec)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_1mab_prev = (1 - ab_prev).sqrt()
        sqrt_ab = ab.sqrt()

        # v = sqrt(ab)*eps - sqrt(1-ab)*z0
        # z_t = sqrt(ab)*z0 + sqrt(1-ab)*eps
        # eps = (z_t - sqrt(ab)*z0) / sqrt(1-ab) = (sqrt(ab)*v + z_t) / (2*sqrt_ab) ...
        # Simpler: from v and z_t, recover z0 = sqrt(ab)*z_t - sqrt(1-ab)*v
        z0_pred = sqrt_ab * x - (1 - ab).sqrt() * v
        x = sqrt_ab_prev * z0_pred + sqrt_1mab_prev * (x - sqrt_ab * z0_pred) / (
            (1 - ab).sqrt() + 1e-8
        )

    return x


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """DDIM-style deterministic sampler for latent diffusion.

    Parameters
    ----------
    model : nn.Module
        DiT denoiser predicting velocity v.
    schedule : CosineSchedule
    c_spec : Tensor (B, spec_dim), optional
    batch_size : int
    steps : int
    seed : int
    device : torch.device

    Returns
    -------
    Tensor (B, latent_channels, latent_size, latent_size)
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    latent_channels = getattr(model, "latent_channels", 4)
    latent_size = getattr(model, "latent_size", 32)

    x = torch.randn(
        batch_size,
        latent_channels,
        latent_size,
        latent_size,
        device=device,
        generator=gen,
    )

    for sig, sig_prev in _pairwise(sigmas.tolist()):
        t = torch.full((batch_size,), int(sig), device=device, dtype=torch.long)
        v = model(x, t, c_spec=c_spec)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()

        # Recover z0 prediction from v
        z0_pred = sqrt_ab * x - sqrt_1mab * v

        # DDIM deterministic step
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_1mab_prev = (1 - ab_prev).sqrt()
        x = sqrt_ab_prev * z0_pred + sqrt_1mab_prev * v

    return x


def sample_ddim_steps(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
) -> Iterator[Tensor]:
    """Yield intermediate latents during DDIM sampling (for visualization)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    latent_channels = getattr(model, "latent_channels", 4)
    latent_size = getattr(model, "latent_size", 32)

    x = torch.randn(
        batch_size,
        latent_channels,
        latent_size,
        latent_size,
        device=device,
        generator=gen,
    )
    yield x

    for sig, sig_prev in _pairwise(sigmas.tolist()):
        t = torch.full((batch_size,), int(sig), device=device, dtype=torch.long)
        v = model(x, t, c_spec=c_spec)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()

        z0_pred = sqrt_ab * x - sqrt_1mab * v

        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_1mab_prev = (1 - ab_prev).sqrt()
        x = sqrt_ab_prev * z0_pred + sqrt_1mab_prev * v
        yield x
