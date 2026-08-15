"""Inference samplers for ALD-SC latent diffusion.

Provides Euler and DDIM-style deterministic samplers that turn noise into
a 1-D audio latent, which can then be decoded through the graph decoder
to produce a waveform.

When a ``SpectralSchedule`` is provided, the sampler uses the
Barontini-inspired heat-death stopping criterion: sampling terminates
when ``Σ ν_k · ᾱ_k(t) < ε``, rather than at a fixed step count.

This module must not add training logic (per AGENTS.md §11).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from ald_sc.schedule import CosineSchedule
from ald_sc.spectral_schedule import SpectralSchedule

__all__ = ["sample_euler", "sample_ddim", "sample_ddim_steps"]


def _pairwise(iterable: list[int]) -> Iterator[tuple[int, int]]:
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for curr in it:
        yield prev, curr
        prev = curr


def _init_noise(
    model: nn.Module, batch_size: int, device: torch.device, gen: torch.Generator
) -> Tensor:
    """Initialise noise latent from the model's latent_shape attribute."""
    latent_channels = getattr(model, "latent_channels", 4)
    latent_shape = getattr(model, "latent_shape", None)
    if latent_shape is not None:
        return torch.randn(batch_size, *latent_shape, device=device, generator=gen)
    # Fallback for models without latent_shape
    latent_length = getattr(model, "latent_length", None)
    if latent_length is not None:
        return torch.randn(
            batch_size, latent_channels, latent_length, device=device, generator=gen
        )
    latent_size = getattr(model, "latent_size", 32)
    return torch.randn(
        batch_size,
        latent_channels,
        latent_size,
        latent_size,
        device=device,
        generator=gen,
    )


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
    spectral_schedule: SpectralSchedule | None = None,
    return_steps: bool = False,
) -> Tensor | tuple[Tensor, int]:
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
        Maximum number of sampling steps.
    seed : int
    device : torch.device
    spectral_schedule : SpectralSchedule, optional
        If provided, sampling stops early when the heat-death criterion
        remaining dissipation ``Σ ν_k · (1 − ᾱ_k(t)) < ε`` is met.
    return_steps : bool
        If True, return (z, steps_used).

    Returns
    -------
    Tensor
        Denoised latent z_0. If return_steps, returns (z, steps_used).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    x = _init_noise(model, batch_size, device, gen)

    steps_used = 0
    for sig, sig_prev in _pairwise(sigmas.tolist()):
        if spectral_schedule is not None:
            t_frac = sig / schedule.num_steps
            if spectral_schedule.is_heat_death(torch.tensor(t_frac, device=device)):
                break
        steps_used += 1
        t = torch.full((batch_size,), int(sig), device=device, dtype=torch.long)
        v = model(x, t, c_spec=c_spec)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_1mab_prev = (1 - ab_prev).sqrt()
        sqrt_ab = ab.sqrt()

        z0_pred = sqrt_ab * x - (1 - ab).sqrt() * v
        x = sqrt_ab_prev * z0_pred + sqrt_1mab_prev * (x - sqrt_ab * z0_pred) / (
            (1 - ab).sqrt() + 1e-8
        )

    if return_steps:
        return x, steps_used
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
    spectral_schedule: SpectralSchedule | None = None,
    return_steps: bool = False,
) -> Tensor | tuple[Tensor, int]:
    """DDIM-style deterministic sampler for latent diffusion.

    Parameters
    ----------
    model : nn.Module
        DiT denoiser predicting velocity v.
    schedule : CosineSchedule
    c_spec : Tensor (B, spec_dim), optional
    batch_size : int
    steps : int
        Maximum number of sampling steps.
    seed : int
    device : torch.device
    spectral_schedule : SpectralSchedule, optional
    return_steps : bool
        If True, return (z, steps_used).

    Returns
    -------
    Tensor
        If return_steps, returns (z, steps_used).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    x = _init_noise(model, batch_size, device, gen)

    steps_used = 0
    for sig, sig_prev in _pairwise(sigmas.tolist()):
        if spectral_schedule is not None:
            t_frac = sig / schedule.num_steps
            if spectral_schedule.is_heat_death(torch.tensor(t_frac, device=device)):
                break
        steps_used += 1
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

    if return_steps:
        return x, steps_used
    return x


def sample_ddim_steps(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
    spectral_schedule: SpectralSchedule | None = None,
) -> Iterator[Tensor]:
    """Yield intermediate latents during DDIM sampling (for visualization).

    If spectral_schedule is provided, stops early at heat death.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    if c_spec is not None:
        c_spec = c_spec.to(device)

    sigmas = schedule.sample_sigmas(steps).to(device)

    x = _init_noise(model, batch_size, device, gen)
    yield x

    for sig, sig_prev in _pairwise(sigmas.tolist()):
        if spectral_schedule is not None:
            t_frac = sig / schedule.num_steps
            if spectral_schedule.is_heat_death(torch.tensor(t_frac, device=device)):
                break
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
