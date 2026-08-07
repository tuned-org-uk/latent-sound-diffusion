"""Inference samplers for ALD-SC latent diffusion.

Provides deterministic (Euler, DDIM eta=0) and stochastic (DDIM eta>0,
DDPM ancestral) samplers that turn noise into a 1-D audio latent, which
can then be decoded through the graph decoder to produce a waveform.

Sampler diversity spectrum (low → high):
  sample_euler / sample_ddim(eta=0)  →  sample_ddim(eta∈(0,1))  →  sample_ddpm

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

__all__ = ["sample_euler", "sample_ddim", "sample_ddim_steps", "sample_ddpm"]


def _cfg_forward(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    c_spec: Tensor | None,
    guidance_scale: float,
) -> Tensor:
    """Forward pass with classifier-free guidance (CFG).

    Standard CFG formulation::

        v = v_uncond + s * (v_cond - v_uncond)

    where ``s`` is the guidance scale.  Special cases are short-circuited
    to a single forward pass for efficiency:

    - ``c_spec is None``       → unconditional (guidance is meaningless)
    - ``guidance_scale == 1`` → pure conditional
    - ``guidance_scale == 0`` → pure unconditional

    Only ``0 < s ≠ 1`` triggers the two-pass blend.

    Parameters
    ----------
    model : nn.Module
        DiT denoiser predicting velocity v.
    x : Tensor
        Current latent.
    t : Tensor
        Timestep indices.
    c_spec : Tensor or None
        Spectral conditioning vector (``None`` = unconditional).
    guidance_scale : float
        CFG scale: 1.0 = pure conditional, 0.0 = pure unconditional,
        >1.0 = amplified conditioning.

    Returns
    -------
    Tensor
        Guided velocity prediction.
    """
    if c_spec is None or guidance_scale == 1.0:
        return model(x, t, c_spec=c_spec)
    if guidance_scale == 0.0:
        return model(x, t, c_spec=None)
    v_cond = model(x, t, c_spec=c_spec)
    v_uncond = model(x, t, c_spec=None)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


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
    guidance_scale: float = 1.0,
) -> Tensor | tuple[Tensor, int]:
    """Euler sampler for latent diffusion.

    Starts from pure noise at t=T and steps backward to t=0.
    Fully deterministic (ODE solver, no stochastic noise injection).
    For sample diversity, prefer ``sample_ddim(eta>0)`` or ``sample_ddpm``.

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
        ``Σ ν_k · ᾱ_k(t) < ε`` is met.
    return_steps : bool
        If True, return (z, steps_used).
    guidance_scale : float
        Classifier-free guidance scale.  1.0 = pure conditional (single
        pass), 0.0 = pure unconditional, >1.0 = amplified conditioning
        (two-pass: conditional + unconditional, blended).  Only effective
        when ``c_spec`` is provided; ignored otherwise.

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
        v = _cfg_forward(model, x, t, c_spec, guidance_scale)

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
    eta: float = 0.0,
    guidance_scale: float = 1.0,
) -> Tensor | tuple[Tensor, int]:
    """DDIM sampler for latent diffusion, with optional stochastic noise.

    When ``eta=0`` (default) this is the standard deterministic DDIM ODE
    solver — identical output for any given seed.  Setting ``eta > 0``
    injects per-step ancestral noise, interpolating between the ODE
    (eta=0) and the full DDPM SDE (eta=1).  Use ``eta=1`` to match DDPM
    diversity while keeping DDIM's flexible step-count.

    The stochastic variance at each step follows the DDIM generalisation
    (Song et al. 2020)::

        sigma_t = eta * sqrt((1 - ab_prev) / (1 - ab)) * sqrt(1 - ab / ab_prev)

    The reverse update then becomes::

        x_prev = sqrt(ab_prev) * z0_pred
               + sqrt(1 - ab_prev - sigma_t**2) * direction
               + sigma_t * eps

    where ``direction`` is the predicted direction toward x_t and
    ``eps ~ N(0, I)`` is fresh noise.

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
    eta : float
        Stochasticity coefficient in [0, 1].  ``eta=0`` is fully
        deterministic DDIM; ``eta=1`` matches DDPM noise level.
        Values > 1 are valid but increase variance beyond DDPM.
        Must be >= 0; negative values raise ``ValueError``.
    guidance_scale : float
        Classifier-free guidance scale.  1.0 = pure conditional (single
        pass), 0.0 = pure unconditional, >1.0 = amplified conditioning
        (two-pass: conditional + unconditional, blended).  Only effective
        when ``c_spec`` is provided; ignored otherwise.

    Returns
    -------
    Tensor
        Denoised latent z_0. If return_steps, returns (z, steps_used).

    Raises
    ------
    ValueError
        If ``eta < 0``.
    """
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")

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
        v = _cfg_forward(model, x, t, c_spec, guidance_scale)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()

        # Predict clean latent z0 from v-parameterisation
        z0_pred = sqrt_ab * x - sqrt_1mab * v

        sqrt_ab_prev = ab_prev.sqrt()

        if eta == 0.0:
            # Deterministic DDIM — no noise injection
            x = sqrt_ab_prev * z0_pred + (1 - ab_prev).sqrt() * v
        else:
            # Stochastic DDIM-ancestral update (Song et al. 2020, eq. 12)
            sigma_t = (
                eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            )
            # Coefficient for the "direction pointing to x_t" term
            coef_direction = (1 - ab_prev - sigma_t**2).clamp(min=0.0).sqrt()
            direction = (x - sqrt_ab * z0_pred) / (sqrt_1mab + 1e-8)
            noise = torch.randn_like(x, generator=gen)
            x = sqrt_ab_prev * z0_pred + coef_direction * direction + sigma_t * noise

    if return_steps:
        return x, steps_used
    return x


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: CosineSchedule,
    c_spec: Tensor | None = None,
    batch_size: int = 1,
    steps: int = 50,
    seed: int = 3407,
    device: torch.device = torch.device("cpu"),
    spectral_schedule: SpectralSchedule | None = None,
    return_steps: bool = False,
    guidance_scale: float = 1.0,
) -> Tensor | tuple[Tensor, int]:
    """Full ancestral DDPM sampler for latent diffusion.

    Implements the standard DDPM reverse process (Ho et al. 2020) with
    per-step Gaussian noise injection.  This is the maximum-diversity
    sampler and is recommended for creative generation tasks where
    sample variety matters more than reconstruction fidelity.

    The update rule at each step is::

        beta_t      = 1 - ab / ab_prev
        mean        = (1/sqrt(1 - beta_t)) * (x - beta_t/sqrt(1-ab) * eps_pred)
        sigma_t     = sqrt(beta_t * (1 - ab_prev) / (1 - ab))
        x_prev      = mean + sigma_t * N(0, I)

    where ``eps_pred`` is the predicted noise derived from the model's
    v-parameterisation.  Given ``v = sqrt(ab)*eps - sqrt(1-ab)*z0``,
    inverting for ``eps`` yields::

        eps_pred = sqrt(ab) * v + sqrt(1 - ab) * x

    Note that ``sigma_t = 0`` at the final step (t=0) so no noise is
    added to the last iterate.

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
        If provided, sampling stops early at heat-death criterion.
    return_steps : bool
        If True, return (z, steps_used).
    guidance_scale : float
        Classifier-free guidance scale.  1.0 = pure conditional (single
        pass), 0.0 = pure unconditional, >1.0 = amplified conditioning
        (two-pass: conditional + unconditional, blended).  Only effective
        when ``c_spec`` is provided; ignored otherwise.

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
    sigma_list = sigmas.tolist()
    for sig, sig_prev in _pairwise(sigma_list):
        if spectral_schedule is not None:
            t_frac = sig / schedule.num_steps
            if spectral_schedule.is_heat_death(torch.tensor(t_frac, device=device)):
                break
        steps_used += 1
        t = torch.full((batch_size,), int(sig), device=device, dtype=torch.long)
        v = _cfg_forward(model, x, t, c_spec, guidance_scale)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()

        # Recover predicted noise eps from v-parameterisation:
        #   v = sqrt(ab)*eps - sqrt(1-ab)*z0  =>  eps = sqrt(ab)*v + sqrt(1-ab)*x
        eps_pred = sqrt_ab * v + sqrt_1mab * x

        # DDPM posterior mean (Ho et al. 2020, eq. 11)
        beta_t = 1.0 - ab / ab_prev
        coef_x = 1.0 / (1.0 - beta_t).sqrt()
        coef_eps = beta_t / (sqrt_1mab + 1e-8)
        mean = coef_x * (x - coef_eps * eps_pred)

        # No noise on the final step. Use sig_prev == sigma_list[-1] rather than
        # an index check so this stays correct when spectral_schedule causes early
        # exit (the index-based i == len-2 would be wrong in that case).
        is_last_step = sig_prev == sigma_list[-1]
        if is_last_step or ab_prev >= 1.0 - 1e-6:
            x = mean
        else:
            sigma_t = (beta_t * (1 - ab_prev) / ((1 - ab) + 1e-8)).sqrt()
            noise = torch.randn_like(x, generator=gen)
            x = mean + sigma_t * noise

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
    eta: float = 0.0,
    guidance_scale: float = 1.0,
) -> Iterator[Tensor]:
    """Yield intermediate latents during DDIM sampling (for visualization).

    If spectral_schedule is provided, stops early at heat death.

    Parameters
    ----------
    eta : float
        Stochasticity coefficient forwarded to the DDIM update rule.
        ``eta=0`` (default) is fully deterministic.  Must be >= 0.
    guidance_scale : float
        Classifier-free guidance scale.  1.0 = pure conditional (single
        pass), 0.0 = pure unconditional, >1.0 = amplified conditioning
        (two-pass).  Only effective when ``c_spec`` is provided.

    Raises
    ------
    ValueError
        If ``eta < 0``.
    """
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")

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
        v = _cfg_forward(model, x, t, c_spec, guidance_scale)

        ab = schedule.alpha_bar[int(sig)]
        ab_prev = schedule.alpha_bar[int(sig_prev)]
        sqrt_ab = ab.sqrt()
        sqrt_1mab = (1 - ab).sqrt()

        z0_pred = sqrt_ab * x - sqrt_1mab * v
        sqrt_ab_prev = ab_prev.sqrt()

        if eta == 0.0:
            x = sqrt_ab_prev * z0_pred + (1 - ab_prev).sqrt() * v
        else:
            sigma_t = (
                eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            )
            coef_direction = (1 - ab_prev - sigma_t**2).clamp(min=0.0).sqrt()
            direction = (x - sqrt_ab * z0_pred) / (sqrt_1mab + 1e-8)
            noise = torch.randn_like(x, generator=gen)
            x = sqrt_ab_prev * z0_pred + coef_direction * direction + sigma_t * noise
        yield x
