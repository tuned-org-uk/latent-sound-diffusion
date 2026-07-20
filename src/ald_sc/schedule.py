"""Noise schedules for latent diffusion.

Provides cosine and linear schedules with v-prediction support. The cosine
schedule follows ScheduleLDM (Nichol & Dhariwal 2021); the linear schedule
follows the original DDPM formulation.

Both schedules expose ``add_noise`` and ``v_target`` for v-prediction training:

    z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps
    v   = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * z_0

The round-trip recovers z_0:

    z_0 = sqrt(alpha_bar_t) * z_t - sqrt(1 - alpha_bar_t) * v
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["CosineSchedule", "LinearSchedule"]


class CosineSchedule:
    """Cosine noise schedule (ScheduleLDM-style).

    alpha_bar(t) = cos^2(pi/2 * (t/T + s) / (1 + s))

    Parameters
    ----------
    num_steps : int
        Number of discrete timesteps T.
    s : float
        Small offset to prevent alpha_bar from being exactly 1 at t=0.
    """

    def __init__(self, num_steps: int = 1000, s: float = 0.008) -> None:
        self.num_steps = num_steps
        self.s = s
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        self.alpha_bar = f[:num_steps]
        self.alpha_bar = self.alpha_bar / self.alpha_bar[0]

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, i: int) -> Tensor:
        return self.alpha_bar[i]

    def add_noise(self, z0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward corruption: z_t = sqrt(ab) * z0 + sqrt(1-ab) * noise."""
        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, *([1] * (z0.dim() - 1)))
        sqrt_1mab = (1 - ab).sqrt().view(-1, *([1] * (z0.dim() - 1)))
        return sqrt_ab * z0 + sqrt_1mab * noise

    def v_target(self, z0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Velocity target: v = sqrt(ab) * noise - sqrt(1-ab) * z0."""
        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, *([1] * (z0.dim() - 1)))
        sqrt_1mab = (1 - ab).sqrt().view(-1, *([1] * (z0.dim() - 1)))
        return sqrt_ab * noise - sqrt_1mab * z0

    def sample_batch(self, x0: Tensor) -> Tensor:
        """Sample random timesteps for a batch."""
        return torch.randint(0, self.num_steps, (x0.shape[0],))

    def sample_sigmas(self, steps: int) -> Tensor:
        """Subsample timesteps for deterministic samplers (DDIM-style)."""
        indices = (
            self.num_steps * (1 - torch.arange(steps) / steps)
        ).round().long() - 1
        return indices.clamp(0, self.num_steps - 1)


class LinearSchedule:
    """Linear noise schedule (DDPM-style).

    alpha_bar(t) = 1 - t/T  (linear in beta)

    Parameters
    ----------
    num_steps : int
        Number of discrete timesteps T.
    beta_start : float
        Starting noise rate.
    beta_end : float
        Ending noise rate.
    """

    def __init__(
        self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> None:
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, i: int) -> Tensor:
        return self.alpha_bar[i]

    def add_noise(self, z0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward corruption: z_t = sqrt(ab) * z0 + sqrt(1-ab) * noise."""
        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, *([1] * (z0.dim() - 1)))
        sqrt_1mab = (1 - ab).sqrt().view(-1, *([1] * (z0.dim() - 1)))
        return sqrt_ab * z0 + sqrt_1mab * noise

    def v_target(self, z0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Velocity target: v = sqrt(ab) * noise - sqrt(1-ab) * z0."""
        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt().view(-1, *([1] * (z0.dim() - 1)))
        sqrt_1mab = (1 - ab).sqrt().view(-1, *([1] * (z0.dim() - 1)))
        return sqrt_ab * noise - sqrt_1mab * z0

    def sample_batch(self, x0: Tensor) -> Tensor:
        """Sample random timesteps for a batch."""
        return torch.randint(0, self.num_steps, (x0.shape[0],))

    def sample_sigmas(self, steps: int) -> Tensor:
        """Subsample timesteps for deterministic samplers (DDIM-style)."""
        indices = (
            self.num_steps * (1 - torch.arange(steps) / steps)
        ).round().long() - 1
        return indices.clamp(0, self.num_steps - 1)
