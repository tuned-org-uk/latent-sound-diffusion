"""Per-mode entropic noise schedule from the ArrowSpace spectrum.

Implements the theoretical bridge from ALD-SC to ESDM's entropic clock
(paper §5). Each spectral mode k receives its own schedule driven by its
frozen Laplacian eigenvalue ν_k:

    τ_k(t) = ν_k · t
    ᾱ_k(τ_k) = cos²(π/2 · τ_k / τ_{k,max})
    τ_{k,max} = ν_k · T

where T is the reference horizon. All modes formally reach equilibrium at
the same external time T, but they traverse their trajectories at different
rates: high-ν_k modes (rough, local detail) accumulate entropic time faster
and cross the 1/2 transition earlier; low-ν_k modes (smooth, global
structure) persist throughout the denoising trajectory.

The entropy rate is:

    dS_k/dt = -ν_k

so each Laplacian eigenvalue *is* the entropy exchange rate for mode k.

Heat death (the intrinsic stopping criterion) is:

    heat death ⟺ Σ ν_k · mmse_k(t) < ε

This module is a thin reproducible stub demonstrating the per-mode formula
on frozen eigenvalues. It is decoupled from ESDM's full entropic clock
(wave recurrence, vibrational pump, density matrices), which remains
Phase 4 / future work.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior

__all__ = ["SpectralSchedule"]


class SpectralSchedule(nn.Module):
    """Per-mode entropic noise schedule driven by frozen Laplacian eigenvalues.

    Parameters
    ----------
    prior : ArrowSpacePrior
        The frozen ArrowSpace prior whose eigenvalues ν_k drive the schedule.
    horizon : float
        Reference horizon T. All modes reach equilibrium (ᾱ_k = 0) at t = T.
    eps : float
        Heat-death threshold for the stopping criterion.
    """

    def __init__(
        self,
        prior: ArrowSpacePrior,
        horizon: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.eps = eps
        self.q = prior.q

        # Declared buffer types for static checkers (see arrow_prior.py).
        self.nu: Tensor
        self.tau_k_max: Tensor
        self.register_buffer("nu", prior.eigvals_q.clone())
        self.register_buffer("tau_k_max", prior.eigvals_q.clone() * horizon)

    def tau_k(self, t: Tensor) -> Tensor:
        """Per-mode entropic time: τ_k(t) = ν_k · t.

        Parameters
        ----------
        t : Tensor (scalar)
            External time.

        Returns
        -------
        Tensor (q,)
            Per-mode entropic time.
        """
        return self.nu * t

    def alpha_bar_k(self, t: Tensor) -> Tensor:
        """Per-mode noise schedule: ᾱ_k = cos²(π/2 · τ_k / τ_{k,max}).

        Parameters
        ----------
        t : Tensor (scalar)
            External time.

        Returns
        -------
        Tensor (q,)
            Per-mode alpha_bar. At t=0 all modes are 1; at t=T all modes are 0.
        """
        tau = self.tau_k(t)
        ratio = (tau / self.tau_k_max).clamp(0.0, 1.0)
        return torch.cos(ratio * (math.pi / 2.0)).pow(2)

    def entropy_rate(self) -> Tensor:
        """Entropy exchange rate per mode: dS_k/dt = -ν_k.

        Returns
        -------
        Tensor (q,)
            Negative eigenvalues (entropy decreases over time).
        """
        return -self.nu

    def heat_death_metric(self, t: Tensor) -> Tensor:
        """Heat-death metric (forward direction): Σ ν_k · ᾱ_k(t).

        This approximates the MMSE-weighted entropy rate from the paper.
        At t=0 the metric is high (modes are active); at t=T it approaches 0
        (all modes at equilibrium).

        Parameters
        ----------
        t : Tensor (scalar)
            External time.

        Returns
        -------
        Tensor (scalar)
            Heat-death metric of the forward process.
        """
        ab_k = self.alpha_bar_k(t)
        return (self.nu * ab_k).sum()

    def remaining_dissipation(self, t: Tensor) -> Tensor:
        """Remaining dissipation of the REVERSE process: Σ ν_k · (1 − ᾱ_k(t)).

        During denoising (t: 1 → 0) each mode resolves on its own entropic
        clock; this measures how much spectral structure is still unresolved.
        It is maximal at t=1 (pure noise) and monotonically decreases to 0 as
        sampling completes. Heat death of the reverse process = nothing
        measurable left to resolve: ``remaining_dissipation(t) < eps``.

        Parameters
        ----------
        t : Tensor (scalar)
            External time in [0, 1].

        Returns
        -------
        Tensor (scalar)
            Remaining dissipation, normalized by Σ ν_k: a fraction in
            [0, 1] (1 = pure noise, 0 = fully resolved), so ``eps`` is
            scale-free with respect to the prior's spectrum.
        """
        ab_k = self.alpha_bar_k(t)
        total = self.nu.sum()
        if float(total.item()) <= 0.0:
            return torch.zeros((), device=self.nu.device, dtype=self.nu.dtype)
        return (self.nu * (1.0 - ab_k)).sum() / total

    def is_heat_death(self, t: Tensor) -> bool:
        """Intrinsic stopping criterion (reverse process): heat death when
        the remaining dissipation Σ ν_k (1 − ᾱ_k(t)) < ε.

        Parameters
        ----------
        t : Tensor (scalar)
            External time.

        Returns
        -------
        bool
            True if all modes are near equilibrium (nothing left to resolve).
        """
        return bool(self.remaining_dissipation(t).item() < self.eps)
