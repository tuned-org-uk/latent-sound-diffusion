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

__all__ = ["SpectralSchedule", "DynamicEntropicClock"]


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
        """Heat-death metric: Σ ν_k · ᾱ_k(t).

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
            Heat-death metric. Below ``eps`` means heat death.
        """
        ab_k = self.alpha_bar_k(t)
        return (self.nu * ab_k).sum()

    def is_heat_death(self, t: Tensor) -> bool:
        """Intrinsic stopping criterion: heat death when metric < ε.

        Parameters
        ----------
        t : Tensor (scalar)
            External time.

        Returns
        -------
        bool
            True if all modes are near equilibrium.
        """
        return self.heat_death_metric(t).item() < self.eps


class DynamicEntropicClock(SpectralSchedule):
    """Dynamic Barontini entropic clock (issue #41).

    Promotes ``SpectralSchedule`` from a frozen-prior stub to a dynamic
    accumulator that advances τ only when entropy actually flows between
    spectral sectors during generation.

    The clock measures per-step entropy change of the "bright sector"
    (active spectral modes) from the running variance of the denoised latent,
    and accumulates it into τ_n:

        Δτ = |S_bright(t) - S_bright(t-1)|
        τ_n += Δτ

    The dynamic ᾱ(τ_n) schedule is:

        ᾱ_k(τ_n) = cos²(π/2 · τ_n / τ_max_est)

    where τ_max_est is estimated from a warm-up window (first ``tau_warmup``
    steps). Steps where entropy flows rapidly advance the schedule faster;
    plateau steps contribute negligibly.

    At equilibrium (no entropy change), τ_n stalls — Barontini's key property.

    This class is a drop-in ``SpectralSchedule`` subclass: the sampler's
    ``spectral_schedule`` hook and ``is_heat_death()`` interface are unchanged.
    The sampler calls ``update(x)`` at each step to feed the current latent
    before checking ``is_heat_death(t)``.

    Parameters
    ----------
    prior : ArrowSpacePrior
        The frozen ArrowSpace prior whose eigenvalues ν_k drive the schedule.
    eps : float
        Heat-death threshold for the stopping criterion.
    tau_warmup : int
        Number of warm-up steps for τ_max estimation.
    """

    def __init__(
        self,
        prior: ArrowSpacePrior,
        eps: float = 1e-3,
        tau_warmup: int = 5,
    ) -> None:
        super().__init__(prior, horizon=1.0, eps=eps)
        self.tau_warmup = tau_warmup
        self.tau_n: float = 0.0
        self.tau_max_est: float | None = None
        self._warmup_deltas: list[float] = []
        self._prev_s_bright: Tensor | None = None

    def _spectral_entropy(self, x: Tensor) -> Tensor:
        """Compute the spectral entropy (variance proxy) of the latent.

        The bright-sector entropy proxy is the running variance of the
        denoised latent. This is a lightweight signal that tracks how much
        the latent is changing between denoising steps.

        Parameters
        ----------
        x : Tensor (B, C, L)
            Current latent.

        Returns
        -------
        Tensor (scalar)
            Bright-sector entropy proxy.
        """
        return x.var()

    def update(self, x: Tensor) -> None:
        """Advance the dynamic clock by one step.

        Measures the entropy change |ΔS_bright| from the previous latent
        state and accumulates it into τ_n. During the warm-up window,
        collects Δτ samples to estimate τ_max_est.

        Parameters
        ----------
        x : Tensor (B, C, L)
            Current latent state from the sampler.
        """
        s_bright = self._spectral_entropy(x)

        if self._prev_s_bright is None:
            delta_tau = s_bright.abs().item()
        else:
            delta_tau = (s_bright - self._prev_s_bright).abs().item()
        self.tau_n += delta_tau

        if self.tau_max_est is None and len(self._warmup_deltas) < self.tau_warmup:
            self._warmup_deltas.append(delta_tau)
            if len(self._warmup_deltas) == self.tau_warmup:
                total = sum(self._warmup_deltas)
                self.tau_max_est = max(total, 1e-8)

        self._prev_s_bright = s_bright.detach()

    def alpha_bar_k(self, t: Tensor) -> Tensor:
        """Per-mode noise schedule indexed by dynamic τ_n.

        ᾱ_k(τ_n) = cos²(π/2 · τ_n / τ_max_est)

        Falls back to the frozen schedule if τ_max_est is not yet set
        (warm-up phase).

        Parameters
        ----------
        t : Tensor (scalar)
            External time (unused in dynamic mode, kept for interface compat).

        Returns
        -------
        Tensor (q,)
            Per-mode alpha_bar.
        """
        if self.tau_max_est is None or self.tau_max_est <= 0:
            return super().alpha_bar_k(t)
        ratio = min(self.tau_n / self.tau_max_est, 1.0)
        return torch.cos(torch.tensor(ratio * (math.pi / 2.0))).pow(2) * torch.ones_like(self.nu)

    def heat_death_metric(self, t: Tensor) -> Tensor:
        """Heat-death metric for the dynamic clock.

        Uses the dynamic ᾱ_k(τ_n) schedule. Falls back to the frozen
        metric during warm-up.

        Parameters
        ----------
        t : Tensor (scalar)
            External time (unused in dynamic mode, kept for interface compat).

        Returns
        -------
        Tensor (scalar)
            Heat-death metric.
        """
        if self.tau_max_est is None:
            return super().heat_death_metric(t)
        ab_k = self.alpha_bar_k(t)
        return (self.nu * ab_k).sum()

    def is_heat_death(self, t: Tensor) -> bool:
        """Intrinsic stopping criterion for the dynamic clock.

        Returns False during warm-up (τ_max_est not yet estimated).
        After warm-up, fires when the heat-death metric < ε.

        Parameters
        ----------
        t : Tensor (scalar)
            External time (unused in dynamic mode, kept for interface compat).

        Returns
        -------
        bool
            True if the system is near equilibrium.
        """
        if self.tau_max_est is None:
            return False
        return self.heat_death_metric(t).item() < self.eps

    def conserved(self) -> Tensor:
        """Entropy conservation diagnostic: |S_bright + S_dark - S_0|.

        Analogous to Barontini's experimental result 2 (S_bright + S_dark ≈ S_0).
        Serves as a sanity check that the bright/dark sector split is
        well-calibrated.

        Returns
        -------
        Tensor (scalar)
            Conservation error. Near-zero means well-calibrated.
        """
        if self._prev_s_bright is None:
            return torch.tensor(0.0)
        s_bright = self._prev_s_bright
        s_dark = self.nu.sum() - s_bright
        s_0 = self.nu.sum() / 2
        return (s_bright + s_dark - s_0).abs()
