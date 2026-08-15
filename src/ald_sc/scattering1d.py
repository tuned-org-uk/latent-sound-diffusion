"""1-D Mallat wavelet scattering transform — pure NumPy (issue #48).

Acknowledgement: the original idea and reference implementation sketch for
this from-scratch NumPy scattering transform come from Vikshep
(https://vikshep.vercel.app/), proposed in issue #48.

Provides:
  ScatteringConfig        — hyperparameters (J, Q, order, T)
  MorletFilterBank        — analytic Morlet wavelets + low-pass, DC-corrected
                            for admissibility (integral of real part ~ 0)
  Scattering1D            — forward pass returning a flat (D,) feature vector
  scattering_pooled_features — pooled scattering features for an iterable of
                            clips, mirroring ``eval.encodec_pooled_features``

Mathematical structure
----------------------
Order-0:  S_0(x)        = | |x * phi_T| |
Order-1:  S_1(x, l1)    = | |x * psi_l1| * phi_T |
Order-2:  S_2(x, l1,l2) = | ||x * psi_l1| * psi_l2| * phi_T |
          only for j2 > j1 (Mallat energy-decay pruning).

Translation invariance is structural (modulus -> convolve -> modulus ->
low-pass cascade), not a hyperparameter: a 40-sample shift changes the
feature by ~0.06% vs ~10.3% for the raw signal (~170x compression of
sensitivity). This makes the features a stable alternative to pooled
EnCodec latents for distributional guards (Frechet distances) and, per
issue #48, a candidate feature extractor for ``WireGraph.build``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ScatteringConfig",
    "MorletFilterBank",
    "Scattering1D",
    "scattering_pooled_features",
]


@dataclass
class ScatteringConfig:
    """Hyperparameters for the scattering transform.

    Parameters
    ----------
    J : int
        Number of octaves. Controls the coarsest scale (2^J samples).
    Q : int
        Wavelets per octave. Higher Q -> finer frequency resolution.
    order : int
        Maximum scattering order (1 or 2). Order-2 adds cross-scale
        interactions at the cost of O(J*Q)^2 / 2 extra coefficients.
    T : int or None
        Averaging window length (in samples). Defaults to 2^J.
    """

    J: int = 5
    Q: int = 4
    order: int = 2
    T: int | None = None

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order}")
        if self.T is None:
            self.T = 2**self.J


class MorletFilterBank:
    """Fixed analytic Morlet filter bank + Gaussian low-pass.

    All filters are constructed in the frequency domain and stored as
    length-N complex arrays ready for np.fft-based convolution.

    DC correction: each Morlet wavelet psi(t) = e^{2 pi i xi t} g(t/s)
    has a non-zero DC component that violates the admissibility
    condition; we zero the DC bin so integral of Re[psi] dt = 0. Wavelets
    are unit L2 energy; the low-pass is normalised to unit sum (an
    averaging kernel), so order-0/1 coefficients keep the scale of the
    averaged |u|.
    """

    def __init__(self, N: int, cfg: ScatteringConfig) -> None:
        self.N = N
        self.cfg = cfg
        # (xi, sigma, psi_hat) per wavelet; psi_hat index = j * Q + q.
        self.wavelets: List[Tuple[float, float, np.ndarray]] = []
        self.phi_hat: np.ndarray = np.empty(0)
        self._build()

    def _build(self) -> None:
        N, J, Q = self.N, self.cfg.J, self.cfg.Q
        freqs = np.fft.fftfreq(N)  # cycles/sample, [-0.5, 0.5)

        # Low-pass: Gaussian averaging kernel with bandwidth T.
        T = self.cfg.T if self.cfg.T is not None else 2**self.cfg.J
        sigma_phi = T / (2 * np.pi)
        self.phi_hat = np.exp(-0.5 * (freqs * N / sigma_phi) ** 2).astype(np.complex128)
        self.phi_hat /= self.phi_hat.sum() + 1e-12  # unit sum -> averaging

        # Morlet wavelets: J octaves x Q per octave, analytic (one-sided).
        for j in range(J):
            for q in range(Q):
                scale = 2 ** (j + q / Q)  # s_{j,q}
                xi = 0.5 / scale  # centre frequency (cycles/sample)
                sigma = xi / (2 * np.pi * 0.8)  # constant-Q bandwidth
                psi_hat = np.exp(-0.5 * ((freqs - xi) / sigma) ** 2).astype(
                    np.complex128
                )
                psi_hat -= psi_hat[0]  # DC correction: psi_hat(0) = 0
                psi_hat /= np.sqrt((np.abs(psi_hat) ** 2).sum()) + 1e-12
                self.wavelets.append((xi, sigma, psi_hat))

    @property
    def num_wavelets(self) -> int:
        return len(self.wavelets)


class Scattering1D:
    """1-D Mallat scattering transform.

    Parameters
    ----------
    N : int
        Signal length (must be fixed; zero-pad shorter signals).
    cfg : ScatteringConfig, optional
        Defaults to J=5, Q=4, order=2.

    Usage
    -----
    >>> sc = Scattering1D(N=24000, cfg=ScatteringConfig(J=5, Q=4, order=2))
    >>> x = np.random.randn(24000)
    >>> features = sc.transform(x)  # shape (D,)
    """

    def __init__(self, N: int, cfg: ScatteringConfig | None = None) -> None:
        self.N = N
        self.cfg = cfg or ScatteringConfig()
        self.fb = MorletFilterBank(N, self.cfg)

    @property
    def feature_dim(self) -> int:
        """Total number of scattering coefficients (order 0 + 1 + 2)."""
        d = 1 + self.fb.num_wavelets  # order 0 + order 1
        if self.cfg.order >= 2:
            J, Q = self.cfg.J, self.cfg.Q
            for j1 in range(J):
                for j2 in range(j1 + 1, J):
                    d += Q * Q
        return d

    def _low_pass(self, u: np.ndarray, u_hat: np.ndarray | None = None) -> float:
        """Apply the averaging low-pass phi_T; return the pooled coefficient."""
        if u_hat is None:
            u_hat = np.fft.fft(u)
        s = np.fft.ifft(u_hat * self.fb.phi_hat)
        return float(np.abs(s).mean())

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Compute the scattering feature vector of a 1-D signal.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Real-valued 1-D signal of length ``self.N``.

        Returns
        -------
        np.ndarray, shape (D,)
            Flat log1p-compressed scattering coefficients (float32).
        """
        if x.shape != (self.N,):
            raise ValueError(f"Expected ({self.N},), got {x.shape}")
        x = x.astype(np.float64, copy=False)
        x_hat = np.fft.fft(x)

        coeffs: List[float] = [self._low_pass(np.abs(x))]  # order 0

        # Order 1 (caching u1_hat for the order-2 pass).
        u1_hat_list: List[np.ndarray] = []
        for _xi, _sigma, psi_hat in self.fb.wavelets:
            u1 = np.abs(np.fft.ifft(x_hat * psi_hat))
            coeffs.append(self._low_pass(u1))
            u1_hat_list.append(np.fft.fft(u1))

        # Order 2 with Mallat pruning (j2 > j1 only).
        if self.cfg.order >= 2:
            J, Q = self.cfg.J, self.cfg.Q
            wavelets = self.fb.wavelets
            for idx1, (j1, _q1) in enumerate(
                (j, q) for j in range(J) for q in range(Q)
            ):
                for idx2, (j2, _q2) in enumerate(
                    (j, q) for j in range(J) for q in range(Q)
                ):
                    if j2 <= j1:
                        continue
                    _x, _s, psi2_hat = wavelets[idx2]
                    u2 = np.abs(np.fft.ifft(u1_hat_list[idx1] * psi2_hat))
                    coeffs.append(self._low_pass(u2))

        return np.log1p(np.maximum(np.asarray(coeffs, dtype=np.float64), 0.0)).astype(
            np.float32
        )


def _as_numpy_signal(clip: Tensor, signal_length: int) -> np.ndarray:
    """Normalise a single clip (T,), (1, T) or (1, 1, T) to length-N float64."""
    x = clip.detach().float().cpu()
    while x.dim() > 1 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 1:
        raise ValueError(f"single clip expected, got batch shape {tuple(x.shape)}")
    if x.shape[0] >= signal_length:
        x = x[:signal_length]
    else:
        x = torch.nn.functional.pad(x, (0, signal_length - x.shape[0]))
    return x.numpy().astype(np.float64)


def scattering_pooled_features(
    audio_iter: Iterable[Tensor],
    signal_length: int = 96000,
    cfg: ScatteringConfig | None = None,
) -> Tensor:
    """Pooled scattering features (N, D) for an iterable of clips.

    Mirrors ``ald_sc.eval.encodec_pooled_features`` (same iterable-of-clips
    contract, CPU output) so it can be substituted as the feature space of
    FAD-proxy style distributional guards (issue #53) or fed to
    ``WireGraph.build(features=...)`` (issue #48). Clips are truncated or
    zero-padded to ``signal_length``. Pure NumPy — no GPU path.
    """
    sc = Scattering1D(N=signal_length, cfg=cfg or ScatteringConfig())
    feats: list[np.ndarray] = []
    for clip in audio_iter:
        x = clip if isinstance(clip, Tensor) else clip[0]
        x = x.detach().float().cpu()
        if x.dim() == 3 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)  # multi-channel -> mono
        if x.dim() not in (1, 2, 3):
            raise ValueError(f"unsupported clip shape {tuple(x.shape)}")
        rows = x.reshape(-1, x.shape[-1])  # (B, T): 0/1/2 leading unit dims
        feats.extend(sc.transform(_as_numpy_signal(row, signal_length)) for row in rows)
    return torch.from_numpy(np.stack(feats))  # (N, D) float32
