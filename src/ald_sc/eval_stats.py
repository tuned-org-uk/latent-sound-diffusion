"""Statistics for PROTOCOL_10S: per-frame diversity and TOST-style CIs.

``cross_clip_frame_excess`` measures whether different seeds explore
different frame-level content (the pooled pairwise metric saturates at
the contraction floor). ``bootstrap_fad_ci`` + ``equivalence_verdict``
implement the protocol's CI-within-margin equivalence test on the
FAD-proxy statistic.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

__all__ = ["cross_clip_frame_excess", "bootstrap_fad_ci", "equivalence_verdict"]


def _mean_pair_distance(a: Tensor, b: Tensor) -> float:
    return float(torch.cdist(a, b).mean().item())


def cross_clip_frame_excess(clouds: list[Tensor]) -> float:
    """Excess cross-clip frame distance over within-clip temporal spread.

    Parameters
    ----------
    clouds : list[Tensor]
        Per-clip frame clouds ``(K, F)``, L2-normalized frames.

    Returns
    -------
    float
        ``mean_{i≠j} D(i, j) − mean_i S_i`` where ``D`` is the mean
        pairwise frame distance across two clips and ``S_i`` the within-
        clip temporal spread. Near zero ⇒ every clip explores the same
        frame cloud (contraction); large positive ⇒ seeds genuinely
        diverge at frame level.
    """
    n = len(clouds)
    if n < 2:
        return 0.0
    within = [_mean_pair_distance(c, c) for c in clouds if c.shape[0] > 1]
    cross = [
        _mean_pair_distance(clouds[i], clouds[j])
        for i, j in itertools.combinations(range(n), 2)
    ]
    excess = sum(cross) / len(cross)
    if within:
        excess -= sum(within) / len(within)
    return float(excess)


def bootstrap_fad_ci(
    arm_feats: Tensor,
    ref_feats: Tensor,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 3407,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the FAD-proxy over arm resamples.

    Resamples the generated arm's clips with replacement (seed-family
    proxy: each clip is one family member) and recomputes the FAD-proxy
    against the fixed reference set. Returns the ``(1−alpha)`` central
    percentile interval.
    """
    from ald_sc.eval import fad_score

    gen = torch.Generator().manual_seed(seed)
    n = int(arm_feats.shape[0])
    values: list[float] = []
    for _ in range(int(n_boot)):
        idx = torch.randint(0, n, (n,), generator=gen)
        values.append(fad_score(arm_feats[idx], ref_feats))
    values.sort()
    lo_i = int((alpha / 2.0) * n_boot)
    hi_i = min(int(n_boot - 1), int((1.0 - alpha / 2.0) * n_boot))
    return float(values[lo_i]), float(values[hi_i])


def equivalence_verdict(ci_low: float, ci_high: float, margin: float) -> str:
    """CI-within-margin (TOST-style) verdict for 'quality unchanged'.

    Returns one of:
    - ``"equivalent"``: the CI lies entirely inside ±margin;
    - ``"inferior"``: entirely above +margin (worse than the baseline);
    - ``"superior"``: entirely below −margin (better than baseline — not
      an equivalence claim);
    - ``"inconclusive"``: otherwise.
    """
    if margin <= 0:
        raise ValueError(f"margin must be > 0; got {margin}")
    if ci_low >= -margin and ci_high <= margin:
        return "equivalent"
    if ci_low > margin:
        return "inferior"
    if ci_high < -margin:
        return "superior"
    return "inconclusive"
