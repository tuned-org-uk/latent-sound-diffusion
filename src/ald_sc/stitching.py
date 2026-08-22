"""Waveform-domain long-form stitching (Track B control arm).

Overlap-add of decoded segments with complementary equal-power windows
(sqrt-Hann halves): uncorrelated segments hold constant power through
every seam, unlike linear fades (-3 dB dip) or latent-space blending
(off the codec manifold). This is the notebook-07 pattern, corrected to
equal power and placed exactly on the overlap interval.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["equal_power_overlap_add"]


def _fade(overlap: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Complementary sqrt-Hann halves: gain_out^2 + gain_in^2 == 1."""
    t = torch.linspace(0.0, math.pi / 2, overlap, device=device)
    return t.cos(), t.sin()


def equal_power_overlap_add(segments: list[Tensor], overlap: int) -> Tensor:
    """Stitch mono segments (1, T) or (B, T) into one continuous render.

    Consecutive segments overlap by ``overlap`` samples; inside the
    overlap the earlier segment tapers by ``cos`` and the later one rises
    by ``sin``, so squared gains sum to unity at every sample. The first
    segment starts at full gain and the last ends at full gain.
    """
    if not segments:
        raise ValueError("equal_power_overlap_add requires at least one segment")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0; got {overlap}")
    shortest = min(int(seg.shape[-1]) for seg in segments)
    if overlap >= shortest:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than the shortest segment ({shortest})"
        )
    if len(segments) > 2:
        middle = min(int(seg.shape[-1]) for seg in segments[1:-1])
        if overlap * 2 >= middle:
            raise ValueError(
                f"middle segments carry head and tail fades: overlap*2 "
                f"({overlap * 2}) must stay below the shortest middle "
                f"segment ({middle})"
            )
    if any(seg.shape[:-1] != segments[0].shape[:-1] for seg in segments[1:]):
        raise ValueError("all segments must share the same leading shape (batch)")

    first = segments[0]
    lead = first.shape[:-1]
    total = sum(int(s.shape[-1]) for s in segments) - overlap * (len(segments) - 1)
    out = torch.zeros(*lead, total, dtype=first.dtype, device=first.device)

    fade_out, fade_in = _fade(overlap, first.device) if overlap > 0 else (None, None)

    pos = 0
    last = len(segments) - 1
    for i, seg in enumerate(segments):
        env = None
        if overlap > 0:
            head = fade_in.expand(*lead, -1) if i > 0 else None
            tail = fade_out.expand(*lead, -1) if i < last else None
            body_len = (
                int(seg.shape[-1])
                - overlap * (head is not None)
                - overlap * (tail is not None)
            )
            parts = [
                p
                for p in (
                    head,
                    torch.ones(*lead, body_len, dtype=seg.dtype, device=seg.device)
                    if body_len > 0
                    else None,
                    tail,
                )
                if p is not None
            ]
            env = torch.cat(parts, dim=-1) if parts else None
        windowed = seg if env is None else seg * env
        end = pos + int(seg.shape[-1])
        out[..., pos:end] += windowed
        pos = end - overlap if i < last else end
    return out
