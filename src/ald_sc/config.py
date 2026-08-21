"""Configuration authority and checkpoint geometry validation.

Single source of truth for model geometry: YAML config files merge over
``DEFAULT_CONFIG`` and CLI flags override the merged result. Loading a
DiT checkpoint against a declared geometry is validated here so a
mismatched ``latent_length`` fails loudly instead of silently producing a
wrong-shape broadcast error deep inside forward.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import yaml

__all__ = [
    "DEFAULT_CONFIG",
    "load_config",
    "resolve_geometry",
    "validate_dit_state_dict",
]

DEFAULT_CONFIG: dict = {
    "dit": {
        "latent_channels": 128,
        "latent_length": 375,
        "patch_size": 8,
        "dim": 256,
        "depth": 4,
        "num_heads": 4,
    },
}


def load_config(path: str | Path | None) -> dict:
    """Load a YAML config merged over ``DEFAULT_CONFIG``.

    ``None`` returns pristine defaults. Sections absent from the file keep
    their defaults; unknown sections are preserved as-is.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    loaded = yaml.safe_load(Path(path).read_text()) or {}
    for section, values in loaded.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def resolve_geometry(cfg: dict, overrides: dict | None = None) -> dict:
    """Merge config geometry with explicit CLI overrides (None-skipping).

    Returns the resolved DiT geometry including derived ``num_patches``.
    """
    dit = dict(cfg["dit"])
    for key, value in (overrides or {}).items():
        if value is not None:
            dit[key] = value
    patch_size = int(dit["patch_size"])
    latent_length = int(dit["latent_length"])
    return {
        **dit,
        "latent_channels": int(dit["latent_channels"]),
        "patch_size": patch_size,
        "latent_length": latent_length,
        "num_patches": math.ceil(latent_length / patch_size),
    }


def validate_dit_state_dict(
    state_dict: dict,
    latent_channels: int,
    latent_length: int,
    patch_size: int,
) -> None:
    """Check checkpoint tensor shapes against declared DiT geometry.

    Raises ValueError naming the mismatched tensor and the repair options
    (rebuild at the checkpoint's true geometry, or interpolate the
    position table).
    """
    weight = state_dict.get("patch_embed.weight")
    pos_embed = state_dict.get("pos_embed")
    if weight is None or pos_embed is None:
        raise ValueError(
            "state_dict is missing patch_embed.weight/pos_embed; not a MinimalDiT checkpoint"
        )
    dim = weight.shape[0]
    if tuple(weight.shape) != (dim, latent_channels, patch_size):
        raise ValueError(
            f"patch_embed.weight shape {tuple(weight.shape)} does not match declared "
            f"geometry (latent_channels={latent_channels}, patch_size={patch_size})"
        )
    expected_pos = (1, math.ceil(latent_length / patch_size), dim)
    if tuple(pos_embed.shape) != expected_pos:
        raise ValueError(
            f"pos_embed shape {tuple(pos_embed.shape)} does not match declared "
            f"geometry (latent_length={latent_length}, patch_size={patch_size} -> "
            f"{expected_pos}); rebuild the DiT at the checkpoint's true "
            f"latent_length, or interpolate the position table before loading"
        )
