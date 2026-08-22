"""Tests for configuration loading and checkpoint geometry validation."""

from __future__ import annotations

import math

import pytest
import torch

from ald_sc.config import (
    DEFAULT_CONFIG,
    load_config,
    resolve_geometry,
    validate_dit_state_dict,
)


class TestLoadConfig:
    def test_none_uses_defaults(self) -> None:
        cfg = load_config(None)
        assert cfg == DEFAULT_CONFIG

    def test_missing_explicit_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_yaml_overrides_defaults(self, tmp_path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("dit:\n  depth: 6\n  dim: 128\n")
        cfg = load_config(path)
        assert cfg["dit"]["depth"] == 6
        assert cfg["dit"]["dim"] == 128
        assert cfg["dit"]["latent_channels"] == DEFAULT_CONFIG["dit"]["latent_channels"]

    def test_partial_section_merges(self, tmp_path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("dit:\n  num_heads: 8\n")
        cfg = load_config(path)
        assert cfg["dit"]["num_heads"] == 8
        assert cfg["dit"]["patch_size"] == DEFAULT_CONFIG["dit"]["patch_size"]


class TestResolveGeometry:
    def test_defaults_resolve(self) -> None:
        geo = resolve_geometry(load_config(None))
        assert geo["latent_length"] == DEFAULT_CONFIG["dit"]["latent_length"]
        assert geo["num_patches"] == math.ceil(geo["latent_length"] / geo["patch_size"])

    def test_cli_overrides_config(self, tmp_path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("dit:\n  latent_length: 375\n")
        geo = resolve_geometry(load_config(path), overrides={"latent_length": 750})
        assert geo["latent_length"] == 750
        assert geo["num_patches"] == 94


class TestValidateDitStateDict:
    def _state_dict(self, channels=4, length=16, patch=2, dim=32):
        from ald_sc.dit import MinimalDiT

        torch.manual_seed(3407)
        return MinimalDiT(
            latent_channels=channels,
            latent_length=length,
            patch_size=patch,
            dim=dim,
            depth=1,
            num_heads=4,
        ).state_dict()

    def test_valid_checkpoint_passes(self) -> None:
        sd = self._state_dict()
        validate_dit_state_dict(sd, latent_channels=4, latent_length=16, patch_size=2)

    def test_length_mismatch_raises_with_hint(self) -> None:
        sd = self._state_dict(length=16)
        with pytest.raises(
            ValueError, match="latent_length.*interpolate|interpolate.*latent_length"
        ):
            validate_dit_state_dict(
                sd, latent_channels=4, latent_length=32, patch_size=2
            )

    def test_channel_mismatch_raises(self) -> None:
        sd = self._state_dict(channels=4)
        with pytest.raises(ValueError, match="latent_channels"):
            validate_dit_state_dict(
                sd, latent_channels=8, latent_length=16, patch_size=2
            )

    def test_patch_size_mismatch_raises(self) -> None:
        sd = self._state_dict(patch=2)
        with pytest.raises(ValueError, match="patch_size"):
            validate_dit_state_dict(
                sd, latent_channels=4, latent_length=16, patch_size=4
            )
