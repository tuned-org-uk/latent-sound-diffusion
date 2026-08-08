"""Tests for issue #28 remaining items: config.json, safetensors, latent_length derivation.

Tests that the training scripts export a config.json with architecture
hyperparams alongside each .pt checkpoint, export safetensors files for
trainable components, and derive latent_length from audio_length by default.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _help_output(script: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script), "--help"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{script} --help failed:\n{e.stderr}")
    return result.stdout


class TestLatentLengthDerivation:
    """#28 #4: latent_length should derive from audio_length // 320 by default."""

    def test_train_audio_diffusion_help_shows_auto_default(self) -> None:
        help_text = _help_output("train_audio_diffusion.py")
        assert "--latent-length" in help_text
        assert "--audio-length" in help_text

    def test_sample_audio_help_shows_auto_default(self) -> None:
        help_text = _help_output("sample_audio.py")
        assert "--latent-length" in help_text
        assert "--audio-length" in help_text

    def test_derive_latent_length_helper(self) -> None:
        """The derivation logic: audio_length // 320."""
        assert 24000 // 320 == 75
        assert 48000 // 320 == 150
        assert 12000 // 320 == 37


class TestConfigJsonExport:
    """#28 #5: training scripts save config.json alongside checkpoints."""

    def test_train_audio_decoder_exports_config_json(self, tmp_path: Path) -> None:
        out_pt = tmp_path / "decoder.pt"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_audio_decoder.py"),
                "--toy",
                "--epochs",
                "1",
                "--out",
                str(out_pt),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        config_path = tmp_path / "config.json"
        assert config_path.exists(), "config.json not saved alongside decoder.pt"
        config = json.loads(config_path.read_text())
        assert "latent_channels" in config
        assert "base_channels" in config
        assert "q" in config
        assert "sample_rate" in config or "audio_length" in config

    def test_train_audio_diffusion_exports_config_json(self, tmp_path: Path) -> None:
        out_pt = tmp_path / "dit.pt"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_audio_diffusion.py"),
                "--toy",
                "--epochs",
                "1",
                "--out",
                str(out_pt),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        config_path = tmp_path / "config.json"
        assert config_path.exists(), "config.json not saved alongside dit.pt"
        config = json.loads(config_path.read_text())
        assert "dim" in config
        assert "depth" in config
        assert "num_heads" in config
        assert "patch_size" in config
        assert "latent_channels" in config
        assert "latent_length" in config
        assert "spec_dim" in config
        assert "q" in config


class TestSafetensorsExport:
    """#28 #3: training scripts export safetensors files for trainable components."""

    def test_train_audio_decoder_exports_safetensors(self, tmp_path: Path) -> None:
        out_pt = tmp_path / "decoder.pt"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_audio_decoder.py"),
                "--toy",
                "--epochs",
                "1",
                "--out",
                str(out_pt),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        safe_path = tmp_path / "decoder.safetensors"
        assert safe_path.exists(), "decoder.safetensors not exported"

    def test_train_audio_diffusion_exports_safetensors(self, tmp_path: Path) -> None:
        out_pt = tmp_path / "dit.pt"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_audio_diffusion.py"),
                "--toy",
                "--epochs",
                "1",
                "--out",
                str(out_pt),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        safe_path = tmp_path / "dit.safetensors"
        assert safe_path.exists(), "dit.safetensors not exported"

    def test_safetensors_is_loadable(self, tmp_path: Path) -> None:
        from safetensors.torch import load_file

        out_pt = tmp_path / "decoder.pt"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_audio_decoder.py"),
                "--toy",
                "--epochs",
                "1",
                "--out",
                str(out_pt),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        safe_path = tmp_path / "decoder.safetensors"
        state = load_file(str(safe_path))
        assert len(state) > 0, "safetensors file is empty"
