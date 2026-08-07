"""Tests for CLI script argument plumbing (issue #24 acceptance).

Verifies that the training scripts expose the hyperparameters required
for generation-grade vs fast-iteration configuration.  Invokes the
scripts with ``--help`` via subprocess (the same entry point a user uses)
and checks the flags are present.  This validates CLI configurability
without running full training — the underlying library functions are
already covered by ``test_trainer.py`` and ``test_losses.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _help_output(script: str) -> str:
    """Run a script with --help and return its stdout."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout


class TestTrainAudioDiffusionCLI:
    def test_dit_capacity_args_present(self) -> None:
        help_text = _help_output("train_audio_diffusion.py")
        for flag in ["--dim", "--depth", "--num-heads", "--patch-size"]:
            assert flag in help_text, f"{flag} missing from train_audio_diffusion CLI"

    def test_cfg_dropout_arg_present(self) -> None:
        help_text = _help_output("train_audio_diffusion.py")
        assert "--cfg-dropout" in help_text

    def test_lr_arg_present(self) -> None:
        help_text = _help_output("train_audio_diffusion.py")
        assert "--lr" in help_text


class TestTrainAudioDecoderCLI:
    def test_decoder_capacity_args_present(self) -> None:
        help_text = _help_output("train_audio_decoder.py")
        assert "--base-channels" in help_text

    def test_loss_weight_args_present(self) -> None:
        help_text = _help_output("train_audio_decoder.py")
        for flag in [
            "--lambda-rec",
            "--lambda-stft",
            "--lambda-chart",
            "--lambda-smooth",
        ]:
            assert flag in help_text, f"{flag} missing from train_audio_decoder CLI"

    def test_lr_arg_present(self) -> None:
        help_text = _help_output("train_audio_decoder.py")
        assert "--lr" in help_text
