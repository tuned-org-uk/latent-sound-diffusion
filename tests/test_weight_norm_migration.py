"""Tests for deprecated ``weight_norm`` hook -> parametrization migration.

PyTorch deprecated ``torch.nn.utils.weight_norm`` in favour of
``torch.nn.utils.parametrizations.weight_norm``. The EnCodec pretrained
model is constructed (by the third-party ``encodec`` package) with the
old hook, which emits a ``FutureWarning`` on every load and will become
a hard error when PyTorch removes the old API.

``EnCodecEncoder._load_model`` suppresses the construction-time warning
and migrates every affected module to the new parametrization so the
effective weights are preserved and the model survives the eventual
removal of the old hook API.

These are small, deterministic unit tests that exercise the migration
helper directly without needing the real EnCodec model (which requires
network/weights).
"""

from __future__ import annotations

import warnings

import torch
from ald_sc.audio_codec import _migrate_weight_norm
from torch import nn


def _make_old_wn_module() -> nn.Conv1d:
    """Conv1d with the *deprecated* weight_norm hook applied."""
    m = nn.Conv1d(8, 16, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        torch.nn.utils.weight_norm(m)  # deprecated hook
    return m


class TestMigrateWeightNorm:
    def test_converts_old_hook_to_parametrization(self) -> None:
        from torch.nn.utils.weight_norm import WeightNorm

        m = _make_old_wn_module()
        old_hooks = [h for h in m._forward_pre_hooks.values() if isinstance(h, WeightNorm)]
        assert old_hooks, "sanity: old weight_norm hook present before migration"

        migrated = _migrate_weight_norm(m)

        assert migrated == 1
        # The deprecated WeightNorm forward pre-hook must be gone.
        assert not any(isinstance(h, WeightNorm) for h in m._forward_pre_hooks.values())
        # The new parametrization must be registered on "weight".
        assert "weight" in m.parametrizations

    def test_preserves_effective_weights(self) -> None:
        m = _make_old_wn_module()
        x = torch.randn(2, 8, 16)
        y_before = m(x).detach().clone()

        _migrate_weight_norm(m)

        y_after = m(x)
        assert torch.allclose(y_before, y_after, atol=1e-6)

    def test_noop_on_clean_module(self) -> None:
        m = nn.Conv1d(8, 16, 3)  # no weight_norm applied

        migrated = _migrate_weight_norm(m)

        assert migrated == 0
        assert not hasattr(m, "parametrizations")

    def test_migrates_every_module_in_full_model(self) -> None:
        model = nn.Sequential(nn.Conv1d(8, 16, 3), nn.Conv1d(16, 8, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            torch.nn.utils.weight_norm(model[0])
            torch.nn.utils.weight_norm(model[1])

        migrated = _migrate_weight_norm(model)

        assert migrated == 2
        assert "weight" in model[0].parametrizations
        assert "weight" in model[1].parametrizations

    def test_is_idempotent(self) -> None:
        m = _make_old_wn_module()

        _migrate_weight_norm(m)
        migrated = _migrate_weight_norm(m)  # already migrated -> noop

        assert migrated == 0
