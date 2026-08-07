"""Tests for c_spec conditioning of the DiT (issue #22 acceptance criteria).

Covers:
- the DiT's predicted velocity responds to ``c_spec`` (conditioning path is
  active on an untrained model once the zero-initialized AdaLN is perturbed)
- ``generate_sound_bank`` forwards ``target_c_spec`` through the inference
  plumbing into the DiT (``_sample_and_decode`` → ``sample_ddim`` → DiT)
- ``_sample_and_decode`` raises ``ValueError`` when ``z_init`` +
  ``c_spec_override`` conflict
- ``train_audio_diffusion`` raises ``TypeError`` for a DiT without
  ``cfg_dropout``

The divergence is asserted at the **v_pred level** (the DiT's forward
output) rather than the decoded audio: on a freshly-initialized, untrained
DiT the AdaLN scale/shift projection is zero by design, so ``c_spec`` has
negligible influence on the DDIM trajectory versus the seeded initial noise,
and the graph decoder re-derives ``c_spec`` post-hoc from the generated
``z`` — washing out residual differences at the audio level.  These tests
can only prove audio-level divergence after training; the v_pred-level
assertion proves the conditioning path is wired and responsive on an
untrained model.  The plumbing (that ``target_c_spec`` reaches the DiT) is
covered separately by a spy-based test.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule
from ald_sc.trainer import train_audio_diffusion

from _helpers import (
    LATENT_CH,
    LATENT_LEN,
    SPEC_DIM,
    make_model,
    perturb_adaln,
)


def _make_model() -> LSDModel:
    return make_model()


# ---------------------------------------------------------------------------
# c_spec conditioning divergence tests (v_pred level)
# ---------------------------------------------------------------------------


class TestCSpecConditioningDivergence:
    """Acceptance criterion: the DiT's predicted velocity depends on c_spec.

    Asserted at the v_pred level because audio-level divergence requires a
    trained model (the decoder re-derives c_spec post-hoc from the generated
    z, and an untrained DiT's zero-initialized AdaLN ignores c_spec).  These
    tests prove the conditioning path is active and responsive on an
    untrained model once AdaLN is perturbed; audio-level divergence is
    expected to emerge after training.
    """

    def test_conditioned_v_pred_differs_from_unconditioned(self) -> None:
        """dit(z, t, c_spec=X) diverges from dit(z, t, c_spec=None)."""
        m = _make_model()
        perturb_adaln(m.dit)
        m.dit.eval()

        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([50])
        c_spec = torch.randn(1, SPEC_DIM)

        uncond = m.dit(z, t)
        cond = m.dit(z, t, c_spec=c_spec)

        assert not torch.allclose(uncond, cond, atol=1e-6), (
            "Conditioned and unconditioned v_preds are identical — c_spec "
            "is not influencing the DiT's prediction."
        )

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_two_different_c_specs_produce_different_v_preds(self, seed: int) -> None:
        """Two distinct c_spec vectors must yield different v_preds (same z, t)."""
        m = _make_model()
        perturb_adaln(m.dit)
        m.dit.eval()

        torch.manual_seed(seed + 1)
        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        t = torch.tensor([seed % 100])
        c_spec_a = torch.randn(1, SPEC_DIM)
        c_spec_b = torch.randn(1, SPEC_DIM)

        out_a = m.dit(z, t, c_spec=c_spec_a)
        out_b = m.dit(z, t, c_spec=c_spec_b)

        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            f"Two different c_spec vectors produced identical v_preds "
            f"(seed={seed}). The DiT spec_proj is not influencing the output."
        )


# ---------------------------------------------------------------------------
# Inference plumbing: target_c_spec reaches the DiT
# ---------------------------------------------------------------------------


class TestCSpecPlumbing:
    """Prove the inference plumbing forwards target_c_spec into the DiT.

    ``generate_sound_bank(target_c_spec=...)`` → ``_sample_and_decode`` →
    ``sample_ddim`` → ``dit(z_t, t, c_spec=...)``.  A spy on the DiT's
    forward captures the ``c_spec`` actually received during sampling and
    verifies it matches the requested ``target_c_spec`` (and that ``None``
    is forwarded in the unconditioned case).  This is plumbing coverage;
    the divergence class above covers that the DiT responds to what it
    receives.
    """

    def _install_spy(self, dit: MinimalDiT) -> list:
        received: list[Tensor | None] = []
        original_forward = dit.forward

        def recording_forward(
            z_t: Tensor, t: Tensor, c_spec: Tensor | None = None, **kw
        ):  # type: ignore[no-untyped-def]
            received.append(None if c_spec is None else c_spec.detach().clone())
            return original_forward(z_t, t, c_spec=c_spec, **kw)

        dit.forward = recording_forward  # type: ignore[method-assign,assignment]
        return received

    def test_unconditioned_bank_forwards_none_c_spec(self) -> None:
        m = _make_model()
        received = self._install_spy(m.dit)

        m.generate_sound_bank(n=1, steps=4, seed=42)

        assert received, "DiT.forward was never called during sampling"
        assert all(cs is None for cs in received), (
            "Unconditioned generation forwarded a non-None c_spec to the DiT"
        )

    def test_conditioned_bank_forwards_target_c_spec(self) -> None:
        m = _make_model()
        received = self._install_spy(m.dit)
        c_spec = torch.randn(1, SPEC_DIM)

        m.generate_sound_bank(n=1, steps=4, seed=42, target_c_spec=c_spec)

        assert received, "DiT.forward was never called during sampling"
        forwarded = [cs for cs in received if cs is not None]
        assert forwarded, "target_c_spec was not forwarded to the DiT"
        # The same c_spec is broadcast across every sampling step / bank sample.
        for cs in forwarded:
            assert torch.equal(cs, c_spec), (
                "DiT received a c_spec that does not match target_c_spec"
            )


# ---------------------------------------------------------------------------
# _sample_and_decode conflict guard
# ---------------------------------------------------------------------------


class TestSampleAndDecodeConflict:
    def test_raises_when_z_init_and_c_spec_both_provided(self) -> None:
        """Providing both z_init and c_spec_override must raise ValueError."""
        m = _make_model()
        z = torch.randn(1, LATENT_CH, LATENT_LEN)
        c_spec = torch.randn(1, SPEC_DIM)

        with pytest.raises(ValueError, match="mutually exclusive"):
            m._sample_and_decode(
                seed=0,
                steps=2,
                temperature=1.0,
                z_init=z,
                c_spec_override=c_spec,
            )


# ---------------------------------------------------------------------------
# cfg_dropout guard in train_audio_diffusion
# ---------------------------------------------------------------------------


class TestCfgDropoutGuard:
    def test_raises_type_error_for_non_conformant_dit(self) -> None:
        """train_audio_diffusion must raise TypeError if DiT lacks cfg_dropout."""
        torch.manual_seed(0)
        embeddings = torch.randn(32, LATENT_CH)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        sched = CosineSchedule(num_steps=100)

        class NoCfgDiT(nn.Module):
            latent_shape = (LATENT_CH, LATENT_LEN)

            def forward(self, z_t, t, c_spec=None):
                return torch.zeros_like(z_t)

        class StubVAE(nn.Module):
            class encoder(nn.Module):
                @staticmethod
                def encode(x, prior):
                    z = torch.zeros(x.shape[0], LATENT_CH, LATENT_LEN)
                    a = z.mean(dim=2)
                    return z, a, prior.chart_energy_descriptor(a)

            def parameters(self):
                return iter([])

        dummy_data = torch.randn(2, 1, LATENT_LEN * 320)
        loader = DataLoader(TensorDataset(dummy_data), batch_size=2)
        vae = StubVAE()

        with pytest.raises(TypeError, match="cfg_dropout"):
            # Exhaust the generator to trigger the error.
            list(
                train_audio_diffusion(
                    loader=loader,
                    audio_vae=vae,
                    dit=NoCfgDiT(),
                    prior=prior,
                    schedule=sched,
                    epochs=1,
                )
            )
