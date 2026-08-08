"""Tests for issue #38: --per-sample-conditioning flag in sample_audio.py.

When active, c_spec is resampled per sample for within-bank spectral
diversity. When inactive (default), a single shared c_spec steers all
samples toward the same spectral region.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_ddim

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


class TestPerSampleConditioningCLI:
    """CLI flags for per-sample conditioning (#38)."""

    def test_num_samples_arg_present(self) -> None:
        help_text = _help_output("sample_audio.py")
        assert "--num-samples" in help_text, (
            "--num-samples missing from sample_audio CLI"
        )

    def test_per_sample_conditioning_arg_present(self) -> None:
        help_text = _help_output("sample_audio.py")
        assert "--per-sample-conditioning" in help_text, (
            "--per-sample-conditioning missing from sample_audio CLI"
        )


class TestPerSampleConditioningDiversity:
    """Per-sample c_spec produces more diverse latents than shared c_spec."""

    def _make_model(self) -> LSDModel:
        from torch import nn

        class StubEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Conv1d(1, 128, 320, stride=320)

            def encode(self, x, prior):
                z = self.proj(x).float()
                a = z.mean(dim=2)
                return z, a, prior.chart_energy_descriptor(a)

            def extract_features(self, x):
                return self.proj(x).float()

        torch.manual_seed(3407)
        embeddings = torch.randn(32, 128)
        prior = build_arrow_prior(embeddings, q=8, k=4)
        encoder = StubEncoder()
        decoder = GraphDecoder(128, 1, 128, 16, prior, (2, 4, 5, 8))
        dit = MinimalDiT(
            latent_channels=128,
            latent_length=16,
            patch_size=4,
            dim=32,
            depth=1,
            num_heads=4,
            spec_dim=24,
        )
        sched = CosineSchedule(num_steps=100)
        return LSDModel(
            prior=prior,
            dit=dit,
            decoder=decoder,
            encoder=encoder,
            schedule=sched,
            sample_rate=24000,
        )

    def test_per_sample_c_spec_more_diverse_latents(self) -> None:
        """Per-sample c_spec produces higher pairwise variance than shared c_spec.

        With high guidance_scale on an untrained model, shared c_spec pulls
        all samples toward the same spectral region (reducing variance),
        while per-sample c_spec pulls each toward a different region
        (preserving variance).  20 steps let the conditioning compound.
        """
        from torch import nn as _nn

        m = self._make_model()
        device = torch.device("cpu")
        n = 8
        base_seed = 200
        guidance = 50.0

        # Strongly perturb AdaLN so c_spec has a large effect.
        for block in m.dit.blocks:
            _nn.init.normal_(block.adaln.proj.weight, std=0.1)
            _nn.init.normal_(block.adaln.proj.bias, std=0.1)

        # Shared c_spec: one probe for all samples
        z_probe = torch.randn(1, 128, 16, device=device)
        c_spec_shared = m.prior.chart_energy_descriptor(z_probe.mean(dim=2))

        shared_latents = [
            sample_ddim(
                m.dit,
                m.schedule,
                c_spec=c_spec_shared,
                steps=20,
                seed=base_seed + i,
                device=device,
                eta=0.0,
                guidance_scale=guidance,
            )
            for i in range(n)
        ]

        # Per-sample c_spec: fresh probe per sample
        per_sample_latents = []
        for i in range(n):
            g = torch.Generator(device=device).manual_seed(base_seed + i + 1000)
            z_probe_i = torch.randn(1, 128, 16, device=device, generator=g)
            c_spec_i = m.prior.chart_energy_descriptor(z_probe_i.mean(dim=2))
            per_sample_latents.append(
                sample_ddim(
                    m.dit,
                    m.schedule,
                    c_spec=c_spec_i,
                    steps=20,
                    seed=base_seed + i,
                    device=device,
                    eta=0.0,
                    guidance_scale=guidance,
                )
            )

        def pairwise_var(latents: list[torch.Tensor]) -> float:
            flat = torch.stack([t.flatten() for t in latents])
            return float(flat.var(dim=0).mean())

        assert pairwise_var(per_sample_latents) > pairwise_var(shared_latents)

    def test_shared_c_spec_condenses_relative_to_unconditioned(self) -> None:
        """Shared c_spec with high guidance reduces variance vs unconditional."""
        from torch import nn as _nn

        m = self._make_model()
        device = torch.device("cpu")
        n = 8
        base_seed = 300
        guidance = 50.0

        for block in m.dit.blocks:
            _nn.init.normal_(block.adaln.proj.weight, std=0.1)
            _nn.init.normal_(block.adaln.proj.bias, std=0.1)

        # Unconditional (guidance_scale=0 → pure unconditional)
        uncond_latents = [
            sample_ddim(
                m.dit,
                m.schedule,
                c_spec=None,
                steps=20,
                seed=base_seed + i,
                device=device,
                eta=0.0,
                guidance_scale=0.0,
            )
            for i in range(n)
        ]

        # Shared c_spec with high guidance
        z_probe = torch.randn(1, 128, 16, device=device)
        c_spec_shared = m.prior.chart_energy_descriptor(z_probe.mean(dim=2))
        shared_latents = [
            sample_ddim(
                m.dit,
                m.schedule,
                c_spec=c_spec_shared,
                steps=20,
                seed=base_seed + i,
                device=device,
                eta=0.0,
                guidance_scale=guidance,
            )
            for i in range(n)
        ]

        def pairwise_var(latents: list[torch.Tensor]) -> float:
            flat = torch.stack([t.flatten() for t in latents])
            return float(flat.var(dim=0).mean())

        # Shared conditioning with high guidance should reduce variance
        # vs unconditional (all samples pulled toward same target).
        assert pairwise_var(shared_latents) < pairwise_var(uncond_latents)

    def test_per_sample_c_spec_is_deterministic(self) -> None:
        """Per-sample c_spec with the same base seed is reproducible."""
        m = self._make_model()
        device = torch.device("cpu")
        base_seed = 42

        def generate() -> list[torch.Tensor]:
            latents = []
            for i in range(4):
                g = torch.Generator(device=device).manual_seed(base_seed + i + 1000)
                z_probe_i = torch.randn(1, 128, 16, device=device, generator=g)
                c_spec_i = m.prior.chart_energy_descriptor(z_probe_i.mean(dim=2))
                latents.append(
                    sample_ddim(
                        m.dit,
                        m.schedule,
                        c_spec=c_spec_i,
                        steps=5,
                        seed=base_seed + i,
                        device=device,
                        eta=0.0,
                    )
                )
            return latents

        a = generate()
        b = generate()
        for la, lb in zip(a, b):
            assert torch.allclose(la, lb, atol=1e-6)
