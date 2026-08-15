"""Measure end-to-end single-clip generation latency (issue #50, §6.3/§9).

Times the full generation path for ONE 4-second clip (matching the
evaluation configuration: latent 128×300, graph decoder 32 base
channels): DDIM sampling -> self-consistent c_spec -> graph-decoder
render -> peak normalisation. Two configs:

  - full50: 50 DDIM steps, no early stop
  - heatdeath_eps0.5: heat-death early stopping at eps = 0.5 (the
    sweep's quality/compute sweet spot: 41 steps, FAD-proxy 858 vs
    46 steps / 1154)

on both MPS and CPU (median of 3 runs each). Reports seconds per clip
and the real-time factor (audio duration / latency; >1 means faster
than playback). Writes results/latency.csv.

Uses the trained NSynth checkpoints saved by the evaluation run in
results/artifacts/ (prior.pt, graph_dec.pt, dit.pt).

Usage:
    uv run python scripts/measure_latency.py            # both devices
    uv run python scripts/measure_latency.py --device mps
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.eval import write_csv
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.sampling import sample_ddim
from ald_sc.schedule import CosineSchedule
from ald_sc.spectral_schedule import SpectralSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "results" / "artifacts"
AUDIO_SECONDS = 4.0
RUNS = 3


def _load(device: torch.device):
    embeddings = torch.load(ARTIFACTS / "embeddings.pt", weights_only=False)
    prior = build_arrow_prior(embeddings, q=8, k=4).to(device)
    graph_dec = GraphDecoder(
        latent_channels=128,
        out_channels=1,
        feature_dim=128,
        base_channels=32,
        prior=prior,
        upsample_strides=(2, 4, 5, 8),
    ).to(device)
    graph_dec.load_state_dict(
        torch.load(ARTIFACTS / "graph_dec.pt", weights_only=False, map_location="cpu")
    )
    graph_dec = graph_dec.to(device)
    dit = MinimalDiT(
        latent_channels=128,
        latent_length=300,
        patch_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        spec_dim=3 * 8,
    )
    dit.load_state_dict(
        torch.load(ARTIFACTS / "dit.pt", weights_only=False, map_location="cpu")
    )
    dit = dit.to(device).eval()
    sched = CosineSchedule(num_steps=1000)
    return prior, dit, graph_dec, sched


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_one_clip(prior, dit, decoder, sched, device, steps, spec_sched, seed):
    """End-to-end single-clip generation (mirrors LSDModel._sample_and_decode)."""
    z = sample_ddim(
        dit,
        sched,
        batch_size=1,
        steps=steps,
        seed=seed,
        device=device,
        spectral_schedule=spec_sched,
    )
    a = z.mean(dim=2)
    c_spec = prior.chart_energy_descriptor(a)
    audio = decoder(z, c_spec).squeeze(1)
    audio = audio.clamp(-1, 1)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak
    return audio


def measure(device_name: str) -> list[dict[str, object]]:
    device = torch.device(device_name)
    prior, dit, decoder, sched = _load(device)
    spec_sched = SpectralSchedule(prior=prior, horizon=1.0, eps=0.5)

    configs = [
        ("full50", None),
        ("heatdeath_eps0.5", spec_sched),
    ]
    rows: list[dict[str, object]] = []
    for label, ss in configs:
        latencies, steps_used = [], []
        for r in range(RUNS):
            _sync(device)
            t0 = time.perf_counter()
            generate_one_clip(prior, dit, decoder, sched, device, 50, ss, 3407 + r)
            _sync(device)
            latencies.append(time.perf_counter() - t0)
            z, used = sample_ddim(
                dit,
                sched,
                batch_size=1,
                steps=50,
                seed=3407 + r,
                device=device,
                spectral_schedule=ss,
                return_steps=True,
            )
            steps_used.append(used)
        med = statistics.median(latencies)
        rows.append(
            {
                "device": device_name,
                "config": label,
                "audio_seconds": AUDIO_SECONDS,
                "runs": RUNS,
                "median_latency_s": round(med, 3),
                "min_latency_s": round(min(latencies), 3),
                "max_latency_s": round(max(latencies), 3),
                "mean_steps_used": round(statistics.mean(steps_used), 1),
                "realtime_factor": round(AUDIO_SECONDS / med, 2),
            }
        )
        print(
            f"[{device_name:>4} | {label:<17}] median {med:.2f}s/clip "
            f"({AUDIO_SECONDS / med:.2f}x realtime), "
            f"{statistics.mean(steps_used):.0f} steps"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=str, default=None, help="mps | cpu")
    args = parser.parse_args()

    devices = [args.device] if args.device else ["mps", "cpu"]
    rows: list[dict[str, object]] = []
    for d in devices:
        try:
            rows.extend(measure(d))
        except RuntimeError as e:
            print(f"[{d}] unavailable: {e}")

    if rows:
        out = REPO_ROOT / "results" / "latency.csv"
        write_csv(out, rows)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
