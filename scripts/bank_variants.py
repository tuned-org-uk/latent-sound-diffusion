"""Bank-generation variant strategies (post-contraction finding).

Context (issue #53, session 2026-08-15): the undertrained unconditional
DiT contracts every noise draw to (nearly) one latent (init-noise
max|diff| 6.1 -> 1.2e-6 after 49 DDIM steps), so ``generate_sound_bank``
returns n near-identical clips and sampler-side dials cannot manufacture
diversity. This experiment tests latent-space variation around the
canonical draw z_bar — where Mode-B interpolation already shows
waveform-level diversity exists (pairwise L1 0.13-0.17).

Arms (n=8 bank each, peak-normalised, graph-decoded):
  - controls: base_ddim (current behaviour), temperature 0.5/1.5
  - jitter:      z_bar + alpha * sigma_z * eps_i        (alpha 0.05-0.5)
  - residual:    z_bar + k*(z_ddim,i - z_bar)           (rel std 0.1/0.3)
  - corpus-anchor: (1-alpha)*z_bar + alpha*z_corpus_i   (alpha 0.25-0.75)
  - stop-time:   same seed, different step counts       (12..49)

Metrics per arm: pairwise waveform L1 (diversity), centroid spread (Hz),
FAD-proxy vs held-out test features (distributional guard), RMS
(degeneracy guard).

Pre-registered gate (decided before first run — do not retune):
  USEFUL iff mean pairwise L1 >= 0.05 AND FAD <= 1200 AND RMS >= 0.35
  AND centroid spread > 20 Hz.

Usage:
    uv run python scripts/bank_variants.py             # MPS (or CPU auto)
    uv run python scripts/bank_variants.py --device cpu

Requires results/artifacts/ checkpoints from scripts/run_evaluation.py.
Writes results/bank_variants.csv.
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import torch

from ald_sc.audio_codec import EnCodecEncoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.data import AudioFolderDataset, build_audio_dataloader
from ald_sc.dit import MinimalDiT
from ald_sc.eval import (
    encodec_pooled_features,
    fad_score,
    spectral_centroid,
    split_files,
    write_csv,
)
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.sampling import sample_ddim
from ald_sc.schedule import CosineSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "results" / "artifacts"
SEED, N_BANK, STEPS = 3407, 8, 50
FAD_CEILING, RMS_FLOOR, L1_FLOOR, SPREAD_FLOOR = 1200.0, 0.35, 0.05, 20.0


@torch.no_grad()
def _decode(dec, prior, z):
    c = prior.chart_energy_descriptor(z.mean(dim=2))
    xh = dec(z, c).clamp(-1, 1).squeeze(0)
    peak = xh.abs().max()
    return (xh / peak if peak > 0 else xh).unsqueeze(0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=str, default=None, help="cpu | mps")
    args = parser.parse_args()
    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    random.seed(SEED)
    torch.manual_seed(SEED)

    embeddings = torch.load(ARTIFACTS / "embeddings.pt", weights_only=False)
    prior = build_arrow_prior(embeddings, q=8, k=4).to(device)
    dec = (
        GraphDecoder(
            latent_channels=128,
            out_channels=1,
            feature_dim=128,
            base_channels=32,
            prior=prior,
            upsample_strides=(2, 4, 5, 8),
        )
        .to(device)
        .eval()
    )
    dec.load_state_dict(
        torch.load(ARTIFACTS / "graph_dec.pt", weights_only=False, map_location="cpu")
    )
    dit = (
        MinimalDiT(
            latent_channels=128,
            latent_length=300,
            patch_size=8,
            dim=64,
            depth=2,
            num_heads=4,
            spec_dim=24,
        )
        .to(device)
        .eval()
    )
    dit.load_state_dict(
        torch.load(ARTIFACTS / "dit.pt", weights_only=False, map_location="cpu")
    )
    sched = CosineSchedule(num_steps=1000)
    encoder = EnCodecEncoder().to(device).eval()

    files = sorted((REPO_ROOT / "data").glob("*.wav"))
    selected = random.Random(SEED).sample(files, 256)
    _, _, test_files = split_files(selected, 0.7, 0.15, seed=SEED)
    test_ds = AudioFolderDataset(root=str(REPO_ROOT / "data"), audio_length=96000)
    test_ds.files = test_files
    test_loader = build_audio_dataloader(test_ds, batch_size=8, shuffle=False)
    ref_feats = encodec_pooled_features((b for b in test_loader), encoder, device)

    anchor_ds = AudioFolderDataset(root=str(REPO_ROOT / "data"), audio_length=96000)
    anchor_ds.files = test_files[:N_BANK]
    z_anchor = []
    for batch in build_audio_dataloader(anchor_ds, batch_size=1, shuffle=False):
        z, _, _ = encoder.encode(batch.to(device), prior)
        z_anchor.append(z)
    print(f"device={device}, anchors={len(z_anchor)}, ref={tuple(ref_feats.shape)}")

    draws = [
        sample_ddim(dit, sched, batch_size=1, steps=STEPS, seed=SEED + i, device=device)
        for i in range(N_BANK)
    ]
    z_bar = draws[0]
    resid = torch.cat([(d - z_bar).flatten().unsqueeze(0) for d in draws[1:]])
    sig = z_bar.std()
    print(f"z_bar std {sig:.3f}; seed-residual std {resid.std():.2e}")

    def bank_metrics(clips):
        l1 = [
            float((clips[a] - clips[b]).abs().mean())
            for a in range(len(clips))
            for b in range(a + 1, len(clips))
        ]
        cents = [float(spectral_centroid(c).mean()) for c in clips]
        feats = encodec_pooled_features(clips, encoder, device)
        rms = float(
            torch.cat([c.pow(2).mean().unsqueeze(0) for c in clips]).sqrt().mean()
        )
        return {
            "l1_mean": statistics.mean(l1),
            "l1_min": min(l1),
            "l1_max": max(l1),
            "centroid_spread_hz": statistics.stdev(cents),
            "centroid_mean_hz": statistics.mean(cents),
            "FAD": round(fad_score(feats, ref_feats.cpu()), 1),
            "RMS": round(rms, 3),
        }

    rows: list[dict[str, object]] = []

    def report(name, clips):
        m = bank_metrics(clips)
        gate = (
            m["l1_mean"] >= L1_FLOOR
            and m["FAD"] <= FAD_CEILING
            and m["RMS"] >= RMS_FLOOR
            and m["centroid_spread_hz"] > SPREAD_FLOOR
        )
        m.update(arm=name, gate="USEFUL" if gate else "-")
        rows.append(m)
        print(
            f"[{name:>14}] L1 {m['l1_mean']:.4f} ({m['l1_min']:.4f}-"
            f"{m['l1_max']:.4f})  spread {m['centroid_spread_hz']:6.1f}Hz"
            f"  FAD {m['FAD']:7.1f}  RMS {m['RMS']:.3f}"
            f"  {'USEFUL' if gate else '-'}"
        )

    # Controls
    report("base_ddim", [_decode(dec, prior, d).cpu() for d in draws])
    for temp in (0.5, 1.5):
        report(f"temp{temp}", [_decode(dec, prior, d * temp).cpu() for d in draws])

    # Jitter bank
    for alpha in (0.05, 0.1, 0.25, 0.5):
        clips = []
        for i in range(N_BANK):
            g = torch.Generator(device=device).manual_seed(SEED + 100 + i)
            eps = torch.randn_like(z_bar, generator=g)
            clips.append(_decode(dec, prior, z_bar + alpha * sig * eps).cpu())
        report(f"jitter_a{alpha}", clips)

    # Residual amplification
    for rel in (0.1, 0.3):
        k = rel * sig / max(resid.std(), 1e-12)
        report(
            f"resid_r{rel}",
            [_decode(dec, prior, z_bar + k * (d - z_bar)).cpu() for d in draws],
        )

    # Corpus-anchored blend
    for alpha in (0.25, 0.5, 0.75):
        report(
            f"anchor_a{alpha}",
            [
                _decode(dec, prior, (1 - alpha) * z_bar + alpha * z_anchor[i]).cpu()
                for i in range(N_BANK)
            ],
        )

    # Stop-time variation
    step_grid = [12, 17, 22, 27, 32, 37, 42, 49]
    report(
        "stopvar",
        [
            _decode(
                dec,
                prior,
                sample_ddim(
                    dit, sched, batch_size=1, steps=s, seed=SEED, device=device
                ),
            ).cpu()
            for s in step_grid
        ],
    )

    out = REPO_ROOT / "results" / "bank_variants.csv"
    write_csv(out, rows)
    useful = [r["arm"] for r in rows if r["gate"] == "USEFUL"]
    print(f"\nUSEFUL arms: {useful or 'none'}")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
