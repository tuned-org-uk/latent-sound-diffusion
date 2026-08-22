"""Comparative similarity test across all bank-generation variants (issue #53).

Generates banks through the shipped library interface
(``LSDModel.generate_sound_bank``) in every ``bank_mode`` and variety
setting, and measures two families of metrics per bank:

Within-bank (diversity — how different are the n clips from each other):
  - pairwise waveform L1 (mean/min/max)
  - pairwise EnCodec-feature cosine similarity (mean/min; 1.0 = identical)

To-canonical (anchoring — how far the variant bank sits from the
canonical "house voice", which shares the same seed / z-bar):
  - mean feature cosine similarity of each variant clip to the canonical
    clip (higher = more anchored to the frozen manifold)
  - centroid spread (Hz) and RMS as distribution guards

Writes ``results/bank_mode_comparison.csv`` and prints the table with a
USEFUL verdict per the #53 gate (FAD excluded here; see
``scripts/bank_variants.py`` for the full pre-registered protocol).

Usage:
    uv run python scripts/compare_bank_modes.py             # MPS auto
    uv run python scripts/compare_bank_modes.py --device cpu

Requires results/artifacts/ checkpoints from scripts/run_evaluation.py.
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import torch

from ald_sc.audio_codec import EnCodecEncoder
from ald_sc.build_prior import build_arrow_prior
from ald_sc.dit import MinimalDiT
from ald_sc.eval import encodec_pooled_features, spectral_centroid, write_csv
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.inference import LSDModel
from ald_sc.schedule import CosineSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "results" / "artifacts"
SEED, N_BANK, STEPS = 3407, 8, 50
RMS_FLOOR, L1_FLOOR, SPREAD_FLOOR = 0.35, 0.05, 20.0

# (label, bank_mode, bank_variety) — every shipped variant.
ARMS: list[tuple[str, str, float | None]] = [
    ("canonical", "canonical", None),
    ("jitter_a0.05", "jitter", 0.05),
    ("jitter_a0.1", "jitter", 0.1),
    ("jitter_a0.25", "jitter", 0.25),
    ("jitter_a0.5", "jitter", 0.5),
    ("resid_r0.1", "residual", 0.1),
    ("resid_r0.3", "residual", 0.3),
    ("stopvar", "stopvar", None),
]


def _load_model(device: torch.device) -> tuple[LSDModel, EnCodecEncoder]:
    embeddings = torch.load(ARTIFACTS / "embeddings.pt", weights_only=True)
    prior = build_arrow_prior(embeddings, q=8, k=4).to(device)
    decoder = (
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
    decoder.load_state_dict(
        torch.load(ARTIFACTS / "graph_dec.pt", weights_only=True, map_location="cpu")
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
        torch.load(ARTIFACTS / "dit.pt", weights_only=True, map_location="cpu")
    )
    encoder = EnCodecEncoder().to(device).eval()
    model = LSDModel(
        prior=prior,
        dit=dit,
        decoder=decoder,
        encoder=encoder,
        schedule=CosineSchedule(num_steps=1000),
    )
    return model, encoder


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

    model, encoder = _load_model(device)
    print(f"device={device}, arms={len(ARMS)}, n={N_BANK}, steps={STEPS}")

    def generate(bank_mode: str, bank_variety: float | None) -> list[torch.Tensor]:
        kwargs: dict[str, object] = dict(n=N_BANK, steps=STEPS, seed=SEED)
        if bank_variety is not None:
            kwargs["bank_variety"] = bank_variety
        return [
            c.cpu() for c in model.generate_sound_bank(bank_mode=bank_mode, **kwargs)
        ]

    canonical = generate("canonical", None)
    canon_feat = encodec_pooled_features(canonical, encoder, device)
    canon_unit = canon_feat / canon_feat.norm(dim=1, keepdim=True).clamp(min=1e-12)

    rows: list[dict[str, object]] = []
    for label, bank_mode, variety in ARMS:
        clips = canonical if bank_mode == "canonical" else generate(bank_mode, variety)

        l1 = [
            float((clips[a] - clips[b]).abs().mean())
            for a in range(len(clips))
            for b in range(a + 1, len(clips))
        ]
        cents = [float(spectral_centroid(c).mean()) for c in clips]
        rms = float(
            torch.cat([c.pow(2).mean().unsqueeze(0) for c in clips]).sqrt().mean()
        )

        feats = encodec_pooled_features(clips, encoder, device)
        unit = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-12)
        iu = torch.triu_indices(len(clips), len(clips), offset=1)
        within = (unit @ unit.T)[iu[0], iu[1]]
        to_canon = unit @ canon_unit[0].T if bank_mode != "canonical" else None

        spread = statistics.stdev(cents)
        gate = (
            statistics.mean(l1) >= L1_FLOOR
            and rms >= RMS_FLOOR
            and spread > SPREAD_FLOOR
        )
        rows.append(
            {
                "arm": label,
                "bank_mode": bank_mode,
                "variety": variety if variety is not None else "",
                "l1_mean": round(statistics.mean(l1), 4),
                "l1_min": round(min(l1), 4),
                "l1_max": round(max(l1), 4),
                "feat_cos_within_mean": round(float(within.mean()), 4),
                "feat_cos_within_min": round(float(within.min()), 4),
                "feat_cos_to_canonical": (
                    round(float(to_canon.mean()), 4) if to_canon is not None else 1.0
                ),
                "centroid_spread_hz": round(spread, 1),
                "centroid_mean_hz": round(statistics.mean(cents), 1),
                "RMS": round(rms, 3),
                "gate": "USEFUL" if gate else "-",
            }
        )
        r = rows[-1]
        print(
            f"[{label:>14}] L1 {r['l1_mean']:.4f} ({r['l1_min']:.4f}-"
            f"{r['l1_max']:.4f})  cos_w {r['feat_cos_within_mean']:.4f}"
            f"  cos_zbar {r['feat_cos_to_canonical']:.4f}"
            f"  spread {r['centroid_spread_hz']:6.1f}Hz  RMS {r['RMS']:.3f}"
            f"  {r['gate']}"
        )

    out = REPO_ROOT / "results" / "bank_mode_comparison.csv"
    write_csv(out, rows)
    useful = [r["arm"] for r in rows if r["gate"] == "USEFUL"]
    print(f"\nUSEFUL arms: {useful or 'none'}")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
