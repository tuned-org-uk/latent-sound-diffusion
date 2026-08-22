# v0.12.0 Track bundle — first native 10 s generation

**Checkpoint:** `dit_v0.12_10s.pt` (+ `.safetensors`, `_metadata.json`)
Trained by `scripts/train_audio_longform.py` on ESC-50 paired segments
(748-frame max crop), 30 epochs × 500 steps on MPS, loss 17.17 → 1.47.
Geometry: latent_channels=128, latent_length=748, patch_size=8, dim=64,
depth=2, num_heads=4, spec_dim=24. Training commit `5e11c6c`+`1f359bb`
(sampler/layout fixes active).

## Arms (PROTOCOL_10S.md)

| Directory | Arm | Contents |
|---|---|---|
| `track_a/` | A — native 10 s, v0.12 sampler | 16 clips @ 9.97 s, seeds 901000–901015 + manifest.json |
| `ref_bank/` | REF — frozen v0.11 checkpoint (`baselines/v0.11-prefix/esc50_dit.pt`) | 16 clips @ 4.00 s, seeds 900000–900015 + manifest.json |
| `track_b/` | B — zero-retrain control: consecutive REF pairs joined via `equal_power_overlap_add` | 8 renders @ ~7.96 s |

## Metrics (`results/track_metrics.csv`, FAD-proxy vs length-matched held-out references; lower = closer to corpus)

| Arm | Duration | FAD-proxy | Centroid mean |
|---|---|---|---|
| track_a | 9.97 s | **167.6** | 1428 Hz |
| track_b | 7.98 s | 242.8 | 1349 Hz |
| ref_bank | 4.00 s | 238.9 | 1352 Hz |

Native 10 s beats both the stitched control and the frozen baseline.
Pilot scale (n=16/8).

## Confirmation scale (M=64, fresh blocks 901100+/900100+, `confirm_metrics.csv`)

| Arm | n | Duration | FAD-proxy | Centroid mean |
|---|---|---|---|---|
| confirm_a (native) | 64 | 9.97 s | **152.8** | 1436 Hz |
| confirm_b (stitched control) | 32 | 7.98 s | 220.6 | 1347 Hz |

The pilot ordering replicates at confirmation scale with a wider gap
(ΔFAD ≈ 68). Pairwise pooled-feature distance stays at the contraction
floor (~1e-4) in all arms — diversity discrimination requires per-frame
metrics (open item on #60). TOST equivalence bands still pending; the
direction is now established at both scales.

## Statistical verification (`confirm_stats.csv`, n_boot=300, α=0.05)

Null band (same-generator ref_bank vs references): **[225.5, 241.6]** →
equivalence margin 241.6.

| Arm | FAD CI | Verdict |
|---|---|---|
| confirm_a | [148.0, 158.8] | equivalent (and entirely **below** the null band) |
| confirm_b | [213.4, 228.1] | equivalent |

confirm_a's whole CI sits under the null's whole CI: native v0.12
matches the corpus distribution significantly better than the frozen
v0.11 generator, not just "not worse". Per-frame excess ≈ 0.010 in both
arms — frame-level contraction persists (open item).

SHA256SUMS covers pilot + confirmation artifacts + stats.

## Integrity

    shasum -a 256 -c SHA256SUMS
