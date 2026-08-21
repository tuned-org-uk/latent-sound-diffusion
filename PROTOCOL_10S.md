# PROTOCOL_10S — "Length extended, quality unchanged" evaluation protocol

Status: pre-registered before any 10 s training run (issue #60).
Baselines: `baselines/v0.11-prefix/` (frozen @ `c7b4f7e`, SHA256-manifested).

## Claims under test

1. **Primary:** native 10 s generation (Track A, 750-frame DiT) is
   *equivalent* (not superior) to the frozen 5 s-class baseline profile on
   quality metrics.
2. **Non-inferiority control:** Track B (zero-retrain waveform stitching of
   existing checkpoints) ships if it is non-inferior to the same baseline.

## Arms

| Arm | Model | Length | Notes |
|---|---|---|---|
| REF   | frozen `esc50_dit.pt` (+ decoders)          | 5 s class | metrics recomputed, not reused |
| A     | v0.12 DiT finetuned/retrained at 750 frames | 10 s      | variable-length crops 300–750 |
| B     | frozen checkpoints + equal-power crossfade stitch | 10 s | no retraining |

All arms run on the **v0.12 sampler stack** (fixed direction term, terminal
sigma). The frozen artifacts define target metric values, not code.

## Seeding rules

- Eval seeds: block **≥ 900 000**, disjoint from all tuning/split/training
  seeds (the historical 3407 family) and from each other per arm:
  REF `[900000, 900999]`, A `[901000, 901999]`, B `[902000, 902999]`.
- No `seed+i`, `seed+100+i` style collisions across arms or stages; one
  seeded `torch.Generator` threaded end-to-end.
- Confirmation block uses a further-disjoint range (`[950000, …]`) never
  touched during piloting.

## References and inputs

- FAD-style references: **random-crop** 10 s windows from the held-out
  corpus split. Zero-padding short files is forbidden (it mechanically
  shifts FAD with length).
- All arms decode at identical sample rate and are loudness-matched
  (LUFS or RMS-over-body) before encoding for feature distances.
- Device pinned to MPS for headline tables; `device` and torch version
  recorded as CSV columns on every row.

## Metrics (all reported with 95 % cluster-bootstrap CIs, resampling seed families)

Quality: STFT distance · latent L2 · FAD-proxy vs length-matched references ·
CLAP cosine similarity/diversity curves (issue #45 protocol).
Diversity (per-second-normalized): pairwise feature L1 · spectral centroid
spread · RMS spread. Diagnostics: init-vs-endpoint diameter ratio
(contraction drift), logged per length.

## Design & statistics

1. **Pilot:** M = 32 independent seed families/arm → estimate metric SDs;
   derive equivalence margins from these SDs only (never reuse the #53
   constants).
2. **Confirmation:** M ≥ 64 clips/arm for FAD-class metrics; M ≥ 30 families
   for pairwise-diversity endpoints.
3. **Test:** TOST (equivalence via CI-within-margin) on the A−REF difference
   per metric; B tested for non-inferiority against REF with the same
   machinery. Holm correction across the metric family.
4. Pilot and confirmation reported separately (no winner's-curse merging).

## Release gates

- Arm A ships iff: TOST-equivalent to REF on all quality metrics after
  Holm correction **and** no diversity regression vs its own length-normalized
  profile.
- Arm B ships iff non-inferior to REF; it may ship alongside A as fallback.
- Absolute thresholds inherited from `bank_variants.py` are retired; only
  relative-to-baseline decisions count.

## Provenance requirements per result artifact

`resolved_seed`, `git_commit`, `torch.__version__`, `device`,
`bank_mode/bank_variety/steps` (where applicable) in metadata/manifest,
plus SHA256 digests. Outputs generated with `seed=None` must not enter
paper tables.
