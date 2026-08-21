# Baseline Freeze — v0.11 pre-fix reference

**Frozen:** 2026-08-22 · **Source commit:** `c7b4f7e` (main, pre-sampler/layout fixes)
**Branch introducing breaking fixes:** `v0.12-tentative` @ `ddc7d6a`
**Environment:** torch 2.13.0 · Python 3.13.12 · macOS/arm64 (MPS)

## Why frozen

`ddc7d6a` changes output distributions: unpatchify layout, DDIM direction
term, sigma-ladder endpoint, and pos-embed interpolation all alter generated
audio. Every table and checkpoint in this directory was produced **before**
those fixes and is the only valid comparison target for the v0.12 equivalence
claim ("length extended, quality unchanged"). Do not regenerate in place.

## Contents

- `esc50_*` — paper-grade checkpoints (ESC-50 corpus, latent_length=300,
  dim=64/depth=2 geometry per `run_evaluation.py`) + prior/embeddings.
  These back `esc50_table1..6`.
- `dit.pt`, `graph_dec.pt`, `baseline_dec.pt`, `prior.pt`, `embeddings.pt` —
  the earlier NSynth-side run backing un-prefixed tables.
- `graph_dec_q{4,8,16,32}.pt`, `graph_dec_noise{0.0,0.25,0.5}.pt` — ablation
  arms for tables 2/4.
- All `results/*.csv` as of the freeze commit.
- `smoketest_*` included for completeness; not used by any paper table.

## Integrity

`SHA256SUMS` covers every `.pt` and `.csv`. Verify:

    shasum -a 256 -c SHA256SUMS

Checkpoint files are gitignored (18 MB); the durable references are this
manifest plus tracked CSVs at commit `c7b4f7e`.

## Usage in the v0.12 protocol

See `PROTOCOL_10S.md`: Track A (native 10 s) must pass TOST equivalence
against these artifacts' metric profile; Track B (zero-retrain stitching)
must be non-inferior to `esc50_table3_fad_clap.csv` baselines on fresh,
disjoint eval seeds.
