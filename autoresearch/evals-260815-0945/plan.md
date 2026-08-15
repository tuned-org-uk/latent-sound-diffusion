# Autoresearch plan — Issue #50 Phase 1: evaluation extensions + MPS rerun

`[autoresearch] mode: orchestrator (build-feature)`

## Goal
Extend the evaluation pipeline per issue #50 Phase 1, rerun all deliverables on
MPS with the new grad-clip trainer (issue #51 fixes), and calibrate the
experiment framing to the paper's objective: **presenting a novel paradigm** —
a workflow introduction with scientific backing, not a benchmark claim.

## Calibration intent (user-confirmed direction)
- The paper introduces a novel workflow (dehydration → diffusion → rehydration
  → recursive variation) backed by honest preliminary-scale measurements.
- No "beats the baseline" claims: numbers are hypothesis-generating evidence
  at N=256/20-epoch scale, with explicit hypotheses for what should improve
  at scale (training length, N, capacity).
- Workflow-affordance metrics (compression, coherence, variant drift,
  diversity-vs-fidelity tradeoffs) stand alongside fidelity metrics; FAD/CLAP
  stay documented proxies.
- Every table carries methodology columns; summary reports numbers verbatim.

## Scope
- `src/ald_sc/eval.py` additions (+ unit tests in `tests/test_eval.py`):
  - `band_energy_retention` — per-mode band energies of original vs
    reconstructed audio via `prior.band_energies` (§6.2 per-band retention).
  - `spectral_rolloff` — companion to `spectral_centroid` (§6.1 rolloff drift).
  - `recursive_variant_drift` — TRUE R-round recursion: feed outputs back
    through `condition_on_audio` → `synthesize_midi`; centroid/rolloff drift
    + CLAP-proxy distance to round 0 (replaces the MIDI-rotation hack).
  - `eps_sweep` — heat-death ε sweep helper (sampling-only, SpectralSchedule).
- `scripts/run_evaluation.py`:
  - `--ablation-q {4,8,16,32}` (decoder retraining per prior),
    `--ablation-noise {0.0,0.1,0.25,0.5}` (diversity-vs-fidelity),
    `--ablation-eps {1e-2,1e-3,1e-4}` (sampling-only).
  - Checkpoints to `results/artifacts/` (gitignored) so sweeps share the
    encoder/prior embeddings.
  - Integrate retention + rolloff + true recursion into the main run.
  - `--tag` prefix for dataset-scoped outputs (ESC-50 vs NSynth).
- Runs (all `--device mps`):
  - Main NSynth 256-subset run with new metrics integrated.
  - q / NOISE_INJECT / ε sweeps.
  - ESC-50 download (~600 MB, CC BY 4.0) + 256-subset tagged run.
- Notebook 07 rebuilt with the new tables.
- `results/artifacts/` added to `.gitignore`.

## Metric (Success predicate)
```
uv run pytest tests/ -q
  && uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
  && test -s results/table1_reconstruction.csv && test -s results/table6_band_retention.csv
  && test -s results/sweep_q.csv && test -s results/sweep_noise.csv && test -s results/sweep_eps.csv
  && test -s results/fig_variant_diversity.png
  && test -s autoresearch/evals-260815-0945/summary.md
```
Expected: exit 0 (all sweeps + main run executed on `--device mps`).

## Terminal choice
stop-at-verified. No ship/deploy/push.

## Non-goals (this loop)
- Paper prose fill-ins (issue #50 Phases 2-3) — follow-up loop.
- ESC-50 numbers go into the paper only via the tagged CSVs; prose deferred.
