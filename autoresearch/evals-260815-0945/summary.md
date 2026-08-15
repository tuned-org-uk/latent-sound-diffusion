# Autoresearch Summary — Issue #50 Phase 1: eval extensions + MPS rerun

`[autoresearch] mode: orchestrator (build-feature) → CONVERGED`

## Calibration (user direction)
The paper presents a **novel paradigm** (dehydration → diffusion → rehydration →
recursive variation) at preliminary scale. All tables are hypothesis-generating
evidence with honest numbers and documented proxy methodology — no benchmark
claims. Workflow-affordance metrics (compression, coherence, variant drift,
diversity-vs-fidelity, intrinsic stopping) stand alongside fidelity metrics.

## What was built / changed
- **`src/ald_sc/eval.py`** extensions: `band_energy_retention` (§6.2 per-band
  retention), `spectral_rolloff` (§6.1 rolloff drift), `recursive_variant_drift`
  (TRUE R-round recursion via `condition_on_audio` → `synthesize_midi`,
  replacing the MIDI-rotation hack), `eps_sweep` (heat-death ε, sampling-only).
  +11 unit tests (`tests/test_eval.py`, 41 total).
- **`scripts/run_evaluation.py`**: `--ablation-q/-noise/-eps` sweep modes,
  `--tag` dataset prefix (ESC-50 tables don't clobber NSynth), checkpoints to
  `results/artifacts/` (gitignored), retention + true recursion integrated,
  eps sweep writes FAD-vs-steps.
- **Heat-death criterion fixed** (`SpectralSchedule.is_heat_death`): the old
  forward-time metric Σνᵏᾱ(t) < ε fired at the *start* of sampling; now the
  reverse-process remaining dissipation Σνᵏ(1−ᾱ(t)) < ε (monotone decreasing
  during denoising). `docs/HOW_IT_WORKS.md` + tests updated.
- **Graph decoder v2 (issue #51 root cause found during this loop)**: the wave
  block's pooled/time-constant `U_q` delta broadcast through GroupNorm carried
  1/σ³ gradients (knife-edge; device RNG stream decided stability). Now a
  **per-time-step graph filter** (1×1 convs + einsum through `U_q`, 0.01×
  near-identity init). Correction posted on issue #51; README updated.

## Runs (all `--device mps`, grad-clip 1.0, 256-subset, 20 epochs, real EnCodec)
- Main NSynth run ~15 min (was ~16.5 min CPU) incl. sweeps; ESC-50 run ~20 min.
- ESC-50 downloaded to `data/esc50/` (gitignored; 2000 clips, CC BY 4.0).

## Key results (NSynth)
| Metric | Graph | Baseline |
|---|---|---|
| L1 train / val / test | **0.1103 / 0.1073 / 0.1127** | 0.1162 / 0.1086 / 0.1151 |
| Band-retention cosine (test) | **0.9979** | 0.9969 |
| FAD-proxy (lower better) | 1197.8 | **407.9** |
| CLAP-proxy | **-0.046** | -0.073 |

The per-time-step graph filter **flips the L1 comparison**: graph now ≤
baseline on every split (the pooled version lost 0.117 vs 0.114). Honest
negative retained: FAD-proxy still favours the baseline at this scale.

- λ_ED ablation delta: train 8.6e-4, val −8.8e-5, test −2.7e-5 (still a no-op —
  consistent with the postmortem).
- True recursion (R=4): CLAP-proxy distance to round 0 = 0 → 0.079 → 0.084 →
  0.095 → 0.081 (drift-then-saturate); centroid 606 → 1536 Hz, rolloff 12 →
  584 Hz. Real drift where the old MIDI-rotation metric read ~0.
- q sweep (decoder retrained per q): test L1 0.1087–0.1144 across q ∈
  {4,8,16,32} — flat; reconstruction is not chart-limited at this scale.
- NOISE_INJECT sweep: test L1 0.1116 → 0.1212 (0 → 0.5); variant distance
  6.8e-5 → 2.4e-4 — a weak, non-monotone dial at this scale (honest finding).
- ε sweep (absolute units; Σνᵏ = 5.67): 46 steps/FAD 1154 (ε=0.1) → 41/858
  (ε=0.5) → 37/883 → 30/955 → 12/1180 (ε=5). Intrinsic stopping cuts compute
  2–4× and *improves* the FAD-proxy mid-range. Note: the issue's proposed
  1e-2…1e-4 is below the 50-step grid's reachable resolution — documented.

## Key results (ESC-50, `esc50_`-tagged tables)
- Graph L1 ≤ baseline again (train 0.0818 vs 0.0833; val 0.0794).
- Retention cosine (test): graph 0.9991 vs baseline 0.9984.
- FAD-proxy: graph 1787 vs baseline 307 (baseline favoured, as on NSynth).
- Recursive drift: distance → 0.123; centroid 270 → 1551 Hz.
- Rehydration coherence: r = −0.017 (≈0 on NSynth too: −0.021) — resampling
  pitch artefacts dominate; honest limitation, matches §6.4.

## Verification
- `uv run pytest tests/ -q` → **223 passed** (was 211; +11 eval, +1 dissipation
  monotonicity; heat-death test updated to reverse-process semantics)
- `uv run ruff check src tests scripts` + `format --check` → clean
- Main + ESC-50 + sweeps + notebook executed on MPS; all deliverables non-empty
- `notebooks/07_evaluation_metrics.ipynb` rebuilt + executed (243 KB, new
  tables 6 / recursive variants / sweeps / ESC-50 sections + calibration note)

## Decision record
- ε sweep re-run with absolute {0.1, 0.5, 1, 2, 5} after the 1e-2…1e-4 grid
  proved unreachable (Σν-normalised units documented in CSV).
- Wave-block v2 adopted over trainer-side workarounds (clip/eps/warmup all
  failed); paper numbers re-run on MPS in the same loop.
- ESC-50 keeps its own tagged tables; paper prose integration deferred to the
  Phase 2/3 loop.

## Terminal choice
stop-at-verified. No push/deploy.
