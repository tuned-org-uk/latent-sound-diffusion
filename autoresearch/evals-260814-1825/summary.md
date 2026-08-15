# Autoresearch Summary — Evaluation pipeline for issue #49

`[autoresearch] mode: orchestrator (build-feature) → CONVERGED`

## What was built
- **`src/ald_sc/eval.py`** (reusable, ~700 lines): all paper-table computations
  - Phase 1: `reconstruction_table`, `ablation_table` (train/val/test L1 + λ_ED ablation)
  - Phase 2: `frechet_distance`, `fad_score`, `text_embedding`, `audio_embedding`, `clap_proxy_score`, `perceptual_table` (FAD-proxy + CLAP-proxy, graph vs baseline)
  - Phase 3a: `compression_ratio`, `compression_ratio_vs_n` (dehydration compression vs N)
  - Phase 3b: `spectral_centroid`, `midi_pitch_contour`, `rehydration_coherence` (MIDI pitch vs spectral centroid, Pearson r)
  - Phase 3c: `variant_diversity` (pairwise CLAP-proxy cosine distance vs recursion depth)
  - FAD/CLAP implemented as dependency-free proxies on frozen EnCodec features (user-confirmed; README + paper §6.4 document fadtk removal)
- **`tests/test_eval.py`**: 30 unit tests (CPU, synthetic data, hermetic)
- **`scripts/run_evaluation.py`**: full runner (--smoke for fast self-check, --device cpu/mps)
- **`scripts/build_eval_notebook.py`**: notebook builder (loads pre-computed results by default)
- **`notebooks/07_evaluation_metrics.ipynb`**: executed notebook with all output cells + embedded figure
- **Infra fixes**: EnCodec MPS lazy-load device bug, CosineSchedule/LinearSchedule device handling, inference _resample_1d + synthesize_midi MPS compatibility

## Decision record
- **FAD/CLAP**: dependency-free proxy (user-confirmed). fadtk/laion-clap NOT re-added (README: removed due to dependency conflicts; paper §6.4 documents the proxy). Methodology recorded in CSV `*_method` columns.
- **Notebook name**: `07_evaluation_metrics.ipynb` (matches issue; nb07_production_workflow.ipynb already exists — non-clobbering via builder script, not a rename conflict).
- **Training device**: CPU for paper-quality numbers. MPS accelerates the baseline decoder + DiT (~5×) but the graph decoder's `WaveReconstructionBlock` diverges on MPS (numerical instability in the U_q projection / gate matmuls: loss 0.76 → 11.1 vs CPU 0.19 → 0.15). Documented in the notebook configuration cell.
- **Variant depths**: [1, 2, 4] (matches issue text "1×, 2×, 4×"). Depth 8 caused MPS OOM; removed.
- **Paper repo link**: added to `docs/spectral-composition/document.tex` (title footnote, §5.1 Software Stack, bibliography).

## Verification
- `uv run pytest tests/ -q` → 203 passed (203, +30 new eval tests)
- `uv run ruff check src/ald_sc/ scripts/ tests/test_eval.py` → All checks passed
- `uv run ruff format --check src/ald_sc/ scripts/ tests/test_eval.py` → 27 files already formatted
- `uv run python scripts/run_evaluation.py --device cpu` → exit 0, 992s, all 6+1 deliverables OK
- `uv run jupyter nbconvert --execute notebooks/07_evaluation_metrics.ipynb` → exit 0, 1s, 10/10 code cells with outputs, 2 embedded images

## Deliverables (all present, non-empty)
| File | Size | Phase |
|---|---|---|
| results/table1_reconstruction.csv | 483 B | 1 |
| results/table2_ablation.csv | 314 B | 1 |
| results/table3_fad_clap.csv | 263 B | 2 |
| results/table4_compression.csv | 587 B | 3a |
| results/table5_coherence.csv | 160 B | 3b |
| results/fig_variant_diversity.png | 51373 B | 3c |
| results/variant_diversity.csv | 163 B | 3c |
| notebooks/07_evaluation_metrics.ipynb | 131536 B | notebook |

## Key results (256-sample NSynth subset, 20 epochs, CPU, real EnCodec)
- Graph decoder L1: train 0.121 / val 0.119 / test 0.117
- Baseline decoder L1: train 0.112 / val 0.107 / test 0.114
- λ_ED ablation delta: ~1e-5 (gating has near-zero effect — consistent with
  the documented c_spec regression postmortem)
- Compression ratio: 2.34× (N=1) → 15.99× (N=9811); prior amortises
- Rehydration coherence: Pearson r = –0.040 (ascending MIDI scale vs spectral centroid)
- Variant diversity: mean pairwise CLAP-proxy distance 0.0 (depth 1-2) → 6.1e-4 (depth 4)

## Terminal choice
stop-at-verified. No ship/deploy/push (per autoresearch safety invariants).