# Autoresearch plan — Evaluation pipeline for issue #49

`[autoresearch] mode: orchestrator (build-feature)`

## Goal
Implement and run the Spectral Composition evaluation pipeline and produce all
issue #49 deliverables.

## Scope
- Phase 1: train/val/test L1 reconstruction + λ_ED ablation → CSV tables 1 & 2.
- Phase 2: FAD-proxy (Fréchet distance on frozen EnCodec features) + CLAP-proxy
  (cosine on deterministic text-audio embedding) + baseline comparison → table 3.
  Strategy: **dependency-free proxy** (user-confirmed). fadtk / laion-clap NOT
  re-added (README: removed due to dependency conflicts). Methodology documented.
- Phase 3a: dehydration compression ratio vs corpus size N → table 4.
- Phase 3b: rehydration coherence (MIDI pitch contour vs spectral centroid, Pearson) → table 5.
- Phase 3c: recursive variant diversity (pairwise CLAP-proxy cosine distance vs depth) → figure.
- Notebook: `notebooks/08_evaluation_metrics.ipynb` (renamed from issue's
  `07_*.ipynb` to avoid clobbering existing `07_production_workflow.ipynb`).
- Eval engine: `src/ald_sc/eval.py` (reusable). Runner: `scripts/run_evaluation.py`.
- Tests: `tests/test_eval.py`.

## Metric (Success predicate)
All deliverable files exist, are non-empty, and validate:
  results/table1_reconstruction.csv
  results/table2_ablation.csv
  results/table3_fad_clap.csv
  results/table4_compression.csv
  results/table5_coherence.csv
  results/fig_variant_diversity.png
  notebooks/08_evaluation_metrics.ipynb

## Verify command
  uv run python scripts/run_evaluation.py --smoke   # fast end-to-end self-check
  uv run pytest tests/test_eval.py -q
  uv run ruff check src/ald_sc/eval.py scripts/run_evaluation.py tests/test_eval.py
  test -s results/table1_reconstruction.csv && test -s results/table2_ablation.csv \
    && test -s results/table3_fad_clap.csv && test -s results/table4_compression.csv \
    && test -s results/table5_coherence.csv && test -s results/fig_variant_diversity.png \
    && test -s notebooks/08_evaluation_metrics.ipynb

## Terminal choice
stop-at-verified (produce all deliverables + tests green + lint clean). No ship/deploy.

## Follow-up (non-blocking)
- Add link to this repository (tuned-org-uk/latent-sound-diffusion) in the
  Spectral Composition paper (docs/spectral-composition/document.tex).