# Autoresearch plan — Issue #50 Phase 2/3: paper writing

`[autoresearch] mode: orchestrator (build-feature)`

## Goal
Fill all 23 `\placeholder{}` macros in `docs/spectral-composition/document.tex`
with data-grounded results (Phase 1 reruns on MPS, autoresearch
evals-260815-0945) and calibrated prose, keeping the novel-paradigm framing:
honest preliminary-scale numbers, no benchmark claims.

## Scope
- Group A (7 data-grounded): §5.3 hyperparameter table, §3.1 compression
  table, §6.1 recursion results + Figure + listening-study protocol, §6.2
  Table 1 + FAD-proxy + retention + ESC-50, §6.3 ablations (CFG deferred,
  one sentence), §6.4 concrete limitations (corrected issue #51 narrative),
  §9 findings summary.
- Group B (16 prose): §2.1 related models, §2.2 ArrowSpace/EDN, §2.3 problem
  of time (port Connes–Rovelli/Barontini framing from the old ald-sc.tex via
  git history), §3 training loss (from losses.py), §3.2 super-dense bank,
  §3.3.1 GPT-MIDI design spec, §3.5 DAW sync spec, §4.1 kNN construction
  (from build_prior.py), §4.5 IPC TikZ diagram, §7.1-7.3 discussion/ethics,
  §8.1-8.4 future work.
- Bibliography: +AudioLM, MusicLM, MusicGen/AudioCraft, Stable Audio,
  Riffusion, AudioLDM 2, Jukebox, Moûsai, ConnesRovelli1994, DDIM,
  DPM-Solver, CLAP, FAD.
- Copy `fig_variant_diversity.png` into `docs/spectral-composition/figures/`.
- README FAD-proxy note; handoff.json chain update.

## Metric (Success predicate)
```
! grep -c "placeholder{"
```
→ 0 macros remain (listening study becomes prose), `pdflatex` compiles clean
(exit 0), `uv run pytest tests/ -q` green, ruff clean (no src changes
expected).

## Terminal choice
stop-at-verified. No push.
