# Autoresearch Summary — Issue #50 Phase 2/3: paper writing

`[autoresearch] mode: orchestrator (build-feature) → CONVERGED`

## What was done
All 23 `\placeholder{}` macros in `docs/spectral-composition/document.tex`
filled (macro definition removed); paper recompiled clean.

### Group A — data-grounded (7)
- **§3.1** compression paragraph + `tab:compression` (2.34× N=1 → 15.99×
  N=9811; prior = 561,664 bits stored once).
- **§5.3** `tab:hyperparams` (full config incl. loss weights, grad-clip,
  sampler settings) + measured MPS times (graph 70s / baseline 59s / DiT 45s
  per 20 epochs; CPU ≈5×).
- **§6.1** true-recursion table + `fig_variant_diversity.png` as Figure
  (drift-then-saturate: 0→0.079→0.084→0.095→0.081; centroid 606→1536 Hz);
  listening study written as a proposed, preregisterable protocol.
- **§6.2** `tab:reconstruction` (L1 + STFT diagnostics, NSynth + ESC-50,
  all splits) + `tab:retention` + FAD/CLAP proxy methodology stated
  honestly (baseline favoured 408 vs 1198; scale hypotheses explicit).
- **§6.3** four ablations verbatim (λ_ED no-op; R depth; q flat; noise dial
  weak) + ε sweep with the units lesson (Σν=5.67; 41 steps/FAD 858 beats
  46/1154); **CFG dropped** → one sentence deferring to Phase 2.
- **§6.4** seven concrete limitations (proxy metrics, preliminary scale,
  idle gating, training-instability history + design rule, resampling
  pitch artefacts (r≈0), graph scalability, no long-form).
- **§9** findings summary in the calibration voice: workflow properties
  each with a first quantitative trace; scale-up framed as the decisive
  experiment, not a defence.

### Group B — prose (16)
Related models (§2.1, three-strategy taxonomy + position), ArrowSpace/EDN
(§2.2), Connes–Rovelli/Barontini problem-of-time framing ported from the
old ald-sc.tex + DDIM/DPM-Solver contrast (§2.3), training loss from
losses.py incl. the honest disclosure that λ_stft=0 and chart/smooth carry
no decoder gradient in the current trainer (§3), super-dense bank (§3.2),
GPT-MIDI as design spec with the engineering-gap admission (§3.3.1), DAW
sync spec (§3.5), kNN construction from build_prior.py + pyarrowspace
fallback truth (§4.1), **§4.3 rewritten to the v2 per-time-step graph
filter** with the instability postmortem paragraph, TikZ IPC diagram
(§4.5), paradigm discussion (§7.1), systems table (§7.2), ethics with
clearance/auditability stance (§7.3), future work ×4 (§8).

### Bibliography
+14 entries: Jukebox, AudioLM, MusicLM, MusicGen/AudioCraft, Riffusion,
AudioLDM 2, Stable Audio, Moûsai, ConnesRovelli1994, DDIM, DPM-Solver,
FAD (Kilgour), CLAP (Wu et al.).

## Verification
- `grep -c "placeholder{"` → **0** (definition also removed)
- `pdflatex` ×3 → exit 0, **23 pages**, 0 undefined references
- `uv run pytest tests/ -q` → 223 passed (no code changes)
- `uv run ruff check/format` → clean
- README FAD note updated to name the proxy methodology

## Terminal choice
stop-at-verified. No push/deploy.
