# Predict Summary — LSD 10-second stem extension
**Run:** predict-260821-2359 · **Depth:** deep (8 personas, 3 rounds) · **Budget:** 40/40 findings used
**Scope:** `src/ald_sc/{dit,sampling,inference}.py`, `scripts/{sample_audio,train_audio_diffusion}.py`, `configs/audio_diffusion.yaml` (+ deps & tests consulted)
**Goal:** architecture + performance, weighted to: extend generation 2–4 s → **10 s stems**, no quality regression, MPS hardware, dense attention stays default.
**Personas:** Software Architect · Security Analyst · Diffusion Theory Researcher · Signal Processing Scientist · Psychoacoustics Scientist · Statistical Learning Scientist · Reproducibility Scientist · ML Numerics Specialist

---

## Consensus verdict (the track question)

**The 10 s goal is blocked by model-side bugs, not by attention cost or data.** Three scientific personas independently agree dense attention is a non-issue at ~94 tokens (10 s @ 75 Hz ÷ patch 8); quadratic-cost work (#46 sparse attention) is **demoted** for this goal. The binding chain is:

1. **Unpatchify layout bug** (`dit.py:236-237`) — output time-slot is `t' = k·N + n` instead of `t = n·ps + k`. A checkpoint trained at N=47 tokens learns scrambled semantics and cannot render at N=94 even with interpolated positional embeddings. *This single bug makes "just interpolate pos_embed" insufficient.*
2. **Sampler math error** (`sampling.py:257`) — production `sample_ddim` feeds raw velocity where the DDIM direction ε̂ is required; every clip ever shipped used it. Empirically tolerated (late-step v-weighting is tiny) but biased; must be A/B-tested before it becomes the fixed baseline.
3. **Sigma ladder stops at t=19** (`schedule.py:72-77`) — ~4 % residual noise floor baked into every latent, at any length.
4. **No valid measurement** — shared seed 3407 across split/train/eval, zero-padded length-mismatched FAD references, device-bound RNG streams: "quality unchanged at 10 s" is currently unverifiable rather than unsupported.

## Top 10 consensus findings (severity × confidence × agreement)

| # | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | Unpatchify temporal scramble locks sequence length; length extension structurally impossible from current checkpoints | CRIT | 95% | dit.py:236-237 |
| 2 | `sample_ddim` uses raw v where DDIM needs ε̂ = √ᾱ·v + √(1−ᾱ)·x; production sampler for all audio | HIGH→CRIT | 85% | sampling.py:257 vs 157-160 |
| 3 | Sigma ladder terminates at t=19 (√(1−ᾱ₁₉)≈0.04): fixed residual-noise floor in every generated latent | HIGH | 93% | schedule.py:72-77 |
| 4 | pos_embed hard gate: crash outside band [(N−1)·ps+1, N·ps], silent pass inside; no interpolation/guard | HIGH | 94% | dit.py:147-208 |
| 5 | No defensible eval protocol: seed 3407 shared across stages, padded references shift FAD mechanically at 10 s, CPU/MPS generators non-portable | HIGH | 92% | run_evaluation.py:556-713, data.py:77-82, sampling.py:229 |
| 6 | Unseeded fallback prior `torch.randn(64,128)` in both scripts — silent per-run random conditioning geometry | HIGH | 98% | sample_audio.py:51, train_audio_diffusion.py:60 |
| 7 | Dead YAML config; four divergent `latent_length` sources (75 / 300 / 375 / artifact-truth 300); metadata never validated on load | MED-HIGH | 95% | configs/audio_diffusion.yaml:7, sample_audio.py:34, bank_variants.py:103 |
| 8 | `synthesize_midi` pitch shift cancels via chained resamples (net pitch = f(dur) only) | HIGH (Mode C) | 92% | inference.py:512-521 |
| 9 | Notebook `overlap_add` fades placed one segment early (periodic dropouts/double-energy seams); latent-space crossfade blends off-manifold EnCodec features | MED-HIGH | 87% | notebooks/02·04 cells 6·9 (ref nb07 cell 21) |
| 10 | Heat-death stopping criterion unreachable with defaults (Σν≈3.05 ≫ ε=1e-3), unit-dependent, docstring states wrong formula | MED | 90% | spectral_schedule.py:62-173 |

Full deduped register (15 items incl. CFG/spec dead weight, Bank.store path traversal, pickled prior forcing `weights_only=False`, post-hoc temperature distorting c_spec, serial batch-size-1 generation) in `debate.md`.

## Risk assessment

- **Blocking risk:** attempting 750-frame training before fixing #1 wastes the run entirely (scrambled supervision). Fix order matters more than speed.
- **Invalidation risk:** fixing #2/#3 changes every output distribution mid-project → freeze current checkpoints+tables as baselines BEFORE fixes land; otherwise no comparison survives.
- **Claim risk:** without #5's protocol (length-matched references, disjoint eval seeds, pinned device, TOST equivalence bands), a "quality unchanged" statement in the paper is attackable.
- **Deferred-risk accepted:** Mode C render bugs (#8, #9) are real but out of scope this cycle (user: materialised quality currently acceptable); recorded for the next cycle.

## Recommendation

**Two-track plan, primary = native 10 s:**
1. **Track A (primary):** fix #1 → #3 → #4 (small diffs + impulse round-trip test) → freeze baselines → finetune/retrain DiT at 750 frames with variable-length crops (300–750) → run #5's equivalence protocol vs Track B control arm.
2. **Track B (control/fallback, zero-retrain):** corrected waveform-domain equal-power crossfade stitching of existing checkpoints (nb07 pattern), as a non-inferiority control so a shippable 10 s exists while Track A matures.
3. **Demoted:** #46 SparseAttn1d — build the `attn_mode` seam opportunistically; expect no quality/speed win at 94 tokens; revisit ≥30 s horizons.
4. **Deferred:** synthesize_midi/notebook render fixes (#8, #9), bank-mode recalibration, security hardening bundle.
