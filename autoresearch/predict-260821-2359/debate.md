# Debate Record — predict-260821-2359

8 isolated personas · 40 raw findings · 3 rounds · budget 40/40.
Live cross-examination resumes were cancelled by operator after Round-1 challenges were drafted; conflicts were adjudicated by the synthesizer via direct code/algebra verification (noted per resolution).

## Persona roster

| # | Persona | Lens | Findings | Session |
|---|---|---|---|---|
| P1 | Software Architect | boundaries, scalability | 5 | ses_fda857d00ffeOzK3a87BNhogy9 |
| P2 | Security Analyst | trust boundaries, deserialization | 5 | ses_fda855ee9ffeUEj6hGR7QfWweu |
| P3 | Diffusion Theory Researcher | sampler math, schedules | 5 | ses_fda718461ffeUyUMQRHVbQ660p |
| P4 | Signal Processing Scientist | codecs, resampling, O&A | 5 | ses_fda7133c9ffeS1T5ong7Ew5d50 |
| P5 | Psychoacoustics Scientist | perception, artifacts | 5 | ses_fda7dc394ffeER8WBSyODMJ7hJ |
| P6 | Statistical Learning Scientist | contraction, diversity, power | 5 | ses_fda702eb5ffedFkfyYIYsvw27n |
| P7 | Reproducibility Scientist | provenance, protocol | 5 | ses_fda64e0b4ffeMl8Otv1Zn7Wvq4 |
| P8 | ML Numerics Specialist | shape/layout algebra, RNG | 5 | ses_fda5b82bfffe411Q8E7cEzynxn |

---

## Round 1 — independent positions (digest)

Convergent clusters (independent discovery = high confidence):
- **Length is hard-bound in the denoiser**: P1-F1, P4-F1, P8-F2 (pos_embed gate), P3-F5.
- **Sampler/schedule quality caps**: P3-F1 (`sample_ddim` direction error), P3-F2 + P8-F4 (sigma ladder t=19 floor — double discovery).
- **Measurement cannot support a length claim**: P6-F5 (equivalence protocol), P7-F1..F4 (seed reuse, fallback prior, config drift, device RNG).
- **Materialisation path broken but deferred**: P5-F1..F4, P4-F2/F3.

Divergences carried into Round 2:
- C1: P4/P1 prescribe "interpolate pos_embed → go"; P8 claims a deeper layout bug makes that insufficient.
- C2: P3 rates `sample_ddim` CRITICAL; empirical track record says outputs are acceptable.
- C3: P6 wants latent-concat as control arm; P4 calls stitching unnecessary at 10 s.
- C4: Issue #46 sparse attention vs "attention cost non-issue at 94 tokens" (P3/P4/P8).

## Round 2 — challenges issued

**C1 → P4/P1:** P8's executed trace of `dit.py:236-237`: reshape merges `(ps,N)` so output slot = `k·N + n`, not `n·ps + k`. Checkpoint trained at N=47 learns "token n owns times ≡ n mod 47" semantics; at N=94 the same weights place correct values at wrong times even with interpolated pos_embed.
*Synthesizer verification:* confirmed independently — line 236 splits channel-major `[c,k]` correctly, but line 237 flattens `(ps,N)` row-major. Correct inverse requires `permute(0,1,3,2)` before the final reshape. Impulse round-trip test would catch it; existing tests assert shapes only.

**C2 → P3:** all shipped audio used `sample_ddim`; owner judges quality acceptable post-materialisation. Also: the final step multiplies v̂ by √(1−ᾱ₁₉)≈0.04, shrinking late-step impact of the wrong term.

**C3 → P6 vs P4:** native retrain risk vs zero-retrain shipping path.

**C4 → all scientific personas:** is #46 worth building for this goal at all?

## Round 3 — resolutions & confidence revisions

- **R1 (RESOLVED for P8):** unpatchify scramble is real and *primary*. P4-F1 recommendation amended: fix layout first, then interpolate pos_embed at load time, then finetune on variable-length crops (300–750). P1-F1 severity unchanged (same root complex). New consensus #1.
- **R2 (REVISED):** `sample_ddim` error stays HIGH (not downgraded to medium): bias is trajectory-wide where v-weighting is largest (high-noise steps dominate direction early); empirically tolerated because perceptual content concentrates in low-noise steps where √(1−ᾱ_prev)·v term is small and ẑ₀-term dominates. Fixing it invalidates all prior comparisons → sequence AFTER baseline freeze; cheap A/B `euler` vs `ddim` first (both already exist). Conf 95→85 pending A/B evidence.
- **R3 (BOTH KEPT):** Track A native 10 s primary (owner goal); Track B = corrected waveform-domain equal-power crossfade (nb07 cell-21 pattern) as non-inferiority control. Latent-space crossfade rejected (off-manifold; decoder RF ≈ ±20 ms smears, doesn't heal).
- **R4 (DEMOTE):** SparseAttn1d/#46 deprioritized for 10 s: dense attention over 94 tokens is microseconds-scale; keep an `attn_mode` seam opportunistically; revisit ≥30 s horizons or thousands-of-token regimes.

### Anti-herd check (mandated counter-arguments)

All personas broadly agreed on "fix model internals + protocol". Counter-arguments the synthesizer MUST register:
1. **Against model-first:** if LSD-studio-style materialisation composes short clips anyway (owner reports materialised quality fine), zero-retrain Track B may satisfy the actual product need this cycle without any training risk.
2. **Against immediate sampler fix:** swapping `sample_ddim`'s update changes every output distribution mid-project, voiding bank_variants gates and paper tables; freeze baselines first or accept documented discontinuity.
3. **Against architecture novelty:** with the layout fixed, plain finetune/retrain at N=94 may beat RoPE swaps or interpolation cleverness; complexity budget should follow the equivalence protocol, not novelty.

---

## Per-persona findings (deduped register)

**P1 Software Architect**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | DiT hard-binds seq length; no attn seam for SparseAttn1d | high | 92 | dit.py:70,147-148,208 |
| 2 | Producer inference O(n) serial batch-size-1 DDIM trajectories | high | 88 | inference.py:236-246,267-277,299-309 |
| 3 | Dead YAML; three divergent geometry sources; spec_dim/cfg_dropout trained-on-zeros dead weight | med | 95 | audio_diffusion.yaml:7,24 |
| 4 | Samplers relocate caller's model (.to(device)); triplicated update math ×3 heat-death blocks | med | 90 | sampling.py:133,230,279 |
| 5 | GraphDecoder embeds whole prior module; pickled prior.pt; no LSDModel.load | med | 85 | graph_decoder.py:157,69; inference.py:157 |

**P2 Security Analyst**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | torch.load(weights_only=False) at 12+ sites — RCE from untrusted checkpoints | high | 95 | sample_audio.py:49,66,99 et al. |
| 2 | store() ships whole-object pickled prior forcing unsafe loads downstream | med | 85 | inference.py:157 |
| 3 | Bank.store() name path traversal (no sanitization, unlike store()) | med | 90 | inference.py:572 |
| 4 | synthesize_midi unvalidated floats: allocation DoS; negative-start silent wraparound writes | low | 80 | inference.py:501-505,523-525 |
| 5 | No artefact digests in manifests; unpinned EnCodec fetch | info | 85 | inference.py:163-174 |

**P3 Diffusion Theory Researcher**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | sample_ddim uses v where ε̂ required (~44% rel-norm deviation at t=500→400); euler is the correct one; parity test vacuous (steps=1 ⇒ 0 updates) | crit→high | 95→85 | sampling.py:257; tests/test_sampling.py:79-85 |
| 2 | Sigma ladder ends at t=19: ~4% residual noise std contradicts docstring | high | 97 | schedule.py:72-77 |
| 3 | Heat-death criterion unit-dependent, unreachable (Σν≈3.05 ≫ ε), returns still-noisy x, docstring formula wrong | med | 90 | spectral_schedule.py:62-173; sampling.py:8-9 |
| 4 | Spectral conditioning trained on constant zero-derived embedding; CFG scaffold inert end-to-end | med | 92 | trainer.py:212,220; dit.py:211-225 |
| 5 | 375→750: learned absolute pos_embed binding constraint; dense attention + 50 steps adequate; LiteFocus sparsity compatible if windows are length-relative | med | 85 | dit.py:147-149 |

**P4 Signal Processing Scientist**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | Length generalization blocked only by pos_embed table (amended in R3 by unpatchify finding); patch bottleneck 16:1 at dim64; attention cost non-issue | high | 95 | dit.py:143-149,170 |
| 2 | nb02/nb04 overlap_add fades placed one segment early (dropouts + double-energy seams); nb04/07 latent_crossfade blends off-manifold features; decoder RF ≈ ±20 ms smears seams | high | 90/85 | notebooks 02·04 cells 6·9; ref nb07 cell 21 |
| 3 | synthesize_midi chained resamples cancel pitch (net pitch = f(L/target)); _resample_1d near-coprime ratios → oversized sinc kernels, per-call CPU round-trip | med-high | 88/80 | inference.py:73-80,509-521 |
| 4 | DC handled by scalar mean vs time-varying decoder bias (+0.4..+0.56); EnCodec wrapper bypasses input normalization; recursive eval feeds peak-normalized audio back → level-mismatch bias in drift metrics | med | 85 | inference.py:83-96; audio_codec.py:187-194; eval.py:844-851 |
| 5 | scattering1d truncates/zero-pads to fixed 96 000 samples — biases distributional guards for 10 s material | low-med | 90 | scattering1d.py:215-253 |

**P5 Psychoacoustics Scientist**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | Mode C renders wrong notes: second resample cancels pitch shift (worked example: +12 st requested → +52 st rendered; equal-length melodies monotone) | crit (Mode C) | 96 | inference.py:512-521 |
| 2 | Every onset a potential click: hard additive overlap, no fades/crossfades, N-clip sum can hit N pre-limiter | high | 90 | inference.py:523-525 |
| 3 | Peak normalization ≠ loudness matching: crest-factor spread → level jumps; global peak norm ducks mix under one transient | high | 85 | inference.py:92-96,534 |
| 4 | Round-robin timbre assigned AFTER time-sort: instrument identity becomes artifact of start times; no voice/channel param | med | 88 | inference.py:499,510 |
| 5 | Three conflicting durations (75/300/375); bank-mode gates measured on different duration AND architecture than production | med | 88 | sample_audio.py:34; bank_variants.py:101-115 |

**P6 Statistical Learning Scientist**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | Contraction is a model property (~440-step dim64×depth2 DiT, 179×4 s clips, deterministic sampler) — reproducible, not sampler-dial-fixable; expected to deepen at longer T | high | 90 | bank_variants.py; inference.py docstrings |
| 2 | Length×diversity interaction unmeasured; measure per-length FAD / draw-pair L1 / spectral spread FIRST to de-risk | high | 85 | — |
| 3 | Variant modes: jitter pushes off-manifold beyond α≈0.15 tolerance; residual directions likely noise at current scale; stopvar biases toward noisy quantiles | med | 80 | inference.py:257-309 |
| 4 | Diversity gates measured at 300 frames/dim64 with n=8 arms vs near-vacuous FAD ceiling — not portable to production geometry | med | 85 | bank_variants.py:22-24,61 |
| 5 | Equivalence protocol: pilot M=32 → confirm M≥64/arm, TOST with pre-registered margins, Holm correction, fresh holdout seeds ≥10000, length-normalized metrics | med | 90 | protocol spec in transcript |

**P7 Reproducibility Scientist**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | One global seed 3407 across split/train/eval; Table 3 and Table 5 share 8 bit-identical latents; structural mode collisions (canonical seed+i vs jitter seed+100+i); zero-padded refs shift FAD mechanically; write_csv overwrites in place | crit | 92 | run_evaluation.py:556-713; eval.py:472-479,943-957; data.py:77-82 |
| 2 | Unseeded randn(64,128) fallback prior — --seed does not cover it; every prior-less run silently re-randomizes conditioning geometry | crit | 98 | sample_audio.py:48-53; train_audio_diffusion.py:56-62 |
| 3 | Dead YAML; four latent_length sources; metadata.json never validated against checkpoint shapes | high | 95 | configs/audio_diffusion.yaml:7; run_evaluation.py:184; inference.py:161-174 |
| 4 | Cross-device comparability absent: CPU-hardcoded script vs auto-MPS evaluator; device-local generators make equal seeds unequal; parity tests don't cover sampled-audio metrics | high | 90 | sample_audio.py:45; run_evaluation.py:86-91; test_device_parity.py:117-151 |
| 5 | Provenance gaps: no SHA256 manifests; Bank manifest omits seed/mode/steps; time-based default seed makes default outputs permanently unregenerable | med-high | 96 | inference.py:157,163-174,582-589 |

**P8 ML Numerics Specialist**
| F | Finding | Sev | Conf | Location |
|---|---|---|---|---|
| 1 | Unpatchify scramble t'=k·N+n (executed trace); checkpoint semantics N-dependent → 750-frame generation structurally impossible from current checkpoints; pad-contamination every N-th frame at T=751 | crit | 97 | dit.py:235-237,200-203,240-241 |
| 2 | Length gate exact failure order: crash at pos_embed add (line 208) before crop; silent band [745,752] at N=94; z_init bypasses DiT entirely (condition_on_audio does zero denoising — no SDEdit) | high | 96 | dit.py:147-208; inference.py:194-204,448 |
| 3 | Device-bound generators verified (CPU≠MPS streams at same seed); generation-path RNG inventory otherwise clean; alpha_bar CPU fp32 → per-step MPS host sync | high | 93 | sampling.py:132,229,278; schedule.py:43 |
| 4 | σ-ladder endpoint (ᾱ₁₉=0.998386, floor coeff 0.0402); euler epsilon-guard inert (√(1−ᾱ₃₉)≈0.073 ≫ 1e-8); temperature applied post-hoc to near-clean z₀ distorts c_spec (a=z.mean(dim=2)) | med-high | 90 | schedule.py:72-77; inference.py:67-70,206-207 |
| 5 | synthesize_midi placement algebra: negative start → slice wraparound RuntimeError (or silent mis-write when end<0); zero duration → click; negative duration → silent drop | med | 95 | inference.py:498-525 |

## Dedup & conflict ledger

- Merged: P3-F2 + P8-F4(sigma ladder) → consensus #3. P1-F1 + P4-F1 + P8-F2(pos gate) → consensus #4 (root complex with #1). P2-F1 + P2-F2 + P7-F5(pickle/provenance) kept split: read-side exposure vs write-side format design.
- Dissent noted: owner states materialised quality currently acceptable vs P5-F1 critical Mode-C verdict — reconciled by scope decision (render polish deferred; LSD-studio materialisation path differs from synthesize_midi code path).
