# Postmortem: c_spec DiT Conditioning Regression (v0.5.4–v0.8.0)

## Summary

Models trained on versions v0.5.4 through v0.8.0 produced significantly less
varied and less interesting sounds, and took longer to train. The root cause
was the decision at v0.5.4 (PR #30) to pass `c_spec` (spectral conditioning)
into the DiT during training and inference. Each subsequent version compounded
this with additional machinery (CFG, eta mapping, Min-SNR weighting, per-sample
conditioning) that patched symptoms rather than addressing the root cause.

This document records the wrong decisions so they are not repeated.

## Root Cause: v0.5.4 (PR #30)

At v0.5.3, `train_audio_diffusion` calls the DiT as `dit(z_t, t)` — a fully
unconditional forward pass. From v0.5.4 this became `dit(z_t, t, c_spec=c_spec)`,
passing the spectral conditioning vector directly into the denoiser. The model
now learns to denoise toward specific spectral regions rather than freely
exploring the latent space. At inference, every sample in a bank is anchored
to the same spectral attractor — diversity comes only from per-sample noise
seeds, not from genuinely diverse latent trajectories.

## Compounding Decisions

| Version | Change | Why it was wrong |
|---------|--------|-----------------|
| v0.5.4 | `c_spec` passed to DiT in training + `target_c_spec` in bank generator | Spectral bias funnel — all bank samples converge on same region |
| v0.5.7 | CFG two-pass `cfg_forward` + `guidance_scale` | Doubles forward cost per step; amplified conditioning reduces off-manifold exploration |
| v0.6.0 | `--temperature` remapped to DDIM `eta` | Removes post-hoc latent scaling that provided cheap diversity |
| v0.7.0 | Min-SNR loss weighting (default) + `chart_loss` fix + `grad_clip` | Training distribution collapses toward lower-variance attractors; longer training time |
| v0.8.0 | Per-sample `c_spec` seeding | Adds complexity to patch a problem that shouldn't exist |

## What v0.5.3 Had Right

- **Unconditional DiT**: `dit(z_t, t)` — no spectral steering during diffusion.
- **`noise_std` in `train_audio_decoder`**: the primary lever for sound diversity.
  Gaussian noise injected into the latent `z` before decoding. The docstring
  explicitly calls it "a feature for artistic exploration rather than a bug."
- **`_apply_temperature`**: post-hoc latent scaling (`z * temperature`) with
  no-op at `temperature == 1.0`. Diversity at default settings comes entirely
  from `seed + i`.
- **`c_spec` for decoder gate only**: derived post-hoc from `z.mean(dim=2)`,
  used by the decoder, never by the DiT.
- **`chart_loss` no-op**: `A_hat = A.detach()` makes `chart_loss` always zero.
  This was a silent no-op due to the detach, not because of a `0.0` default.
  The v0.7.0 "fix" activated it, contributing to the training collapse.

## Principles to Prevent Recurrence

1. **Noise-seed diversity is the primary mechanism.** The diffusion model's
   job is to explore the latent manifold freely. Spectral steering constrains
   it to a subspace. If diversity is lacking, increase `noise_std` or vary
   seeds — do not add conditioning to the DiT.

2. **`c_spec` is a decoder gate, not a DiT conditioner.** The spectral chart
   descriptor shapes how the decoder turns a latent into a waveform. It must
   never steer the diffusion process itself.

3. **Preserve the `chart_loss` no-op deliberately.** The `A_hat = A.detach()`
   pattern in `trainer.py` makes `chart_loss` zero. This is intentional for
   Phase 1 — the constraint is too strong and collapses training toward
   lower-variance outputs. Do not "fix" this without understanding why it was
   a no-op in the first place.

4. **`noise_std` / `NOISE_INJECT` is the diversity lever.** It corresponds to
   `NOISE_INJECT` in notebook 07. It must not be removed or disabled.

5. **Avoid compounding fixes.** When a change reduces diversity, do not add
   more machinery to compensate. Revert the change. Each layer of
   compensation makes the system harder to understand and debug.

## Resolution

- `main` re-pointed to `simplify/v053-core` (v0.5.3 + neutral fixes only)
- Post-v0.5.3 work preserved on `stash/post-053-complexity` branch
- Neutral fixes cherry-picked back: weight_norm migration, STFT window
  caching, safetensors export + config.json, CI workflow
- Abandoned tags v0.5.4–v0.8.0 deleted
- See [issue #42](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/42)
  for the full investigation
