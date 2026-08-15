# How LSD Sound Generation Works

> A concise explanation of the Latent Sound Diffusion (LSD) model,
> with references to its predecessors ALD-SC and ESDM.

## The one-paragraph version

LSD generates audio by running a **diffusion process on EnCodec's
continuous latent space** and then **decoding the denoised latent through a
graph-structured decoder** that uses a frozen ArrowSpace feature-space
Laplacian $L_F$ to define reconstruction paths and a dispersion network
$\lambda^{\mathrm{ED}}$ to allocate reconstruction energy. The central
claim — inherited from [ALD-SC](https://github.com/tuned-org-uk/arrowspace-latent-diffusion)
and ultimately from [ESDM](https://github.com/tuned-org-uk/entropic-semantic-diffusion) —
is that decoding on this feature-space manifold produces better global
semantic coherence than decoding through unconstrained convolutions.

---

## Lineage: ESDM → ALD-SC → LSD

| Model | Modality | Key innovation | What it added |
|---|---|---|---|
| **ESDM** | Images | Frozen ArrowSpace prior + full entropic clock (wave recurrence, density matrices, vibrational harness) | The research programme: decoding on the feature-space manifold |
| **ALD-SC** | Images | Simplified ESDM: frozen prior retained, vibrational machinery deferred; DiT backbone + spectral chart conditioning | Practical implementation of the project→gate→lift decoder |
| **LSD** | Audio | Replaces image VAE with frozen EnCodec; 1-D DiT + 1-D graph decoder; self-consistent c_spec from z | Extends the claim to audio; controlled graph-vs-baseline comparison |

LSD is a sound-generation-specific fork of ALD-SC. The research programme
(`docs/00.md` § "The research programme") is modality-agnostic and
carries over unchanged. Only the data representation and spatial
dimensions change: 2-D image latents become 1-D audio latents.

---

## The pipeline

```
                    ┌─────────── FROZEN (built once from corpus) ───────────┐
                    │  ArrowSpace prior: L_F, U_q, ν_k, λ_ED → c_spec       │
                    └───────────────────────────────────────────────────────┘
                                          │
  audio x ── EnCodec ──► z, A = pool(z), c_spec = prior.chart_energy_descriptor(A)
  (frozen)    │           (B,128,T)  (B,128)        (B, 3q)
  encoder     │
              │         ┌─── TRAINABLE ───────────────────────────┐
              │         │                                          │
  noise ───► 1-D DiT ──► z_gen ───► GraphDecoder(z_gen, c_spec) ──► waveform
              │         (Conv1d        (WaveReconstructionBlock:    (B, 1, L)
              │          patchify      project → gate → lift
              │          + AdaLN)      along U_q, gated by λ_ED)
              │                        + 320× Conv1d upsampling
              │         └──────────────────────────────────────────┘
```

### Step-by-step

1. **Build the prior** (once, offline). Extract EnCodec features from a
   corpus of audio clips (e.g. ESC-50). Pool each clip's features over
   time → $(N, 128)$ embedding matrix. Build the ArrowSpace prior:
   - **$L_F$** $(128 \times 128)$: feature-space graph Laplacian via kNN
     over feature columns
   - **$U_q$** $(128 \times q)$: leading $q$ eigenvectors (smooth modes)
   - **$\nu_k$** $(q)$: eigenvalues (entropy exchange rates)
   - **$\lambda^{\mathrm{ED}}$** $(128)$: per-feature energy-dispersion
     distribution

   These are frozen buffers — zero trainable parameters. See
   `arrow_prior.py`, `build_prior.py`, `wire_graph.py`.

2. **Encode audio** (frozen EnCodec). A raw waveform $x \in \mathbb{R}^{B
   \times 1 \times L}$ passes through EnCodec's encoder (24 kHz, stride
   320) to produce a continuous 1-D latent $z \in \mathbb{R}^{B \times
   128 \times T}$ where $T = L/320$ (75 Hz frame rate). We use
   **pre-quantization** features — the continuous encoder output before
   RVQ — because diffusion needs a continuous space.

3. **Derive conditioning** (self-consistent). Pool $z$ over time:
   $A = \mathrm{mean}_T(z) \in \mathbb{R}^{B \times 128}$. Compute the
   spectral chart conditioning vector:
   $$c_{\mathrm{spec}} = [\tilde{e}, \; \lambda^{\mathrm{chart}}, \; \nu]
   \in \mathbb{R}^{B \times 3q}$$
   where $\tilde{e}_k = \|A u_k\|^2 / \sum_j \|A u_j\|^2$ are normalized
   band energies and $\lambda^{\mathrm{chart}}_k = \sum_f
   \lambda^{\mathrm{ED}}_f U_{f,k}^2$ is the dispersion projected onto the
   chart. This vector encodes *how the clip's energy is distributed over
   the graph's smooth modes*.

4. **Diffuse** (1-D DiT). The 1-D DiT (`dit.py`) is trained to predict
   velocity $v$ in v-prediction parameterization:
   $$z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,
   \quad v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1-\bar\alpha_t}\, z_0$$
   The DiT patchifies $z_t$ with `Conv1d` (patch size 8 → ~47 tokens for
   $T=375$), applies AdaLN conditioning on timestep only (unconditional
   Phase 1), and predicts $v$. A cosine noise schedule
   (`schedule.py`) controls $\bar\alpha_t$.

5. **Sample** (DDIM/Euler). Starting from pure noise, the sampler
   (`sampling.py`) iteratively denoises using the DiT's velocity
   predictions to recover $z_0$. The Barontini entropic clock
   (`spectral_schedule.py`) provides an optional intrinsic stopping
   criterion: sampling terminates when $\sum_k \nu_k
   \nu_k (1-\bar\alpha_k(t)) < \varepsilon$ (heat death of the reverse
   process: nothing measurable left to resolve).

6. **Decode** (graph decoder). This is the research contribution. The
   `GraphDecoder` (`graph_decoder.py`) takes $z_0$ and $c_{\mathrm{spec}}$
   and produces a waveform via 4 upsampling stages (strides 2, 4, 5, 8 →
   320×, matching EnCodec's stride). At each resolution, a
   `WaveReconstructionBlock` performs:

   **Project**: $\hat{H} = A \cdot U_q$ — project pooled features onto
   the smooth chart basis (decode along graph eigenvector directions)

   **Gate**: $g = \sigma(W \cdot c_{\mathrm{spec}})$ — dispersion network
   gates how much reconstruction energy each mode receives

   **Lift**: $A' = (\hat{H} \odot g) \cdot U_q^\top$ — reconstruct in
   feature space, then broadcast back to the temporal axis

   **Residual conv**: $h_{\mathrm{out}} = h + \mathrm{Conv1d}(\mathrm{SiLU}(
   \mathrm{GroupNorm}(h + \delta)))$

   The `ClockGatedGraphDecoder` variant modulates gate strength by the
   entropic clock: $\mathrm{tempo} = \mathrm{mean}(\bar\alpha_k(t))$ —
   gates are weak early (high noise) and strong late (low noise).

---

## The central claim and controlled experiment

**Claim**: decoding on the feature-space manifold $(L_F,
\lambda^{\mathrm{ED}})$ yields better global semantic coherence under
compression than decoding on an unconstrained ambient latent.

**Test**: LSD trains two decoders with **identical capacity** (same
channels, same upsampling strides):

- **Graph decoder**: uses `WaveReconstructionBlock` (project→gate→lift
  along $U_q$, gated by $\lambda^{\mathrm{ED}}$)
- **Baseline decoder**: uses plain `ResBlock1d` (no $U_q$, no
  $\lambda^{\mathrm{ED}}$)

The only variable is graph structure. Both decode the same $z_0$ from the
same frozen EnCodec encoder. Reconstruction quality is compared via L1,
multi-scale STFT loss, chart-energy error, and off-manifold ratio.

---

## What comes from ESDM, what is simplified, what is new

### Retained from ESDM (via ALD-SC)

- **Frozen ArrowSpace prior** — $L_F$, $U_q$ are buffers, never parameters
  (`arrow_prior.py`). The graph defines valid semantic geometry; learning
  happens on top of it.
- **Projection** $\Phi(A) = A U_q U_q^\top$ — smooth subspace projection
  (`arrow_prior.py:project_to_chart`).
- **Spectral chart conditioning** — $c_{\mathrm{spec}} = [\tilde{e},
  \lambda^{\mathrm{chart}}, \nu]$ as compact conditioning
  (`arrow_prior.py:chart_energy_descriptor`).
- **Heat-death stopping** — $\sum_k \nu_k \bar\alpha_k(t) < \varepsilon$
  (`spectral_schedule.py`, used in `sampling.py`).

### Simplified from ESDM (deferred to Phase 3)

- **Single graph-filter step** per block instead of full second-order wave
  recurrence $Q_{t+1} = 2Q_t - Q_{t-1} - \Delta\tau^2 L_F Q_t$.
- **FiLM-like spectral gate** instead of wave-based reconstruction with
  density matrices.
- **No Rayleigh-gradient restoring force**, no learned entropic pump.
- **Corpus-level frozen prior**, not per-sample dynamic Laplacian.

See `docs/01.md` § "What is deferred" for the full ESDM→ALD-SC transfer
table.

### New in LSD (audio-specific)

- **Frozen EnCodec encoder** replaces the trainable image VAE encoder.
  The encoder is never trained — only the decoder and DiT have trainable
  parameters.
- **1-D architecture**: `Conv1d` patchify in DiT, `Conv1d` + temporal
  pooling in graph decoder, 1-D upsampling matching EnCodec's 320× stride.
- **Self-consistent $c_{\mathrm{spec}}$**: derived from $z$ itself
  ($A = \mathrm{mean}_T(z)$), not from a separate feature head. This
  means the decoder is self-consistent at both training and generation
  time — no separate $c_{\mathrm{spec}}$ sampling needed.
- **Controlled graph-vs-baseline comparison** as the core experimental
  design.

---

## Key files

| File | Role | Trainable? |
|---|---|---|
| `arrow_prior.py` | Frozen prior: $L_F$, $U_q$, $\lambda^{\mathrm{ED}}$, $c_{\mathrm{spec}}$ | No (buffers) |
| `build_prior.py` | Build prior from corpus embeddings via kNN + eigendecomposition | No |
| `wire_graph.py` | ArrowSpace adapter (pyarrowspace or kNN fallback) | No |
| `audio_codec.py` | `EnCodecEncoder` (frozen), `BaselineAudioDecoder`, `AudioVAE` | Encoder: No; Baseline decoder: Yes |
| `dit.py` | 1-D DiT denoiser (Conv1d patchify + AdaLN + CFG) | Yes |
| `graph_decoder.py` | `WaveReconstructionBlock` (project→gate→lift), `GraphDecoder`, `ClockGatedGraphDecoder` | Yes |
| `schedule.py` | Cosine/linear noise schedules (v-prediction) | No |
| `spectral_schedule.py` | Per-mode $\tau_k(t)$, $\bar\alpha_k(t)$, heat-death criterion | No |
| `sampling.py` | DDIM/Euler samplers with spectral stopping | No |
| `losses.py` | L1 + multi-scale STFT + chart + smooth | No |
| `trainer.py` | `train_audio_decoder()`, `train_audio_diffusion()` | — |

---

## References

- **ESDM**: [entropic-semantic-diffusion](https://github.com/tuned-org-uk/entropic-semantic-diffusion) — full vibrational harness, entropy clock, density matrices
- **ALD-SC**: [arrowspace-latent-diffusion](https://github.com/tuned-org-uk/arrowspace-latent-diffusion) — simplified ESDM for images, project→gate→lift decoder
- **LSD**: [latent-sound-diffusion](https://github.com/tuned-org-uk/latent-sound-diffusion) — this repo, audio fork
- **ArrowSpace**: [pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace) — graph Laplacian + dispersion network library
- **Energy Dispersion Networks**: [arXiv:2606.21535](https://arxiv.org/abs/2606.21535)
- **EnCodec**: Défossez et al., 2022
- **Barontini clock**: Barontini, *Testing the problem of time with cold atoms* (PRL 2026)
- **Design docs**: [`docs/00.md`](00.md) (research programme), [`docs/01.md`](01.md) (ESDM transfer), [`docs/02.md`](02.md) (audio adaptation)
