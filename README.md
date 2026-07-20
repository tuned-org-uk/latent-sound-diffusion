# Arrowspace Latent Diffusion (ALD-SC)

**ArrowSpace Latent Diffusion with Spectral Chart Conditioning** — a
spectral latent diffusion model in which decoding is performed on the
feature-space manifold defined by a frozen ArrowSpace graph Laplacian
$L_F$ and its associated energy-dispersion network $\lambda^{\mathrm{ED}}$.

The model builds on the theoretical framework of the
[Entropic Semantic Diffusion Model (ESDM)](https://github.com/tuned-org-uk/entropic-semantic-diffusion)
while keeping the implementation minimal: the full vibrational machinery
(wave recurrence, density matrices, entropic pump) is deferred. What is
retained is the central geometric contract — a frozen ArrowSpace prior
defines the valid semantic subspace, and the decoder reconstructs along
the graph's smooth directions rather than through unconstrained
convolutions.

> **Central claim:** decoding on the feature-space manifold
> $(L_F, \lambda^{\mathrm{ED}})$ yields better global semantic coherence
> under compression than decoding on an unconstrained ambient latent.

---

## The research programme

The basic point of this research programme is to design **decoding using
three structures**, all computed from the training corpus via the
[ArrowSpace library](https://github.com/tuned-org-uk/pyarrowspace):

1. **The item-space** — the spatial latent $z$ carrying local image detail.
2. **The feature-space graph Laplacian** $L_F$ — its eigenvectors $U_q$
   define the smooth semantic subspace; its eigenvalues $\nu_k$ define
   entropy exchange rates.
3. **The dispersion network** $\lambda^{\mathrm{ED}}$ — ArrowSpace's
   per-feature energy-dispersion distribution
   ([arXiv:2606.21535](https://arxiv.org/abs/2606.21535)). Not a
   diagnostic, but a constructive representation of how semantic structure
   is distributed over the feature graph.

**Decoding uses $L_F$ and $\lambda^{\mathrm{ED}}$ as constructive elements
of the decoding operator**, not merely as conditioning signals. The
`WaveReconstructionBlock` propagates information along $U_q$ directions,
gated by dispersion-derived weights — the graph-theoretic analogue of the
VAE reparameterization trick.

The Barontini entropic clock governs *when* reconstruction effort is
allocated: the sampler terminates intrinsically when
$\sum_k \nu_k \bar\alpha_k(t) < \varepsilon$, and the
`ClockGatedGraphDecoder` modulates decoding tempo by $\bar\alpha_k(t)$.

See [`docs/00.md`](docs/00.md) § "The research programme" and
[`AGENTS.md`](AGENTS.md) §1.1 for the full design statement.

---

## Architecture

```text
                         Frozen ArrowSpace prior
              ┌──────────────────────────────────────┐
              │ L_F, U_q, Λ_q, λ_ED  (from ArrowSpace)│
              └──────────────────────────────────────┘
                              │
 image x ── Encoder ──► (z, A) ──► c_spec ──► Latent DiT ──► ẑ
              │         │  │                      │                │
              │         │  └─ project: A U_q U_q^T                 │
              │         │                                           │
              │         └─ DualSpaceMatrix M_N (2.5-D target)      │
              │                                                     │
              │              SpectralSchedule (Barontini clock)     │
              │                     │                               ▼
              └────────────── GraphDecoder ◄── ClockGated tempo ──► x̂
                              (WaveReconstructionBlock:
                               project → gate → lift along U_q)
```

### The 2.5-D latent

Each image encodes to:
- **z** — spatial VAE latent (local detail, what the DiT denoises)
- **A** — feature field projected onto $U_q$ (global semantic structure)
- **c_spec** $\in \mathbb{R}^{3q}$ — `[ẽ, λ_chart, ν]` (conditioning vector)

The 2.5-D encoding target is the **DualSpaceMatrix**
$M_N = \alpha\|VV^\top\|_F - \beta\|V L_F V^\top\|_F$, which fuses
item-space geometry with feature-space topology.

### The graph-structured decoder

The `WaveReconstructionBlock` at each resolution:
1. Pool feature activations → $A$
2. Project to chart: $\hat{H} = A \cdot U_q$ (decode along smooth directions)
3. Gate by dispersion: $g = \sigma(W \cdot c_{\mathrm{spec}})$ (energy allocation)
4. Lift back: $A' = (\hat{H} \odot g) \cdot U_q^\top$ (reconstruct in feature space)
5. Residual conv update

The `ClockGatedGraphDecoder` modulates gate strength by
$\bar\alpha_k(t)$: early in denoising (high noise) gates are weak; late
(low noise) gates are strong.

---

## What is implemented

| Component | File | Description |
|-----------|------|-------------|
| ArrowSpace adapter | `wire_graph.py` | $L_F$ + $\lambda^{\mathrm{ED}}$ via pyarrowspace or kNN fallback |
| Frozen prior | `arrow_prior.py`, `build_prior.py` | $L_F$, $U_q$, $\Pi_q$, $c_{\mathrm{spec}}$ as buffers (zero `nn.Parameter`) |
| 2.5-D encoding target | `dual_space.py` | $M_N = \alpha\|VV^\top\|_F - \beta\|V L_F V^\top\|_F$ |
| Spectral VAE | `vae.py` | Dual-head encoder (spatial + feature), legacy single-gate decoder |
| DiT denoiser | `dit.py` | Patchify + AdaLN + CFG dropout on $c_{\mathrm{spec}}$ |
| Schedules | `schedule.py` | Cosine + linear, v-prediction (`add_noise`, `v_target`) |
| Graph decoder | `graph_decoder.py` | `WaveReconstructionBlock`, `GraphDecoder`, `ClockGatedGraphDecoder` |
| Entropic clock | `spectral_schedule.py` | $\tau_k(t)$, $\bar\alpha_k(t)$, heat-death stopping criterion |
| Samplers | `sampling.py` | DDIM + Euler with spectral stopping criterion |
| Losses | `losses.py` | $L_{\mathrm{diff}}$ + $L_{\mathrm{rec}}$ + $L_{\mathrm{chart}}$ + $L_{\mathrm{smooth}}$ + $L_{\mathrm{kl}}$ |
| Training | `trainer.py` | `train_vae()` + `train_diffusion()` (yields loss dicts) |
| Data | `data.py` | `ImageFolderDataset`, `ToyImageDataset`, `build_dataloader()` |
| CLI | `scripts/sample.py` | End-to-end image generation |

**107 unit tests**, all on CPU. `uv run pytest tests/ -v`.

---

## Status

| Phase | Scope | State |
|---|---|---|
| Phase 1 | Spectral VAE + DiT + sampling (image generation) | ✅ Complete |
| Phase 2 | Paper honesty pass + entropic clock in samplers | ✅ Complete |
| Phase 3 | Graph-structured decoding (research contribution) | ✅ Complete |
| Phase 4 | Real-data experiments + metrics + wave recurrence | Tracked in issues |

### Open issues (limitations)

- [#9](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/9) — Real-data experiments: CIFAR-10 with DINO/SigLIP embeddings
- [#10](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/10) — Full second-order wave recurrence in WaveReconstructionBlock
- [#11](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/11) — Entropic training schedule (not just inference stopping)
- [#12](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/12) — Quantitative metrics: FID, PSNR, SSIM, LPIPS, spectral diagnostics
- [#8](https://github.com/tuned-org-uk/arrowspace-latent-diffusion/issues/8) — *(stretch)* Joint fine-tuning & controllable editing

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and requires Python ≥ 3.13 with PyTorch ≥ 2.2.

```bash
git clone https://github.com/tuned-org-uk/arrowspace-latent-diffusion.git
cd arrowspace-latent-diffusion
uv sync
```

## Usage

```bash
# Run the test suite (CPU; 107 tests)
uv run pytest tests/ -v

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/

# Generate an image
uv run python scripts/sample.py --out results/sample.png

# With options
uv run python scripts/sample.py --steps 50 --seed 3407 --epochs 20 --out results/sample.png
```

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_noise_schedule.ipynb` | Cosine/linear schedules, v-prediction, forward corruption |
| 02 | `02_arrow_prior.ipynb` | Frozen ArrowSpace prior, eigenvalues, projector, c_spec |
| 03 | `03_spectral_vae.ipynb` | VAE training, reconstruction, band-energy comparison |
| 04 | `04_dit_conditioning.ipynb` | DiT velocity prediction, c_spec sensitivity, CFG dropout |
| 05 | `05_train_diffusion.ipynb` | Latent diffusion training, v-prediction loss |
| 06 | `06_sampling.ipynb` | DDIM sampling, with-vs-without c_spec ablation |
| 07 | `07_spectral_schedule.ipynb` | Per-mode entropic schedule, heat-death criterion |
| 08 | `08_graph_decoder.ipynb` | Graph decoder vs clock-gated decoder at different times |

---

## Repository layout

```
arrowspace-latent-diffusion/
├── pyproject.toml                # uv / hatchling project config
├── AGENTS.md                     # contributor guide (read this first)
├── docs/
│   ├── 00.md                     # design document — the research programme
│   ├── 01.md                     # design document — ESDM transfer
│   └── paper/
│       └── ald-sc.tex            # the paper
├── notebooks/                    # 01–08 (numbered milestones)
├── scripts/
│   └── sample.py                 # CLI image generation
├── src/ald_sc/
│   ├── __init__.py
│   ├── arrow_prior.py            # ArrowSpacePrior: frozen spectral prior
│   ├── build_prior.py            # build_arrow_prior() from corpus embeddings
│   ├── data.py                   # ImageFolderDataset, ToyImageDataset
│   ├── dit.py                    # MinimalDiT: patchify + AdaLN + CFG
│   ├── dual_space.py             # DualSpaceMatrix M_N (2.5-D encoding target)
│   ├── graph_decoder.py          # WaveReconstructionBlock, GraphDecoder,
│   │                             #   ClockGatedGraphDecoder
│   ├── losses.py                 # ALDSCLoss: diff + rec + chart + smooth + kl
│   ├── sampling.py               # sample_euler(), sample_ddim() + spectral stopping
│   ├── schedule.py               # CosineSchedule, LinearSchedule (v-prediction)
│   ├── spectral_schedule.py      # Per-mode τ_k, ᾱ_k, heat-death criterion
│   ├── trainer.py                # train_vae(), train_diffusion()
│   ├── vae.py                    # SpectralVAE: dual-head encoder
│   └── wire_graph.py             # ArrowSpace adapter: L_F + λ_ED
└── tests/                        # 13 test files, 107 tests
```

---

## Design constraints

- **Frozen prior.** $L_F$, $U_q$ are buffers, never parameters. The graph
  defines the valid semantic geometry; learning happens on top of it.
- **Decoding on the feature-space manifold.** $L_F$ defines reconstruction
  paths (via $U_q$); $\lambda^{\mathrm{ED}}$ defines energy allocation. Not
  conditioning bolted on top — the graph structures *are* the decoding
  operator.
- **Diffusion runs on $z$ only.** No second diffusion process over the
  spectral chart $s$.
- **Corpus-level prior, not per-image.** Do not construct a new graph per
  image.
- **Barontini clock governs when, not how much.** The entropic clock
  provides an intrinsic stopping criterion and decoder tempo modulation,
  not a per-mode noise schedule (ν_k cancels in external time).

See [`AGENTS.md`](AGENTS.md) §1.1 and §6 for the full design constraints.

---

## References

- Design documents: [`docs/00.md`](docs/00.md), [`docs/01.md`](docs/01.md)
- Paper: [`docs/paper/ald-sc.tex`](docs/paper/ald-sc.tex)
- [Diffusion as spectral-geometric projection](https://www.tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/) — theoretical background
- [`entropic-semantic-diffusion`](https://github.com/tuned-org-uk/entropic-semantic-diffusion) — predecessor (full entropic clock)
- [`arrowspace-diffusion-from-scratch`](https://github.com/tuned-org-uk/arrowspace-diffusion-from-scratch) — pedagogical template
- [`pyarrowspace`](https://github.com/tuned-org-uk/pyarrowspace) — ArrowSpace library (Rust bindings)
- [Energy Dispersion Networks](https://arxiv.org/abs/2606.21535) — dispersion network concept
- [ArrowSpace — Spectral Search for Embeddings](https://doi.org/10.21105/joss.09002) (JOSS)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022)
- Barontini, *Testing the problem of time with cold atoms* (PRL 2026)
- Stancevic et al., *Entropic Time Schedulers for Generative Diffusion Models* (arXiv 2025)

## License

MIT
