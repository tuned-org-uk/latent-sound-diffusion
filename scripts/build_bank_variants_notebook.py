"""Build notebooks/08_bank_variant_modes.ipynb for issue #53 (v0.11.0).

The notebook exercises the bank-variant modes wired into
``LSDModel.generate_sound_bank`` (``bank_mode`` / ``bank_variety``) on the
trained checkpoints in ``results/artifacts/``: generates banks in every
mode, measures the diversity metrics used by the pre-registered gate
(pairwise waveform L1, centroid spread, RMS), sweeps the ``bank_variety``
dial, writes audition WAVs, and compares against the recorded experiment
CSV (``results/bank_variants.csv``).

Requires ``results/artifacts/`` from a prior ``scripts/run_evaluation.py``
run (same as ``scripts/bank_variants.py``).

    uv run python scripts/build_bank_variants_notebook.py
    uv run jupyter nbconvert --to notebook --execute notebooks/08_bank_variant_modes.ipynb
"""

from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython",
        "version": "3.13.12",
    },
}

cells = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# Bank Variant Modes — Testing the Variations System (v0.11.0, issue #53)\n"
    "\n"
    "v0.11.0 added explicit **variant modes** to bank generation. The\n"
    "undertrained unconditional DiT contracts every noise draw to (nearly)\n"
    "one latent (#52), so raw DDIM draws yield n near-identical clips. The\n"
    "decoder, however, trained with NOISE_INJECT, tolerates a latent\n"
    "neighbourhood — so we vary *around the canonical draw* $\\bar{z}$ and\n"
    "decode each variant.\n"
    "\n"
    "Interface (`LSDModel.generate_sound_bank`):\n"
    "\n"
    "| `bank_mode` | latent rule | `bank_variety` |\n"
    "|---|---|---|\n"
    "| `canonical` (default) | n independent DDIM draws | — |\n"
    "| `jitter` | $\\bar{z} + \\alpha\\,\\sigma_z\\,\\varepsilon_i$ | noise amplitude $\\alpha$ |\n"
    "| `residual` | $\\bar{z} + k\\,(z_i - \\bar{z})$ | amplified residual rel. std |\n"
    "| `stopvar` | same seed, step count swept 0.24–0.98·steps | unused |\n"
    "\n"
    "Pre-registered gate (#53): **USEFUL iff pairwise L1 ≥ 0.05 AND FAD ≤ 1200\n"
    "AND RMS ≥ 0.35 AND centroid spread > 20 Hz**. Gate-passing arms:\n"
    "`jitter_a0.5` (L1 0.182), `resid_r0.3` (0.112), `stopvar` (0.073).\n"
    "\n"
    "This notebook: loads the trained checkpoints, generates banks in every\n"
    "mode, measures the gate metrics, sweeps the variety dial, writes\n"
    "audition WAVs, and cross-checks against `results/bank_variants.csv`."
)

md(
    "## Setup\n"
    "\n"
    "Loads the prior, graph decoder and DiT from `results/artifacts/`\n"
    "(produced by `scripts/run_evaluation.py`). MPS is used when available."
)

code(
    "import random\n"
    "import statistics\n"
    "from pathlib import Path\n"
    "\n"
    "import soundfile as sf\n"
    "import torch\n"
    "\n"
    "from ald_sc.audio_codec import EnCodecEncoder\n"
    "from ald_sc.build_prior import build_arrow_prior\n"
    "from ald_sc.dit import MinimalDiT\n"
    "from ald_sc.eval import spectral_centroid\n"
    "from ald_sc.graph_decoder import GraphDecoder\n"
    "from ald_sc.inference import BANK_MODES, LSDModel\n"
    "from ald_sc.schedule import CosineSchedule\n"
    "\n"
    "REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "ARTIFACTS = REPO / 'results' / 'artifacts'\n"
    "OUT_DIR = REPO / 'notebooks' / 'results' / 'bank_variants'\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "assert (ARTIFACTS / 'dit.pt').exists(), (\n"
    "    'results/artifacts/ missing — run scripts/run_evaluation.py first'\n"
    ")\n"
    "\n"
    "SEED, N_BANK, STEPS = 3407, 8, 50\n"
    "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')\n"
    "random.seed(SEED)\n"
    "torch.manual_seed(SEED)\n"
    "print(f'device={device}')"
)

code(
    "prior = build_arrow_prior(\n"
    "    torch.load(ARTIFACTS / 'embeddings.pt', weights_only=False), q=8, k=4\n"
    ").to(device)\n"
    "decoder = GraphDecoder(\n"
    "    latent_channels=128, out_channels=1, feature_dim=128,\n"
    "    base_channels=32, prior=prior, upsample_strides=(2, 4, 5, 8),\n"
    ").to(device).eval()\n"
    "decoder.load_state_dict(\n"
    "    torch.load(ARTIFACTS / 'graph_dec.pt', weights_only=False, map_location='cpu')\n"
    ")\n"
    "dit = MinimalDiT(\n"
    "    latent_channels=128, latent_length=300, patch_size=8,\n"
    "    dim=64, depth=2, num_heads=4, spec_dim=24,\n"
    ").to(device).eval()\n"
    "dit.load_state_dict(\n"
    "    torch.load(ARTIFACTS / 'dit.pt', weights_only=False, map_location='cpu')\n"
    ")\n"
    "encoder = EnCodecEncoder().to(device).eval()\n"
    "model = LSDModel(\n"
    "    prior=prior, dit=dit, decoder=decoder, encoder=encoder,\n"
    "    schedule=CosineSchedule(num_steps=1000),\n"
    ")\n"
    "print(f'model loaded — BANK_MODES = {BANK_MODES}')"
)

md(
    "## Generate a bank in every mode\n"
    "\n"
    "Defaults follow the gate-passing arms: `jitter` at variety 0.5,\n"
    "`residual` at 0.3. Each bank is n=8 clips, 50 DDIM steps."
)

code(
    "banks = {\n"
    "    'canonical': model.generate_sound_bank(n=N_BANK, steps=STEPS, seed=SEED),\n"
    "    'jitter_0.5': model.generate_sound_bank(\n"
    "        n=N_BANK, steps=STEPS, seed=SEED, bank_mode='jitter', bank_variety=0.5\n"
    "    ),\n"
    "    'residual_0.3': model.generate_sound_bank(\n"
    "        n=N_BANK, steps=STEPS, seed=SEED, bank_mode='residual', bank_variety=0.3\n"
    "    ),\n"
    "    'stopvar': model.generate_sound_bank(\n"
    "        n=N_BANK, steps=STEPS, seed=SEED, bank_mode='stopvar'\n"
    "    ),\n"
    "}\n"
    "print({k: len(v) for k, v in banks.items()})"
)

md(
    "## Measure the gate metrics per mode\n"
    "\n"
    "Pairwise waveform L1 (diversity), centroid spread (Hz), RMS\n"
    "(degeneracy guard) — the same metrics as the pre-registered gate\n"
    "(FAD-proxy omitted here for speed; see `scripts/bank_variants.py`\n"
    "for the full protocol)."
)

code(
    "def bank_metrics(clips):\n"
    "    l1 = [\n"
    "        float((clips[a] - clips[b]).abs().mean())\n"
    "        for a in range(len(clips))\n"
    "        for b in range(a + 1, len(clips))\n"
    "    ]\n"
    "    cents = [float(spectral_centroid(c).mean()) for c in clips]\n"
    "    rms = float(\n"
    "        torch.cat([c.pow(2).mean().unsqueeze(0) for c in clips]).sqrt().mean()\n"
    "    )\n"
    "    return {\n"
    "        'L1_mean': round(statistics.mean(l1), 4),\n"
    "        'L1_min': round(min(l1), 4),\n"
    "        'L1_max': round(max(l1), 4),\n"
    "        'spread_hz': round(statistics.stdev(cents), 1),\n"
    "        'centroid_hz': round(statistics.mean(cents), 1),\n"
    "        'RMS': round(rms, 3),\n"
    "    }\n"
    "\n"
    "rows = []\n"
    "for name, clips in banks.items():\n"
    "    m = bank_metrics([c.cpu() for c in clips])\n"
    "    m.update(mode=name, gate='USEFUL' if (\n"
    "        m['L1_mean'] >= 0.05 and m['RMS'] >= 0.35 and m['spread_hz'] > 20\n"
    "    ) else '-')\n"
    "    rows.append(m)\n"
    "\n"
    "print(f\"{'mode':<12} {'L1':>7} {'spread':>7} {'RMS':>6}  gate\")\n"
    "for r in rows:\n"
    "    print(f\"{r['mode']:<12} {r['L1_mean']:>7} {r['spread_hz']:>7} \"\n"
    "          f\"{r['RMS']:>6}  {r['gate']}\")"
)

md(
    "Expected (from `results/bank_variants.csv`, gate-passing arms):\n"
    "\n"
    "- `canonical`: L1 ≈ 0.000 — the contraction; every draw the same clip\n"
    "- `jitter_0.5`: L1 ≈ 0.182, spread ≈ 41 Hz\n"
    "- `residual_0.3`: L1 ≈ 0.112, spread ≈ 192 Hz\n"
    "- `stopvar`: L1 ≈ 0.073, spread ≈ 51 Hz"
)

md(
    "## Sweep the `bank_variety` dial (jitter)\n"
    "\n"
    "`bank_variety` is the 0–1 diversity slider (LSD-studio mapping:\n"
    "dropdown = `bank_mode`, slider = `bank_variety`). Diversity should be\n"
    "monotone in the dial, with 0 collapsing to the canonical clip."
)

code(
    "sweep = []\n"
    "for a in (0.0, 0.05, 0.1, 0.25, 0.5):\n"
    "    bank = model.generate_sound_bank(\n"
    "        n=N_BANK, steps=STEPS, seed=SEED,\n"
    "        bank_mode='jitter', bank_variety=a,\n"
    "    )\n"
    "    m = bank_metrics([c.cpu() for c in bank])\n"
    "    m.update(alpha=a)\n"
    "    sweep.append(m)\n"
    "\n"
    "print(f\"{'alpha':>6} {'L1':>7} {'spread':>7} {'RMS':>6}\")\n"
    "for r in sweep:\n"
    "    print(f\"{r['alpha']:>6} {r['L1_mean']:>7} {r['spread_hz']:>7} {r['RMS']:>6}\")"
)

md(
    "## Audition: write WAVs\n"
    "\n"
    "One subdirectory per mode, `NN.wav` per clip. Listen and judge the\n"
    "qualitative claim behind the gate: a coherent palette (same frozen\n"
    "manifold) with audible intra-bank variation — vs the canonical bank's\n"
    "n identical clips."
)

code(
    "for name, clips in banks.items():\n"
    "    mode_dir = OUT_DIR / name\n"
    "    mode_dir.mkdir(parents=True, exist_ok=True)\n"
    "    for i, clip in enumerate(clips):\n"
    "        sf.write(str(mode_dir / f'{i:02d}.wav'), clip.squeeze(0).cpu().numpy(), 24000)\n"
    "print(f'wrote banks under {OUT_DIR}')\n"
    "for p in sorted(OUT_DIR.glob('*/*.wav'))[:4]:\n"
    "    print(' ', p.relative_to(REPO))"
)

md(
    "## Cross-check against the experiment record\n"
    "\n"
    "`results/bank_variants.csv` is the frozen decision record (13 arms).\n"
    "The wired library modes must reproduce its gate-arm rows bit-for-bit."
)

code(
    "import csv\n"
    "\n"
    "csv_path = REPO / 'results' / 'bank_variants.csv'\n"
    "with open(csv_path) as f:\n"
    "    record = {r['arm']: r for r in csv.DictReader(f)}\n"
    "\n"
    "for name, arm in (\n"
    "    ('jitter_0.5', 'jitter_a0.5'),\n"
    "    ('residual_0.3', 'resid_r0.3'),\n"
    "    ('stopvar', 'stopvar'),\n"
    "):\n"
    "    rec = record[arm]\n"
    "    row = next(r for r in rows if r['mode'] == name)\n"
    "    match = abs(row['L1_mean'] - float(rec['l1_mean'])) < 0.005\n"
    "    print(f\"{name:<12} notebook L1 {row['L1_mean']:.4f}  \"\n"
    "          f\"csv {float(rec['l1_mean']):.4f}  match={match}\")"
)

md(
    "## Reproducibility & interface notes\n"
    "\n"
    "- Same seed → identical banks per mode (unit-tested in\n"
    "  `tests/test_inference.py::TestBankModes`).\n"
    "- `bank_variety=0` reduces `jitter`/`residual` exactly to the canonical\n"
    '  clip — the slider\'s left end is a safe "same as Stock" anchor.\n'
    "- Invalid `bank_mode` raises `ValueError`; the UI should bind its\n"
    "  dropdown directly to the exported `BANK_MODES` tuple.\n"
    "- `Bank.from_generation(model, bank_mode=..., bank_variety=...)`\n"
    "  wraps generation with provenance for studio storage.\n"
    "\n"
    "Related: [issue #53](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/53) "
    "(decision record), [issue #58](https://github.com/tuned-org-uk/latent-sound-diffusion/issues/58) "
    "(output-noise follow-up), PR #55 (implementation)."
)

nb.cells = cells

out = __import__("pathlib").Path(__file__).resolve().parent.parent / "notebooks"
out.mkdir(exist_ok=True)
path = out / "08_bank_variant_modes.ipynb"
nbf.write(nb, str(path))
print(f"wrote {path} ({len(cells)} cells)")
