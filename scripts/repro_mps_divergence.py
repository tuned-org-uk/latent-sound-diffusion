"""CPU/MPS parity harness for the graph decoder (issue #51).

Verifies that the previously reported MPS divergence does not regress:
checks forward parity, loss parity, per-op parity (GroupNorm, Conv1d,
gate Linear, U_q round trip, STFT), and short identical-init training
runs on both devices.

Usage:
    uv run python scripts/repro_mps_divergence.py                # all checks
    uv run python scripts/repro_mps_divergence.py --steps 100    # longer runs
    uv run python scripts/repro_mps_divergence.py --no-train     # parity only

Findings from 2026-08-15 (torch 2.13.0, post EnCodec-device fixes):
all parity checks agree to ~1e-6 and identical-init training matches
across devices — the historical divergence was collateral of the
EnCodec lazy-load device bug, not WaveReconstructionBlock numerics.
See issue #51 for the full analysis.
"""

from __future__ import annotations

import argparse
import copy

import torch

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.build_prior import build_arrow_prior
from ald_sc.graph_decoder import GraphDecoder
from ald_sc.losses import ALDSCLoss

F, Q, B, T_LAT = 128, 8, 4, 32
LR, NOISE = 1e-3, 0.1


def build() -> tuple[ArrowSpacePrior, GraphDecoder]:
    torch.manual_seed(1234)
    prior = build_arrow_prior(torch.randn(256, F), q=Q, k=8)
    torch.manual_seed(0)
    dec = GraphDecoder(
        latent_channels=128,
        out_channels=1,
        feature_dim=F,
        base_channels=64,
        prior=prior,
    )
    return prior, dec


def make_batch(prior: ArrowSpacePrior, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, 128, T_LAT, generator=g)
    x = torch.randn(B, 1, T_LAT * 320, generator=g) * 0.3
    A = torch.randn(B, F, generator=g)
    return z, x, A, prior.chart_energy_descriptor(A)


def check(cond: bool, msg: str, failures: list[str]) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        failures.append(msg)


def forward_parity(prior, dec, failures) -> None:
    print("\n== Forward parity (identical weights/inputs) ==")
    z, x, A, c_spec = make_batch(prior)
    dec.eval()
    with torch.no_grad():
        xh_c = dec(z, c_spec)
        xh_m = copy.deepcopy(dec).to("mps")(z.to("mps"), c_spec.to("mps"))
    d = (xh_c - xh_m.cpu()).abs().max().item()
    check(d < 1e-4, f"decoder output max|diff| = {d:.3e} (< 1e-4)", failures)
    U = prior.U_q.cpu().double()
    err = (U.T @ U - torch.eye(Q, dtype=torch.float64)).norm().item()
    check(err < 1e-4, f"||U_q^T U_q - I||_F = {err:.3e} (< 1e-4)", failures)


def loss_parity(prior, dec, failures) -> None:
    print("\n== Loss parity (same x, x_hat on both devices) ==")
    z, x, A, c_spec = make_batch(prior)
    dec.eval()
    with torch.no_grad():
        xh = dec(z, c_spec)
    lf_c = ALDSCLoss(prior=prior)
    lf_m = ALDSCLoss(prior=copy.deepcopy(prior).to("mps"))
    with torch.no_grad():
        lc = lf_c(x, xh, A, A.detach())
        lm = lf_m(x.to("mps"), xh.to("mps"), A.to("mps"), A.to("mps"))
    for k in ("rec", "stft", "chart", "smooth", "total"):
        d = abs(lc[k].item() - lm[k].item())
        check(d < 1e-4, f"{k:>6}: |cpu - mps| = {d:.3e} (< 1e-4)", failures)


def micro_ops(failures) -> None:
    print("\n== Micro-op parity (fwd + bwd) ==")
    g = torch.Generator().manual_seed(7)

    def cmp(name, mod, shape, post=lambda y: y):
        x = torch.randn(*shape, generator=g)
        outs = {}
        for dev in ("cpu", "mps"):
            m = copy.deepcopy(mod).to(dev)
            xi = x.clone().to(dev).requires_grad_(True)
            y = post(m(xi))
            y.sum().backward()
            outs[dev] = (y.detach().cpu(), xi.grad.detach().cpu())
        df = (outs["cpu"][0] - outs["mps"][0]).abs().max().item()
        dg = (outs["cpu"][1] - outs["mps"][1]).abs().max().item()
        check(
            df < 1e-4 and dg < 1e-3, f"{name}: fwd {df:.1e} / grad {dg:.1e}", failures
        )

    cmp("GroupNorm(8,64)", torch.nn.GroupNorm(8, 64), (B, 64, 128))
    cmp("Conv1d", torch.nn.Conv1d(64, 64, 3, padding=1), (B, 64, 128))
    cmp(
        "gate Linear+sigmoid", torch.nn.Linear(3 * Q, Q), (B, 3 * Q), post=torch.sigmoid
    )

    torch.manual_seed(1234)
    prior = build_arrow_prior(torch.randn(256, F), q=Q, k=8)
    a = torch.randn(B, F, generator=g)
    res = {}
    for dev in ("cpu", "mps"):
        U = prior.U_q.to(dev)
        ai = a.clone().to(dev).requires_grad_(True)
        r = (ai @ U) @ U.T
        r.sum().backward()
        res[dev] = (r.detach().cpu(), ai.grad.detach().cpu())
    dr = (res["cpu"][0] - res["mps"][0]).abs().max().item()
    dg = (res["cpu"][1] - res["mps"][1]).abs().max().item()
    check(
        dr < 1e-4 and dg < 1e-3,
        f"U_q round trip: fwd {dr:.1e} / grad {dg:.1e}",
        failures,
    )

    w = torch.randn(1, T_LAT * 320, generator=g)
    for n_fft in (512, 1024, 2048):
        win = torch.hann_window(n_fft)
        specs = {}
        for dev in ("cpu", "mps"):
            s = torch.stft(
                w.to(dev),
                n_fft,
                hop_length=n_fft // 4,
                return_complex=True,
                window=win.to(dev),
            )
            specs[dev] = s.abs().cpu()
        ds = (specs["cpu"] - specs["mps"]).abs().max().item()
        check(ds < 1e-3, f"stft mag n_fft={n_fft}: max|diff| = {ds:.1e}", failures)


def train(prior, dec, dev, steps, lr=LR, noise=NOISE) -> list[float]:
    model = copy.deepcopy(dec).to(dev).train()
    z, x, A, c_spec = (t.to(dev) for t in make_batch(prior))
    lf = ALDSCLoss(prior=copy.deepcopy(prior).to(dev), lambda_rec=1.0, lambda_stft=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(99)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        zi = z + noise * torch.randn_like(z) if noise > 0 else z
        loss = lf(x, model(zi, c_spec), A, A.detach())["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def training_parity(prior, dec, steps, failures) -> None:
    print(f"\n== Training parity ({steps} steps, lr={LR}, noise={NOISE}) ==")
    eps = train(prior, dec, "cpu", steps)
    mps = train(prior, dec, "mps", steps)
    check(
        eps[-1] < eps[0],
        f"cpu loss {eps[0]:.4f} -> {eps[-1]:.4f} (decreases)",
        failures,
    )
    check(
        mps[-1] < mps[0],
        f"mps loss {mps[0]:.4f} -> {mps[-1]:.4f} (decreases)",
        failures,
    )
    rel = abs(eps[-1] - mps[-1]) / max(eps[-1], mps[-1])
    check(rel < 0.2, f"final loss rel gap cpu vs mps = {rel:.1%} (< 20%)", failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=60, help="training steps")
    parser.add_argument("--no-train", action="store_true", help="skip training parity")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available — nothing to compare against.")

    prior, dec = build()
    failures: list[str] = []
    forward_parity(prior, dec, failures)
    loss_parity(prior, dec, failures)
    micro_ops(failures)
    if not args.no_train:
        training_parity(prior, dec, args.steps, failures)

    print(f"\n{len(failures)} failure(s)")
    if failures:
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
