"""
Limited-feedback variants of iterated interference pricing, at matched budget.

The existing budgeted-pricing arm scores below the silent floor, and the reason is a specific
defect rather than a property of pricing: `pricing_budget.fit_levels` places quantiser levels once,
offline, from prices sampled at the equal-power reference p = P_max, and never refits them. The
loop starts at 0.5 P_max and walks the price distribution off that operating point on the first
iteration, so every later round quantises against a stale codebook. More rounds cannot help, and
the measured plateau (b=1 near 0.56 whether K is 6 or 192) is exactly what that predicts.

A reviewer would call that a misconfigured baseline, correctly. The fix is not exotic: coding a
*moving* scalar over a rate-limited link is a solved problem, and the standard answers are
differential coding, sign-based increments with step adaptation, and a range-adaptive quantiser.
All four variants below carry the identical budget, K rounds x b bits x (N-1) edges, so the
comparison against the learned arm stays honest.

  absolute      the existing arm, kept as the control: memoryless, offline-fitted levels.
  differential  quantise the CHANGE in log-price since the previous round; receiver integrates.
                Round 1 sends the absolute value so the receiver has an anchor.
  sign          1-bit incremental: send sign(delta log-price); receiver steps by a state variable
                that halves on a sign reversal (classical delta modulation with step adaptation).
  adaptive      uniform quantiser in the log domain whose range tracks a running min/max, Jayant
                style, so the 15-decade dynamic range is not spent on values that never occur.
  dithered      absolute levels plus subtractive dither, as a control: it tests whether the
                deficit is quantisation *noise* (dither would help) or quantiser *mismatch*
                (dither would not).

    OMP_NUM_THREADS=1 python3 pricing_variants.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import cached_pool                                       # noqa: E402
from metrics import energy_efficiency, spectral_efficiency            # noqa: E402
from pricing_budget import fit_levels, quantise                       # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W          # noqa: E402

RESULTS = HERE / "results"
N_PAIRS = 8
VARIANTS = ("absolute", "differential", "sign", "adaptive", "dithered")
BITS = (1, 2, 3, 4, 6, 8, 10, 12)   # b>8 included: differential keeps improving there
BUDGETS = (42, 84, 168, 336, 672, 1344, 2688, 5376, 10752, 22601)
K_CAP = 512


def _uniform_levels(lo, hi, bits):
    n = 2 ** bits
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def priced_rounds(a, noise, p_max, lam, se_ref, ee_ref, pc, n_iter,
                  variant="absolute", bits=4, levels=None, rng=None, damping=0.5):
    """Iterated pricing where the wire carries `bits` per edge per round, coded per `variant`."""
    n = len(a)
    diag = np.diag(a)
    p = np.full(n, p_max * 0.5)
    ln2 = np.log(2.0)
    K = max(int(n_iter), 1)

    # Receiver-side state. `rx` is what the receiver believes log-harm to be.
    rx = None
    step = np.full((n, n), 1.0)          # sign-variant adaptive step, in log units
    last_sign = np.zeros((n, n))
    lo = hi = None                       # adaptive-range tracker

    for k in range(K):
        denom = float(p.sum() + n * pc)
        se = float(spectral_efficiency(p, a, noise))
        alpha = lam / se_ref + (1.0 - lam) / (ee_ref * denom)
        beta = (1.0 - lam) * se / (ee_ref * denom * denom)
        interf = a @ p - diag * p + noise
        sig = diag * p
        price = alpha * sig / (ln2 * interf * (interf + sig))
        harm = a * price[:, None]
        lh = np.log(np.maximum(harm, 1e-300))

        if variant == "absolute":
            rx_log = np.log(np.maximum(quantise(harm, levels), 1e-300))

        elif variant == "dithered":
            d = (rng.random(lh.shape) - 0.5) * (levels[1] - levels[0] if len(levels) > 1 else 0.0)
            rx_log = np.log(np.maximum(quantise(np.exp(lh + d), levels), 1e-300)) - d

        elif variant == "differential":
            if rx is None:
                rx_log = np.log(np.maximum(quantise(harm, levels), 1e-300))
            else:
                delta = lh - rx
                # A delta quantiser only needs to span the per-round change, not 15 decades.
                dl = _uniform_levels(-3.0, 3.0, bits)
                idx = np.abs(delta[..., None] - dl[None, None, :]).argmin(axis=-1)
                rx_log = rx + dl[idx]

        elif variant == "sign":
            if rx is None:
                rx_log = np.log(np.maximum(quantise(harm, levels), 1e-300))
                step = np.full((n, n), 1.0)
            else:
                s = np.sign(lh - rx)
                # Jayant-style: halve the step on a reversal, grow it slightly on agreement.
                rev = (s * last_sign) < 0
                step = np.where(rev, step * 0.5, np.minimum(step * 1.2, 4.0))
                last_sign = s
                rx_log = rx + s * step

        elif variant == "adaptive":
            lo = lh.min() if lo is None else 0.9 * lo + 0.1 * lh.min()
            hi = lh.max() if hi is None else 0.9 * hi + 0.1 * lh.max()
            al = _uniform_levels(lo, hi, bits)
            idx = np.abs(lh[..., None] - al[None, None, :]).argmin(axis=-1)
            rx_log = al[idx]
        else:
            raise ValueError(variant)

        rx = rx_log
        recv = np.exp(np.clip(rx_log, -700, 700))
        cost = recv.sum(axis=0) - diag * price
        total = np.maximum(cost + beta, 1e-30)
        target = np.clip(alpha / (total * ln2) - interf / np.maximum(diag, 1e-30), 0.0, p_max)
        p = (1.0 - damping) * p + damping * target
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="64 instances, one budget, proves it runs")
    args = ap.parse_args()
    size = 64 if args.smoke else 512
    budgets = (42, 336) if args.smoke else BUDGETS
    bits_list = (1, 4) if args.smoke else BITS

    pool = cached_pool(f"test_N{N_PAIRS}_2048", size=2048, n_pairs=N_PAIRS, area_m=AREA_M,
                       seed=999, lambdas=LAMBDAS, device="cpu")
    A = pool.gains.numpy()[:size]
    A_obs = pool.gains_obs.numpy()[:size]          # stale CSI, matching Table III
    se_ref, ee_ref = pool.se_ref.numpy()[:size], pool.ee_ref.numpy()[:size]
    noise, pc = pool.noise_power, CIRCUIT_POWER_W
    oracle = {float(k): v.numpy()[:size] for k, v in pool.oracle.items()}
    E = N_PAIRS - 1

    sample = []
    for m in range(min(128, size)):
        a = A_obs[m]; d = np.diag(a); p0 = np.full(N_PAIRS, P_MAX_W)
        interf = a @ p0 - d * p0 + noise; sig = d * p0
        pr = sig / (np.log(2.0) * interf * (interf + sig))
        sample.append((a * pr[:, None]).ravel())
    sample = np.concatenate(sample)
    levels = {b: fit_levels(sample, b) for b in bits_list}

    out = RESULTS / ("pricing_variants_smoke.json" if args.smoke
                     else "pricing_budgeted_variants.json")
    # Resume. The JSON is rewritten after every cell so a kill is survivable, but until now a
    # restart recomputed the whole grid, which for a four-hour sweep makes that write pointless.
    # Cells are keyed by (variant, price width, budget) and are independent of each other, so
    # anything already on disk is kept verbatim: no cell is ever recomputed by a different code
    # path than the one that produced its neighbours in the same file.
    rows = []
    if out.exists() and not args.smoke:
        rows = json.loads(out.read_text())
        print(f"resuming from {out.name}: {len(rows)} cells already done", flush=True)
    have = {(r["variant"], r["price_bits"], r["total_bits"]) for r in rows}
    for variant in VARIANTS:
        for bits in bits_list:
            for total in budgets:
                K = total / (bits * E)
                if K < 1 or K > K_CAP:
                    continue
                if (variant, bits, total) in have:
                    continue
                t0 = time.time()
                per_lam = {}
                for lam in LAMBDAS:
                    rng = np.random.default_rng(7)
                    rs = []
                    for m in range(size):
                        p = priced_rounds(A_obs[m], noise, P_MAX_W, lam, float(se_ref[m]),
                                          float(ee_ref[m]), pc, K, variant=variant, bits=bits,
                                          levels=levels[bits], rng=rng)
                        se = float(spectral_efficiency(p, A[m], noise))
                        ee = float(energy_efficiency(p, A[m], noise, pc))
                        r = lam * se / float(se_ref[m]) + (1 - lam) * ee / float(ee_ref[m])
                        rs.append(r / float(oracle[lam][m]))
                    per_lam[str(lam)] = float(np.mean(rs))
                rows.append({"variant": variant, "price_bits": bits, "total_bits": total,
                             "rounds": K, "n_pairs": N_PAIRS, "n_instances": size,
                             "csi": "stale", "per_lambda": per_lam,
                             "mean_ratio": float(np.mean(list(per_lam.values()))),
                             "seconds": time.time() - t0})
                RESULTS.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rows, indent=2))
                print(f"  {variant:13s} b={bits} T={total:6d} K={K:6.1f} -> "
                      f"{rows[-1]['mean_ratio']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
