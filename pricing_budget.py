"""
Distributed interference pricing under a hard total signalling budget.

The `priced` arm of the bit-budget table is one-shot: a single B-bit price per edge, evaluated at
p = P_max, carried over the learned architecture. That is the right control for isolating message
*content*, but it is not what a pricing scheme does. A real one iterates, and the reviewer's
objection is fair: comparing a learned protocol against a deliberately one-shot classical arm is not
a comparison against distributed pricing.

So here pricing is run as pricing -- iterated to a fixed round count K -- but charged for every bit
it sends. With b bits per price per edge and N-1 edges, K rounds cost

    T = K * b * (N-1)   bits per agent per slot,

so a total budget T buys K = T / (b (N-1)) rounds. Sweeping T from the learned protocol's 42 bits up
to the 22,601 bits converged pricing actually needs traces the classical arm's own
communication-performance curve, and that curve is what the learned arm should be read against.

Each cell also runs unquantised at the same K, which separates the two ways a budget can fail: too
few rounds to converge, versus a price too coarse to act on.

Fairness note for the paper: the learned arm spends its whole budget in one round, while this arm
may spread the same budget over as many rounds as it likes. Any learned advantage at equal total
bits is therefore a lower bound.

    OMP_NUM_THREADS=1 python3 pricing_budget.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import cached_pool                                       # noqa: E402
from metrics import energy_efficiency, spectral_efficiency            # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W          # noqa: E402

RESULTS = HERE / "results"
SIZE = 512          # enough for a mean over 5 preference points; the axis here is budget, not noise
N_PAIRS = 8


def fit_levels(samples: np.ndarray, bits: int) -> np.ndarray:
    """Quantiser levels in the log domain, from sample quantiles (Lloyd-Max-like, no iteration)."""
    q = np.linspace(0.0, 1.0, 2 ** bits + 1)[1:-1]
    edges = np.quantile(np.log(np.maximum(samples, 1e-300)), q)
    lo, hi = np.log(np.maximum(samples, 1e-300)).min(), np.log(np.maximum(samples, 1e-300)).max()
    bounds = np.concatenate([[lo], edges, [hi]])
    return 0.5 * (bounds[:-1] + bounds[1:])          # level centroids, in log space


def quantise(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Snap to the nearest level, in the log domain, and return to linear."""
    lx = np.log(np.maximum(x, 1e-300))
    idx = np.abs(lx[..., None] - levels[None, None, :]).argmin(axis=-1)
    return np.exp(levels[idx])


def priced_rounds(a, noise, p_max, lam, se_ref, ee_ref, pc, n_iter, levels=None, damping=0.5):
    """Interference pricing for exactly `n_iter` rounds; quantise the price if levels are given."""
    n = len(a)
    diag = np.diag(a)
    p = np.full(n, p_max * 0.5)
    ln2 = np.log(2.0)
    for _ in range(max(int(n_iter), 1)):
        denom = float(p.sum() + n * pc)
        se = float(spectral_efficiency(p, a, noise))
        alpha = lam / se_ref + (1.0 - lam) / (ee_ref * denom)
        beta = (1.0 - lam) * se / (ee_ref * denom * denom)
        interf = a @ p - diag * p + noise
        sig = diag * p
        price = alpha * sig / (ln2 * interf * (interf + sig))
        # The wire carries pi_i * a_ij -- the marginal harm j does to i -- so that is what is
        # quantised, not the price alone.
        harm = a * price[:, None]
        if levels is not None:
            harm = quantise(harm, levels)
        cost = harm.sum(axis=0) - diag * price
        total = np.maximum(cost + beta, 1e-30)
        target = np.clip(alpha / (total * ln2) - interf / np.maximum(diag, 1e-30), 0.0, p_max)
        p = (1.0 - damping) * p + damping * target
    return p


def main() -> None:
    pool = cached_pool(f"test_N{N_PAIRS}_2048", size=2048, n_pairs=N_PAIRS, area_m=AREA_M,
                       seed=999, lambdas=LAMBDAS, device="cpu")
    A = pool.gains.numpy()[:SIZE]
    A_obs = pool.gains_obs.numpy()[:SIZE]
    se_ref, ee_ref = pool.se_ref.numpy()[:SIZE], pool.ee_ref.numpy()[:SIZE]
    noise, pc = pool.noise_power, CIRCUIT_POWER_W
    oracle = {float(k): v.numpy()[:SIZE] for k, v in pool.oracle.items()}
    E = N_PAIRS - 1

    # Fit the quantiser on prices actually seen at the equal-power reference, the only operating
    # point an agent can compute with zero signalling.
    print("fitting price quantisers", flush=True)
    sample = []
    for m in range(128):
        a = A_obs[m]
        d = np.diag(a)
        p0 = np.full(N_PAIRS, P_MAX_W)
        interf = a @ p0 - d * p0 + noise
        sig = d * p0
        pr = sig / (np.log(2.0) * interf * (interf + sig))
        sample.append((a * pr[:, None]).ravel())
    sample = np.concatenate(sample)
    levels = {b: fit_levels(sample, b) for b in (1, 2, 3, 4, 6, 8)}

    rows = []
    for bits in (1, 2, 3, 4, 6, 8):
        for total in (42, 84, 168, 336, 672, 1344, 2688, 5376, 11424, 22601):
            K = total / (bits * E)
            if K < 1:
                continue
            t0 = time.time()
            per_lam_q, per_lam_u = {}, {}
            for lam in LAMBDAS:
                rq, ru = [], []
                for m in range(SIZE):
                    for tag, lv, acc in (("q", levels[bits], rq), ("u", None, ru)):
                        p = priced_rounds(A_obs[m], noise, P_MAX_W, lam, float(se_ref[m]),
                                          float(ee_ref[m]), pc, K, levels=lv)
                        se = float(spectral_efficiency(p, A[m], noise))
                        ee = float(energy_efficiency(p, A[m], noise, pc))
                        r = lam * se / float(se_ref[m]) + (1 - lam) * ee / float(ee_ref[m])
                        acc.append(r / float(oracle[lam][m]))
                per_lam_q[str(lam)] = float(np.mean(rq))
                per_lam_u[str(lam)] = float(np.mean(ru))
            rows.append({
                "arm": "priced_rounds", "price_bits": bits, "total_bits": total,
                "rounds": K, "n_pairs": N_PAIRS, "n_instances": SIZE,
                "mean_ratio": float(np.mean(list(per_lam_q.values()))),
                "mean_ratio_unquantised": float(np.mean(list(per_lam_u.values()))),
                "per_lambda": per_lam_q, "seconds": time.time() - t0,
            })
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / "pricing_budget.json").write_text(json.dumps(rows, indent=2))
            print(f"  b={bits} T={total:6d} K={K:7.1f}  quantised {rows[-1]['mean_ratio']:.4f}"
                  f"  unquantised {rows[-1]['mean_ratio_unquantised']:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {RESULTS / 'pricing_budget.json'}")


if __name__ == "__main__":
    main()
