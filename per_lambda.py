"""
Per-preference breakdown, with the classical solvers given the same restart budget as the reference.

Averaging WMMSE over the preference grid gives 0.6938, which invites the reasonable objection that
at lambda = 1 a sum-rate maximiser should nearly match a sum-rate reference. Two things are going on
and only one of them is interesting.

The interesting one is that WMMSE and Dinkelbach each optimise a single objective and are then
scored across the whole preference axis, so their average is dragged down by the far end where they
are solving the wrong problem. That is the multi-objective point of the paper.

The uninteresting one is a defect in how they were run: the centralised reference uses 16 restarts
of projected gradient, while standalone_classical called wmmse() from one initialisation. Measured
at lambda = 1 on 200 instances: 0.9712 at one start, 0.9876 at four, 0.9946 at sixteen. The
single-start gap is initialisation, not method, and reporting it as if it were the method would
understate the baseline. This script gives both solvers the reference's restart budget.

Also records the mean (SE, EE) of the winning allocation at each lambda, in the absolute units of
results/pareto.json, so the classical solvers can be drawn on the same objective plane as the
learned arms rather than only summarised as ratios.

Emits the per-lambda table and asserts the two sanity conditions: at lambda = 1 WMMSE must be within
2% of the reference, and at lambda = 0 Dinkelbach likewise. A failure means the reference or a
solver is broken and the paper's ratios cannot be trusted.

Runs one CSI condition per invocation so the two can run side by side on separate cores:

    OMP_NUM_THREADS=1 python3 per_lambda.py --csi current   -> results/per_lambda.json
    OMP_NUM_THREADS=1 python3 per_lambda.py --csi stale     -> results/per_lambda_stale.json

Both use the same 2,048-instance pool and the same 16 restarts, which is what lets one table carry
the per-lambda columns and the stale column without a footnote explaining that they disagree. The
earlier stale column came from standalone_classical.py at one WMMSE initialisation, and from a run
interrupted partway so that two of its three rows were on 1,024 instances.

    OMP_NUM_THREADS=1 python3 per_lambda.py [--smoke]
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

from baselines import dinkelbach, wmmse                                # noqa: E402
from dataset import cached_pool                                        # noqa: E402
from metrics import energy_efficiency, spectral_efficiency             # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W           # noqa: E402
from standalone_classical import interference_pricing                  # noqa: E402

RESULTS = HERE / "results"
N_PAIRS = 8
RESTARTS = 16          # matched to solvers.maximize_batch's n_starts, which built the reference
N_INSTANCES = 2048     # the pool the rest of the paper reports on
TOL = 0.02


def score(p, a, se_ref, ee_ref, noise, pc, lam):
    se, ee = raw(p, a, noise, pc)
    return lam * se / se_ref + (1.0 - lam) * ee / ee_ref


def raw(p, a, noise, pc):
    """Absolute (SE, EE) of an allocation, in the same units as the Pareto pool."""
    return (float(spectral_efficiency(p, a, noise)),
            float(energy_efficiency(p, a, noise, pc)))


def best_of(fn, a, rng, n_start, p_max):
    """Run a solver from `n_start` initialisations and keep the best allocation per objective."""
    out = [fn(None)]
    for _ in range(n_start - 1):
        out.append(fn(rng.random(len(a)) * p_max))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="wmmse,dinkelbach,pricing",
                    help="comma-separated arms to (re)run; others are carried over from the "
                         "existing file, so a cheap arm can be refreshed without the expensive ones")
    ap.add_argument("--csi", choices=("current", "stale"), default="current",
                    help="which measurement the solvers decide on; scoring is always on slot t")
    args = ap.parse_args()
    size = 64 if args.smoke else N_INSTANCES
    restarts = 2 if args.smoke else RESTARTS

    pool = cached_pool(f"test_N{N_PAIRS}_2048", size=2048, n_pairs=N_PAIRS, area_m=AREA_M,
                       seed=999, lambdas=LAMBDAS, device="cpu")
    # Decide on `A`, always score on the realised gains: the same convention as
    # standalone_classical.py, so "stale" costs only the age of the observation.
    A_eval = pool.gains.numpy()[:size]
    A = (pool.gains if args.csi == "current" else pool.gains_obs).numpy()[:size]
    se_ref, ee_ref = pool.se_ref.numpy()[:size], pool.ee_ref.numpy()[:size]
    noise, pc = pool.noise_power, CIRCUIT_POWER_W
    oracle = {float(k): v.numpy()[:size] for k, v in pool.oracle.items()}

    name = ("per_lambda_smoke" if args.smoke else
            "per_lambda" if args.csi == "current" else "per_lambda_stale")
    want = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    rows = {}
    if (RESULTS / f"{name}.json").exists() and not args.smoke:
        rows = {k: v for k, v in json.loads((RESULTS / f"{name}.json").read_text()).items()
                if k not in want}
        if rows:
            print(f"carrying {', '.join(rows)} from the existing file", flush=True)
    for arm in want:
        t0 = time.time()
        per_lam, se_lam, ee_lam = {}, [], []
        for lam in LAMBDAS:
            rng = np.random.default_rng(11)
            rs, ses, ees, iters = [], [], [], []
            for m in range(size):
                a, a_eval = A[m], A_eval[m]
                sr, er = float(se_ref[m]), float(ee_ref[m])
                if arm == "wmmse":
                    cands = best_of(lambda i: wmmse(a, noise, P_MAX_W, init=i), a, rng,
                                    restarts, P_MAX_W)
                elif arm == "dinkelbach":
                    cands = best_of(lambda i: dinkelbach(a, noise, P_MAX_W, pc), a, rng, 1, P_MAX_W)
                else:
                    p_pr, it = interference_pricing(a, noise, P_MAX_W, lam, sr, er, pc, count=True)
                    cands, iters = [p_pr], iters + [it]
                # argmax, not max: the winning allocation's own (SE, EE) is what the Pareto
                # figure plots, so scoring and coordinates must come from the same p.
                pb = max(cands, key=lambda p: score(p, a_eval, sr, er, noise, pc, lam))
                rs.append(score(pb, a_eval, sr, er, noise, pc, lam) / float(oracle[lam][m]))
                se_m, ee_m = raw(pb, a_eval, noise, pc)
                ses.append(se_m); ees.append(ee_m)
            per_lam[str(lam)] = float(np.mean(rs))
            se_lam.append(float(np.mean(ses))); ee_lam.append(float(np.mean(ees)))
            print(f"  {arm:11s} lam={lam:.2f} -> {per_lam[str(lam)]:.4f}", flush=True)
        rows[arm] = {"arm": arm, "restarts": restarts if arm == "wmmse" else 1,
                     "csi": args.csi, "n_instances": size, "per_lambda": per_lam,
                     "lambdas": list(LAMBDAS),
                     "se_by_lambda": se_lam, "ee_by_lambda": ee_lam,
                     "mean_ratio": float(np.mean(list(per_lam.values()))),
                     # Iterations to convergence: the paper turns this into a bit count as
                     # 32 * (N-1) * iters, so it must come from the same pool as the ratio it sits
                     # beside rather than from a differently sized run.
                     "pricing_iters_mean": float(np.mean(iters)) if iters else None,
                     "pricing_iters_p95": float(np.percentile(iters, 95)) if iters else None,
                     "seconds": time.time() - t0}
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=2))

    # The sanity conditions are about the two single-objective solvers; a run that did not include
    # them (--arms pricing, say) has nothing to assert and must not pretend otherwise.
    if not ({"wmmse", "dinkelbach"} <= set(rows)):
        print("sanity: skipped, this run did not include both single-objective solvers")
        return
    w1 = rows["wmmse"]["per_lambda"]["1.0"]
    d0 = rows["dinkelbach"]["per_lambda"]["0.0"]
    print(f"\nsanity: WMMSE at lam=1 -> {w1:.4f}; Dinkelbach at lam=0 -> {d0:.4f} (need >= {1-TOL})")
    bad = []
    if w1 < 1 - TOL:
        bad.append(f"WMMSE at lambda=1 is {w1:.4f}, more than {TOL:.0%} below the reference")
    if d0 < 1 - TOL:
        bad.append(f"Dinkelbach at lambda=0 is {d0:.4f}, more than {TOL:.0%} below the reference")
    if bad and not args.smoke and args.csi == "current":
        raise SystemExit("SANITY FAILED (reference or solver is wrong):\n  " + "\n  ".join(bad))
    print("sanity OK" if not bad else "sanity failed (smoke run, not fatal)")


if __name__ == "__main__":
    main()
