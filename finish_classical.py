"""
Finish the standalone-classical table without redoing the arms already measured.

Two defects made the first run far slower than it needed to be, and both are fixed here:
WMMSE and Dinkelbach are single-objective, so their allocation does not depend on lambda and is
solved once per instance rather than five times; and the pool is 1024 instances rather than 2048,
which is ample for a mean over 5 preference points and halves the cost again.

Completed arms are read from standalone_classical_partial.json and passed through untouched, so the
current-CSI rows keep the exact numbers already reported.

    python3 finish_classical.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from baselines import dinkelbach, wmmse                              # noqa: E402
from dataset import cached_pool                                       # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W          # noqa: E402
from standalone_classical import interference_pricing, scalarised     # noqa: E402

OUT = HERE / "results" / "standalone_classical.json"
PARTIAL = HERE / "results" / "standalone_classical_partial.json"
SIZE = 1024


def main() -> None:
    rows = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else []
    have = {(r["arm"], r["csi"]) for r in rows}
    print(f"keeping {sorted(have)}", flush=True)

    pool = cached_pool("test_N8_2048", size=2048, n_pairs=8, area_m=AREA_M, seed=999,
                       lambdas=LAMBDAS, device="cpu")
    A_now = pool.gains.numpy()[:SIZE]
    A_obs = pool.gains_obs.numpy()[:SIZE]
    se_ref, ee_ref = pool.se_ref.numpy()[:SIZE], pool.ee_ref.numpy()[:SIZE]
    noise, pc = pool.noise_power, CIRCUIT_POWER_W
    oracle = {float(k): v.numpy()[:SIZE] for k, v in pool.oracle.items()}

    todo = [(a, c) for c in ("current", "stale") for a in ("wmmse", "dinkelbach", "pricing")
            if (a, c) not in have]
    print(f"to run: {todo} on {SIZE} instances", flush=True)

    for arm, csi in todo:
        A = A_now if csi == "current" else A_obs
        t0 = time.time()
        fixed = None
        if arm == "wmmse":
            fixed = np.stack([wmmse(A[m], noise, P_MAX_W) for m in range(SIZE)])
        elif arm == "dinkelbach":
            fixed = np.stack([dinkelbach(A[m], noise, P_MAX_W, pc) for m in range(SIZE)])
        if fixed is not None:
            print(f"  {arm} solved {SIZE} instances once in {time.time()-t0:.0f}s", flush=True)

        per_lam, iters, ses, ees = {}, [], [], []
        for lam in LAMBDAS:
            ratios = []
            for m in range(SIZE):
                if fixed is not None:
                    p = fixed[m]
                else:
                    p, it = interference_pricing(A[m], noise, P_MAX_W, lam, float(se_ref[m]),
                                                 float(ee_ref[m]), pc, count=True)
                    iters.append(it)
                r, se, ee = scalarised(p, A_now[m], noise, float(se_ref[m]), float(ee_ref[m]),
                                       lam, pc)
                ratios.append(r / float(oracle[lam][m]))
                ses.append(se)
                ees.append(ee)
            per_lam[str(lam)] = float(np.mean(ratios))
            print(f"  {arm:11s} {csi:7s} lam={lam:.2f} -> {np.mean(ratios):.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        rows.append({
            "arm": arm, "csi": csi, "n_pairs": 8, "n_instances": SIZE,
            "per_lambda": per_lam,
            "mean_ratio": float(np.mean(list(per_lam.values()))),
            "abs_se": float(np.mean(ses)), "abs_ee": float(np.mean(ees)),
            "pricing_iters_mean": float(np.mean(iters)) if iters else None,
            "pricing_iters_p95": float(np.percentile(iters, 95)) if iters else None,
            "seconds": time.time() - t0,
        })
        OUT.write_text(json.dumps(rows, indent=2))
        print(f"  -> {arm}/{csi} mean {rows[-1]['mean_ratio']:.4f}\n", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
