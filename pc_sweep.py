"""
Does the configured regime actually admit a trade-off, and at what circuit power does it stop?

EE = SE / (sum_i p_i + N*Pc). When N*Pc dominates the controllable term the denominator is nearly
constant, EE becomes nearly proportional to SE, the two objectives stop conflicting, and every
"multi-objective" claim in the paper describes a front that is an artefact of the operating point.
Remark 1 states this with numbers; those numbers were typed by hand and one of them had drifted from
what the code computes. This produces them instead.

For each circuit power we ask the centralised oracle -- not any policy, since a policy that ignores
lambda would confound the measurement -- how far its own allocation moves in SE and in EE between
lambda = 0 and lambda = 1, on one fixed pool.

    OMP_NUM_THREADS=1 python3 pc_sweep.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import build_pool                                         # noqa: E402
from env import ee_torch, se_torch                                     # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, P_MAX_W                    # noqa: E402
from solvers import oracle_batch                                       # noqa: E402

RESULTS = HERE / "results"
N_PAIRS = 8
SEED = 7
PC_W = (CIRCUIT_POWER_W, 0.100)     # the adopted value, and the one that destroys the trade-off


def movement(pool, pc: float, starts: int, steps: int) -> dict:
    vals = {}
    for lam_val in (0.0, 1.0):
        lam = torch.full((len(pool),), float(lam_val), device=pool.gains.device)
        with torch.no_grad():
            p = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam, pool.se_ref, pool.ee_ref,
                             pc, n_starts=starts, n_steps=steps)
        vals[lam_val] = (float(se_torch(p, pool.gains, pool.noise_power).mean()),
                         float(ee_torch(p, pool.gains, pool.noise_power, pc).mean()))
    (se0, ee0), (se1, ee1) = vals[0.0], vals[1.0]
    return {"circuit_power_w": pc, "circuit_power_mw": 1e3 * pc,
            "se_lam0": se0, "se_lam1": se1, "ee_lam0": ee0, "ee_lam1": ee1,
            "se_move_pct": 100.0 * abs(se1 - se0) / max(se0, 1e-9),
            "ee_move_pct": 100.0 * abs(ee1 - ee0) / max(ee0, 1e-9)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    size, starts, steps = (48, 4, 200) if args.smoke else (512, 16, 800)

    pool = build_pool(size=size, n_pairs=N_PAIRS, area_m=AREA_M, seed=SEED, device="cpu",
                      p_max=P_MAX_W, circuit_power_w=CIRCUIT_POWER_W)
    rows = []
    for pc in PC_W:
        t0 = time.time()
        r = movement(pool, pc, starts, steps)
        r.update({"n_instances": size, "n_starts": starts, "n_steps": steps, "seed": SEED,
                  "adopted": pc == CIRCUIT_POWER_W, "seconds": time.time() - t0})
        rows.append(r)
        print(f"  Pc={r['circuit_power_mw']:5.1f} mW -> SE moves {r['se_move_pct']:5.1f}%, "
              f"EE moves {r['ee_move_pct']:5.1f}%  ({r['seconds']:.0f}s)", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ("pc_sweep_smoke.json" if args.smoke else "pc_sweep.json")
    out.write_text(json.dumps(rows, indent=2))

    adopted = next(r for r in rows if r["adopted"])
    if adopted["se_move_pct"] <= 10 or adopted["ee_move_pct"] <= 10:
        raise SystemExit("SANITY FAILED: at the adopted circuit power the objectives barely "
                         "conflict, so every multi-objective claim in the paper is an artefact")
    print(f"wrote {out}; sanity OK at the adopted Pc")


if __name__ == "__main__":
    main()
