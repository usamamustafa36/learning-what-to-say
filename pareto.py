"""
Pareto fronts and hypervolume: the multi-objective claim, made checkable.

A paper that conditions a policy on a preference weight owes the reader a front, not an assertion.
Sweeping lambda produces one (SE, EE) point per operating point; the non-dominated subset of those
points is the achieved front, and its dominated area is the standard scalar summary -- it rewards
both proximity to the ideal point and spread along the front, so it cannot be gamed by a method that
excels at one lambda and collapses elsewhere.

Two normalisations are needed before the number means anything.

*Scale.* Raw SE is ~20 b/s/Hz and raw EE is ~200 b/J/Hz at this operating point, so an unnormalised
dominated area is essentially the EE axis with SE as rounding error. Both axes are divided by the
centralised oracle's best value on that axis, which puts the ideal point at (1, 1) and makes the
hypervolume the fraction of the ideal box a method dominates.

*Reference point.* The origin. With normalised axes this is the honest choice: it charges a method
for everything it fails to achieve rather than for its distance from an arbitrary nadir that could
be tuned to flatter one arm.

No new training is required. `train.evaluate` already records SE and EE at every lambda for every
arm in the budget sweep, so the fronts are recovered from the stored results; only the oracle's own
front is computed here, because it is the normaliser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from metrics import hypervolume_2d, pareto_front
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, N_PAIRS, P_MAX_W

RESULTS = Path(__file__).parent / "results"


def oracle_front(n_pairs: int = N_PAIRS, size: int = 1024, seed: int = 999,
                 device: str | None = None) -> dict:
    """The centralised genie's own (SE, EE) trace across lambda -- the ideal point and the bound."""
    from dataset import build_pool
    from env import ee_torch, se_torch
    from solvers import oracle_batch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = build_pool(size=size, n_pairs=n_pairs, area_m=AREA_M, seed=seed, device=dev,
                      p_max=P_MAX_W, circuit_power_w=CIRCUIT_POWER_W)
    se_pts, ee_pts = [], []
    for lam_val in LAMBDAS:
        lam = torch.full((len(pool),), float(lam_val), device=pool.gains.device)
        with torch.no_grad():
            p = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam, pool.se_ref, pool.ee_ref,
                             CIRCUIT_POWER_W, n_starts=16, n_steps=800)
        se_pts.append(float(se_torch(p, pool.gains, pool.noise_power).mean()))
        ee_pts.append(float(ee_torch(p, pool.gains, pool.noise_power, CIRCUIT_POWER_W).mean()))
    return {"lambdas": list(LAMBDAS), "se": se_pts, "ee": ee_pts,
            "se_max": max(se_pts), "ee_max": max(ee_pts)}


def normalised_hypervolume(se: list[float], ee: list[float], se_max: float, ee_max: float) -> float:
    """Fraction of the ideal box [0, SE*] x [0, EE*] dominated by this arm's front."""
    pts = np.stack([np.asarray(se) / max(se_max, 1e-12),
                    np.asarray(ee) / max(ee_max, 1e-12)], axis=1)
    return hypervolume_2d(pts, reference=(0.0, 0.0))


def analyse(tag: str = "pareto", sweep_tag: str = "bitsweep_fixed",
            size: int = 1024) -> dict:
    rows = json.loads((RESULTS / f"{sweep_tag}.json").read_text())
    orc = oracle_front(size=size)
    se_max, ee_max = orc["se_max"], orc["ee_max"]

    out = {"oracle": orc,
           "oracle_hypervolume": normalised_hypervolume(orc["se"], orc["ee"], se_max, ee_max),
           "arms": []}

    keys = sorted({(r["arm"], r["bits"]) for r in rows},
                  key=lambda k: (k[0], -1 if k[1] is None else k[1]))
    for arm, bits in keys:
        sel = [r for r in rows if r["arm"] == arm and r["bits"] == bits]
        hvs, spreads = [], []
        for r in sel:
            hv = normalised_hypervolume(r["se_by_lambda"], r["ee_by_lambda"], se_max, ee_max)
            hvs.append(hv)
            front = pareto_front(np.stack([r["se_by_lambda"], r["ee_by_lambda"]], axis=1))
            spreads.append(len(front))
        out["arms"].append({
            "arm": arm, "bits": bits,
            "hypervolume": float(np.mean(hvs)), "hypervolume_std": float(np.std(hvs)),
            "fraction_of_oracle": float(np.mean(hvs)) / max(out["oracle_hypervolume"], 1e-12),
            "front_points": float(np.mean(spreads)),
            "se_by_lambda": sel[0]["se_by_lambda"], "ee_by_lambda": sel[0]["ee_by_lambda"],
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    summarise(out)
    return out


def summarise(out: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'arm':>12} {'B':>5} {'hypervolume':>13} {'of oracle':>11} {'non-dominated':>15}")
    print("-" * 78)
    print(f"{'oracle':>12} {'--':>5} {out['oracle_hypervolume']:>13.4f} {1.0:>11.3f} "
          f"{len(LAMBDAS):>15}")
    for a in out["arms"]:
        b = "cont." if a["bits"] is None else str(a["bits"])
        print(f"{a['arm']:>12} {b:>5} {a['hypervolume']:>13.4f} {a['fraction_of_oracle']:>11.3f} "
              f"{a['front_points']:>15.1f}")
    print("=" * 78)
    print("non-dominated = how many of the 5 preference points survive domination. A policy that")
    print("ignores lambda collapses its front to a single point and reports 1.")


if __name__ == "__main__":
    # 1. The hypervolume must behave: a dominating front scores higher, a collapsed one lower.
    a = [(1.0, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 1.0)]      # a spread front
    b = [(0.6, 0.6)] * 4                                       # a policy that ignores the preference
    hv_a = hypervolume_2d(np.array(a)), hypervolume_2d(np.array(b))
    print(f"spread front HV {hv_a[0]:.4f}  vs collapsed front HV {hv_a[1]:.4f}")
    assert hv_a[0] > hv_a[1], "hypervolume does not reward spread"
    print(f"non-dominated points: spread {len(pareto_front(np.array(a)))}, "
          f"collapsed {len(pareto_front(np.array(b)))}")

    # 2. Full analysis over the stored sweep.
    analyse()
