"""
Does the saturation budget move with network size?

Reads `results/scale_N*.json` and reports, per N, the silent floor, the unquantised ceiling, the
learned curve, and the budget at which the curve saturates. Saturation is defined operationally and
stated rather than eyeballed: B* is the smallest budget whose mean is within `TOL` of the best mean
achieved at that N. A knee read off a plot is not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
TOL = 0.005     # within half a point of the best budget at that N


def load(n: int):
    p = RESULTS / f"scale_N{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(rows, arm, bits):
    return [r["mean_ratio"] for r in rows if r["arm"] == arm and r["bits"] == bits]


def main() -> None:
    summary = {}
    print(f"{'N':>3} {'pool':>11} {'floor':>7} {'ceil':>7} "
          + " ".join(f"{'B='+str(b):>7}" for b in (1, 2, 3, 4, 6, 8)) + f" {'B*':>3} {'recov@B*':>9}")
    for n in (4, 8, 16, 32):
        rows = load(n)
        if not rows:
            print(f"{n:>3}  (no results yet)")
            continue
        floor = np.mean(cell(rows, "learned", 0)) if cell(rows, "learned", 0) else float("nan")
        ceil = np.mean(cell(rows, "continuous", None)) if cell(rows, "continuous", None) else float("nan")
        means = {}
        for b in (1, 2, 3, 4, 6, 8):
            c = cell(rows, "learned", b)
            if c:
                means[b] = float(np.mean(c))
        if not means:
            print(f"{n:>3}  (references only so far)")
            continue
        best = max(means.values())
        bstar = min(b for b, v in means.items() if v >= best - TOL)
        recov = 100 * (means[bstar] - floor) / (ceil - floor) if ceil > floor else float("nan")
        tr = rows[0]["train_size"]; te = rows[0]["test_size"]
        cells = " ".join(f"{means.get(b, float('nan')):>7.4f}" for b in (1, 2, 3, 4, 6, 8))
        print(f"{n:>3} {tr}/{te:<5} {floor:>7.4f} {ceil:>7.4f} {cells} {bstar:>3} {recov:>8.1f}%")
        summary[n] = {"floor": floor, "ceiling": ceil, "means": means, "b_star": bstar,
                      "recovered_pct": recov, "train_size": tr, "test_size": te,
                      "seeds": len(cell(rows, "learned", bstar)),
                      "complete": len(means) == 6}

    if summary:
        (RESULTS / "scale_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {RESULTS/'scale_summary.json'}")
        done = [n for n, v in summary.items() if v["complete"]]
        if len(done) > 1:
            bs = {n: summary[n]["b_star"] for n in done}
            print(f"B* by N: {bs}")
            print("B* is constant across the sizes measured" if len(set(bs.values())) == 1
                  else "B* MOVES with N -- the headline budget is size-dependent")


if __name__ == "__main__":
    main()
