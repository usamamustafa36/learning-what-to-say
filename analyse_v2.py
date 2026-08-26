"""
Merge and summarise the rebuilt bit-budget sweep.

Three things this reports that the original could not.

1. **A confidence interval that means something.** The original quoted a paired t-test on three
   seeds. Three is not a sample. Here the per-instance ratios are averaged over seeds to give one
   paired observation per test instance, and the interval is a BCa-free percentile bootstrap over
   those 2048 instances. Seed spread is still reported, as a separate and much smaller number, so
   the two sources of variation are not conflated.

2. **The right centralised reference.** `centralised` is the same GNN holding the whole gain
   matrix. It must weakly dominate every decentralised arm; if it does not, something is wrong with
   the run and this script says so rather than letting the number reach a table.

3. **Three classical arms, not one**, so "learned beats classical" is stated against the strongest
   classical option rather than the weakest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).parent / "results"
ARMS = ("learned", "priced", "quantised_embed", "quantised")
LABEL = {
    "learned": "learned symbol",
    "priced": "quantised interference price",
    "quantised_embed": "quantised CSI (matched)",
    "quantised": "quantised CSI (raw bit-planes)",
}


def load() -> list[dict]:
    rows: list[dict] = []
    for tag in ("bitsweep_v2_a", "bitsweep_v2_b"):
        p = RESULTS / f"{tag}.json"
        if not p.exists():
            raise SystemExit(f"missing {p} -- has the sweep finished?")
        rows += json.loads(p.read_text())
    return rows


def instance_matrix(rows, arm, bits) -> np.ndarray | None:
    """(seeds, instances) of per-instance ratios for one cell."""
    sel = [r for r in rows if r["arm"] == arm and r["bits"] == bits and r.get("per_instance_ratio")]
    if not sel:
        return None
    return np.array([r["per_instance_ratio"] for r in sel], dtype=float)


def boot_ci(x: np.ndarray, n: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def main() -> None:
    rows = load()
    (RESULTS / "bitsweep_v2.json").write_text(json.dumps(rows, indent=2))

    bits_list = sorted({r["bits"] for r in rows if r["bits"] is not None})
    n_test = next((r["n_instances"] for r in rows if r.get("n_instances")), 0)

    print(f"test pool: {n_test} instances, "
          f"{len({r['seed'] for r in rows})} seeds\n")

    refs = {}
    for name in ("continuous", "centralised"):
        m = instance_matrix(rows, name, None)
        if m is not None:
            refs[name] = m
            lo, hi = boot_ci(m.mean(axis=0))
            print(f"{name:12s} {m.mean():.4f}  [{lo:.4f}, {hi:.4f}]  "
                  f"seed sd {m.mean(axis=1).std():.4f}")

    floor = instance_matrix(rows, "learned", 0)
    if floor is not None:
        print(f"{'silent':12s} {floor.mean():.4f}")

    # The invariant that the old centralised reference violated.
    if "centralised" in refs:
        c = refs["centralised"].mean()
        worst = max((instance_matrix(rows, a, b).mean()
                     for a in ARMS for b in bits_list
                     if instance_matrix(rows, a, b) is not None), default=0.0)
        status = "OK" if c >= worst else "VIOLATED"
        print(f"\ncentralised dominance: {c:.4f} vs best decentralised {worst:.4f}  [{status}]")
        if c < worst:
            raise SystemExit("centralised reference does not dominate -- do not report this run")

    print(f"\n{'B':>3}  " + "  ".join(f"{LABEL[a]:>30}" for a in ARMS))
    table = []
    for b in bits_list:
        if b == 0:
            continue
        cells, line = {}, f"{b:>3}  "
        for a in ARMS:
            m = instance_matrix(rows, a, b)
            if m is None:
                line += f"{'--':>30}  "
                continue
            per_inst = m.mean(axis=0)
            lo, hi = boot_ci(per_inst)
            cells[a] = dict(mean=float(m.mean()), seed_sd=float(m.mean(axis=1).std()),
                            ci=[float(lo), float(hi)], per_inst=per_inst)
            line += f"{m.mean():.4f} [{lo:.4f},{hi:.4f}]".rjust(30) + "  "
        print(line)
        table.append((b, cells))

    # Learned vs each classical arm, paired over test instances.
    print(f"\npaired over {n_test} instances, learned minus classical:")
    print(f"{'B':>3}  {'vs priced':>26}  {'vs quantised (matched)':>26}")
    summary = []
    for b, cells in table:
        row = {"bits": b}
        line = f"{b:>3}  "
        for a in ("priced", "quantised_embed"):
            if a not in cells or "learned" not in cells:
                line += f"{'--':>26}  "
                continue
            d = cells["learned"]["per_inst"] - cells[a]["per_inst"]
            t, pv = stats.ttest_rel(cells["learned"]["per_inst"], cells[a]["per_inst"])
            lo, hi = boot_ci(d)
            row[a] = {"delta": float(d.mean()), "ci": [float(lo), float(hi)], "p": float(pv)}
            line += f"{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]".rjust(26) + "  "
        print(line)
        summary.append(row)

    ref_out = {}
    for k, v in refs.items():
        lo, hi = boot_ci(v.mean(axis=0))
        ref_out[k] = {"mean": float(v.mean()), "ci": [float(lo), float(hi)],
                      "seed_sd": float(v.mean(axis=1).std())}
    out = {
        "n_instances": n_test,
        "references": {k: r["mean"] for k, r in ref_out.items()},
        "reference_detail": ref_out,
        "silent": float(floor.mean()) if floor is not None else None,
        "cells": [{"bits": b, **{a: {k: v for k, v in c.items() if k != "per_inst"}
                                for a, c in cells.items()}} for b, cells in table],
        "paired": summary,
    }
    (RESULTS / "bitsweep_v2_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'bitsweep_v2_summary.json'}")


if __name__ == "__main__":
    main()
