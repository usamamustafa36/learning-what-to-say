"""
B*(eps, N): the empirical minimum per-edge communication budget.

The bit-budget sweep answers "how well does B bits do". Inverting it answers the question the paper
is actually about -- "how many bits does a target cost you" -- and that inversion is the object we
define rather than the six-bit operating point we happened to measure:

    B*(eps, N, rho) = min { B in N : Ubar(B, N, rho) >= (1 - eps) * Ubar_ref(N, rho) },   min {} = +inf

Two reference conventions, reported separately because they answer different questions and blurring
them is how a "minimum bits" claim becomes unfalsifiable:

  central   Ubar_ref is the best-known centralised multi-start reference. The stored `mean_ratio`
            is already this ratio, so B* reads straight off. "How many bits to come within eps of
            what a centralised allocator achieves."
  window    Ubar_ref is the unquantised-message ceiling, normalised against the silent floor:
            (r_B - r_0) / (r_inf - r_0). "How many bits to capture 1-eps of what signalling can buy
            at all." This is the convention behind the existing ScaleNxRecovSix macros.

Two rigour properties, because a reviewer will look for both:

  censoring     If no budget on the grid reaches the target, B* is right-censored. We emit
                censored=True and a ">8" string rather than extrapolating off the end of the grid.
  monotonicity  B* presumes the curve is non-decreasing in B, which seed means do not guarantee --
                at N=4 the measured B=8 sits *below* B=6. We compute B* on the raw curve and
                cross-check against an isotonic (non-decreasing) fit; a cell where the two disagree
                is one whose grid or seed count is too thin, and it is flagged rather than reported
                as if it were solid.

    python3 bstar.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
SIZES = (4, 8, 16)
EPSILONS = (0.10, 0.07, 0.05, 0.03)


def curve(rows: list[dict]) -> dict:
    """bits -> seed-mean ratio, plus 'cont' for the unquantised-message arm."""
    by: dict = {}
    for r in rows:
        key = "cont" if r["arm"] == "continuous" else r["config"]["bits"]
        by.setdefault(key, []).append(r["mean_ratio"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def isotonic(xs: list[int], ys: list[float]) -> list[float]:
    """Pool-adjacent-violators: the closest non-decreasing fit in least squares."""
    y = list(ys)
    w = [1.0] * len(y)
    i = 0
    while i < len(y) - 1:
        if y[i] <= y[i + 1] + 1e-15:
            i += 1
            continue
        tot = w[i] + w[i + 1]
        y[i] = (w[i] * y[i] + w[i + 1] * y[i + 1]) / tot
        w[i] = tot
        del y[i + 1], w[i + 1]
        while i > 0 and y[i - 1] > y[i]:
            tot = w[i - 1] + w[i]
            y[i - 1] = (w[i - 1] * y[i - 1] + w[i] * y[i]) / tot
            w[i - 1] = tot
            del y[i], w[i]
            i -= 1
    out, k = [], 0
    for wi, yi in zip(w, y):
        out.extend([yi] * int(round(wi)))
        k += 1
    return out[:len(xs)]


def b_star(c: dict, eps: float, ref: str = "central") -> dict:
    """Smallest budget on the grid meeting the target, with censoring and a monotonicity check."""
    bits = sorted(b for b in c if b != "cont" and b > 0)
    if ref == "central":
        vals = [c[b] for b in bits]
        target = 1.0 - eps
    else:
        f0, fc = c[0], c["cont"]
        vals = [(c[b] - f0) / (fc - f0) for b in bits]
        target = 1.0 - eps

    hit = next((b for b, v in zip(bits, vals) if v >= target), None)
    iso = isotonic(bits, vals)
    hit_iso = next((b for b, v in zip(bits, iso) if v >= target), None)

    return {
        "eps": eps, "ref": ref,
        "b_star": hit,
        "censored": hit is None,
        "display": str(hit) if hit is not None else f">{bits[-1]}",
        "b_star_isotonic": hit_iso,
        "monotone_agrees": hit == hit_iso,
        "grid": bits,
        "values": [round(v, 4) for v in vals],
    }


def main() -> None:
    out: list[dict] = []
    for n in SIZES:
        path = RESULTS / f"scale_N{n}.json"
        if not path.exists():
            continue
        c = curve(json.loads(path.read_text()))
        for ref in ("central", "window"):
            for eps in EPSILONS:
                row = b_star(c, eps, ref)
                row["n_pairs"] = n
                out.append(row)

    (RESULTS / "bstar.json").write_text(json.dumps(out, indent=2))

    for ref in ("central", "window"):
        print(f"\nB*(eps, N)  --  reference: {ref}")
        print(f"{'N':>4} " + " ".join(f"{'eps='+str(int(e*100))+'%':>8}" for e in EPSILONS))
        for n in SIZES:
            cells = [r for r in out if r["n_pairs"] == n and r["ref"] == ref]
            if not cells:
                continue
            line = f"{n:>4} "
            for e in EPSILONS:
                r = next(c for c in cells if c["eps"] == e)
                mark = "" if r["monotone_agrees"] else "*"
                line += f"{r['display'] + mark:>8} "
            print(line)
    flagged = [r for r in out if not r["monotone_agrees"]]
    if flagged:
        print(f"\n* {len(flagged)} cell(s) where the raw and isotonic curves disagree -- grid or "
              f"seed count too thin, reported with the flag rather than as a solid number:")
        for r in flagged:
            print(f"    N={r['n_pairs']} eps={r['eps']:.2f} ref={r['ref']}: "
                  f"raw {r['display']} vs isotonic {r['b_star_isotonic']}")
    print(f"\nwrote {RESULTS / 'bstar.json'}")


if __name__ == "__main__":
    main()
