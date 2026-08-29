"""
Why a directly-consumed quantised price needs so many bits.

pricing_budget.py shows a classical pricing loop fed a b-bit price performing far below the silent
floor at small b, while the same loop with an unquantised price converges normally. That could mean
the quantiser is a straw man, which would invalidate the comparison, so it is measured rather than
assumed.

The quantiser is quantile-fitted in the log domain -- the right construction for a positive quantity
spanning decades. The difficulty is intrinsic: the marginal harm pi_i * a_ij spans roughly fifteen
decades across a deployment, because it is a product of a path loss and a price that are each
themselves spread over orders of magnitude. Two levels cannot represent that, and a pricing update
consuming the *value* is wrong by whatever the quantiser is wrong by.

This is the contrast the paper rests on. A learned receiver is handed an index and learns what to do
with it, so two bits are usable. A classical update is handed a number and trusts it, so it needs
enough bits for that number to be approximately right.

    python3 price_dynamic_range.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import cached_pool                                        # noqa: E402
from pricing_budget import fit_levels, quantise                        # noqa: E402
from regime import AREA_M, LAMBDAS, P_MAX_W                            # noqa: E402

RESULTS = HERE / "results"


def main() -> None:
    pool = cached_pool("test_N8_2048", size=2048, n_pairs=8, area_m=AREA_M, seed=999,
                       lambdas=LAMBDAS, device="cpu")
    A = pool.gains_obs.numpy()[:256]
    noise = pool.noise_power
    samples = []
    for m in range(len(A)):
        a = A[m]
        d = np.diag(a)
        p0 = np.full(a.shape[0], P_MAX_W)
        interf = a @ p0 - d * p0 + noise
        sig = d * p0
        price = sig / (np.log(2.0) * interf * (interf + sig))
        samples.append((a * price[:, None]).ravel())
    samples = np.concatenate(samples)
    pos = samples[samples > 0]

    out = {
        "decades": float(np.log10(pos.max() / pos.min())),
        "min": float(pos.min()), "max": float(pos.max()),
        "n_samples": int(pos.size), "levels": [],
    }
    for b in (1, 2, 3, 4, 6, 8):
        lv = fit_levels(samples, b)
        rel = np.abs(quantise(pos, lv) - pos) / pos
        out["levels"].append({
            "bits": b, "n_levels": 2 ** b,
            "median_rel_error": float(np.median(rel)),
            "p90_rel_error": float(np.percentile(rel, 90)),
        })
        print(f"  b={b}: {2**b:>3} levels, median rel. error {np.median(rel)*100:7.1f}%", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "price_dynamic_range.json").write_text(json.dumps(out, indent=2))
    print(f"\ndynamic range {out['decades']:.1f} decades")
    print(f"wrote {RESULTS / 'price_dynamic_range.json'}")


if __name__ == "__main__":
    main()
