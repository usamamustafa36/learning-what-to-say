"""
Zero-shot transfer across network size, and the one thing that stops it working.

The manuscript claimed a model trained at one network size "can be evaluated at another without
retraining", on the strength of the architecture being permutation-equivariant with N-independent
parameter count. That is an argument about the architecture, not evidence about the model, and the
scale sweep retrained from scratch at every size -- so the claim was never tested.

It is testable with no new machinery: `train()` returns the net and `evaluate()` accepts any pool.

The interesting part is that the parameters are N-free but the *input statistics* are not. The
normaliser standardises the total received interference, which is a sum over N-1 interferers in a
fixed area, so its mean drifts with N by roughly 0.6-0.7 sigma per doubling. That gives two arms
which together localise the failure rather than merely reporting it:

  frozen    carry the N=8 normaliser to the target size. The honest zero-shot condition.
  refit     recompute Normaliser.fit on the target-size pool. Needs no gradients and no labels --
            just unlabelled channel measurements, which a deployment has by definition.

If frozen degrades and refit does not, the architecture claim survives with a stated caveat: what
fails to transfer is six frozen constants, and recovering them costs no training.

    python3 generalisation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from agents import Normaliser                                          # noqa: E402
from checkpoints import train_cached                                   # noqa: E402
from dataset import cached_pool                                        # noqa: E402
from regime import AREA_M, LAMBDAS                                     # noqa: E402
from train import Config, evaluate                                     # noqa: E402

RESULTS = HERE / "results"
TRAIN_N = 8
TEST_NS = (4, 8, 12, 16, 20, 32)
BITS = (0, 2, 4, 6)
SEEDS = (0, 1, 2)
TEST_SIZE = 1024          # smaller than the 2048 of Table IV: the axis here is N, not sampling noise


def norm_drift(ns=TEST_NS, size=512, seed=999) -> list[dict]:
    """The N-dependence of the frozen constants, measured rather than asserted."""
    out = []
    for n in ns:
        pool = cached_pool(f"drift_N{n}_{size}", size=size, n_pairs=n, area_m=AREA_M, seed=seed)
        nm = Normaliser.fit(pool.gains_obs)
        out.append({"n_pairs": n, "direct_mean": nm.direct_mean, "direct_std": nm.direct_std,
                    "recv_mean": nm.recv_mean, "recv_std": nm.recv_std,
                    "edge_mean": nm.edge_mean, "edge_std": nm.edge_std})
        print(f"  N={n:>2}  direct {nm.direct_mean:+.3f}  recv {nm.recv_mean:+.3f}"
              f"  edge {nm.edge_mean:+.3f}", flush=True)
    return out


def transfer_sweep(tag="transfer") -> list[dict]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to run on CPU: the stored rows were measured on cuda", flush=True)
        sys.exit(1)

    tr = cached_pool(f"train_N{TRAIN_N}_8192", size=8192, n_pairs=TRAIN_N, area_m=AREA_M,
                     seed=0, device=dev)
    tests = {}
    for n in TEST_NS:
        t0 = time.time()
        tests[n] = cached_pool(f"test_N{n}_{TEST_SIZE}", size=TEST_SIZE, n_pairs=n, area_m=AREA_M,
                               seed=999, lambdas=LAMBDAS, device=dev)
        print(f"  pool N={n} ready in {time.time()-t0:.0f}s", flush=True)

    rows: list[dict] = []
    for bits in BITS:
        for seed in SEEDS:
            cfg = Config(bits=bits, mode="vq", steps=8000, seed=seed)
            t0 = time.time()
            net = train_cached(cfg, tr)
            frozen_norm = net.norm
            print(f"  trained B={bits} seed={seed} in {time.time()-t0:.0f}s", flush=True)

            for n, te in tests.items():
                for arm in ("frozen", "refit"):
                    net.norm = frozen_norm if arm == "frozen" else Normaliser.fit(te.gains_obs)
                    r = evaluate(net, cfg, te)
                    rows.append({
                        "arm": arm, "train_n": TRAIN_N, "test_n": n, "bits": bits, "seed": seed,
                        "mean_ratio": r["mean_ratio"], "per_lambda": r["per_lambda"],
                        "n_instances": len(te),
                    })
                    print(f"    B={bits} s={seed} N={n:>2} {arm:>6}: {r['mean_ratio']:.4f}",
                          flush=True)
                net.norm = frozen_norm
                RESULTS.mkdir(parents=True, exist_ok=True)
                (RESULTS / f"{tag}.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> None:
    print("normaliser drift with N:", flush=True)
    drift = norm_drift()
    (RESULTS / "norm_drift.json").write_text(json.dumps(drift, indent=2))
    print("\nzero-shot transfer:", flush=True)
    rows = transfer_sweep()
    print(f"\nwrote {RESULTS / 'transfer.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
