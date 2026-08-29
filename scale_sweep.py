"""
Does the saturation budget B* move with network size?

The paper answers its title question at N=8. That is the single largest gap in the evidence: the
architecture is permutation equivariant with N-independent parameter count, signalling is quoted at
N=32, and yet no performance number is reported at any N but 8. This sweep supplies the missing
axis.

Run as `python3 scale_sweep.py <N>`; each N is a separate process so partial results survive and so
three sizes can share the machine. Results are written incrementally -- a run killed halfway still
leaves usable rows on disk.

Pools are held at the main sweep's 8192/2048 for N <= 16 and halved at N=32, where the oracle -- a
multi-start projected gradient over N variables per instance -- dominates the cost. Pool size
affects the *width* of a confidence interval, not the location of a saturation knee, so the trend
being measured is unaffected; the per-N pool sizes are recorded in every row so that no table can
quietly compare across them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataset import build_pool
from regime import AREA_M, LAMBDAS
from train import Config, run_one

RESULTS = Path(__file__).parent / "results"
BITS = (1, 2, 3, 4, 5, 6, 7, 8)   # 5 and 7 added: B*(eps) is read off this ladder
SEEDS = (0, 1, 2)

# (train pool, test pool, torch threads) by network size.
POOLS = {4: (8192, 2048, 4), 8: (8192, 2048, 4), 16: (8192, 2048, 4), 32: (4096, 1024, 4)}


def main(n: int) -> None:
    train_size, test_size, threads = POOLS[n]
    torch.set_num_threads(threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"scale_N{n}"
    out_path = RESULTS / f"{tag}.json"
    rows: list[dict] = []

    t0 = time.time()
    print(f"N={n} on {dev}, {threads} threads, pools {train_size}/{test_size}", flush=True)
    tr = build_pool(size=train_size, n_pairs=n, area_m=AREA_M, seed=0, device=dev)
    te = build_pool(size=test_size, n_pairs=n, area_m=AREA_M, seed=999, lambdas=LAMBDAS, device=dev)
    print(f"  pools ready in {time.time()-t0:.0f}s", flush=True)

    def record(cfg, arm, bits):
        r = run_one(cfg, tr, te)
        r.update(arm=arm, bits=bits, seed=cfg.seed, n_pairs=n,
                 train_size=train_size, test_size=test_size, device=dev)
        # Drop the per-instance vector here: across four network sizes it would dominate the file
        # and the scaling question is answered by the means.
        r.pop("per_instance_ratio", None)
        rows.append(r)
        out_path.write_text(json.dumps(rows, indent=2))    # incremental: survives a kill
        return r

    for seed in SEEDS:
        r = record(Config(bits=0, mode="continuous", steps=8000, seed=seed,
                          usage_bonus=0.2, device=dev), "continuous", None)
        print(f"  continuous seed {seed}: {r['mean_ratio']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    for seed in SEEDS:
        record(Config(bits=0, steps=8000, seed=seed, usage_bonus=0.2, device=dev), "learned", 0)
    print(f"  B=0 silent    : {np.mean([r['mean_ratio'] for r in rows[-len(SEEDS):]]):.4f}", flush=True)

    for bits in BITS:
        for seed in SEEDS:
            record(Config(bits=bits, mode="vq", steps=8000, seed=seed,
                          usage_bonus=0.2, device=dev), "learned", bits)
        got = [r["mean_ratio"] for r in rows[-len(SEEDS):]]
        print(f"  B={bits} learned    : {np.mean(got):.4f} +/- {np.std(got):.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"N={n} done in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
