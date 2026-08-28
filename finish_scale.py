"""
Finish a partly-run scale sweep without recomputing the rows already on disk.

`scale_sweep.py` does not resume: it starts from an empty row list and overwrites the JSON, so
re-running a size recomputes everything. This loads the stored rows, works out which
(arm, bits, seed) combinations are absent, rebuilds the pools with the arguments `scale_sweep.main`
uses for that N so the new rows stay comparable, and refuses to run on CPU.

Usage:  python3 finish_scale.py 16

Finish one size of the scale sweep without recomputing what is already on disk.

scale_sweep.py starts from an empty row list and overwrites its JSON, so re-running it for N=8
would redo the 21 rows already stored (about two hours) to get the three that are missing. This
script loads the existing rows, works out which (arm, bits, seed) combinations are absent, and runs
only those.

The pools are rebuilt with exactly the arguments scale_sweep.main uses for N=8, so the new rows are
comparable with the old ones: train pool 8192 at seed 0, test pool 2048 at seed 999 over LAMBDAS.

Usage:  python3 finish_scale.py <N>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from dataset import build_pool  # noqa: E402
from regime import AREA_M, LAMBDAS  # noqa: E402
from scale_sweep import BITS, POOLS, SEEDS  # noqa: E402
from train import Config, run_one  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT = HERE / "results" / f"scale_N{N}.json"


def main() -> None:
    train_size, test_size, threads = POOLS[N]
    torch.set_num_threads(threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to run on CPU: the stored rows were measured on cuda", flush=True)
        sys.exit(1)

    rows = json.loads(OUT.read_text())
    have = {(r["arm"], r["bits"], r["seed"]) for r in rows}

    wanted = [("continuous", None, s) for s in SEEDS]
    wanted += [("learned", 0, s) for s in SEEDS]
    wanted += [("learned", b, s) for b in BITS for s in SEEDS]
    todo = [w for w in wanted if w not in have]

    print(f"N={N} on {dev}, {threads} threads, pools {train_size}/{test_size}", flush=True)
    print(f"  {len(rows)} rows on disk, {len(todo)} to run: {todo}", flush=True)
    if not todo:
        print("  nothing to do", flush=True)
        return

    t0 = time.time()
    tr = build_pool(size=train_size, n_pairs=N, area_m=AREA_M, seed=0, device=dev)
    te = build_pool(size=test_size, n_pairs=N, area_m=AREA_M, seed=999, lambdas=LAMBDAS, device=dev)
    print(f"  pools ready in {time.time()-t0:.0f}s", flush=True)

    for arm, bits, seed in todo:
        if arm == "continuous":
            cfg = Config(bits=0, mode="continuous", steps=8000, seed=seed,
                         usage_bonus=0.2, device=dev)
        elif bits == 0:
            cfg = Config(bits=0, steps=8000, seed=seed, usage_bonus=0.2, device=dev)
        else:
            cfg = Config(bits=bits, mode="vq", steps=8000, seed=seed,
                         usage_bonus=0.2, device=dev)
        r = run_one(cfg, tr, te)
        r.update(arm=arm, bits=bits, seed=cfg.seed, n_pairs=N,
                 train_size=train_size, test_size=test_size, device=dev)
        r.pop("per_instance_ratio", None)
        rows.append(r)
        OUT.write_text(json.dumps(rows, indent=2))   # incremental, survives a kill
        print(f"  {arm} B={bits} seed {seed}: {r['mean_ratio']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    done = [r["mean_ratio"] for r in rows if r["arm"] == "learned" and r["bits"] == 8]
    if len(done) == len(SEEDS):
        import numpy as np
        print(f"  B=8 learned    : {np.mean(done):.4f} +/- {np.std(done):.4f}", flush=True)
    print(f"N={N} complete, {len(rows)} rows in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
