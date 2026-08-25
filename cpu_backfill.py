"""
Backfill the results still blocked, on CPU, cheapest-first.

Ordered so the paper's unresolved macros resolve in order of how much they matter: the adversarial
detection figures at B=6 are the paper's security claim, the LLM protocol row is a table cell. If the
GPU returns mid-run, `when_gpu_back.sh` supersedes this at full breadth.
"""
import time
import traceback

import torch

print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

JOBS = [
    ("adversarial B=6", lambda: __import__("adversarial").adversarial_sweep(
        bits_list=(6,), attacker_counts=(1, 2, 3), steps=8000)),
    ("llm", lambda: __import__("llm_agent").llm_experiment(n_test=64)),
]

for name, fn in JOBS:
    print(f"\n{'='*70}\n  {name}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        fn()
        print(f"  [{name}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    except Exception:
        print(f"  [{name}] FAILED after {(time.time()-t0)/60:.1f} min", flush=True)
        traceback.print_exc()
