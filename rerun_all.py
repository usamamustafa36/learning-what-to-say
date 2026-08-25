"""
Regenerate every result under the corrected regime (see regime.py).

Sequential by design: these all want the same GPU, and interleaving them would make the timings in
oran.py and llm_agent.py meaningless. Ordered cheapest-first among the things that answer a claim,
so the evidence register fills up early, with the two long sweeps last.
"""
import time
import traceback

import regime

ORDER = [
    ("oran", lambda: __import__("oran").deployment_report()),
    ("temporal", lambda: __import__("prior_methods").temporal_experiment()),
    ("prior", lambda: __import__("experiments").prior_experiment()),
    ("llm", lambda: __import__("llm_agent").llm_experiment(n_test=64)),
    ("bitsweep", lambda: _bitsweep()),
    ("abstraction", lambda: __import__("analysis").abstraction_sweep()),
    ("symbolic", lambda: __import__("symbolic").distillation_sweep()),
    ("intent", lambda: __import__("intent").intent_experiment()),
    ("rho", lambda: __import__("experiments").rho_sweep()),
    ("sensing", lambda: __import__("sensing").sensing_sweep()),
]


def _bitsweep():
    import experiments

    res = experiments.sweep(tag="bitsweep_fixed")
    experiments.summarise(res)
    return res


if __name__ == "__main__":
    print(f"regime: {regime.summary()}\n", flush=True)
    t_all = time.time()
    for name, fn in ORDER:
        print(f"\n{'='*78}\n  {name}\n{'='*78}", flush=True)
        t0 = time.time()
        try:
            fn()
            print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            print(f"  [{name}] FAILED after {time.time()-t0:.0f}s", flush=True)
            traceback.print_exc()
    print(f"\nall experiments finished in {(time.time()-t_all)/60:.1f} min", flush=True)
