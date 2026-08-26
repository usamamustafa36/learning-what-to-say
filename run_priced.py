"""
The strong classical arm: B bits of quantised interference price.

Run as a separate job so it does not discard the bitsweep_v2 run already in flight. Pools are
built from the same seeds (0 for train, 999 for test) and `build_pool` is deterministic, so this
lands on byte-identical instances and the two files merge without a caveat.
"""
from experiments import sweep_v2

sweep_v2(seeds=(0, 1, 2, 3, 4), steps=8000, n_pairs=8,
         arms=("priced",), references=False, tag="bitsweep_v2_priced")
