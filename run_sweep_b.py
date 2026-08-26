"""The strong classical arm, plus the raw-bit-plane ablation.

Split from run_sweep_a so the two halves run concurrently. `build_pool` is seeded (0 train,
999 test), so both halves land on byte-identical instances and the files merge without a caveat.
B=0 is omitted here: with no bits transmitted every arm collapses to the silent floor, which
run_sweep_a already measures.
"""
from experiments import sweep_v2

sweep_v2(bits_list=(1, 2, 3, 4, 6, 8), seeds=(0, 1, 2, 3, 4), steps=8000, n_pairs=8,
         arms=("priced", "quantised"), references=False, tag="bitsweep_v2_b")
