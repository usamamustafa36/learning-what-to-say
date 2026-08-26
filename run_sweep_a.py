"""References + the learned arm + the matched classical control."""
from experiments import sweep_v2

sweep_v2(seeds=(0, 1, 2, 3, 4), steps=8000, n_pairs=8,
         arms=("learned", "quantised_embed"), references=True, tag="bitsweep_v2_a")
