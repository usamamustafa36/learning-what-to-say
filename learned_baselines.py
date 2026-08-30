"""
Phases 3 and 4: a learned competitor that differs in HOW the message is made, and rounds vs bits.

Every competitor so far is classical, so "learning the protocol helps" is only ever tested against
things that do not learn. Two arms fix that, both sharing the graph, rounds, aggregation, readout,
optimiser, schedule, steps, seeds and budget with the learned arm:

  binary    DIAL/CommNet style. The message is a continuous vector pushed through a plain
            straight-through sigmoid, so the receiver gets B learned bits with no codebook and no
            entropy bonus. Same budget, different mechanism for producing the symbol.
  vq-noent  the learned codebook with the entropy bonus switched off, which measures what codebook
            collapse costs. Realised entropy is reported alongside the score, because a collapsed
            codebook transmits fewer bits than its budget claims and the comparison is otherwise
            mislabelled.

Phase 4 sweeps rounds against bits at a matched product: (B,R) in {(6,1),(3,2),(2,3)} all cost
B*R*(N-1) = 42 bits per agent per slot, with (12,1) as an unmatched anchor at double the budget.
R > 1 is otherwise untouched in this study, and the overhead accounting assumes R = 1.

The (12,1) anchor does not fit an 8 GiB card: evaluation one-hots every edge of the test pool
against 4096 codewords, about 2 GiB, on top of the model. It is recorded as skipped rather than
allowed to kill the arms that already ran. Running it needs chunked evaluation; a smaller test pool
would make the number incomparable to the rest of the grid. The manuscript does not cite it.

    python3 learned_baselines.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset import cached_pool                                        # noqa: E402
from regime import AREA_M, LAMBDAS                                     # noqa: E402
from train import Config, run_one                                      # noqa: E402

RESULTS = HERE / "results"
N_PAIRS = 8
BITS = (1, 2, 3, 4, 6, 8)
SEEDS = (0, 1, 2, 3, 4)
ROUND_CELLS = ((6, 1), (3, 2), (2, 3), (12, 1))


def realised_entropy(net, pool, cfg) -> float | None:
    """Bits actually transmitted, so a collapsed codebook is not reported at its nominal budget."""
    from agents import graph_inputs
    from train import node_extras, price_ref, wants_full_csi
    if cfg.bits == 0 or cfg.mode != "vq":
        return None
    with torch.no_grad():
        lam = torch.full((len(pool),), 0.5, device=pool.gains.device)
        gen = torch.Generator(device=pool.gains.device).manual_seed(0)
        node, edge = graph_inputs(pool.gains_obs, lam, extra_node=node_extras(cfg, None, None, gen),
                                  norm=getattr(net, "norm", None),
                                  full_csi=wants_full_csi(cfg), price=price_ref(cfg, pool))
        _, syms = net(node, edge, return_symbols=True)
        s = syms[0].flatten().cpu().numpy()
    _, counts = np.unique(s, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps = 200 if args.smoke else 8000
    seeds = (0,) if args.smoke else SEEDS
    bits_list = (2,) if args.smoke else BITS
    cells = (((6, 1), (3, 2)) if args.smoke else ROUND_CELLS)
    tr_n, te_n = (256, 64) if args.smoke else (8192, 2048)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda" and not args.smoke:
        print("refusing to run on CPU", flush=True); sys.exit(1)
    tr = cached_pool(f"train_N{N_PAIRS}_{tr_n}", size=tr_n, n_pairs=N_PAIRS, area_m=AREA_M,
                     seed=0, device=dev)
    te = cached_pool(f"test_N{N_PAIRS}_{te_n}", size=te_n, n_pairs=N_PAIRS, area_m=AREA_M,
                     seed=999, lambdas=LAMBDAS, device=dev)
    out = RESULTS / ("learned_baselines_smoke.json" if args.smoke else "learned_baselines.json")
    rows = []

    skipped = []

    def record(cfg, arm, extra=None):
        t0 = time.time()
        try:
            r = run_one(cfg, tr, te)
        except torch.cuda.OutOfMemoryError as e:
            # A 12-bit codebook is 4096 codewords, and evaluation one-hots every edge of the test
            # pool against all of them: 2048 x 8 x 7 x 4096 floats, about 2 GiB on an 8 GiB card.
            # One cell that will not fit must not take the arms that already ran with it, so the
            # failure is recorded and the sweep continues. Fixing it needs chunked evaluation, not
            # a smaller pool, or the number stops being comparable to the rest of the grid.
            torch.cuda.empty_cache()
            skipped.append({"arm": arm, "bits": cfg.bits, "rounds": cfg.rounds, "seed": cfg.seed,
                            "reason": "CUDA OOM", "detail": str(e).split("\n")[0]})
            print(f"  {arm:10s} B={cfg.bits} R={cfg.rounds} s={cfg.seed}: SKIPPED (CUDA OOM)",
                  flush=True)
            return
        r.pop("per_instance_ratio", None)
        r.update({"arm": arm, "bits": cfg.bits, "rounds": cfg.rounds, "seed": cfg.seed,
                  "mode": cfg.mode, "usage_bonus": cfg.usage_bonus,
                  "signalling_bits": cfg.bits * (N_PAIRS - 1) * cfg.rounds,
                  "n_instances": len(te)})
        if extra:
            r.update(extra)
        rows.append(r)
        RESULTS.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        ent = f" ent {r['entropy_bits']:.2f}" if r.get("entropy_bits") else ""
        print(f"  {arm:10s} B={cfg.bits} R={cfg.rounds} s={cfg.seed}: "
              f"{r['mean_ratio']:.4f}{ent}  ({time.time()-t0:.0f}s)", flush=True)

    # --- Phase 3: how the message is produced -----------------------------------------------
    for bits in bits_list:
        for seed in seeds:
            record(Config(bits=bits, mode="binary", steps=steps, seed=seed), "binary")
    for bits in (bits_list if args.smoke else (2, 4, 6)):
        for seed in (seeds if args.smoke else (0, 1, 2)):
            cfg = Config(bits=bits, mode="vq", steps=steps, seed=seed, usage_bonus=0.0)
            from checkpoints import train_cached
            from train import evaluate
            net = train_cached(cfg, tr)
            r = evaluate(net, cfg, te)
            r.pop("per_instance_ratio", None)
            r.update({"arm": "vq-noent", "bits": bits, "rounds": 1, "seed": seed, "mode": "vq",
                      "usage_bonus": 0.0, "signalling_bits": bits * (N_PAIRS - 1),
                      "n_instances": len(te), "entropy_bits": realised_entropy(net, te, cfg)})
            rows.append(r); out.write_text(json.dumps(rows, indent=2))
            print(f"  vq-noent   B={bits} s={seed}: {r['mean_ratio']:.4f} "
                  f"ent {r['entropy_bits']:.2f}", flush=True)

    # --- Phase 4: rounds against bits at matched budget --------------------------------------
    for (b, rnd) in cells:
        for seed in (seeds if args.smoke else (0, 1, 2)):
            record(Config(bits=b, mode="vq", rounds=rnd, steps=steps, seed=seed), "rounds")
    print(f"wrote {out} ({len(rows)} rows)")
    if skipped:
        (RESULTS / (out.stem + "_skipped.json")).write_text(json.dumps(skipped, indent=2))
        print(f"{len(skipped)} cells skipped and recorded in {out.stem}_skipped.json:")
        for sk in skipped:
            print(f"  {sk['arm']} B={sk['bits']} R={sk['rounds']} s={sk['seed']}: {sk['reason']}")


if __name__ == "__main__":
    main()
