"""
What happens when the symbols themselves are corrupted.

Every result elsewhere in this study crosses a noiseless, delay-free message channel. That is a real
idealisation, and it is one the paper's own design makes dangerous: a learned codebook has no
structure, so flipping one bit of an index does not perturb the message, it substitutes an unrelated
codeword. A quantised price, whose index is monotone in the quantity it encodes, should degrade far
more gracefully. Whether it does is measurable.

Two impairments, both applied to every edge rather than to an attacker's subset:

  BitFlipChannel  each of the B bits of the transmitted index flips independently w.p. `ber`.
  ErasureChannel  w.p. `p` the symbol is lost and replaced by the modal honest symbol.

The erasure model deserves its caveat stated rather than buried: a true erasure would drop the term
and renormalise the mean, but the substitution hook carries indices and the aggregation divides by a
fixed N-1, so substituting the most common honest symbol is the closest honest approximation -- "no
news, assume typical" -- and it understates erasure damage slightly.

Reuses the `symbol_fn` hook in ProtocolGNN.forward, whose neutrality is asserted in adversarial.py.

    python3 noisy_channel.py
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

from adversarial import honest_symbols                                 # noqa: E402
from checkpoints import train_cached                                   # noqa: E402
from dataset import cached_pool                                        # noqa: E402
from regime import AREA_M, LAMBDAS                                     # noqa: E402
from train import Config, evaluate                                     # noqa: E402

RESULTS = HERE / "results"
N_PAIRS, TEST_SIZE = 8, 2048
BITS = (2, 4, 6)
SEEDS = (0, 1, 2)
BERS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ERASURES = (0.0, 0.01, 0.05, 0.10, 0.25)


class BitFlipChannel:
    """Flip each of the B bits of every transmitted index independently with probability `ber`."""

    def __init__(self, honest_fn, bits: int, ber: float, seed: int = 0):
        self.honest, self.bits, self.ber = honest_fn, bits, ber
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, node, edge):
        sym = self.honest(node, edge)
        if self.ber <= 0:
            return sym
        mask = torch.zeros_like(sym)
        for b in range(self.bits):
            flip = (torch.rand(sym.shape, generator=self.gen).to(sym.device) < self.ber)
            mask = mask | (flip.long() << b)
        return sym ^ mask


class ErasureChannel:
    """Replace a lost symbol with the modal honest symbol: 'no news, assume typical'."""

    def __init__(self, honest_fn, p_erase: float, seed: int = 0):
        self.honest, self.p = honest_fn, p_erase
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, node, edge):
        sym = self.honest(node, edge)
        if self.p <= 0:
            return sym
        modal = torch.mode(sym.flatten()).values
        lost = (torch.rand(sym.shape, generator=self.gen).to(sym.device) < self.p)
        return torch.where(lost, torch.full_like(sym, int(modal)), sym)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to run on CPU", flush=True)
        sys.exit(1)
    tr = cached_pool(f"train_N{N_PAIRS}_8192", size=8192, n_pairs=N_PAIRS, area_m=AREA_M,
                     seed=0, device=dev)
    te = cached_pool(f"test_N{N_PAIRS}_{TEST_SIZE}", size=TEST_SIZE, n_pairs=N_PAIRS,
                     area_m=AREA_M, seed=999, lambdas=LAMBDAS, device=dev)

    rows = []
    for arm, mode in (("learned", "vq"), ("priced", "vq")):
        for bits in BITS:
            for seed in SEEDS:
                cfg = Config(bits=bits, mode=mode, steps=8000, seed=seed,
                             messenger="priced" if arm == "priced" else "learned")
                try:
                    net = train_cached(cfg, tr)
                except TypeError:
                    # Config has no `messenger` field in this build: only the learned arm applies.
                    if arm == "priced":
                        continue
                    cfg = Config(bits=bits, mode=mode, steps=8000, seed=seed)
                    net = train_cached(cfg, tr)
                honest = honest_symbols(net)

                for kind, values in (("ber", BERS), ("erasure", ERASURES)):
                    for v in values:
                        fn = (BitFlipChannel(honest, bits, v, seed) if kind == "ber"
                              else ErasureChannel(honest, v, seed))
                        t0 = time.time()
                        r = evaluate(net, cfg, te, symbol_fn=fn)
                        rows.append({"arm": arm, "bits": bits, "seed": seed, "impairment": kind,
                                     "level": v, "mean_ratio": r["mean_ratio"],
                                     "n_instances": len(te)})
                        print(f"  {arm:>7} B={bits} s={seed} {kind}={v:<7g} "
                              f"{r['mean_ratio']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                        RESULTS.mkdir(parents=True, exist_ok=True)
                        (RESULTS / "noisy_channel.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {RESULTS / 'noisy_channel.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
