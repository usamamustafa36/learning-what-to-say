"""
Why does a matched-budget classical arm stop improving with B?

The bit budget is spent at the *sender*. What the receiver observes is the mean of N-1 messages,
and mean-pooling is lossy. The most any permutation-invariant aggregator can recover is the
unordered *multiset* of neighbour symbols, and whether it recovers even that depends on how the
B-bit index is represented on the wire.

    raw bit-planes  the index written out in binary, as a naive matched-budget control does.
    codebook        the index selects a row of a msg_dim-wide codebook -- what the learned arm uses,
                    and what QuantisedCSIEmbedGNN gives the classical arm so that the two are
                    matched on representation as well as on bit count.

**Both quantities are computed exactly or near-exactly. Nothing here is a plug-in entropy estimate
over a large alphabet.** An earlier version of this script estimated all three entropies by counting
distinct aggregates over 200k random draws. That is invalid above about B=4: the alphabet of
multisets is far larger than the sample, so the estimate saturates at log2(draws) = 17.61 bits and
the measured gap appears to *close* at high B when in truth it widens. The three fixes:

1. `H(raw bit-planes)` is analytic. If the index is uniform on 2^B then its B bits are i.i.d.
   uniform, so the B plane-sums are *independent* Binomial(n, 1/2) and

       H(raw agg) = B * H(Binomial(n, 1/2)).

2. `H(multiset)` uses H(M) = H(X) - H(X|M) with X the ordered tuple. Given the multiset every
   ordering is equally likely, so

       H(M) = n*log2(k) - E[log2(n! / prod_i c_i!)],

   and the correction term is a bounded *scalar* whose expectation Monte Carlo estimates to three
   decimals in seconds -- no large-alphabet estimation anywhere.

3. `H(codebook agg)` equals H(multiset) whenever the mean of codebook rows is injective on
   multisets. That is checked here rather than assumed, and injectivity is a property of the
   codebook geometry alone -- it does not depend on how often each index occurs.

**Both entropies are the uniform-index idealisation.** H(raw) assumes the B index bits are i.i.d.
uniform; H(multiset) assumes the symbols are i.i.d. uniform. A Lloyd-Max quantiser minimises MSE,
not entropy, so its cells are not equiprobable in general and neither is a learned codebook's usage.
The last section of this script therefore measures the real index entropy on real edge features, so
the size of the idealisation is reported rather than assumed away. What does *not* depend on
uniformity is the qualitative conclusion: injectivity of the codebook map and many-to-one-ness of
the bit-plane map are properties of the representations, not of the index distribution.

Run: python3 diagnostics/aggregation_capacity.py
"""

from __future__ import annotations

import json
import sys
from math import comb, factorial, log2
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_NEIGHBOURS = 7            # N = 8 agents, so each receiver aggregates 7 messages
DRAWS = 2_000_000           # for the scalar correction term only
MSG_DIM = 16
BITS = (1, 2, 3, 4, 6, 8)


def h_binomial_half(n: int) -> float:
    """H(Binomial(n, 1/2)) in bits -- the entropy of one aggregated bit-plane."""
    p = np.array([comb(n, k) for k in range(n + 1)], dtype=float) / 2.0**n
    return float(-(p * np.log2(p)).sum())


def h_multiset(n: int, bits: int, rng: np.random.Generator, draws: int = DRAWS) -> float:
    """H of the unordered multiset of n i.i.d. uniform symbols from 2^bits levels."""
    k = 1 << bits
    idx = rng.integers(0, k, size=(draws, n))
    idx.sort(axis=1)
    # Running multiplicity within each run of equal values; summing log2 of it over the row gives
    # exactly sum_i log2(c_i!) for that draw.
    mult = np.ones((draws, n), dtype=np.int64)
    for j in range(1, n):
        mult[:, j] = np.where(idx[:, j] == idx[:, j - 1], mult[:, j - 1] + 1, 1)
    e_log_orderings = log2(factorial(n)) - float(np.log2(mult).sum(axis=1).mean())
    return n * bits - e_log_orderings


def codebook_is_injective(n: int, bits: int, rng: np.random.Generator,
                          msg_dim: int = MSG_DIM, trials: int = 200_000) -> bool:
    """
    Does the mean of codebook rows separate distinct multisets?

    Collisions are checked against the *multiset*, not the tuple: two orderings of the same multiset
    are supposed to collide, and counting those as failures would be wrong.
    """
    k = 1 << bits
    codebook = rng.standard_normal((k, msg_dim)) * 0.5
    idx = rng.integers(0, k, size=(trials, n))
    idx.sort(axis=1)
    agg = codebook[idx].mean(axis=1)
    keys = {}
    for a, m in zip(np.round(agg, 9), map(tuple, idx)):
        t = tuple(a)
        if keys.setdefault(t, m) != m:
            return False
    return True


def empirical_index_entropy(bits_list=BITS, size: int = 1024, n_pairs: int = 8) -> dict:
    """
    How far from uniform is a real Lloyd-Max index on real edge features?

    This is what licenses reading the analytic columns as tight rather than merely as bounds. It
    needs the channel model and the quantiser, so it is imported lazily -- the entropy analysis
    above stands on its own without torch.
    """
    import torch

    from agents import Normaliser, graph_inputs
    from baselines import LloydMaxQuantizer
    from dataset import build_pool
    from regime import AREA_M

    pool = build_pool(size=size, n_pairs=n_pairs, area_m=AREA_M, seed=0, device="cpu")
    norm = Normaliser.fit(pool.gains_obs)
    _, edge = graph_inputs(pool.gains_obs, torch.rand(len(pool)), norm=norm)
    v = edge[..., 0].numpy().ravel()
    out = {}
    for bits in bits_list:
        idx = LloydMaxQuantizer(bits).fit(v).indices(v)
        cnt = np.bincount(idx, minlength=1 << bits).astype(float)
        pr = cnt[cnt > 0] / cnt.sum()
        out[bits] = {"h_index": float(-(pr * np.log2(pr)).sum()), "cells_used": int(len(pr))}
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    hb = h_binomial_half(N_NEIGHBOURS)
    print(f"{N_NEIGHBOURS} neighbours, codebook width {MSG_DIM}")
    print(f"H(Binomial({N_NEIGHBOURS}, 1/2)) = {hb:.4f} bits per aggregated plane\n")
    print(f"{'B':>2}  {'H(multiset)':>12}  {'H(raw agg)':>11}  {'H(codebook agg)':>16}  {'loss':>7}"
          f"  {'cb injective':>13}")

    out = []
    for bits in BITS:
        hm = h_multiset(N_NEIGHBOURS, bits, rng)
        hr = bits * hb
        inj = codebook_is_injective(N_NEIGHBOURS, bits, rng)
        hc = hm if inj else float("nan")
        print(f"{bits:>2}  {hm:>12.3f}  {hr:>11.3f}  {hc:>16.3f}  {hm - hr:>7.3f}  {str(inj):>13}")
        out.append({"bits": bits, "h_multiset": hm, "h_raw_planes": hr, "h_codebook": hc,
                    "loss_bits": hm - hr, "codebook_injective": bool(inj),
                    "n_neighbours": N_NEIGHBOURS, "msg_dim": MSG_DIM, "draws": DRAWS})

    emp = empirical_index_entropy()
    print(f"\nreal Lloyd-Max index entropy vs the uniform idealisation:")
    print(f"{'B':>2}  {'H(index)':>9}  {'shortfall':>10}  {'cells used':>11}")
    for bits, d in emp.items():
        print(f"{bits:>2}  {d['h_index']:>9.3f}  {bits - d['h_index']:>10.3f}"
              f"  {d['cells_used']:>6}/{1 << bits}")
        for r in out:
            if r["bits"] == bits:
                r["h_index_empirical"] = d["h_index"]
                r["index_shortfall_bits"] = bits - d["h_index"]
                r["cells_used"] = d["cells_used"]

    res = Path(__file__).resolve().parent.parent / "results" / "aggregation.json"
    res.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {res}")
    print("H(raw agg) is exact; H(multiset) is a scalar-expectation estimate; H(codebook agg) is")
    print("H(multiset) wherever the mean map was verified injective on multisets.")


if __name__ == "__main__":
    main()
