"""
State abstraction: what the agents actually chose to encode.

The advert names state abstraction directly, and it is the difference between a protocol paper and a
black-box RL paper. A bit-budget curve says coordination needs six bits per edge; it does not say
what those six bits are *about*. This module answers that, and the answer has to survive a null.

Everything here is measured on the transmitted symbol -- the integer index actually sent -- not on
the continuous pre-quantisation vector. What the receiver gets is the index; anything the index does
not carry did not cross the channel.

Four quantities are tested against the symbol, and the fourth is the reason the other three can be
believed:

    a_sr  the sender's own measurement of this edge -- the interference it is being told about
    a_ss  the sender's own direct gain
    lam   the commanded preference weight
    a_rs  what the *receiver* privately measures about the sender -- unobservable to the sender

Mutual information with a_rs must come out at the permutation null. It is the information-theoretic
form of the partial-information check in qa.py: if the sender's symbol knows something about a
quantity the sender cannot measure, the observation model has leaked and every other number on this
page is worthless.

On the estimator. Mutual information between a 2^B-valued symbol and a binned continuous quantity is
biased upward at finite sample size -- with 64 symbols and 16 bins there are 1024 cells, and noise
alone fills them unevenly. So every MI is reported beside the null obtained by shuffling the
quantity against the symbol, which measures that bias directly on the same data, and the excess
`I - I_null` is what gets interpreted. Raw MI is never quoted on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from agents import graph_inputs
from regime import AREA_M

RESULTS = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- estimators


def discrete_mi(sym: np.ndarray, x: np.ndarray, bins: int = 16) -> float:
    """I(symbol ; quantile-binned x) in bits."""
    sym = np.asarray(sym).ravel()
    x = np.asarray(x).ravel()
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    xb = np.searchsorted(edges, x)
    joint = np.histogram2d(sym, xb, bins=(sym.max() + 1, bins))[0]
    joint = joint / joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def mi_with_null(sym: np.ndarray, x: np.ndarray, bins: int = 16, n_perm: int = 8,
                 rng: np.random.Generator | None = None) -> dict:
    """MI, the permutation null, and the excess. Only the excess is interpretable."""
    rng = rng or np.random.default_rng(0)
    obs = discrete_mi(sym, x, bins)
    null = [discrete_mi(sym, rng.permutation(np.asarray(x).ravel()), bins) for _ in range(n_perm)]
    return {"mi_bits": obs, "null_bits": float(np.mean(null)), "null_std": float(np.std(null)),
            "excess_bits": obs - float(np.mean(null))}


def symbol_entropy(sym: np.ndarray) -> dict:
    """How many bits the channel is really carrying, against how many it is allowed to carry."""
    sym = np.asarray(sym).ravel()
    counts = np.bincount(sym)
    p = counts[counts > 0] / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    return {"entropy_bits": h, "codewords_used": int((counts > 0).sum()),
            "codewords_available": int(counts.size)}


def variance_explained(sym: np.ndarray, x: np.ndarray) -> float:
    """
    Fraction of the variance of x explained by symbol identity (eta^2).

    This is also exactly the R^2 of the best possible decoder that sees only the symbol, so it
    answers "how well can a receiver reconstruct this quantity from the message" without training
    a probe to find out.
    """
    sym, x = np.asarray(sym).ravel(), np.asarray(x).ravel()
    grand = x.mean()
    between = 0.0
    for s in np.unique(sym):
        m = sym == s
        between += m.sum() * (x[m].mean() - grand) ** 2
    total = ((x - grand) ** 2).sum()
    return float(between / max(total, 1e-30))


def interval_purity(sym: np.ndarray, x: np.ndarray, max_leaves: int | None = None) -> float:
    """
    Is the emergent vocabulary *ordinal* -- can the code be written as a set of intervals on x?

    Near 1.0 means the agents have discovered a scalar quantiser: a learned interference price with
    an ordering, which is a specification a receiver could reason about. Near chance means the code
    is a lookup table with no order to it -- still a protocol, but a much less interpretable one.

    Measured as the accuracy of the best interval partition of x into as many pieces as there are
    symbols in use, which is what a depth-unlimited 1-D decision tree finds. An earlier version cut
    at the midpoints between symbol means, which is not the optimal partition and topped out at
    0.92 on a code that was a quantiser by construction -- it understated ordinality by ~8 points
    and would have done so silently.

    This is a fit statistic, not a generalisation one: the question is whether the code *is*
    expressible in intervals, so it is scored on the data it is fitted to, deliberately.
    """
    from sklearn.tree import DecisionTreeClassifier

    sym, x = np.asarray(sym).ravel(), np.asarray(x).ravel()
    k = max_leaves or len(np.unique(sym))
    if k < 2:
        return 1.0
    tree = DecisionTreeClassifier(max_leaf_nodes=k, random_state=0).fit(x.reshape(-1, 1), sym)
    return float(tree.score(x.reshape(-1, 1), sym))


# --------------------------------------------------------------------------- collection


@torch.no_grad()
def collect(net, pool, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), max_instances: int = 2048) -> dict:
    """
    Run the trained protocol and pair every transmitted symbol with what the sender knew.

    Only off-diagonal edges are kept -- an agent does not signal to itself -- and the pairing is
    exact: entry [b, r, s] of the symbol tensor is what sender s put on the wire for receiver r, so
    it is matched with a_sr, the gain s measured for that same edge.

    Collection sweeps lambda rather than fixing it. With lambda held at a single value, I(m ; lam)
    is zero by construction and the column would report "the protocol does not encode the
    preference" when nothing had been asked. The commanded preference has to vary across the sample
    before the question means anything.
    """
    net.eval()
    m = min(max_instances, len(pool))
    g_obs = pool.gains_obs[:m]
    n = pool.n_pairs
    off = ~torch.eye(n, dtype=torch.bool, device=g_obs.device)
    idx = off.expand(m, n, n)

    direct = torch.diagonal(g_obs, dim1=-2, dim2=-1)                 # (M, N)
    a_sr = g_obs.transpose(1, 2)                                     # [b, r, s] = a_{s,r}
    a_rs = g_obs                                                     # [b, r, s] = a_{r,s}
    sender_direct = direct[:, None, :].expand(m, n, n)
    recv_direct = direct[:, :, None].expand(m, n, n)
    take = lambda t: t[idx].cpu().numpy()

    cols = {k: [] for k in ("symbol", "a_sr", "a_rs", "sender_direct", "recv_direct", "lam")}
    for lam_val in lambdas:
        lam = torch.full((m,), float(lam_val), device=g_obs.device)
        node, edge = graph_inputs(g_obs, lam, norm=getattr(net, "norm", None))
        _, syms = net(node, edge, return_symbols=True)
        if not syms or syms[0].numel() == 0:
            net.train()
            return {}
        cols["symbol"].append(syms[0][idx].cpu().numpy())
        cols["a_sr"].append(np.log10(take(a_sr) + 1e-30))
        cols["a_rs"].append(np.log10(take(a_rs) + 1e-30))
        cols["sender_direct"].append(np.log10(take(sender_direct) + 1e-30))
        cols["recv_direct"].append(np.log10(take(recv_direct) + 1e-30))
        cols["lam"].append(np.full(int(idx.sum()), float(lam_val)))
    net.train()
    return {k: np.concatenate(v) for k, v in cols.items()}


def analyse(net, pool, bits: int, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0) -> dict:
    """The full abstraction report for one trained protocol."""
    rng = np.random.default_rng(seed)
    d = collect(net, pool, lambdas)
    if not d:
        return {"bits": bits, "note": "silent protocol -- nothing transmitted"}

    sym = d["symbol"]
    out = {"bits": bits, "lambdas": list(lambdas), "n_edges": int(sym.size)}
    out.update(symbol_entropy(sym))

    out["mutual_information"] = {}
    for k in ("a_sr", "sender_direct", "recv_direct", "lam", "a_rs"):
        out["mutual_information"][k] = mi_with_null(sym, d[k], rng=rng)

    out["variance_explained"] = {k: variance_explained(sym, d[k])
                                 for k in ("a_sr", "sender_direct", "a_rs")}
    out["interval_purity_a_sr"] = interval_purity(sym, d["a_sr"])

    # The invariant: a B-bit symbol cannot carry more than B bits about anything.
    worst = max(v["mi_bits"] for v in out["mutual_information"].values())
    out["budget_respected"] = bool(worst <= bits + 1e-6)
    out["entropy_within_budget"] = bool(out["entropy_bits"] <= bits + 1e-6)
    return out


def codebook_geometry(net, n_clusters: int = 4) -> dict:
    """
    Does the learned codebook have structure, or is it 2^B unrelated points?

    Cluster the codeword vectors and report the silhouette. A vocabulary organised into a few
    families is the compositional structure the abstraction literature predicts; a flat cloud is a
    lookup table. Reported either way -- this is a measurement, not a claim.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    cb = getattr(net.channel, "codebook", None)
    if cb is None:
        return {}
    x = cb.detach().cpu().numpy()
    if len(x) < n_clusters * 2:
        return {"n_codewords": int(len(x)), "note": "too few codewords to cluster"}
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(x)
    return {"n_codewords": int(len(x)), "n_clusters": n_clusters,
            "silhouette": float(silhouette_score(x, labels)),
            "cluster_sizes": np.bincount(labels).tolist()}


# --------------------------------------------------------------------------- experiment


def abstraction_sweep(bits_list=(1, 2, 3, 4, 6, 8), seeds=(0,), steps: int = 8000,
                      n_pairs: int = 8, tag: str = "analysis") -> list[dict]:
    """Train at each budget and report what the resulting vocabulary encodes."""
    from dataset import build_pool
    from train import Config, evaluate, train

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=2048, n_pairs=n_pairs, area_m=AREA_M, seed=999, lambdas=(0.0, 0.5, 1.0))
    out = []
    for bits in bits_list:
        for seed in seeds:
            cfg = Config(bits=bits, steps=steps, seed=seed, usage_bonus=0.2)
            net = train(cfg, tr)
            row = analyse(net, te, bits, seed=seed)
            row["seed"] = seed
            row["mean_ratio"] = evaluate(net, cfg, te)["mean_ratio"]
            row["codebook"] = codebook_geometry(net)
            out.append(row)
            mi = row["mutual_information"]
            print(f"  B={bits} seed {seed}: H={row['entropy_bits']:.2f}b  "
                  f"I(m;a_sr)={mi['a_sr']['excess_bits']:.3f}b  "
                  f"I(m;a_rs)={mi['a_rs']['excess_bits']:.3f}b (must be ~0)  "
                  f"purity={row['interval_purity_a_sr']:.3f}  ratio={row['mean_ratio']:.4f}",
                  flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    summarise(out)
    return out


def summarise(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(f"{'B':>3} {'H(m)':>7} {'used':>9} {'I(m;a_sr)':>11} {'I(m;a_ss)':>11} "
          f"{'I(m;lam)':>10} {'I(m;a_rs)':>11} {'purity':>8} {'R2':>7}")
    print("-" * 92)
    for r in rows:
        if "mutual_information" not in r:
            continue
        mi = r["mutual_information"]
        print(f"{r['bits']:>3} {r['entropy_bits']:>7.2f} "
              f"{r['codewords_used']:>4}/{r['codewords_available']:<4} "
              f"{mi['a_sr']['excess_bits']:>11.3f} {mi['sender_direct']['excess_bits']:>11.3f} "
              f"{mi['lam']['excess_bits']:>10.3f} {mi['a_rs']['excess_bits']:>11.3f} "
              f"{r['interval_purity_a_sr']:>8.3f} {r['variance_explained']['a_sr']:>7.3f}")
    print("=" * 92)
    print("I(m;a_rs) is the receiver's private measurement: it must sit at the null (~0).")
    print("purity: fraction of edges whose symbol matches a scalar-threshold rule on a_sr.")


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. The estimator must recover a known quantity. A symbol that is a deterministic function of
    #    x carries exactly H(symbol) bits about it; an independent symbol carries none.
    x = rng.normal(size=20000)
    q = np.searchsorted(np.quantile(x, [0.25, 0.5, 0.75]), x)      # 2 bits, deterministic in x
    print(f"deterministic 2-bit code: I={discrete_mi(q, x):.3f} bits (expect ~2.0)")
    indep = rng.integers(0, 4, size=20000)
    r = mi_with_null(indep, x, rng=rng)
    print(f"independent code:         I={r['mi_bits']:.3f}  null={r['null_bits']:.3f}  "
          f"excess={r['excess_bits']:+.4f} (expect ~0)")
    assert abs(discrete_mi(q, x) - 2.0) < 0.05 and abs(r["excess_bits"]) < 0.02

    # 2. Interval purity is 1.0 for a quantiser and at chance for a shuffled code.
    print(f"quantiser purity: {interval_purity(q, x):.3f} (expect 1.0)")
    print(f"shuffled  purity: {interval_purity(rng.permutation(q), x):.3f} (expect ~0.25)")
    assert interval_purity(q, x) > 0.99

    # 3. Variance explained matches R^2 of the symbol-mean decoder.
    print(f"variance explained by the 2-bit code: {variance_explained(q, x):.3f}")

    # 4. End to end on a briefly trained protocol.
    from dataset import build_pool
    from train import Config, train

    tr = build_pool(size=1024, n_pairs=6, area_m=AREA_M, seed=0)
    te = build_pool(size=512, n_pairs=6, area_m=AREA_M, seed=999, lambdas=(0.5,))
    cfg = Config(bits=4, steps=800, seed=0, usage_bonus=0.2)
    net = train(cfg, tr)
    rep = analyse(net, te, bits=4)
    mi = rep["mutual_information"]
    print(f"\ntrained B=4: H={rep['entropy_bits']:.2f} bits, "
          f"{rep['codewords_used']}/{rep['codewords_available']} codewords")
    for k, v in mi.items():
        print(f"  I(m; {k:14s}) = {v['mi_bits']:.3f}  null {v['null_bits']:.3f}  "
              f"excess {v['excess_bits']:+.3f}")
    print("  budget respected:", rep["budget_respected"], " entropy within budget:",
          rep["entropy_within_budget"])
    print("  codebook:", codebook_geometry(net))
    assert rep["budget_respected"] and rep["entropy_within_budget"]
