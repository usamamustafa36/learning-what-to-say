"""
Neuro-symbolic distillation: turning a learned code into a specification.

A hand-designed protocol has a specification, so a receiver can reject a malformed message. A
learned protocol does not, and that is the security hole this project is aimed at: with no spec,
there is nothing to check an incoming symbol against. Distillation is the repair. If the emergent
code can be rewritten as a short rule over quantities the sender can measure, then the rule *is* the
missing specification, and a receiver can hold senders to it.

Three questions, in the order that matters:

1.  **Fidelity.** How often does a depth-limited rule reproduce the neural encoder's symbol?
2.  **Performance.** Does the system still work when the rule *replaces* the encoder? Fidelity
    alone proves nothing -- a rule can agree 90% of the time and disagree exactly where it counts,
    so the distilled encoder is swapped in and scored end to end against the same oracle.
3.  **Detection.** Given the rule, can a receiver tell an honest sender from a lying one?

The features the rule is allowed to use are exactly the sender's own observations -- its measurement
of this edge, its own direct gain, the total interference it suffers, and the commanded lambda.
Nothing else is available to it, which is what makes the resulting spec implementable at a real
transmitter rather than a description written with hindsight.

Depth is capped deliberately low. A tree deep enough to memorise the encoder would score perfect
fidelity and be no more a specification than the network was; the point is a rule a person can read
and a receiver can check cheaply.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from agents import graph_inputs
from regime import AREA_M

RESULTS = Path(__file__).parent / "results"
FEATURES = ("a_sr", "own_direct", "own_interference", "lambda")


def edge_features(node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """
    (B, N, N, 4) of sender-observable quantities, aligned with the symbol tensor's [b, r, s].

    Built from the same standardised tensors the network itself consumes, so the rule is fitted on
    the sender's actual observation and cannot smuggle in anything the sender lacks.
    """
    b, n, _ = node.shape
    sender = node.unsqueeze(1).expand(b, n, n, node.shape[-1])       # [b, r, s] -> sender s
    return torch.cat([edge[..., :1], sender[..., 0:1], sender[..., 1:2], sender[..., 2:3]], dim=-1)


@torch.no_grad()
def gather(net, pool, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), max_instances: int = 2048):
    """Sender features and emitted symbols for every off-diagonal edge, over a sweep of lambda."""
    net.eval()
    m = min(max_instances, len(pool))
    g_obs = pool.gains_obs[:m]
    n = pool.n_pairs
    off = ~torch.eye(n, dtype=torch.bool, device=g_obs.device)
    idx = off.expand(m, n, n)

    xs, ys = [], []
    for lam_val in lambdas:
        lam = torch.full((m,), float(lam_val), device=g_obs.device)
        node, edge = graph_inputs(g_obs, lam, norm=getattr(net, "norm", None))
        _, syms = net(node, edge, return_symbols=True)
        if not syms or syms[0].numel() == 0:
            net.train()
            return None, None
        feats = edge_features(node, edge)
        xs.append(feats[idx].cpu().numpy())
        ys.append(syms[0][idx].cpu().numpy())
    net.train()
    return np.concatenate(xs), np.concatenate(ys)


class SymbolicEncoder:
    """
    A decision tree standing in for the learned message encoder.

    Callable as a `symbol_fn` for ProtocolGNN.forward, so the distilled rule can be dropped into the
    real forward pass and measured rather than described.
    """

    def __init__(self, tree, device) -> None:
        self.tree, self.device = tree, device

    def __call__(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        b, n, _ = node.shape
        feats = edge_features(node, edge).reshape(-1, len(FEATURES)).cpu().numpy()
        sym = self.tree.predict(feats)
        return torch.as_tensor(sym, device=self.device, dtype=torch.long).view(b, n, n)

    def rules(self, max_chars: int = 4000) -> str:
        from sklearn.tree import export_text

        return export_text(self.tree, feature_names=list(FEATURES), max_depth=10)[:max_chars]


def distil(net, pool, max_depth: int = 4, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0):
    """Fit the rule, held out honestly: fitted on one half of the edges, scored on the other."""
    from sklearn.tree import DecisionTreeClassifier

    x, y = gather(net, pool, lambdas)
    if x is None:
        return None, {"note": "silent protocol -- nothing to distil"}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    cut = len(x) // 2
    tr, te = perm[:cut], perm[cut:]

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(x[tr], y[tr])
    enc = SymbolicEncoder(tree, pool.gains.device)
    # Chance is the majority-class rate: a code collapsed onto one symbol is trivially "faithful".
    counts = np.bincount(y)
    return enc, {
        "max_depth": max_depth,
        "fidelity_train": float(tree.score(x[tr], y[tr])),
        "fidelity_held_out": float(tree.score(x[te], y[te])),
        "majority_class_rate": float(counts.max() / counts.sum()),
        "n_leaves": int(tree.get_n_leaves()),
        "feature_importance": dict(zip(FEATURES, [float(v) for v in tree.feature_importances_])),
    }


@torch.no_grad()
def evaluate_symbolic(net, cfg, pool, enc: SymbolicEncoder) -> dict:
    """
    Score the system end to end with the distilled rule in place of the learned encoder.

    Mirrors train.evaluate() exactly apart from the substitution, so the two numbers are comparable.
    """
    from env import ee_torch, se_torch

    net.eval()
    out = {}
    for lam_val, oracle_val in pool.oracle.items():
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        node, edge = graph_inputs(pool.gains_obs, lam, norm=getattr(net, "norm", None))
        p = net(node, edge, symbol_fn=enc)
        se = se_torch(p, pool.gains, pool.noise_power) / pool.se_ref.clamp_min(1e-12)
        ee = ee_torch(p, pool.gains, pool.noise_power, cfg.circuit_power_w) / pool.ee_ref.clamp_min(1e-12)
        obj = lam * se + (1.0 - lam) * ee
        out[lam_val] = float((obj / oracle_val.clamp_min(1e-12)).mean())
    net.train()
    return {"per_lambda": out, "mean_ratio": float(np.mean(list(out.values())))}


# --------------------------------------------------------------------------- validation


class MessageValidator:
    """
    The thing a learned protocol cannot otherwise have: a receiver-side check on incoming symbols.

    A receiver cannot verify a single symbol -- it does not know the sender's channel, which is the
    whole reason the message exists. What it *can* do is hold a sender to the spec in aggregate: the
    distilled rule implies how often each symbol should appear, and a sender whose symbol histogram
    departs from that over a window is either broken or lying.

    A chi-square goodness-of-fit test over a window of `window` symbols, with the expected
    distribution taken from the rule on honest traffic. This is deliberately weak -- it detects a
    shifted distribution, not a single well-chosen lie -- and that limitation is the finding, not a
    defect to hide: a bit-budgeted emergent protocol gives a receiver very little to check against.
    """

    MIN_EXPECTED = 5.0          # Cochran's rule of thumb for a chi-square cell
    CELLS_PER_CODE = 10         # target expected count per codeword when choosing a window

    def __init__(self, expected: np.ndarray, window: int | None = None,
                 alpha: float = 0.01) -> None:
        self.expected = expected / max(expected.sum(), 1e-12)
        self.alpha = alpha
        self.window = window if window is not None else self.min_window(len(expected))

    @classmethod
    def min_window(cls, n_codes: int) -> int:
        """
        Shortest window at which the test can actually run for this alphabet.

        This is not a tuning knob, it is a correctness condition, and getting it wrong produced the
        most dangerous kind of bug in this project: a 64-symbol window over a 64-codeword alphabet
        gives expected counts near one, fewer than two cells clear Cochran's threshold, and the
        detector returned "not flagged" for *every* sender. Detection and false-alarm rates both
        went to zero and the result read as though a crude attack had become undetectable, when in
        fact no test had been performed. A detector that cannot run must say so, never return a
        negative.
        """
        return max(64, cls.CELLS_PER_CODE * int(n_codes))

    @staticmethod
    def fit(symbols: np.ndarray, n_codes: int, **kw) -> "MessageValidator":
        counts = np.bincount(np.asarray(symbols).ravel(), minlength=n_codes).astype(float)
        return MessageValidator(counts + 1e-9, **kw)

    def usable_cells(self, n_samples: int) -> int:
        """How many chi-square cells this many samples can support under the fitted marginal."""
        return int((self.expected * n_samples > self.MIN_EXPECTED).sum())

    def flag(self, symbols: np.ndarray) -> bool:
        """True if this window of symbols is inconsistent with the specification."""
        from scipy import stats

        obs = np.bincount(np.asarray(symbols).ravel(), minlength=len(self.expected)).astype(float)
        exp = self.expected * obs.sum()
        keep = exp > self.MIN_EXPECTED
        if keep.sum() < 2:
            raise ValueError(
                f"chi-square is underpowered: {int(keep.sum())} usable cells from "
                f"{int(obs.sum())} symbols over {len(self.expected)} codewords. "
                f"Use window >= {self.min_window(len(self.expected))}."
            )
        chi = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum())
        p = 1.0 - stats.chi2.cdf(chi, df=int(keep.sum()) - 1)
        return bool(p < self.alpha)

    def false_alarm_rate(self, honest: np.ndarray, n_windows: int = 200,
                         rng: np.random.Generator | None = None) -> float:
        """Flag rate on traffic known to be honest -- the number the detector must be judged with."""
        rng = rng or np.random.default_rng(0)
        s = np.asarray(honest).ravel()
        return float(np.mean([self.flag(rng.choice(s, self.window)) for _ in range(n_windows)]))


# --------------------------------------------------------------------------- experiment


def distillation_sweep(bits_list=(2, 3, 4, 6), depths=(2, 3, 4, 6), steps: int = 8000,
                       n_pairs: int = 8, seed: int = 0, tag: str = "symbolic") -> list[dict]:
    """Fidelity and, more importantly, retained performance, across budgets and rule depths."""
    from dataset import build_pool
    from train import Config, evaluate, train

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=2048, n_pairs=n_pairs, area_m=AREA_M, seed=999,
                    lambdas=(0.0, 0.25, 0.5, 0.75, 1.0))
    out = []
    for bits in bits_list:
        cfg = Config(bits=bits, steps=steps, seed=seed, usage_bonus=0.2)
        net = train(cfg, tr)
        neural = evaluate(net, cfg, te)["mean_ratio"]
        for depth in depths:
            enc, info = distil(net, te, max_depth=depth, seed=seed)
            if enc is None:
                continue
            sym_ratio = evaluate_symbolic(net, cfg, te, enc)["mean_ratio"]
            info.update({"bits": bits, "seed": seed, "neural_ratio": neural,
                         "symbolic_ratio": sym_ratio,
                         "retained": sym_ratio / max(neural, 1e-9)})
            out.append(info)
            print(f"  B={bits} depth={depth}: fidelity {info['fidelity_held_out']:.3f} "
                  f"(chance {info['majority_class_rate']:.3f})  "
                  f"neural {neural:.4f} -> symbolic {sym_ratio:.4f} "
                  f"({100*info['retained']:.1f}% retained)", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import numpy as np

    from dataset import build_pool
    from train import Config, evaluate, train

    torch.manual_seed(0)
    tr = build_pool(size=1024, n_pairs=6, area_m=AREA_M, seed=0)
    te = build_pool(size=512, n_pairs=6, area_m=AREA_M, seed=999, lambdas=(0.0, 0.5, 1.0))
    cfg = Config(bits=4, steps=1200, seed=0, usage_bonus=0.2)
    net = train(cfg, tr)
    neural = evaluate(net, cfg, te)["mean_ratio"]

    # 1. Fidelity must beat the majority-class rate, or the "rule" is just naming the common symbol.
    print(f"neural protocol: {neural:.4f} of oracle")
    for depth in (2, 4, 6):
        enc, info = distil(net, te, max_depth=depth)
        ratio = evaluate_symbolic(net, cfg, te, enc)["mean_ratio"]
        print(f"  depth {depth}: fidelity {info['fidelity_held_out']:.3f} "
              f"(chance {info['majority_class_rate']:.3f}, {info['n_leaves']} leaves)  "
              f"symbolic {ratio:.4f} = {100*ratio/neural:.1f}% of neural")
        assert info["fidelity_held_out"] > info["majority_class_rate"]

    # 2. The rule must be readable, and must use only sender-observable features.
    enc, info = distil(net, te, max_depth=3)
    print("\n  feature importance:", {k: round(v, 3) for k, v in info["feature_importance"].items()})
    print("  " + "\n  ".join(enc.rules(600).splitlines()[:10]))

    # 3. The validator must almost never flag honest traffic, or its detections mean nothing.
    x, y = gather(net, te)
    val = MessageValidator.fit(y, n_codes=1 << cfg.bits, alpha=0.01)
    print(f"  alphabet {1 << cfg.bits} codewords -> minimum usable window {val.window} symbols")
    fa = val.false_alarm_rate(y)
    print(f"\n  validator false-alarm rate on honest traffic: {fa:.3f} (alpha = 0.01)")
    assert fa < 0.10

    # 4. ...and must flag a sender that has shifted its symbol distribution.
    rng = np.random.default_rng(0)
    liar = rng.choice(np.unique(y)[: max(2, len(np.unique(y)) // 4)], 64)   # a narrowed vocabulary
    print(f"  flags a sender using a quarter of the vocabulary: {val.flag(liar)}")
    assert val.flag(liar)
