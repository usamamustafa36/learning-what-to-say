"""
The prior repo's two optimisers, ported to N agents and finally given an honest test.

`prior/llm-d2d-resource-allocation/main.py` contains two methods. Both are carried forward here as
baseline arms, because both answer questions this paper has to answer anyway -- and because the
claims made for them are checkable, and were never checked.

**EnhancedNeuralNetworkOptimizer** -> `SupervisedAllocator`. A feedforward net regressing the
channel matrix onto powers, trained on labels from a multi-start solver. Its reported result was
"100% of optimal", which is circular: the labels and the yardstick were the same solver, so 100%
says the regression converged, not that the allocation is good. Scored here against the same genie
oracle every other arm is scored against, it stops being circular and becomes the arm this paper
actually needs: **centralised, full-CSI, no communication**. That is the correct reference for
"what does message passing buy over collecting everything at one point", and the prior repo is
where it comes from.

**TemporalOptimizer** -> `TrendPredictor`. Extrapolates the channel by the mean first difference of
the last five slots, then optimises on the extrapolation. Its reported +58% SE/EE advantage was
measured on i.i.d. channels -- `h = uniform(0.7, 1.0)` redrawn every sample -- so the quantity it
claims to exploit was absent from the data. Here the channel is Gauss-Markov with a Jakes-derived
rho, so temporal correlation exists and the claim can be tested.

It is worth writing down what the test should find, before running it. For a first-order
Gauss-Markov process the conditional mean of h[t] given the history is rho * h[t-1] -- the last
sample, shrunk. It depends on h[t-1] alone; earlier samples add nothing. A linear trend over five
slots therefore estimates a slope that does not exist and adds its estimation noise to the
prediction. Extrapolation should be *worse* than simply reusing the last measurement, at every rho.
`PersistencePredictor` is that control, and it is the one comparison the prior repo never ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from env import BatchChannel, ee_torch, se_torch
from solvers import maximize_batch, oracle_batch, reference_values_batch
from regime import AREA_M, CIRCUIT_POWER_W, P_MAX_W

FAST = dict(n_starts=8, n_steps=300)


# --------------------------------------------------------------- centralised supervised allocator


class SupervisedAllocator(nn.Module):
    """
    Port of the prior repo's network: full CSI in, powers out, trained by regression on oracle
    labels.

    Two generalisations were needed. It takes N(N-1) interference gains plus N direct gains rather
    than the four gains of a two-user channel, and it is conditioned on the preference weight
    lambda, which the prior version fixed at 0.5/0.5 and therefore could not sweep. Architecture,
    depth, dropout and the training recipe are otherwise the original's.

    What it is *not*: decentralised. It consumes the entire gain matrix, which is exactly the
    quantity the protocol under study is trying to avoid transporting. It belongs in the table as a
    reference line, not as a competitor.
    """

    def __init__(self, n_agents: int, p_max: float, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.n, self.p_max = n_agents, p_max
        # Frozen input statistics, fitted once on the training pool. Registered as buffers so they
        # travel with the model and evaluation cannot silently restandardise on test data.
        self.register_buffer("x_mean", torch.tensor(-9.0))
        self.register_buffer("x_std", torch.tensor(1.0))
        self.net = nn.Sequential(
            nn.Linear(n_agents * n_agents + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_agents),
        )

    def forward(self, gains: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        b = gains.shape[0]
        x = torch.log10(gains.reshape(b, -1) + 1e-30)
        x = (x - self.x_mean) / self.x_std
        x = torch.cat([x, lam[:, None]], dim=-1)
        return self.p_max * torch.sigmoid(self.net(x))


def make_labels(pool, lambdas, p_max: float, circuit_power_w: float, **kw) -> tuple:
    """Oracle powers for every (instance, lambda) pair -- the supervised targets."""
    xs_g, xs_l, ys = [], [], []
    for lam_val in lambdas:
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        with torch.no_grad():
            p = oracle_batch(pool.gains, pool.noise_power, p_max, lam,
                             pool.se_ref, pool.ee_ref, circuit_power_w, **kw)
        xs_g.append(pool.gains_obs); xs_l.append(lam); ys.append(p)
    return torch.cat(xs_g), torch.cat(xs_l), torch.cat(ys)


def train_supervised(pool, p_max=P_MAX_W, circuit_power_w=CIRCUIT_POWER_W, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
                     epochs: int = 300, batch: int = 256, lr: float = 1e-3, seed: int = 0,
                     loss: str = "mse", verbose: bool = False) -> SupervisedAllocator:
    """
    Train the centralised allocator, either the prior repo's way or the honest way.

    `loss="mse"` is the port: regress onto oracle powers, which is what produced the "100% of
    optimal" number. `loss="objective"` is the control that has to accompany it -- same network,
    same full CSI, same everything, but maximising the objective directly instead of imitating a
    solution to it. Without that control, a poor result for the ported method cannot be attributed:
    it might be the imitation, or it might be that this architecture cannot do the job at all.

    Imitation has a specific failure mode here that the two-user case hides. At N agents the
    scalarised objective has many near-equivalent optima -- permutations and near-ties in which
    subset of agents is switched off -- so the label is multi-modal in the input. Least squares
    against a multi-modal target returns the conditional *mean*, which is an average of several good
    allocations and is typically not a good allocation itself.

    Note which channel goes in: `gains_obs`, the measurement from slot t-1, while the label is the
    optimum for slot t. Feeding it `gains` would hand this arm the realised channel and no other
    arm gets that, which would turn a reference line into a genie.
    """
    torch.manual_seed(seed)
    dev = pool.gains.device
    net = SupervisedAllocator(pool.n_pairs, p_max).to(dev)
    lg = torch.log10(pool.gains_obs + 1e-30)
    net.x_mean.fill_(float(lg.mean())); net.x_std.fill_(float(lg.std().clamp_min(1e-6)))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if loss == "mse":
        g, lam, y = make_labels(pool, lambdas, p_max, circuit_power_w)
        m = g.shape[0]
    else:
        m = len(pool)

    gen = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        idx = torch.randperm(m, device=dev, generator=gen)
        total = 0.0
        for k in range(0, m, batch):
            j = idx[k : k + batch]
            if loss == "mse":
                l = nn.functional.mse_loss(net(g[j], lam[j]) / p_max, y[j] / p_max)
            else:
                lam_j = torch.rand(len(j), device=dev, generator=gen)
                p = net(pool.gains_obs[j], lam_j)
                se = se_torch(p, pool.gains[j], pool.noise_power) / pool.se_ref[j].clamp_min(1e-12)
                ee = ee_torch(p, pool.gains[j], pool.noise_power, circuit_power_w) / pool.ee_ref[j].clamp_min(1e-12)
                l = -(lam_j * se + (1.0 - lam_j) * ee).mean()
            opt.zero_grad(set_to_none=True); l.backward(); opt.step()
            total += float(l) * len(j)
        sched.step()
        if verbose and ep % 100 == 0:
            print(f"    epoch {ep:4d}  {loss} {total/m:.6f}", flush=True)
    return net


@torch.no_grad()
def evaluate_supervised(net: SupervisedAllocator, pool, circuit_power_w: float = CIRCUIT_POWER_W) -> dict:
    """Ratio to the genie oracle, per lambda -- directly comparable with train.evaluate()."""
    net.eval()
    out = {}
    for lam_val, oracle_val in pool.oracle.items():
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        p = net(pool.gains_obs, lam)
        se = se_torch(p, pool.gains, pool.noise_power) / pool.se_ref.clamp_min(1e-12)
        ee = ee_torch(p, pool.gains, pool.noise_power, circuit_power_w) / pool.ee_ref.clamp_min(1e-12)
        obj = lam * se + (1.0 - lam) * ee
        out[lam_val] = float((obj / oracle_val.clamp_min(1e-12)).mean())
    net.train()
    return {"per_lambda": out, "mean_ratio": float(np.mean(list(out.values()))),
            "signalling_bits_per_agent": float("inf"),      # full CSI collected centrally
            "arm": "supervised-fullcsi"}


def prior_arms(train_pool, test_pool, epochs: int = 400, **kw) -> list[dict]:
    """Both centralised arms, ported and controlled, ready to sit in the bit-budget table."""
    out = []
    for loss, label in (("mse", "imitation (prior repo)"), ("objective", "direct objective")):
        net = train_supervised(train_pool, epochs=epochs, loss=loss, **kw)
        r = evaluate_supervised(net, test_pool)
        r["arm"], r["loss"], r["label"] = f"supervised-{loss}", loss, label
        out.append(r)
        print(f"  full-CSI centralised, {label:24s}: {r['mean_ratio']:.4f}", flush=True)
    return out


# ------------------------------------------------------------------------ channel predictors


@dataclass
class TrendPredictor:
    """
    The prior repo's `_predict_channel`: last sample plus the mean first difference of the last
    `window` samples, then `np.clip(predicted, 0.01, 1.0)`.

    That clip is doing more work than it looks. The prior repo's gains live in [0.01, 1], so
    clipping to [0.01, 1] confines the extrapolation to the range the channel actually occupies and
    silently repairs an overshoot. Here the gains are ~1e-9 and span decades, so a literal [0.01, 1]
    clip would send every prediction to the floor. The faithful port is the *relative* one: clip
    into the range the history has occupied, per entry, which is what [0.01, 1] meant there.

    Ported with that clip rather than without it, because the point of running this arm is to test
    the method, not to win against a crippled version of it.
    """

    window: int = 5
    name: str = "trend"
    clip: bool = True

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        recent = history[-self.window :]
        if recent.shape[0] < 2:
            return history[-1]
        trend = torch.diff(recent, dim=0).mean(dim=0)
        pred = recent[-1] + trend
        if self.clip:
            lo = history.min(dim=0).values
            hi = history.max(dim=0).values
            pred = torch.minimum(torch.maximum(pred, lo), hi)
        return pred.clamp_min(1e-30)


@dataclass
class PersistencePredictor:
    """Reuse the last measurement. The control the prior comparison omitted."""

    name: str = "persistence"

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        return history[-1]


@dataclass
class ShrunkPredictor:
    """
    rho^2 * h[t-1] + (1 - rho^2) * E[h], the conditional mean of the gain under Gauss-Markov fading.

    Gains are |fading|^2, so the correlation that carries over to power is rho^2, not rho. Included
    because if any predictor should beat persistence it is this one, and it costs nothing to check.
    """

    rho: float = 0.9
    name: str = "shrunk"

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        r2 = self.rho**2
        return r2 * history[-1] + (1.0 - r2) * history.mean(dim=0)


@dataclass
class GeniePredictor:
    """Sees the realised channel. Upper bound, not a method."""

    name: str = "genie"
    future: torch.Tensor | None = None

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        assert self.future is not None
        return self.future


# ------------------------------------------------------------------------ trajectory experiment


def trajectory(n_slots: int, batch: int, n_pairs: int, rho: float, area_m: float = AREA_M,
               seed: int = 0, device: str = "cuda") -> tuple[torch.Tensor, float]:
    """(T, B, N, N) gains from one Gauss-Markov run, plus the noise power."""
    ch = BatchChannel(batch=batch, n_pairs=n_pairs, area_m=area_m, rho=rho,
                      rng=np.random.default_rng(seed))
    dev = torch.device(device)
    slots = [torch.as_tensor(ch.gains(), dtype=torch.float32, device=dev)]
    for _ in range(n_slots - 1):
        slots.append(torch.as_tensor(ch.step(), dtype=torch.float32, device=dev))
    return torch.stack(slots), ch.noise_power


def temporal_experiment(rhos=(0.0, 0.5, 0.9, 0.99), n_slots: int = 10, batch: int = 128,
                        n_pairs: int = 8, lam_val: float = 0.5, p_max: float = P_MAX_W,
                        circuit_power_w: float = CIRCUIT_POWER_W, warmup: int = 5, seed: int = 0,
                        tag: str = "temporal") -> list[dict]:
    """
    Does extrapolating the channel beat reusing the last measurement?

    Every predictor is handed the same history, produces a channel estimate, and the *same* solver
    is run on that estimate; the resulting powers are then scored on the channel that actually
    occurred. Only the prediction differs, so any difference is the prediction's.
    """
    import json
    from pathlib import Path

    out = []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rho in rhos:
        gains, noise = trajectory(n_slots, batch, n_pairs, rho, seed=seed, device=str(dev))
        preds = [TrendPredictor(), PersistencePredictor(), ShrunkPredictor(rho=rho), GeniePredictor()]

        for t in range(warmup, n_slots):
            truth = gains[t]
            with torch.no_grad():
                se_ref, ee_ref = reference_values_batch(truth, noise, p_max, circuit_power_w, **FAST)
                lam = torch.full((batch,), lam_val, device=dev)
                p_or = oracle_batch(truth, noise, p_max, lam, se_ref, ee_ref, circuit_power_w, **FAST)

            def score(p):
                se = se_torch(p, truth, noise) / se_ref.clamp_min(1e-12)
                ee = ee_torch(p, truth, noise, circuit_power_w) / ee_ref.clamp_min(1e-12)
                return lam_val * se + (1.0 - lam_val) * ee

            best = score(p_or)
            for pr in preds:
                if isinstance(pr, GeniePredictor):
                    pr.future = truth
                est = pr(gains[:t])                     # history strictly before t
                with torch.no_grad():
                    def obj(p, est=est):
                        reps = p.shape[0] // batch
                        e = est.repeat(reps, 1, 1)
                        se = se_torch(p, e, noise) / se_ref.repeat(reps).clamp_min(1e-12)
                        ee = ee_torch(p, e, noise, circuit_power_w) / ee_ref.repeat(reps).clamp_min(1e-12)
                        return lam_val * se + (1.0 - lam_val) * ee

                    p_hat = maximize_batch(obj, batch, n_pairs, p_max, dev, **FAST)
                ratio = (score(p_hat) / best.clamp_min(1e-12))
                out.append({"rho": rho, "slot": t, "arm": pr.name,
                            "mean_ratio": float(ratio.mean()),
                            "per_instance": [float(x) for x in ratio.cpu()]})
            done = {r["arm"]: r["mean_ratio"] for r in out if r["rho"] == rho and r["slot"] == t}
            print(f"  rho={rho:.2f} slot {t}: " +
                  "  ".join(f"{k} {v:.4f}" for k, v in done.items()), flush=True)

    results = Path(__file__).parent / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{tag}.json").write_text(json.dumps(out, indent=2))
    summarise_temporal(out)
    return out


def summarise_temporal(rows: list[dict]) -> None:
    """The prior repo's claim, restated as a paired test against the control it never ran."""
    from evaluator import compare, report

    rhos = sorted({r["rho"] for r in rows})
    print("\n" + "=" * 78)
    print(f"{'rho':>6} {'trend':>10} {'persistence':>13} {'shrunk':>10} {'genie':>8} "
          f"{'trend - persist':>16}")
    print("-" * 78)
    for rho in rhos:
        m = {a: np.mean([r["mean_ratio"] for r in rows if r["rho"] == rho and r["arm"] == a])
             for a in ("trend", "persistence", "shrunk", "genie")}
        print(f"{rho:>6.2f} {m['trend']:>10.4f} {m['persistence']:>13.4f} {m['shrunk']:>10.4f} "
              f"{m['genie']:>8.4f} {m['trend']-m['persistence']:>+16.4f}")
    print("=" * 78)

    for rho in rhos:
        a = [{"objective_ratio": x} for r in rows if r["rho"] == rho and r["arm"] == "trend"
             for x in r["per_instance"]]
        b = [{"objective_ratio": x} for r in rows if r["rho"] == rho and r["arm"] == "persistence"
             for x in r["per_instance"]]
        c = compare(a, b, "trend", "persistence")[0]
        verdict = {"a": "trend wins", "b": "persistence wins", "tie": "no difference",
                   "degenerate": "no variance"}[c.verdict]
        print(f"  rho={rho:.2f}: {verdict:18s} p={c.p_value:.2e}" if c.p_value is not None
              else f"  rho={rho:.2f}: {verdict}")


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. The predictors run and produce sane channel estimates.
    g, noise = trajectory(8, 32, 6, rho=0.9, seed=1, device=str(dev))
    print("trajectory:", tuple(g.shape))
    for pr in (TrendPredictor(), TrendPredictor(clip=False, name="trend-noclip"),
               PersistencePredictor(), ShrunkPredictor(rho=0.9)):
        est = pr(g[:6])
        err = float((torch.log10(est + 1e-30) - torch.log10(g[6] + 1e-30)).abs().mean())
        print(f"  {pr.name:12s} mean |log10 error| vs the realised channel: {err:.4f}")

    # 2. Prediction error, in the units the claim was made in. If the trend helped, its error would
    #    be below persistence's; the ordering here is the whole argument, so it is printed at every
    #    rho rather than at one.
    print("\n  mean |log10 prediction error| by rho:")
    for rho in (0.0, 0.5, 0.9, 0.99):
        g, _ = trajectory(8, 64, 6, rho=rho, seed=2, device=str(dev))
        errs = {}
        for pr in (TrendPredictor(), TrendPredictor(clip=False, name="trend-noclip"),
                   PersistencePredictor(), ShrunkPredictor(rho=rho)):
            est = pr(g[:6])
            errs[pr.name] = float((torch.log10(est + 1e-30) - torch.log10(g[6] + 1e-30)).abs().mean())
        print(f"    rho={rho:.2f}: " + "  ".join(f"{k} {v:.4f}" for k, v in errs.items())
              + f"   trend better? {errs['trend'] < errs['persistence']}")

    # 3. The supervised allocator trains and stays inside the box.
    from dataset import build_pool

    tr = build_pool(size=256, n_pairs=6, area_m=AREA_M, seed=0, device=str(dev))
    te = build_pool(size=128, n_pairs=6, area_m=AREA_M, seed=999, lambdas=(0.0, 0.5, 1.0), device=str(dev))
    net = train_supervised(tr, epochs=40, lambdas=(0.0, 0.5, 1.0))
    r = evaluate_supervised(net, te)
    print(f"\n  supervised full-CSI allocator: {r['mean_ratio']:.4f} of oracle "
          f"(per lambda { {k: round(v,3) for k,v in r['per_lambda'].items()} })")
    assert 0.0 < r["mean_ratio"] <= 1.0 + 1e-6, "an arm above the oracle is a bug"
