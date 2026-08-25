"""
Intent-based networking: a declared outcome, compiled onto the knob the protocol already has.

An intent is a statement of what the network should achieve -- "every URLLC link clears 3 b/s/Hz,
and spend as little energy as possible doing it" -- with no instruction about how. Something has to
turn that into a control action. Here the actuator already exists: the policy is conditioned on a
preference weight lambda at inference time, so one trained model spans the whole trade-off and the
compiler's job is to find the operating point that satisfies the intent most cheaply.

That is the substantive claim this module tests, and it is worth stating precisely, because it is
the thing preference-conditioning buys that a fixed-weight policy cannot:

    a new intent is served by *searching lambda at run time*, not by retraining.

`train.py` samples lambda per instance during training for exactly this reason. If the trained
policy could only serve the operating point it was trained at, every new SLO would mean a new
training run -- minutes to hours in the non-RT RIC, against milliseconds for a lambda change.

Two things are checked rather than assumed:

**Monotonicity.** Bisection on lambda is only valid if satisfaction increases with it. Whether it
does is an empirical property of a trained policy, not a theorem, so it is measured on a grid first
and the compiler falls back to grid search when the measurement says the assumption fails.

**Whether the intent is achievable at all.** An SLO can simply exceed what the channel allows at
this SNR. The compiler reports `satisfied: False` with the best achievable value rather than
returning the closest lambda and letting a table imply success.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from agents import graph_inputs
from env import ee_torch, se_torch
from tasks import per_link_rate
from regime import AREA_M, CIRCUIT_POWER_W

RESULTS = Path(__file__).parent / "results"


@dataclass
class Intent:
    """
    A declarative service-level objective.

    `min_rate_bps_hz` is the floor every agent's link must clear; `min_satisfied` is the fraction of
    agents that must clear it. `minimise` names what to spend as little of as possible once the
    guarantee is met -- which is what makes this an optimisation rather than a threshold.
    """

    name: str
    min_rate_bps_hz: float = 2.0
    min_satisfied: float = 0.9
    minimise: str = "energy"                  # "energy" | "power"
    description: str = ""


@dataclass
class IntentResult:
    intent: dict
    satisfied: bool
    lambda_star: float | None
    achieved_satisfaction: float
    achieved_se: float
    achieved_ee: float
    mean_power_w: float
    method: str
    grid: list = field(default_factory=list)


@torch.no_grad()
def measure(net, pool, lam_val: float, circuit_power_w: float = CIRCUIT_POWER_W,
            extra=None) -> dict:
    """Run the policy at one commanded preference and report what the network actually did."""
    net.eval()
    m = len(pool)
    lam = torch.full((m,), float(lam_val), device=pool.gains.device)
    node, edge = graph_inputs(pool.gains_obs, lam, extra_node=extra, norm=getattr(net, "norm", None))
    p = net(node, edge)
    rate = per_link_rate(p, pool.gains, pool.noise_power)
    net.train()
    return {
        "lambda": float(lam_val),
        "rate_mean": float(rate.mean()),
        "se": float(se_torch(p, pool.gains, pool.noise_power).mean()),
        "ee": float(ee_torch(p, pool.gains, pool.noise_power, circuit_power_w).mean()),
        "mean_power_w": float(p.mean()),
        "rates": rate,
    }


def satisfaction(rate: torch.Tensor, min_rate: float) -> float:
    """Fraction of agent-instances clearing the floor."""
    return float((rate >= min_rate).float().mean())


def sweep(net, pool, intent: Intent, n_grid: int = 11, **kw) -> list[dict]:
    """Satisfaction and cost across the whole preference axis -- the compiler's map."""
    out = []
    for lam_val in np.linspace(0.0, 1.0, n_grid):
        r = measure(net, pool, float(lam_val), **kw)
        out.append({"lambda": r["lambda"], "satisfaction": satisfaction(r["rates"], intent.min_rate_bps_hz),
                    "se": r["se"], "ee": r["ee"], "mean_power_w": r["mean_power_w"]})
    return out


def is_monotone(grid: list[dict], tol: float = 0.02) -> bool:
    """Does satisfaction rise with lambda? Bisection is only sound if it does."""
    s = [g["satisfaction"] for g in grid]
    drops = sum(max(0.0, s[i] - s[i + 1]) for i in range(len(s) - 1))
    return drops <= tol


def compile_intent(net, pool, intent: Intent, n_grid: int = 11, n_bisect: int = 8, **kw) -> IntentResult:
    """
    Find the cheapest lambda that satisfies the intent.

    Cheapest means smallest, because lambda weights throughput against energy efficiency: the least
    lambda that still meets the guarantee is the operating point that meets it with the least energy
    spent chasing rate beyond it.
    """
    grid = sweep(net, pool, intent, n_grid, **kw)
    best = max(g["satisfaction"] for g in grid)
    if best < intent.min_satisfied:
        worst = max(grid, key=lambda g: g["satisfaction"])
        return IntentResult(asdict(intent), False, None, best, worst["se"], worst["ee"],
                            worst["mean_power_w"], "infeasible", grid)

    if is_monotone(grid):
        lo, hi, method = 0.0, 1.0, "bisection"
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            r = measure(net, pool, mid, **kw)
            if satisfaction(r["rates"], intent.min_rate_bps_hz) >= intent.min_satisfied:
                hi = mid
            else:
                lo = mid
        lam_star = hi
    else:
        # Satisfaction is not monotone in lambda for this policy; take the smallest grid point that
        # satisfies the intent instead of trusting a bisection that could land anywhere.
        method = "grid (satisfaction not monotone in lambda)"
        ok = [g for g in grid if g["satisfaction"] >= intent.min_satisfied]
        lam_star = min(g["lambda"] for g in ok)

    final = measure(net, pool, lam_star, **kw)
    return IntentResult(asdict(intent), True, float(lam_star),
                        satisfaction(final["rates"], intent.min_rate_bps_hz),
                        final["se"], final["ee"], final["mean_power_w"], method, grid)


def calibrated_catalogue(net, pool, **kw) -> list[Intent]:
    """
    Build the intent set from what this network can actually deliver.

    Absolute rate floors written by hand do not survive a change of operating point: the first
    catalogue here used 1-4 b/s/Hz, which was reasonable at one deployment density and reported
    every intent infeasible at another. That tells you nothing about the compiler.

    So the floors are taken from the rate distribution the policy achieves at lambda = 1 -- its most
    throughput-hungry setting -- and the *coverage* asked of each floor is set from the same
    distribution, which is the part that is easy to get backwards. A floor at the q-th percentile
    can be cleared by at most (1 - q) of the links, by definition of a percentile. Asking that 75%
    of links clear the 40th-percentile rate is therefore not a demanding intent, it is an
    arithmetically impossible one, and the first version of this catalogue did exactly that and
    reported the compiler broken.

    Each intent asks for 90% of the headroom the percentile leaves, so it is satisfiable at lambda=1
    and tight enough that a lower lambda misses it. The last entry is deliberately beyond the
    ceiling, because a compiler that cannot say "no" is not a compiler.
    """
    r = measure(net, pool, 1.0, **kw)
    qs = (0.4, 0.6, 0.8)
    vals = torch.quantile(r["rates"].flatten().float(),
                          torch.tensor(qs, device=r["rates"].device))
    lo, mid, hi = (float(x) for x in vals)
    names = ("best-effort", "assured", "premium")
    out = [
        Intent(name, round(v, 2), round((1.0 - q) * 0.9, 2), "energy",
               f"{q:.0%}-percentile rate, asked of {(1-q)*0.9:.0%} of links "
               f"({(1-q):.0%} clear it at lambda=1)")
        for name, q, v in zip(names, qs, (lo, mid, hi))
    ]
    out.append(Intent("urllc-strict", round(hi * 2.0, 2), 0.95, "energy",
                      "deliberately beyond the ceiling -- the compiler must refuse this one"))
    return out


# Kept for reference: absolute floors, valid only at the regime they were written for.
CATALOGUE = [
    Intent("best-effort", 1.0, 0.90, "energy", "a low floor almost everything clears"),
    Intent("assured-2", 2.0, 0.90, "energy", "a moderate rate guarantee"),
    Intent("assured-3", 3.0, 0.80, "energy", "a firmer guarantee for most links"),
    Intent("urllc-strict", 4.0, 0.95, "energy", "close to what the old SNR regime allowed"),
]


def intent_experiment(bits: int = 6, steps: int = 8000, n_pairs: int = 8, seed: int = 0,
                      tag: str = "intent") -> list[dict]:
    """
    One trained policy, four intents, no retraining -- against the fixed-weight alternative.

    The comparison that matters is the last two columns: what a fixed lambda = 0.5 policy would have
    delivered for the same intent, and what it costs in power to meet the intent by simply running
    at lambda = 1.0 instead of compiling it.
    """
    from dataset import build_pool
    from train import Config, train

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=2048, n_pairs=n_pairs, area_m=AREA_M, seed=999, lambdas=(0.5,))
    cfg = Config(bits=bits, steps=steps, seed=seed, usage_bonus=0.2)
    net = train(cfg, tr)

    out = []
    fixed_half = measure(net, te, 0.5)
    fixed_one = measure(net, te, 1.0)
    for it in calibrated_catalogue(net, te):
        r = compile_intent(net, te, it)
        row = asdict(r)
        row["fixed_half_satisfaction"] = satisfaction(fixed_half["rates"], it.min_rate_bps_hz)
        row["fixed_one_satisfaction"] = satisfaction(fixed_one["rates"], it.min_rate_bps_hz)
        row["fixed_one_power_w"] = fixed_one["mean_power_w"]
        row["power_saved_vs_lambda_one"] = (
            1.0 - r.mean_power_w / max(fixed_one["mean_power_w"], 1e-12) if r.satisfied else 0.0
        )
        row["bits"] = bits
        out.append(row)
        status = "satisfied" if r.satisfied else "INFEASIBLE"
        lam = "n/a" if r.lambda_star is None else f"{r.lambda_star:.3f}"
        print(f"  {it.name:14s} floor {it.min_rate_bps_hz:.1f} b/s/Hz x {it.min_satisfied:.0%}: "
              f"{status:11s} lambda*={lam:>5}  achieved {r.achieved_satisfaction:.3f}  "
              f"power {r.mean_power_w*1e3:5.1f} mW  "
              f"({100*row['power_saved_vs_lambda_one']:+.1f}% vs lambda=1)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    return out


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from dataset import build_pool
    from train import Config, train

    torch.manual_seed(0)
    tr = build_pool(size=1024, n_pairs=6, area_m=AREA_M, seed=0)
    te = build_pool(size=512, n_pairs=6, area_m=AREA_M, seed=999, lambdas=(0.5,))
    cfg = Config(bits=4, steps=1200, seed=0, usage_bonus=0.2)
    net = train(cfg, tr)

    # 1. The knob must do something. If satisfaction and power do not move with lambda, there is no
    #    actuator here and the rest of the module is theatre.
    it = Intent("probe", 2.0, 0.9)
    grid = sweep(net, te, it, n_grid=6)
    print(f"{'lambda':>8}{'satisfaction':>14}{'SE':>9}{'EE':>9}{'power mW':>10}")
    for g in grid:
        print(f"{g['lambda']:>8.2f}{g['satisfaction']:>14.3f}{g['se']:>9.2f}{g['ee']:>9.2f}"
              f"{g['mean_power_w']*1e3:>10.1f}")
    spread = max(g["satisfaction"] for g in grid) - min(g["satisfaction"] for g in grid)
    print(f"\nsatisfaction spans {spread:.3f} across lambda; monotone: {is_monotone(grid)}")
    assert spread > 0.01, "lambda does not move satisfaction -- no actuator"

    # 2. Compile each intent and check the compiler's own verdict against the grid it measured.
    print()
    catalogue = calibrated_catalogue(net, te)
    satisfied_any = False
    for it in catalogue:
        r = compile_intent(net, te, it, n_grid=6)
        lam = "n/a" if r.lambda_star is None else f"{r.lambda_star:.3f}"
        print(f"  {it.name:14s}: {'satisfied' if r.satisfied else 'INFEASIBLE':11s} "
              f"lambda*={lam:>5}  achieved {r.achieved_satisfaction:.3f}  "
              f"({r.method})")
        if r.satisfied:
            satisfied_any = True
            assert r.achieved_satisfaction >= it.min_satisfied - 0.05, \
                "compiler claimed satisfaction it did not deliver"
        else:
            assert r.achieved_satisfaction < it.min_satisfied
    assert satisfied_any, "no intent was satisfiable -- the catalogue is not calibrated to this policy"

    # 3. An impossible intent must be reported infeasible, not silently approximated.
    absurd = Intent("impossible", 50.0, 0.99)
    r = compile_intent(net, te, absurd, n_grid=6)
    print(f"\n  a 50 b/s/Hz floor: satisfied={r.satisfied}, best achievable "
          f"{r.achieved_satisfaction:.3f}, method '{r.method}'")
    assert not r.satisfied
