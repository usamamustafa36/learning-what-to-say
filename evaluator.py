"""
The evaluation layer, ported from `prior/llm-d2d-resource-allocation/main.py`.

The twelve-metric evaluator with paired t-tests is the best thing in the prior repo and it is worth
carrying forward. It is ported here rather than re-invented, generalised from two users to N, and
audited on the way in -- because five of the twelve metrics do not measure anything.

    >>> ev = PerformanceEvaluator(Pc=1.0)
    >>> a = ev.evaluate(P, h, N0, W, Pc, opt_time)     # prior repo
    >>> a['reliability'], a['network_iq']
    (0.9, 85.0)                                        # for every P, every h, always

`reliability_index()`, `adaptability_score()` and `network_iq()` take only keyword defaults and
never read the allocation or the channel. `sustainability` is `renewable/(total+renewable)` with
`renewable = 0.25*total`, which is 0.2 by construction. `user_satisfaction` asks whether SE >= 0.5
and EE >= 0.5, which every allocation in the study satisfies, so it is 1.0 throughout.

That is not a small bookkeeping point. The prior repo's stored comparison runs a t-test on all
twelve, and the five degenerate ones return p = 1.0 or None; the headline "Neural Network Wins: 3,
Temporal Wins: 9" counts those ties as Temporal wins. A metric that cannot vary cannot be won.

So: seven metrics survive, five are dropped with the reason recorded in DROPPED below, and one is
repaired -- user satisfaction is re-pointed at the task floors in tasks.py, where the requirement is
something an allocation can actually fail.

Two further repairs, both to things that would otherwise be inherited silently:

1.  `composite_score` in the prior repo is a weighted sum of *unnormalised* metrics, so it is
    dominated by whichever has the largest numerical scale. In the stored results
    `computational_efficiency` has mean 77,734 against a spectral efficiency of 6.0, and the
    composite is therefore a speed ranking with the physics as rounding error. Here every term is
    normalised against a reference before weighting.

2.  Those stored results do not come from the code that ships beside them. The current
    `comp_eff()` is bounded by roughly SE + EE ~ 20 and its own comment says it "keeps the metric
    in a reasonable range (0-10)"; the stored values are in the tens of thousands, from an earlier
    quality/time form. `dual_optimization_results/` is dated 2025-08-16, `main.py` 2026-01-01. The
    bounded form is the one ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from metrics import energy_efficiency, jain_fairness, spectral_efficiency
from regime import CIRCUIT_POWER_W

# Metric -> why it is not carried forward. Kept in code so the audit travels with the port.
DROPPED = {
    "sustainability": "renewable = 0.25*total_power makes it 0.2 for every allocation",
    "reliability": "reliability_index() returns (0.9+0.85+0.95)/3 from keyword defaults; "
                   "reads neither powers nor channel",
    "adaptability": "adaptability_score() returns (0.8+0.75)/2 from keyword defaults",
    "network_iq": "network_iq() returns (0.8+0.85+0.9)/3*100 from keyword defaults",
    "user_satisfaction (as written)": "thresholds SE >= 0.5 and EE >= 0.5 are met by every "
                                      "allocation in the study, so the metric is 1.0 throughout; "
                                      "repaired here against per-agent task floors instead",
}

MEASURED = (
    "spectral_efficiency",
    "energy_efficiency",
    "jain_fairness",
    "efficiency",
    "green_efficiency",
    "computational_efficiency",
    "user_satisfaction",
)

# Composite weights, renormalised from the prior repo's after dropping the dead terms.
WEIGHTS = {
    "spectral_efficiency": 0.25,
    "energy_efficiency": 0.20,
    "green_efficiency": 0.15,
    "efficiency": 0.15,
    "jain_fairness": 0.10,
    "computational_efficiency": 0.10,
    "user_satisfaction": 0.05,
}


@dataclass
class References:
    """
    Per-metric scales, so the composite adds comparable quantities.

    Fitted from a reference arm -- normally the centralised oracle -- and then frozen, for the same
    reason agents.Normaliser is frozen: a scale computed over the arms being compared makes each
    arm's score depend on which other arms happen to be in the table.
    """

    values: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def fit(rows: list[dict]) -> "References":
        keys = [k for k in MEASURED if k in rows[0]]
        return References({k: max(float(np.mean([r[k] for r in rows])), 1e-12) for k in keys})

    def normalise(self, row: dict) -> dict:
        return {k: row[k] / self.values.get(k, 1.0) for k in row if k in self.values}


def computational_efficiency(opt_time_s: float, quality: float) -> float:
    """
    Quality discounted by the time it took to produce it -- the prior repo's bounded form.

    The unbounded quality/time version it replaced is what put 77,734 in the stored results and made
    the composite a stopwatch. Bounded, a method cannot buy the ranking with speed alone.
    """
    time_ms = float(opt_time_s) * 1000.0
    return float(quality) * (1.0 - np.tanh(time_ms / 10.0) * 0.5)


def user_satisfaction(rates: np.ndarray, r_min: np.ndarray) -> float:
    """
    Fraction of agents meeting their own rate floor.

    The repaired metric. `rates` and `r_min` are per-agent, so a URLLC floor of 4 b/s/Hz is a real
    requirement an allocation can miss -- unlike the SE >= 0.5 test it replaces.
    """
    return float(np.mean(np.asarray(rates) >= np.asarray(r_min)))


def evaluate_allocation(
    powers: np.ndarray,
    gains: np.ndarray,
    noise_power: float,
    circuit_power_w: float = CIRCUIT_POWER_W,
    opt_time_s: float = 0.0,
    r_min: np.ndarray | None = None,
) -> dict[str, float]:
    """Every surviving metric for one allocation. N agents, not two."""
    powers = np.asarray(powers, dtype=float)
    se = float(spectral_efficiency(powers, gains, noise_power))
    ee = float(energy_efficiency(powers, gains, noise_power, circuit_power_w))
    n = powers.shape[-1]
    total_power = float(powers.sum()) + circuit_power_w * n

    row = {
        "spectral_efficiency": se,
        "energy_efficiency": ee,
        "jain_fairness": float(jain_fairness(powers, gains, noise_power)),
        "efficiency": 0.5 * (se + ee),
        "green_efficiency": (se + ee) / (0.5 * total_power + 1e-10),
        "computational_efficiency": computational_efficiency(opt_time_s, se + ee),
        "sum_power": float(powers.sum()),
    }
    if r_min is not None:
        from metrics import sinr

        rates = np.log2(1.0 + sinr(powers, gains, noise_power))
        row["user_satisfaction"] = user_satisfaction(rates, r_min)
    return row


def composite_score(row: dict, refs: References) -> float:
    """Weighted sum over normalised metrics. Terms absent from `row` are dropped and reweighted."""
    norm = refs.normalise(row)
    used = {k: w for k, w in WEIGHTS.items() if k in norm}
    total_w = sum(used.values())
    return float(sum(w * norm[k] for k, w in used.items()) / max(total_w, 1e-12))


# --------------------------------------------------------------------------- comparison


@dataclass
class Comparison:
    metric: str
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    t_statistic: float | None
    p_value: float | None
    verdict: str          # "a" | "b" | "tie" | "degenerate"

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05


def compare(rows_a: list[dict], rows_b: list[dict], name_a="A", name_b="B",
            paired: bool = True) -> list[Comparison]:
    """
    Per-metric statistical comparison of two arms.

    Two departures from the prior repo, both of which it got wrong:

    * A metric with no variance in either arm is reported as `degenerate`, not awarded to one side.
      The prior code assigned every such tie to `temporal` and counted it in the win total.
    * The test is paired by default. Both arms are run on the *same* channel scenarios, so an
      independent-samples t-test throws away the pairing and inflates the variance it is testing
      against. `ttest_ind` was the wrong test for the design that produced the data.
    """
    out = []
    keys = [k for k in rows_a[0] if k in rows_b[0]]
    for k in keys:
        a = np.asarray([r[k] for r in rows_a], dtype=float)
        b = np.asarray([r[k] for r in rows_b], dtype=float)
        degenerate = a.std() < 1e-12 and b.std() < 1e-12
        if degenerate:
            t = p = None
            verdict = "degenerate"
        else:
            if paired and len(a) == len(b):
                t, p = stats.ttest_rel(a, b)
            else:
                t, p = stats.ttest_ind(a, b, equal_var=False)
            t, p = float(t), float(p)
            verdict = ("a" if a.mean() > b.mean() else "b") if p < 0.05 else "tie"
        out.append(Comparison(k, float(a.mean()), float(a.std()), float(b.mean()), float(b.std()),
                              t, p, verdict))
    return out


def report(comps: list[Comparison], name_a="A", name_b="B") -> str:
    """The prior repo's comparison table, with an honest win count underneath it."""
    w = max(len(c.metric) for c in comps) + 2
    lines = ["=" * (w + 58),
             f"{'metric':<{w}}{name_a:>16}{name_b:>16}{'p':>10}{'winner':>12}",
             "-" * (w + 58)]
    for c in comps:
        p = "  n/a" if c.p_value is None else f"{c.p_value:.4f}"
        win = {"a": name_a, "b": name_b, "tie": "tie", "degenerate": "no variance"}[c.verdict]
        lines.append(f"{c.metric:<{w}}{c.mean_a:>10.4f}+-{c.std_a:<4.2f}"
                     f"{c.mean_b:>10.4f}+-{c.std_b:<4.2f}{p:>10}{win:>12}")
    wins_a = sum(c.verdict == "a" for c in comps)
    wins_b = sum(c.verdict == "b" for c in comps)
    ties = sum(c.verdict == "tie" for c in comps)
    dead = sum(c.verdict == "degenerate" for c in comps)
    lines.append("=" * (w + 58))
    lines.append(f"  {name_a}: {wins_a} significant wins   {name_b}: {wins_b}   "
                 f"ties: {ties}   metrics with no variance (not counted): {dead}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from channel import InterferenceChannel

    rng = np.random.default_rng(0)
    ch = InterferenceChannel(n_pairs=6, rng=rng)
    g, n0 = ch.gains(), ch.noise_power

    # 1. Every surviving metric must actually respond to the allocation. This is the test the
    #    ported evaluator exists to pass and the original fails.
    rows = []
    for _ in range(64):
        p = rng.uniform(1e-3, 0.1, size=6)
        rows.append(evaluate_allocation(p, g, n0, 0.1, opt_time_s=rng.uniform(1e-4, 1.0),
                                        r_min=np.full(6, 1.0)))
    print(f"{'metric':<28}{'min':>12}{'max':>12}   varies?")
    dead = []
    for k in rows[0]:
        v = [r[k] for r in rows]
        varies = max(v) - min(v) > 1e-9
        if not varies:
            dead.append(k)
        print(f"{k:<28}{min(v):>12.4f}{max(v):>12.4f}   {'yes' if varies else 'NO'}")
    assert not dead, f"ported a constant metric: {dead}"

    # 2. Dropped metrics stay dropped, with reasons attached.
    print(f"\ndropped {len(DROPPED)} metrics from the prior evaluator:")
    for k, why in DROPPED.items():
        print(f"  {k:<32} {why}")

    # 3. The composite must not be buyable with speed. Same allocation, 10000x faster.
    refs = References.fit(rows)
    p = np.full(6, 0.05)
    slow = evaluate_allocation(p, g, n0, 0.1, opt_time_s=1.0, r_min=np.full(6, 1.0))
    fast = evaluate_allocation(p, g, n0, 0.1, opt_time_s=1e-4, r_min=np.full(6, 1.0))
    cs, cf = composite_score(slow, refs), composite_score(fast, refs)
    print(f"\ncomposite: slow {cs:.4f} -> fast {cf:.4f}  "
          f"({100*(cf-cs)/cs:+.1f}% for a 10,000x speedup, bounded by design)")
    assert cf < 2.0 * cs, "speed still dominates the composite"

    # 4. A metric with no variance must be reported as degenerate, never won.
    a = [{"real": float(x), "constant": 0.9} for x in rng.normal(1.0, 0.2, 40)]
    b = [{"real": float(x), "constant": 0.9} for x in rng.normal(1.3, 0.2, 40)]
    comps = compare(a, b, "left", "right")
    print("\n" + report(comps, "left", "right"))
    by = {c.metric: c for c in comps}
    assert by["constant"].verdict == "degenerate" and by["real"].verdict == "b"
    print("\nconstant metric is not awarded to either arm:", by["constant"].verdict)
