"""
The bit-budget sweep: Figure 2 of the paper.

Two reference lines bound everything:

    silent (B=0)  -- the floor. Message passing with nothing said.
    continuous    -- the ceiling. Unquantised real-valued messages.

Between them, the question: how many bits per edge does coordination actually need, and does a
*learned* B-bit symbol beat B bits spent on classical quantised CSI feedback at the same budget?

Every cell is repeated over seeds and reported as mean +/- std with a paired t-test against the
matching quantised-CSI cell.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from analysis import abstraction_sweep
from dataset import build_pool
from intent import intent_experiment
from llm_agent import llm_experiment, llm_rendering_sweep
from oran import deployment_report, summarise as summarise_oran
from pareto import analyse as pareto_analyse
from prior_methods import prior_arms, temporal_experiment
from sensing import sensing_sweep
from symbolic import distillation_sweep
from train import Config, run_one
from regime import AREA_M, LAMBDAS

RESULTS = Path(__file__).parent / "results"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sweep(
    bits_list=(0, 1, 2, 3, 4, 6, 8),
    seeds=(0, 1, 2),
    n_pairs=8,
    area_m=AREA_M,
    rounds=1,
    steps=8000,
    train_size=8192,
    test_size=2048,
    usage_bonus=0.2,
    rho=None,
    tag="bitsweep",
) -> list[dict]:
    print(f"building pools (N={n_pairs}, area={area_m}m)...", flush=True)
    t0 = time.time()
    kw = {} if rho is None else {"rho": rho}
    tr = build_pool(size=train_size, n_pairs=n_pairs, area_m=area_m, seed=0, **kw)
    te = build_pool(size=test_size, n_pairs=n_pairs, area_m=area_m, seed=999, lambdas=LAMBDAS, **kw)
    print(f"  pools ready in {time.time()-t0:.0f}s\n", flush=True)

    results = []

    # Ceiling: continuous messages, no budget.
    for seed in seeds:
        cfg = Config(bits=0, mode="continuous", rounds=rounds, steps=steps, seed=seed,
                     usage_bonus=usage_bonus)
        r = run_one(cfg, tr, te)
        r["arm"], r["bits"] = "continuous", None
        r["rho"] = te.rho
        results.append(r)
        print(f"  continuous  seed {seed}: {r['mean_ratio']:.4f}", flush=True)

    for bits in bits_list:
        for messenger in ("learned", "quantised"):
            if bits == 0 and messenger == "quantised":
                continue                      # identical to the silent floor
            for seed in seeds:
                cfg = Config(
                    bits=bits, mode="vq", messenger=messenger,
                    rounds=rounds, steps=steps, seed=seed, usage_bonus=usage_bonus,
                )
                r = run_one(cfg, tr, te)
                r["arm"], r["bits"] = messenger, bits
                results.append(r)
            got = [x["mean_ratio"] for x in results[-len(seeds):]]
            print(f"  B={bits} {messenger:9s}: {np.mean(got):.4f} +/- {np.std(got):.4f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(results, indent=2))
    return results


def summarise(results: list[dict]) -> None:
    def cell(arm, bits):
        return [r["mean_ratio"] for r in results if r["arm"] == arm and r["bits"] == bits]

    cont = cell("continuous", None)
    silent = cell("learned", 0)
    print("\n" + "=" * 78)
    print(f"{'budget':>8} {'learned':>18} {'quantised CSI':>18} {'gap':>9} {'p':>9}")
    print("-" * 78)
    print(f"{'silent':>8} {np.mean(silent):>10.4f} +/-{np.std(silent):.3f} {'':>18}")

    bits_seen = sorted({r["bits"] for r in results if r["bits"] not in (None, 0)})
    for b in bits_seen:
        le, qu = cell("learned", b), cell("quantised", b)
        if not le or not qu:
            continue
        t, p = stats.ttest_rel(le, qu)
        print(f"{b:>8} {np.mean(le):>10.4f} +/-{np.std(le):.3f} "
              f"{np.mean(qu):>10.4f} +/-{np.std(qu):.3f} "
              f"{np.mean(le)-np.mean(qu):>+9.4f} {p:>9.4f}")
    print(f"{'cont.':>8} {np.mean(cont):>10.4f} +/-{np.std(cont):.3f}")
    print("=" * 78)

    floor, ceil = np.mean(silent), np.mean(cont)
    print(f"\nwindow attributable to messaging: {floor:.4f} -> {ceil:.4f} "
          f"({100*(ceil-floor):.1f} points)")
    for b in bits_seen:
        le = cell("learned", b)
        if le:
            frac = (np.mean(le) - floor) / max(ceil - floor, 1e-9)
            print(f"  B={b}: recovers {100*frac:5.1f}% of that window")


def rho_sweep(rhos=(0.0, 0.5, 0.9, 0.99), bits=6, seeds=(0, 1, 2), steps=8000, tag="rho_sweep"):
    """
    How much is a stale measurement worth?

    The policy observes slot t-1 and is judged on slot t. At rho = 1 the observation is perfect and
    the problem is static; at rho = 0 it is noise. This is the axis the original repo claimed to
    exploit while training on i.i.d. data, so it is measured here rather than asserted.
    """
    out = []
    for rho in rhos:
        tr = build_pool(size=8192, n_pairs=8, area_m=AREA_M, seed=0, rho=rho)
        te = build_pool(size=2048, n_pairs=8, area_m=AREA_M, seed=999, lambdas=LAMBDAS, rho=rho)
        for arm_bits in (0, bits):
            got = []
            for seed in seeds:
                cfg = Config(bits=arm_bits, steps=steps, seed=seed, usage_bonus=0.2)
                r = run_one(cfg, tr, te)
                r["arm"], r["bits"], r["rho"] = "learned", arm_bits, rho
                out.append(r); got.append(r["mean_ratio"])
            print(f"  rho={rho:.2f} B={arm_bits}: {np.mean(got):.4f} +/- {np.std(got):.4f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


def prior_experiment(n_pairs: int = 8, tag: str = "prior") -> list[dict]:
    """The prior repo's centralised allocator, ported, against the message-passing arms."""
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=2048, n_pairs=n_pairs, area_m=AREA_M, seed=999, lambdas=lambdas)
    rows = prior_arms(tr, te, epochs=400)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(rows, indent=2))
    return rows


def tasks_experiment(bits_list=(0, 2, 4, 6), steps: int = 8000, n_pairs: int = 8,
                     seed: int = 0, tag: str = "tasks") -> list[dict]:
    """
    Heterogeneous tasks: does the budget buy task satisfaction, or only throughput?

    The task-oriented arm has its own oracle (see train.evaluate_tasks) because the SE/EE optimum is
    not the task optimum. Hard success rate is reported alongside the smooth objective, and the two
    are allowed to disagree -- qa.py has a check that fails if the ratio improves while success does
    not, which is the failure mode that would make "task-oriented" a misnomer.
    """
    from train import Config, evaluate_tasks, train

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=1024, n_pairs=n_pairs, area_m=AREA_M, seed=999, lambdas=LAMBDAS)
    out = []
    for bits in bits_list:
        cfg = Config(bits=bits, steps=steps, seed=seed, usage_bonus=0.2, use_tasks=True)
        r = evaluate_tasks(train(cfg, tr), cfg, te)
        r["bits"], r["seed"] = bits, seed
        out.append(r)
        succ = float(np.mean(list(r["task_success"].values())))
        print(f"  B={bits}: task objective {r['mean_ratio']:.4f} of task oracle, "
              f"hard success {succ:.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


# Every experiment in the paper, by name. `python3 experiments.py <name>` runs one.
EXPERIMENTS = {
    "bitsweep": lambda a: summarise(sweep(seeds=tuple(range(a.seeds)), steps=a.steps,
                                          n_pairs=a.n_pairs, tag=a.tag)),
    "rho": lambda a: rho_sweep(seeds=tuple(range(a.seeds)), steps=a.steps),
    "sensing": lambda a: sensing_sweep(seeds=tuple(range(a.seeds)), steps=a.steps,
                                       n_pairs=a.n_pairs),
    "abstraction": lambda a: abstraction_sweep(steps=a.steps, n_pairs=a.n_pairs),
    "symbolic": lambda a: distillation_sweep(steps=a.steps, n_pairs=a.n_pairs),
    "intent": lambda a: intent_experiment(steps=a.steps, n_pairs=a.n_pairs),
    "llm": lambda a: llm_experiment(n_pairs=a.n_pairs),
    "llm_renderings": lambda a: llm_rendering_sweep(n_pairs=a.n_pairs),
    "oran": lambda a: summarise_oran(deployment_report(n_agents=a.n_pairs)),
    "temporal": lambda a: temporal_experiment(n_pairs=a.n_pairs),
    "prior": lambda a: prior_experiment(n_pairs=a.n_pairs),
    "tasks": lambda a: tasks_experiment(steps=a.steps, n_pairs=a.n_pairs),
    "pareto": lambda a: pareto_analyse(),
    "adversarial": lambda a: __import__("adversarial").adversarial_sweep(steps=a.steps,
                                                                        n_pairs=a.n_pairs),
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="?", default="bitsweep", choices=sorted(EXPERIMENTS))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--tag", default="bitsweep")
    ap.add_argument("--rho-sweep", action="store_true")     # kept: older scripts call it this way
    a = ap.parse_args()
    EXPERIMENTS["rho" if a.rho_sweep else a.experiment](a)
