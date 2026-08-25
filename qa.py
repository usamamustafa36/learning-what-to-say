"""
Continuous QA for the emergent-signalling codebase: technical and conceptual.

Run it after any change:

    python3 qa.py            # fast checks (~1 min)
    python3 qa.py --full     # adds training-dependent checks (~10 min)
    python3 qa.py --loop 900 # re-run every 15 minutes

Technical checks catch code that is wrong. Conceptual checks catch code that is right but does not
support the claim being made of it -- stale results, unmet objectives, evidence that predates the
fix it was supposed to survive. The second kind is what actually sinks papers, so it is checked
here rather than left to memory.

Severities: FAIL blocks, WARN is a known gap to be honest about, PASS is verified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from regime import AREA_M, CIRCUIT_POWER_W, P_MAX_W

HERE = Path(__file__).parent
RESULTS = HERE / "results"

MODULES = ["channel", "metrics", "baselines", "env", "solvers", "dataset", "agents", "tasks",
           "evaluator", "sensing", "analysis", "symbolic", "oran", "intent", "prior_methods",
           "adversarial", "regime", "pareto"]


@dataclass
class Check:
    name: str
    kind: str            # "technical" | "conceptual"
    status: str          # PASS | FAIL | WARN | SKIP
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, kind, status, detail="") -> None:
        self.checks.append(Check(name, kind, status, detail))
        icon = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn ", "SKIP": " skip "}[status]
        print(f"[{icon}] {name}" + (f"\n           {detail}" if detail else ""), flush=True)

    def summary(self) -> int:
        f = sum(c.status == "FAIL" for c in self.checks)
        w = sum(c.status == "WARN" for c in self.checks)
        p = sum(c.status == "PASS" for c in self.checks)
        print("\n" + "=" * 74)
        print(f"  {p} passed, {w} warnings, {f} failures")
        if f:
            print("\n  FAILURES:")
            for c in self.checks:
                if c.status == "FAIL":
                    print(f"    - {c.name}: {c.detail}")
        if w:
            print("\n  WARNINGS (known gaps, be honest about these):")
            for c in self.checks:
                if c.status == "WARN":
                    print(f"    - {c.name}: {c.detail}")
        print("=" * 74)
        return f


# ---------------------------------------------------------------- technical


def check_self_tests(rep: Report, full: bool) -> None:
    for m in MODULES:
        if not full and m in ("dataset", "solvers"):
            rep.add(f"self-test: {m}", "technical", "SKIP", "slow; run with --full")
            continue
        r = subprocess.run([sys.executable, f"{m}.py"], cwd=HERE, capture_output=True, timeout=1800)
        if r.returncode == 0:
            rep.add(f"self-test: {m}", "technical", "PASS")
        else:
            tail = r.stderr.decode().strip().splitlines()[-1:] or ["(no stderr)"]
            rep.add(f"self-test: {m}", "technical", "FAIL", tail[-1])


def check_physics(rep: Report) -> None:
    from baselines import dinkelbach, equal_power, wmmse
    from channel import InterferenceChannel
    from metrics import energy_efficiency, spectral_efficiency

    rng = np.random.default_rng(4)
    bad_se = bad_ee = 0
    for _ in range(10):
        ch = InterferenceChannel(n_pairs=6, rng=rng)
        g, n0 = ch.gains(), ch.noise_power
        pe = equal_power(g, P_MAX_W)
        if spectral_efficiency(wmmse(g, n0, P_MAX_W), g, n0) < spectral_efficiency(pe, g, n0) - 1e-9:
            bad_se += 1
        if energy_efficiency(dinkelbach(g, n0, P_MAX_W, CIRCUIT_POWER_W), g, n0, CIRCUIT_POWER_W) \
                < energy_efficiency(pe, g, n0, CIRCUIT_POWER_W) - 1e-9:
            bad_ee += 1
    rep.add("physics: WMMSE >= equal power (SE)", "technical",
            "PASS" if bad_se == 0 else "FAIL", f"{bad_se}/10 violations")
    rep.add("physics: Dinkelbach >= equal power (EE)", "technical",
            "PASS" if bad_ee == 0 else "FAIL", f"{bad_ee}/10 violations")


def check_gradients(rep: Report) -> None:
    from baselines import se_gradient
    from channel import InterferenceChannel
    from metrics import spectral_efficiency

    rng = np.random.default_rng(2)
    ch = InterferenceChannel(n_pairs=5, rng=rng)
    g, n0 = ch.gains(), ch.noise_power
    p0 = rng.uniform(1e-3, 0.1, size=5)
    ana, num, eps = se_gradient(p0, g, n0), np.zeros(5), 1e-9
    for k in range(5):
        e = np.zeros(5); e[k] = eps
        num[k] = (spectral_efficiency(p0 + e, g, n0) - spectral_efficiency(p0 - e, g, n0)) / (2 * eps)
    err = float(np.max(np.abs(ana - num) / (np.abs(num) + 1e-12)))
    rep.add("numerics: analytic vs finite-difference gradient", "technical",
            "PASS" if err < 1e-4 else "FAIL", f"max rel err {err:.2e}")


def check_bits_are_bits(rep: Report) -> None:
    from agents import ProtocolGNN, graph_inputs

    g = torch.rand(8, 5, 5) + 0.05
    node, edge = graph_inputs(g, torch.rand(8))
    net = ProtocolGNN(bits=4, p_max=P_MAX_W, mode="binary").eval()
    _, syms = net(node, edge, return_symbols=True)
    ok = bool(((syms[0] == 0) | (syms[0] == 1)).all())
    rep.add("discreteness: transmitted symbols are exactly 0/1", "technical",
            "PASS" if ok else "FAIL", "straight-through must not leak float residue")


def check_partial_information(rep: Report) -> None:
    """
    The sender must not be able to see what only the receiver can measure.

    Perturb a_{r,s} -- the harm the *receiver* suffers from someone else -- and confirm the message
    the sender emits does not move. If it does, the model has leaked receiver-side knowledge into
    the sender and the whole premise is void.
    """
    from agents import ProtocolGNN, graph_inputs

    torch.manual_seed(0)
    net = ProtocolGNN(bits=6, p_max=P_MAX_W).eval()
    g = torch.rand(4, 5, 5) + 0.05

    node, edge = graph_inputs(g, torch.full((4,), 0.5))
    _, s1 = net(node, edge, return_symbols=True)

    g2 = g.clone()
    g2[:, 0, 1] *= 5.0                       # what receiver 0 measures about 1 -- sender 1 cannot know
    node2, edge2 = graph_inputs(g2, torch.full((4,), 0.5))
    _, s2 = net(node2, edge2, return_symbols=True)

    # Message emitted BY sender 1 TO receivers other than 0 must be unchanged.
    others = [r for r in range(5) if r != 0]
    unchanged = bool((s1[0][:, others, 1] == s2[0][:, others, 1]).all())
    rep.add("information: sender cannot see receiver-only measurements", "conceptual",
            "PASS" if unchanged else "FAIL",
            "perturbing a_{r,s} must not change what s transmits elsewhere")


def check_temporal_is_live(rep: Report, full: bool) -> None:
    """rho must actually reach the data, or the temporal model is decoration."""
    if not full:
        rep.add("temporal: rho affects observation quality", "technical", "SKIP", "run with --full")
        return
    from dataset import build_pool

    def fading_corr(rho):
        q = build_pool(size=256, n_pairs=6, area_m=AREA_M, seed=1, rho=rho)
        now = torch.diagonal(q.gains, dim1=1, dim2=2).cpu().numpy()
        obs = torch.diagonal(q.gains_obs, dim1=1, dim2=2).cpu().numpy()
        scale = np.sqrt((obs * now).mean(axis=1, keepdims=True))
        return float(np.corrcoef((obs / scale).ravel(), (now / scale).ravel())[0, 1])

    hi, lo = fading_corr(0.99), fading_corr(0.0)
    ok = hi > 0.8 and lo < 0.3
    rep.add("temporal: rho affects observation quality", "technical",
            "PASS" if ok else "FAIL",
            f"fading corr {hi:.3f} at rho=0.99 vs {lo:.3f} at rho=0")


def check_determinism(rep: Report) -> None:
    from agents import ProtocolGNN, graph_inputs

    g = torch.rand(4, 5, 5) + 0.05
    node, edge = graph_inputs(g, torch.full((4,), 0.5))
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        outs.append(ProtocolGNN(bits=4, p_max=P_MAX_W).eval()(node, edge))
    rep.add("reproducibility: same seed gives same output", "technical",
            "PASS" if torch.allclose(outs[0], outs[1]) else "FAIL")


def check_finite(rep: Report) -> None:
    from agents import ProtocolGNN, QuantisedCSIGNN, graph_inputs

    g = torch.rand(6, 7, 7) + 1e-6
    node, edge = graph_inputs(g, torch.rand(6))
    bad = []
    for bits in (0, 1, 4, 8):
        for mode in ("vq", "binary", "continuous"):
            p = ProtocolGNN(bits=bits, p_max=P_MAX_W, mode=mode)(node, edge)
            if not torch.isfinite(p).all() or (p < 0).any() or (p > 0.1 + 1e-6).any():
                bad.append(f"{mode}/B={bits}")
    q = QuantisedCSIGNN(bits=4, p_max=P_MAX_W); q.fit_quantizer(edge)
    if not torch.isfinite(q(node, edge)).all():
        bad.append("quantised/B=4")
    rep.add("robustness: all arms give finite in-range powers", "technical",
            "PASS" if not bad else "FAIL", ", ".join(bad))


def check_oracle_invariant(rep: Report) -> None:
    """No decentralised policy may exceed the centralised oracle in any stored result."""
    files = list(RESULTS.glob("*.json"))
    if not files:
        rep.add("invariant: no result exceeds the oracle", "technical", "SKIP", "no results yet")
        return
    total = viol = 0
    for f in files:
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        for r in rows if isinstance(rows, list) else []:
            for v in (r.get("per_lambda") or {}).values():
                total += 1
                viol += int(v > 1.0 + 1e-6)
    rep.add("invariant: no result exceeds the oracle", "technical",
            "PASS" if viol == 0 else "FAIL", f"{viol} of {total} cells above 1.0")


def check_codebook_health(rep: Report, full: bool) -> None:
    """Collapse detector: a budget of B bits that transmits ~1 bit is not measuring what it claims."""
    if not full:
        rep.add("collapse: codebook entropy tracks the budget", "technical", "SKIP", "run with --full")
        return
    from agents import graph_inputs
    from dataset import build_pool
    from train import Config, train

    tr = build_pool(size=2048, n_pairs=8, area_m=AREA_M, seed=0)
    cfg = Config(bits=6, steps=2500, seed=0, usage_bonus=0.2)
    net = train(cfg, tr)
    net.eval()
    with torch.no_grad():
        node, edge = graph_inputs(tr.gains_obs[:512], torch.rand(512, device=tr.gains.device))
        _, syms = net(node, edge, return_symbols=True)
    off = ~torch.eye(8, dtype=torch.bool, device=syms[0].device)
    c = torch.bincount(syms[0][:, off].reshape(-1), minlength=64).float()
    p = c / c.sum()
    ent = float(-(p[p > 0] * p[p > 0].log2()).sum())
    ok = ent > 0.5 * 6
    rep.add("collapse: codebook entropy tracks the budget", "technical",
            "PASS" if ok else "FAIL",
            f"H={ent:.2f} bits of a 6-bit budget, {int((c>0).sum())}/64 codewords used")


def check_preference_tradeoff(rep: Report) -> None:
    """
    The configured regime must admit a trade-off at all, or "multi-objective" is a word.

    This is the root-cause check, and it is deliberately upstream of any policy: it asks the
    *oracle* what changes between lambda = 0 and lambda = 1. If the centralised optimum barely
    moves, then SE and EE are nearly the same objective in this operating point and no policy,
    however well trained, can trace a Pareto front -- there is no front to trace.

    The mechanism to watch is the circuit power. EE = SE / (sum_P + N*Pc), so when N*Pc dominates
    sum_P the denominator is almost constant and EE becomes proportional to SE. The two objectives
    then agree everywhere and lambda selects between identical things.
    """
    from dataset import build_pool
    from env import ee_torch, se_torch
    from solvers import oracle_batch

    pool = build_pool(size=192, n_pairs=8, area_m=AREA_M, seed=7)
    vals = {}
    for lam_val in (0.0, 1.0):
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        with torch.no_grad():
            p = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam, pool.se_ref, pool.ee_ref,
                             CIRCUIT_POWER_W, n_starts=8, n_steps=400)
        vals[lam_val] = (float(se_torch(p, pool.gains, pool.noise_power).mean()),
                         float(ee_torch(p, pool.gains, pool.noise_power, CIRCUIT_POWER_W).mean()))
    (se0, ee0), (se1, ee1) = vals[0.0], vals[1.0]
    d_se, d_ee = abs(se1 - se0) / max(se0, 1e-9), abs(ee1 - ee0) / max(ee0, 1e-9)
    ok = d_se > 0.10 and d_ee > 0.10
    rep.add("multi-objective: the regime admits a trade-off", "conceptual",
            "PASS" if ok else "FAIL",
            f"oracle moves {100*d_se:.1f}% in SE and {100*d_ee:.1f}% in EE from lambda 0 to 1"
            + ("" if ok else "  -- SE and EE are near-collinear here; circuit power dominates"))


def check_preference_response(rep: Report, full: bool) -> None:
    """
    A preference-conditioned policy must respond to the preference it is conditioned on.

    Separate from the check above and downstream of it: even where a trade-off exists, a policy can
    ignore lambda and settle on one compromise operating point. Then every per-lambda number in
    every results file is the same allocation reported five times, and the Pareto front and the
    hypervolume are both fictions. Measured as the spread in transmit power across the lambda axis,
    which is the control action itself rather than a derived score.
    """
    if not full:
        rep.add("multi-objective: the policy responds to lambda", "conceptual", "SKIP",
                "run with --full")
        return
    from agents import graph_inputs
    from dataset import build_pool
    from train import Config, train

    tr = build_pool(size=2048, n_pairs=8, area_m=AREA_M, seed=0)
    te = build_pool(size=512, n_pairs=8, area_m=AREA_M, seed=999)
    net = train(Config(bits=6, steps=2500, seed=0, usage_bonus=0.2), tr)
    net.eval()
    powers = []
    with torch.no_grad():
        for lam_val in (0.0, 0.5, 1.0):
            lam = torch.full((len(te),), lam_val, device=te.gains.device)
            node, edge = graph_inputs(te.gains_obs, lam, norm=getattr(net, "norm", None))
            powers.append(float(net(node, edge).mean()))
    spread = (max(powers) - min(powers)) / max(max(powers), 1e-12)
    ok = spread > 0.10
    rep.add("multi-objective: the policy responds to lambda", "conceptual",
            "PASS" if ok else "FAIL",
            f"mean transmit power moves {100*spread:.1f}% across lambda "
            f"({', '.join(f'{1e3*p:.1f}mW' for p in powers)})"
            + ("" if ok else "  -- lambda is an input the policy ignores; no Pareto front exists"))


def check_task_calibration(rep: Report, full: bool) -> None:
    """
    A task-oriented objective that does not improve task satisfaction is misnamed.

    Observed: the smooth utility rises with the bit budget while the hard success rate falls
    (0.195 -> 0.154 between B=0 and B=4). The satisfaction gate is soft enough that partial
    satisfaction plus energy saving beats meeting the requirement, so the objective is not really
    optimising the thing its name claims. Either the gate sharpens, or the requirement levels come
    down to what this SNR regime can deliver, or the paper must not call it task-oriented.
    """
    if not full:
        rep.add("tasks: hard success tracks the objective", "conceptual", "SKIP", "run with --full")
        return
    from dataset import build_pool
    from train import Config, evaluate_tasks, train

    tr = build_pool(size=1024, n_pairs=8, area_m=AREA_M, seed=0)
    te = build_pool(size=256, n_pairs=8, area_m=AREA_M, seed=999, lambdas=(0.5,))
    out = {}
    for bits in (0, 4):
        cfg = Config(bits=bits, steps=1500, seed=0, usage_bonus=0.2, use_tasks=True)
        r = evaluate_tasks(train(cfg, tr), cfg, te)
        out[bits] = (r["mean_ratio"], float(np.mean(list(r["task_success"].values()))))
    (r0, s0), (r4, s4) = out[0], out[4]
    ok = s4 >= s0 - 0.01
    rep.add("tasks: hard success tracks the objective", "conceptual",
            "PASS" if ok else "WARN",
            f"ratio {r0:.3f}->{r4:.3f} with bits, but hard success {s0:.3f}->{s4:.3f}"
            + ("" if ok else "  -- gate too soft or requirements too high for this SNR"))


# ---------------------------------------------------------------- conceptual

OBJECTIVES = {
    "O-RAN / edge / cloud domains": ("oran.py", True),
    "large language models (LLMs)": ("llm_agent.py", True),
    "intent-based networking": ("intent.py", True),
    "neuro-symbolic AI": ("symbolic.py", True),
    "resource-efficient protocols (bit budget)": ("agents.py", True),
    "distributed, task-oriented agents": ("tasks.py", True),
    "emergent semantic communication": ("tasks.py", True),
    "learning-based protocol design": ("agents.py", True),
    "multi-agent systems": ("agents.py", True),
    "AI-native network architectures": ("oran.py", True),
    "multi-agent communication": ("agents.py", True),
    "multi-objective optimization": ("pareto.py", True),
    "state abstraction": ("analysis.py", True),
    "new signaling mechanisms": ("agents.py", True),
    "scalable learning strategies": ("agents.py", True),
    "optimization approaches (WMMSE/Dinkelbach)": ("baselines.py", True),
    "generative AI": ("llm_agent.py", True),
    "multimodal sensing": ("sensing.py", True),
    "validation in 6G use cases": ("experiments.py", True),
}


def check_objective_coverage(rep: Report) -> None:
    """Every advertised objective must map to a module that exists AND is wired into the pipeline."""
    src = {p.name: p.read_text() for p in HERE.glob("*.py")}
    imported = set()
    for name, text in src.items():
        for other in src:
            base = other[:-3]
            if f"from {base} import" in text or f"import {base}" in text:
                if name != other:
                    imported.add(other)

    done = partial = missing = 0
    lines = []
    for obj, (mod, _) in OBJECTIVES.items():
        if mod not in src:
            missing += 1
            lines.append(f"MISSING  {obj:45s} -> {mod} not written")
        elif mod not in imported and mod not in ("experiments.py",):
            partial += 1
            lines.append(f"ORPHAN   {obj:45s} -> {mod} exists but nothing imports it")
        else:
            done += 1
    total = len(OBJECTIVES)
    status = "PASS" if missing == 0 and partial == 0 else "WARN"
    rep.add("objectives: advert coverage", "conceptual", status,
            f"{done}/{total} wired, {partial} orphaned, {missing} missing\n           "
            + "\n           ".join(lines))


def check_results_freshness(rep: Report) -> None:
    """Results computed before the code that produced them changed are not evidence."""
    # qa.py is not a dependency of any result -- editing the checker must not invalidate evidence.
    deps = [p for p in HERE.glob("*.py") if p.name != "qa.py"]
    code_mtime = max((p.stat().st_mtime for p in deps), default=0)
    stale = []
    for f in RESULTS.glob("*.json"):
        if f.name == "qa_report.json":
            continue
        if f.stat().st_mtime < code_mtime:
            age = (code_mtime - f.stat().st_mtime) / 3600
            stale.append(f"{f.name} ({age:.1f}h older than the code)")
    rep.add("evidence: stored results are newer than the code", "conceptual",
            "PASS" if not stale else "WARN",
            "; ".join(stale) + ("  -- regenerate before citing" if stale else ""))


def check_claims_register(rep: Report) -> None:
    """Headline claims must each point at evidence that exists."""
    register = {
        "learned beats quantised CSI at equal budget": RESULTS / "bitsweep_fixed.json",
        "message passing beats no messaging": RESULTS / "bitsweep_fixed.json",
        "bit budget sweep measures bits, not parameters": RESULTS / "bitsweep_fixed.json",
        "rho ablation": RESULTS / "rho_sweep.json",
        "task-oriented / semantic messages": RESULTS / "tasks.json",
        "symbolic distillation fidelity": RESULTS / "symbolic.json",
        "adversarial degradation and recovery": RESULTS / "adversarial.json",
        "state abstraction: what the messages encode": RESULTS / "analysis.json",
        "multimodal sensing helps most when CSI is stale": RESULTS / "sensing.json",
        "O-RAN placement and signalling overhead": RESULTS / "oran.json",
        "intents compile to lambda without retraining": RESULTS / "intent.json",
        "LLM allocator cost and accuracy": RESULTS / "llm.json",
        "Pareto front and hypervolume vs budget": RESULTS / "pareto.json",
        "prior repo's trend predictor vs persistence": RESULTS / "temporal.json",
    }
    unsupported = [c for c, p in register.items() if not p.exists()]
    rep.add("claims: every headline claim has evidence on disk", "conceptual",
            "PASS" if not unsupported else "WARN",
            f"{len(register)-len(unsupported)}/{len(register)} supported; unsupported: "
            + ", ".join(unsupported))


def run(full: bool = False) -> int:
    t0 = time.time()
    rep = Report()
    print("=" * 74)
    print("  TECHNICAL")
    print("=" * 74)
    check_self_tests(rep, full)
    check_physics(rep)
    check_gradients(rep)
    check_bits_are_bits(rep)
    check_finite(rep)
    check_determinism(rep)
    check_oracle_invariant(rep)
    check_temporal_is_live(rep, full)
    check_codebook_health(rep, full)

    print("\n" + "=" * 74)
    print("  CONCEPTUAL")
    print("=" * 74)
    check_partial_information(rep)
    check_preference_tradeoff(rep)
    check_preference_response(rep, full)
    check_task_calibration(rep, full)
    check_objective_coverage(rep)
    check_results_freshness(rep)
    check_claims_register(rep)

    fails = rep.summary()
    print(f"  completed in {time.time()-t0:.0f}s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "qa_report.json").write_text(
        json.dumps([c.__dict__ for c in rep.checks], indent=2)
    )
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include slow training-dependent checks")
    ap.add_argument("--loop", type=int, default=0, help="re-run every N seconds")
    a = ap.parse_args()
    if a.loop:
        while True:
            print(f"\n\n{'#'*74}\n# QA run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*74}")
            run(a.full)
            time.sleep(a.loop)
    else:
        sys.exit(1 if run(a.full) else 0)
