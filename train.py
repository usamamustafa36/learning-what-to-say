"""
Training over cached instance pools.

Direct gradient ascent on the preference-scalarised objective. No reinforcement learning: SE and EE
are analytic and differentiable in the powers, the powers come out of the network, and the discrete
message channel is made differentiable by Gumbel-softmax with a straight-through estimator. The
whole system is one graph.

An earlier version recomputed per-instance reference values inside every gradient step, which meant
each outer step ran a multi-start inner optimisation and training was ~100x slower than necessary.
References depend only on the channel, so they are precomputed once per pool (see dataset.py) and
train and test pools are disjoint by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from agents import Normaliser, ProtocolGNN, QuantisedCSIGNN, graph_inputs
from dataset import Pool
from env import ee_torch, se_torch
from sensing import sensing_features
from solvers import maximize_batch
from tasks import TaskMix, task_objective, task_success_rate
from regime import AREA_M, CIRCUIT_POWER_W, P_MAX_W


@dataclass
class Config:
    bits: int = 4
    mode: str = "vq"                    # "vq" | "binary" | "continuous"
    messenger: str = "learned"          # "learned" | "quantised"
    rounds: int = 1
    hidden: int = 64
    msg_dim: int = 16
    steps: int = 8000
    batch: int = 512
    lr: float = 1e-3
    temp_start: float = 2.0
    temp_end: float = 0.5
    p_max: float = P_MAX_W
    circuit_power_w: float = CIRCUIT_POWER_W
    grad_clip: float = 5.0
    usage_bonus: float = 0.0     # entropy bonus on codeword usage; counters codebook collapse
    use_tasks: bool = False      # task-oriented objective instead of raw SE/EE
    use_sensing: bool = False    # second modality: geometry from positioning/ISAC (sensing.py)
    sensing_noise_db: float = 3.0
    sensing_shuffle: bool = False  # information-free control arm; keeps the width, breaks the pairing
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def objective(p, g, lam, se_ref, ee_ref, noise, pc):
    se = se_torch(p, g, noise) / se_ref.clamp_min(1e-12)
    ee = ee_torch(p, g, noise, pc) / ee_ref.clamp_min(1e-12)
    return lam * se + (1.0 - lam) * ee


def extra_node_dim(cfg: Config) -> int:
    """Width of everything appended to the three base node features."""
    return 2 * int(cfg.use_tasks) + 2 * int(cfg.use_sensing)


def node_extras(
    cfg: Config,
    path_gain: torch.Tensor | None,
    task_feats: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """
    Assemble the per-agent side information: its own task descriptor, its own sensing readout.

    Both are strictly local. An agent knows its own job and senses its own neighbourhood; neither
    tells it anything about what a neighbour wants or measures. That is the line the message channel
    exists to cross, and side information must not quietly cross it instead.
    """
    parts = []
    if task_feats is not None:
        parts.append(task_feats)
    if cfg.use_sensing:
        if path_gain is None:
            raise ValueError("use_sensing needs a pool carrying path gains (rebuild it)")
        parts.append(sensing_features(path_gain, cfg.sensing_noise_db, generator, cfg.sensing_shuffle))
    return torch.cat(parts, dim=-1) if parts else None


def build(cfg: Config, edge_sample: torch.Tensor | None = None) -> ProtocolGNN:
    cls = QuantisedCSIGNN if cfg.messenger == "quantised" else ProtocolGNN
    net = cls(
        bits=cfg.bits, p_max=cfg.p_max, rounds=cfg.rounds, hidden=cfg.hidden,
        msg_dim=cfg.msg_dim, mode=cfg.mode, temperature=cfg.temp_start,
        node_dim=3 + extra_node_dim(cfg),        # +2 per local side channel: task, sensing
    ).to(cfg.device)
    if cfg.messenger == "quantised" and edge_sample is not None:
        net.fit_quantizer(edge_sample)
    return net


def train(cfg: Config, pool: Pool, verbose: bool = False) -> ProtocolGNN:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=pool.gains.device).manual_seed(cfg.seed)

    # Fit the feature normaliser ONCE on the training pool and freeze it. It travels with the
    # model so evaluation uses identical constants -- no batch statistics, no test-set leakage,
    # and no cross-agent coupling through the normaliser.
    _, _, _, g0_obs, _ = pool.sample(min(2048, len(pool)), gen)
    norm = Normaliser.fit(g0_obs)
    _, edge0 = graph_inputs(g0_obs, torch.rand(g0_obs.shape[0], device=g0_obs.device), norm=norm)
    net = build(cfg, edge0)
    net.norm = norm

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    mix, task_rng = TaskMix(), np.random.default_rng(cfg.seed + 7919)

    for step in range(cfg.steps):
        frac = step / max(cfg.steps - 1, 1)
        net.channel.temperature = cfg.temp_start + (cfg.temp_end - cfg.temp_start) * frac

        g, sr, er, g_obs, pg = pool.sample(cfg.batch, gen)
        lam = torch.rand(cfg.batch, device=g.device)

        task_feats = None
        if cfg.use_tasks:
            _, r_min, beta = mix.sample(cfg.batch, pool.n_pairs, g.device, task_rng)
            task_feats = mix.features(r_min, beta)
        extra = node_extras(cfg, pg, task_feats, gen)

        # Observe the channel one slot in the past, be judged on the channel that actually occurs.
        node, edge = graph_inputs(g_obs, lam, extra_node=extra, norm=net.norm)
        p = net(node, edge)
        if cfg.use_tasks:
            obj = task_objective(
                p, g, pool.noise_power, lam, r_min, beta,
                torch.ones(cfg.batch, device=g.device), er, cfg.circuit_power_w,
            )
        else:
            obj = objective(p, g, lam, sr, er, pool.noise_power, cfg.circuit_power_w)

        loss = -obj.mean()
        if cfg.usage_bonus > 0.0 and getattr(net.channel, "last_logits", None) is not None:
            # Maximise the entropy of the *marginal* codeword distribution. Gumbel-softmax VQ is
            # known to collapse onto a couple of codewords; without this the nominal budget B and
            # the entropy actually transmitted are different quantities.
            probs = torch.softmax(net.channel.last_logits, dim=-1)
            marginal = probs.reshape(-1, probs.shape[-1]).mean(0)
            entropy = -(marginal * (marginal + 1e-12).log()).sum()
            loss = loss - cfg.usage_bonus * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if verbose and step % 2000 == 0:
            print(f"    step {step:5d}  obj {float(obj.mean()):.4f}", flush=True)

    return net


@torch.no_grad()
def evaluate_tasks(net: ProtocolGNN, cfg: Config, pool: Pool, seed: int = 4242) -> dict:
    """
    Task-oriented evaluation, with its own genie oracle.

    The SE/EE oracle cached in the pool is the wrong yardstick here: it maximises throughput and
    energy, not task satisfaction. So the oracle is recomputed against the task objective for the
    exact task draw being evaluated, otherwise the ratio would be measuring two different problems.
    """
    net.eval()
    mix, rng = TaskMix(), np.random.default_rng(seed)
    dev = pool.gains.device
    m = len(pool)
    _, r_min, beta = mix.sample(m, pool.n_pairs, dev, rng)
    gen = torch.Generator(device=dev).manual_seed(seed)
    extra = node_extras(cfg, pool.path_gain, mix.features(r_min, beta), gen)

    out, succ = {}, {}
    for lam_val in (pool.oracle or {0.5: None}):
        lam = torch.full((m,), lam_val, device=dev)
        node, edge = graph_inputs(pool.gains_obs, lam, extra_node=extra, norm=getattr(net, "norm", None))
        p = net(node, edge)
        ones = torch.ones(m, device=dev)
        obj = task_objective(p, pool.gains, pool.noise_power, lam, r_min, beta,
                             ones, pool.ee_ref, cfg.circuit_power_w)

        def task_obj(pw, lam=lam, r_min=r_min, beta=beta):
            reps = pw.shape[0] // m
            return task_objective(
                pw, pool.gains.repeat(reps, 1, 1), pool.noise_power, lam.repeat(reps),
                r_min.repeat(reps, 1), beta.repeat(reps, 1), ones.repeat(reps),
                pool.ee_ref.repeat(reps), cfg.circuit_power_w,
            )

        p_or = maximize_batch(task_obj, m, pool.n_pairs, cfg.p_max, dev, n_starts=8, n_steps=400)
        obj_or = task_objective(p_or, pool.gains, pool.noise_power, lam, r_min, beta,
                                ones, pool.ee_ref, cfg.circuit_power_w)
        out[lam_val] = float((obj / obj_or.clamp_min(1e-12)).mean())
        succ[lam_val] = float(task_success_rate(p, pool.gains, pool.noise_power, r_min).mean())

    net.train()
    return {"per_lambda": out, "mean_ratio": float(np.mean(list(out.values()))),
            "task_success": succ,
            "signalling_bits_per_agent": net.signalling_bits(pool.n_pairs)}


@torch.no_grad()
def evaluate(net: ProtocolGNN, cfg: Config, pool: Pool) -> dict:
    """
    Ratio to the centralised oracle, per lambda.

    Scale-free, and it makes the primary invariant checkable: a decentralised policy scoring above
    1.0 is a bug, not a result.

    Note the oracle is a *genie* bound: it optimises on the realised channel while the policy sees
    only the previous slot. Under fast fading the residual gap therefore includes the cost of stale
    information and is not attributable to decentralisation alone. Stated, not hidden.
    """
    net.eval()
    out, se_all, ee_all = {}, [], []
    # One sensing realisation, shared across lambdas: the preference is a command, not a new slot,
    # so redrawing the sensor noise per lambda would average away exactly what is being measured.
    gen = torch.Generator(device=pool.gains.device).manual_seed(4242)
    extra = node_extras(cfg, pool.path_gain, None, gen)
    for lam_val, oracle_val in pool.oracle.items():
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        node, edge = graph_inputs(pool.gains_obs, lam, extra_node=extra, norm=getattr(net, 'norm', None))
        p = net(node, edge)
        obj = objective(p, pool.gains, lam, pool.se_ref, pool.ee_ref, pool.noise_power, cfg.circuit_power_w)
        out[lam_val] = float((obj / oracle_val.clamp_min(1e-12)).mean())
        se_all.append(float(se_torch(p, pool.gains, pool.noise_power).mean()))
        ee_all.append(float(ee_torch(p, pool.gains, pool.noise_power, cfg.circuit_power_w).mean()))
    net.train()
    return {
        "per_lambda": out,
        "mean_ratio": float(np.mean(list(out.values()))),
        "se_by_lambda": se_all,
        "ee_by_lambda": ee_all,
        "signalling_bits_per_agent": net.signalling_bits(pool.n_pairs),
    }


def run_one(cfg: Config, train_pool: Pool, test_pool: Pool) -> dict:
    """
    Train and evaluate one arm, on the yardstick that matches the objective it was trained on.

    A task-trained policy scored against the SE/EE oracle is being marked on a different exam from
    the one it sat, and -- because the two objectives take different node features -- it also fails
    outright on a shape mismatch. Dispatch here rather than leaving that to every caller.
    """
    net = train(cfg, train_pool)
    res = evaluate_tasks(net, cfg, test_pool) if cfg.use_tasks else evaluate(net, cfg, test_pool)
    res["config"] = asdict(cfg)
    return res


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    from dataset import build_pool

    LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
    print("smoke test: silent vs 4-bit vs continuous, one seed", flush=True)
    tr = build_pool(size=4096, n_pairs=8, area_m=AREA_M, seed=0)
    te = build_pool(size=1024, n_pairs=8, area_m=AREA_M, seed=999, lambdas=LAMS)
    for label, cfg in [
        ("B=0  (silent)", Config(bits=0, steps=3000)),
        ("B=4  (learned vq)", Config(bits=4, steps=3000)),
        ("continuous (ceiling)", Config(bits=0, mode="continuous", steps=3000)),
    ]:
        r = run_one(cfg, tr, te)
        print(f"  {label:22s} {r['mean_ratio']:.4f} of oracle "
              f"({r['signalling_bits_per_agent']} bits/agent/slot)", flush=True)
