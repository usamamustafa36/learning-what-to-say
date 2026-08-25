"""
Heterogeneous task classes: what turns learned messages into *semantic* messages.

If every agent wants the same thing -- more rate -- then a message about the channel is all a
message can usefully be, and "semantic communication" is just a word attached to a learned encoder.
Give agents different jobs and that changes. A URLLC agent that already clears its latency budget
gains nothing from more power and should be telling its neighbours to take the headroom; an eMBB
agent always wants more. The useful content of a message is then the agent's *task state*, not its
channel state, and whether the learned protocol discovers that is an empirical question this module
makes askable.

Three classes, each a (r_min, beta) pair:

    eMBB   : low floor, high beta   -- always wants more throughput
    URLLC  : high floor, beta = 0   -- satisfy the requirement, then stop
    mMTC   : tiny floor, beta = 0   -- satisfy a small requirement at minimum power

Utility is a smooth satisfaction gate times an optional beyond-target reward, so it stays
differentiable and trains with the same machinery as everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EMBB, URLLC, MMTC = 0, 1, 2
TASK_NAMES = {EMBB: "eMBB", URLLC: "URLLC", MMTC: "mMTC"}

# (rate floor in bits/s/Hz, beyond-target coefficient)
TASK_PARAMS: dict[int, tuple[float, float]] = {
    EMBB: (0.5, 1.0),
    URLLC: (4.0, 0.0),
    MMTC: (0.2, 0.0),
}


@dataclass
class TaskMix:
    """Per-agent task assignment, redrawn whenever the deployment is redrawn."""

    probabilities: tuple[float, float, float] = (0.4, 0.3, 0.3)
    sharpness: float = 4.0          # k in the satisfaction sigmoid
    rate_scale: float = 4.0         # r_ref, normalises the beyond-target term

    def sample(
        self, batch: int, n_agents: int, device: torch.device, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (task_id, r_min, beta), each (batch, n_agents)."""
        ids = rng.choice(3, size=(batch, n_agents), p=self.probabilities)
        r_min = np.vectorize(lambda t: TASK_PARAMS[int(t)][0])(ids).astype(np.float32)
        beta = np.vectorize(lambda t: TASK_PARAMS[int(t)][1])(ids).astype(np.float32)
        to = lambda x, dt: torch.as_tensor(x, device=device, dtype=dt)
        return to(ids, torch.long), to(r_min, torch.float32), to(beta, torch.float32)

    def features(self, r_min: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Task descriptor appended to each agent's observation, (B, N, 2).

        An agent must know its own job; it must not be told anyone else's. Whether a neighbour is
        latency-bound is exactly the kind of thing the protocol has to learn to say.
        """
        return torch.stack([r_min / self.rate_scale, beta], dim=-1)


def per_link_rate(powers: torch.Tensor, gains: torch.Tensor, noise_power: float) -> torch.Tensor:
    """Per-link spectral efficiency, (B, N) -- the quantity a task requirement is written against."""
    desired = torch.diagonal(gains, dim1=-2, dim2=-1) * powers
    total = torch.bmm(gains, powers.unsqueeze(-1)).squeeze(-1)
    return torch.log2(1.0 + desired / (total - desired + noise_power))


def task_utility(
    powers: torch.Tensor,
    gains: torch.Tensor,
    noise_power: float,
    r_min: torch.Tensor,
    beta: torch.Tensor,
    sharpness: float = 4.0,
    rate_scale: float = 4.0,
) -> torch.Tensor:
    """
    Mean per-agent task utility, (B,).

        u_i = sigmoid(k * (rate_i - r_min_i)) * (1 + beta_i * relu(rate_i - r_min_i) / r_ref)

    The gate is what makes a URLLC agent stop caring once it is satisfied; the second factor is
    what keeps an eMBB agent hungry. Both are smooth, so the whole thing backpropagates.
    """
    rate = per_link_rate(powers, gains, noise_power)
    satisfied = torch.sigmoid(sharpness * (rate - r_min))
    surplus = torch.relu(rate - r_min) / rate_scale
    return (satisfied * (1.0 + beta * surplus)).mean(dim=-1)


def task_success_rate(
    powers: torch.Tensor, gains: torch.Tensor, noise_power: float, r_min: torch.Tensor
) -> torch.Tensor:
    """Hard fraction of agents meeting their rate floor, (B,). Reported, never optimised."""
    return (per_link_rate(powers, gains, noise_power) >= r_min).float().mean(dim=-1)


def task_objective(
    powers: torch.Tensor,
    gains: torch.Tensor,
    noise_power: float,
    lam: torch.Tensor,
    r_min: torch.Tensor,
    beta: torch.Tensor,
    util_ref: torch.Tensor,
    ee_ref: torch.Tensor,
    circuit_power_w: float,
    sharpness: float = 4.0,
    rate_scale: float = 4.0,
) -> torch.Tensor:
    """
    The task-oriented replacement for the raw SE/EE scalarisation.

    Still two objectives and still preference-conditioned -- task utility against energy efficiency
    -- so the multi-objective machinery, the Pareto front and the hypervolume all carry over
    unchanged. What changes is that objective one is now *task success*, not raw throughput.
    """
    from env import ee_torch

    u = task_utility(powers, gains, noise_power, r_min, beta, sharpness, rate_scale)
    ee = ee_torch(powers, gains, noise_power, circuit_power_w)
    return lam * u / util_ref.clamp_min(1e-12) + (1.0 - lam) * ee / ee_ref.clamp_min(1e-12)


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from env import Environment

    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment(batch=256, n_pairs=6, device=dev, rng=rng)
    mix = TaskMix()
    ids, r_min, beta = mix.sample(256, 6, dev, rng)

    counts = {TASK_NAMES[t]: int((ids == t).sum()) for t in (EMBB, URLLC, MMTC)}
    print("task mix:", counts, "of", ids.numel())
    print("task features shape:", tuple(mix.features(r_min, beta).shape))

    # 1. Utility must rise with power, but not without limit for the gated classes.
    for scale in (0.1, 0.5, 1.0):
        p = torch.full((256, 6), env.p_max * scale, device=dev)
        u = task_utility(p, env.gains, env.noise_power, r_min, beta).mean()
        s = task_success_rate(p, env.gains, env.noise_power, r_min).mean()
        print(f"  power {scale:4.1f} x p_max -> utility {float(u):.4f}  hard success {float(s):.3f}")

    # 2. A satisfied URLLC agent must gain far less from extra power than an eMBB agent.
    p_lo = torch.full((256, 6), env.p_max * 0.5, device=dev)
    p_hi = torch.full((256, 6), env.p_max, device=dev)
    rate_lo = per_link_rate(p_lo, env.gains, env.noise_power)
    rate_hi = per_link_rate(p_hi, env.gains, env.noise_power)

    def gain(mask):
        if mask.sum() == 0:
            return float("nan")
        g = torch.sigmoid(4.0 * (rate_hi - r_min)) * (1 + beta * torch.relu(rate_hi - r_min) / 4.0) \
            - torch.sigmoid(4.0 * (rate_lo - r_min)) * (1 + beta * torch.relu(rate_lo - r_min) / 4.0)
        return float(g[mask].mean())

    satisfied_urllc = (ids == URLLC) & (rate_lo >= r_min)
    embb = ids == EMBB
    print(f"  utility gain from doubling power -- eMBB: {gain(embb):.4f}, "
          f"already-satisfied URLLC: {gain(satisfied_urllc):.4f}")
    print("  satisfied URLLC gains less than eMBB:", gain(satisfied_urllc) < gain(embb))

    # 3. Differentiability.
    p = (torch.rand(64, 6, device=dev) * env.p_max).requires_grad_(True)
    task_utility(p, env.gains[:64], env.noise_power, r_min[:64], beta[:64]).sum().backward()
    print("  gradient flows through task utility:", bool(torch.isfinite(p.grad).all()))

    # 4. The objective stays two-dimensional and preference-conditioned.
    lam = torch.rand(64, device=dev)
    obj = task_objective(
        p, env.gains[:64], env.noise_power, lam, r_min[:64], beta[:64],
        torch.ones(64, device=dev), torch.full((64,), 20.0, device=dev), env.circuit_power_w,
    )
    print("  task objective shape:", tuple(obj.shape), "| finite:", bool(torch.isfinite(obj).all()))
