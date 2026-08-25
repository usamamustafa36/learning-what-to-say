"""
Batched GPU solvers: WMMSE, and a generic multi-start maximiser for the scalarised objective.

Why these exist. Training needs a per-instance SE and EE reference to normalise the two objectives
before scalarising, and evaluation needs a centralised oracle for every instance in every batch.
Calling the scipy solvers in baselines.py once per sample would cost minutes per epoch. These are
vectorised over the batch and run on the GPU.

They are validated against the scipy implementations in the self-test rather than trusted. The
scipy versions remain the reference; these are the fast path.
"""

from __future__ import annotations

import torch

from env import ee_torch, se_torch


def wmmse_batch(
    gains: torch.Tensor, noise_power: float, p_max: float, n_iter: int = 200, tol: float = 1e-10
) -> torch.Tensor:
    """Vectorised WMMSE over a batch of gain matrices. gains (B, N, N) -> powers (B, N)."""
    b, n, _ = gains.shape
    direct = torch.diagonal(gains, dim1=-2, dim2=-1)
    root = direct.sqrt()
    v = torch.full((b, n), p_max, device=gains.device, dtype=gains.dtype).sqrt()

    prev = torch.full((b,), -float("inf"), device=gains.device, dtype=gains.dtype)
    for _ in range(n_iter):
        p = v**2
        recv = torch.bmm(gains, p.unsqueeze(-1)).squeeze(-1) + noise_power
        u = root * v / recv
        w = 1.0 / (1.0 - u * root * v).clamp_min(1e-12)

        num = w * u * root
        den = torch.bmm(gains.transpose(1, 2), (w * u**2).unsqueeze(-1)).squeeze(-1)
        v = (num / den.clamp_min(1e-30)).clamp(0.0, p_max**0.5)

        rate = se_torch(v**2, gains, noise_power)
        if torch.max(torch.abs(rate - prev)) < tol:
            break
        prev = rate

    return v**2


def maximize_batch(
    objective,
    batch: int,
    n_agents: int,
    p_max: float,
    device: torch.device,
    n_starts: int = 16,
    n_steps: int = 800,
    lr: float = 0.005,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Multi-start projected Adam on the powers directly, maximising a batched objective.

    `objective` takes powers (S*B, N) and returns (S*B,). Restarts are stacked into the batch
    dimension so they all run in one pass; the best start per instance is returned. `lr` is
    relative to p_max, so the caller does not have to think about power units.

    Parameterisation note. An earlier version optimised p = p_max * sigmoid(z). That fails on the
    energy-efficiency objective, whose optimum is typically *sparse* -- several agents switched off
    entirely -- because reaching p = 0 needs z -> -inf where the sigmoid gradient has already
    vanished. It fell as low as 0.85x the scipy Dinkelbach optimum. Optimising p directly and
    clamping into the box after each step reaches the boundary exactly, and seeding some starts
    with a random subset of agents switched off gives the sparse optima a basin to fall into.
    """
    s, step = n_starts, lr * p_max
    p = torch.empty(s * batch, n_agents, device=device, dtype=dtype).uniform_(0.0, p_max)
    p[:batch] = p_max                                   # all on
    if s > 1:
        p[batch : 2 * batch] = p_max / 2.0              # half power
    if s > 2:                                           # sparse starts
        tail = p[2 * batch :]
        tail *= (torch.rand_like(tail) > 0.5).to(dtype)
    p.requires_grad_(True)

    # enable_grad so this still works when called from inside a torch.no_grad() block -- which is
    # the normal case, since the reference values and the oracle are not part of any training graph.
    with torch.enable_grad():
        opt = torch.optim.Adam([p], lr=step)
        for _ in range(n_steps):
            opt.zero_grad(set_to_none=True)
            loss = -objective(p.clamp(1e-12, p_max)).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(1e-12, p_max)

    with torch.no_grad():
        vals = objective(p).view(s, batch)
        best = vals.argmax(dim=0)
        return p.view(s, batch, n_agents)[best, torch.arange(batch, device=device)]


def reference_values_batch(
    gains: torch.Tensor, noise_power: float, p_max: float, circuit_power_w: float, **kw
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-instance SE and EE single-objective optima, used to normalise before scalarising."""
    b, n, _ = gains.shape
    se_ref = se_torch(wmmse_batch(gains, noise_power, p_max), gains, noise_power)
    ee_p = maximize_batch(
        lambda p: ee_torch(p, gains.repeat(p.shape[0] // b, 1, 1), noise_power, circuit_power_w),
        b, n, p_max, gains.device, **kw,
    )
    ee_ref = ee_torch(ee_p, gains, noise_power, circuit_power_w)
    return se_ref, ee_ref


def oracle_batch(
    gains: torch.Tensor,
    noise_power: float,
    p_max: float,
    lam: torch.Tensor,
    se_ref: torch.Tensor,
    ee_ref: torch.Tensor,
    circuit_power_w: float,
    **kw,
) -> torch.Tensor:
    """Centralised full-CSI ceiling on the scalarised objective, for every instance in the batch."""
    b, n, _ = gains.shape

    def obj(p):
        reps = p.shape[0] // b
        g = gains.repeat(reps, 1, 1)
        l = lam.repeat(reps)
        se = se_torch(p, g, noise_power) / se_ref.repeat(reps).clamp_min(1e-12)
        ee = ee_torch(p, g, noise_power, circuit_power_w) / ee_ref.repeat(reps).clamp_min(1e-12)
        return l * se + (1.0 - l) * ee

    return maximize_batch(obj, b, n, p_max, gains.device, **kw)


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import numpy as np

    from baselines import centralised_oracle, dinkelbach, reference_values, wmmse
    from env import Environment
    from metrics import energy_efficiency, spectral_efficiency

    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    env = Environment(batch=24, n_pairs=6, device=dev, rng=rng)
    g = env.gains
    g_np = g.double().cpu().numpy()
    n0, pmax, pc = env.noise_power, env.p_max, env.circuit_power_w

    # 1. Batched WMMSE must match the scipy-free reference implementation.
    p_t = wmmse_batch(g, n0, pmax).cpu().double().numpy()
    se_t = np.array([spectral_efficiency(p_t[i], g_np[i], n0) for i in range(24)])
    se_r = np.array([spectral_efficiency(wmmse(g_np[i], n0, pmax), g_np[i], n0) for i in range(24)])
    print(f"WMMSE batched vs reference, max rel diff in SE: {np.max(np.abs(se_t-se_r)/se_r):.2e}")

    # 2. Batched EE maximiser must not fall short of scipy Dinkelbach.
    se_ref_t, ee_ref_t = reference_values_batch(g, n0, pmax, pc)
    ee_ref_np = np.array([
        energy_efficiency(dinkelbach(g_np[i], n0, pmax, pc), g_np[i], n0, pc) for i in range(24)
    ])
    ratio = ee_ref_t.cpu().double().numpy() / ee_ref_np
    print(f"batched EE / scipy Dinkelbach EE: min {ratio.min():.4f}  mean {ratio.mean():.4f}")
    print("  batched maximiser is at least as good on every instance:", bool((ratio > 0.999).all()))

    # 3. Batched oracle vs scipy oracle, and the invariant that it beats every baseline.
    for lam_val in (0.0, 0.5, 1.0):
        lam = torch.full((24,), lam_val, device=dev)
        p_o = oracle_batch(g, n0, pmax, lam, se_ref_t, ee_ref_t, pc)
        obj_t = (lam_val * se_torch(p_o, g, n0) / se_ref_t
                 + (1 - lam_val) * ee_torch(p_o, g, n0, pc) / ee_ref_t).cpu().double().numpy()

        obj_s = []
        for i in range(24):
            sr, er = float(se_ref_t[i]), float(ee_ref_t[i])
            p = centralised_oracle(g_np[i], n0, pmax, lam_val, sr, er, pc, rng=rng)
            obj_s.append(lam_val * spectral_efficiency(p, g_np[i], n0) / sr
                         + (1 - lam_val) * energy_efficiency(p, g_np[i], n0, pc) / er)
        obj_s = np.array(obj_s)
        print(f"  lambda={lam_val:.1f}  batched {obj_t.mean():.4f}  scipy {obj_s.mean():.4f}  "
              f"ratio {np.mean(obj_t/obj_s):.4f}")

    # 4. Timing, which is the reason this module exists.
    import time
    big = Environment(batch=512, n_pairs=8, device=dev, rng=rng)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    reference_values_batch(big.gains, big.noise_power, big.p_max, big.circuit_power_w)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"batched refs for 512 instances x N=8: {time.time()-t0:.2f}s")
