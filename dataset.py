"""
Precomputed instance pools.

The first training loop recomputed per-instance reference values inside every gradient step. Each
call runs a multi-start inner optimisation, so a single outer step cost hundreds of inner steps and
training was roughly two orders of magnitude slower than it needed to be -- 4000 steps did not
finish in two minutes.

References depend only on the channel, not on the policy, so they are computed once for a pool of
instances and cached. Oracle values for evaluation are cached the same way, per lambda. Training
then samples minibatches from the pool, which is both far faster and a cleaner experimental design:
train and test pools are explicitly disjoint and reproducible from their seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from env import BatchChannel
from solvers import oracle_batch, reference_values_batch
from regime import AREA_M, CIRCUIT_POWER_W, P_MAX_W

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FULL = dict(n_starts=16, n_steps=800)
CACHE = Path(__file__).parent / "results" / "pools"


@dataclass
class Pool:
    """
    A fixed set of channel instances with their normalisers, and optionally oracle values.

    Two gain matrices per instance, which is the point of the temporal model:

        gains_obs : the channel as *measured* at slot t-1, what the agents get to see
        gains     : the channel at slot t, over which the allocation is actually evaluated

    They are linked by the Gauss-Markov process, so rho controls how much the measurement is worth.
    At rho = 1 the two coincide and the problem is the static one; at rho = 0 the observation is
    pure noise and no allocation policy can do better than a blind one. Everything -- references,
    oracle, reported metrics -- is computed on `gains`, because that is the channel the transmission
    actually experiences. The oracle therefore sees the future and is a genie bound, which is stated
    rather than hidden.

    The first version of this pool took a single snapshot and never advanced the channel, so rho
    reached no experiment at all -- the same defect this project criticises in the prior repo.
    """

    gains: torch.Tensor                 # (M, N, N) at slot t -- evaluation
    se_ref: torch.Tensor                # (M,)
    ee_ref: torch.Tensor                # (M,)
    noise_power: float
    n_pairs: int
    oracle: dict[float, torch.Tensor] | None = None    # lambda -> (M,) objective value
    gains_obs: torch.Tensor | None = None              # (M, N, N) at slot t-1 -- observation
    rho: float = 1.0
    path_gain: torch.Tensor | None = None              # (M, N, N) large-scale term, no fading

    def __post_init__(self) -> None:
        if self.gains_obs is None:
            self.gains_obs = self.gains

    def __len__(self) -> int:
        return self.gains.shape[0]

    def to(self, device) -> "Pool":
        dev = torch.device(device)
        return Pool(
            self.gains.to(dev), self.se_ref.to(dev), self.ee_ref.to(dev),
            self.noise_power, self.n_pairs,
            None if self.oracle is None else {k: v.to(dev) for k, v in self.oracle.items()},
            self.gains_obs.to(dev), self.rho,
            None if self.path_gain is None else self.path_gain.to(dev),
        )

    def sample(self, batch: int, generator: torch.Generator | None = None):
        idx = torch.randint(0, len(self), (batch,), device=self.gains.device, generator=generator)
        pg = None if self.path_gain is None else self.path_gain[idx]
        return self.gains[idx], self.se_ref[idx], self.ee_ref[idx], self.gains_obs[idx], pg


def build_pool(
    size: int,
    n_pairs: int,
    p_max: float = P_MAX_W,
    circuit_power_w: float = CIRCUIT_POWER_W,
    device: str = DEFAULT_DEVICE,
    seed: int = 0,
    chunk: int = 512,
    lambdas: tuple[float, ...] | None = None,
    **channel_kwargs,
) -> Pool:
    """Generate `size` instances, compute reference values, and optionally cache oracle values."""
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    gains, gains_obs, se_refs, ee_refs, path_gains = [], [], [], [], []
    oracle: dict[float, list[torch.Tensor]] = {l: [] for l in (lambdas or ())}

    done = 0
    while done < size:
        m = min(chunk, size - done)
        ch = BatchChannel(batch=m, n_pairs=n_pairs, rng=rng, **channel_kwargs)
        # Observation at slot t-1, evaluation at slot t. One Gauss-Markov step apart.
        g_obs = torch.as_tensor(ch.gains(), dtype=torch.float32, device=dev)
        g = torch.as_tensor(ch.step(), dtype=torch.float32, device=dev)
        # Geometry, without fading. Drawn once per deployment and identical at both slots, so it is
        # the one thing a stale measurement does not lose -- see sensing.py.
        pg = torch.as_tensor(ch._path_gain, dtype=torch.float32, device=dev)
        with torch.no_grad():
            sr, er = reference_values_batch(g, ch.noise_power, p_max, circuit_power_w, **FULL)
            for l in oracle:
                lam = torch.full((m,), l, device=dev)
                p = oracle_batch(g, ch.noise_power, p_max, lam, sr, er, circuit_power_w, **FULL)
                from env import ee_torch, se_torch
                val = (lam * se_torch(p, g, ch.noise_power) / sr.clamp_min(1e-12)
                       + (1 - lam) * ee_torch(p, g, ch.noise_power, circuit_power_w) / er.clamp_min(1e-12))
                oracle[l].append(val)
        gains.append(g); gains_obs.append(g_obs); se_refs.append(sr); ee_refs.append(er)
        path_gains.append(pg)
        done += m

    return Pool(
        gains=torch.cat(gains), se_ref=torch.cat(se_refs), ee_ref=torch.cat(ee_refs),
        noise_power=ch.noise_power, n_pairs=n_pairs,
        oracle={l: torch.cat(v) for l, v in oracle.items()} if lambdas else None,
        gains_obs=torch.cat(gains_obs), rho=ch.rho, path_gain=torch.cat(path_gains),
    )


def cached_pool(tag: str, **kwargs) -> Pool:
    """Build a pool once and reuse it. Tag must encode every parameter that changes the data."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        # Honour the caller's device on a cache HIT as well as a miss. The cache stores CPU
        # tensors, so without this a cached pool silently comes back on CPU while the model is
        # built on CUDA, and training dies on a device mismatch several frames deep.
        return Pool(**blob).to(kwargs.get("device", "cpu"))
    pool = build_pool(**kwargs)
    torch.save(
        dict(gains=pool.gains.cpu(), se_ref=pool.se_ref.cpu(), ee_ref=pool.ee_ref.cpu(),
             noise_power=pool.noise_power, n_pairs=pool.n_pairs,
             oracle=None if pool.oracle is None else {k: v.cpu() for k, v in pool.oracle.items()},
             gains_obs=pool.gains_obs.cpu(), rho=pool.rho, path_gain=pool.path_gain.cpu()),
        path,
    )
    return pool


if __name__ == "__main__":
    import time

    t0 = time.time()
    pool = build_pool(size=1024, n_pairs=8, area_m=AREA_M, seed=0, lambdas=(0.0, 0.5, 1.0))
    dt = time.time() - t0
    print(f"built pool of {len(pool)} instances, N={pool.n_pairs}, in {dt:.1f}s "
          f"({1000*dt/len(pool):.1f} ms/instance)")
    print("se_ref  mean {:.3f}  min {:.3f}".format(float(pool.se_ref.mean()), float(pool.se_ref.min())))
    print("ee_ref  mean {:.3f}  min {:.3f}".format(float(pool.ee_ref.mean()), float(pool.ee_ref.min())))
    for l, v in pool.oracle.items():
        print(f"  oracle objective at lambda={l}: {float(v.mean()):.4f}")
    g, sr, er, g_obs, pg = pool.sample(64)
    print("sampled minibatch:", tuple(g.shape), tuple(sr.shape), tuple(er.shape),
          tuple(g_obs.shape), tuple(pg.shape))
    # Geometry must be identical at both slots; only the fading moves.
    ratio = (pool.gains / pool.path_gain).mean()
    print(f"mean |fading|^2 = {float(ratio):.3f} (expect ~1.0), "
          f"path gain shared across slots: {bool(pool.path_gain is not None)}")
    print(f"pool rho = {pool.rho:.4f}")
    import numpy as _np
    d_now = torch.diagonal(pool.gains, dim1=1, dim2=2).cpu().numpy().ravel()
    d_obs = torch.diagonal(pool.gains_obs, dim1=1, dim2=2).cpu().numpy().ravel()
    print(f"corr(observed, actual) direct gains = {_np.corrcoef(d_obs, d_now)[0,1]:.4f}  "
          f"(expect ~rho^2 = {pool.rho**2:.4f})")

    # rho=0 ablation. Note the raw gain correlation does NOT go to zero, and should not: geometry
    # is redrawn per instance but fixed across the two slots, so the path-loss component persists
    # whatever the fading does. The quantity that must vanish is the correlation of the *fading*,
    # which is isolated by dividing out the shared large-scale term.
    for r in (0.99, 0.0):
        q = build_pool(size=512, n_pairs=6, area_m=AREA_M, seed=1, rho=r)
        now = torch.diagonal(q.gains, dim1=1, dim2=2).cpu().numpy().ravel()
        obs = torch.diagonal(q.gains_obs, dim1=1, dim2=2).cpu().numpy().ravel()
        shared = _np.sqrt(now * obs).mean()          # crude common scale, enough to de-trend
        raw = _np.corrcoef(obs, now)[0, 1]
        resid = _np.corrcoef(obs / shared - (obs / shared).mean(),
                             now / shared - (now / shared).mean())[0, 1]
        # Per-instance de-trending: remove each instance's own mean gain, leaving fading only.
        o2 = obs.reshape(-1, 6); n2 = now.reshape(-1, 6)
        scale = _np.sqrt((o2 * n2).mean(axis=1, keepdims=True))
        fade = _np.corrcoef((o2 / scale).ravel(), (n2 / scale).ravel())[0, 1]
        print(f"  rho={r:.2f}: raw gain corr {raw:+.4f} (path loss persists), "
              f"fading-only corr {fade:+.4f}")
