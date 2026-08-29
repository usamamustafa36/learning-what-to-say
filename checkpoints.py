"""
Persist trained protocols, so that evaluation-only experiments stop paying to retrain.

`train.run_one` trains a network and throws it away. That is fine when every experiment trains its
own arm, but most of what we now want to ask -- how a policy transfers to a network size it never
saw, how it degrades under a noisy symbol channel, under CSI estimation error, at preference weights
outside the training grid -- are questions about an *already trained* policy. Retraining for each is
pure waste: one checkpoint set answers all of them.

Three things that will silently corrupt a cache here, all avoided below:

  net.norm is a plain dataclass attribute, NOT a registered buffer, so `state_dict()` does not carry
  it. A checkpoint restored without it falls back to DEFAULT_NORM via `getattr(net, "norm", None)`
  and evaluates against constants the model was never trained with -- quietly, and wrongly. We store
  and restore it explicitly.

  A cache keyed only on the config will serve a model trained on a *different pool*. We key on the
  config and a pool fingerprint together.

  Caching inside `train.train` itself would make qa.py's determinism check vacuous, since it calls
  train() twice and compares. The cache lives here, and callers opt in.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from agents import Normaliser
from dataset import Pool
from train import Config, train

CKPT = Path(__file__).parent / "results" / "checkpoints"


def pool_key(pool: Pool) -> str:
    """Fingerprint the data, not just its shape: two pools of equal size are not interchangeable."""
    g = pool.gains
    digest = hashlib.sha1(
        f"{pool.n_pairs}|{len(pool)}|{pool.rho}|{float(g.sum()):.6e}|{float(g.std()):.6e}".encode()
    ).hexdigest()[:12]
    return f"N{pool.n_pairs}_M{len(pool)}_{digest}"


def cfg_key(cfg: Config) -> str:
    d = asdict(cfg)
    keep = {k: d[k] for k in sorted(d) if k not in ("verbose",)}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_net(net, cfg: Config, pool: Pool) -> Path:
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / f"{cfg_key(cfg)}_{pool_key(pool)}.pt"
    torch.save(
        {
            "state": net.state_dict(),
            "cfg": asdict(cfg),
            # Explicitly, because state_dict() drops it -- see the module docstring.
            "norm": asdict(net.norm),
            "pool_key": pool_key(pool),
        },
        path,
    )
    return path


def load_net(cfg: Config, pool: Pool, device=None):
    """Return the cached net for this (config, pool) pair, or None."""
    path = CKPT / f"{cfg_key(cfg)}_{pool_key(pool)}.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location=device or "cpu", weights_only=False)
    from train import build                                   # local: avoids a circular import
    from agents import graph_inputs
    from train import price_ref, wants_full_csi

    gen = torch.Generator(device=pool.gains.device).manual_seed(cfg.seed)
    _, _, _, g0, _ = pool.sample(min(2048, len(pool)), gen)
    norm = Normaliser(**blob["norm"])
    _, edge0 = graph_inputs(g0, torch.rand(g0.shape[0], device=g0.device), norm=norm,
                            full_csi=wants_full_csi(cfg), price=price_ref(cfg, pool))
    net = build(cfg, edge0)
    net.load_state_dict(blob["state"])
    net.norm = norm
    net.eval()
    return net


def train_cached(cfg: Config, pool: Pool, verbose: bool = False):
    """train(), but reuse a stored network when one exists for this exact (config, pool)."""
    net = load_net(cfg, pool, device=pool.gains.device)
    if net is not None:
        return net
    net = train(cfg, pool, verbose=verbose)
    save_net(net, cfg, pool)
    return net
