"""
Two impairments the study assumed away: imperfect CSI, and preference weights it never reported on.

CSI ESTIMATION ERROR. The observation model has exactly one impairment -- staleness, a one-slot
Gauss-Markov lag. A deployed agent also mis-measures. We add a log-normal multiplicative error in
dB, copying the model already used for the sensing modality, together with the same
information-free control: an arm whose perturbation is drawn from a *different* deployment, which
has identical input statistics and no usable information. The gap between treatment and control is
information; anything they share is capacity.

We perturb the pool rather than adding a field to `Pool`, because `Pool.to()` and the pool cache
both enumerate fields explicitly and a new one would have to be kept in sync in three places.

Note which pool the normaliser is fitted on. At deployment an agent only ever sees estimates, so the
statistics it standardises against are the statistics of estimates; fitting on clean gains would
hand it a calibration it cannot have. Evaluation here uses the model's frozen training-time
normaliser, so this is a train-clean / test-noisy robustness measurement and is labelled as such.

PREFERENCE GRID. lambda is a raw continuous input drawn uniformly during training, so the five-point
reporting grid is a convention, not a model constraint. Evaluating on 21 points -- 16 of which
appear in no training or reporting grid -- tests whether the conditioning generalises or merely
interpolates between the points it was scored on.

    python3 robustness.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from checkpoints import train_cached                                   # noqa: E402
from dataset import Pool, cached_pool                                  # noqa: E402
from regime import AREA_M, LAMBDAS, USAGE_BONUS  # noqa: E402
from train import Config, evaluate                                     # noqa: E402

RESULTS = HERE / "results"
N_PAIRS, TEST_SIZE = 8, 2048
STEPS = 8000
BITS = (0, 2, 4, 6)
SEEDS = (0, 1, 2)
SIGMAS_DB = (0.0, 1.0, 2.0, 3.0, 6.0)


def perturb_observation(pool: Pool, sigma_db: float, seed: int, shuffle: bool = False) -> Pool:
    """A copy of `pool` whose gains_obs carries a log-normal estimation error, in dB."""
    if sigma_db <= 0:
        return pool
    g = torch.Generator(device="cpu").manual_seed(seed)
    err = torch.randn(pool.gains_obs.shape, generator=g).to(pool.gains_obs.device) * sigma_db
    if shuffle:
        # Same marginal error distribution, drawn against a different deployment: the control.
        idx = torch.randperm(err.shape[0], generator=g).to(err.device)
        err = err[idx]
    return replace(pool, gains_obs=pool.gains_obs * (10.0 ** (err / 10.0)))


SUFFIX = ""            # "_smoke" on a smoke run, so a trial never overwrites a real result


def _smoke() -> None:
    """Tiny grid, one seed, short training: proves the code path before the real pool."""
    global TEST_SIZE, STEPS, BITS, SEEDS, SIGMAS_DB, SUFFIX
    SUFFIX = "_smoke"
    TEST_SIZE, STEPS = 64, 200
    BITS, SEEDS, SIGMAS_DB = (6,), (0,), (0.0, 3.0)
    print("SMOKE: 64 instances, 200 steps, one seed, one budget", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="64 instances, 200 steps, one seed, one budget: proves the path runs")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to run on CPU", flush=True)
        sys.exit(1)
    tr = cached_pool(f"train_N{N_PAIRS}_8192", size=8192, n_pairs=N_PAIRS, area_m=AREA_M,
                     seed=0, device=dev)
    te = cached_pool(f"test_N{N_PAIRS}_{TEST_SIZE}", size=TEST_SIZE, n_pairs=N_PAIRS,
                     area_m=AREA_M, seed=999, lambdas=LAMBDAS, device=dev)

    # ---------------------------------------------------------------- CSI estimation error
    rows = []
    for bits in BITS:
        for seed in SEEDS:
            cfg = Config(bits=bits, mode="vq", steps=STEPS, seed=seed,
                         usage_bonus=USAGE_BONUS)
            net = train_cached(cfg, tr)
            for sigma in SIGMAS_DB:
                for arm in ("sensing", "shuffled") if sigma > 0 else ("sensing",):
                    t0 = time.time()
                    pert = perturb_observation(te, sigma, seed=1000 + seed,
                                               shuffle=(arm == "shuffled"))
                    r = evaluate(net, cfg, pert)
                    rows.append({"arm": arm, "bits": bits, "seed": seed, "sigma_db": sigma,
                                 "mean_ratio": r["mean_ratio"], "n_instances": len(te)})
                    print(f"  csi B={bits} s={seed} sigma={sigma:.0f}dB {arm:>8}: "
                          f"{r['mean_ratio']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                    RESULTS.mkdir(parents=True, exist_ok=True)
                    (RESULTS / f"csi_error{SUFFIX}.json").write_text(json.dumps(rows, indent=2))

    # ---------------------------------------------------------------- 21-point preference grid
    lam21 = tuple(round(float(x), 3) for x in np.linspace(0.0, 1.0, 21))
    t0 = time.time()
    te21 = cached_pool(f"test_N{N_PAIRS}_{TEST_SIZE}_lam21", size=TEST_SIZE, n_pairs=N_PAIRS,
                       area_m=AREA_M, seed=999, lambdas=lam21, device=dev)
    print(f"  21-lambda pool ready in {time.time()-t0:.0f}s", flush=True)
    unseen = [l for l in lam21 if l not in set(LAMBDAS)]
    lrows = []
    for bits in BITS:
        for seed in SEEDS:
            cfg = Config(bits=bits, mode="vq", steps=STEPS, seed=seed,
                         usage_bonus=USAGE_BONUS)
            net = train_cached(cfg, tr)
            r = evaluate(net, cfg, te21)
            per = {float(k): v for k, v in r["per_lambda"].items()}
            lrows.append({
                "bits": bits, "seed": seed, "per_lambda": r["per_lambda"],
                "mean_all": float(np.mean(list(per.values()))),
                "mean_reported": float(np.mean([per[l] for l in LAMBDAS if l in per])),
                "mean_unseen": float(np.mean([per[l] for l in unseen if l in per])),
                "n_unseen": len(unseen), "n_instances": len(te21),
            })
            print(f"  lam21 B={bits} s={seed}: all {lrows[-1]['mean_all']:.4f} "
                  f"reported {lrows[-1]['mean_reported']:.4f} "
                  f"unseen {lrows[-1]['mean_unseen']:.4f}", flush=True)
            (RESULTS / f"lambda_grid{SUFFIX}.json").write_text(json.dumps(lrows, indent=2))
    print(f"wrote csi_error{SUFFIX}.json and lambda_grid{SUFFIX}.json")


if __name__ == "__main__":
    main()
