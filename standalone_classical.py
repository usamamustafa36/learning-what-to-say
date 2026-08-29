"""
The classical algorithms as algorithms, not as messages inside our architecture.

The bit-budget table's "classical" arms are quantised quantities carried over the learned
architecture, which isolates message *content* but is open to the objection that no practitioner
ships a GNN. This script answers that objection directly by running the real methods to convergence
with unquantised information and no learned components at all:

  wmmse        Shi et al. 2011, full CSI, centralised, run to convergence. The SE workhorse.
  dinkelbach   fractional programming for EE, full CSI, centralised.
  pricing      distributed interference pricing (Schmidt et al. 2009; Shi et al. 2009): every
               receiver publishes a real-valued marginal interference price, every transmitter
               best-responds, iterate. This is the algorithm whose *dual variable* the B-bit
               "interference price" arm quantises, now run properly with unlimited signalling.

Each is scored on the same preference-scalarised reward and the same test pool as Table IV, and each
is run twice: on the realised channel at slot t, and on the stale measurement at t-1 that the
learned policy actually observes. The stale pair is what separates the cost of decentralisation from
the cost of acting on an old measurement -- the two are conflated in every ratio quoted against the
centralised reference.

    python3 standalone_classical.py            # N=8, the manuscript's operating point
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from baselines import dinkelbach, wmmse                              # noqa: E402
from dataset import cached_pool                                       # noqa: E402
from metrics import energy_efficiency, spectral_efficiency            # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W          # noqa: E402

OUT = HERE / "results" / "standalone_classical.json"


def scalarised(p, a, noise, se_ref, ee_ref, lam, pc):
    se = float(spectral_efficiency(p, a, noise))
    ee = float(energy_efficiency(p, a, noise, pc))
    return lam * se / se_ref + (1.0 - lam) * ee / ee_ref, se, ee


def interference_pricing(a, noise, p_max, lam, se_ref, ee_ref, pc,
                         n_iter=300, damping=0.5, tol=1e-10, count=False):
    """
    Distributed interference pricing on the scalarised objective.

    Receiver i publishes the marginal cost of interference it suffers,
        pi_i = d/dI_i [ -alpha * log2(1 + a_ii p_i / (noise + I_i)) ],
    transmitter j best-responds to the total price it pays, sum_{i!=j} pi_i a_ij, plus a linear
    power cost beta that carries the energy-efficiency half of the objective. Both terms are what a
    Dinkelbach linearisation of EE = SE / (sum_p + N*Pc) contributes at the current operating point,
    so the outer loop re-linearises as the allocation moves.

    Messages are real-valued and exchanged every iteration; the point of the arm is to show what the
    classical method achieves when signalling is *not* the binding constraint.
    """
    n = len(a)
    diag = np.diag(a)
    p = np.full(n, p_max * 0.5)
    ln2 = np.log(2.0)

    iters = 0
    for _i in range(n_iter):
        denom = float(p.sum() + n * pc)
        se = float(spectral_efficiency(p, a, noise))
        # d r / d SE and d r / d (sum p), from r = lam*SE/se_ref + (1-lam)*(SE/denom)/ee_ref
        alpha = lam / se_ref + (1.0 - lam) / (ee_ref * denom)
        beta = (1.0 - lam) * se / (ee_ref * denom * denom)

        interf = a @ p - diag * p + noise                 # noise + interference at each receiver
        sig = diag * p
        # marginal value of removing a unit of interference at receiver i
        price = alpha * sig / (ln2 * interf * (interf + sig))

        # transmitter j pays price_i * a_ij to every other receiver i
        cost = (a * price[:, None]).sum(axis=0) - diag * price       # sum_{i != j} pi_i a_ij
        total = np.maximum(cost + beta, 1e-30)
        target = np.clip(alpha / (total * ln2) - interf / np.maximum(diag, 1e-30), 0.0, p_max)

        new = (1.0 - damping) * p + damping * target
        done = np.max(np.abs(new - p)) < tol
        p = new
        iters = _i + 1
        if done:
            break
    return (p, iters) if count else p


def main() -> None:
    n_pairs, size = 8, 2048
    pool = cached_pool(f"test_N{n_pairs}_{size}", size=size, n_pairs=n_pairs, area_m=AREA_M,
                       seed=999, lambdas=LAMBDAS, device="cpu")
    A_now = pool.gains.numpy()
    A_obs = pool.gains_obs.numpy()
    se_ref = pool.se_ref.numpy()
    ee_ref = pool.ee_ref.numpy()
    noise, pc = pool.noise_power, CIRCUIT_POWER_W
    oracle = {float(k): v.numpy() for k, v in pool.oracle.items()}

    print(f"pool {len(pool)} instances, N={n_pairs}, lambdas {LAMBDAS}", flush=True)
    rows = []
    for csi, A in (("current", A_now), ("stale", A_obs)):
        for arm in ("wmmse", "dinkelbach", "pricing"):
            t0 = time.time()
            per_lam = {}
            for lam in LAMBDAS:
                ratios, ses, ees = [], [], []
                iter_counts = []
                for m in range(len(pool)):
                    a_dec, a_eval = A[m], A_now[m]        # decide on `a_dec`, always score on t
                    if arm == "wmmse":
                        p = wmmse(a_dec, noise, P_MAX_W)
                    elif arm == "dinkelbach":
                        p = dinkelbach(a_dec, noise, P_MAX_W, pc)
                    else:
                        p, it = interference_pricing(a_dec, noise, P_MAX_W, lam,
                                                     float(se_ref[m]), float(ee_ref[m]), pc,
                                                     count=True)
                        iter_counts.append(it)
                    r, se, ee = scalarised(p, a_eval, noise, float(se_ref[m]), float(ee_ref[m]),
                                           lam, pc)
                    ratios.append(r / float(oracle[lam][m]))
                    ses.append(se)
                    ees.append(ee)
                per_lam[str(lam)] = float(np.mean(ratios))
                print(f"  {arm:11s} {csi:7s} lam={lam:.2f} -> {np.mean(ratios):.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            rows.append({
                "arm": arm, "csi": csi, "n_pairs": n_pairs, "n_instances": len(pool),
                "per_lambda": per_lam,
                "mean_ratio": float(np.mean(list(per_lam.values()))),
                "abs_se": float(np.mean(ses)), "abs_ee": float(np.mean(ees)),
                "pricing_iters_mean": float(np.mean(iter_counts)) if iter_counts else None,
                "pricing_iters_p95": float(np.percentile(iter_counts, 95)) if iter_counts else None,
                "seconds": time.time() - t0,
            })
            OUT.write_text(json.dumps(rows, indent=2))
            print(f"  -> {arm}/{csi} mean {rows[-1]['mean_ratio']:.4f}\n", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
