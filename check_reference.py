"""
How loose is the centralised reference, and where?

validate_scoring.py found Dinkelbach and interference pricing beating the 16-restart
projected-gradient reference on a sixth of instances at lambda = 0. The reference is used as the
denominator of every ratio in the paper, so its looseness is a systematic bias in the headline
numbers and needs measuring rather than noting.

Reports, per lambda: how often each classical arm exceeds the reference, and by how much.
"""
import sys, numpy as np
sys.path.insert(0, '.')
from dataset import cached_pool
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W
from baselines import dinkelbach, wmmse
from standalone_classical import interference_pricing, scalarised

M = 256
pool = cached_pool("test_N8_2048", size=2048, n_pairs=8, area_m=AREA_M, seed=999,
                   lambdas=LAMBDAS, device="cpu")
A = pool.gains.numpy(); se_ref = pool.se_ref.numpy(); ee_ref = pool.ee_ref.numpy()
noise, pc = pool.noise_power, CIRCUIT_POWER_W
orc = {float(k): v.numpy() for k, v in pool.oracle.items()}

print(f"{'lam':>5} {'arm':>11} {'frac>ref':>9} {'mean excess':>12} {'max excess':>11}", flush=True)
worst = {}
for lam in LAMBDAS:
    best_over = np.zeros(M)
    for name in ("wmmse", "dinkelbach", "pricing"):
        ex = []
        for m in range(M):
            p = (wmmse(A[m], noise, P_MAX_W) if name == "wmmse" else
                 dinkelbach(A[m], noise, P_MAX_W, pc) if name == "dinkelbach" else
                 interference_pricing(A[m], noise, P_MAX_W, lam, float(se_ref[m]),
                                      float(ee_ref[m]), pc))
            r = scalarised(p, A[m], noise, float(se_ref[m]), float(ee_ref[m]), lam, pc)[0]
            ex.append(r / float(orc[lam][m]) - 1.0)
        ex = np.array(ex)
        best_over = np.maximum(best_over, ex)
        pos = ex[ex > 1e-9]
        print(f"{lam:5.2f} {name:>11} {len(pos)/M:9.3f} "
              f"{(pos.mean() if len(pos) else 0):12.5f} {(ex.max() if len(ex) else 0):11.5f}",
              flush=True)
    worst[lam] = best_over
    print(f"{'':5} {'ANY arm':>11} {(best_over>1e-9).mean():9.3f} "
          f"{best_over[best_over>1e-9].mean() if (best_over>1e-9).any() else 0:12.5f} "
          f"{best_over.max():11.5f}", flush=True)
print("\nIf a tightened reference were used, every ratio at that lambda would fall by")
print("roughly the mean excess -- uniformly across arms, so between-arm gaps are unaffected.")
for lam in LAMBDAS:
    b = worst[lam]
    print(f"  lam={lam:.2f}: reference too low on {100*(b>1e-9).mean():.1f}% of instances, "
          f"mean shortfall {100*b[b>1e-9].mean() if (b>1e-9).any() else 0:.3f}%", flush=True)
