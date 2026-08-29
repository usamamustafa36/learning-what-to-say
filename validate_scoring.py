"""
Does the standalone-classical scoring path agree with the pipeline that produced Table IV?

The classical arms are about to carry a headline claim, and they are scored by code written for
them rather than by the trainer's evaluator. If `scalarised()` disagreed with the objective the
oracle maximises, every ratio in the new table would be wrong in a way no amount of proofreading
would catch. Three checks:

  1. Re-score the oracle's own allocation through this path and divide by the stored oracle value.
     The oracle maximises lam*SE/SE* + (1-lam)*EE/EE*, so the result must be 1.0.
  2. Pricing must respect the power box and must beat equal-power and max-power.
  3. No arm may exceed the oracle. One that does means a broken reference or broken scoring.
"""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from dataset import cached_pool
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W
from baselines import equal_power, max_power, wmmse, dinkelbach
from solvers import oracle_batch
from standalone_classical import interference_pricing, scalarised

M = 64
pool = cached_pool("test_N8_2048", size=2048, n_pairs=8, area_m=AREA_M, seed=999,
                   lambdas=LAMBDAS, device="cpu")
A = pool.gains.numpy(); se_ref = pool.se_ref.numpy(); ee_ref = pool.ee_ref.numpy()
noise, pc = pool.noise_power, CIRCUIT_POWER_W
orc = {float(k): v.numpy() for k, v in pool.oracle.items()}
g = torch.tensor(A[:M]); sr = torch.tensor(se_ref[:M]); er = torch.tensor(ee_ref[:M])

print("TEST 1  scoring path vs stored oracle value (must be 1.0)", flush=True)
for lam in (0.0, 0.5, 1.0):
    lt = torch.full((M,), lam, dtype=g.dtype)
    P = oracle_batch(g, noise, P_MAX_W, lt, sr, er, pc).numpy()
    rs = [scalarised(P[m], A[m], noise, float(se_ref[m]), float(ee_ref[m]), lam, pc)[0]
          / float(orc[lam][m]) for m in range(M)]
    ok = "OK" if abs(np.mean(rs) - 1) < 0.02 else "MISMATCH"
    print(f"  lam={lam:.1f} mean {np.mean(rs):.6f} min {np.min(rs):.6f} max {np.max(rs):.6f}  {ok}",
          flush=True)

print("TEST 2  pricing feasibility and dominance (lam=0.5)", flush=True)
viol = beq = bmx = 0
for m in range(M):
    p = interference_pricing(A[m], noise, P_MAX_W, 0.5, float(se_ref[m]), float(ee_ref[m]), pc)
    if p.min() < -1e-12 or p.max() > P_MAX_W + 1e-12:
        viol += 1
    r = lambda q: scalarised(q, A[m], noise, float(se_ref[m]), float(ee_ref[m]), 0.5, pc)[0]
    beq += r(p) < r(equal_power(A[m], P_MAX_W))
    bmx += r(p) < r(max_power(A[m], P_MAX_W))
print(f"  violations {viol}/{M}  worse-than-equal {beq}/{M}  worse-than-max {bmx}/{M}", flush=True)

print("TEST 3  any arm above the oracle?", flush=True)
for lam in (0.0, 0.5, 1.0):
    out = []
    for name in ("wmmse", "dinkelbach", "pricing", "equal", "max"):
        c = 0
        for m in range(M):
            p = (wmmse(A[m], noise, P_MAX_W) if name == "wmmse" else
                 dinkelbach(A[m], noise, P_MAX_W, pc) if name == "dinkelbach" else
                 interference_pricing(A[m], noise, P_MAX_W, lam, float(se_ref[m]),
                                      float(ee_ref[m]), pc) if name == "pricing" else
                 equal_power(A[m], P_MAX_W) if name == "equal" else max_power(A[m], P_MAX_W))
            c += scalarised(p, A[m], noise, float(se_ref[m]), float(ee_ref[m]), lam,
                            pc)[0] / float(orc[lam][m]) > 1 + 1e-9
        out.append(f"{name} {c}")
    print(f"  lam={lam:.1f}: " + "  ".join(out) + f"  (of {M})", flush=True)
print("done", flush=True)
