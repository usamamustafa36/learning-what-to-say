"""
The operating point every experiment shares, in one place, with the reason it is what it is.

These three numbers were scattered as literal defaults across eight modules and one of them was
quietly wrong for the entire study. Collecting them here is not tidiness -- it is what makes the
defect below impossible to reintroduce in one file while the others stay correct.

**Why the circuit power changed from 0.1 W to 0.01 W.**

Energy efficiency is `EE = SE / (sum_P + N*Pc)`. With N = 8 agents at Pc = 0.1 W, the circuit term
is 0.8 W while the entire transmit budget is also 0.8 W, so the denominator moves by at most a
factor of two no matter what the allocation does. EE is then very nearly proportional to SE, the
two objectives point the same way, and the preference weight lambda selects between two things that
are not actually different.

Measured on the centralised oracle, sweeping lambda from 0 to 1:

    area 50 m,  Pc 0.10 W :  SE +3.0%   EE  -6.2%    <- the old regime. No front to trace.
    area 200 m, Pc 0.10 W :  SE +9.0%   EE -14.2%
    area 50 m,  Pc 0.01 W :  SE +23.7%  EE -50.1%
    area 200 m, Pc 0.01 W :  SE +36.4%  EE -66.7%    <- this regime

Under the old numbers every trained policy ignored lambda, because ignoring it cost almost nothing:
transmit power at lambda = 0 and lambda = 1 differed by 0.1 mW out of 18. Every per-lambda column in
every results file was one allocation reported five times, and the Pareto front and hypervolume
computed from them were fictions. `qa.check_preference_tradeoff` now tests this regime directly,
before any policy is trained, because it is a property of the physics and not of the learning.

**Why the area changed from 50 m to 200 m.** At 50 m the deployment is interference-limited enough
that the sum-rate optimum switches off roughly 80% of the transmitters at every lambda, which
flattens the trade-off further and makes the oracle's own solutions degenerate -- the few-shot LLM
arm copied that sparsity and returned all zeros. 200 m is the channel model's own default and leaves
a working population of active links.

Nothing else moved. `p_max` is unchanged, the channel model is unchanged, and the bit budgets, the
message channel and the training recipe are all as they were.
"""

from __future__ import annotations

AREA_M = 200.0             # side of the square deployment area
CIRCUIT_POWER_W = 0.01     # per-agent circuit power; see above
P_MAX_W = 0.1              # per-agent transmit power ceiling
N_PAIRS = 8                # default number of transmitter-receiver pairs
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)     # the preference grid every evaluation reports over


def summary() -> str:
    return (f"area {AREA_M:.0f} m, Pc {CIRCUIT_POWER_W*1e3:.0f} mW, "
            f"p_max {P_MAX_W*1e3:.0f} mW, N {N_PAIRS}")


if __name__ == "__main__":
    """The regime must admit a trade-off. Measured, not asserted -- this is the check the old
    numbers would have failed."""
    import torch

    from dataset import build_pool
    from env import ee_torch, se_torch
    from solvers import oracle_batch

    print("regime:", summary())
    pool = build_pool(size=256, n_pairs=N_PAIRS, area_m=AREA_M, seed=7,
                      p_max=P_MAX_W, circuit_power_w=CIRCUIT_POWER_W)
    vals = {}
    for lam_val in (0.0, 0.5, 1.0):
        lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
        with torch.no_grad():
            p = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam, pool.se_ref, pool.ee_ref,
                             CIRCUIT_POWER_W, n_starts=16, n_steps=800)
        vals[lam_val] = (float(se_torch(p, pool.gains, pool.noise_power).mean()),
                         float(ee_torch(p, pool.gains, pool.noise_power, CIRCUIT_POWER_W).mean()),
                         float(p.mean()) * 1e3,
                         float((p < 1e-3).float().mean()))
    print(f"{'lambda':>8}{'SE':>9}{'EE':>10}{'power mW':>11}{'agents off':>12}")
    for k, (se, ee, mw, off) in vals.items():
        print(f"{k:>8.2f}{se:>9.3f}{ee:>10.2f}{mw:>11.2f}{off:>11.1%}")

    (se0, ee0, *_), (se1, ee1, *_) = vals[0.0], vals[1.0]
    d_se, d_ee = (se1 - se0) / se0, (ee1 - ee0) / ee0
    print(f"\nlambda 0 -> 1 moves SE by {100*d_se:+.1f}% and EE by {100*d_ee:+.1f}%")
    assert abs(d_se) > 0.10 and abs(d_ee) > 0.10, "no trade-off: lambda would be inert again"
    print("the regime admits a Pareto front")
