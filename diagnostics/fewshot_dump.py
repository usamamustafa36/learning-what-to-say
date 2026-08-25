"""
Dump the raw few-shot generation per instance at the real n_test.

Written because `llm-few-shot` records mean_ratio exactly 0.0 with zero parse failures at
n_test=64, and two prior explanations for that were wrong. This settles it with data instead:
it records, for every test instance, the exact prompt tail, the raw decoded generation, what
`parse_powers` made of it, and the resulting per-instance ratio -- plus the exemplar block,
verbatim, because the exemplars depend on n_test (`bank_idx = arange(n_test, n_test+64)`) and
so differ between the full-scale run and any small-scale reproduction.

Read-only with respect to results/: writes results/diagnostics/fewshot_dump.json and nothing else.
No paper number changes on the strength of this script alone.

    python3 fewshot_dump.py            # n_test=64, the real setting
    python3 fewshot_dump.py --n-test 8 # the small-scale case that scored 0.0547
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # importable from anywhere

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from llm_agent import LLMAllocator, build_shots, describe, parse_powers
from regime import AREA_M, CIRCUIT_POWER_W, N_PAIRS, P_MAX_W  # single source of truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--k-shots", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/diagnostics/fewshot_dump.json")
    a = ap.parse_args()

    from dataset import build_pool
    from env import ee_torch, se_torch
    from solvers import oracle_batch

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise SystemExit("asked for cuda and did not get it -- refusing to dump a CPU run silently")

    # Identical construction to llm_experiment, including the seeds, or the dump describes
    # a different experiment than the one that produced the 0.0.
    pool = build_pool(size=a.n_test + 64, n_pairs=N_PAIRS, area_m=AREA_M, seed=999,
                      lambdas=(a.lam,), device=str(dev), p_max=P_MAX_W,
                      circuit_power_w=CIRCUIT_POWER_W)
    bank_idx = np.arange(a.n_test, a.n_test + 64)

    with torch.no_grad():
        lam_t = torch.full((len(pool),), a.lam, device=dev)
        p_or = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam_t,
                            pool.se_ref, pool.ee_ref, CIRCUIT_POWER_W, n_starts=16, n_steps=800)

    test = np.arange(a.n_test)
    gains_np = pool.gains_obs[test].cpu().numpy()
    shots = build_shots(pool, p_or, a.lam, a.k_shots, bank_idx)

    def ratio_of(p_np: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Per-instance obj/oracle -- the same quantity llm_experiment.score() averages."""
        p = torch.as_tensor(p_np, dtype=torch.float32, device=dev)
        g = pool.gains[idx]
        se = se_torch(p, g, pool.noise_power) / pool.se_ref[idx].clamp_min(1e-12)
        ee = ee_torch(p, g, pool.noise_power, CIRCUIT_POWER_W) / pool.ee_ref[idx].clamp_min(1e-12)
        obj = a.lam * se + (1.0 - a.lam) * ee
        return (obj / pool.oracle[a.lam][idx].clamp_min(1e-12)).cpu().numpy()

    llm = LLMAllocator(device=str(dev))
    prompts = ["\n\n".join(shots + [describe(gains_np[i], a.lam)]) for i in range(a.n_test)]

    # Same batching as allocate(), so any batch-boundary effect reproduces here.
    raw: list[str] = []
    t0 = time.perf_counter()
    for k in range(0, a.n_test, 8):
        raw.extend(llm._chat(prompts[k : k + 8]))
    dt = time.perf_counter() - t0

    powers = np.zeros((a.n_test, N_PAIRS))
    records = []
    for i, txt in enumerate(raw):
        p = parse_powers(txt, N_PAIRS)
        parsed_ok = p is not None
        fallback = not parsed_ok
        if p is None:
            p = np.full(N_PAIRS, 0.05)
        powers[i] = p
        records.append({
            "i": int(i),
            "raw_generation": txt,
            "raw_repr": repr(txt),
            "parsed_ok": bool(parsed_ok),
            "used_equal_power_fallback": bool(fallback),
            "parsed_mw": [round(float(x) * 1000.0, 3) for x in p],
            "parsed_all_zero": bool(parsed_ok and float(np.max(p)) == 0.0),
            "parsed_sum_mw": round(float(p.sum()) * 1000.0, 3),
        })

    r = ratio_of(powers, test)
    for rec, v in zip(records, r):
        rec["ratio"] = float(v)

    n_zero_vec = sum(x["parsed_all_zero"] for x in records)
    n_fail = sum(x["used_equal_power_fallback"] for x in records)
    n_zero_ratio = int((r == 0.0).sum())

    # Must mirror build_shots exactly, or the dump reports a rendering the model never saw.
    exemplar_mw = [[round(float(x) * 1000.0, 1) for x in p_or[int(i)].cpu().numpy()]
                   for i in bank_idx[: a.k_shots]]

    summary = {
        "n_test": a.n_test, "lam": a.lam, "k_shots": a.k_shots, "device": str(dev),
        "model": llm.model_id,
        "mean_ratio": float(r.mean()),
        "seconds_per_instance": dt / a.n_test,
        "parse_failures": n_fail,
        "parse_failure_rate": n_fail / a.n_test,
        "n_parsed_all_zero_vectors": n_zero_vec,
        "n_instances_with_zero_ratio": n_zero_ratio,
        "ratio_min": float(r.min()), "ratio_max": float(r.max()),
        "bank_idx_used": bank_idx[: a.k_shots].tolist(),
        "exemplar_powers_mw_as_rendered": exemplar_mw,
        "exemplar_zero_fraction": float(np.mean([v == 0.0 for row in exemplar_mw for v in row])),
        "shots_text": shots,
        "prompt_0": prompts[0],
    }

    with open(a.out, "w") as f:
        json.dump({"summary": summary, "instances": records}, f, indent=2)

    print(f"mean_ratio                {summary['mean_ratio']:.6f}")
    print(f"parse failures            {n_fail}/{a.n_test}  ({summary['parse_failure_rate']:.1%})")
    print(f"parsed all-zero vectors   {n_zero_vec}/{a.n_test}")
    print(f"instances with ratio 0    {n_zero_ratio}/{a.n_test}")
    print(f"exemplar idx              {summary['bank_idx_used']}")
    print(f"exemplar zero fraction    {summary['exemplar_zero_fraction']:.1%}")
    for row in exemplar_mw:
        print(f"  exemplar mW             {row}")
    print("\nfirst 5 raw generations:")
    for rec in records[:5]:
        print(f"  [{rec['i']:2d}] ratio={rec['ratio']:.4f} ok={rec['parsed_ok']} {rec['raw_repr'][:110]}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
