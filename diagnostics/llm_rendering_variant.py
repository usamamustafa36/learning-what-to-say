"""
Third rendering: integers, but in units of 0.1 mW so the oracle's small powers survive.

Keeps the "integers" contract (which the model parses reliably) while giving 10x the resolution
of whole milliwatts. Monkeypatched so nothing in the repo changes until a variant proves itself.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # importable from anywhere

import numpy as np, torch
import llm_agent as L

_orig_describe = L.describe

def describe(gains, lam, p_max_mw=100.0):
    s = _orig_describe(gains, lam, p_max_mw)
    old = (f"Reply with exactly {gains.shape[0]} numbers in mW to one decimal place, comma "
           f"separated, inside square brackets. No other text.")
    new = (f"Reply with exactly {gains.shape[0]} integers, each in units of 0.1 mW "
           f"(so 100 means 10 mW, range 0 to 1000), comma separated, inside square brackets. "
           f"No other text.")
    assert old in s, "instruction anchor moved"
    return s.replace(old, new)

def build_shots(pool, oracle_powers, lam, k=3, idx=None):
    idx = np.arange(k) if idx is None else idx[:k]
    shots = []
    for i in idx:
        g = pool.gains_obs[int(i)].cpu().numpy()
        p = (oracle_powers[int(i)].cpu().numpy() * 10000.0).round().astype(int)  # 0.1 mW units
        shots.append(describe(g, lam) + "\n[" + ", ".join(str(int(x)) for x in p) + "]")
    return shots

_orig_parse = L.parse_powers
def parse_powers(text, n, p_max_mw=100.0):
    """Same parser, but the reply is in 0.1 mW units."""
    p = _orig_parse(text, n, p_max_mw * 10.0)   # allow up to 1000 units
    return None if p is None else p / 10.0      # 0.1mW-units -> W

L.describe, L.build_shots, L.parse_powers = describe, build_shots, parse_powers

rows = L.llm_experiment(n_test=64, tag="llm_variant01mw")
