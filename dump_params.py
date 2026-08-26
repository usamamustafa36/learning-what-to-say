"""
Emit `results/params.json` -- every simulation and training constant, read out of the code itself.

The paper's parameter table is generated from this file, so it cannot drift from what actually ran.
Nothing here is typed by hand: channel constants come from `InterferenceChannel`'s signature, the
operating point from `regime.py`, the training recipe from `train.Config`, and the pool sizes from
`experiments.sweep_v2`'s signature.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import channel
import regime
from experiments import sweep_v2
from train import Config


def main() -> None:
    ch = inspect.signature(channel.InterferenceChannel.__init__).parameters
    cfg, sw = Config(), inspect.signature(sweep_v2).parameters
    out = {
        "area_m": regime.AREA_M, "circuit_power_w": regime.CIRCUIT_POWER_W,
        "p_max_w": regime.P_MAX_W, "n_pairs": regime.N_PAIRS, "lambdas": len(regime.LAMBDAS),
        "carrier_hz": ch["carrier_hz"].default, "bandwidth_hz": ch["bandwidth_hz"].default,
        "noise_figure_db": ch["noise_figure_db"].default,
        "path_loss_exp": ch["path_loss_exp"].default,
        "reference_loss_db": ch["reference_loss_db"].default,
        "speed_mps": ch["speed_mps"].default, "slot_s": ch["slot_s"].default,
        "pair_distance_m": list(ch["pair_distance_m"].default),
        "steps": cfg.steps, "batch": cfg.batch, "lr": cfg.lr, "hidden": cfg.hidden,
        "msg_dim": cfg.msg_dim, "rounds": cfg.rounds, "grad_clip": cfg.grad_clip,
        "temp_start": cfg.temp_start, "temp_end": cfg.temp_end,
        "train_size": sw["train_size"].default, "test_size": sw["test_size"].default,
        "seeds": len(sw["seeds"].default), "usage_bonus": sw["usage_bonus"].default,
    }
    fd = channel.doppler_from_speed(out["speed_mps"], out["carrier_hz"])
    out["doppler_hz"] = fd
    out["rho"] = channel.jakes_rho(fd, out["slot_s"])
    out["noise_dbm"] = float(-174.0 + 10.0 * np.log10(out["bandwidth_hz"])
                             + out["noise_figure_db"])
    p = Path(__file__).parent / "results" / "params.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
