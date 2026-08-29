"""
What a queue-blind policy delivers to a queue.

Every result elsewhere assumes saturated buffers, so a per-agent rate is a physical-layer floor and
not a delay guarantee. That is the largest gap between this study and a schedulable system, and the
reviewer is right to press on it.

It is also the one gap that cannot be closed by adding a metric. With a backlog the transition
kernel stops being action-independent -- what I transmit now changes my queue, hence my state next
slot -- and Remark~2's myopia argument, which is what lets the Dec-POMDP drop the discount factor,
no longer holds. A queue-aware protocol is a different paper.

So the queue here is deliberately EVALUATION-ONLY: an already-trained, queue-blind policy is rolled
over T slots against stochastic arrivals and a finite buffer, and we report what it delivers. That
adds realism without invalidating the formulation, and it measures something worth knowing -- a
policy optimising instantaneous SE/EE should mistreat a latency-bound user in a way a queue-aware
one would not, and if so that is a finding rather than a defect.

    python3 traffic.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from agents import graph_inputs                                        # noqa: E402
from checkpoints import train_cached                                   # noqa: E402
from dataset import cached_pool                                        # noqa: E402
from env import BatchChannel                                           # noqa: E402
from regime import AREA_M, CIRCUIT_POWER_W, LAMBDAS, P_MAX_W           # noqa: E402
from tasks import per_link_rate                                        # noqa: E402
from train import Config, node_extras, price_ref, wants_full_csi       # noqa: E402

RESULTS = HERE / "results"
N_PAIRS = 8
BITS = (0, 1, 2, 4, 6)
SEEDS = (0, 1, 2)
LOADS = (0.3, 0.5, 0.7, 0.9)


@dataclass
class TrafficConfig:
    n_slots: int = 200
    n_episodes: int = 256
    bandwidth_hz: float = 10e6
    slot_s: float = 1e-3
    buffer_bits: float = 2.0e5
    lam: float = 0.5


@torch.no_grad()
def rollout(net, cfg: Config, tcfg: TrafficConfig, load: float, seed: int = 0) -> dict:
    """Roll a trained policy over T slots against Poisson arrivals and a finite buffer."""
    dev = next(net.parameters()).device
    ch = BatchChannel(batch=tcfg.n_episodes, n_pairs=N_PAIRS, area_m=AREA_M,
                      rng=np.random.default_rng(10_000 + seed))
    g_prev = torch.as_tensor(ch.reset(), dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)

    # Offered load is a fraction of the capacity the silent policy would deliver, so `load` means
    # the same thing at every budget rather than drifting with the arm being measured.
    slot_bits = tcfg.bandwidth_hz * tcfg.slot_s
    q = torch.zeros(tcfg.n_episodes, N_PAIRS, device=dev)
    served_tot = torch.zeros_like(q)
    arrived_tot = torch.zeros_like(q)
    dropped_tot = torch.zeros_like(q)
    delay_sum = torch.zeros_like(q)
    delay_n = torch.zeros_like(q)
    power_tot = 0.0

    lam_t = torch.full((tcfg.n_episodes,), tcfg.lam, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    extra = node_extras(cfg, None, None, gen)

    for _ in range(tcfg.n_slots):
        g_now = torch.as_tensor(ch.step(), dtype=torch.float32, device=dev)
        node, edge = graph_inputs(g_prev, lam_t, extra_node=extra,
                                  norm=getattr(net, "norm", None),
                                  full_csi=wants_full_csi(cfg), price=None)
        p = net(node, edge)
        rate = per_link_rate(p, g_now, ch.noise_power)              # bits/s/Hz
        served = torch.minimum(q, rate * slot_bits)
        q = q - served
        # Backlog still waiting at the end of the slot contributes one slot of delay each.
        delay_sum += q
        delay_n += (q > 0).float()

        mean_rate = float(rate.mean())
        arrivals = torch.as_tensor(
            rng.poisson(load * mean_rate * slot_bits, size=(tcfg.n_episodes, N_PAIRS)),
            dtype=torch.float32, device=dev)
        q = q + arrivals
        over = (q - tcfg.buffer_bits).clamp_min(0.0)
        q = q - over

        served_tot += served
        arrived_tot += arrivals
        dropped_tot += over
        power_tot += float(p.sum(dim=-1).mean())
        g_prev = g_now

    thr = float(served_tot.sum() / (tcfg.n_episodes * tcfg.n_slots * tcfg.slot_s)) / 1e6
    energy = power_tot * tcfg.slot_s + N_PAIRS * CIRCUIT_POWER_W * tcfg.n_slots * tcfg.slot_s
    return {
        "throughput_mbps": thr,
        "drop_rate": float(dropped_tot.sum() / arrived_tot.sum().clamp_min(1.0)),
        "mean_delay_slots": float((delay_sum.sum() / served_tot.sum().clamp_min(1.0)) * 1.0),
        "backlog_frac": float((delay_n.sum() / (tcfg.n_episodes * N_PAIRS * tcfg.n_slots))),
        "energy_eff_mbit_per_j": float(served_tot.sum() / tcfg.n_episodes / max(energy, 1e-9)) / 1e6,
    }


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("refusing to run on CPU", flush=True)
        sys.exit(1)
    tr = cached_pool(f"train_N{N_PAIRS}_8192", size=8192, n_pairs=N_PAIRS, area_m=AREA_M,
                     seed=0, device=dev)
    tcfg = TrafficConfig()
    rows = []
    for bits in BITS:
        for seed in SEEDS:
            cfg = Config(bits=bits, mode="vq", steps=8000, seed=seed)
            net = train_cached(cfg, tr)
            net.eval()
            for load in LOADS:
                t0 = time.time()
                m = rollout(net, cfg, tcfg, load, seed=seed)
                m.update({"bits": bits, "seed": seed, "load": load,
                          "n_slots": tcfg.n_slots, "n_episodes": tcfg.n_episodes})
                rows.append(m)
                print(f"  B={bits} s={seed} load={load:.1f}: thr {m['throughput_mbps']:.2f} Mb/s"
                      f"  drop {m['drop_rate']*100:5.2f}%  backlog {m['backlog_frac']*100:5.1f}%"
                      f"  EE {m['energy_eff_mbit_per_j']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
                RESULTS.mkdir(parents=True, exist_ok=True)
                (RESULTS / "traffic.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {RESULTS / 'traffic.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
