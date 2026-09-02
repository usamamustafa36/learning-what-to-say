"""
Multimodal sensing: what a position fix is worth when the CSI is stale.

Multimodal sensing is easy to claim: the temptation is to bolt a second input onto the observation
and declare the box ticked. The question worth asking is narrower and testable:

    a stale CSI measurement loses the fading, but it cannot lose the geometry.

The channel here is `path_gain * |fading|^2`. Geometry is drawn once per deployment and is identical
at slot t-1 and slot t; the fading decorrelates at rate rho. So an agent observing CSI at t-1 holds a
measurement whose useful content decays with rho, while a sensing modality that reports *where things
are* -- ISAC echo, GNSS, a positioning reference signal, a radio map -- reports something that has not
decayed at all.

That gives a prediction with a sign, not just a hope: sensing should be worth little when rho is high
(CSI already carries the geometry, plus the fading) and progressively more as rho falls. If the
measured curve does not have that shape, the modality is decoration and this module says so.

The control arm matters as much as the treatment. Adding two features also adds parameters, and a
model can improve for that reason alone. The `shuffled` arm therefore supplies sensing features drawn
from *another instance in the batch*: same input width, same parameter count, same everything, with
the geometric information destroyed. Any gap between `sensing` and `shuffled` is information; anything
they share is capacity.

What the sensing modality is not allowed to be: an oracle. Position is converted to large-scale gain
through the same log-distance model the channel uses, then corrupted by a log-normal error of
`sigma_db`, which stands in for ranging error and shadowing the sensor cannot resolve. At
sigma_db = 0 this module would be feeding the agent an exact path-loss matrix and the result would
mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from regime import AREA_M

# Frozen feature constants, fitted once over deployments at the regime in regime.py and hardcoded
# for the same
# reason agents.Normaliser freezes its own: statistics taken over a batch make every agent's features
# depend on every other agent's geometry, which is a leak, not a normalisation.
DIRECT_MEAN, DIRECT_STD = -9.04, 0.66
INTERF_MEAN, INTERF_STD = -9.29, 0.92


@dataclass
class SensingConfig:
    sigma_db: float = 3.0        # log-normal error on each sensed large-scale gain
    enabled: bool = True


def sensing_features(
    path_gain: torch.Tensor,
    sigma_db: float = 3.0,
    generator: torch.Generator | None = None,
    shuffle: bool = False,
) -> torch.Tensor:
    """
    Per-agent sensing observation, (B, N, 2).

    From its own position and the positions it can sense, agent i forms two large-scale quantities:

        own link      : the geometric gain of its own transmitter-receiver pair
        neighbourhood : the summed geometric gain of every interferer reaching its receiver

    Both are what a positioning or ISAC subsystem can produce without any radio measurement of the
    instantaneous channel, and both are constant over the coherence time of the fading -- that is the
    whole point of including them.

    `shuffle` is the information-free control: the same features, taken from a different deployment.
    """
    b, n, _ = path_gain.shape
    direct = torch.diagonal(path_gain, dim1=-2, dim2=-1)          # (B, N)
    interf = path_gain.sum(-1) - direct                           # (B, N)

    if sigma_db > 0:
        err = torch.randn(b, n, 2, device=path_gain.device, generator=generator) * sigma_db
        direct = direct * 10.0 ** (err[..., 0] / 10.0)
        interf = interf * 10.0 ** (err[..., 1] / 10.0)

    ld = (torch.log10(direct + 1e-30) - DIRECT_MEAN) / DIRECT_STD
    li = (torch.log10(interf + 1e-30) - INTERF_MEAN) / INTERF_STD
    feats = torch.stack([ld, li], dim=-1)

    if shuffle:
        # Break the pairing between an agent's sensing features and its own channel, keeping the
        # marginal distribution and the input width identical.
        perm = torch.randperm(b, device=feats.device, generator=generator)
        feats = feats[perm]
    return feats


def fit_constants(n_samples: int = 4096, n_pairs: int = 8, area_m: float = AREA_M) -> dict:
    """Recompute the frozen constants above. Run this if the deployment geometry changes."""
    from dataset import build_pool

    pool = build_pool(size=n_samples, n_pairs=n_pairs, area_m=area_m, seed=17)
    pg = pool.path_gain
    direct = torch.diagonal(pg, dim1=-2, dim2=-1)
    interf = pg.sum(-1) - direct
    ld, li = torch.log10(direct + 1e-30), torch.log10(interf + 1e-30)
    return {
        "DIRECT_MEAN": round(float(ld.mean()), 2), "DIRECT_STD": round(float(ld.std()), 2),
        "INTERF_MEAN": round(float(li.mean()), 2), "INTERF_STD": round(float(li.std()), 2),
    }


# --------------------------------------------------------------------------- experiment


def sensing_sweep(
    rhos=(0.0, 0.5, 0.9, 0.99),
    bits: int = 6,
    seeds=(0, 1, 2),
    steps: int = 8000,
    n_pairs: int = 8,
    area_m: float = AREA_M,
    sigma_db: float = 3.0,
    tag: str = "sensing",
) -> list[dict]:
    """Three arms per rho: CSI alone, CSI + sensing, CSI + shuffled sensing."""
    import json
    from pathlib import Path

    from dataset import build_pool
    from train import Config, run_one

    results = Path(__file__).parent / "results"
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    out = []

    for rho in rhos:
        tr = build_pool(size=8192, n_pairs=n_pairs, area_m=area_m, seed=0, rho=rho)
        te = build_pool(size=2048, n_pairs=n_pairs, area_m=area_m, seed=999,
                        lambdas=lambdas, rho=rho)
        for arm in ("csi", "sensing", "shuffled"):
            got = []
            for seed in seeds:
                cfg = Config(
                    bits=bits, steps=steps, seed=seed, usage_bonus=0.2,
                    use_sensing=arm != "csi", sensing_noise_db=sigma_db,
                    sensing_shuffle=arm == "shuffled",
                )
                r = run_one(cfg, tr, te)
                r["arm"], r["rho"], r["bits"] = arm, rho, bits
                out.append(r); got.append(r["mean_ratio"])
            print(f"  rho={rho:.2f} {arm:9s}: {np.mean(got):.4f} +/- {np.std(got):.4f}", flush=True)

    results.mkdir(parents=True, exist_ok=True)
    (results / f"{tag}.json").write_text(json.dumps(out, indent=2))
    summarise(out)
    return out


def summarise(rows: list[dict]) -> None:
    def cell(rho, arm):
        return [r["mean_ratio"] for r in rows if r["rho"] == rho and r["arm"] == arm]

    rhos = sorted({r["rho"] for r in rows})
    print("\n" + "=" * 74)
    print(f"{'rho':>6} {'CSI only':>12} {'+ sensing':>12} {'+ shuffled':>12} "
          f"{'information':>12} {'capacity':>10}")
    print("-" * 74)
    for rho in rhos:
        c, s, z = np.mean(cell(rho, "csi")), np.mean(cell(rho, "sensing")), np.mean(cell(rho, "shuffled"))
        print(f"{rho:>6.2f} {c:>12.4f} {s:>12.4f} {z:>12.4f} {s - z:>+12.4f} {z - c:>+10.4f}")
    print("=" * 74)
    print("information = sensing - shuffled (what the modality tells the agent)")
    print("capacity    = shuffled - CSI      (what two more input features buy on their own)")


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    torch.manual_seed(0)
    from dataset import build_pool

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool = build_pool(size=512, n_pairs=8, area_m=AREA_M, seed=3, device=dev)
    gen = torch.Generator(device=pool.gains.device).manual_seed(0)

    # 1. Shape and range.
    f = sensing_features(pool.path_gain, sigma_db=3.0, generator=gen)
    print("features:", tuple(f.shape), f"mean {float(f.mean()):+.3f}  std {float(f.std()):.3f}")
    assert f.shape == (512, 8, 2) and torch.isfinite(f).all()

    # 2. The frozen constants must still standardise this geometry, or every sensing feature is
    #    silently off-scale relative to the CSI features it sits beside.
    assert abs(float(f.mean())) < 0.3 and 0.7 < float(f.std()) < 1.4, "refit fit_constants()"
    print(f"  standardised: mean {float(f.mean()):+.3f}, std {float(f.std()):.3f}  "
          f"(frozen constants still fit)")

    # 3. Sensing must be stale-proof: its correlation with the *realised* channel is unchanged by
    #    rho, whereas the CSI observation's collapses. This is the premise of the whole module, so
    #    it is measured rather than asserted.
    for rho in (0.99, 0.0):
        q = build_pool(size=512, n_pairs=8, area_m=AREA_M, seed=5, rho=rho, device=dev)
        now = torch.diagonal(q.gains, dim1=1, dim2=2).cpu().numpy().ravel()
        obs = torch.diagonal(q.gains_obs, dim1=1, dim2=2).cpu().numpy().ravel()
        sen = torch.diagonal(q.path_gain, dim1=1, dim2=2).cpu().numpy().ravel()
        lc = lambda a, b: float(np.corrcoef(np.log10(a), np.log10(b))[0, 1])
        print(f"  rho={rho:.2f}: corr(log CSI@t-1, log channel@t) = {lc(obs, now):+.3f}   "
              f"corr(log sensing, log channel@t) = {lc(sen, now):+.3f}")

    # 4. Noise must degrade the sensed value monotonically -- no free oracle.
    for sig in (0.0, 3.0, 10.0):
        g2 = torch.Generator(device=pool.gains.device).manual_seed(1)
        f = sensing_features(pool.path_gain, sigma_db=sig, generator=g2)
        truth = sensing_features(pool.path_gain, sigma_db=0.0)
        print(f"  sigma={sig:4.1f} dB: corr with noiseless feature "
              f"{float(torch.corrcoef(torch.stack([f.ravel(), truth.ravel()]))[0,1]):.3f}")

    # 5. The shuffle control must destroy the pairing without changing the marginals. Correlate
    #    each feature channel separately: raveling both channels together would correlate the
    #    alternating direct/interference pattern, which survives any permutation of instances and
    #    would report a healthy control as a broken one.
    g3 = torch.Generator(device=pool.gains.device).manual_seed(2)
    a = sensing_features(pool.path_gain, sigma_db=0.0)
    b = sensing_features(pool.path_gain, sigma_db=0.0, generator=g3, shuffle=True)
    same_marginal = abs(float(a.mean() - b.mean())) < 1e-6 and abs(float(a.std() - b.std())) < 1e-6
    corrs = [float(torch.corrcoef(torch.stack([a[..., k].ravel(), b[..., k].ravel()]))[0, 1])
             for k in (0, 1)]
    print(f"  shuffled control: same marginals {same_marginal}, "
          f"pairing corr per channel {corrs[0]:+.3f} / {corrs[1]:+.3f}")
    assert same_marginal and max(abs(c) for c in corrs) < 0.2
