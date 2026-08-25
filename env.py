"""
Batched, differentiable multi-agent environment.

Two things matter here.

1. **Partial observability is the point.** Each agent sees only its own direct gain, the aggregate
   interference it measured in the previous slot, and its own preference weight. No agent sees the
   gain matrix. That is what forces coordination to be earned through messages rather than assumed,
   and it is the single change that turns a centralised optimiser into a multi-agent problem.

2. **The objective is differentiable in torch**, so the message encoder and power policy train by
   direct gradient ascent rather than policy gradients. SE and EE are analytic functions of the
   powers, so there is no reason to estimate a gradient that can be computed.

Channel generation is numpy and outside the gradient path; everything from gains onward is torch.
"""

from __future__ import annotations

import numpy as np
import torch

from channel import doppler_from_speed, jakes_rho
from regime import CIRCUIT_POWER_W, P_MAX_W

EPS = 1e-30


# --------------------------------------------------------------------------- batched channel


class BatchChannel:
    """
    Vectorised version of channel.InterferenceChannel over a batch of independent deployments.

    Mirrors the single-instance model exactly: log-distance path loss over random geometry, and
    first-order Gauss-Markov small-scale fading with a Jakes-derived correlation coefficient.
    Verified against InterferenceChannel in the self-test at the bottom of this file.
    """

    def __init__(
        self,
        batch: int,
        n_pairs: int,
        area_m: float = 200.0,
        pair_distance_m: tuple[float, float] = (10.0, 50.0),
        carrier_hz: float = 3.5e9,
        bandwidth_hz: float = 10e6,
        noise_figure_db: float = 7.0,
        path_loss_exp: float = 3.5,
        reference_loss_db: float = 40.0,
        speed_mps: float = 3.0,
        slot_s: float = 1e-3,
        rho: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.batch = batch
        self.n = n_pairs
        self.area_m = area_m
        self.pair_distance_m = pair_distance_m
        self.path_loss_exp = path_loss_exp
        self.reference_loss_db = reference_loss_db
        self.rng = rng if rng is not None else np.random.default_rng()

        noise_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
        self.noise_power = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        self.doppler_hz = doppler_from_speed(speed_mps, carrier_hz)
        self.rho = jakes_rho(self.doppler_hz, slot_s) if rho is None else float(rho)

        self._path_gain: np.ndarray | None = None
        self._fading: np.ndarray | None = None
        self.reset()

    def _cgauss(self) -> np.ndarray:
        shape = (self.batch, self.n, self.n)
        return self.rng.normal(0.0, np.sqrt(0.5), shape) + 1j * self.rng.normal(0.0, np.sqrt(0.5), shape)

    def reset(self) -> np.ndarray:
        b, n = self.batch, self.n
        tx = self.rng.uniform(0.0, self.area_m, size=(b, n, 2))
        lo, hi = self.pair_distance_m
        radius = self.rng.uniform(lo, hi, size=(b, n))
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=(b, n))
        rx = tx + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)

        # d[b, i, j] = distance from transmitter j to receiver i
        d = np.linalg.norm(rx[:, :, None, :] - tx[:, None, :, :], axis=-1)
        d = np.maximum(d, 1.0)
        loss_db = self.reference_loss_db + 10.0 * self.path_loss_exp * np.log10(d)
        self._path_gain = 10.0 ** (-loss_db / 10.0)
        self._fading = self._cgauss()
        return self.gains()

    def step(self) -> np.ndarray:
        self._fading = self.rho * self._fading + np.sqrt(1.0 - self.rho**2) * self._cgauss()
        return self.gains()

    def gains(self) -> np.ndarray:
        return self._path_gain * np.abs(self._fading) ** 2


# --------------------------------------------------------------------------- torch objectives


def se_torch(powers: torch.Tensor, gains: torch.Tensor, noise_power: float) -> torch.Tensor:
    """Sum spectral efficiency. powers (B, N), gains (B, N, N) -> (B,)"""
    desired = torch.diagonal(gains, dim1=-2, dim2=-1) * powers
    total = torch.bmm(gains, powers.unsqueeze(-1)).squeeze(-1)
    sinr = desired / (total - desired + noise_power)
    return torch.log2(1.0 + sinr).sum(dim=-1)


def ee_torch(
    powers: torch.Tensor, gains: torch.Tensor, noise_power: float, circuit_power_w: float
) -> torch.Tensor:
    """Energy efficiency = sum-SE / total consumed power."""
    n = powers.shape[-1]
    return se_torch(powers, gains, noise_power) / (powers.sum(dim=-1) + circuit_power_w * n)


def scalarized_torch(
    powers: torch.Tensor,
    gains: torch.Tensor,
    noise_power: float,
    lam: torch.Tensor,
    se_ref: torch.Tensor,
    ee_ref: torch.Tensor,
    circuit_power_w: float,
) -> torch.Tensor:
    """lam * SE/se_ref + (1 - lam) * EE/ee_ref, with lam per batch element."""
    se = se_torch(powers, gains, noise_power) / se_ref.clamp_min(1e-12)
    ee = ee_torch(powers, gains, noise_power, circuit_power_w) / ee_ref.clamp_min(1e-12)
    return lam * se + (1.0 - lam) * ee


# --------------------------------------------------------------------------- environment


class Environment:
    """
    One training batch of episodes.

    Observation per agent, all in log domain and standardised:
        [ log10(own direct gain),
          log10(interference + noise measured last slot),
          log10(own power last slot),
          lambda ]

    The interference term is a *measurement from the previous slot*, not an oracle read of the
    current interference, because an agent cannot know what its neighbours are about to transmit.
    This is what makes the coordination problem real.
    """

    OBS_DIM = 4

    def __init__(
        self,
        batch: int,
        n_pairs: int,
        p_max: float = P_MAX_W,
        circuit_power_w: float = CIRCUIT_POWER_W,
        device: str | torch.device = "cpu",
        rng: np.random.Generator | None = None,
        **channel_kwargs,
    ) -> None:
        self.batch = batch
        self.n = n_pairs
        self.p_max = p_max
        self.circuit_power_w = circuit_power_w
        self.device = torch.device(device)
        self.channel = BatchChannel(batch, n_pairs, rng=rng, **channel_kwargs)
        self.noise_power = self.channel.noise_power
        self._prev_power: torch.Tensor | None = None
        self._prev_interf: torch.Tensor | None = None
        self.reset()

    def reset(self) -> torch.Tensor:
        g = self.channel.reset()
        self.gains = torch.as_tensor(g, dtype=torch.float32, device=self.device)
        self._prev_power = torch.full((self.batch, self.n), self.p_max / 2.0, device=self.device)
        self._prev_interf = self._interference(self._prev_power, self.gains)
        return self.gains

    def advance(self) -> torch.Tensor:
        self.gains = torch.as_tensor(self.channel.step(), dtype=torch.float32, device=self.device)
        return self.gains

    def _interference(self, powers: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        desired = torch.diagonal(gains, dim1=-2, dim2=-1) * powers
        total = torch.bmm(gains, powers.unsqueeze(-1)).squeeze(-1)
        return total - desired + self.noise_power

    def observations(self, lam: torch.Tensor) -> torch.Tensor:
        """(B, N, OBS_DIM). lam is (B,) and is broadcast to every agent."""
        direct = torch.diagonal(self.gains, dim1=-2, dim2=-1)
        obs = torch.stack(
            [
                torch.log10(direct + EPS),
                torch.log10(self._prev_interf + EPS),
                torch.log10(self._prev_power + EPS),
                lam.unsqueeze(-1).expand(-1, self.n),
            ],
            dim=-1,
        )
        return self._standardise(obs)

    @staticmethod
    def _standardise(obs: torch.Tensor) -> torch.Tensor:
        """
        Centre the three log-domain channels; leave lambda alone since it is already in [0, 1] and
        carries meaning that must not be rescaled away.
        """
        out = obs.clone()
        logs = out[..., :3]
        mean = logs.mean(dim=(0, 1), keepdim=True)
        std = logs.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        out[..., :3] = (logs - mean) / std
        return out

    def commit(self, powers: torch.Tensor) -> None:
        """Record the slot's allocation so the next observation reflects what was actually measured."""
        self._prev_power = powers.detach()
        self._prev_interf = self._interference(powers.detach(), self.gains)

    def objective(self, powers, lam, se_ref, ee_ref) -> torch.Tensor:
        return scalarized_torch(
            powers, self.gains, self.noise_power, lam, se_ref, ee_ref, self.circuit_power_w
        )


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from channel import InterferenceChannel
    from metrics import energy_efficiency, spectral_efficiency

    rng = np.random.default_rng(7)

    # 1. BatchChannel must reproduce InterferenceChannel's statistics.
    bc = BatchChannel(batch=4000, n_pairs=4, rng=rng)
    single = InterferenceChannel(n_pairs=4, rng=np.random.default_rng(7))
    many = np.array([InterferenceChannel(n_pairs=4, rng=rng).gains() for _ in range(4000)])
    print("batched  mean log10 direct gain:", round(float(np.mean(np.log10(np.diagonal(bc.gains(), axis1=1, axis2=2)))), 4))
    print("single   mean log10 direct gain:", round(float(np.mean(np.log10(np.diagonal(many, axis1=1, axis2=2)))), 4))
    print("noise power matches:", np.isclose(bc.noise_power, single.noise_power))

    # 2. Temporal correlation survives batching.
    bc2 = BatchChannel(batch=200, n_pairs=3, speed_mps=3.0, rng=rng)
    traj = np.stack([bc2.gains()] + [bc2.step() for _ in range(500)])   # (T, B, N, N)
    d = np.diagonal(traj, axis1=2, axis2=3)                              # (T, B, N)
    corr = np.mean([np.corrcoef(d[:-1, b, i], d[1:, b, i])[0, 1] for b in range(50) for i in range(3)])
    print(f"batched lag-1 corr of |h|^2: {corr:.4f}  (rho^2 = {bc2.rho**2:.4f})")

    # 3. torch objectives must agree with the numpy reference.
    env = Environment(batch=64, n_pairs=5, rng=rng)
    p = torch.rand(64, 5) * env.p_max
    se_t = se_torch(p, env.gains, env.noise_power)
    ee_t = ee_torch(p, env.gains, env.noise_power, env.circuit_power_w)
    g_np, p_np = env.gains.numpy().astype(np.float64), p.numpy().astype(np.float64)
    se_n = np.array([spectral_efficiency(p_np[b], g_np[b], env.noise_power) for b in range(64)])
    ee_n = np.array([energy_efficiency(p_np[b], g_np[b], env.noise_power, env.circuit_power_w) for b in range(64)])
    print("max rel err SE torch vs numpy:", f"{np.max(np.abs(se_t.numpy()-se_n)/se_n):.2e}")
    print("max rel err EE torch vs numpy:", f"{np.max(np.abs(ee_t.numpy()-ee_n)/ee_n):.2e}")

    # 4. The objective must be differentiable with respect to the powers.
    p = (torch.rand(env.batch, 5) * env.p_max).requires_grad_(True)
    lam = torch.rand(env.batch)
    se_ref = torch.full((env.batch,), 10.0)
    ee_ref = torch.full((env.batch,), 20.0)
    obj = env.objective(p, lam, se_ref, ee_ref).sum()
    obj.backward()
    print("gradient flows to powers:", p.grad is not None and torch.isfinite(p.grad).all().item())

    # 5. Observations have the advertised shape and no NaNs.
    obs = env.observations(lam=torch.rand(env.batch))
    print("obs shape:", tuple(obs.shape), "| finite:", torch.isfinite(obs).all().item())
