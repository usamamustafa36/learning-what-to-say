"""
Temporally correlated N-pair interference channel for beyond-5G/6G resource allocation.

Fixes the fatal modelling flaw in the original 6g-resource-allocation repo, where channels were
drawn i.i.d. from a uniform distribution while a "temporal optimizer" claimed to exploit temporal
correlation that did not exist in the data.

Here the small-scale fading follows a first-order Gauss-Markov (autoregressive) process whose
correlation coefficient is derived from the Jakes model, so temporal correlation is a controlled,
physically motivated quantity that a predictive method can be honestly evaluated against.

    h[t+1] = rho * h[t] + sqrt(1 - rho^2) * w[t],    w[t] ~ CN(0, 1)
    rho    = J0(2 * pi * f_d * T_s)

Setting rho = 0 recovers block-i.i.d. Rayleigh fading, which is the correct ablation for any claim
that a method exploits temporal structure.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0

SPEED_OF_LIGHT = 299_792_458.0


def jakes_rho(doppler_hz: float, slot_duration_s: float) -> float:
    """Gauss-Markov correlation coefficient implied by the Jakes autocorrelation J0(2*pi*f_d*tau)."""
    return float(np.clip(j0(2.0 * np.pi * doppler_hz * slot_duration_s), -1.0, 1.0))


def doppler_from_speed(speed_mps: float, carrier_hz: float) -> float:
    """Maximum Doppler shift f_d = v * f_c / c."""
    return speed_mps * carrier_hz / SPEED_OF_LIGHT


class InterferenceChannel:
    """
    N transmitter-receiver pairs sharing one band.

    Gain matrix G has shape (N, N); G[i, j] is the power gain from transmitter j to receiver i.
    Diagonal entries are the desired links, off-diagonal entries are interference. Geometry is
    drawn once per episode (slow), fading evolves every slot (fast).

    Parameters
    ----------
    n_pairs        : number of transmitter-receiver pairs
    area_m         : side length of the square deployment area
    pair_distance_m: (min, max) distance between a transmitter and its own receiver
    carrier_hz     : carrier frequency
    bandwidth_hz   : system bandwidth
    noise_figure_db: receiver noise figure
    path_loss_exp  : path-loss exponent
    speed_mps      : terminal speed, sets the Doppler and hence the temporal correlation
    slot_s         : slot duration
    rho            : if given, overrides the Jakes-derived correlation (use 0.0 for i.i.d. fading)
    """

    def __init__(
        self,
        n_pairs: int = 4,
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
        self.n = n_pairs
        self.area_m = area_m
        self.pair_distance_m = pair_distance_m
        self.carrier_hz = carrier_hz
        self.bandwidth_hz = bandwidth_hz
        self.path_loss_exp = path_loss_exp
        self.reference_loss_db = reference_loss_db
        self.slot_s = slot_s
        self.rng = rng if rng is not None else np.random.default_rng()

        # Thermal noise power: -174 dBm/Hz + 10log10(BW) + noise figure.
        noise_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db
        self.noise_power_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        self.doppler_hz = doppler_from_speed(speed_mps, carrier_hz)
        self.rho = jakes_rho(self.doppler_hz, slot_s) if rho is None else float(rho)

        self._path_gain: np.ndarray | None = None   # (N, N) large-scale, linear
        self._fading: np.ndarray | None = None      # (N, N) complex small-scale
        self.reset()

    # ------------------------------------------------------------------ geometry

    def _draw_geometry(self) -> np.ndarray:
        """Place transmitters uniformly, receivers at a random offset, return linear path gains."""
        tx = self.rng.uniform(0.0, self.area_m, size=(self.n, 2))
        lo, hi = self.pair_distance_m
        radius = self.rng.uniform(lo, hi, size=self.n)
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        rx = tx + np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

        # d[i, j] = distance from transmitter j to receiver i
        d = np.linalg.norm(rx[:, None, :] - tx[None, :, :], axis=2)
        d = np.maximum(d, 1.0)  # avoid singularity inside the reference distance

        loss_db = self.reference_loss_db + 10.0 * self.path_loss_exp * np.log10(d)
        return 10.0 ** (-loss_db / 10.0)

    # ------------------------------------------------------------------ dynamics

    def reset(self) -> np.ndarray:
        """New episode: redraw geometry and initialise fading from its stationary distribution."""
        self._path_gain = self._draw_geometry()
        self._fading = self._complex_gaussian()
        return self.gains()

    def step(self) -> np.ndarray:
        """Advance the fading process by one slot and return the new gain matrix."""
        innovation = self._complex_gaussian()
        self._fading = self.rho * self._fading + np.sqrt(1.0 - self.rho**2) * innovation
        return self.gains()

    def _complex_gaussian(self) -> np.ndarray:
        real = self.rng.normal(0.0, np.sqrt(0.5), size=(self.n, self.n))
        imag = self.rng.normal(0.0, np.sqrt(0.5), size=(self.n, self.n))
        return real + 1j * imag

    def gains(self) -> np.ndarray:
        """Instantaneous power gain matrix G[i, j] = path_gain[i, j] * |fading[i, j]|^2."""
        assert self._path_gain is not None and self._fading is not None
        return self._path_gain * np.abs(self._fading) ** 2

    # ------------------------------------------------------------------ helpers

    @property
    def noise_power(self) -> float:
        return self.noise_power_w

    def trajectory(self, n_slots: int) -> np.ndarray:
        """Return (n_slots, N, N) gains for one episode without redrawing geometry."""
        out = np.empty((n_slots, self.n, self.n))
        out[0] = self.gains()
        for t in range(1, n_slots):
            out[t] = self.step()
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for speed in (0.5, 3.0, 30.0):
        ch = InterferenceChannel(n_pairs=4, speed_mps=speed, rng=rng)
        traj = ch.trajectory(2000)
        direct = np.array([traj[:, i, i] for i in range(ch.n)])          # (N, T)
        # Empirical lag-1 correlation of the gain sequence, averaged over links.
        emp = np.mean([np.corrcoef(d[:-1], d[1:])[0, 1] for d in direct])
        print(f"speed={speed:5.1f} m/s  f_d={ch.doppler_hz:7.2f} Hz  "
              f"rho={ch.rho:.4f}  empirical lag-1 corr of |h|^2={emp:.4f}")
    ch = InterferenceChannel(n_pairs=4, rho=0.0, rng=rng)
    traj = ch.trajectory(2000)
    direct = np.array([traj[:, i, i] for i in range(ch.n)])
    emp = np.mean([np.corrcoef(d[:-1], d[1:])[0, 1] for d in direct])
    print(f"rho=0 (i.i.d. ablation)             empirical lag-1 corr of |h|^2={emp:.4f}")
