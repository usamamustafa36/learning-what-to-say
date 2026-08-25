"""
Objectives and evaluation metrics for multi-objective power allocation.

The scalarisation is preference-conditioned: the trade-off weight lambda is an *input*, not a
constant. This is the difference between a genuinely multi-objective study, which characterises a
Pareto front, and a single-objective one that happens to sum two terms with fixed 0.5/0.5 weights.
"""

from __future__ import annotations

import numpy as np
from regime import CIRCUIT_POWER_W


def sinr(powers: np.ndarray, gains: np.ndarray, noise_power: float) -> np.ndarray:
    """
    SINR per receiver.

    powers : (..., N) transmit powers in watts
    gains  : (N, N) or (..., N, N); gains[i, j] is the gain from transmitter j to receiver i
    """
    powers = np.asarray(powers, dtype=float)
    gains = np.asarray(gains, dtype=float)
    desired = np.einsum("...ii,...i->...i", gains, powers)
    total = np.einsum("...ij,...j->...i", gains, powers)
    interference = total - desired
    return desired / (interference + noise_power)


def spectral_efficiency(powers: np.ndarray, gains: np.ndarray, noise_power: float) -> np.ndarray:
    """Sum spectral efficiency in bits/s/Hz."""
    return np.sum(np.log2(1.0 + sinr(powers, gains, noise_power)), axis=-1)


def energy_efficiency(
    powers: np.ndarray, gains: np.ndarray, noise_power: float, circuit_power_w: float = CIRCUIT_POWER_W
) -> np.ndarray:
    """Energy efficiency in bits/Joule/Hz: sum-SE divided by total consumed power."""
    se = spectral_efficiency(powers, gains, noise_power)
    consumed = np.sum(np.asarray(powers, dtype=float), axis=-1) + circuit_power_w * _n_from(powers)
    return se / consumed


def _n_from(powers: np.ndarray) -> int:
    return np.asarray(powers).shape[-1]


def jain_fairness(powers: np.ndarray, gains: np.ndarray, noise_power: float) -> np.ndarray:
    """Jain's fairness index over per-link rates. 1.0 is perfectly fair, 1/N is maximally unfair."""
    rates = np.log2(1.0 + sinr(powers, gains, noise_power))
    num = np.sum(rates, axis=-1) ** 2
    den = _n_from(powers) * np.sum(rates**2, axis=-1)
    return num / np.maximum(den, 1e-12)


class Scalarizer:
    """
    Preference-conditioned scalarisation of (SE, EE).

    Both objectives are normalised by reference values so that lambda sweeps a meaningful front
    rather than being dominated by whichever objective has the larger numerical scale. Reference
    values are the single-objective optima, estimated once per problem instance.
    """

    def __init__(self, se_ref: float, ee_ref: float) -> None:
        self.se_ref = max(float(se_ref), 1e-12)
        self.ee_ref = max(float(ee_ref), 1e-12)

    def __call__(
        self,
        powers: np.ndarray,
        gains: np.ndarray,
        noise_power: float,
        lam: float,
        circuit_power_w: float = CIRCUIT_POWER_W,
    ) -> np.ndarray:
        se = spectral_efficiency(powers, gains, noise_power) / self.se_ref
        ee = energy_efficiency(powers, gains, noise_power, circuit_power_w) / self.ee_ref
        return lam * se + (1.0 - lam) * ee


def pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated subset of a set of (SE, EE) points, both maximised.

    points : (M, 2)
    """
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
        if np.any(dominated):
            keep[i] = False
    return points[keep]


def hypervolume_2d(points: np.ndarray, reference: tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Hypervolume (dominated area) of a 2-D maximisation front with respect to a reference point.

    This is the standard scalar summary of multi-objective quality: it rewards both getting close
    to the ideal point and spreading across the front, so it cannot be gamed by a method that does
    well at one lambda and collapses elsewhere.
    """
    front = pareto_front(np.asarray(points, dtype=float))
    front = front[np.all(front > np.asarray(reference), axis=1)]
    if len(front) == 0:
        return 0.0
    order = np.argsort(-front[:, 0])          # descending in objective 1
    front = front[order]
    area = 0.0
    prev_y = reference[1]
    for x, y in front:
        if y > prev_y:
            area += (x - reference[0]) * (y - prev_y)
            prev_y = y
    return float(area)


METRIC_NAMES = ("spectral_efficiency", "energy_efficiency", "jain_fairness", "sum_power")


def evaluate(
    powers: np.ndarray, gains: np.ndarray, noise_power: float, circuit_power_w: float = CIRCUIT_POWER_W
) -> dict[str, float]:
    """Full metric dictionary for one allocation."""
    return {
        "spectral_efficiency": float(spectral_efficiency(powers, gains, noise_power)),
        "energy_efficiency": float(energy_efficiency(powers, gains, noise_power, circuit_power_w)),
        "jain_fairness": float(jain_fairness(powers, gains, noise_power)),
        "sum_power": float(np.sum(powers)),
    }


if __name__ == "__main__":
    from channel import InterferenceChannel

    rng = np.random.default_rng(0)
    ch = InterferenceChannel(n_pairs=4, rng=rng)
    g, n0 = ch.gains(), ch.noise_power

    full = np.full(4, 0.1)
    half = np.full(4, 0.05)
    print("full power:", {k: round(v, 4) for k, v in evaluate(full, g, n0).items()})
    print("half power:", {k: round(v, 4) for k, v in evaluate(half, g, n0).items()})
    print("SE rises with power:", spectral_efficiency(full, g, n0) > spectral_efficiency(half, g, n0))
    print("EE falls with power:", energy_efficiency(full, g, n0) < energy_efficiency(half, g, n0))

    pts = np.array([[10.0, 1.0], [8.0, 3.0], [5.0, 5.0], [1.0, 6.0], [4.0, 2.0]])
    print("pareto front:", pareto_front(pts).tolist())
    print("hypervolume:", round(hypervolume_2d(pts), 3))
