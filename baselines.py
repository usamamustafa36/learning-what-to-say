"""
Classical and reference allocators.

These are the credibility spine of the study. The original 6g-resource-allocation repo compared a
learned allocator only against an LLM, which avoided the comparison that matters: twenty years of
power-control literature. Nothing learned in this project is reported without WMMSE and Dinkelbach
standing next to it, and no decentralised method may exceed the centralised oracle.

Contents
--------
wmmse                : weighted MMSE sum-rate maximiser (SE ceiling, full CSI)
dinkelbach           : fractional programming for energy efficiency (EE ceiling, full CSI)
centralised_oracle   : multi-start projected gradient on the scalarised objective (the ceiling)
equal/max/random     : trivial floors
LloydMaxQuantizer    : B-bit scalar quantiser, the "quantised CSI" competitor to learned messages
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from metrics import energy_efficiency, sinr, spectral_efficiency
from regime import CIRCUIT_POWER_W, P_MAX_W

LN2 = np.log(2.0)


# --------------------------------------------------------------------------- floors


def equal_power(gains: np.ndarray, p_max: float) -> np.ndarray:
    return np.full(len(gains), p_max / 2.0)


def max_power(gains: np.ndarray, p_max: float) -> np.ndarray:
    return np.full(len(gains), p_max)


def random_power(gains: np.ndarray, p_max: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, p_max, size=len(gains))


# --------------------------------------------------------------------------- gradients


def se_gradient(powers: np.ndarray, gains: np.ndarray, noise_power: float) -> np.ndarray:
    """
    Analytic gradient of sum spectral efficiency with respect to the power vector.

    d(SE)/d(p_k) = (1/ln2) [ a_kk / ((1+s_k) D_k) - sum_{i != k} s_i a_ik / ((1+s_i) D_i) ]

    where D_i is the interference-plus-noise seen by receiver i and s_i its SINR. The first term is
    the benefit to your own link, the second the harm you do to everyone else's.
    """
    p = np.asarray(powers, dtype=float)
    a = np.asarray(gains, dtype=float)
    desired = np.diag(a) * p
    total = a @ p
    d = total - desired + noise_power           # interference + noise per receiver
    s = desired / d

    own = np.diag(a) / ((1.0 + s) * d)
    coupling = (s / ((1.0 + s) * d))            # per receiver i
    # sum_{i != k} coupling_i * a_ik  ->  a^T coupling, minus the i == k term
    cross = a.T @ coupling - np.diag(a) * coupling
    return (own - cross) / LN2


def ee_gradient(
    powers: np.ndarray, gains: np.ndarray, noise_power: float, circuit_power_w: float
) -> np.ndarray:
    """Quotient rule on EE = SE / (sum p + N * Pc)."""
    p = np.asarray(powers, dtype=float)
    se = spectral_efficiency(p, gains, noise_power)
    p_tot = np.sum(p) + circuit_power_w * len(p)
    return (se_gradient(p, gains, noise_power) * p_tot - se) / (p_tot**2)


# --------------------------------------------------------------------------- WMMSE


def wmmse(
    gains: np.ndarray,
    noise_power: float,
    p_max: float,
    n_iter: int = 200,
    tol: float = 1e-9,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Weighted MMSE sum-rate maximisation (Shi et al., 2011) for the SISO interference channel.

    Gains are power gains, so the amplitude channel is sqrt(gains); phases are irrelevant to the
    achievable rate here and are absorbed. Converges to a stationary point of sum-rate; run from
    several initialisations if the ceiling matters.
    """
    a = np.asarray(gains, dtype=float)
    n = len(a)
    root = np.sqrt(a)
    v = np.sqrt(np.full(n, p_max)) if init is None else np.sqrt(np.clip(init, 0.0, p_max))

    prev = -np.inf
    for _ in range(n_iter):
        p = v**2
        recv = a @ p + noise_power                       # total received + noise at each receiver
        u = np.diag(root) * v / recv                     # MMSE receiver
        w = 1.0 / np.maximum(1.0 - u * np.diag(root) * v, 1e-12)

        num = w * u * np.diag(root)
        den = a.T @ (w * u**2)                           # sum_j w_j u_j^2 a_ji
        v = np.clip(num / np.maximum(den, 1e-30), 0.0, np.sqrt(p_max))

        rate = spectral_efficiency(v**2, a, noise_power)
        if abs(rate - prev) < tol:
            break
        prev = rate

    return v**2


# --------------------------------------------------------------------------- Dinkelbach


def dinkelbach(
    gains: np.ndarray,
    noise_power: float,
    p_max: float,
    circuit_power_w: float = CIRCUIT_POWER_W,
    n_outer: int = 30,
    n_starts: int = 6,
    tol: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Dinkelbach's method for max EE = SE(p) / (sum p + N*Pc).

    The outer loop updates the fractional parameter q. The inner problem max SE(p) - q * P(p) is
    still nonconvex on an interference channel, so it is solved by multi-start L-BFGS-B with the
    analytic gradient rather than claimed to be solved exactly -- reported as such.

    Equal power is always among the starts, so the returned allocation can never be worse than
    equal power. An earlier hand-rolled projected-gradient inner loop used a fixed step against
    gradients of magnitude ~30 and slammed into the box boundary, losing to equal power on 12 of
    20 random instances. Let the line search choose the step.
    """
    a = np.asarray(gains, dtype=float)
    n = len(a)
    rng = rng if rng is not None else np.random.default_rng(0)
    bounds = [(1e-12, p_max)] * n

    starts = [np.full(n, p_max / 2.0), np.full(n, p_max), np.full(n, p_max / 50.0)]
    starts += [rng.uniform(1e-6, p_max, size=n) for _ in range(max(0, n_starts - 3))]

    p = np.full(n, p_max / 2.0)
    q = float(energy_efficiency(p, a, noise_power, circuit_power_w))

    for _ in range(n_outer):
        def neg_inner(x, q=q):
            return -(spectral_efficiency(x, a, noise_power) - q * (np.sum(x) + circuit_power_w * n))

        def neg_inner_grad(x, q=q):
            return -(se_gradient(x, a, noise_power) - q)

        best_x, best_v = p, np.inf
        for x0 in [p] + starts:
            res = minimize(neg_inner, x0, jac=neg_inner_grad, bounds=bounds, method="L-BFGS-B")
            if res.fun < best_v:
                best_v, best_x = res.fun, res.x

        p = np.clip(best_x, 1e-12, p_max)
        q_new = float(energy_efficiency(p, a, noise_power, circuit_power_w))
        if abs(q_new - q) < tol:
            q = q_new
            break
        q = q_new

    return p


# --------------------------------------------------------------------------- oracle


def centralised_oracle(
    gains: np.ndarray,
    noise_power: float,
    p_max: float,
    lam: float,
    se_ref: float,
    ee_ref: float,
    circuit_power_w: float = CIRCUIT_POWER_W,
    n_starts: int = 12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Multi-start L-BFGS-B on the preference-scalarised objective with full CSI.

    This is the ceiling every decentralised method in the paper is measured against. It is not a
    method being proposed — it is the number that says how much coordination costs.
    """
    a = np.asarray(gains, dtype=float)
    n = len(a)
    rng = rng if rng is not None else np.random.default_rng(0)
    se_ref = max(se_ref, 1e-12)
    ee_ref = max(ee_ref, 1e-12)

    def neg_obj(p):
        se = spectral_efficiency(p, a, noise_power) / se_ref
        ee = energy_efficiency(p, a, noise_power, circuit_power_w) / ee_ref
        return -(lam * se + (1.0 - lam) * ee)

    def neg_grad(p):
        g_se = se_gradient(p, a, noise_power) / se_ref
        g_ee = ee_gradient(p, a, noise_power, circuit_power_w) / ee_ref
        return -(lam * g_se + (1.0 - lam) * g_ee)

    bounds = [(1e-12, p_max)] * n
    starts = [np.full(n, p_max), np.full(n, p_max / 2.0), np.full(n, p_max / 100.0)]
    starts += [rng.uniform(1e-6, p_max, size=n) for _ in range(max(0, n_starts - 3))]

    best_p, best_val = starts[0], np.inf
    for x0 in starts:
        res = minimize(neg_obj, x0, jac=neg_grad, bounds=bounds, method="L-BFGS-B")
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return np.clip(best_p, 0.0, p_max)


def reference_values(
    gains: np.ndarray, noise_power: float, p_max: float, circuit_power_w: float = CIRCUIT_POWER_W
) -> tuple[float, float]:
    """Single-objective optima used to normalise the two objectives before scalarising."""
    se_ref = float(spectral_efficiency(wmmse(gains, noise_power, p_max), gains, noise_power))
    ee_ref = float(
        energy_efficiency(
            dinkelbach(gains, noise_power, p_max, circuit_power_w), gains, noise_power, circuit_power_w
        )
    )
    return se_ref, ee_ref


# --------------------------------------------------------------------------- quantisation


class LloydMaxQuantizer:
    """
    B-bit scalar quantiser fitted by Lloyd's algorithm on a training sample.

    This is the competitor that decides the paper. A learned B-bit message must beat B bits spent
    on quantising the agent's own channel state, or there is no case for learning the protocol.
    """

    def __init__(self, bits: int) -> None:
        self.bits = int(bits)
        self.levels = 1 << self.bits
        self.codebook: np.ndarray | None = None

    def fit(self, samples: np.ndarray, n_iter: int = 100, tol: float = 1e-9) -> "LloydMaxQuantizer":
        x = np.asarray(samples, dtype=float).ravel()
        if self.bits == 0:
            self.codebook = np.array([x.mean()])
            return self
        # Initialise on quantiles so empty cells are unlikely on skewed channel distributions.
        qs = (np.arange(self.levels) + 0.5) / self.levels
        self.codebook = np.quantile(x, qs)
        for _ in range(n_iter):
            edges = 0.5 * (self.codebook[1:] + self.codebook[:-1])
            idx = np.searchsorted(edges, x)
            new = self.codebook.copy()
            for k in range(self.levels):
                sel = x[idx == k]
                if len(sel):
                    new[k] = sel.mean()
            if np.max(np.abs(new - self.codebook)) < tol:
                self.codebook = new
                break
            self.codebook = new
        return self

    def quantize(self, values: np.ndarray) -> np.ndarray:
        assert self.codebook is not None, "call fit() first"
        v = np.asarray(values, dtype=float)
        if self.bits == 0:
            return np.full_like(v, self.codebook[0])
        edges = 0.5 * (self.codebook[1:] + self.codebook[:-1])
        return self.codebook[np.searchsorted(edges, v)]

    def indices(self, values: np.ndarray) -> np.ndarray:
        assert self.codebook is not None, "call fit() first"
        if self.bits == 0:
            return np.zeros_like(np.asarray(values), dtype=int)
        edges = 0.5 * (self.codebook[1:] + self.codebook[:-1])
        return np.searchsorted(edges, np.asarray(values, dtype=float))


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from channel import InterferenceChannel

    rng = np.random.default_rng(1)
    p_max, pc = P_MAX_W, CIRCUIT_POWER_W
    n_ok = 0
    n_trials = 20

    se_gain, ee_gain = [], []
    for _ in range(n_trials):
        ch = InterferenceChannel(n_pairs=6, rng=rng)
        g, n0 = ch.gains(), ch.noise_power

        p_eq = equal_power(g, p_max)
        p_w = wmmse(g, n0, p_max)
        p_d = dinkelbach(g, n0, p_max, pc)

        se_w = spectral_efficiency(p_w, g, n0)
        se_e = spectral_efficiency(p_eq, g, n0)
        ee_d = energy_efficiency(p_d, g, n0, pc)
        ee_e = energy_efficiency(p_eq, g, n0, pc)

        se_gain.append(se_w / se_e)
        ee_gain.append(ee_d / ee_e)
        n_ok += int(se_w >= se_e - 1e-9 and ee_d >= ee_e - 1e-9)

    print(f"WMMSE >= equal on SE and Dinkelbach >= equal on EE: {n_ok}/{n_trials}")
    print(f"  mean SE gain of WMMSE over equal power  : {np.mean(se_gain):.3f}x")
    print(f"  mean EE gain of Dinkelbach over equal   : {np.mean(ee_gain):.3f}x")

    # Gradient check against finite differences.
    ch = InterferenceChannel(n_pairs=5, rng=rng)
    g, n0 = ch.gains(), ch.noise_power
    p0 = rng.uniform(1e-3, p_max, size=5)
    ana = se_gradient(p0, g, n0)
    num = np.zeros(5)
    eps = 1e-9
    for k in range(5):
        e = np.zeros(5); e[k] = eps
        num[k] = (spectral_efficiency(p0 + e, g, n0) - spectral_efficiency(p0 - e, g, n0)) / (2 * eps)
    rel = np.max(np.abs(ana - num) / (np.abs(num) + 1e-12))
    print(f"analytic vs numerical SE gradient, max rel err: {rel:.2e}")

    # Oracle must dominate at both extremes of lambda.
    se_ref, ee_ref = reference_values(g, n0, p_max, pc)
    dominates = True
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        p_o = centralised_oracle(g, n0, p_max, lam, se_ref, ee_ref, pc, rng=rng)
        obj = lambda p: lam * spectral_efficiency(p, g, n0) / se_ref + (1 - lam) * energy_efficiency(p, g, n0, pc) / ee_ref
        best_other = max(obj(equal_power(g, p_max)), obj(wmmse(g, n0, p_max)), obj(dinkelbach(g, n0, p_max, pc)))
        ok = obj(p_o) >= best_other - 1e-6
        dominates &= ok
        print(f"  lambda={lam:.2f}  oracle={obj(p_o):.4f}  best baseline={best_other:.4f}  ok={ok}")
    print("oracle dominates every baseline at every lambda:", dominates)

    # Quantiser sanity: distortion must fall as bits rise.
    samples = rng.lognormal(mean=0.0, sigma=1.0, size=20000)
    prev = np.inf
    monotone = True
    for b in (1, 2, 3, 4, 6):
        q = LloydMaxQuantizer(b).fit(samples)
        mse = float(np.mean((q.quantize(samples) - samples) ** 2))
        monotone &= mse < prev
        prev = mse
        print(f"  {b}-bit Lloyd-Max MSE: {mse:.5f}")
    print("quantiser distortion strictly decreasing in bits:", monotone)
