"""
A learned protocol has no specification, so there is nothing to check a message against.

That is the security argument this project exists to make, and it is not rhetorical. A hand-designed
protocol defines what each field means, so a receiver can reject a malformed or out-of-range message.
An emergent protocol defines nothing: the symbol alphabet is whatever the training run converged on,
every symbol in it is by construction a legal symbol, and the quantity a symbol refers to lives in
the sender's private measurement where no receiver can audit it. A sender that transmits the symbol
that suits it rather than the symbol its channel warrants is not violating any rule, because there
are no rules.

What is measured here:

    attack   : one or more agents replace their learned encoder with a symbol chosen to maximise
               their own rate, given how the honest agents will respond. Everything else -- the
               codebook, the aggregation, the power readout, the honest agents' policies -- is
               untouched, so any effect is attributable to the message content alone.
    damage   : the social objective, and separately what the attacker gains and the victims lose.
               A selfish attack that lowers everyone's outcome including its own is a different
               finding from one that transfers utility, and they are reported separately.
    defence  : the distilled specification from symbolic.py, used as a detector -- and honestly
               scored, with its false-alarm rate on traffic that is not attacking anything.

The attack is white-box and needs the victims' network to search against. That is the right threat
model for this question: it establishes what the *worst case* costs, which is what a protocol
designer needs to know. A weaker attacker is a weaker result, not a safer protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from agents import graph_inputs
from env import ee_torch, se_torch
from regime import CIRCUIT_POWER_W, LAMBDAS
from tasks import per_link_rate

RESULTS = Path(__file__).parent / "results"


class ConstantLiar:
    """
    An attacker that transmits one fixed symbol on all of its outgoing edges.

    The simplest attack that is still optimised: the symbol is chosen by exhaustive search over the
    2^B alphabet against the victims' actual responses, so it is the best constant lie rather than
    an arbitrary one. Attackers are honest about nothing and clever about one thing, which keeps the
    result attributable to the message rather than to a complicated adversary.
    """

    def __init__(self, attackers: list[int], symbol: int, honest_fn=None) -> None:
        self.attackers, self.symbol, self.honest_fn = attackers, int(symbol), honest_fn

    def __call__(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        sym = self.honest_fn(node, edge)
        # [b, r, s]: column s is what sender s transmits. Overwrite the attackers' columns only.
        for s in self.attackers:
            sym[:, :, s] = self.symbol
        return sym


class StealthyLiar:
    """
    An attacker that relabels its own honest symbols under a fixed permutation.

    The constant lie above is devastating and trivially detectable: sending one symbol forever is
    visible to any test on the symbol distribution. That makes it the wrong attack to conclude
    anything from, because it invites the reply "so validate the distribution".

    This attacker breaks the pairing between channel state and symbol without inventing any symbol
    it would not otherwise have sent. It is the naive answer to that reply, and it does not work.

    **It is not marginal-preserving, despite sending only symbols it would have sent anyway.** A
    permutation preserves the *multiset* of symbol counts while moving each count onto a different
    symbol, so the marginal is preserved only when the honest marginal is uniform (or the
    permutation happens to fix it). Ours is not uniform -- counts span roughly 45 to 250 at B=4 --
    so a non-identity permutation shifts the histogram by a total variation of about 0.11, and a
    chi-square test with an adequate window catches it in most windows. An earlier docstring here
    claimed the marginal was "identical by construction"; that was wrong, and it went unnoticed
    because the detector's fixed 64-symbol window was too underpowered to contradict it.

    Keep this arm as the negative control it actually is: it shows that merely conserving symbol
    *usage* is not enough to evade distributional validation. `MarginalMatchedLiar` below is the
    attack that does evade it, by constraining emitted counts to equal the honest counts exactly.
    That one carries the paper's claim; this one motivates why it has to be stated so carefully.
    """

    def __init__(self, attackers: list[int], perm: torch.Tensor, honest_fn=None) -> None:
        self.attackers, self.perm, self.honest_fn = attackers, perm, honest_fn

    def __call__(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        sym = self.honest_fn(node, edge)
        perm = self.perm.to(sym.device)
        for s in self.attackers:
            sym[:, :, s] = perm[sym[:, :, s]]
        return sym


def best_permutation(net, pool, attackers: list[int], lam_val: float, n_codes: int,
                     n_random: int = 24, n_swaps: int = 40, seed: int = 0) -> tuple[torch.Tensor, float]:
    """
    Search the permutation group for the relabelling that most helps the attacker.

    The group has 2^B! elements, so this is a heuristic: random restarts followed by greedy pairwise
    swaps. It gives a lower bound on what a stealthy attacker can achieve, which is the conservative
    direction -- a better search would make the protocol look worse, not better.
    """
    dev = pool.gains.device
    hon = honest_symbols(net)
    rng = np.random.default_rng(seed)

    def gain(perm):
        r = outcome(net, pool, lam_val, StealthyLiar(attackers, perm, hon))["rates"]
        return float(r[:, attackers].mean())

    identity = torch.arange(n_codes, device=dev)
    best, best_val = identity.clone(), gain(identity)
    for _ in range(n_random):
        cand = torch.as_tensor(rng.permutation(n_codes), device=dev)
        v = gain(cand)
        if v > best_val:
            best, best_val = cand, v
    for _ in range(n_swaps):
        i, j = rng.integers(0, n_codes, size=2)
        if i == j:
            continue
        cand = best.clone()
        cand[[i, j]] = cand[[j, i]]
        v = gain(cand)
        if v > best_val:
            best, best_val = cand, v
    return best, best_val


class MarginalMatchedLiar:
    """
    The strong stealthy attack: lie where it pays, conform on average.

    The permutation attack relabels every symbol the same way, which destroys the attacker's own
    coordination along with everyone else's -- and, because a permutation moves a non-uniform
    histogram rather than preserving it, does not even buy stealth (see `StealthyLiar`). This one
    is genuinely stealthy, and useful. Per slot the attacker sends whichever symbol most
    raises its own rate, but the *choice across slots* is constrained so that its emitted symbol
    counts match honest traffic exactly. It tells the truth often enough to keep its books clean and
    lies precisely when lying is worth something.

    Formally: with per-slot benefit r_c(i) for sending symbol c in instance i, and honest marginal
    counts n_c, choose an assignment maximising sum_i r_{c(i)}(i) subject to |{i : c(i)=c}| = n_c.
    That is a transportation problem; solved greedily here, which is a lower bound on the damage.

    The detector in symbolic.py tests the symbol marginal. Against this attacker the marginal is
    correct by construction, so the test has no power -- not because the test is badly chosen, but
    because a specification distilled from an emergent protocol constrains syntax and the attack is
    on semantics. That is the finding.
    """

    def __init__(self, attackers: list[int], assignment: torch.Tensor, honest_fn=None) -> None:
        self.attackers, self.assignment, self.honest_fn = attackers, assignment, honest_fn

    def __call__(self, node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        sym = self.honest_fn(node, edge)
        assert sym.shape[0] == self.assignment.shape[0], \
            "assignment was solved for a different instance set"
        a = self.assignment.to(sym.device)
        for s in self.attackers:
            sym[:, :, s] = a[:, :, s]
        return sym


def solve_marginal_matched(net, pool, attackers: list[int], lam_val: float, n_codes: int):
    """
    Per-*edge* benefit of every symbol, then a greedy reassignment that conserves the histogram.

    The formulation matters, and the first version here got it wrong in an instructive way. It
    constrained the attacker to one symbol per slot across all of its edges. Honest traffic sends a
    different symbol on each edge, so the honest policy was not even a member of the feasible set,
    and the "best" constrained attack came out 7% *worse* for the attacker than telling the truth.
    A search whose feasible set excludes the baseline cannot report a gain over it.

    Corrected: the decision variable is the symbol on each (slot, edge) individually, and the
    constraint is that the multiset of emitted symbols matches the honest multiset exactly. Honest
    is now feasible by construction, so the optimum is at least as good as honesty and any reported
    gain is real.

    Benefit is measured one edge at a time -- override edge (r, s) to symbol c, leave everything
    else honest, record the attacker's rate -- which is 2^B * (N-1) forward passes. The additive
    approximation that follows is a heuristic, but the *final* policy is evaluated exactly, so the
    damage reported is measured rather than predicted.
    """
    dev = pool.gains.device
    hon = honest_symbols(net)
    m, n = len(pool), pool.n_pairs
    lam = torch.full((m,), lam_val, device=dev)
    node, edge = graph_inputs(pool.gains_obs, lam, norm=getattr(net, "norm", None))
    honest = hon(node, edge)

    base = outcome(net, pool, lam_val)["rates"][:, attackers].mean(dim=-1)

    class _Override:
        """Honest everywhere except one (receiver, sender) edge, forced to `c`."""

        def __init__(self, r, s, c):
            self.r, self.s, self.c = r, s, c

        def __call__(self, node, edge):
            sym = hon(node, edge)
            sym[:, self.r, self.s] = self.c
            return sym

    # gain[s][r][c] : (M,) change in the attackers' mean rate from that single-edge deviation.
    slots = [(s, r) for s in attackers for r in range(n) if r != s]
    gain = torch.zeros(len(slots), n_codes, m, device=dev)
    for k, (s, r) in enumerate(slots):
        for c in range(n_codes):
            out = outcome(net, pool, lam_val, _Override(r, s, c))["rates"]
            gain[k, c] = out[:, attackers].mean(dim=-1) - base

    # Target histogram: exactly what the attackers emitted honestly.
    honest_slots = torch.stack([honest[:, r, s] for (s, r) in slots], dim=0)      # (K, M)
    quota = torch.bincount(honest_slots.reshape(-1), minlength=n_codes).clone()

    # Greedy transport: fill the highest-benefit (slot, symbol) pairs first, respecting the quota.
    #
    # The sorts are vectorised on the device; the fill itself is strictly sequential -- each
    # decision consumes quota the next one sees -- so it runs on the host. Kept on the device it
    # costs one synchronising launch per `left[c]` read and per decrement: K*M outer steps times
    # up to C inner ones, which is millions of round trips for B=6, and is what wedged the CUDA
    # context mid-sweep. Moving it here is a placement change only; the arithmetic is unchanged.
    flat = gain.permute(0, 2, 1).reshape(-1, n_codes)                             # (K*M, C)
    order = torch.argsort(flat.max(dim=-1).values, descending=True).cpu().tolist()
    ranked = torch.argsort(flat, dim=-1, descending=True).cpu().numpy()
    left = quota.cpu().tolist()
    assignment = [-1] * (len(slots) * m)
    for idx in order:
        for c in ranked[idx]:
            if left[c] > 0:
                assignment[idx] = int(c)
                left[c] -= 1
                break
    fallback = int(quota.argmax())
    assignment = torch.tensor([fallback if a < 0 else a for a in assignment],
                              dtype=torch.long, device=dev)

    full = honest.clone()
    a = assignment.view(len(slots), m)
    for k, (s, r) in enumerate(slots):
        full[:, r, s] = a[k]
    return full, quota


def honest_symbols(net):
    """The network's own encoder, exposed as a symbol_fn so attacks can overwrite part of it."""

    def fn(node, edge):
        b, n, _ = node.shape
        h = net.embed(node)
        hs = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        raw = net.msg(torch.cat([hs, edge], dim=-1))
        return net.channel.to_logits(raw).argmax(dim=-1)

    return fn


@torch.no_grad()
def outcome(net, pool, lam_val: float, symbol_fn=None) -> dict:
    """Per-agent rates and the social objective for one message policy."""
    lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
    node, edge = graph_inputs(pool.gains_obs, lam, norm=getattr(net, "norm", None))
    p = net(node, edge, symbol_fn=symbol_fn)
    rates = per_link_rate(p, pool.gains, pool.noise_power)
    se = se_torch(p, pool.gains, pool.noise_power) / pool.se_ref.clamp_min(1e-12)
    ee = ee_torch(p, pool.gains, pool.noise_power, CIRCUIT_POWER_W) / pool.ee_ref.clamp_min(1e-12)
    obj = lam_val * se + (1.0 - lam_val) * ee
    return {"rates": rates, "objective": obj, "powers": p}


def best_lie(net, pool, attackers: list[int], lam_val: float, n_codes: int) -> tuple[int, float]:
    """Exhaustive search over the alphabet for the symbol that maximises the attackers' own rate."""
    hon = honest_symbols(net)
    best_sym, best_gain = 0, -float("inf")
    base = outcome(net, pool, lam_val)["rates"][:, attackers].mean()
    for c in range(n_codes):
        r = outcome(net, pool, lam_val, ConstantLiar(attackers, c, hon))["rates"]
        gain = float(r[:, attackers].mean() - base)
        if gain > best_gain:
            best_sym, best_gain = c, gain
    return best_sym, best_gain


def attack(net, pool, n_attackers: int, bits: int, lam_val: float = 0.5,
           seed: int = 0, strategy: str = "constant") -> dict:
    """
    Run one attack and report who gained, who lost, and what it cost the system.

    `strategy` selects the threat model: "constant" is the loud, maximally damaging lie; "stealth"
    is the permutation attack that leaves the symbol histogram untouched.
    """
    net.eval()
    rng = np.random.default_rng(seed)
    n = pool.n_pairs
    attackers = sorted(rng.choice(n, size=n_attackers, replace=False).tolist())
    victims = [i for i in range(n) if i not in attackers]
    n_codes = 1 << bits

    hon = honest_symbols(net)
    before = outcome(net, pool, lam_val)
    sym = perm = assign = None
    if strategy == "constant":
        sym, _ = best_lie(net, pool, attackers, lam_val, n_codes)
        policy = ConstantLiar(attackers, sym, hon)
    elif strategy == "permutation":
        perm, _ = best_permutation(net, pool, attackers, lam_val, n_codes, seed=seed)
        policy = StealthyLiar(attackers, perm, hon)
    else:
        assign, _ = solve_marginal_matched(net, pool, attackers, lam_val, n_codes)
        policy = MarginalMatchedLiar(attackers, assign, hon)
    after = outcome(net, pool, lam_val, policy)

    def m(d, idx):
        return float(d["rates"][:, idx].mean())

    obj_b, obj_a = float(before["objective"].mean()), float(after["objective"].mean())
    net.train()
    return {
        "bits": bits, "n_attackers": n_attackers, "attackers": attackers, "lambda": lam_val,
        "strategy": strategy, "symbol": sym,
        "permutation": None if perm is None else perm.cpu().tolist(),
        "assignment": None if assign is None else assign.cpu().tolist(),
        "objective_before": obj_b, "objective_after": obj_a,
        "objective_change": (obj_a - obj_b) / max(obj_b, 1e-12),
        "attacker_rate_before": m(before, attackers), "attacker_rate_after": m(after, attackers),
        "attacker_gain": (m(after, attackers) - m(before, attackers)) / max(m(before, attackers), 1e-12),
        "victim_rate_before": m(before, victims), "victim_rate_after": m(after, victims),
        "victim_loss": (m(after, victims) - m(before, victims)) / max(m(before, victims), 1e-12),
    }


# --------------------------------------------------------------------------- defence


@torch.no_grad()
def detection(net, pool, attackers: list[int], policy, bits: int, lam_val: float = 0.5,
              window: int | None = None, alpha: float = 0.01, seed: int = 0) -> dict:
    """
    Can the distilled specification tell the liar from the honest senders?

    The validator is fitted on honest traffic from *all* senders, then applied per sender over
    windows. Detection rate is measured on the attackers, false alarms on everyone else, and both
    are reported -- a detector that flags every sender catches every attacker and is worthless.
    """
    from symbolic import MessageValidator

    rng = np.random.default_rng(seed)
    lam = torch.full((len(pool),), lam_val, device=pool.gains.device)
    node, edge = graph_inputs(pool.gains_obs, lam, norm=getattr(net, "norm", None))
    honest = honest_symbols(net)(node, edge)                  # (M, N, N)
    lying = policy(node, edge)

    n = pool.n_pairs
    off = ~torch.eye(n, dtype=torch.bool, device=honest.device)
    # Window defaults to the shortest one at which chi-square can run for this alphabet. Passing a
    # fixed 64 here silently disabled the test at B=6 and reported zero detection for an attack that
    # is trivially detectable.
    val = MessageValidator.fit(honest[off.expand(len(pool), n, n)].cpu().numpy(),
                               n_codes=1 << bits, window=window, alpha=alpha)

    def per_sender(sym_t, s):
        col = sym_t[:, [r for r in range(n) if r != s], s].reshape(-1).cpu().numpy()
        return np.mean([val.flag(rng.choice(col, val.window)) for _ in range(50)])

    detected = [float(per_sender(lying, s)) for s in attackers]
    false_alarm = [float(per_sender(lying, s)) for s in range(n) if s not in attackers]
    return {"detection_rate": float(np.mean(detected)),
            "false_alarm_rate": float(np.mean(false_alarm)),
            "window": val.window, "usable_cells": val.usable_cells(val.window), "alpha": alpha}


def policy_for(net, row: dict):
    """Rebuild the attack described by an `attack()` row, for the detector to be run against."""
    hon = honest_symbols(net)
    if row["strategy"] == "constant":
        return ConstantLiar(row["attackers"], row["symbol"], hon)
    if row["strategy"] == "permutation":
        return StealthyLiar(row["attackers"], torch.as_tensor(row["permutation"]), hon)
    return MarginalMatchedLiar(row["attackers"], torch.as_tensor(row["assignment"]), hon)


# --------------------------------------------------------------------------- experiment


def adversarial_sweep(bits_list=(2, 4, 6), attacker_counts=(1, 2, 3), steps: int = 8000,
                      n_pairs: int = 8, seed: int = 0, tag: str = "adversarial") -> list[dict]:
    """Damage and detectability, across budgets and numbers of colluding attackers."""
    from dataset import build_pool
    from regime import AREA_M
    from train import Config, train

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0)
    te = build_pool(size=2048, n_pairs=n_pairs, area_m=AREA_M, seed=999, lambdas=LAMBDAS)
    out = []
    for bits in bits_list:
        cfg = Config(bits=bits, steps=steps, seed=seed, usage_bonus=0.2)
        net = train(cfg, tr)
        for k in attacker_counts:
            for strategy in ("constant", "permutation", "matched"):
                row = attack(net, te, k, bits, seed=seed, strategy=strategy)
                row.update({f"detect_{a}": b for a, b in
                            detection(net, te, row["attackers"], policy_for(net, row), bits,
                                      seed=seed).items()})
                out.append(row)
                print(f"  B={bits} {k} attacker(s) {strategy:8s}: "
                      f"objective {100*row['objective_change']:+6.1f}%  "
                      f"attacker {100*row['attacker_gain']:+6.1f}%  "
                      f"victims {100*row['victim_loss']:+6.1f}%  "
                      f"detected {row['detect_detection_rate']:.2f} "
                      f"(false alarm {row['detect_false_alarm_rate']:.2f})", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from dataset import build_pool
    from regime import AREA_M
    from train import Config, train

    torch.manual_seed(0)
    tr = build_pool(size=1024, n_pairs=6, area_m=AREA_M, seed=0)
    te = build_pool(size=512, n_pairs=6, area_m=AREA_M, seed=999, lambdas=(0.5,))
    cfg = Config(bits=4, steps=1200, seed=0, usage_bonus=0.2)
    net = train(cfg, tr)

    # 1. The honest symbol_fn must reproduce the network's own behaviour exactly. If it does not,
    #    every "attack effect" below is really a substitution artefact.
    net.eval()
    lam = torch.full((len(te),), 0.5, device=te.gains.device)
    node, edge = graph_inputs(te.gains_obs, lam, norm=getattr(net, "norm", None))
    with torch.no_grad():
        p_native = net(node, edge)
        p_hook = net(node, edge, symbol_fn=honest_symbols(net))
    gap = float((p_native - p_hook).abs().max())
    print(f"honest hook reproduces the native forward pass: max |dP| = {gap:.2e}")
    assert gap < 1e-6, "the substitution hook is not neutral"
    net.train()

    # 2. A lie must actually change the outcome, and the best lie must beat an average one.
    r = attack(net, te, n_attackers=1, bits=4)
    print(f"\n1 attacker of 6, B=4, symbol {r['symbol']}:")
    print(f"  attacker rate {r['attacker_rate_before']:.3f} -> {r['attacker_rate_after']:.3f} "
          f"({100*r['attacker_gain']:+.1f}%)")
    print(f"  victim rate   {r['victim_rate_before']:.3f} -> {r['victim_rate_after']:.3f} "
          f"({100*r['victim_loss']:+.1f}%)")
    print(f"  social objective {100*r['objective_change']:+.1f}%")
    assert r["attacker_gain"] > 0, "the searched lie does not help the attacker"

    # 3. More attackers should not help the victims.
    print()
    for k in (1, 2, 3):
        rk = attack(net, te, n_attackers=k, bits=4)
        print(f"  {k} attacker(s): social {100*rk['objective_change']:+6.1f}%  "
              f"attacker {100*rk['attacker_gain']:+6.1f}%  victims {100*rk['victim_loss']:+6.1f}%")

    # 4. Detection, with its false-alarm rate attached.
    d = detection(net, te, r["attackers"], policy_for(net, r), bits=4)
    print(f"\n  constant liar -- detector catches it {d['detection_rate']:.2f} of windows, "
          f"flags honest senders {d['false_alarm_rate']:.2f}")

    # 5. The two stealthy attacks, and what actually decides whether an attack is invisible.
    #
    #    A test on the symbol marginal can see exactly one quantity: the total variation between
    #    what the attacker emits and what it would have emitted honestly. So that is what this
    #    tests, rather than testing detection alone and inferring the mechanism.
    #
    #    Only the matched attack drives that distance to zero. Permuting symbol labels preserves
    #    the *multiset* of counts, not the marginal -- it moves each count to a different symbol,
    #    which a goodness-of-fit test reads off directly. It is marginal-preserving only when the
    #    honest marginal is uniform, and ours is not (counts here span roughly 45 to 250).
    #
    #    An earlier version of this test asserted invisibility for *both* attacks. It passed only
    #    because the detector then used a fixed 64-symbol window, which is too underpowered at
    #    B=4 to resolve either one. Fixing the window to 10*2^B exposed the false claim. Keep the
    #    assertions tied to TV: a test that checks detection alone cannot tell an attack that is
    #    invisible from a detector that cannot see.
    lam = torch.full((len(te),), 0.5, device=te.gains.device)
    node, edge = graph_inputs(te.gains_obs, lam, norm=getattr(net, "norm", None))
    honest_sym = honest_symbols(net)(node, edge).cpu()

    def tv_to_honest(sym: torch.Tensor, s0: int) -> float:
        """Total variation between what sender s0 emits and what it would emit honestly."""
        others = [r for r in range(6) if r != s0]
        h = torch.bincount(sym[:, others, s0].reshape(-1), minlength=16).float()
        hh = torch.bincount(honest_sym[:, others, s0].reshape(-1), minlength=16).float()
        return 0.5 * float((h / h.sum() - hh / hh.sum()).abs().sum())

    for strat in ("permutation", "matched"):
        rs = attack(net, te, n_attackers=1, bits=4, strategy=strat)
        pol = policy_for(net, rs)
        ds = detection(net, te, rs["attackers"], pol, bits=4)
        with torch.no_grad():
            emitted = pol(node, edge).cpu()
        s0 = rs["attackers"][0]
        tv = tv_to_honest(emitted, s0)
        floor = max(0.15, 3 * ds["false_alarm_rate"] + 0.05)
        print(f"\n  {strat:11s} -- attacker {100*rs['attacker_gain']:+.1f}%  "
              f"victims {100*rs['victim_loss']:+.1f}%  social {100*rs['objective_change']:+.1f}%")
        print(f"                 TV to honest histogram {tv:.4f}")
        print(f"                 detector {ds['detection_rate']:.2f} of windows, "
              f"honest senders {ds['false_alarm_rate']:.2f}")

        if strat == "matched":
            assert tv < 0.02, "the matched attack is not actually marginal-preserving"
            assert ds["detection_rate"] <= floor, \
                "the matched attack should be invisible to a test on the symbol marginal"
        else:
            perm = torch.as_tensor(rs["permutation"])
            if bool((perm == torch.arange(len(perm))).all()):
                # The search found nothing better than honesty, so there is no attack to detect.
                assert tv < 1e-9, "the identity permutation must emit honest symbols exactly"
                assert ds["detection_rate"] <= floor, \
                    "an identity permutation must not be flagged above the false-alarm rate"
            else:
                assert tv > 0.02, (
                    "a non-identity permutation of a non-uniform marginal must move the histogram; "
                    "if this fails the honest code has gone uniform and the arm is degenerate")
                assert ds["detection_rate"] > floor, \
                    "the permutation attack moves the marginal and must therefore be detectable"
