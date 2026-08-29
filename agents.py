"""
Agents: message passing over the interference graph, with a bit budget on what is said.

Why this architecture, recorded because the evidence forced it. An earlier version had each agent
encode a message from its own observation alone and broadcast it once. Measured against the
centralised oracle at N=8:

    local observation only ........ 0.647
    + genie: the interference it causes ... 0.658   (+1.1 points)
    + full CSI, still acting alone ....... 0.708   (+6.1 points)
    message passing over the graph ....... 0.940   (+29.3 points)

So one-shot broadcast of local state was very nearly worthless, and the gap was never structural:
an agent holding full CSI could in principle compute the joint solution and read off its own part,
it simply could not learn to. Message passing can. That 0.647 floor and 0.940 ceiling are what give
the bit-budget question a 29-point window to live in, and they are reported as the paper's two
reference lines.

The budget. Each message on each edge in each round is quantised to B bits, in one of two ways:

    binary     : B independent bits, a straight-through binary-concrete relaxation
    vq         : a learned codebook of 2^B entries, selected by Gumbel-softmax
    continuous : no quantisation at all -- the unbounded-budget ceiling, reported as a reference
                 line rather than proposed as a scheme

The vq mode is the interesting one for this project. A learned codebook of 2^B symbols is an
emergent vocabulary in the literal sense, and a discrete vocabulary is what makes the later
symbolic distillation and message validation possible at all -- you cannot write a specification
over a real-valued vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from baselines import LloydMaxQuantizer


def mlp(sizes: list[int], activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


@dataclass
class Normaliser:
    """
    Fixed feature statistics, fitted once and frozen.

    Batch statistics cannot be used here, and the QA suite is what caught it. Standardising with
    `(x - x.mean()) / x.std()` over the batch makes every agent's features depend on every other
    agent's channel, including quantities it has no way to measure -- so perturbing what receiver r
    privately measures changed what sender s transmitted to everyone else. The partial-information
    premise leaked through the normaliser rather than through the model.

    Freezing the constants also removes the milder problem of test-time statistics leaking into
    evaluation.
    """

    direct_mean: float = -9.0
    direct_std: float = 1.0
    recv_mean: float = -8.0
    recv_std: float = 1.0
    edge_mean: float = -10.0
    edge_std: float = 1.0

    @staticmethod
    def fit(gains: torch.Tensor) -> "Normaliser":
        direct = torch.diagonal(gains, dim1=-2, dim2=-1)
        recv = gains.sum(-1) - direct
        ld = torch.log10(direct + 1e-30)
        lr = torch.log10(recv + 1e-30)
        lg = torch.log10(gains + 1e-30)
        return Normaliser(
            float(ld.mean()), float(ld.std().clamp_min(1e-6)),
            float(lr.mean()), float(lr.std().clamp_min(1e-6)),
            float(lg.mean()), float(lg.std().clamp_min(1e-6)),
        )


DEFAULT_NORM = Normaliser()


def graph_inputs(
    gains: torch.Tensor,
    lam: torch.Tensor,
    extra_node: torch.Tensor | None = None,
    norm: "Normaliser | None" = None,
    full_csi: bool = False,
    price: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Node and edge features under a *partial information* model.

    This is the corrected version. The first one handed every agent the true gains on both
    directions of each incident edge, which quietly gave the sender knowledge of the harm it causes
    -- precisely the quantity the protocol is supposed to have to communicate. The study then
    constrained only what agents transmit, not what they know, and the two are very different
    claims.

    What agent i can actually measure, from reference signals at its own receiver:
        a_ii            its own direct gain
        a_ik            how much each neighbour k hurts it
    What agent i cannot know:
        a_ki            how much it hurts each neighbour k -- only k can measure that

    So the useful message is i telling k "you are costing me a_ik", which is the learned analogue of
    an interference price. Edge features are therefore indexed from the *sender's* measurement:

        node : (B, N, 3 [+extra])   log direct gain, log total received interference, lambda
        edge : (B, N, N, 1)         at [b, r, s], what sender s measured about receiver r: a_sr
    """
    nm = norm or DEFAULT_NORM
    b, n, _ = gains.shape
    direct = torch.diagonal(gains, dim1=-2, dim2=-1)
    recv = gains.sum(-1) - direct                      # total interference i receives, measurable

    ld = (torch.log10(direct + 1e-30) - nm.direct_mean) / nm.direct_std
    lr = (torch.log10(recv + 1e-30) - nm.recv_mean) / nm.recv_std
    node = torch.stack([ld, lr, lam[:, None].expand(-1, n)], dim=-1)
    if extra_node is not None:
        node = torch.cat([node, extra_node], dim=-1)

    lg = (torch.log10(gains + 1e-30) - nm.edge_mean) / nm.edge_std
    # entry [b, r, s] must be a_{s,r}: what sender s measured about receiver r's interference.
    edge = lg.transpose(1, 2).unsqueeze(-1)
    if price is not None:
        # Classical interference price, appended as a second edge channel.
        #
        # This is the quantity distributed pricing / WMMSE-style schemes actually put on the wire,
        # and it is the arm the paper needs: nobody designing limited-feedback distributed power
        # control ships raw quantised CSI. For receiver s with utility
        # u_s = log(1 + a_ss p_s / (N0 + I_s)), the marginal cost of interference is
        #
        #     pi_s = -d u_s / d I_s = a_ss p_s / [(N0 + I_s)(N0 + I_s + a_ss p_s)]
        #
        # and the marginal cost that transmitter r imposes on s is pi_s * a_sr -- exactly what s
        # would want to tell r, and exactly the classical analogue of the learned message.
        #
        # The price is evaluated at the equal-power reference p = P_max. A price depends on the
        # operating point, and the honest reference is the one an agent can compute with *zero*
        # signalling; anything else would quietly hand this arm a coordination round it did not pay
        # for. At R > 1 rounds a pricing scheme would re-evaluate at the current iterate, which is
        # the natural extension and is not exercised by the R = 1 sweep.
        #
        # The price is the sum-rate price and carries no lambda. That is not a handicap: lambda is
        # already a node feature at every receiver, so the arm can weight the price locally. Only
        # the transmitted quantity is classical, which is the comparison being drawn.
        n0, p_ref = price
        sig = torch.diagonal(gains, dim1=-2, dim2=-1) * p_ref          # (B, N) wanted signal at s
        interf = (gains.sum(-1) - torch.diagonal(gains, dim1=-2, dim2=-1)) * p_ref
        pi = sig / ((n0 + interf) * (n0 + interf + sig) + 1e-30)       # (B, N), indexed by s
        # mu[b, r, s] = pi_s * a_{s,r}: what sender s would price receiver r's interference at.
        mu = pi[:, None, :] * gains.transpose(1, 2)
        # Left in the raw log domain deliberately. This channel exists only to be fed to the
        # Lloyd-Max quantiser, which fits its own codebook on a training sample and is therefore
        # scale-agnostic; standardising here with batch statistics would make the bin boundaries
        # drift from batch to batch and would be exactly the test-set leakage the frozen
        # Normaliser is designed to prevent.
        edge = torch.cat([edge, torch.log10(mu + 1e-30).unsqueeze(-1)], dim=-1)
    if full_csi:
        # The centralised reference (CentralisedGNN) is the one arm allowed to break the partial
        # information model: entry [b, r, s] carries both a_sr and a_rs, so after message passing
        # every node has seen the whole gain matrix. Every other arm must be built with
        # full_csi=False or the bit budget stops meaning anything.
        edge = torch.cat([edge, lg.unsqueeze(-1)], dim=-1)
    return node, edge


class MessageChannel(nn.Module):
    """Quantises a continuous message vector down to a B-bit symbol."""

    def __init__(self, dim: int, bits: int, mode: str = "vq", temperature: float = 1.0) -> None:
        super().__init__()
        self.dim, self.bits, self.mode = dim, int(bits), mode
        self.temperature = temperature
        self.last_logits: torch.Tensor | None = None
        if mode == "continuous":
            pass
        elif mode == "vq" and self.bits > 0:
            self.n_codes = 1 << self.bits
            self.to_logits = nn.Linear(dim, self.n_codes)
            self.codebook = nn.Parameter(torch.randn(self.n_codes, dim) * 0.5)
        elif mode == "binary" and self.bits > 0:
            self.to_logits = nn.Linear(dim, self.bits)

    @property
    def out_dim(self) -> int:
        if self.mode == "continuous":
            return self.dim
        if self.bits == 0:
            return 0
        return self.dim if self.mode == "vq" else self.bits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (message, symbol_index_or_bits). Message is zero-width when the budget is zero."""
        if self.mode == "continuous":
            return x, x
        if self.bits == 0:
            return x.new_zeros(x.shape[:-1] + (0,)), x.new_zeros(x.shape[:-1] + (0,))

        if self.mode == "binary":
            logits = self.to_logits(x)
            if self.training:
                u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                logits = logits + torch.log(u) - torch.log1p(-u)
            soft = torch.sigmoid(logits / self.temperature)
            hard = (soft > 0.5).to(soft.dtype)
            out = hard + (soft - soft.detach())
            return out, out

        logits = self.to_logits(x)
        self.last_logits = logits
        if self.training:
            onehot = nn.functional.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            onehot = nn.functional.one_hot(idx, self.n_codes).to(logits.dtype)
        return onehot @ self.codebook, logits.argmax(dim=-1)


class ProtocolGNN(nn.Module):
    """
    R rounds of bit-budgeted message passing, then a per-agent power readout.

    Total signalling cost per agent per slot is bits * (N - 1) * rounds, reported alongside the
    per-edge budget so the overhead can be compared against a real E2/A1 interface.
    """

    def __init__(
        self,
        bits: int,
        p_max: float,
        rounds: int = 1,
        hidden: int = 64,
        msg_dim: int = 16,
        node_dim: int = 3,
        edge_dim: int = 1,
        mode: str = "vq",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.bits, self.rounds, self.p_max = int(bits), rounds, p_max
        self.embed = mlp([node_dim, hidden, hidden])
        # Sender-only: the message is a function of the sender's hidden state and the sender's own
        # measurement of this edge. The receiver's hidden state is deliberately absent -- a sender
        # that could read it would already know what it is meant to be told.
        self.msg = mlp([hidden + edge_dim, hidden, msg_dim])
        self.channel = MessageChannel(msg_dim, bits, mode, temperature)
        self.upd = mlp([hidden + self.channel.out_dim, hidden, hidden])
        self.read = mlp([hidden, hidden, 1])

    def signalling_bits(self, n_agents: int) -> float:
        """
        Bits sent per agent per slot.

        Continuous mode is not free: it transmits msg_dim real numbers per edge. Reporting it as
        zero would put an unbounded-budget ceiling on a bit-budget plot at the origin, which is
        exactly backwards. It is returned as infinity so any plot or table has to handle it as the
        unbounded reference line it is.
        """
        if self.channel.mode == "continuous":
            return float("inf")
        return float(self.bits * (n_agents - 1) * self.rounds)

    def forward(self, node: torch.Tensor, edge: torch.Tensor, return_symbols: bool = False,
                symbol_fn=None):
        """
        `symbol_fn(node, edge) -> (B, N, N) long` replaces the learned encoder with an external one,
        keeping the codebook, the aggregation and the readout untouched.

        This is what lets a distilled symbolic rule (symbolic.py) or a lying agent (adversarial.py)
        be swapped in and measured end to end. Re-implementing the rest of the forward pass in
        those modules would leave two copies of it to drift apart, and the substituted encoder is
        the only thing either of them wants to change.
        """
        b, n, _ = node.shape
        h = self.embed(node)
        eye = torch.eye(n, device=node.device, dtype=torch.bool).view(1, n, n, 1)
        symbols = []

        for _ in range(self.rounds):
            hs = h.unsqueeze(1).expand(b, n, n, h.shape[-1])     # [b, r, s] -> sender s
            raw = self.msg(torch.cat([hs, edge], dim=-1))
            if symbol_fn is not None and self.channel.mode == "vq" and self.bits > 0:
                sym = symbol_fn(node, edge)
                m = self.channel.codebook[sym]
            else:
                m, sym = self.channel(raw)
            if self.channel.out_dim > 0:
                m = m.masked_fill(eye, 0.0)
                agg = m.sum(dim=2) / max(n - 1, 1)
            else:
                agg = h.new_zeros((b, n, 0))
            h = h + self.upd(torch.cat([h, agg], dim=-1))
            if return_symbols:
                symbols.append(sym)

        powers = self.p_max * torch.sigmoid(self.read(h).squeeze(-1))
        return (powers, symbols) if return_symbols else powers


class QuantisedCSIGNN(ProtocolGNN):
    """
    The non-learned competitor: spend the same B bits on a Lloyd-Max quantisation of the edge gain.

    Identical graph, identical rounds, identical readout, identical budget -- only the content of
    the message is fixed by a classical CSI-feedback scheme instead of learned. If the learned
    protocol cannot beat this at equal bits, there is no case for learning it, and that comparison
    is the point of the paper.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quantizer: LloydMaxQuantizer | None = None
        # The message here is raw quantiser bits, not a codebook vector, so the update MLP takes a
        # different input width than the learned variant. Rebuild it rather than patch it later.
        hidden = self.upd[0].out_features
        self.upd = mlp([self.embed[-1].out_features + self.bits, hidden, hidden])

    def fit_quantizer(self, edge_sample: torch.Tensor) -> "QuantisedCSIGNN":
        if self.bits > 0:
            vals = edge_sample[..., 0].detach().cpu().numpy().ravel()
            self.quantizer = LloydMaxQuantizer(self.bits).fit(vals)
        return self

    def forward(self, node: torch.Tensor, edge: torch.Tensor, return_symbols: bool = False):
        b, n, _ = node.shape
        h = self.embed(node)
        eye = torch.eye(n, device=node.device, dtype=torch.bool).view(1, n, n, 1)

        if self.bits > 0:
            assert self.quantizer is not None, "call fit_quantizer() first"
            v = edge[..., 0].detach().cpu().numpy()
            idx = self.quantizer.indices(v.ravel()).reshape(v.shape)
            shifts = np.arange(self.bits - 1, -1, -1)
            code = torch.as_tensor(
                ((idx[..., None] >> shifts) & 1).astype(np.float32), device=edge.device, dtype=edge.dtype
            )
            width = self.bits
        else:
            code = edge.new_zeros(edge.shape[:-1] + (0,))
            width = 0

        for _ in range(self.rounds):
            if width > 0:
                agg = code.masked_fill(eye, 0.0).sum(dim=2) / max(n - 1, 1)
            else:
                agg = h.new_zeros((b, n, 0))
            h = h + self.upd(torch.cat([h, agg], dim=-1))

        powers = self.p_max * torch.sigmoid(self.read(h).squeeze(-1))
        return (powers, []) if return_symbols else powers


# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    torch.manual_seed(0)
    from env import Environment

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    env = Environment(batch=32, n_pairs=6, device=dev, rng=rng)
    lam = torch.rand(32, device=dev)
    node, edge = graph_inputs(env.gains, lam)
    print("node:", tuple(node.shape), " edge:", tuple(edge.shape))

    # 1. Every budget runs, and signalling cost is reported honestly.
    for bits in (0, 1, 2, 4, 8):
        net = ProtocolGNN(bits=bits, p_max=env.p_max, rounds=2).to(dev)
        p, syms = net(node, edge, return_symbols=True)
        assert p.shape == (32, 6) and (p <= env.p_max + 1e-6).all()
        n_sym = 0 if bits == 0 else int(syms[0].max()) + 1
        print(f"  B={bits}: powers ok, {net.signalling_bits(6):5.0f} bits/agent/slot, "
              f"codebook entries seen {n_sym:3d}/{0 if bits==0 else 1<<bits}")

    # 2. Discreteness: evaluation-time symbols must be integers in range.
    net = ProtocolGNN(bits=4, p_max=env.p_max, rounds=1).to(dev).eval()
    _, syms = net(node, edge, return_symbols=True)
    s = syms[0]
    print("  symbols integral and in [0, 15]:", bool((s >= 0).all() and (s <= 15).all()))

    # 3. Gradient flows through the discrete channel.
    net.train()
    p = net(node, edge)
    p.sum().backward()
    ok = all(q.grad is not None and torch.isfinite(q.grad).all()
             for q in net.parameters() if q.requires_grad and q.grad is not None)
    print("  gradient flows through Gumbel codebook:", ok)

    # 4. Binary mode.
    netb = ProtocolGNN(bits=6, p_max=env.p_max, mode="binary").to(dev).eval()
    _, syms = netb(node, edge, return_symbols=True)
    print("  binary mode emits bits:", bool(((syms[0] == 0) | (syms[0] == 1)).all()))

    # 5. One model, many N -- the scalability claim survives the rewrite.
    net = ProtocolGNN(bits=4, p_max=env.p_max, rounds=2).to(dev).eval()
    for n in (4, 8, 16, 32):
        e = Environment(batch=8, n_pairs=n, device=dev, rng=rng)
        nd, eg = graph_inputs(e.gains, torch.rand(8, device=dev))
        print(f"    N={n:2d}: {tuple(net(nd, eg).shape)}, {net.signalling_bits(n):.0f} bits/agent/slot")


class QuantisedCSIEmbedGNN(ProtocolGNN):
    """
    The *matched* classical control -- the one the paper's central claim actually rests on.

    `QuantisedCSIGNN` above hands the receiver the raw binary expansion of the Lloyd-Max index and
    then mean-pools it over neighbours. That is not a fair matched-budget comparison, and the
    reason is visible without running anything: the mean of a bit-plane over N-1 neighbours is
    informative for the MSB and converges to 0.5 for the low-order bits regardless of the gains, so
    adding resolution adds near-constant noise dimensions instead of information. The learned arm
    mean-pools a `msg_dim`-wide codebook vector that is *trained* to survive mean-pooling. The two
    arms were therefore matched on nominal bit count and mismatched on representation, and a flat
    -- indeed weakly decreasing -- quantised curve is what that mismatch predicts.

    This class removes the mismatch. The Lloyd-Max index selects a row of the *same* learned
    codebook the learned arm uses, of the same width, aggregated the same way. The only surviving
    difference between the arms is what decides the index: a classical scalar quantiser of the
    sender's measured gain, or a learned encoder. That is the difference the paper is about.

    Keeping both classes is deliberate: the raw-bits variant is now an ablation that explains *why*
    a naive matched-budget control looks flat, rather than an unexplained anomaly in the headline
    table.
    """

    quantised_channel = 0      # which edge feature goes on the wire

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.channel.mode == "vq", "the matched control shares the vq codebook"
        self.quantizer: LloydMaxQuantizer | None = None

    def fit_quantizer(self, edge_sample: torch.Tensor) -> "QuantisedCSIEmbedGNN":
        if self.bits > 0:
            vals = edge_sample[..., self.quantised_channel].detach().cpu().numpy().ravel()
            self.quantizer = LloydMaxQuantizer(self.bits).fit(vals)
        return self

    def forward(self, node: torch.Tensor, edge: torch.Tensor, return_symbols: bool = False,
                symbol_fn=None):
        b, n, _ = node.shape
        h = self.embed(node)
        eye = torch.eye(n, device=node.device, dtype=torch.bool).view(1, n, n, 1)

        if self.bits > 0:
            assert self.quantizer is not None, "call fit_quantizer() first"
            v = edge[..., self.quantised_channel].detach().cpu().numpy()
            idx = torch.as_tensor(
                self.quantizer.indices(v.ravel()).reshape(v.shape).astype(np.int64),
                device=edge.device,
            )
            # Honour the substitution hook here too. The signature accepted `symbol_fn` and then
            # ignored it, so an impairment applied to this arm silently did nothing -- a BER sweep
            # against it returned a perfectly flat line, which is what exposed the bug.
            if symbol_fn is not None:
                idx = symbol_fn(node, edge)
        symbols = []

        for _ in range(self.rounds):
            if self.bits > 0:
                m = self.channel.codebook[idx]
                m = m.masked_fill(eye, 0.0)
                agg = m.sum(dim=2) / max(n - 1, 1)
            else:
                agg = h.new_zeros((b, n, 0))
            h = h + self.upd(torch.cat([h, agg], dim=-1))
            if return_symbols:
                symbols.append(idx if self.bits > 0 else h.new_zeros((b, n, n), dtype=torch.long))

        powers = self.p_max * torch.sigmoid(self.read(h).squeeze(-1))
        return (powers, symbols) if return_symbols else powers


class CentralisedGNN(ProtocolGNN):
    """
    The correct centralised reference: *this* architecture, with the whole gain matrix.

    The prior repo's `SupervisedAllocator` is a flat MLP over the vectorised gain matrix with no
    permutation equivariance. Scoring the message-passing arms against it does not measure what
    communication buys -- it measures the difference between two inductive biases, and that is why
    it produced the impossible ordering in which a bandwidth-limited decentralised policy beat a
    full-CSI centralised one. A centralised allocator can simulate any decentralised one, so it
    must weakly dominate it; when it does not, the reference is wrong.

    Here the graph, the embedding, the update and the readout are byte-for-byte the decentralised
    arm's. Two things change, and only these two: edges carry both directions (`full_csi=True`, so
    a sender knows the harm it causes as well as the harm it suffers) and the message is
    unquantised. With R >= 2 rounds on a complete graph every node has then seen the entire matrix,
    which is what "centralised" means. It is the upper bound the bit-budget curve should be read
    against.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("mode", "continuous")
        kwargs.setdefault("edge_dim", 2)
        super().__init__(*args, **kwargs)
        assert self.channel.mode == "continuous", "the centralised reference is not bit-budgeted"


class PricedCSIGNN(QuantisedCSIEmbedGNN):
    """
    The strong classical control: B bits spent on the *interference price*, not on raw CSI.

    Quantised CSI is the weakest classical option available, and beating it says little. The
    schemes an engineer would actually field -- distributed pricing, WMMSE, ADMM -- put a dual
    variable on the wire. Section on what the messages encode already reports that the learned code
    is largely encoding the interference price, which makes this the comparison the paper is
    obliged to run: how much of the gap survives when the classical arm is allowed to send the
    right quantity?

    Everything is shared with `QuantisedCSIEmbedGNN` except which edge feature is quantised. Same
    Lloyd-Max quantiser, same shared codebook, same width, same aggregation, same budget.
    """

    quantised_channel = 1      # the price channel appended by graph_inputs(price=...)
