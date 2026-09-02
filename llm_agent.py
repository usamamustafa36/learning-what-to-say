"""
The generative-AI arm: an LLM allocating power from a text description of the network.

Large language models and generative AI are where the prior repo begins -- `llm-d2d-resource-allocation`
is named for it. It should be said plainly that the repo
contains no LLM: no `transformers`, no PEFT, no LoRA, no adapter weights, and the Phi-3 / Llama /
Mistral figures in its summary are labelled there as "Prior Work: ICC 2025 Paper", i.e. someone
else's measurements. Those numbers are cited in this project as prior work and are not reproduced as
though they were ours.

What is run here instead is a real, local, small instruct model -- Qwen2.5-0.5B-Instruct -- given the
same observation the centralised allocator gets, asked for a power vector, and scored against the
same genie oracle as every other arm. Three prompting regimes, which are the ones the prior README
described:

    zero-shot   : the task, the numbers, the output format
    few-shot    : k solved instances in the prompt
    retrieval   : the k *nearest* solved instances, by channel similarity -- prompt-level RAG

This is a deployment-cost measurement, not a ceiling on what LLMs can do at this task. A 0.5B model
is small, and a larger one would score better; what does not change with scale is the shape of the
cost, and that is the axis this arm exists to populate. Reported alongside: parameters, model size,
and seconds per allocation, against a protocol network of ~50 KB deciding in under a millisecond.

One metric here has no analogue in the other arms and is reported because it is real: the
**parse failure rate**. A text interface can return something that is not a power vector. When it
does, the allocation falls back to equal power, and the rate is stated rather than quietly dropped
-- an allocator that fails to answer 8% of the time has an availability problem, not just an
accuracy one.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from regime import AREA_M, CIRCUIT_POWER_W, N_PAIRS, P_MAX_W

RESULTS = Path(__file__).parent / "results"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _quiet_transformers() -> None:
    """Progress bars and generation warnings make the experiment logs unreadable; the run is
    reproducible from the seed, not from watching weights load."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _stub_torchvision() -> bool:
    """
    Environment shim: torchvision is installed here but its compiled ops are not registered, so
    `import torchvision` raises `operator torchvision::nms does not exist`.

    `transformers` imports torchvision unconditionally through `image_utils`, on the path to every
    model including text-only ones, so a broken torchvision makes Qwen unloadable for reasons that
    have nothing to do with Qwen. Rather than reinstall packages in someone else's environment,
    a namespace stub is installed *before* transformers is imported. Nothing here is called: this
    arm never touches an image.

    Returns True if the stub was installed, so the experiment record can say so.
    """
    import types

    try:
        import torchvision  # noqa: F401
        return False
    except Exception:
        pass

    class _AnyAttr(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            v = type(name, (), {})
            setattr(cls, name, v)
            return v

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            v = _AnyAttr(name, (), {})
            setattr(self, name, v)
            return v

    for name in ("torchvision", "torchvision.transforms", "torchvision.transforms.v2",
                 "torchvision.transforms.functional", "torchvision.io", "torchvision.ops"):
        mod = _Stub(name)
        mod.__version__ = "0.0.0-stub"
        mod.__file__ = f"<stub {name}>"
        mod.__path__ = []
        mod.__spec__ = types.SimpleNamespace(name=name, origin="<stub>")
        mod.__all__ = []
        sys.modules[name] = mod
    return True


_quiet_transformers()
TORCHVISION_STUBBED = _stub_torchvision()

SYSTEM = (
    "You are a radio resource controller. You allocate transmit power to N transmitter-receiver "
    "pairs sharing one channel. Higher power raises your own rate but interferes with everyone "
    "else. Answer with the power vector only."
)


class Rendering:
    """
    How a power vector is written into the prompt and read back out of the reply.

    This is not cosmetic. The oracle's median power is 3.74 mW, so whole-milliwatt integers flatten
    45.8% of the exemplars' entries to a literal `0`; the model copies that sparsity and answers
    `[0, 0, ...]` for every instance, scoring exactly 0.0 of oracle. Change the rendering and the
    same model on the same data scores up to 0.50. Reporting a single rendering therefore reports
    an arbitrary point in that range, so `llm_rendering_sweep` measures all of them and the paper
    quotes each arm's best.

    `unit_per_w` converts watts into the unit the prompt talks in; `decimals` is how the number is
    written; `instruction` is the sentence that tells the model what to reply with. The three must
    agree, or the model is asked for one format and scored against another.
    """

    def __init__(self, name: str, unit_per_w: float, decimals: int, instruction: str) -> None:
        self.name, self.unit_per_w, self.decimals = name, unit_per_w, decimals
        self.instruction = instruction

    def render(self, p_w: np.ndarray) -> str:
        v = np.asarray(p_w, dtype=float) * self.unit_per_w
        return "[" + ", ".join(f"{x:.{self.decimals}f}" for x in v) + "]"

    def ceiling(self, p_max_w: float) -> float:
        return p_max_w * self.unit_per_w

    def instruction_for(self, n: int, p_max_w: float) -> str:
        return self.instruction.format(n=n, ceil=self.ceiling(p_max_w))


RENDERINGS = {
    # The original. Kept first because it is what every result before 24 Aug 2026 used.
    "int-mw": Rendering(
        "int-mw", 1e3, 0,
        "Reply with exactly {n} integers in mW, comma separated, inside square brackets. "
        "No other text."),
    "dec-mw": Rendering(
        "dec-mw", 1e3, 1,
        "Reply with exactly {n} numbers in mW to one decimal place, comma separated, inside "
        "square brackets. No other text."),
    "int-dmw": Rendering(
        "int-dmw", 1e4, 0,
        "Reply with exactly {n} integers, each in units of 0.1 mW (so 100 means 10 mW, range 0 to "
        "{ceil:.0f}), comma separated, inside square brackets. No other text."),
}

DEFAULT_RENDERING = "int-mw"


def describe(gains: np.ndarray, lam: float, p_max_mw: float = 100.0,
             rendering: str = DEFAULT_RENDERING) -> str:
    """
    Render one instance as text.

    Gains are given in dB, which is how they are discussed in the field and is far kinder to a
    tokeniser than 3.7e-10. The diagonal is separated from the interference matrix because that is
    the structure of the problem and burying it in a flat list makes the task harder for no reason
    that has anything to do with allocation.
    """
    n = gains.shape[0]
    db = 10.0 * np.log10(gains + 1e-30)
    direct = np.diag(db)
    lines = [f"N = {n} pairs. Transmit power range: 0 to {p_max_mw:.0f} mW per transmitter.",
             f"Preference weight lambda = {lam:.2f} "
             f"(1.0 = maximise total throughput, 0.0 = maximise energy efficiency).",
             "",
             "Direct link gain of each pair, dB:"]
    lines.append("  " + ", ".join(f"pair {i}: {direct[i]:.1f}" for i in range(n)))
    lines.append("")
    lines.append("Interference gain from transmitter j to receiver i, dB (row i, column j):")
    for i in range(n):
        row = ", ".join(f"{db[i, j]:.1f}" for j in range(n) if j != i)
        lines.append(f"  receiver {i}: {row}")
    lines.append("")
    lines.append(RENDERINGS[rendering].instruction_for(n, p_max_mw / 1000.0))
    return "\n".join(lines)


def parse_powers(text: str, n: int, p_max_mw: float = 100.0,
                 rendering: str = DEFAULT_RENDERING) -> np.ndarray | None:
    """
    Pull an N-vector out of whatever the model said. None means it did not answer the question.

    The numbers are in the rendering's unit, not necessarily mW, so they are scaled back to watts
    through the same `unit_per_w` used to write the prompt. Asking in one unit and reading in
    another silently rescales every allocation.

    An all-zero vector parses *successfully*: it clips to [0, p_max] and returns a valid, terrible
    allocation. That is deliberate -- "transmit nothing" is an answer, not a refusal -- but it means
    a degenerate reply is invisible in the parse-failure rate, so `allocate` counts those separately
    as `degenerate_zero_replies`.
    """
    r = RENDERINGS[rendering]
    m = re.search(r"\[([^\]]*)\]", text)
    body = m.group(1) if m else text
    nums = re.findall(r"-?\d+\.?\d*", body)
    if len(nums) < n:
        return None
    vals = np.array([float(x) for x in nums[:n]])
    if not np.isfinite(vals).all():
        return None
    return np.clip(vals, 0.0, r.ceiling(p_max_mw / 1000.0)) / r.unit_per_w


class LLMAllocator:
    """A local instruct model behind the same interface as every other arm."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 max_new_tokens: int = 48) -> None:
        import logging

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device, self.max_new_tokens, self.model_id = device, max_new_tokens, model_id

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def size_mb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2

    def _chat(self, prompts: list[str]) -> list[str]:
        texts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    def allocate(self, gains: np.ndarray, lam: float, shots: list[str] | None = None,
                 batch: int = 8, rendering: str = DEFAULT_RENDERING) -> tuple[np.ndarray, dict]:
        """gains (M, N, N) -> powers (M, N), plus timing and parse statistics."""
        m, n, _ = gains.shape
        prompts = []
        for i in range(m):
            body = describe(gains[i], lam, rendering=rendering)
            prompts.append("\n\n".join(shots + [body]) if shots else body)

        powers, failures, degenerate = np.zeros((m, n)), 0, 0
        replies: list[str] = []
        t0 = time.perf_counter()
        for k in range(0, m, batch):
            for j, txt in enumerate(self._chat(prompts[k : k + batch])):
                replies.append(txt)
                p = parse_powers(txt, n, rendering=rendering)
                if p is None:
                    failures += 1
                    p = np.full(n, 0.05)                   # equal power fallback
                elif not p.any():
                    # Parses cleanly, allocates nothing. Scores 0 while counting as a success, so a
                    # row reading 0.0 of oracle next to 0% unparseable is otherwise unreadable.
                    degenerate += 1
                powers[k + j] = p
        dt = time.perf_counter() - t0
        # How many *distinct* replies came back. One distinct reply over 64 varied instances means
        # the model is copying a fixed vector rather than allocating, which a mean ratio hides.
        distinct = len({t.strip() for t in replies})
        return powers, {"seconds_per_instance": dt / m, "parse_failures": failures,
                        "parse_failure_rate": failures / m, "n_instances": m,
                        "degenerate_zero_replies": degenerate, "degenerate_zero_rate": degenerate / m,
                        "distinct_replies": distinct, "rendering": rendering}


def build_shots(pool, oracle_powers: torch.Tensor, lam: float, k: int = 3,
                idx: np.ndarray | None = None,
                rendering: str = DEFAULT_RENDERING) -> list[str]:
    """
    Solved examples, rendered in the same format as the question.

    The exemplar must be written through the *same* Rendering that produced the instruction. When
    these disagreed -- integer mW exemplars under any instruction -- 45.8% of exemplar entries
    became a literal 0 and the model reproduced the sparsity for every instance.
    """
    r = RENDERINGS[rendering]
    idx = np.arange(k) if idx is None else idx[:k]
    shots = []
    for i in idx:
        g = pool.gains_obs[int(i)].cpu().numpy()
        p = oracle_powers[int(i)].cpu().numpy()
        shots.append(describe(g, lam, rendering=rendering) + "\n" + r.render(p))
    return shots


def nearest(pool, query_idx: int, bank_idx: np.ndarray, k: int) -> np.ndarray:
    """
    Retrieval by channel similarity -- the RAG arm, without a vector database.

    Similarity is Euclidean distance between log-gain matrices, which is the quantity a text
    embedding of these prompts would be a noisy proxy for. Using it directly measures retrieval on
    its best day and keeps the arm about allocation rather than about embedding quality.
    """
    q = torch.log10(pool.gains_obs[query_idx] + 1e-30).ravel()
    bank = torch.log10(pool.gains_obs[bank_idx] + 1e-30).reshape(len(bank_idx), -1)
    d = torch.linalg.norm(bank - q, dim=-1)
    return bank_idx[d.argsort()[:k].cpu().numpy()]


# --------------------------------------------------------------------------- experiment


def llm_experiment(n_test: int = 64, n_pairs: int = N_PAIRS, lam: float = 0.5, k_shots: int = 3,
                   regimes=("zero-shot", "few-shot", "retrieval"), device: str = "cuda",
                   seed: int = 0, tag: str = "llm",
                   rendering: str = DEFAULT_RENDERING) -> list[dict]:
    """Score the LLM arm against the same oracle as everything else, and price it."""
    from dataset import build_pool
    from env import ee_torch, se_torch
    from solvers import oracle_batch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    pool = build_pool(size=n_test + 64, n_pairs=n_pairs, area_m=AREA_M, seed=999,
                      lambdas=(lam,), device=str(dev), p_max=P_MAX_W,
                      circuit_power_w=CIRCUIT_POWER_W)
    bank_idx = np.arange(n_test, n_test + 64)              # exemplars, disjoint from the test set

    with torch.no_grad():
        lam_t = torch.full((len(pool),), lam, device=dev)
        p_or = oracle_batch(pool.gains, pool.noise_power, P_MAX_W, lam_t,
                            pool.se_ref, pool.ee_ref, CIRCUIT_POWER_W, n_starts=16, n_steps=800)

    def score(p_np, idx):
        p = torch.as_tensor(p_np, dtype=torch.float32, device=dev)
        g = pool.gains[idx]
        se = se_torch(p, g, pool.noise_power) / pool.se_ref[idx].clamp_min(1e-12)
        ee = ee_torch(p, g, pool.noise_power, CIRCUIT_POWER_W) / pool.ee_ref[idx].clamp_min(1e-12)
        obj = lam * se + (1.0 - lam) * ee
        return float((obj / pool.oracle[lam][idx].clamp_min(1e-12)).mean())

    test = np.arange(n_test)
    gains_np = pool.gains_obs[test].cpu().numpy()
    llm = LLMAllocator(device=str(dev))
    out = []

    for regime in regimes:
        if regime == "zero-shot":
            p, stats = llm.allocate(gains_np, lam, rendering=rendering)
        elif regime == "few-shot":
            shots = build_shots(pool, p_or, lam, k_shots, bank_idx, rendering=rendering)
            p, stats = llm.allocate(gains_np, lam, shots=shots, rendering=rendering)
        else:
            p = np.zeros((n_test, n_pairs)); fails = 0; degen = 0; t0 = time.perf_counter()
            seen: set[str] = set()
            for i in test:
                near = nearest(pool, int(i), bank_idx, k_shots)
                shots = build_shots(pool, p_or, lam, k_shots, near, rendering=rendering)
                pi, si = llm.allocate(gains_np[i : i + 1], lam, shots=shots, batch=1,
                                      rendering=rendering)
                p[i] = pi[0]; fails += si["parse_failures"]; degen += si["degenerate_zero_replies"]
                seen.add(np.array2string(pi[0], precision=6))
            stats = {"seconds_per_instance": (time.perf_counter() - t0) / n_test,
                     "parse_failures": fails, "parse_failure_rate": fails / n_test,
                     "n_instances": n_test, "degenerate_zero_replies": degen,
                     "degenerate_zero_rate": degen / n_test, "distinct_replies": len(seen),
                     "rendering": rendering}
        row = {"arm": f"llm-{regime}", "model": llm.model_id, "regime": regime,
               "rendering": rendering,
               "mean_ratio": score(p, test), "parameters": llm.parameters,
               "size_mb": llm.size_mb, **stats}
        out.append(row)
        print(f"  {regime:10s}: {row['mean_ratio']:.4f} of oracle   "
              f"{row['seconds_per_instance']*1e3:8.1f} ms/instance   "
              f"parse failures {row['parse_failure_rate']:.1%}   "
              f"distinct replies {row['distinct_replies']:3d}/{n_test}", flush=True)

    # Reference points on the same instances, so the cost axis has something to be read against.
    from baselines import equal_power

    eq = np.stack([equal_power(gains_np[i], P_MAX_W) for i in test])
    out.append({"arm": "equal-power", "mean_ratio": score(eq, test), "parameters": 0,
                "size_mb": 0.0, "seconds_per_instance": 0.0, "parse_failure_rate": 0.0})
    print(f"  {'equal power':10s}: {out[-1]['mean_ratio']:.4f} of oracle   (reference)")

    # The learned protocol on *these* instances at *this* lambda. Quoting its number from the
    # bit-budget sweep instead would compare two different test sets and two different lambda
    # averages, which is the most common way a cost table misleads.
    from agents import graph_inputs
    from train import Config
    from train import train as train_protocol

    tr = build_pool(size=8192, n_pairs=n_pairs, area_m=AREA_M, seed=0, device=str(dev),
                    p_max=P_MAX_W, circuit_power_w=CIRCUIT_POWER_W)
    net = train_protocol(Config(bits=6, steps=8000, seed=0, usage_bonus=0.2), tr)
    net.eval()
    with torch.no_grad():
        lam_t = torch.full((n_test,), lam, device=dev)
        node, edge = graph_inputs(pool.gains_obs[test], lam_t, norm=getattr(net, "norm", None))
        t0 = time.perf_counter()
        p_net = net(node, edge)
        dt = (time.perf_counter() - t0) / n_test
    params = sum(q.numel() for q in net.parameters())
    out.append({"arm": "protocol-6bit", "mean_ratio": score(p_net.cpu().numpy(), test),
                "parameters": int(params),
                "size_mb": sum(q.numel() * q.element_size() for q in net.parameters()) / 1024**2,
                "seconds_per_instance": dt, "parse_failure_rate": 0.0})
    print(f"  {'protocol':10s}: {out[-1]['mean_ratio']:.4f} of oracle   "
          f"{dt*1e3:8.3f} ms/instance   {params:,} parameters")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out


def llm_rendering_sweep(n_test: int = 64, n_pairs: int = N_PAIRS, lam: float = 0.5,
                        renderings=tuple(RENDERINGS), device: str = "cuda",
                        tag: str = "llm_renderings") -> dict:
    """
    Score the LLM arm under every prompt rendering, and report each arm's best.

    Why this exists. The LLM arm's score is not a property of the model alone: on identical data,
    the same Qwen2.5-0.5B scores anywhere from 0.0000 to 0.4987 of oracle on the few-shot arm
    depending only on how the power vector is written into the prompt. Quoting one rendering picks
    an arbitrary point in that range, and the rendering we happened to start with was the worst one
    -- for a reason (integer milliwatts against a 3.74 mW median) that is a defect in our prompt
    rather than a limitation of the model. Reporting the *best* rendering per arm is the version of
    this comparison the baseline cannot complain about, and the spread is itself worth reporting.

    The learned protocol is re-scored inside each rendering's run on the same instances, so the
    contrast is never across different test sets.
    """
    per_rendering = {}
    for r in renderings:
        print(f"\n--- rendering: {r} ---", flush=True)
        per_rendering[r] = llm_experiment(n_test=n_test, n_pairs=n_pairs, lam=lam, device=device,
                                          tag=f"{tag}_{r}", rendering=r)

    arms = [a for a in (row["arm"] for row in next(iter(per_rendering.values())))]
    best = {}
    for arm in arms:
        cand = [(r, next(row for row in rows if row["arm"] == arm))
                for r, rows in per_rendering.items()]
        ratios = {r: row["mean_ratio"] for r, row in cand}
        r_best, row_best = max(cand, key=lambda kv: kv[1]["mean_ratio"])
        best[arm] = {**row_best, "best_rendering": r_best,
                     "ratio_by_rendering": ratios,
                     "ratio_min": min(ratios.values()), "ratio_max": max(ratios.values()),
                     "ratio_spread": max(ratios.values()) - min(ratios.values())}

    out = {"n_test": n_test, "lam": lam, "renderings": list(renderings),
           "per_rendering": per_rendering, "best_per_arm": best}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'='*88}")
    print(f"  {'arm':22s} {'best':>8s}  {'rendering':>10s}  {'min':>8s}  {'max':>8s}  {'spread':>8s}")
    print(f"  {'-'*84}")
    for arm, b in best.items():
        print(f"  {arm:22s} {b['mean_ratio']:8.4f}  {b['best_rendering']:>10s}  "
              f"{b['ratio_min']:8.4f}  {b['ratio_max']:8.4f}  {b['ratio_spread']:8.4f}")
    print(f"{'='*88}")
    return out



# --------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. Serialisation and parsing round-trip, including the ways a model can misbehave.
    g = rng.uniform(1e-11, 1e-9, size=(4, 4))
    txt = describe(g, 0.5)
    print(txt[:280], "\n...")
    assert parse_powers("[10, 20, 30, 40]", 4) is not None
    assert np.allclose(parse_powers("[10, 20, 30, 40]", 4), [0.01, 0.02, 0.03, 0.04])
    assert np.allclose(parse_powers("Sure! [500, -3, 30, 40] mW", 4), [0.1, 0.0, 0.03, 0.04])
    assert parse_powers("I cannot help with that", 4) is None
    assert parse_powers("[1, 2]", 4) is None
    print("parse: clips out-of-range, rejects short and non-numeric answers")

    # 2. The model loads and answers in the required format.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    llm = LLMAllocator(device=dev)
    print(f"\n{llm.model_id}: {llm.parameters/1e6:.0f}M parameters, {llm.size_mb:.0f} MB")
    p, stats = llm.allocate(np.stack([g, g]), 0.5, batch=2)
    print(f"  answered 2 instances in {stats['seconds_per_instance']*1e3:.0f} ms each, "
          f"parse failures {stats['parse_failures']}/2")
    print("  powers (W):", np.round(p, 4).tolist())
    assert p.shape == (2, 4) and (p >= 0).all() and (p <= P_MAX_W + 1e-9).all()
