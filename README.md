# Learning What to Say

**Bit-budgeted emergent signalling for multi-objective resource allocation in 6G**

Usama Mustafa, Imran Rashid — Department of Information Security, Military College of Signals,
National University of Sciences and Technology (NUST), Islamabad, Pakistan

Code, experiment logs and the stored result files behind every number in the accompanying
manuscript (in preparation). **Every figure in the paper is generated from `results/*.json`; nothing
is transcribed by hand.** If an experiment has not been run, its macro renders as a visible
`[NOT RUN]` marker rather than a plausible number.

---

## The question

Emergent inter-agent communication is being pushed toward ever greater capability — semantic,
goal-oriented, agentic, and lately mediated by large language models. This work asks the opposite
question: **what is the least an agent must transmit for decentralised coordination to work at all?**

Multi-objective power allocation in an *N*-user interference channel is cast as a decentralised
decision problem under an explicit per-edge information bottleneck. Each agent observes only its own
direct gain and the interference at its own receiver, exchanges one *B*-bit learned symbol with each
neighbour per slot, and selects power against a preference weight supplied at inference time. The
message channel is a learned vector-quantised codebook trained end-to-end through a
straight-through Gumbel-softmax estimator, so the **protocol** — not merely the policy — is what is
learned.

## What comes out of it

| | |
|---|---|
| Learned protocol, *B*=6 | **0.9482** of the centralised genie optimum |
| No-communication floor (*B*=0) | 0.8513 |
| Unquantised-message ceiling | 0.9562 |
| Coordination gain recovered at 6 bits | **92%** |
| Policy size | 25,361 parameters (99 KB) |

**The control that carries the argument.** Spending the same budget classically — Lloyd–Max
quantised CSI feedback — is *flat* in *B*: 0.8632 at one bit, 0.8613 at eight. The bits are not the
constraint; what the agents choose to say with them is.

**Against centralisation.** An allocator holding the *complete* gain matrix, trained on the same
objective, reaches 0.9057. Two learned bits per edge per slot, exchanged locally, already match it
(0.9061); three exceed it (0.9267).

**What the code encodes.** Mutual-information and decision-tree analyses agree the emergent symbols
carry both quantities a sender can measure — its own link quality and the interference price at its
receiver — and which of the two leads depends on the budget, reversing at one bit. A 64-leaf
distilled rule retains 98.7% of performance, so the protocol is inspectable, not just effective.

**A learned protocol has no specification to defend.** An attacker whose symbol histogram matches
honest traffic *exactly* costs its neighbours 5.7% while remaining statistically undetectable to any
test on the symbol marginal. Conserving symbol usage is not enough to evade distributional
validation — a permutation attack, which preserves the multiset but not the marginal, is detected.

## Layout

```
*.py                 17 self-testing modules; each runs its own checks under `qa.py`
results/*.json       stored results — the source of every number in the paper
results/diagnostics/ investigations that produce no cited number
diagnostics/         the scripts that produce the above
figures/            generated figures
```

`regime.py` is the single source of truth for area, circuit power and `p_max`.

## Reproducing

```bash
pip install torch numpy scipy            # tested on torch 2.6, numpy 2.2, scipy 1.15
python3 qa.py                            # fast validation suite
python3 qa.py --full                     # adds training-dependent checks (~10 min)
python3 experiments.py bitsweep          # the headline sweep (Figure 2)
```

`experiments.py` takes one of: `bitsweep`, `rho`, `sensing`, `abstraction`, `symbolic`, `intent`,
`llm`, `llm_renderings`, `oran`, `temporal`, `prior`, `tasks`, `pareto`, `adversarial`.

`qa.py` checks the self-tests, that every headline claim has evidence on disk, and that stored
results are not older than the code that produced them. Current state: **27 passed, 1 warning
(result-file timestamps), 0 failures; 14/14 headline claims supported.**

A CUDA-capable GPU is used when available; everything runs on CPU, more slowly.

## Known limitations

Stated here rather than left for a reader to find.

- **The LLM arm is sensitive to prompt rendering.** The few-shot arm's headline 0.0000 is *not* a
  capability measurement: the model emitted an all-zero power vector on 64 of 64 instances, and the
  parser accepts all-zero as a successful parse. The cause is our own prompt — exemplar powers were
  rendered as integer milliwatts, flattening 45.8% of exemplar entries to a literal `0`.
  Re-rendering to one decimal place moves the arm to 0.4987. No conclusion in the paper rests on
  that number, and the manuscript says so explicitly. A systematic sweep over renderings
  (`experiments.py llm_renderings`) is the right way to report this arm and has not yet been run to
  completion.
- The setting is a single-hop interference channel with a fixed agent population and a
  differentiable channel; the protocol is learned offline against a stationary environment. Changing
  agent counts, partial observability of the neighbourhood, compositional message structure across
  tasks, and interoperation with an existing standardised layer are all out of scope here.
- Five metrics from an earlier version of this work (`reliability`, `adaptability`, `network_iq`,
  `sustainability`, `user_satisfaction`) were **dropped**: as originally written none varied with
  the allocation. Reasons are recorded in `evaluator.DROPPED`.

## Licence

MIT — see `LICENSE`.
