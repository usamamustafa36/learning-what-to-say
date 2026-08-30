# Audit, round 2

Answers read off the code and the stored result files. Nothing here is inferred from the manuscript.

---

## 1. Which checkpoint does Section VIII load, and was the entropy bonus on?

**The bonus was off. Section VIII does not characterise the Table II arm.**

`robustness.py:79,104` and `noisy_channel.py:100,108` all build

```python
cfg = Config(bits=bits, mode="vq", steps=8000, seed=seed)
net = train_cached(cfg, tr)
```

`Config.usage_bonus` defaults to `0.0` (`train.py:57`) and none of these call sites overrides it.
The Table II sweep ran at `usage_bonus = 0.2` (confirmed in `bitsweep_v2.json`: every `learned` row
carries `"usage_bonus": 0.2`).

The evidence is stronger than a config comparison. The clean-channel seeds in `noisy_channel.json`
(BER = 0, learned, B = 6) are **0.9287, 0.9373, 0.9397 → 0.9352**, and `csi_error.json` at
σ = 0 dB holds the same three values. Those are bit-identical to the `vq-noent` B = 6 rows in
`learned_baselines.json` (0.92867287, 0.93728653, 0.93973154). `train_cached` is keyed on the config
plus a pool fingerprint, so both experiments were served the *same cached network*.

So every number in Section VIII — BER, erasure, CSI error, ρ, ISAC, the 21-point λ grid — describes
a policy trained without the entropy bonus, scoring 0.9352 clean, while Table II's headline arm
scores 0.9481. The robustness section and the results section are not evaluating the same policy.

## 2. Was the rounds-versus-bits experiment run with the bonus off? Why?

**Yes, and not by design.** `learned_baselines.py:123` builds
`Config(bits=b, mode="vq", rounds=rnd, steps=steps, seed=seed)` — again taking the `0.0` default.
There is no comment or argument suggesting it was deliberate; the script simply never passed `0.2`.

The consequence is exact, not approximate: the `(6,1)` cell of the rounds sweep and the `vq-noent`
B = 6 cell are the same three floats, because they are the same config and `train_cached` returned
one checkpoint for both. That is why the manuscript currently needs a paragraph explaining a
cross-reference instead of comparing directly to Table II.

## 3. Table III: are the stale column and the per-λ columns run the same way?

**No, on both counts, and the stale column is not internally consistent either.**

| | source | pool | WMMSE restarts |
|---|---|---:|---:|
| per-λ columns + mean | `per_lambda.json` | 512 | **16** |
| stale, WMMSE | `standalone_classical.json` | **2,048** | **1** |
| stale, Dinkelbach | `standalone_classical.json` | **1,024** | n/a |
| stale, pricing | `standalone_classical.json` | **1,024** | n/a |

`standalone_classical.py:120` calls `wmmse(a_dec, noise, P_MAX_W)` from a single initialisation.
That is the defect `per_lambda.py` was written to fix, and the stale column still carries it.

The pool split is a provenance accident. `standalone_classical.py:96` sets `size = 2048` and stamps
`"n_instances": len(pool)` on every row, so a single run cannot produce both values.
`standalone_classical_partial.json` holds exactly the first four rows, all at 2,048; the final file's
last two rows are at 1,024. The run was interrupted after `wmmse/stale` and finished from a
1,024-instance pool.

Two published numbers come from those 1,024-instance rows: `\ClsPriceStale` = 0.9725 and
`\ClsDinkStale` = 0.9141. `\ClsPriceStale` is also `\FrontPricingRatio`, the converged-pricing point
in Fig. 2 and the 0.9725 quoted in Section VII-C and the conclusion. Table III's caption currently
says 2,048 for the whole stale column, which is true of one row of three.

**Runtime for the fix.** `per_lambda.py` on 512 instances takes 1,036 s (WMMSE 633 s at 16 restarts,
Dinkelbach 387 s, pricing 16 s). Cost is linear in the pool, so 2,048 instances is **~69 min**
single-threaded. That is affordable, and it is the better direction: it puts the whole table on the
2,048-instance pool the rest of the paper uses, at 16 restarts, from one script.

## 4. Does the imitation arm exist?

**Yes.** `prior_methods.py:98 train_supervised(..., loss="mse")` regresses a full-CSI feedforward
network onto oracle powers — the "regress channel state onto a classical solver's output" template.
`prior_methods.py:174 prior_arms` runs it against `loss="objective"`, the same architecture and the
same full CSI maximising the objective directly, which is the control that separates *imitation*
from *architecture*. Driven by `experiments.py:225 prior_experiment`: train pool 8,192, test pool
2,048, N = 8, test seed 999, **one seed**, 400 epochs.

`results/prior.json`:

| arm | label | mean ratio |
|---|---|---:|
| `supervised-mse` | imitation (prior repo) | 0.8123 |
| `supervised-objective` | direct objective | 0.9057 |

Imitation costs **0.0934** against the identical architecture. `make_numbers.py` already emits
`\PriorImitation`, `\PriorDirectObj` and `\PriorImitationCost`.

**None of the three is cited anywhere in `main.tex`.** Section II-B says "We also implement the first
template, which yields a finding bearing on its own literature… (Section VII-A)" and Section VII-A
contains no such finding. The forward reference is dangling — the result exists on disk and was
never put in the paper.

## 5. The five coding schemes in the 345-configuration sweep

All in `pricing_variants.py:priced_rounds`. Each sends `b` bits per edge per round for `K` rounds
with `K·b·(N−1)` fixed to the budget; they differ only in what those bits mean.

| internal name | what the bits carry |
|---|---|
| `absolute` | memoryless. Levels are sample quantiles of `log(π_s a_sq)` fitted once, before the round loop, from a draw taken at `p = P_max·1`, and never refitted. |
| `differential` | the change since the receiver's last reconstruction, on a fixed ±3 log-unit uniform grid, so the codebook spans a round's movement rather than the full range. |
| `sign` | one sign bit per edge with a Jayant-adapted step: halve on a reversal, ×1.2 on agreement, capped at 4 log units. |
| `adaptive` | uniform levels over a range tracked by an EMA (0.9/0.1) of the observed min and max in the log domain. |
| `dithered` | `absolute` with subtractive dither of one level spacing, added before quantisation and removed after. |

**The winner is `differential`**, and it is the only scheme that ever reaches the learned arm: 9 of
345 cells, all differential, none below a 10-bit price width. It is the "best budgeted price" point
in Fig. 2 (0.9713 at 10,752 bits) and the "budgeted" row of Table III. Ceilings on any budget:
absolute 0.8723, dithered 0.8739, adaptive 0.8189, sign 0.7073.

## 6. B⋆ cells in Fig. 3, and how many are censored

**Fig. 3 shows 9 cells, of which 3 are censored.**

`make_figures.py:fig_bstar_vs_N` plots `styles = [(0.10, …), (0.07, …), (0.05, …)]` against
N ∈ {4, 8, 16}. The censored three are N = 8 at ε = 5%, and N = 16 at ε = 7% and ε = 5%.

`bstar.json` holds **24** rows: 12 at `ref="central"` and 12 at `ref="window"`, each being
ε ∈ {10, 7, 5, 3}% × N ∈ {4, 8, 16}. Among the 12 `central` rows, 6 are censored — which is where
the manuscript's "6 of the 12 cells in Fig. 3" came from. But ε = 3% is not plotted, so neither
number describes the figure. All 12 central cells agree with their isotonic cross-check
(`monotone_agrees` is true throughout), so none is withheld as unresolved.

---

## What this implies for the phases that follow

- Q1 and Q2 are the same root cause: `Config.usage_bonus` defaults to `0.0`, and three scripts
  written after the main sweep never set it. Any rerun must pass `usage_bonus=0.2` explicitly.
- Q3 needs a rerun, not a caption fix: the numbers themselves are from mixed pools.
- Q4 is an addition, not a deletion — the result exists.
- Q6 is a generated-count fix; the figure and the sentence disagree because the sentence counts rows
  in a file rather than markers on a plot.
