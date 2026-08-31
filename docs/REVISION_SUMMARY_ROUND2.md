# Round 2: consistency pass and page-budget recovery

**Final page count: 10.** 0 overfull boxes, 0 undefined references, 0 unresolved `[NOT RUN]`
macros, `numbers.tex` byte-identical on regeneration, `audit_literals.py --check` clean. Gated by
`code/verify.sh`.

The audit that drove this round is `docs/AUDIT_ROUND2.md`.

---

## 1. Reruns

All were **fully run**, not smoke-only. Each was smoke-tested first on the tiny configuration the
brief specifies; the smoke flags for `robustness.py` and `noisy_channel.py` did not exist before this
round and were added.

| Rerun | Why | Scale | Outcome |
|---|---|---|---|
| `noisy_channel.py` | ran at `usage_bonus=0.0` while Table II used `0.2` | 198 rows, 3 seeds, full pool | clean-channel value **0.9352 → 0.9482** against Table II's 0.9481 |
| `robustness.py` | same | 108 CSI rows + 21-λ grid, 3 seeds | same checkpoint family; Section VIII now describes the Table II policy |
| `learned_baselines.py --arms rounds` | same | 9 cells, 3 seeds | **the finding did not survive** (below) |
| `learned_baselines.py --arms binary` | Table II needs a percentile bootstrap over instances, and the script discarded the per-instance vector | 30 runs, 5 seeds | per-instance ratios kept; binary column now has matched intervals |
| `per_lambda.py --csi current` | Table III mixed pools and restart counts | 2,048 instances, 16 restarts | WMMSE mean 0.7455 |
| `per_lambda.py --csi stale` | same | 2,048 instances, 16 restarts | WMMSE stale **0.6887 → 0.7419** |
| `per_lambda.py --arms pricing` (both CSI) | the bit figure was derived from a differently sized run | 2,048 instances | iterations **100.9 → 130.2** |
| `pc_sweep.py`, `oran.py` | carried over from round 1 | — | unchanged |

### What the reruns changed

**Section VIII was characterising the wrong policy.** `Config.usage_bonus` defaults to `0.0`; the
bit-budget sweep passed `0.2`; three scripts written afterwards took the default. Nothing failed —
`train_cached` served them a perfectly good network trained under a different recipe. The clean-
channel seeds in `noisy_channel.json` were bit-identical to the `vq-noent` arm, which is how it was
caught. `USAGE_BONUS` now lives in `regime.py`, the module that exists because scattered literal
defaults went wrong once before, and every experiment imports it.

**"Rounds beat width" is withdrawn.** Bonus-off, the three matched arrangements spanned 0.0219 with
B=2/R=3 best. Like-for-like they span **0.0047** against a worst-case seed s.d. of 0.0033, the best
is **+0.0043** on the single-round cell, and B=2/R=3 is statistically tied with B=6/R=1. The old gap
was codebook collapse hurting the 6-bit single-round arm most, not rounds being worth more than
width. The single-round cell now reproduces Table II to 0.9482, which is the control the comparison
always needed. The paragraph says the split barely matters and R=1 is the right default.

**Table III is one pool and one restart setting.** It previously mixed three pool sizes (2,048 /
1,024 / 512) and two restart counts, because its stale column came from `standalone_classical.py`,
which runs WMMSE from a single initialisation and whose run was interrupted and finished from a
smaller pool. Consequences: WMMSE stale 0.6887 → 0.7419; pricing stale 0.9725 → 0.9738; pricing
iterations 100.9 → 130.2, so converged pricing costs **29,175 bits, not 22,601**, at **695×** the
learned budget rather than 538×. Fig. 2's converged-pricing marker is rebased on the same
measurement, so figure and table can no longer disagree. The frontier claim strengthens:
centralisation beats it by 0.0067 at **1/14** of the signalling, not 1/11.

## 2. Claims corrected

- **The imitation arm existed and was never cited.** `prior.json` had it all along (0.8123 regressed
  onto oracle powers against 0.9057 for the same network maximising the objective directly) with its
  macros generated, while Section II-B forward-referenced a finding Section VII-A never made. Now
  stated, with its pool and one-seed count.
- **The censoring counts described a file, not a figure.** "6 of the 12 cells in Fig. 3" counted
  every `ref="central"` row in `bstar.json`, including an ε the figure does not plot. Fig. 3 shows
  **9 cells, 3 censored**. The macros now mirror the figure's own ε list.
- **Two "order of magnitude" claims were factors of 3.5 and 2.3.** The staleness ratio is generated
  as `ClsStaleVsGap`; the imitation comparison now quotes both numbers and asserts no multiplier.
- Section III-D no longer says Fig. 1 draws N = 2 — it points at the observability boundary the
  architecture diagram marks. Table II's 0.9562 row is labelled *R = 1* and "ceiling at one round",
  since the multi-round arm exceeds it.
- "Our first attempt coded it badly" is a neutral description of the memoryless scheme. Section VII-C
  names all five schemes; Fig. 2's legend and Table III's caption name the winner.

## 3. Table II restructured

The straight-through binary arm becomes a **column**, with means and percentile-bootstrap intervals
at every budget, built exactly as the other columns are (average each instance over seeds, then
10,000 replicates over instances). The raw-bit-plane column is removed and reduced to one sentence
in VII-A carrying its two numbers (7.2 bits lost at B = 4, 24.2 at B = 8): it is a representation
ablation, not a competitor. Column count is unchanged.

## 4. Related work

Three additions, 89 words, each read on its arXiv listing:
`lee2021decentralised` (decentralised GNN inference over a real channel — analogue transmission, no
budget to invert), `yin2025quantisation` (learned CSI-feedback quantiser — one link to a base
station, adaptive rather than fixed per-edge), `farooq2026bandwidth` (VQ plus information bottleneck
for multi-agent messages — a penalty weight, not a budget swept to a target). Every bibliography
field comes from the listing; none carries a fabricated volume or page range. All three are cited as
preprints and listed in `docs/CITATIONS_TO_VERIFY.md` with what to substitute before submission.

## 5. Textual cuts (Phase 6, applied in order)

| # | Cut | Effect |
|---|---|---|
| 1 | Table II's seed-variance sentence moved into VI-B | net +69 ch (caption type is smaller than body) |
| 2 | Section I-B to three sentences | −470 ch, **11 → 10 pages** |
| 3 | Section V-A sender-only paragraph made one point three times | −? (with 4) |
| 4 | Section VII-D(b) RIC placement sentences dropped, overhead accounting kept | −528 ch with (3), **11 → 10** |
| 5 | Section IV-C coarser-grid paragraph to one clause | −109 ch |
| 6 | Section VI-C to a single tighter paragraph | −222 ch |
| 7 | `seo2026reasoning`, cited only for a dismissal | −341 ch |
| 8 | Table I drops the three derived constants and merges two paired rows | −36 ch, 2 rows |
| — | four bibliography entries abbreviated (`et al.`, `Proc. Allerton`) | −151 ch, **11 → 10** |

Also removed as rhetorical residue: "the whole point", "not cosmetic", "the honest zero-shot
condition", "a list of difficulties is not an agenda", "and we state that once here rather than at
every number", "worth making explicit", "the informative result", "is exactly that".

Section VII-F is retitled *What the messages encode, and two consequences*, with its three parts as
run-in `\paragraph`s. Index terms gain cognitive networking and graph neural networks.
`dump_params.py` in Table I's caption rendered as spaced small caps (`D U M P _ P A R A M S . P Y`)
because IEEEtran sets table captions in small caps; a `\normalfont` group fixes it.

## 6. `\todo{}` remaining

**None.** No `\todo{}` is in the manuscript and no number is typed. A macro whose experiment has not
finished resolves to a red `[NOT RUN]` marker, which `verify.sh` fails on; the current build has
zero.

## 7. Defects found in the tooling

- **`robustness.py --smoke` overwrote the real result files.** Proving the code path destroyed
  `csi_error.json` and `lambda_grid.json`. Both scripts now write `*_smoke.json`.
- `learned_baselines.py` and `per_lambda.py` gained `--arms`, which is how one arm was fixed without
  repeating hours of unaffected work.
- `pricing_variants.py` is resumable.
- The GPU wedged twice, both times after a CUDA process died abnormally (an OOM, then a `kill -9`),
  leaving `Xid 31` faults and a stale UVM context. Cleared with a `nvidia_uvm` reload; a CPU-only
  torch job pins that module unless launched with `CUDA_VISIBLE_DEVICES=`.

## 8. Not done, and why

- **`learned_baselines.py` B=12, R=1** still cannot run: evaluation one-hots every edge of the
  2,048-instance pool against 4,096 codewords, ~2 GiB on an 8 GiB card. Recorded as skipped. The
  manuscript does not cite it.
- **`paper/` and `figures/` remain outside the git repository**, whose root is `code/`. The
  manuscript changes in this round are therefore in no commit; the snapshots are
  `main.tex.phase1.bak`, `main.tex.phase2.bak` and `main.tex.final.bak`. Moving `.git` to the project
  root would rename every tracked path on a repo with a remote, so it is left as the author's call.
- **There is no `references.bib`.** The bibliography is a `thebibliography` environment inside
  `main.tex`, so the requested diff of that file has no subject; bibliography changes are listed in
  §4 and §5 above.
