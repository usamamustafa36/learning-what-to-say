# Round 3: five pre-submission fixes

**Final page count: 10.** 0 overfull boxes, 0 undefined references, 0 unresolved `[NOT RUN]` macros,
0 `?` in the PDF, every `\cite` key resolves, every bib entry cited, `numbers.tex` byte-identical on
regeneration, literals audit clean. Gated by `code/verify.sh`.

Manuscript diff against `main.tex.round2.bak`: **+18 characters**, 37 references before and after
(two keys corrected, none added or dropped), 8 floats unchanged, section structure unchanged. There
is no `references.bib` — the bibliography is a `thebibliography` environment inside `main.tex`.

---

## Fix 1 — the imitation paragraph (path taken: **rewrite**, not delete)

**What was found.** Both numbers come from `prior_methods.py:train_supervised`, driven by
`experiments.py:prior_experiment`, stored in `results/prior.json`. The 0.9057 arm is **not** a
weaker centralised allocator:

| | 0.9057 arm | 0.9804 arm (Table II) |
|---|---|---|
| class | `SupervisedAllocator` | `CentralisedGNN` |
| architecture | flat MLP over the vectorised gain matrix, `Linear(N²+1→128)→128→64→N`, dropout 0.3 | the message-passing network, byte-for-byte the decentralised arm's graph, embedding, update, readout |
| permutation equivariance | none | yes |
| full CSI reaches it via | flattening | edges carrying both directions, R ≥ 2 rounds |
| training | 400 epochs, batch 256, lr 1e-3, **1 seed** | 8,000 steps, batch 512, **5 seeds**, `usage_bonus=0.2` |

The difference is architecture plus recipe, and it states in one clause, so the paragraph was
rewritten rather than deleted. `agents.py:489` already documented exactly this hazard: scoring
message-passing arms against the flat MLP "measures the difference between two inductive biases, and
that is why it produced the impossible ordering in which a bandwidth-limited decentralised policy
beat a full-CSI centralised one." The manuscript now names the architecture and distinguishes both
numbers from 0.9804 **in the same sentence**.

The Section II-B sentence was also wrong once the framing was corrected — the pair isolates the
*loss*, not "the architecture" — and now reads "Section VII-A reports what imitating a solver costs
against maximising the objective with the same network."

Grep confirms no orphaned reference: `0.9057` and `0.8123` appear nowhere as literals (both are
generated macros), and the two surviving uses of "imitation"/"regress" are the corrected sentences.

## Fix 2 — Table III caption

"all of them differential" replaced with "the best of 345 configurations over the 5 coding schemes of
Section VII-C; the best is differentially coded." The caption is in `main.tex`, not generated.

## Fix 3 — ρ and ISAC provenance: **no rerun needed, and here is why**

The four numbers are generated from `rho_sweep.json` and `sensing.json`. Provenance:

- **Neither file is written by either buggy script.** `robustness.py` writes only `csi_error.json`
  and `lambda_grid.json`; `noisy_channel.py` writes only `noisy_channel.json`. `rho_sweep.json` and
  `sensing.json` come from the `experiments.py` registry and `sensing.py`.
- **Every row of both files records `usage_bonus: 0.2` and `steps: 8000`** — the Table II recipe —
  checked across all 24 rows of `rho_sweep.json` (arm `learned`) and all 36 of `sensing.json` (arms
  `csi`, `sensing`, `shuffled`), not just the first.
- File dates are 24 August, before the 30 August rerun.

So these numbers were never affected by the entropy-bonus defect: they were produced on the correct
recipe from the start, by scripts that did not have the bug. Four-decimal agreement across the rerun
is expected, because the files were never regenerated and never needed to be.

Also in that section: "Symbol erasure at 10% costs 0.9331" now reads "leaves the arm at 0.9331",
since 0.9331 is a level, not a loss.

## Fix 4 — Section V-E promised three classical arms

"We run three" → "We run two"; the raw-bit-plane description is deleted from V-E; "the strongest of
the three" → "the stronger of the two". The surviving sentence in VII-A is self-contained and does
not reference V-E, but its lead-in said "A third format", counting V-E's old three, and now reads "A
third format we tried". Grep for "bit-plane" and "bit planes" finds only that sentence. Recovered
about four lines.

## Fix 5 — citation verification

Full record in `docs/CITATIONS_VERIFIED.md`. Two of the four were wrong.

- **[2] carried a fabricated author list.** It read "M. Kim, S. Samarakoon, and M. Bennis"; the paper
  is by **Mohamed K. Abdel-Aziz, Mohammed S. Elbamby, Sumudu Samarakoon and Mehdi Bennis**. There is
  no author called M. Kim, and a fourth author was missing. Verified on arXiv:2306.11336; venue and
  year confirmed through `doi.org/10.1109/TCOMM.2024.3365520`. This entry was **pre-existing**, not
  added in the recent rounds, and had been cited twice.
- **[33] had a published version**, as suspected. Now cited as *IEEE Trans. Mobile Comput.*, 2023,
  with the DOI confirmed through `doi.org/10.1109/TMC.2021.3125793`.
- **[34] and [35] verified unchanged** against their arXiv abstract pages, character by character.

IEEE Xplore article pages returned empty bodies from this machine, so volume/issue/pages for [2] and
[33] could not be obtained from a primary source. They are **not printed in the paper** and are
listed as "to confirm" in the verification file, with the aggregator values noted but not used.

Because a pre-existing entry carried a wrong author list, **a full sweep of the other 33 references
is worth doing before submission.** This round only covered the four the brief named.

## Small fixes

- **0.0067 is generated, not typed.** `0.9804219 − 0.9737697 = 0.006652 → 0.0067`; subtracting the
  displayed figures gives 0.0066. A comment in `make_numbers.py` now records that the mismatch is
  rounding and that the unrounded difference is the honest one to quote.
- **Fig. 4** panels share one y-range (0.85–0.953), derived from both panels' data in
  `make_figures.py`, so the left–right comparison is direct. Verified by rendering.
- Section III-D: "The point that most often reads as a contradiction is this:" → "Note that".
- Section VII-B: "reproduces Table II to 0.9482" → "matches Table II's 0.9481 to within 0.0001".

## `\todo{}` remaining

**None.** No `\todo{}` is in the manuscript and no number is typed.

## Still open from earlier rounds

- The **B = 12 rounds anchor** is training seed 2 of 3 (2h43m per seed, measured). The paper does not
  cite it, and nothing here depends on it.
- **`paper/` and `figures/` remain outside the git repository**, whose root is `code/`, so this
  round's manuscript changes are in `main.tex.round3.bak` rather than in a commit.
