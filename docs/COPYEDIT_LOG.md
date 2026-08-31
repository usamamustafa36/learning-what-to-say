# Copy-edit log

Page count **10 before, 10 after**. `main.tex` 77,481 → 77,504 characters (+23). Numeric literals in
the body: **196 before, 196 after, identical as a list** (checked by extracting and diffing them, not
by eye). No new overfull boxes; `verify.sh` passes all gates.

**12 edits.** The brief expected 20–50. Three editorial passes preceded this one, which is the likely
reason the count is low; I did not manufacture edits to reach a target, and §3 records what a generic
copy-editor would have changed that I left alone.

## 1. Every edit

| # | Rule | Before | After |
|---|---|---|---|
| 1 | A — comma splice | `separates them in one experiment, if $B^\star$ stops rising, the growth was th` | `separates them in one experiment: if $B^\star$ stops rising, the growth was th` |
| 2 | A — comma splice | `The parameter count is independent of $N$, the same weights serve any number o` | `The parameter count is independent of $N$: the same weights serve any number o` |
| 3 | B — IEEE Fig./Figure | `\label{sec:scale} Fig.~\ref{fig:bstar} evaluates` | `\label{sec:scale} Figure~\ref{fig:bstar} evaluates` |
| 4 | C — sentence-initial symbol | `\subsection{Preference conditioning} $\lambda$ is drawn afresh per training in` | `\subsection{Preference conditioning} The preference weight $\lambda$ is drawn ` |
| 5 | D — list inconsistency | `$\epsilon=10\%$ for $N=4$, $\RegimeN$, $16$.` | `$\epsilon=10\%$ for $N=4$, $\RegimeN$ and $16$.` |
| 6 | F — And/But merge | `and reuse them every round. But the loop leaves that operating point immediate` | `and reuse them every round, but the loop leaves that operating point immediate` |
| 7 | F — And/But merge | `so there is more to coordinate about. And what the \emph{first} bit recovers o` | `so there is more to coordinate about, and what the \emph{first} bit recovers o` |
| 8 | G — punctuation mechanics | ```how well does a budget do''.` | ```how well does a budget do.''` |
| 9 | G — punctuation mechanics | `frozen in a specification, and implemented against it for a decade` | `frozen in a specification and implemented against it for a decade` |
| 10 | G — punctuation mechanics | `\item \textbf{Generalisation, robustness, and what the budget is spent on.}` | `\item \textbf{Generalisation, robustness and what the budget is spent on.}` |
| 11 | G — punctuation mechanics | `entries, selects one, and transmits its index` | `entries, selects one and transmits its index` |
| 12 | G — punctuation mechanics | `\emph{adaptive} levels over a tracked range, and \emph{dithered} quantisation.` | `\emph{adaptive} levels over a tracked range and \emph{dithered} quantisation.` |

### Notes on three of them

**Edit 5 (D).** The same list appears three times in the paper. The abstract and Section VII-D both
write "for $N=4$, $\RegimeN$ and $16$"; contribution 1 wrote "$N=4$, $\RegimeN$, $16$". Aligned to
the majority form. No value changed.

**Edits 9–12 (G, serial comma).** The brief said "the paper mostly uses the serial comma — make it
uniform." **The premise is inverted:** counting three-item lists in the prose gives **13 without a
serial comma and 0 with** (after stripping macros and math). The four edits therefore remove serial
commas rather than add them, which is the direction that makes the manuscript uniform with itself.

**Edit 8 (G, quotation).** `do''.` → `do.''` — IEEE follows American placement, with the period
inside the closing quotes. This was the only instance in the paper where punctuation met a closing
quote.

## 2. Sentence-initial "And"/"But": 7 found, 2 merged, 5 kept

| Location | Decision | Reason |
|---|---|---|
| V-C, "But the loop leaves that operating point immediately" | **merged** | contrast survives as `, but`; result is 46 words, within the paper's range |
| VII-D, "And what the *first* bit recovers of that window collapses" | **merged** | second of two mechanisms; `, and` keeps the colon-consequence intact |
| I, "And a classical distributed algorithm…" | kept | merging gives a 67-word sentence — rule F forbids creating an overlong one |
| III-A, "And there is no queueing:" | kept-emphasis | a deliberate beat, and named in the brief as one to leave |
| IV-C, "And it *presumes a monotonicity*…" | kept-emphasis | third of a "Three properties bound…" triad |
| VII-F, "And information about $a_{qs}$…" | kept-emphasis | second of "Two readings need care" |
| Conclusion, "And an emergent protocol has no defence…" | kept-emphasis | third of "Three limits bound all of it" |

## 3. Declined edits

Things a generic editor would change, deliberately left.

- **"Two boundaries bound every number here"** (I-B) — "boundaries bound" is an echo, but the
  sentence is correct. *Voice.*
- **"Inverting it answers what a designer asks, what a target costs, and that inversion is the object
  we define"** (IV-C) — the third item breaks parallel with the first two, but reads as an
  independent clause joined by `, and`, which is grammatical. *Correct as written.*
- **"the symbol channel is noiseless, the observation carries no estimation error, and the
  measurement is one slot old"** (VIII) — a serial comma survives here against the convention,
  because the list items are full clauses and removing it impedes parsing. *Correct as written.*
- **"It is \EntOne, \EntTwo, … and \EntEight~bits at $B=1,2,3,4,6,8$"** (V-D) — terse, but a
  singular subject with a distributed list is grammatical. *Voice.*
- **"Signalling cost appears there as an overhead to reduce"** (II-B) — "there" reaches back across
  a paragraph break; loose but unambiguous. *Voice.*
- **Every "per edge" / "per agent per slot" left unhyphenated** — checked all 17 instances: each
  open form is adverbial (postnominal) and each hyphenated form is a prenominal modifier, which is
  the correct rule already applied consistently. Rule E asks for fixes only where the *same
  grammatical position* differs; none does. *Correct as written.*
- **Every `-ize` spelling left alone** — all 30 are inside `\bibitem` entries, i.e. the published
  titles of cited works. Anglicising them would misquote the sources. *Citation accuracy overrides
  the house spelling.*

## 4. Two things outside this pass that you should know about

Both are content, not copy-editing, so I did not touch them.

- **`main.tex` line 247 hard-codes `$[10, 50]$~m`** for the pair-distance range, and line 545
  hard-codes **"16 restarts"**. Both have generated macros that render identically
  (`\PrmPairLo`/`\PrmPairHi` and `\PLRestarts`), so the paper's "no number is typed" rule has two
  exceptions. `audit_literals.py` misses them because its pattern only catches decimals and 3+ digit
  integers. Substituting the macros would change no printed value; it is a one-line fix whenever you
  want it.


---

# Second, independent read

Requested after the first pass. It was worth doing, because **the first pass had a coverage gap I
did not notice at the time**: the prose extractor started at `\section{Introduction}` and dropped any
paragraph beginning with `\caption`, so the **abstract, all nine captions and all three footnotes
were never read**, even though the brief named them explicitly. This read covered exactly those,
then re-ran the mechanical checks over that region and added category-D checks the first pass had
not automated.

Page count **10 before, 10 after**. Numeric literals **196 before, 196 after, identical as a list**.

## Edits found (2, both in the abstract)

| # | Rule | Before | After |
|---|---|---|---|
| 13 | D — dropped comma causing a misparse | `…observes only its own direct gain and its received interference and exchanges one $B$-bit learned symbol…` | `…observes only its own direct gain and its received interference, and exchanges one $B$-bit learned symbol…` |
| 14 | G — missing comma after an introductory phrase | `At $N=\RegimeN$ six learned bits reach` | `At $N=\RegimeN$, six learned bits reach` |

**Edit 13.** Three coordinated `and`s in a row let a reader take "exchanges" as a third thing the
agent *observes*, which inverts the sentence's meaning: the point is that the agent observes two
things and transmits a third. The comma separates the compound predicates. This is not a serial
comma and does not conflict with the manuscript's no-serial-comma convention.

**Edit 14.** The sentence immediately above it — "At $\epsilon=10\%$, $B^\star$ is…" — has the comma
after the same kind of introductory phrase. Without it, `$N=\RegimeN$` and "six" collide as two
adjacent quantities.

## Checked and clean

- **All nine captions.** No comma splices, no sentence-initial abbreviations, no serial commas
  against convention. `tab:budget`'s "Point estimates are means, brackets 95\% percentile-bootstrap
  intervals" is gapping with an elided verb, which conventionally takes a comma — *correct as
  written*, not a splice.
- **All three footnotes**, the title, and the index terms.
- **Category D across the body**, mechanically: doubled words, `each/every` with a plural verb,
  `data/criteria is`, and a/an mismatches. Zero hits.

## Total across both passes

**14 edits.** Still below the brief's 20–50 estimate, and the reason is unchanged: three editorial
passes preceded this work. I did not manufacture edits to reach the range.
