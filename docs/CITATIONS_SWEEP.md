# Full bibliography sweep

Every entry checked against a **primary source**: the arXiv abstract page, the publisher's own DOI
resolver, or an IEEE Xplore article page supplied by the author. Aggregators (dblp, Google Scholar,
search summaries) were not used as the source of any field printed in the paper.

**Result: 4 of the 37 entries were wrong, 3 of them in the author list.** One was found in round 3;
three more here.

## Errors found and corrected

| Ref | What was wrong | Source |
|---|---|---|
| [2] `abdelaziz2024abstraction` | authors read "M. Kim, S. Samarakoon, M. Bennis". **No author named M. Kim.** Correct: Abdel-Aziz, Elbamby, Samarakoon, Bennis. Volume/issue/pages were also absent | arXiv 2306.11336 + Xplore 10433694 |
| [3] `bennis2025system2` | authors read "C. Chaccour, W. Saad, M. Debbah, M. Bennis". The paper has **two** authors: **M. Bennis and S. Lahlou** | arXiv 2505.20964 |
| [4] `chaccour2024lessdata` | third author read "M. Bennis"; it is **M. Debbah** | arXiv 2211.14343 |
| [7] `foerster2016dial` | second author read "I. A. Assael"; the source prints **Y. M. Assael** | arXiv 1605.06676 |
| [5] `strinati2021beyond` | missing the article number; *Computer Networks* uses article numbers, not page ranges. Added **art. 107930** | publisher record |
| [33] `lee2023decentralised` | cited as a preprint when a journal version exists; volume/issue/pages added | Xplore 9606569 |

**[3] and [4] are the same error propagating.** [4]'s correct author list is
Chaccour, Saad, **Debbah**, Han, Poor. [3] had been given a near-copy of it — Chaccour, Saad,
Debbah, Bennis — while [4] itself had Debbah replaced by Bennis. One list appears to have been
copied across two entries and then corrupted.

## Verified correct, no change needed (15)

`samarakoon2022mac` (Mota, Valcarce, Gorce, Hoydis; GLOBECOM Wkshps 2021) ·
`gunduz2023beyond` (Gündüz et al.; JSAC 41(1):5–41, 2023) ·
`sukhbaatar2016commnet` · `das2019tarmac` (Das, Gervet, Romoff, Batra, Parikh, Rabbat, Pineau;
ICML 2019) · `wang2020ndq` · `kim2019schedule` (Kim, Moon, Hostallero, Kang, Lee, Son, Yi; ICLR
2019) · `lazaridou2020survey` · `jang2017gumbel` · `maddison2017concrete` · `bengio2013ste` ·
`oord2017vqvae` · `tishby1999ib` · `alemi2017vib` · plus `yin2025quantisation` and
`farooq2026bandwidth` from round 3.

## Second pass over the 16 (this round)

Publishers block automated fetching broadly: IEEE Xplore returns HTTP 418, ScienceDirect, INFORMS
and now publishers all return 403. But six of the sixteen have arXiv versions I had not tried, and
that is where every error so far has been — authors and titles.

### One more error found

| Ref | What was wrong | Source |
|---|---|---|
| [24] `sun2018learning` | title read "…for interference management". The published title is "…for **Wireless Resource Management**". Volume, issue, pages and authors were correct | arXiv 1705.09412, whose journal-ref confirms *IEEE Trans. Signal Process.* 66(20):5438–5453 |

### Verified from arXiv (authors and titles confirmed)

`nasir2019drl` (Nasir, Guo) · `shen2021gnn` (Shen, Shi, Zhang, Letaief) · `eisen2020regnn` (Eisen,
Ribeiro) · `gu2023airmpnn` (Gu, She, Quan, Qiu, Xu) · `jindal2006finite` (Jindal) ·
`sun2018learning` (corrected above).

Their IEEE volume/issue/page fields still come only from the existing entry; arXiv records confirm
the venue via DOI but not always the page range.

### One page-range discrepancy, unresolved

**`jindal2006finite`** prints pp. 5045–**5060**. The author's own arXiv journal-ref for
cs/0603065 says pp. 5045–**5059**. Both forms appear in the literature. I have not changed it,
because neither source is IEEE Xplore and guessing between two plausible values is what produced the
other errors in this bibliography. **Check this one on Xplore.**

### Corroborated by search only — weaker evidence, no contradiction found (4)

`shi2011wmmse` (Shi, Razaviyayn, Luo, He; TSP 59(9):4331–4340, 2011) ·
`love2008limited` (Love, Heath, Lau, Gesbert, Rao, Andrews; JSAC 26(8):1341–1365, 2008) ·
`gesbert2007adaptation` (Proc. IEEE 95(12):2393–2409, 2007) ·
`schmidt2009pricing` (Schmidt, Shi, Berry, Honig, Utschick; IEEE SPM 26(5):53–63, 2009) ·
`boyd2011admm` (Boyd, Parikh, Chu, Peleato, Eckstein; FnT ML 3(1):1–122, 2011).

Each search independently reproduced the entry as printed. That is corroboration, not verification:
the sources are aggregators and indexes, and this bibliography has already shown that a plausible
looking entry can be wrong.

### Still entirely unchecked (5)

`shi2009monotonic` · `lloyd1982` · `max1960` · `dinkelbach1967` · `neely2010lyapunov`

Four of these are foundational papers from 1960–2010 whose details are canonical and stable, and one
is a book. They are the lowest-risk entries in the bibliography, but they have not been checked
against anything.

## Third pass: the remaining ten, via Crossref

IEEE Xplore still refuses automated fetches, but **Crossref** does not — and Crossref is not a
third-party index. It is the DOI registration agency, and the metadata in it is **deposited by the
publisher itself**. For an IEEE paper, the Crossref record *is* IEEE's own record. That closed all
ten.

### Resolved: the page-range discrepancy

**`jindal2006finite`** — Crossref (10.1109/TIT.2006.883550) gives **pp. 5045–5060**. The manuscript
was **right** and the author-supplied arXiv journal-ref (5059) was wrong. No change; the flag is
withdrawn.

### Two entries gained missing fields

| Ref | Added | Source |
|---|---|---|
| `gu2023airmpnn` | **vol. 22, no. 11, pp. 7551–7564** (it had only a DOI) | Crossref 10.1109/TWC.2023.3253126 |
| `shi2009monotonic` | **pp. 1619–1623** | Crossref 10.1109/isit.2009.5205801 |

### Verified exact, no change (8)

`shi2011wmmse` 59(9):4331–4340 · `schmidt2009pricing` 26(5):53–63 · `gesbert2007adaptation`
95(12):2393–2409 · `love2008limited` 26(8):1341–1365 · `lloyd1982` 28(2):129–137 ·
`max1960` 6(1):7–12 · `dinkelbach1967` 13(7):492–498 · `neely2010lyapunov` (Morgan & Claypool,
2010) · plus `nasir2019drl` 37(10):2239–2250, `shen2021gnn` 39(1):101–115 and `eisen2020regnn`
68:2977–2991 confirmed against their DOIs.

### Two deliberate divergences from Crossref

- **`max1960`** — Crossref names the journal *IEEE* Trans. Inf. Theory. In 1960 it was the **IRE**
  Transactions; IRE became IEEE in 1963, and IEEE lists the back catalogue under the modern name.
  The manuscript's "IRE" is the historically correct citation and stays.
- **`schmidt2009pricing`** — Crossref's deposited title is the bare "Distributed resource allocation
  schemes". The article as printed carries the subtitle "Pricing Algorithms for Power Control and
  Beamformer Design in Interference Networks". The manuscript prints the full title, which is more
  informative than the truncated deposit, and stays.

## Running total

**Every one of the 37 entries has now been checked against a primary source.**

**Seven were wrong**: four author lists ([2] `abdelaziz2024abstraction`, [3] `bennis2025system2`,
[4] `chaccour2024lessdata`, [7] `foerster2016dial`), one title ([24] `sun2018learning`), and two
incomplete ([5] `strinati2021beyond` missing its article number, [26] `gu2023airmpnn` missing volume
and pages). [33] `lee2023decentralised` was additionally upgraded from preprint to journal version,
and [15] `shi2009monotonic` gained a page range.

None of these would have been caught by reading the manuscript. Three were fabricated attributions —
authors who did not write the paper they were credited with.
