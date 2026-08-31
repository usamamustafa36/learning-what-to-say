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

## Not verifiable from this machine (16)

IEEE Xplore refuses automated fetches (empty bodies, then HTTP 418) and ScienceDirect returns 403.
These entries are **unchanged and unconfirmed**:

`shi2011wmmse` · `schmidt2009pricing` · `shi2009monotonic` · `gesbert2007adaptation` ·
`love2008limited` · `jindal2006finite` · `lloyd1982` · `max1960` · `sun2018learning` ·
`nasir2019drl` · `gu2023airmpnn` · `shen2021gnn` · `eisen2020regnn` · `boyd2011admm` ·
`dinkelbach1967` · `neely2010lyapunov`

Most are long-established papers whose details are stable and widely reproduced, which is mildly
reassuring but is not verification. **Given that three author lists in this bibliography were wrong,
these sixteen should be checked before submission.** The fastest route is the one that worked for
[2] and [33]: open the Xplore or publisher page and paste it in.

Two specific things to look at while you are there:

- **`gu2023airmpnn`** still has no volume, issue or pages, only a DOI.
- **`strinati2021beyond`** is printed as "E.~C. Strinati". The surname is *Calvanese Strinati*, so
  the correct form may be "E. Calvanese Strinati". ScienceDirect was unreachable, so this is
  unresolved.
