# Citation verification, round 3

Every entry below was checked against a primary source: the arXiv abstract page, or the publisher's
own DOI resolver at `doi.org`. Third-party aggregators were not used as the source of any field that
appears in the bibliography. Where a field could only be found on an aggregator it is recorded here
as **to confirm** and is *not* printed in the paper.

IEEE Xplore article pages could not be read directly from this machine — every fetch returned an
empty body — so Xplore-only fields (volume, issue, page range) are the ones marked to confirm.

| Ref | Status | Source used |
|---|---|---|
| `abdelaziz2024abstraction` | **corrected** — author list was wrong | arXiv abs page + `doi.org` |
| `lee2023decentralised` | **corrected** — replaced preprint with the journal version | arXiv abs page + `doi.org` |
| `yin2025quantisation` | verified, unchanged | arXiv abs page |
| `farooq2026bandwidth` | verified, unchanged | arXiv abs page |

---

## `abdelaziz2024abstraction` (was `samarakoon2023abstraction`, reference [2]) — CORRECTED

The bibliography read **"M. Kim, S. Samarakoon, and M. Bennis"**. That author list is wrong. There is
no author named M. Kim on this paper, and a fourth author was missing entirely.

Verified on <https://arxiv.org/abs/2306.11336>:

- **Title:** Cooperative Multi-Agent Learning for Navigation via Structured State Abstraction ✓
- **Authors:** Mohamed K. Abdel-Aziz, Mohammed S. Elbamby, Sumudu Samarakoon, Mehdi Bennis
- **arXiv:** 2306.11336, v1 20 June 2023, v2 12 February 2024. No journal-ref field.

`https://doi.org/10.1109/TCOMM.2024.3365520` resolves (302) to IEEE Xplore document 10433694,
confirming the venue and the year through the publisher's own resolver.

**Entry now reads:** M. K. Abdel-Aziz, M. S. Elbamby, S. Samarakoon, and M. Bennis, "Cooperative
multi-agent learning for navigation via structured state abstraction," *IEEE Trans. Commun.*, 2024.

**RESOLVED.** The author supplied the IEEE Xplore article page (document 10433694), which gives
*IEEE Trans. Commun.*, vol. 72, no. 6, pp. 3454–3462, June 2024, DOI 10.1109/TCOMM.2024.3365520,
with the author list confirmed as printed above. The entry is now complete. The value a search
result had reported matched, but it is now sourced from the publisher rather than from an
aggregator.

The likely origin of the error is that "Mohamed K." was collapsed to "M. Kim". This entry was
pre-existing, not added in the recent rounds, and had been cited twice since.

## `lee2023decentralised` (was `lee2021decentralised`, reference [33]) — CORRECTED

A published journal version exists, as suspected.

Verified on <https://arxiv.org/abs/2104.09027>:

- **Title:** Decentralized Inference with Graph Neural Networks in Wireless Communication Systems ✓
- **Authors:** Mengyuan Lee, Guanding Yu, Huaiyu Dai ✓
- **arXiv:** 2104.09027, v1 19 April 2021, v2 14 November 2021. No journal-ref field; the comments
  field says "The paper was accpeted by TMC" (sic).

`https://doi.org/10.1109/TMC.2021.3125793` resolves (302) to IEEE Xplore document 9606569, which a
title search independently identifies as this article in IEEE Journals & Magazine.

**Entry now reads:** M. Lee, G. Yu, and H. Dai, "Decentralized inference with graph neural networks
in wireless communication systems," *IEEE Trans. Mobile Comput.*, 2023.

**RESOLVED.** The author supplied the IEEE Xplore article page (document 9606569), which gives
*IEEE Trans. Mobile Comput.*, vol. 22, no. 5, pp. 2582–2598, 1 May 2023, DOI
10.1109/TMC.2021.3125793. The entry is now complete. dblp's figures matched, but the fields in the
paper now come from the publisher.

## `yin2025quantisation` (reference [34]) — VERIFIED, unchanged

Verified on <https://arxiv.org/abs/2503.08125>:

- **Title:** Quantization Design for Deep Learning-Based CSI Feedback ✓
- **Authors:** Manru Yin, Shengqian Han, Chenyang Yang ✓
- **arXiv:** 2503.08125, submitted 11 March 2025 ✓. No journal-ref; correctly cited as a preprint.

## `farooq2026bandwidth` (reference [35]) — VERIFIED, unchanged

Verified on <https://arxiv.org/abs/2602.02035>:

- **Title:** Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector
  Quantization ✓
- **Authors:** Ahmad Farooq, Kamran Iqbal ✓
- **arXiv:** 2602.02035, v1 2 February 2026, v2 11 August 2026 ✓. Preprint only.

---

## Remaining by hand

**Eyeball `yin2025quantisation` and `farooq2026bandwidth` yourself.** Both were verified here against
their arXiv abstract pages, but checking author names and titles by eye rather than trusting a tool's
report is thirty seconds each: <https://arxiv.org/abs/2503.08125> and
<https://arxiv.org/abs/2602.02035>.

## Not re-verified

The remaining 33 entries were not checked in this pass. Given that [2] carried a wrong author list
undetected, **a full sweep of the bibliography against primary sources is worth doing before
submission**; this round only covered the entries the brief named.
