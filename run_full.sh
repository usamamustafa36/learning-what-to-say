#!/usr/bin/env bash
# FULL-POOL RUNS -- the real pool, not the smoke grid. Each script writes its JSON after every cell,
# so any of these is resumable by re-running it, and each also takes --smoke (64 instances, one
# seed) to prove the code path first.
#
#   ./run_full.sh pricing     Phase 1: budgeted pricing variants, 345 cells, CPU.        ~4 h
#   ./run_full.sh perlambda   Phase 2: per-preference breakdown, matched restarts, CPU.  ~7 min
#   ./run_full.sh learned     Phases 3-4: straight-through arm, entropy ablation,
#                             rounds vs bits. GPU; the B=12 cell trains a 4096-word
#                             codebook and dominates the runtime.                        ~3 h+
#   ./run_full.sh pc          Remark 1: does the operating point admit a trade-off, CPU. ~2 min
#   ./run_full.sh all
#
# Then, in order:
#   cd ../figures && PYTHONPATH=../code python3 make_figures.py
#   cd ../paper   && PYTHONPATH=../code python3 make_numbers.py && pdflatex main.tex && pdflatex main.tex
#   cd ../paper   && python3 audit_literals.py --check
#   cd ../code    && python3 qa.py
#
# make_numbers.py and make_figures.py both refuse to summarise a partial grid: withheld numbers
# reach the page as red [NOT RUN] markers, never as a plausible value.
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-all}" in
  pricing)   python3 pricing_variants.py ;;
  perlambda) python3 per_lambda.py ;;
  learned)   OMP_NUM_THREADS=4 python3 learned_baselines.py ;;
  pc)        python3 pc_sweep.py ;;
  all)       python3 pricing_variants.py && python3 per_lambda.py && python3 pc_sweep.py \
             && OMP_NUM_THREADS=4 python3 learned_baselines.py ;;
  *) echo "usage: $0 {pricing|perlambda|learned|pc|all}"; exit 1 ;;
esac
