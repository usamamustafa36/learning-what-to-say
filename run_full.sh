#!/usr/bin/env bash
# FULL-POOL RUNS -- run these on the real pool; each is CPU-only and resumable by re-running.
#
#   ./run_full.sh pricing     Phase 1: budgeted pricing variants (512 instances). ~1-3 h.
#   ./run_full.sh perlambda   Phase 2: per-preference breakdown, matched restarts.
#   ./run_full.sh learned     Phases 3-4: DIAL-style arm, no-entropy ablation, rounds vs bits (GPU).
#   ./run_full.sh all
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-all}" in
  pricing)   python3 pricing_variants.py ;;
  perlambda) python3 per_lambda.py ;;
  learned)   OMP_NUM_THREADS=4 python3 learned_baselines.py ;;
  all)       python3 pricing_variants.py && python3 per_lambda.py \
             && OMP_NUM_THREADS=4 python3 learned_baselines.py ;;
  *) echo "usage: $0 {pricing|perlambda|all}"; exit 1 ;;
esac
