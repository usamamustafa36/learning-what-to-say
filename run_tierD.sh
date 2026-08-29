#!/usr/bin/env bash
# The evaluation-only queue, after Tier C releases the GPU.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
echo "[$(date -Is)] waiting for Tier C"
until grep -q "tier C done" results/tierC.log 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] === traffic ==="
python3 traffic.py && echo "[$(date -Is)] traffic ok" || echo "[$(date -Is)] traffic FAILED"
cd ../paper && PYTHONPATH=../code python3 make_numbers.py
echo "[$(date -Is)] tier D done"
