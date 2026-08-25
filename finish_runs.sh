#!/bin/bash
# Wait for the main regeneration to finish, then close the remaining evidence gaps and re-QA.
cd "$(dirname "$0")"
while pgrep -f rerun_all.py > /dev/null; do sleep 30; done
echo "=== main sweeps done, running adversarial ==="
python3 experiments.py adversarial 2>&1
echo "=== tasks ==="
python3 experiments.py tasks 2>&1
echo "=== final QA (full) ==="
python3 qa.py --full 2>&1
