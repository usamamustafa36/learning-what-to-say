#!/usr/bin/env bash
# Tier C: everything that is evaluation-only once a checkpoint set exists.
# Waits for Tier B to release the GPU, reloads the driver if the card wedges (it has four times),
# and runs each stage independently so one failure does not sink the rest.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4

cuda_ok() { python3 -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
wait_gpu() {
  for i in $(seq 180); do
    cuda_ok && return 0
    [ "$i" -eq 1 ] && echo "[$(date -Is)] GPU unavailable, waiting"
    sleep 60
  done
  echo "[$(date -Is)] gave up waiting for GPU"; return 1
}

echo "[$(date -Is)] waiting for Tier B to finish"
while pgrep -f "[f]inish_scale.py" >/dev/null || pgrep -f "[r]un_tierB.sh" >/dev/null; do sleep 60; done
echo "[$(date -Is)] Tier B clear"

for stage in generalisation noisy_channel robustness; do
  wait_gpu || break
  echo "[$(date -Is)] === $stage ==="
  python3 "$stage.py" && echo "[$(date -Is)] $stage ok" || echo "[$(date -Is)] $stage FAILED"
done

echo "[$(date -Is)] regenerating B* and paper macros"
python3 bstar.py > /dev/null 2>&1
cd ../paper && PYTHONPATH=../code python3 make_numbers.py
echo "[$(date -Is)] tier C done"
