#!/usr/bin/env bash
# Finish the scale sweep as soon as the GPU is usable, then rebuild the paper.
#
# The card on this machine intermittently wedges: nvidia-smi shows it idle while torch reports
# "CUDA unknown error" and zero devices. Clearing it needs root:
#
#     sudo systemctl stop ollama; sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
#
# Start this first and it will sit waiting; run that command and the sweep begins on its own.
#
#     nohup ./run_scale_when_gpu.sh > results/scale_auto.log 2>&1 &
#
# N=16 needs 15 runs, N=32 needs 22. finish_scale.py fills only what is missing, so an interrupted
# run costs nothing but the row it was on.

set -u
cd "$(dirname "$0")"
PAPER="$(cd .. && pwd)/paper"
DEADLINE=$(( $(date +%s) + 86400 ))   # give up waiting after 24 h

gpu_ready() {
    python3 - <<'PY' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

echo "[$(date -Is)] waiting for a usable GPU"
until gpu_ready; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[$(date -Is)] gave up after 24 h; GPU never came back"
        exit 1
    fi
    sleep 60
done
echo "[$(date -Is)] GPU is up, starting"

for N in 16 32; do
    echo "[$(date -Is)] === finishing N=$N ==="
    python3 finish_scale.py "$N" || { echo "[$(date -Is)] N=$N failed, stopping"; exit 1; }
    echo "[$(date -Is)] N=$N done"
done

echo "[$(date -Is)] regenerating numbers and rebuilding the paper"
cd "$PAPER" || exit 1
PYTHONPATH=../code python3 make_numbers.py || exit 1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "[$(date -Is)] pages: $(pdfinfo main.pdf | awk '/^Pages/{print $2}')"
echo "[$(date -Is)] NOTE: Table VI gains rows, so the paper will exceed 13 pages."
echo "[$(date -Is)] PARKED.md P4-P6 lists what came out; page 13 was full to ~7300 chars."
echo "[$(date -Is)] all done"
