#!/bin/bash
# Everything still blocked on a working CUDA context, in dependency order.
#
# Two rules learned the hard way, on 24 Aug 2026:
#
#   1. A step that fails must stop the run. The previous version had no `set -e`, so when the
#      adversarial sweep faulted the CUDA context mid-sweep, every later step silently fell back
#      to CPU and the run went on to rebuild the paper. A paper rebuilt from a half-finished
#      sweep is worse than no rebuild, because it looks finished.
#   2. CUDA is re-checked before *every* step, not just at the start. The context can wedge in
#      the middle of a run; torch then reports no device and the next step quietly runs on CPU
#      rather than failing. Checking once at the top cannot catch that.
#
# Pass step names to run a subset, e.g. `./when_gpu_back.sh adversarial pareto`.
cd "$(dirname "$0")"
set -uo pipefail

need_cuda() {
  python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo ""
    echo "!! CUDA unavailable before step '$1' -- stopping."
    echo "!! Recover with: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
    exit 1; }
}

run_step() {
  need_cuda "$1"
  echo "+ python3 experiments.py $1"
  python3 experiments.py "$1" || { echo ""; echo "!! step '$1' FAILED -- stopping, nothing downstream ran."; exit 1; }
}

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(llm adversarial prior temporal pareto)
fi

for s in "${STEPS[@]}"; do run_step "$s"; done

need_cuda "qa"
python3 qa.py --full 2>&1 | tail -25
qa_rc=${PIPESTATUS[0]}
[ "$qa_rc" -ne 0 ] && { echo "!! qa.py --full FAILED (rc=$qa_rc) -- paper NOT rebuilt."; exit 1; }

cd ../paper && python3 make_numbers.py || { echo "!! make_numbers.py FAILED -- paper NOT rebuilt."; exit 1; }
if grep -q "NOT RUN" numbers.tex; then
  echo "!! numbers.tex still contains [NOT RUN] markers:"
  grep -n "NOT RUN" numbers.tex
fi
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
echo "paper rebuilt"
