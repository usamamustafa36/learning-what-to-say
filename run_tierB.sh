#!/usr/bin/env bash
# Fill B=5,7 at every completed size, then regenerate B*.
# The card wedges intermittently (four times so far); a wedge mid-run would silently turn every
# remaining size into a "refusing to run on CPU" no-op, so check before each size and after failure.
set -u
cd "$(dirname "$0")"
cuda_ok() { python3 -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }
for N in 8 16 4; do
  if ! cuda_ok; then
    echo "[$(date -Is)] GPU wedged before N=$N -- waiting for it to come back"
    for i in $(seq 120); do sleep 60; cuda_ok && { echo "[$(date -Is)] GPU back"; break; }; done
  fi
  echo "[$(date -Is)] === filling N=$N ==="
  python3 finish_scale.py "$N" && echo "[$(date -Is)] N=$N ok" || echo "[$(date -Is)] N=$N FAILED"
done
echo "[$(date -Is)] regenerating B*"
python3 bstar.py
echo "[$(date -Is)] tier B done"
