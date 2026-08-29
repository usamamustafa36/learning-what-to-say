#!/usr/bin/env bash
# Re-run the noisy-channel sweep after the fix to QuantisedCSIEmbedGNN.forward.
# The first pass accepted symbol_fn on the priced arm and ignored it, so those rows recorded a
# perfectly flat line under every BER -- the impairment never reached the model. The learned rows
# were always valid; this redoes both so one file is internally consistent.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
echo "[$(date -Is)] waiting for tier D"
until grep -q "tier D done" results/tierD.log 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] === noisy_channel (re-run, symbol_fn honoured on both arms) ==="
python3 noisy_channel.py && echo "[$(date -Is)] noisy ok" || echo "[$(date -Is)] noisy FAILED"
cd ../paper && PYTHONPATH=../code python3 make_numbers.py
echo "[$(date -Is)] tier E done"
