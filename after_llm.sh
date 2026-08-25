#!/bin/bash
# Wait on a specific PID -- not `pgrep -f <name>`, which matches its own command line and hangs.
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "=== llm done; refreshing the two results now older than the code ==="
python3 experiments.py prior 2>&1
python3 experiments.py temporal 2>&1
echo "=== final QA ==="
python3 qa.py 2>&1 | tail -20
