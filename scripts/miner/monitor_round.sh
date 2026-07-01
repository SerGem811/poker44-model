#!/bin/bash
set -euo pipefail
REPO="/root/work/Poker44-subnet"
cd "$REPO"
export PYTHONPATH="$REPO"
source "$REPO/.venv/bin/activate"
python3 scripts/miner/monitor_round.py >> logs/round_metrics.run.log 2>&1
