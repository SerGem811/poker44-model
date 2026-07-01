#!/bin/bash
# Daily safe auto-retrain wrapper for cron. Logs to logs/auto_retrain.log.
# Promotes a new model only if it beats the running one on unseen data and is
# FPR-cliff-safe; otherwise leaves the live miner untouched.
set -euo pipefail

REPO="/root/work/Poker44-subnet"
cd "$REPO"

mkdir -p logs
LOCK="$REPO/logs/auto_retrain.lock"

# prevent overlapping runs
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -u +%FT%TZ)] another auto_retrain run holds the lock; skipping" >> logs/auto_retrain.log
  exit 0
fi

export PYTHONPATH="$REPO"
# the trained miner reads these so a promoted model + manifest stay consistent
export POKER44_MODEL_PATH="${POKER44_MODEL_PATH:-models/poker44_gbdt.joblib}"

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

python3 scripts/miner/train/auto_retrain.py >> logs/auto_retrain.log 2>&1
