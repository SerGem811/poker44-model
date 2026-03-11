#!/bin/bash

# Poker44 Validator Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-test-ck}"
HOTKEY="${HOTKEY:-poker44-hk}"
NETWORK="${NETWORK:-finney}"
VALIDATOR_SCRIPT="${VALIDATOR_SCRIPT:-./neurons/validator.py}"
PM2_NAME="${PM2_NAME:-poker44_validator}"  ##  name of validator, as you wish
POKER44_HUMAN_JSON_PATH="${POKER44_HUMAN_JSON_PATH:-/path/to/private/poker_data_combined.json}"
POKER44_CHUNK_COUNT="${POKER44_CHUNK_COUNT:-40}"
POKER44_REWARD_WINDOW="${POKER44_REWARD_WINDOW:-40}"
POKER44_POLL_INTERVAL_SECONDS="${POKER44_POLL_INTERVAL_SECONDS:-300}"
NEURON_TIMEOUT="${NEURON_TIMEOUT:-60}"

if [ ! -f "$VALIDATOR_SCRIPT" ]; then
    echo "Error: Validator script not found at $VALIDATOR_SCRIPT"
    exit 1
fi

if [ ! -f "$POKER44_HUMAN_JSON_PATH" ]; then
    echo "Error: Private validator human dataset not found at $POKER44_HUMAN_JSON_PATH"
    echo "Set POKER44_HUMAN_JSON_PATH in scripts/validator/run/run_vali.sh before starting."
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
export POKER44_HUMAN_JSON_PATH="$POKER44_HUMAN_JSON_PATH"
export POKER44_CHUNK_COUNT="$POKER44_CHUNK_COUNT"
export POKER44_REWARD_WINDOW="$POKER44_REWARD_WINDOW"
export POKER44_POLL_INTERVAL_SECONDS="$POKER44_POLL_INTERVAL_SECONDS"

pm2 start $VALIDATOR_SCRIPT \
  --name $PM2_NAME -- \
  --netuid $NETUID \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $HOTKEY \
  --subtensor.network $NETWORK \
  --neuron.timeout $NEURON_TIMEOUT \
  --logging.debug

pm2 save

echo "Validator started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY"
echo "Profile: chunks=$POKER44_CHUNK_COUNT reward_window=$POKER44_REWARD_WINDOW poll_interval_s=$POKER44_POLL_INTERVAL_SECONDS timeout_s=$NEURON_TIMEOUT"
