#!/bin/bash

# Poker44 Validator Startup Script

NETUID=294  ## 87 if mainnet, 294 if testnet
WALLET_NAME="poker44-test-ck"
HOTKEY="poker44-hk"
NETWORK="test"  ## "finney" for mainnet; "test" for testnet
VALIDATOR_SCRIPT="./neurons/validator.py"
PM2_NAME="poker44_validator"  ##  name of validator, as you wish

if [ ! -f "$VALIDATOR_SCRIPT" ]; then
    echo "Error: Validator script not found at $VALIDATOR_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="/root/Poker44-subnet"

pm2 start $VALIDATOR_SCRIPT \
  --name $PM2_NAME -- \
  --netuid $NETUID \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $HOTKEY \
  --subtensor.network $NETWORK \
  --logging.debug

pm2 save

echo "Validator started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
