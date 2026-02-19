#!/bin/bash

# Poker44 Miner Startup Script

NETUID=294  ## 87 if mainnet, 294 if testnet
WALLET_NAME="poker44-miner-ck"
HOTKEY="poker44-miner-hk"
NETWORK="test"  ## "finney" for mainnet; "test" for testnet
MINER_SCRIPT="./neurons/miner.py"
PM2_NAME="poker44_miner"  ##  name of Miner, as you wish

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="/root/Poker44-subnet"

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME -- \
  --netuid $NETUID \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $HOTKEY \
  --subtensor.network $NETWORK \
  --blacklist.force_validator_permit \
  --blacklist.allow_non_registered false \
    --logging.debug

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
