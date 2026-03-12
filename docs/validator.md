# 🔐 Poker44 Validator Guide

Production validator guide for Poker44 subnet `126`.

---

## Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+
- Registered validator hotkey on netuid `126`

---

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use the helper script:

```bash
./scripts/validator/main/setup.sh
```

---

## Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name p44_cold --subtensor.network finney
```

---

## Required Environment

Mandatory:

- `POKER44_HUMAN_JSON_PATH` (private local human dataset JSON)

Optional tuning:

- `POKER44_DATASET_REFRESH_SECONDS` (default `3600`)
- `POKER44_POLL_INTERVAL_SECONDS` (default `300`)
- `POKER44_REWARD_WINDOW` (default `40`)
- `POKER44_CHUNK_COUNT` (default `40`)
- `POKER44_MIN_HANDS_PER_CHUNK` (default `60`)
- `POKER44_MAX_HANDS_PER_CHUNK` (default `120`)
- `POKER44_HUMAN_RATIO` (default `0.5`)
- `POKER44_MINERS_PER_CYCLE` (default `16`; set `0` or a negative value to query all eligible miners)
- `POKER44_TARGET_MINER_UIDS` (comma-separated UIDs, useful for controlled local tests)
- `--neuron.timeout` (default `60s`, validator -> miner query timeout)

---

## Run Validator

### PM2 command

```bash
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
POKER44_CHUNK_COUNT=40 \
POKER44_REWARD_WINDOW=40 \
POKER44_POLL_INTERVAL_SECONDS=300 \
POKER44_MINERS_PER_CYCLE=16 \
pm2 start python --name poker44_validator -- \
  ./neurons/validator.py \
  --netuid 126 \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --subtensor.network finney \
  --neuron.timeout 60 \
  --logging.debug
```

### Script

Script path: `scripts/validator/run/run_vali.sh`

```bash
chmod +x ./scripts/validator/run/run_vali.sh
./scripts/validator/run/run_vali.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `POKER44_HUMAN_JSON_PATH`

The script is environment-driven. Example:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
POKER44_CHUNK_COUNT=40 \
POKER44_REWARD_WINDOW=40 \
POKER44_POLL_INTERVAL_SECONDS=300 \
POKER44_MINERS_PER_CYCLE=16 \
NEURON_TIMEOUT=60 \
./scripts/validator/run/run_vali.sh
```

PM2:

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

---

## Runtime Behavior

Per cycle, validator:

1. Builds mixed labeled chunks from private human data + generated bot data.
2. Sanitizes payloads before sending to miners.
3. Queries miners and scores returned `risk_scores`.
4. Updates internal scores and attempts `set_weights` on-chain.

Default production cadence:

- dataset refresh: every `3600s`
- query loop: every `300s` unless overridden
- miner fanout: `16` miners per cycle by default, rotating across the eligible set

Validated starting profile:

- `POKER44_CHUNK_COUNT=40`
- `POKER44_REWARD_WINDOW=40`
- `--neuron.timeout 60`
- `POKER44_MINERS_PER_CYCLE=16`

These defaults were validated as a practical starting point for production-like runs:

- `80` chunks with the current heuristic miners caused validator query timeouts;
- `40` chunks with `60s` timeout completed successfully;
- setting `POKER44_REWARD_WINDOW=40` allows miners to receive non-zero weights from the first completed cycle.
- querying the full eligible set in one cycle degraded useful miner responses; rotating a subset per cycle was more stable.

---

## Production Checklist

- Validator logs show forward cycles and eligible miner UIDs.
- Miners return non-empty `risk_scores` with expected chunk count.
- Validator logs periodic successful weight submissions:
  - `set_weights on chain successfully!`

---

## Help

- Open a GitHub issue for bugs or missing behavior.
