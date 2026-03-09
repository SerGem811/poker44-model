# 🛠️ Poker44 Miner Guide

This guide covers the production-facing miner flow for Poker44 subnet `126`.

---

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Wallet and Registration

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey --wallet.name my_cold --wallet.hotkey my_poker44_hotkey

btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

---

## Run Miner

Script path: `scripts/miner/run/run_miner.sh`

```bash
chmod +x ./scripts/miner/run/run_miner.sh
./scripts/miner/run/run_miner.sh
```

PM2:

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 stop poker44_miner
pm2 delete poker44_miner
```

---

## Request/Response Contract

Miners receive `DetectionSynapse(chunks=...)`, where:

- `chunks` is a list of chunks.
- each chunk is a list of hands.
- return exactly one `risk_score` per chunk.

Expected output fields:

- `risk_scores: List[float]` in `[0, 1]`
- `predictions: List[bool]` (optional but recommended)

Important: validator payloads are sanitized to remove label/identity leakage before querying miners.

---

## Production Access Policy

Miners should run with validator-only access enabled:

- `blacklist.force_validator_permit=true`

Meaning:

- requests from non-permitted peers are rejected;
- your miner must stay reachable and correctly served on-chain.

---

## Training Data (Miner Side)

Public human corpus:

`hands_generator/human_hands/poker_hands_combined.json.gz`

Bot generation:

```bash
python3 hands_generator/bot_hands/generate_poker_data.py
```

Output:

`hands_generator/bot_hands/bot_hands.json`

Validators evaluate with private human data (`POKER44_HUMAN_JSON_PATH`), not with the public training corpus.

---

## Health Checklist

- Miner hotkey registered on netuid `126`.
- Axon served and visible on-chain.
- Validator queries are accepted.
- Miner returns non-empty `risk_scores` with correct chunk count.
