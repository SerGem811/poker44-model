# 🔐 Poker44 Validator Guide

Welcome to Poker44 – the poker anti-bot subnet with objective, evolving
evaluation. This guide covers the lean validator scaffold introduced in v0.

> **Goal for v0:** fetch labeled hands from Poker44, query miners, score
> them with F1-centric rewards, and log results. On-chain publishing and
> attestations follow in the next milestone.

---

## ✅ Requirements

- Ubuntu 22.04+ (or any Linux with Python 3.10/3.11 available)
- Python 3.10+

---

## 🛠️ Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

___
Validators do not use the public human corpus shipped in the repo. For
evaluation, each validator must have a separate private local human-hand JSON
and point `POKER44_HUMAN_JSON_PATH` at that file. Bot hands are created on the
fly during validator dataset construction. No manual player list is required;
the validator builds labeled human/bot chunks internally.

---

### Register on Subnet 87

```bash
# Register your validator on Poker44 subnet
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 87 \
  --subtensor.network finney

# Check registration status
btcli wallet overview \
   --wallet.name p44_cold \
   --subtensor.network finney
```
---

## ▶️ Run the loop


#### Run validator using pm2
`POKER44_HUMAN_JSON_PATH` is required. Without it, the validator will fail fast
at startup.

```bash
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
pm2 start python --name poker44_validator -- \
  ./neurons/validator.py \
  --netuid 87 \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --subtensor.network finney \
  --logging.debug
```

Example with explicit private human corpus:

```bash
POKER44_HUMAN_JSON_PATH=/path/to/private/poker_data_combined.json \
pm2 start python --name poker44_validator -- \
  ./neurons/validator.py \
  --netuid 87 \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --subtensor.network finney \
  --logging.debug
```

#### Run validator using script
If you want to run it with the help of bash script;
Script for running the validator is at `scripts/validator/run/run_vali.sh`

- Update the hotkey, coldkey, name, network as needed
- Set `POKER44_HUMAN_JSON_PATH` inside the script to your private local human dataset
- Make the script executable: `chmod + x ./scripts/validator/run/run_vali.sh`
- Run the script: `./scripts/validator/run/run_vali.sh`



#### Logs:
```
pm2 logs poker44_validator
```

#### Stop / restart / delete:
```
pm2 stop poker44_validator

pm2 restart poker44_validator

pm2 delete poker44_validator
```


What happens each cycle:

1. Labeled hands (actions, timing, integrity signals) are fetched.
2. A batch is generated consisting of a single hand type & multiple batches are used to create a chunk.
3. Chunks are dispatched to miners; responses are scored with average precision,
   bot recall, and a hard false-positive penalty on humans.
4. Rewards are logged and used to update weights with a winner-take-all policy:
   97% to UID 0 and 3% to the single top-scoring eligible miner. If no miner
   achieves a positive score, 100% goes to UID 0 for that cycle.

The script currently prints results and sleeps for `poll_interval` seconds before repeating.

---

## 🧭 Road to full validator

- ✅ Poker44 ingestion + heuristic scoring loop
- ⏳ Persist receipts + publish weights on-chain
- ⏳ Held-out bot families + early-detection challenges
- ⏳ Dashboarding and operator-facing APIs

Track progress in [docs/roadmap.md](roadmap.md).

---

## 🆘 Help

- Open an issue on GitHub for bugs or missing APIs.
- Reach us on Discord (@sachhp) for any doubts.
