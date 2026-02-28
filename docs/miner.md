# 🛠️ Poker44 Miner Guide

Poker44 treats miners as bot-hunters: your job is to classify chunks, where each chunk contain multiple batches and each batches are made up of multiple poker hands and then
return a bot classification result per batch. Validators curate labeled hands from a
controlled poker environment & real human hands and reward miners who deliver
accurate, low–false-positive predictions.

This guide covers how to keep your miner hotkey active while you score hands and
how validators translate your responses into on-chain incentives.

---

## 🚀 Quick start

## 🛠️ Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🔐 Wallet prep

Create keys ahead of time so you can register the minute miner incentives go
live:

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey  --wallet.name my_cold --wallet.hotkey my_poker44_hotkey
```

---

### Register on Subnet 87

```bash
# Register your miner on Poker44 subnet
btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 87 \
  --subtensor.network finney

# Check registration status
btcli wallet overview \
   --wallet.name my_cold \
   --subtensor.network finney
```

---

---

## ▶️ Run the loop

#### Run miner using script
You want to run it with the help of bash script;
Script for running the miner is at `scripts/miner/run/run_miner.sh`

- Update the hotkey, coldkey, name, network as needed
- Make the script executable: `chmod + x ./scripts/miner/run/run_miner.sh`
- Run the script: `./scripts/miner/run/run_miner.sh`



#### Logs:
```
pm2 logs poker44_miner
```

#### Stop / restart / delete:
```
pm2 stop poker44_miner

pm2 restart poker44_miner

pm2 delete poker44_miner
```


---

Keep the process running so validators can send canonical hand payloads to your
axon. The reference miner ships with a simple heuristic model; swap in your own
in `neurons/miner.py` for better scores.

### Public training corpus

The public repo includes a compressed human-hand corpus at:

`hands_generator/human_hands/poker_hands_combined.json.gz`

This file is meant to be a starting point for miner training. Poker44 does not
ship a public mixed human+bot training dataset. Instead, miners are expected to:

1. Use the public human corpus as a base.
2. Generate bot hands with the provided generator in `hands_generator/bot_hands/`.
3. Build their own labels, features, balancing strategy, and training pipeline.

This is intentional: the subnet rewards generalization, not memorization of a
single public benchmark.

This public human corpus is for miners. Validators should evaluate on a
separate private local human dataset that is not distributed in the public repo.

To generate a starter bot corpus locally:

```bash
cd Poker44-subnet
python3 hands_generator/bot_hands/generate_poker_data.py
```

This writes `hands_generator/bot_hands/bot_hands.json`, which you can combine
with the public human corpus however you want for training.

### What arrives in each request?

Validators send a `DetectionSynapse` containing:

- **Event log:** ordered actions with amounts, street, stack and pot states.
- **Timing:** decision windows and optional client latency buckets.
- **Context:** table/game metadata (blinds, seat map, format flags).
- **Integrity:** bot provenance tags (for bots), session multi-tabling buckets.

Return a probability in `[0,1]` plus a binary guess; risk scores closer to 1
indicate "bot".

---

## 🧭 How miners earn now

1. **Serve your axon.** Keep your node online so validators can hit it with
   hand-history queries.
2. **Return calibrated bot-risk scores.** Miners are rewarded on average
   precision, bot recall, and low false positives on humans. Only the single
   top-scoring eligible miner receives non-zero miner emissions in each scoring
   window.
3. **Generalise.** Datasets evolve with harder, more human-like bots. Models that
   adapt quickly keep their rewards as difficulty ramps.

---

## 🤝 Contribute ideas

- Share new heuristics/features that help catch bots without harming humans.
- Add adapters for new bot families or integrity signals.
- Stress-test the scoring loop with adversarial patterns.

Keep your node online, push better models, and help keep poker tables fair. 🂡
