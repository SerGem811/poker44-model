<div align="center">
  <h1>🂡 <strong>Poker44</strong> — Poker Bot Detection Subnet</h1>
  <img src="poker44/assets/logopoker44.png" alt="Poker44 logo" style="width:320px;">
  <p>
    <a href="docs/validator.md">🔐 Validator Guide</a> &bull;
    <a href="docs/miner.md">🛠️ Miner Guide</a> &bull;
    <a href="docs/anti-leakage.md">🛡️ Anti-Leakage</a> &bull;
    <a href="docs/roadmap.md">🗺️ Roadmap</a>
  </p>
</div>

---

## Official Links

- X: https://x.com/poker44subnet
- Web: https://poker44.net
- Whitepaper: https://poker44.net/Poker44_Whitepaper.pdf

---

## What is Poker44?

Poker44 is a Bittensor subnet focused on one problem: detecting bots in online poker with
objective, reproducible evaluation.

Validators query miners with sanitized poker-behavior payloads, score predictions, and publish
weights on-chain. Miners compete by returning robust bot-risk predictions that generalize to
evolving live-table behavior.

Poker44 is security infrastructure, not a poker room.

---

## Current Production Model

The current production direction is:

- a live provider table runs on Poker44 platform infrastructure;
- that table includes both human and bot seats;
- hands are persisted to central platform SQL;
- `poker44-platform-backend` builds sanitized evaluation batches from those hands;
- validators do **not** run their own tables;
- validators fetch the active canonical batch set through the central eval API;
- validators send those batches to miners, compute rewards, and set weights.

The old local `mixed_dataset` validator path still exists in code, but it is no longer the target
production operating model.

---

## What Miners Receive Today

Miners receive `DetectionSynapse(chunks=...)`.

Current semantics:

- `chunks` is a list of chunks;
- each chunk is a list of sanitized hand payloads;
- validators expect one `risk_score` per chunk;
- today, in the live `provider_runtime` path, each chunk is usually one sanitized hand/example.

This means:

- the overall validator request can contain both human-labeled and bot-labeled chunks;
- but miners are not currently receiving a single mixed multi-hand chunk with one global label.

See:

- [Miner Guide](docs/miner.md)
- [Validator Guide](docs/validator.md)

---

## Data Model Boundary

### Production evaluation

Production validators now target:

- live hands from Poker44 platform tables;
- SQL-persisted events and hand results;
- centralized sanitized batch generation through `/internal/eval/*`.

### Public benchmark

The repo still includes a public benchmark/training path for miner development.

That public benchmark is:

- useful for local training and offline testing;
- aligned with the sanitized schema;
- **not** a mirror of the live production evaluation stream.

See:

- [Public benchmark + W&B](docs/public-benchmark.md)

---

## Open-Source Miner Standard

Poker44 supports a lightweight `model_manifest` attached to miner responses.

This does not change validator scoring or on-chain `set_weights`. It adds:

- traceability
- training-data disclosure
- transparency metadata
- anti-leakage observability

Recommended manifest fields include:

- repo URL
- repo commit or tag
- model name and version
- framework
- license
- training-data statement
- private-data attestation

---

## Quick Start

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then follow:

- [Validator setup](docs/validator.md)
- [Miner setup](docs/miner.md)
- [Public benchmark + W&B](docs/public-benchmark.md)

Validated current production-like validator profile:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `POKER44_CHUNK_COUNT=80`
- `POKER44_REWARD_WINDOW=40`
- `POKER44_POLL_INTERVAL_SECONDS=300`
- `--neuron.timeout 60`

---

## Repository Links

- Validator docs: [`docs/validator.md`](docs/validator.md)
- Miner docs: [`docs/miner.md`](docs/miner.md)
- Anti-leakage policy: [`docs/anti-leakage.md`](docs/anti-leakage.md)
- Open-sourced roadmap: [`docs/opensourced_roadmap.md`](docs/opensourced_roadmap.md)
- Roadmap: [`docs/roadmap.md`](docs/roadmap.md)

---

## License

MIT — see [`LICENSE`](LICENSE).
