# hands_for_validator

Utilities to expose poker hands as JSON via FastAPI.

The public repo now ships a compressed human-hand corpus for miner training at:

`hands_generator/human_hands/poker_hands_combined.json.gz`

This folder remains useful for local validator workflows that need to expose a
separate local JSON file over an API instead of using the bundled corpus.

## Files

- `rar_to_json.py`: converts a `.rar` archive of histories into a local JSON file.
- `json_api.py`: FastAPI app that reads a local JSON file.

## Local usage

1. Generate a local JSON file:

```bash
cd poker_hand_service
PYTHONPATH=. python3 scripts/rar_to_json.py \
  --rar-path ../poker_hands.rar \
  --extract-dir ../extracted_rar \
  --source-dir ../extracted_rar/poker_hands \
  --output-json ../Poker44-subnet/hands_for_validator/poker_hands_from_rar.json
```

2. Start the API:

```bash
cd Poker44-subnet
. .venv/bin/activate
uvicorn hands_for_validator.json_api:app --host 127.0.0.1 --port 8000
```

Optionally, point it at a different local file:

```bash
POKER_JSON_PATH=/path/to/local/poker_hands_from_rar.json uvicorn hands_for_validator.json_api:app --host 127.0.0.1 --port 8000
```
