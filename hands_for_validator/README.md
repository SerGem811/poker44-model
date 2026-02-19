# hands_for_validator

Esta carpeta contiene utilidades para exponer manos de poker en JSON con FastAPI, cargando el archivo desde local (sin subirlo al repo).

## Archivos

- `rar_to_json.py`: convierte un `.rar` de historiales a un `poker_hands_from_rar.json`.
- `json_api.py`: API FastAPI que lee ese JSON local.

## Uso local

1. Genera el JSON local:

```bash
cd poker_hand_service
PYTHONPATH=. python3 scripts/rar_to_json.py \
  --rar-path ../poker_hands.rar \
  --extract-dir ../extracted_rar \
  --source-dir ../extracted_rar/poker_hands \
  --output-json ../Poker44-subnet/hands_for_validator/poker_hands_from_rar.json
```

2. Levanta la API:

```bash
cd Poker44-subnet
. .venv/bin/activate
uvicorn hands_for_validator.json_api:app --host 127.0.0.1 --port 8000
```

Opcionalmente, define una ruta distinta:

```bash
POKER_JSON_PATH=/ruta/local/poker_hands_from_rar.json uvicorn hands_for_validator.json_api:app --host 127.0.0.1 --port 8000
```
