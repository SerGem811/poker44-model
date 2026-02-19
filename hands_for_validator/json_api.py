from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query


DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "poker_hands_from_rar.json"
JSON_PATH = Path(os.getenv("POKER_JSON_PATH", str(DEFAULT_JSON_PATH))).expanduser().resolve()
app = FastAPI(title="Poker Hands JSON API", version="1.0.0")


@lru_cache(maxsize=1)
def load_payload() -> dict:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"No existe JSON en {JSON_PATH}. Exporta POKER_JSON_PATH con tu ruta local.")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    payload = load_payload()
    stats = payload.get("stats", {})
    return {
        "ok": True,
        "json_path": str(JSON_PATH),
        "parsed_hands": stats.get("parsed_hands", 0),
    }


@app.get("/hands")
def list_hands(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    player_uid: str | None = None,
):
    payload = load_payload()
    hands = payload.get("hands", [])

    if player_uid:
        hands = [
            hand
            for hand in hands
            if any(p.get("player_uid") == player_uid for p in hand.get("data", {}).get("players", []))
        ]

    total = len(hands)
    return {"total": total, "limit": limit, "offset": offset, "items": hands[offset : offset + limit]}


@app.get("/hands/{external_hand_id}")
def get_hand(external_hand_id: str):
    payload = load_payload()
    for hand in payload.get("hands", []):
        if hand.get("external_hand_id") == external_hand_id:
            return hand
    raise HTTPException(status_code=404, detail="Hand no encontrada")
