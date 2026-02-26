from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import bittensor as bt

from hands_generator.bot_hands.generate_poker_data import BotProfile
from hands_generator.data_generator import _default_bot_profiles, generate_bot_chunk
from poker44.core.hand_json import from_standard_json
from poker44.core.models import LabeledHandBatch

DEFAULT_HUMAN_JSON_PATH = Path(__file__).resolve().parents[2] / "poker_data_combined.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "validator_mixed_chunks.json"


@dataclass
class MixedDatasetConfig:
    human_json_path: Path = DEFAULT_HUMAN_JSON_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    chunk_count: int = 80
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    refresh_seconds: int = 12 * 60 * 60
    seed: Optional[int] = None


def _iter_top_level_array_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield object JSON strings from a top-level JSON array without loading the whole file."""
    with path.open("r", encoding="utf-8") as f:
        in_string = False
        escape = False
        depth = 0
        collecting = False
        buf: List[str] = []
        seen_array_start = False

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for ch in chunk:
                if collecting:
                    buf.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "[":
                    depth += 1
                    if depth == 1:
                        seen_array_start = True
                    continue

                if ch == "{":
                    if seen_array_start and depth == 1 and not collecting:
                        collecting = True
                        buf = ["{"]
                    depth += 1
                    continue

                if ch == "}":
                    depth -= 1
                    if collecting and depth == 1:
                        yield "".join(buf)
                        collecting = False
                        buf = []
                    continue

                if ch == "]":
                    depth -= 1


def _is_valid_human_hand(hand: Dict[str, Any]) -> bool:
    players = hand.get("players")
    actions = hand.get("actions")
    if not isinstance(players, list) or len(players) < 2:
        return False
    if not isinstance(actions, list) or len(actions) == 0:
        return False
    return True


def _reservoir_sample_humans(path: Path, sample_size: int, rng: random.Random) -> List[Dict[str, Any]]:
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for raw in _iter_top_level_array_objects(path):
        try:
            hand = json.loads(raw)
        except Exception:
            continue

        if not _is_valid_human_hand(hand):
            continue

        # Force canonical label for the validator pipeline.
        hand["label"] = "human"

        if len(reservoir) < sample_size:
            reservoir.append(hand)
        else:
            j = rng.randint(0, seen)
            if j < sample_size:
                reservoir[j] = hand
        seen += 1

    return reservoir


def _split_chunk_sizes(rng: random.Random, n_chunks: int, min_hands: int, max_hands: int) -> List[int]:
    return [rng.randint(min_hands, max_hands) for _ in range(n_chunks)]


def _compute_dataset_hash(labeled_chunks: List[Dict[str, Any]]) -> str:
    payload = json.dumps(labeled_chunks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_mixed_labeled_chunks(cfg: MixedDatasetConfig) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    rng = random.Random(cfg.seed)

    if cfg.chunk_count <= 0:
        raise ValueError("chunk_count must be > 0")

    if cfg.min_hands_per_chunk <= 0 or cfg.max_hands_per_chunk < cfg.min_hands_per_chunk:
        raise ValueError("Invalid hands per chunk range")

    if not cfg.human_json_path.exists():
        raise FileNotFoundError(f"Missing human JSON source: {cfg.human_json_path}")

    n_human = int(round(cfg.chunk_count * cfg.human_ratio))
    n_human = max(1, min(cfg.chunk_count - 1, n_human))
    n_bot = cfg.chunk_count - n_human

    human_sizes = _split_chunk_sizes(rng, n_human, cfg.min_hands_per_chunk, cfg.max_hands_per_chunk)
    bot_sizes = _split_chunk_sizes(rng, n_bot, cfg.min_hands_per_chunk, cfg.max_hands_per_chunk)

    needed_human_hands = sum(human_sizes)
    human_pool = _reservoir_sample_humans(cfg.human_json_path, needed_human_hands, rng)
    if not human_pool:
        raise RuntimeError("Could not sample any valid human hands from source JSON")

    if len(human_pool) < needed_human_hands:
        human_pool.extend(rng.choices(human_pool, k=needed_human_hands - len(human_pool)))

    human_chunks: List[Dict[str, Any]] = []
    cursor = 0
    for size in human_sizes:
        human_chunks.append({"hands": human_pool[cursor : cursor + size], "is_bot": False})
        cursor += size

    bot_profiles: List[BotProfile] = _default_bot_profiles()
    bot_chunks: List[Dict[str, Any]] = []
    for size in bot_sizes:
        bot_hands = generate_bot_chunk(size=size, profiles=bot_profiles)
        for hand in bot_hands:
            hand["label"] = "bot"
        bot_chunks.append({"hands": bot_hands, "is_bot": True})

    labeled_chunks = human_chunks + bot_chunks
    rng.shuffle(labeled_chunks)

    dataset_hash = _compute_dataset_hash(labeled_chunks)

    stats = {
        "chunk_count": len(labeled_chunks),
        "human_chunks": n_human,
        "bot_chunks": n_bot,
        "total_hands": sum(len(c["hands"]) for c in labeled_chunks),
        "human_hands": sum(len(c["hands"]) for c in labeled_chunks if not c["is_bot"]),
        "bot_hands": sum(len(c["hands"]) for c in labeled_chunks if c["is_bot"]),
        "dataset_hash": dataset_hash,
        "generated_at": int(time.time()),
    }
    return labeled_chunks, dataset_hash, stats


def save_mixed_dataset(output_path: Path, labeled_chunks: List[Dict[str, Any]], dataset_hash: str, stats: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "stats": stats,
        "dataset_hash": dataset_hash,
        "labeled_chunks": labeled_chunks,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def load_mixed_dataset(path: Path) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        labeled_chunks = payload
        dataset_hash = _compute_dataset_hash(labeled_chunks)
        stats = {
            "chunk_count": len(labeled_chunks),
            "dataset_hash": dataset_hash,
            "generated_at": int(path.stat().st_mtime),
        }
        return labeled_chunks, dataset_hash, stats

    labeled_chunks = payload.get("labeled_chunks", [])
    dataset_hash = payload.get("dataset_hash") or _compute_dataset_hash(labeled_chunks)
    stats = payload.get("stats") or {
        "chunk_count": len(labeled_chunks),
        "dataset_hash": dataset_hash,
        "generated_at": int(path.stat().st_mtime),
    }
    return labeled_chunks, dataset_hash, stats


class TimedMixedDatasetProvider:
    """Serves a stable mixed human/bot dataset and refreshes it every N seconds."""

    def __init__(self, cfg: MixedDatasetConfig):
        self.cfg = cfg
        self._data: List[Dict[str, Any]] = []
        self._dataset_hash: str = ""
        self._stats: Dict[str, Any] = {}
        self._last_refresh_ts: float = 0.0

        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        if self.cfg.output_path.exists():
            try:
                data, ds_hash, stats = load_mixed_dataset(self.cfg.output_path)
                if data:
                    self._data = data
                    self._dataset_hash = ds_hash
                    self._stats = stats
                    self._last_refresh_ts = float(stats.get("generated_at", int(time.time())))
                    bt.logging.info(
                        f"Loaded mixed dataset from disk: {self.cfg.output_path} | chunks={len(self._data)} hash={self._dataset_hash[:12]}"
                    )
                    return
            except Exception as e:
                bt.logging.warning(f"Failed to load mixed dataset from disk, regenerating: {e}")

        self.force_refresh()

    def force_refresh(self) -> None:
        labeled_chunks, ds_hash, stats = build_mixed_labeled_chunks(self.cfg)
        save_mixed_dataset(self.cfg.output_path, labeled_chunks, ds_hash, stats)
        self._data = labeled_chunks
        self._dataset_hash = ds_hash
        self._stats = stats
        self._last_refresh_ts = time.time()
        bt.logging.info(
            f"Generated mixed dataset | chunks={len(self._data)} hash={self._dataset_hash[:12]} saved={self.cfg.output_path}"
        )

    def refresh_if_due(self) -> None:
        now = time.time()
        if (now - self._last_refresh_ts) < self.cfg.refresh_seconds:
            return

        bt.logging.info("Mixed dataset refresh window reached. Regenerating candidate dataset...")
        labeled_chunks, new_hash, new_stats = build_mixed_labeled_chunks(self.cfg)

        if new_hash != self._dataset_hash:
            bt.logging.info(
                f"New mixed dataset differs from current one ({self._dataset_hash[:12]} -> {new_hash[:12]}). Replacing."
            )
            save_mixed_dataset(self.cfg.output_path, labeled_chunks, new_hash, new_stats)
            self._data = labeled_chunks
            self._dataset_hash = new_hash
            self._stats = new_stats
        else:
            bt.logging.info("Regenerated dataset is identical. Keeping current dataset.")

        self._last_refresh_ts = now

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def fetch_hand_batch(
        self,
        *,
        limit: int = 80,
        include_integrity: bool = True,
    ) -> List[LabeledHandBatch]:
        if not self._data:
            return []

        selected = self._data[: max(0, limit)]
        batches: List[LabeledHandBatch] = []
        for entry in selected:
            hands_raw = entry.get("hands", [])
            is_bot = bool(entry.get("is_bot", False))
            hands = [from_standard_json(hand) for hand in hands_raw]
            batches.append(LabeledHandBatch(hands=hands, is_human=not is_bot))
        return batches
