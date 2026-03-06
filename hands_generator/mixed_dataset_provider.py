from __future__ import annotations

import hashlib
import gzip
import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import bittensor as bt

from hands_generator.bot_hands.generate_poker_data import BotProfile
from hands_generator.data_generator import _default_bot_profiles, generate_bot_chunk
from poker44.core.hand_json import from_standard_json
from poker44.core.models import LabeledHandBatch
from poker44.validator.seed_manager import SynchronizedSeedManager

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_JSON_PATH = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "validator_mixed_chunks.json"


@dataclass
class MixedDatasetConfig:
    human_json_path: Path = DEFAULT_HUMAN_JSON_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    chunk_count: int = 80
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    refresh_seconds: int = 60 * 60
    seed: Optional[int] = None
    validator_secret_key: Optional[str] = None


def _current_window_id(refresh_seconds: int, now: Optional[float] = None) -> int:
    if refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be > 0")
    ts = time.time() if now is None else now
    return int(ts // refresh_seconds)


def _effective_seed(base_seed: Optional[int], window_id: int) -> int:
    seed_material = f"{0 if base_seed is None else int(base_seed)}:{window_id}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_hand_fingerprint(hand: Dict[str, Any]) -> str:
    payload = json.dumps(hand, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _window_effective_seed(
    cfg: MixedDatasetConfig, window_id: int, *, window_start_iso: Optional[str] = None
) -> int:
    if cfg.validator_secret_key and window_start_iso:
        seed_material = f"{cfg.validator_secret_key}:{window_start_iso}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
    return _effective_seed(cfg.seed, window_id)


def _window_start_iso_for_id(cfg: MixedDatasetConfig, window_id: int) -> str:
    anchor_ts = (window_id * cfg.refresh_seconds) + 1
    return datetime.fromtimestamp(anchor_ts, tz=UTC).isoformat()


def _window_human_sizes(
    cfg: MixedDatasetConfig, window_id: int, *, window_start_iso: Optional[str] = None
) -> List[int]:
    resolved_window_start_iso = window_start_iso
    if cfg.validator_secret_key and resolved_window_start_iso is None:
        resolved_window_start_iso = _window_start_iso_for_id(cfg, window_id)
    effective_seed = _window_effective_seed(
        cfg, window_id, window_start_iso=resolved_window_start_iso
    )
    rng = random.Random(effective_seed)

    n_human = int(round(cfg.chunk_count * cfg.human_ratio))
    n_human = max(1, min(cfg.chunk_count - 1, n_human))
    return _split_chunk_sizes(
        rng, n_human, cfg.min_hands_per_chunk, cfg.max_hands_per_chunk
    )


def _cumulative_human_offset(cfg: MixedDatasetConfig, window_id: int) -> int:
    offset = 0
    for historical_window_id in range(window_id):
        offset += sum(_window_human_sizes(cfg, historical_window_id))
    return offset


def _iter_top_level_array_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield object JSON strings from a top-level JSON array without loading the whole file."""
    if path.suffix != ".gz":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected top-level JSON array in {path}")
        for item in payload:
            yield json.dumps(item, ensure_ascii=False)
        return

    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle as f:
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


def _deterministic_human_selection(
    path: Path,
    sample_size: int,
    cfg: MixedDatasetConfig,
    window_id: int,
) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in {path}")

    valid_hands: List[Dict[str, Any]] = []
    for raw_hand in payload:
        if not isinstance(raw_hand, dict) or not _is_valid_human_hand(raw_hand):
            continue
        hand = dict(raw_hand)
        hand["label"] = "human"
        valid_hands.append(hand)

    if not valid_hands:
        raise RuntimeError("Could not sample any valid human hands from source JSON")

    secret = cfg.validator_secret_key or str(cfg.seed or 0)
    ordered_hands = sorted(
        valid_hands,
        key=lambda hand: hashlib.sha256(
            f"{secret}:{_stable_hand_fingerprint(hand)}".encode("utf-8")
        ).hexdigest(),
    )

    offset = _cumulative_human_offset(cfg, window_id) % len(ordered_hands)
    selected: List[Dict[str, Any]] = []
    for index in range(sample_size):
        selected.append(ordered_hands[(offset + index) % len(ordered_hands)])
    return selected


def _split_chunk_sizes(rng: random.Random, n_chunks: int, min_hands: int, max_hands: int) -> List[int]:
    return [rng.randint(min_hands, max_hands) for _ in range(n_chunks)]


def _compute_dataset_hash(labeled_chunks: List[Dict[str, Any]]) -> str:
    payload = json.dumps(labeled_chunks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_mixed_labeled_chunks(
    cfg: MixedDatasetConfig, *, window_id: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    resolved_window_id = (
        _current_window_id(cfg.refresh_seconds) if window_id is None else int(window_id)
    )
    window_start_iso = None
    window_end_iso = None
    if cfg.validator_secret_key:
        seed_manager = SynchronizedSeedManager(
            secret_key=cfg.validator_secret_key,
            window_minutes=max(1, cfg.refresh_seconds // 60),
        )
        anchor_ts = (resolved_window_id * cfg.refresh_seconds) + 1
        effective_seed, window_start, window_end = seed_manager.generate_seed(
            current_time=datetime.fromtimestamp(anchor_ts, tz=UTC)
        )
        window_start_iso = window_start.isoformat()
        window_end_iso = window_end.isoformat()
    else:
        effective_seed = _effective_seed(cfg.seed, resolved_window_id)
    rng = random.Random(effective_seed)

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
    if cfg.validator_secret_key and cfg.human_json_path.suffix != ".gz":
        human_pool = _deterministic_human_selection(
            cfg.human_json_path, needed_human_hands, cfg, resolved_window_id
        )
    else:
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
        bot_hands = generate_bot_chunk(
            size=size,
            profiles=bot_profiles,
            reference_hands=human_pool,
            seed=rng.randint(0, 10**9),
        )
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
        "window_id": resolved_window_id,
        "effective_seed": effective_seed,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
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
        self._window_id: Optional[int] = None

        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        current_window_id = _current_window_id(self.cfg.refresh_seconds)
        if self.cfg.output_path.exists():
            try:
                data, ds_hash, stats = load_mixed_dataset(self.cfg.output_path)
                stored_window_id = stats.get("window_id")
                if data and stored_window_id == current_window_id:
                    self._data = data
                    self._dataset_hash = ds_hash
                    self._stats = stats
                    self._window_id = int(stored_window_id)
                    self._last_refresh_ts = float(stats.get("generated_at", int(time.time())))
                    bt.logging.info(
                        f"Loaded mixed dataset from disk: {self.cfg.output_path} | chunks={len(self._data)} hash={self._dataset_hash[:12]}"
                    )
                    return
            except Exception as e:
                bt.logging.warning(f"Failed to load mixed dataset from disk, regenerating: {e}")

        self.force_refresh(window_id=current_window_id)

    def force_refresh(self, *, window_id: Optional[int] = None) -> None:
        resolved_window_id = (
            _current_window_id(self.cfg.refresh_seconds) if window_id is None else int(window_id)
        )
        labeled_chunks, ds_hash, stats = build_mixed_labeled_chunks(
            self.cfg, window_id=resolved_window_id
        )
        save_mixed_dataset(self.cfg.output_path, labeled_chunks, ds_hash, stats)
        self._data = labeled_chunks
        self._dataset_hash = ds_hash
        self._stats = stats
        self._window_id = int(stats["window_id"])
        self._last_refresh_ts = time.time()
        bt.logging.info(
            f"Generated mixed dataset | chunks={len(self._data)} hash={self._dataset_hash[:12]} saved={self.cfg.output_path}"
        )

    def refresh_if_due(self) -> None:
        current_window_id = _current_window_id(self.cfg.refresh_seconds)
        if self._window_id == current_window_id:
            return

        bt.logging.info("Mixed dataset refresh window reached. Regenerating candidate dataset...")
        labeled_chunks, new_hash, new_stats = build_mixed_labeled_chunks(
            self.cfg, window_id=current_window_id
        )

        if new_hash != self._dataset_hash:
            bt.logging.info(
                f"New mixed dataset differs from current one ({self._dataset_hash[:12]} -> {new_hash[:12]}). Replacing."
            )
            save_mixed_dataset(self.cfg.output_path, labeled_chunks, new_hash, new_stats)
            self._data = labeled_chunks
            self._dataset_hash = new_hash
            self._stats = new_stats
            self._window_id = int(new_stats["window_id"])
        else:
            bt.logging.info("Regenerated dataset is identical. Keeping current dataset.")

        self._window_id = current_window_id
        self._last_refresh_ts = time.time()

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
