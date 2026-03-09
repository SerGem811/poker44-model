from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hands_generator.mixed_dataset_provider import (
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
    build_mixed_labeled_chunks,
)
from poker44.validator.seed_manager import SynchronizedSeedManager


HUMAN_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "hands_generator"
    / "human_hands"
    / "poker_hands_combined.json.gz"
)


def _build_cfg(tmp_path: Path) -> MixedDatasetConfig:
    return MixedDatasetConfig(
        human_json_path=HUMAN_DATASET_PATH,
        output_path=tmp_path / "validator_mixed_chunks.json",
        chunk_count=6,
        min_hands_per_chunk=2,
        max_hands_per_chunk=3,
        human_ratio=0.5,
        refresh_seconds=12 * 60 * 60,
        seed=126,
    )


def test_dataset_generation_is_deterministic_within_same_window(tmp_path):
    cfg = _build_cfg(tmp_path)

    chunks_a, hash_a, stats_a = build_mixed_labeled_chunks(cfg, window_id=100)
    chunks_b, hash_b, stats_b = build_mixed_labeled_chunks(cfg, window_id=100)

    assert hash_a == hash_b
    assert stats_a["effective_seed"] == stats_b["effective_seed"]
    assert json.dumps(chunks_a, sort_keys=True) == json.dumps(chunks_b, sort_keys=True)


def test_dataset_generation_changes_across_windows(tmp_path):
    cfg = _build_cfg(tmp_path)

    _, hash_a, stats_a = build_mixed_labeled_chunks(cfg, window_id=100)
    _, hash_b, stats_b = build_mixed_labeled_chunks(cfg, window_id=101)

    assert hash_a != hash_b
    assert stats_a["window_id"] != stats_b["window_id"]
    assert stats_a["effective_seed"] != stats_b["effective_seed"]


def test_provider_returns_labeled_batches(tmp_path):
    cfg = _build_cfg(tmp_path)
    provider = TimedMixedDatasetProvider(cfg)

    batches = provider.fetch_hand_batch(limit=4)

    assert len(batches) == 4
    assert any(batch.is_human for batch in batches)
    assert any(not batch.is_human for batch in batches)
    assert all(batch.hands for batch in batches)
    assert all(batch.hands[0].to_payload()["label"] in {"human", "bot"} for batch in batches)


def test_synchronized_seed_manager_is_stable_within_window():
    manager = SynchronizedSeedManager("secret-126", window_minutes=720)

    seed_a, start_a, end_a = manager.generate_seed(
        current_time=datetime(2026, 3, 4, 10, 15, tzinfo=UTC)
    )
    seed_b, start_b, end_b = manager.generate_seed(
        current_time=datetime(2026, 3, 4, 11, 59, tzinfo=UTC)
    )

    assert seed_a == seed_b
    assert start_a == start_b
    assert end_a == end_b


def test_no_human_reuse_across_consecutive_windows_with_shared_secret(tmp_path):
    hands = []
    for index in range(12):
        hands.append(
            {
                "metadata": {"game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 6, "hero_seat": 1, "hand_ended_on_street": "preflop", "button_seat": 1, "sb": 0.01, "bb": 0.02, "ante": 0.0, "rng_seed_commitment": None},
                "players": [{"player_uid": f"p{index}_{seat}", "seat": seat, "starting_stack": 1.0, "hole_cards": None, "showed_hand": False} for seat in range(1, 3)],
                "streets": [],
                "actions": [{"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "fold", "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0, "pot_before": 0.0, "pot_after": 0.0}],
                "outcome": {"winners": [f"p{index}_1"], "payouts": {f"p{index}_1": 0.0}, "total_pot": 0.0, "rake": 0.0, "result_reason": "fold", "showdown": False},
                "label": "human",
                "source_index": index,
            }
        )

    source_path = tmp_path / "tiny_private_humans.json"
    source_path.write_text(__import__("json").dumps(hands), encoding="utf-8")

    cfg = MixedDatasetConfig(
        human_json_path=source_path,
        output_path=tmp_path / "mixed.json",
        chunk_count=4,
        min_hands_per_chunk=2,
        max_hands_per_chunk=2,
        human_ratio=0.5,
        refresh_seconds=12 * 60 * 60,
        seed=126,
        validator_secret_key="secret-126",
    )

    chunks_a, _, _ = build_mixed_labeled_chunks(cfg, window_id=10)
    chunks_b, _, _ = build_mixed_labeled_chunks(cfg, window_id=11)

    human_a = {
        hand["source_index"]
        for chunk in chunks_a
        if not chunk["is_bot"]
        for hand in chunk["hands"]
    }
    human_b = {
        hand["source_index"]
        for chunk in chunks_b
        if not chunk["is_bot"]
        for hand in chunk["hands"]
    }

    assert human_a
    assert human_b
    assert human_a.isdisjoint(human_b)
UTC = timezone.utc
