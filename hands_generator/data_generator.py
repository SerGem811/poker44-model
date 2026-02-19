"""
Data generator that builds arrays of poker hands composed of sub-arrays, where each
sub-array is either entirely human or entirely bot hands.

FINAL validator-oriented behavior:
- Number of chunks is RANDOM in [40, 60].
- Hands per chunk is RANDOM in [60, 100].
- Human ratio is RANDOM per execution in [40%, 60%].
- Each chunk is entirely human or entirely bot.
- Final chunk order is shuffled to avoid positional patterns.

No external caller (validator/miner) can control these parameters.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from hands_generator.bot_hands.generate_poker_data import (
    PokerHandGenerator,
    TableSession,
    BotProfile,
)

# ---------------------------------------------------------------------
# Constants (fixed, not controllable by validator)
# ---------------------------------------------------------------------

HUMAN_HANDS_PATH = Path(__file__).parent / "human_hands" / "human_hands.json"

CHUNK_COUNT_RANGE: Tuple[int, int] = (40, 60)
HANDS_PER_CHUNK_RANGE: Tuple[int, int] = (60, 100)
HUMAN_RATIO_RANGE: Tuple[float, float] = (0.40, 0.60)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_human_hands(path: Path = HUMAN_HANDS_PATH) -> List[Dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def _default_bot_profiles() -> List[BotProfile]:
    return [
        BotProfile(name="balanced", tightness=0.55, aggression=0.55, bluff_freq=0.08),
        BotProfile(name="tight_aggressive", tightness=0.70, aggression=0.75, bluff_freq=0.05),
        BotProfile(name="loose_aggressive", tightness=0.40, aggression=0.80, bluff_freq=0.12),
        BotProfile(name="tight_passive", tightness=0.68, aggression=0.35, bluff_freq=0.03),
        BotProfile(name="loose_passive", tightness=0.42, aggression=0.30, bluff_freq=0.08),
    ]


def sample_human_chunk(
    hands: List[Dict[str, Any]],
    size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not hands or size <= 0:
        return []
    if len(hands) >= size:
        return rng.sample(hands, size)
    return [rng.choice(hands) for _ in range(size)]


def generate_bot_chunk(
    size: int,
    profiles: List[BotProfile],
) -> List[Dict[str, Any]]:
    generator = PokerHandGenerator()
    session = TableSession(table_id="Generated", bot_profiles=profiles)
    session.initialize_table()

    chunk: List[Dict[str, Any]] = []
    while len(chunk) < size:
        hand = generator._generate_single_hand(session)
        if hand:
            chunk.append(hand)
        session.rotate_button()
        session.handle_player_changes()

    return chunk


# ---------------------------------------------------------------------
# Core dataset builders
# ---------------------------------------------------------------------

def build_random_dataset_with_labels(
    human_hands: Optional[List[Dict[str, Any]]] = None,
    bot_profiles: Optional[List[BotProfile]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build a randomized dataset and retain ground-truth labels per chunk.

    Returns:
        [{"hands": [...], "is_bot": bool}, ...]
    """

    rng = random.Random(seed)

    if human_hands is None:
        human_hands = load_human_hands()

    if bot_profiles is None:
        bot_profiles = _default_bot_profiles()

    # --- Randomized parameters (NOT externally controllable) ---
    min_chunks, max_chunks = CHUNK_COUNT_RANGE
    min_hands, max_hands = HANDS_PER_CHUNK_RANGE
    min_ratio, max_ratio = HUMAN_RATIO_RANGE

    num_chunks = rng.randint(min_chunks, max_chunks)
    human_ratio = rng.uniform(min_ratio, max_ratio)

    labeled_chunks: List[Dict[str, Any]] = []
    count = 0

    for _ in range(num_chunks):
        count +=1
        chunk_size = rng.randint(min_hands, max_hands)
        is_human = rng.random() < human_ratio

        if is_human:
            hands = sample_human_chunk(human_hands, chunk_size, rng)
            labeled_chunks.append({"hands": hands, "is_bot": False})
        else:
            hands = generate_bot_chunk(chunk_size, bot_profiles)
            labeled_chunks.append({"hands": hands, "is_bot": True})
            

    # Critical: remove any positional signal
    rng.shuffle(labeled_chunks)

    return labeled_chunks


def build_random_dataset(
    human_hands: Optional[List[Dict[str, Any]]] = None,
    bot_profiles: Optional[List[BotProfile]] = None,
    seed: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    labeled = build_random_dataset_with_labels(
        human_hands=human_hands,
        bot_profiles=bot_profiles,
        seed=seed,
    )
    return [c["hands"] for c in labeled]


# ---------------------------------------------------------------------
# Public API (kept stable for imports)
# ---------------------------------------------------------------------

def generate_dataset_array(
    include_labels: bool = False,
    human_hands: Optional[List[Dict[str, Any]]] = None,
    bot_profiles: Optional[List[BotProfile]] = None,
    seed: Optional[int] = None,
) -> List[Any]:
    """
    Public helper for validator usage.

    IMPORTANT:
    - No external parameter controls chunk counts, sizes, or human ratio.
    - Everything is randomized internally per execution.
    """
    if include_labels:
        return build_random_dataset_with_labels(
            human_hands=human_hands,
            bot_profiles=bot_profiles,
            seed=seed,
        )
    return build_random_dataset(
        human_hands=human_hands,
        bot_profiles=bot_profiles,
        seed=seed,
    )


# ---------------------------------------------------------------------
# CLI (optional, for debugging / inspection)
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate mixed human/bot hand arrays (fully randomized, validator-safe)."
    )
    parser.add_argument("--include-labels", action="store_true", help="Include is_bot labels.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility.")
    parser.add_argument("--output", type=Path, default=Path("mixed_hands.json"))
    args = parser.parse_args()

    human_hands = load_human_hands()

    dataset = generate_dataset_array(
        include_labels=args.include_labels,
        human_hands=human_hands,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(dataset, f, indent=2)

    if args.include_labels:
        chunks = len(dataset)
        hands = sum(len(c["hands"]) for c in dataset)
    else:
        chunks = len(dataset)
        hands = sum(len(c) for c in dataset)

    print(f"✓ Wrote {chunks} chunks ({hands} hands) to {args.output}")


if __name__ == "__main__":
    main()
