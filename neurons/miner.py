"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import time
from collections import Counter
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # Check if hotkey is registered
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            if not self.config.blacklist.allow_non_registered:
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

        # Get the caller's UID
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        # Check validator permit if required
        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # Get the caller's UID and stake
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
