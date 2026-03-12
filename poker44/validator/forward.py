"""Asynchronous forward loop for the Poker44 validator."""
## poker44/validator/forward.py

from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import bittensor as bt
import numpy as np

from poker44.score.scoring import reward
from poker44.validator.synapse import DetectionSynapse

from poker44.validator.constants import (
    BURN_EMISSIONS,
    BURN_FRACTION,
    KEEP_FRACTION,
    UID_ZERO,
    WINNER_TAKE_ALL,
)

_LEAKAGE_KEYS = {
    "label",
    "label_flag",
    "is_bot",
    "bot_family_id",
    "bot_version",
}
_MINER_ACTION_WINDOW = 12
_DEFAULT_MAX_SEATS = 6
_SANITIZED_SB = 0.01
_SANITIZED_BB = 0.02
_SANITIZED_ANTE = 0.0


def _strip_leakage_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _LEAKAGE_KEYS:
                continue
            cleaned[key] = _strip_leakage_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_leakage_fields(item) for item in value]
    return value


def _sanitize_hand_for_miner(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ground-truth and identity leakage before sending chunks to miners."""
    cleaned = _strip_leakage_fields(hand_payload)
    if not isinstance(cleaned, dict):
        return {}

    metadata = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
    players_raw = cleaned.get("players") if isinstance(cleaned.get("players"), list) else []
    actions_raw = cleaned.get("actions") if isinstance(cleaned.get("actions"), list) else []
    outcome = cleaned.get("outcome")

    # Keep a fixed table shape to remove class shortcuts via structural variance.
    max_seats = _DEFAULT_MAX_SEATS

    seat_to_stack: Dict[int, float] = {}
    for player in players_raw:
        if not isinstance(player, dict):
            continue
        seat = player.get("seat")
        try:
            seat_i = int(seat)
        except (TypeError, ValueError):
            continue
        if seat_i <= 0:
            continue
        seat_to_stack[seat_i] = float(player.get("starting_stack", 0.0) or 0.0)

    sanitized_players: List[Dict[str, Any]] = []
    for seat_i in range(1, max_seats + 1):
        sanitized_players.append(
            {
                # Remove persistent identity across hands/classes.
                "player_uid": f"seat_{seat_i}",
                "seat": seat_i,
                "starting_stack": float(seat_to_stack.get(seat_i, 0.0)),
                # Keep schema identical while removing card-reveal leakage.
                "hole_cards": None,
                "showed_hand": False,
            }
        )

    raw_actions: List[Dict[str, Any]] = []
    for action in actions_raw:
        if not isinstance(action, dict):
            continue
        raw_actions.append(
            {
                "action_id": "",
                "street": str(action.get("street", "")),
                # Neutralize absolute seat identifiers to avoid shortcut leakage.
                "actor_seat": 0,
                # Neutralize exact action token to prevent trivial class shortcuts.
                "action_type": "action",
                # Monetary fields are neutralized to remove shortcut leakage.
                "amount": 0.0,
                "raise_to": None,
                "call_to": None,
                "normalized_amount_bb": 0.0,
                "pot_before": 0.0,
                "pot_after": 0.0,
            }
        )

    # Normalize action length to avoid class-separation shortcuts by hand/chunk length.
    sanitized_actions: List[Dict[str, Any]] = []
    if raw_actions:
        if len(raw_actions) >= _MINER_ACTION_WINDOW:
            if _MINER_ACTION_WINDOW == 1:
                sanitized_actions = [raw_actions[0]]
            else:
                last_idx = len(raw_actions) - 1
                indices = [
                    int(round(i * last_idx / (_MINER_ACTION_WINDOW - 1)))
                    for i in range(_MINER_ACTION_WINDOW)
                ]
                sanitized_actions = [raw_actions[i] for i in indices]
        else:
            sanitized_actions = list(raw_actions)
            last = raw_actions[-1]
            while len(sanitized_actions) < _MINER_ACTION_WINDOW:
                pad = dict(last)
                pad["action_id"] = "pad"
                sanitized_actions.append(pad)

    for idx, action in enumerate(sanitized_actions, start=1):
        action["action_id"] = str(idx)

    if not isinstance(outcome, dict):
        outcome = {}

    return {
        "metadata": {
            "game_type": str(metadata.get("game_type", "")),
            "limit_type": str(metadata.get("limit_type", "")),
            "max_seats": max_seats,
            # Normalize seat-level metadata so it cannot be used as an identity hint.
            "hero_seat": 0,
            "hand_ended_on_street": "",
            "button_seat": 0,
            "sb": _SANITIZED_SB,
            "bb": _SANITIZED_BB,
            "ante": _SANITIZED_ANTE,
            "rng_seed_commitment": None,
        },
        "players": sanitized_players,
        # Normalize street list to avoid class separation by board-depth shape alone.
        "streets": [],
        "actions": sanitized_actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": float(outcome.get("total_pot", 0.0) or 0.0),
            "rake": float(outcome.get("rake", 0.0) or 0.0),
            "result_reason": "",
            "showdown": False,
        }
    }


async def forward(validator) -> None:
    """Entry point invoked by :class:`neurons.validator.Validator`."""
    try:
        await _run_forward_cycle(validator)
    except Exception:
        bt.logging.error(f"Unexpected error in forward cycle:\n{traceback.format_exc()}")


async def _run_forward_cycle(validator) -> None:
    validator.forward_count = getattr(validator, "forward_count", 0) + 1
    bt.logging.info(f"[Forward #{validator.forward_count}] start")

    if hasattr(validator.provider, "refresh_if_due"):
        validator.provider.refresh_if_due()

    # Fetch all configured chunks from the stable dataset snapshot.
    chunk_limit = int(getattr(validator, "chunk_batch_size", 80))
    batches = validator.provider.fetch_hand_batch(limit=chunk_limit)
    if not batches:
        bt.logging.info("No hands fetched from dataset; sleeping.")
        await asyncio.sleep(validator.poll_interval)
        return
    
    miner_uids, axons = _get_candidate_miners(validator)
    responses: Dict[int, List[float]] = {uid: [] for uid in miner_uids}

    if not miner_uids:
        bt.logging.info("No eligible miner UIDs available for this cycle.")
        await asyncio.sleep(validator.poll_interval)
        return
    
    # Prepare chunks and labels
    chunks = []  # List of batches (each batch is a list of hand dicts)
    batch_labels = []  # One label per batch
    
    for batch in batches:
        # Convert HandHistory objects to dicts
        chunk_dicts = []
        for hand in batch.hands:
            hand_payload: Dict[str, Any]
            if isinstance(hand, dict):
                hand_payload = hand
            else:
                # Assume hand has a to_payload() or to_dict() method
                try:
                    hand_payload = hand.to_payload()
                except AttributeError:
                    # Fallback: convert dataclass to dict
                    import dataclasses
                    if dataclasses.is_dataclass(hand):
                        hand_payload = dataclasses.asdict(hand)
                    else:
                        hand_payload = hand.__dict__

            chunk_dicts.append(_sanitize_hand_for_miner(hand_payload))
        
        chunks.append(chunk_dicts)
        
        # batch.is_human is False for bots, True for humans
        # We need: 1=bot, 0=human
        batch_label = 0 if batch.is_human else 1
        batch_labels.append(batch_label)
    
    bt.logging.info(f"Processing {len(chunks)} chunks with labels: {batch_labels} (1=bot, 0=human)")
    bt.logging.info(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    
    # Create synapse with all chunks (now as list of dicts)
    synapse = DetectionSynapse(chunks=chunks)
    
    # Get timeout from config
    timeout = 20
    if hasattr(validator.config, "neuron") and hasattr(validator.config.neuron, "timeout"):
        try:
            timeout = float(validator.config.neuron.timeout)
        except (ValueError, TypeError):
            timeout = 20
    
    total_hands = sum(len(chunk) for chunk in chunks)
    bt.logging.info(f"Querying {len(axons)} miners with {len(chunks)} chunks ({total_hands} total hands)...")
    
    synapse_responses = await _dendrite_with_retries(
        validator.dendrite,
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        attempts=3,
    )
    bt.logging.info(f"Received {len(synapse_responses)} responses from miners")
    
    for uid, resp in zip(miner_uids, synapse_responses):
        if resp is None:
            bt.logging.debug(f"Miner {uid} returned None response")
            continue
            
        scores = getattr(resp, "risk_scores", None)
        if scores is None:
            bt.logging.debug(f"Miner {uid} returned no risk_scores")
            continue
            
        try:
            scores_f = [float(s) for s in scores]
            
            # Miners should return one score per chunk
            if len(scores_f) != len(chunks):
                bt.logging.warning(
                    f"Miner {uid} returned {len(scores_f)} scores but expected {len(chunks)} (one per chunk)"
                )
                # Continue anyway, use what we have
                min_len = min(len(scores_f), len(chunks))
                scores_f = scores_f[:min_len]
                effective_labels = batch_labels[:min_len]
            else:
                effective_labels = batch_labels
            
            responses[uid].extend(scores_f)
            
            # Store predictions and labels (one per chunk)
            if not hasattr(validator, "prediction_buffer"):
                validator.prediction_buffer = {}
            if not hasattr(validator, "label_buffer"):
                validator.label_buffer = {}
            
            validator.prediction_buffer.setdefault(uid, []).extend(scores_f)
            validator.label_buffer.setdefault(uid, []).extend(effective_labels)
            
            bt.logging.info(f"Miner {uid} scored {len(scores_f)} chunks successfully")
        except Exception as e:
            bt.logging.warning(f"Error processing response from miner {uid}: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            continue
    
    if not any(responses.values()):
        bt.logging.info("No miner responses this cycle.")
        await asyncio.sleep(validator.poll_interval)
        return
    
    rewards_array, metrics = _compute_windowed_rewards(validator, miner_uids)
    reward_map = dict(zip(miner_uids, rewards_array.tolist()))
    metrics_map = {uid: metric for uid, metric in zip(miner_uids, metrics)}
    bt.logging.info(f"Reward map by UID: {reward_map}")
    bt.logging.info(f"Reward metrics by UID: {metrics_map}")
    winner_uids, winner_rewards = _select_weight_targets(reward_map)

    validator.update_scores(winner_rewards, winner_uids)
    bt.logging.info(f"Rewards issued for {len(winner_rewards)} UID(s).")
    bt.logging.info(
        f"[Forward #{validator.forward_count}] complete. Sleeping {validator.poll_interval}s before next tick.",
    )
    await asyncio.sleep(validator.poll_interval)


def _get_candidate_miners(validator) -> Tuple[List[int], List]:
    miner_uids: List[int] = []
    axons: List = []
    target_uids_env = os.getenv("POKER44_TARGET_MINER_UIDS", "").strip()
    miners_per_cycle_env = os.getenv("POKER44_MINERS_PER_CYCLE", "16").strip()
    miners_per_cycle = 16
    target_uids = None
    if target_uids_env:
        try:
            target_uids = {
                int(uid.strip())
                for uid in target_uids_env.split(",")
                if uid.strip() != ""
            }
            bt.logging.info(f"Restricting miner queries to target UIDs: {sorted(target_uids)}")
        except ValueError:
            bt.logging.warning(
                f"Invalid POKER44_TARGET_MINER_UIDS={target_uids_env!r}; ignoring filter."
            )
            target_uids = None
    try:
        miners_per_cycle = int(miners_per_cycle_env)
    except ValueError:
        bt.logging.warning(
            f"Invalid POKER44_MINERS_PER_CYCLE={miners_per_cycle_env!r}; defaulting to 16."
        )
        miners_per_cycle = 16

    for uid, axon in enumerate(validator.metagraph.axons):
        if uid == UID_ZERO:
            continue
        if target_uids is not None and uid not in target_uids:
            continue
        if bool(validator.metagraph.validator_permit[uid]):
            continue
        ip = str(getattr(axon, "ip", "") or "")
        port = int(getattr(axon, "port", 0) or 0)
        if ip in {"", "0.0.0.0", "::", "[::]"} or port <= 0:
            continue
        miner_uids.append(uid)
        axons.append(axon)

    if target_uids is None and miners_per_cycle > 0 and len(miner_uids) > miners_per_cycle:
        # Rotate deterministically through the eligible set so coverage expands over time
        # without blasting every miner on each cycle.
        offset = ((getattr(validator, "forward_count", 1) - 1) * miners_per_cycle) % len(miner_uids)
        rotated = list(zip(miner_uids, axons))
        rotated = rotated[offset:] + rotated[:offset]
        selected = rotated[:miners_per_cycle]
        miner_uids = [uid for uid, _ in selected]
        axons = [axon for _, axon in selected]
        bt.logging.info(
            f"Sampling {miners_per_cycle} miners this cycle from {len(rotated)} eligible miners "
            f"(rotation offset={offset})."
        )

    bt.logging.info(f"Eligible miners this cycle: {miner_uids}")
    return miner_uids, axons


def _compute_windowed_rewards(validator, miner_uids: List[int]) -> tuple[np.ndarray, list]:
    window = getattr(validator, "reward_window", 20)
    rewards: List[float] = []
    metrics: List[dict] = []

    for uid in miner_uids:
        pred_buf = validator.prediction_buffer.get(uid, [])
        label_buf = validator.label_buffer.get(uid, [])

        if len(pred_buf) < window or len(label_buf) < window:
            rewards.append(0.0)
            metrics.append(
                {
                    "fpr": 1.0,
                    "bot_recall": 0.0,
                    "ap_score": 0.0,
                    "human_safety_penalty": 0.0,
                    "base_score": 0.0,
                    "reward": 0.0,
                }
            )
            continue

        preds_window = np.asarray(pred_buf[-window:], dtype=float)
        labels_window = np.asarray(label_buf[-window:], dtype=bool)
        rew, metric = reward(preds_window, labels_window)
        rewards.append(rew)
        metrics.append(metric)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    
    return rewards_array, metrics


def _select_weight_targets(reward_map: Dict[int, float]) -> tuple[List[int], np.ndarray]:
    if not reward_map:
        bt.logging.info("No eligible rewards computed; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
    winner_uid, winner_reward = sorted_rewards[0]

    if not WINNER_TAKE_ALL:
        positive = [(uid, max(0.0, float(reward))) for uid, reward in sorted_rewards]
        positive = [(uid, reward) for uid, reward in positive if reward > 0.0]

        if not positive:
            bt.logging.info(
                "No miner achieved positive reward; assigning 100%% to UID 0."
            )
            return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

        total_positive = float(sum(reward for _, reward in positive))
        if total_positive <= 0.0:
            bt.logging.info(
                "Positive-reward sum is zero; assigning 100%% to UID 0."
            )
            return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

        norm = [(uid, reward / total_positive) for uid, reward in positive]

        if BURN_EMISSIONS:
            uids = [UID_ZERO] + [uid for uid, _ in norm]
            rewards = [BURN_FRACTION] + [KEEP_FRACTION * frac for _, frac in norm]
            bt.logging.info(
                f"Proportional mode + burn: UID 0 gets {BURN_FRACTION * 100:.2f}%, "
                f"{KEEP_FRACTION * 100:.2f}% split across {len(norm)} miner(s)."
            )
            return uids, np.asarray(rewards, dtype=np.float32)

        uids = [uid for uid, _ in norm]
        rewards = [frac for _, frac in norm]
        bt.logging.info(f"Proportional mode: 100% split across {len(norm)} miner(s).")
        return uids, np.asarray(rewards, dtype=np.float32)

    if winner_reward <= 0.0:
        bt.logging.info("No miner achieved positive reward; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    if BURN_EMISSIONS:
        bt.logging.info(
            f"Winner-take-all burn enabled: UID 0 gets {BURN_FRACTION * 100:.2f}%, "
            f"winner UID {winner_uid} gets {KEEP_FRACTION * 100:.2f}%."
        )
        return [UID_ZERO, winner_uid], np.asarray(
            [BURN_FRACTION, KEEP_FRACTION], dtype=np.float32
        )

    bt.logging.info(f"Winner-take-all enabled: winner UID {winner_uid} gets 100%.")
    return [winner_uid], np.asarray([1.0], dtype=np.float32)

async def _dendrite_with_retries(
    dendrite: bt.dendrite,
    *,
    axons: Sequence,
    synapse: DetectionSynapse,
    timeout: float,
    attempts: int = 3,
):
    """
    Simple retry loop around dendrite calls to avoid transient failures.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await dendrite(
                axons=axons,
                synapse=synapse,
                timeout=timeout,
            )
        except Exception as exc:
            last_exc = exc
            bt.logging.warning(f"dendrite attempt {attempt}/{attempts} failed: {exc}")
            await asyncio.sleep(0.5)
    bt.logging.error(f"dendrite retries exhausted: {last_exc}")
    return [None] * len(axons)
