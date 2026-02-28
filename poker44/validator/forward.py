"""Asynchronous forward loop for the Poker44 validator."""
## poker44/validator/forward.py

from __future__ import annotations

import asyncio
import traceback
import time
from typing import Dict, List, Sequence, Tuple

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


async def forward(validator) -> None:
    """Entry point invoked by :class:`neurons.validator.Validator`."""
    try:
        await _run_forward_cycle(validator)
    except Exception:
        bt.logging.error("Unexpected error in forward cycle:\n%s", traceback.format_exc())


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
            if isinstance(hand, dict):
                chunk_dicts.append(hand)
            else:
                # Assume hand has a to_payload() or to_dict() method
                try:
                    chunk_dicts.append(hand.to_payload())
                except AttributeError:
                    # Fallback: convert dataclass to dict
                    import dataclasses
                    if dataclasses.is_dataclass(hand):
                        chunk_dicts.append(dataclasses.asdict(hand))
                    else:
                        chunk_dicts.append(hand.__dict__)
        
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
    winner_uids, winner_rewards = _select_weight_targets(reward_map)

    validator.update_scores(winner_rewards, winner_uids)
    bt.logging.info("Rewards issued for %d UID(s).", len(winner_rewards))
    bt.logging.info(
        f"[Forward #{validator.forward_count}] complete. Sleeping {validator.poll_interval}s before next tick.",
    )
    await asyncio.sleep(validator.poll_interval)


def _get_candidate_miners(validator) -> Tuple[List[int], List]:
    miner_uids: List[int] = []
    axons: List = []

    for uid, axon in enumerate(validator.metagraph.axons):
        if uid == UID_ZERO:
            continue
        if bool(validator.metagraph.validator_permit[uid]):
            continue
        miner_uids.append(uid)
        axons.append(axon)

    bt.logging.info("Eligible miners this cycle: %s", miner_uids)
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
        uids = [uid for uid, _ in sorted_rewards]
        rewards = np.asarray([reward for _, reward in sorted_rewards], dtype=np.float32)
        return uids, rewards

    if winner_reward <= 0.0:
        bt.logging.info("No miner achieved positive reward; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    if BURN_EMISSIONS:
        bt.logging.info(
            "Winner-take-all burn enabled: UID 0 gets %.2f%%, winner UID %s gets %.2f%%.",
            BURN_FRACTION * 100,
            winner_uid,
            KEEP_FRACTION * 100,
        )
        return [UID_ZERO, winner_uid], np.asarray(
            [BURN_FRACTION, KEEP_FRACTION], dtype=np.float32
        )

    bt.logging.info("Winner-take-all enabled: winner UID %s gets 100%%.", winner_uid)
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
    bt.logging.error("dendrite retries exhausted: %s", last_exc)
    return [None] * len(axons)
