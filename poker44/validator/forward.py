"""Asynchronous forward loop for the Poker44 validator."""
## poker44/validator/forward.py

from __future__ import annotations

import asyncio
import traceback
import time
from typing import Dict, List, Sequence

import bittensor as bt
import numpy as np

from poker44.score.scoring import reward
from poker44.validator.synapse import DetectionSynapse

from poker44.validator.constants import BURN_EMISSIONS, BURN_FRACTION, KEEP_FRACTION, UID_ZERO


async def forward(validator) -> None:
    """Entry point invoked by :class:`neurons.validator.Validator`."""
    try:
        await _run_forward_cycle(validator)
    except Exception:
        bt.logging.error("Unexpected error in forward cycle:\n%s", traceback.format_exc())


async def _run_forward_cycle(validator) -> None:
    validator.forward_count = getattr(validator, "forward_count", 0) + 1
    bt.logging.info(f"[Forward #{validator.forward_count}] start")
    
    # Fetch multiple batches (e.g., 10 batches at once)
    batches = validator.provider.fetch_hand_batch(limit=10)
    if not batches:
        bt.logging.info("No hands fetched from dataset; sleeping.")
        await asyncio.sleep(validator.poll_interval)
        return
    
    axons = validator.metagraph.axons
    miner_uids = list(range(len(axons)))
    responses: Dict[int, List[float]] = {uid: [] for uid in miner_uids}
    
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
    validator.update_scores(rewards_array, miner_uids)
    bt.logging.info("Rewards issued for %d miners.", len(rewards_array))
    bt.logging.info(
        f"[Forward #{validator.forward_count}] complete. Sleeping {validator.poll_interval}s before next tick.",
    )
    await asyncio.sleep(validator.poll_interval)


def _compute_windowed_rewards(validator, miner_uids: List[int]) -> tuple[np.ndarray, list]:
    window = getattr(validator, "reward_window", 20)
    rewards: List[float] = []
    metrics: List[dict] = []

    for uid in miner_uids:
        pred_buf = validator.prediction_buffer.get(uid, [])
        label_buf = validator.label_buffer.get(uid, [])

        if len(pred_buf) < window or len(label_buf) < window:
            rewards.append(0.0)
            metrics.append({"fp_score": 0, "f1_score": 0, "ap_score": 0, "penalty": 0})
            continue

        preds_window = np.asarray(pred_buf[-window:], dtype=float)
        labels_window = np.asarray(label_buf[-window:], dtype=bool)
        rew, metric = reward(preds_window, labels_window)
        metric["penalty"] = 1.0
        rewards.append(rew)
        metrics.append(metric)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    
    # **95% BURN TO UID 0**: Redistribute weights
    if BURN_EMISSIONS:
        if len(rewards_array) > 0:
            # Normalize rewards to sum to 1
            total_reward = np.sum(rewards_array)
            if total_reward > 0:
                normalized_rewards = rewards_array / total_reward
            else:
                normalized_rewards = np.ones_like(rewards_array) / len(rewards_array)
            
            # Allocate 95% to UID 0, 5% distributed among all miners by their performance
            burned_rewards = normalized_rewards * KEEP_FRACTION  # Scale everyone down to 5%
            burned_rewards[UID_ZERO] = BURN_FRACTION  # Give 95% to UID 0
            
            bt.logging.info(f"95% burn applied: UID 0 gets {burned_rewards[0]:.4f}, others share {1-burned_rewards[0]:.4f}")
            
            return burned_rewards, metrics
    
    return rewards_array, metrics

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
