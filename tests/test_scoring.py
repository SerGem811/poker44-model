from __future__ import annotations

import numpy as np

from poker44.score.scoring import reward


def test_reward_is_maximal_for_perfect_predictions():
    preds = np.array([0.0, 0.1, 0.9, 1.0], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=int)

    rew, metrics = reward(preds, labels)

    assert rew == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["bot_recall"] == 1.0
    assert metrics["ap_score"] == 1.0


def test_reward_zeroes_out_when_false_positive_rate_is_too_high():
    preds = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=int)

    rew, metrics = reward(preds, labels)

    assert rew == 0.0
    assert metrics["fpr"] == 1.0
    assert metrics["human_safety_penalty"] == 0.0


def test_reward_handles_no_bot_labels_without_crashing():
    preds = np.array([0.0, 0.2, 0.1], dtype=float)
    labels = np.array([0, 0, 0], dtype=int)

    rew, metrics = reward(preds, labels)

    assert rew == 0.0
    assert metrics["ap_score"] == 0.0
    assert metrics["bot_recall"] == 0.0
