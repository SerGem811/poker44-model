from __future__ import annotations

import asyncio
import importlib
import sys
import types

from poker44.validator.synapse import DetectionSynapse


class _BaseMinerNeuron:
    pass


def _load_miner_module():
    stub_module = types.SimpleNamespace(BaseMinerNeuron=_BaseMinerNeuron)
    sys.modules["poker44.base.miner"] = stub_module
    sys.modules.pop("neurons.miner", None)
    return importlib.import_module("neurons.miner")


def test_miner_forward_returns_one_score_and_prediction_per_chunk():
    miner_module = _load_miner_module()
    miner = object.__new__(miner_module.Miner)
    synapse = DetectionSynapse(chunks=[[{"label": "human"}], [{"label": "bot"}]])

    result = asyncio.run(miner.forward(synapse))

    assert result is synapse
    assert len(result.risk_scores) == 2
    assert len(result.predictions) == 2
    assert all(0.0 <= score <= 1.0 for score in result.risk_scores)
    assert all(isinstance(prediction, bool) for prediction in result.predictions)


def test_miner_blacklist_rejects_non_validator_when_permit_is_required():
    miner_module = _load_miner_module()
    miner = object.__new__(miner_module.Miner)
    miner.config = types.SimpleNamespace(
        blacklist=types.SimpleNamespace(
            allow_non_registered=False,
            force_validator_permit=True,
        )
    )
    miner.metagraph = types.SimpleNamespace(
        hotkeys=["validator-hotkey", "miner-hotkey"],
        validator_permit=[True, False],
        S=[10.0, 5.0],
    )
    synapse = DetectionSynapse()
    synapse.dendrite = types.SimpleNamespace(hotkey="miner-hotkey")

    blocked, reason = asyncio.run(miner.blacklist(synapse))

    assert blocked is True
    assert reason == "Non-validator hotkey"


def test_miner_priority_uses_caller_stake():
    miner_module = _load_miner_module()
    miner = object.__new__(miner_module.Miner)
    miner.metagraph = types.SimpleNamespace(
        hotkeys=["validator-hotkey"],
        S=[42.5],
    )
    synapse = DetectionSynapse()
    synapse.dendrite = types.SimpleNamespace(hotkey="validator-hotkey")

    priority = asyncio.run(miner.priority(synapse))

    assert priority == 42.5


def test_miner_scores_bot_like_chunk_higher_than_human_like_chunk():
    miner_module = _load_miner_module()

    human_like_chunk = [
        {
            "players": [{"seat": i} for i in range(1, 7)],
            "streets": [],
            "actions": [
                {"action_type": "small_blind"},
                {"action_type": "big_blind"},
                {"action_type": "raise"},
                {"action_type": "fold"},
                {"action_type": "fold"},
                {"action_type": "fold"},
                {"action_type": "fold"},
                {"action_type": "uncalled_bet_return"},
            ],
            "outcome": {"showdown": False},
        }
    ]
    bot_like_chunk = [
        {
            "players": [{"seat": i} for i in range(1, 6)],
            "streets": [{"street": "flop"}, {"street": "turn"}, {"street": "river"}],
            "actions": [
                {"action_type": "small_blind"},
                {"action_type": "big_blind"},
                {"action_type": "call"},
                {"action_type": "call"},
                {"action_type": "check"},
                {"action_type": "bet"},
                {"action_type": "call"},
                {"action_type": "check"},
                {"action_type": "bet"},
                {"action_type": "call"},
            ],
            "outcome": {"showdown": True},
        }
    ]

    human_score = miner_module.Miner.score_chunk(human_like_chunk)
    bot_score = miner_module.Miner.score_chunk(bot_like_chunk)

    assert 0.0 <= human_score <= 1.0
    assert 0.0 <= bot_score <= 1.0
    assert bot_score > human_score
