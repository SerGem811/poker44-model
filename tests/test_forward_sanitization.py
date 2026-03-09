from poker44.validator.forward import _sanitize_hand_for_miner


def test_sanitize_hand_removes_label_and_bot_markers():
    hand = {
        "label": "bot",
        "metadata": {"game_type": "Hold'em"},
        "players": [
            {
                "player_uid": "p_abc",
                "seat": 1,
                "starting_stack": 1.0,
                "is_bot": True,
            }
        ],
        "outcome": {
            "winners": ["u_secret", "p_abc"],
            "payouts": {"u_secret": 1.23, "p_abc": 0.5},
            "total_pot": 1.8,
            "rake": 0.1,
            "result_reason": "showdown",
            "showdown": True,
        },
        "bot_family_id": "family-x",
    }

    sanitized = _sanitize_hand_for_miner(hand)

    assert "label" not in sanitized
    assert "bot_family_id" not in sanitized
    assert "is_bot" not in sanitized["players"][0]
    assert sanitized["outcome"]["winners"] == []
    assert sanitized["outcome"]["payouts"] == {}
    assert sanitized["outcome"]["showdown"] is False
    assert sanitized["outcome"]["result_reason"] == ""
