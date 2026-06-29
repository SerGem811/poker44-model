"""Deterministic chunk-level feature extraction for Poker44 bot detection.

The validator never gives miners labels at runtime and heavily degrades the
payload before sending it (see ``poker44/validator/payload_view.py``):

  * only a 5-8 action window per hand survives,
  * bet sizes are normalised to big-blind units and bucketed with seeded noise,
  * hole cards, board cards and outcomes are stripped.

So the only learnable signal is *behavioural regularity*. Bots tend to:
  * concentrate bet sizing on a few modal buckets (low sizing entropy/variance),
  * keep aggression/continuation patterns unusually consistent,
  * deviate from the human distribution of action-type mixes and street depth.

This module turns one chunk (a list of miner-visible hand dicts) into a fixed,
ordered numeric vector. It is pure-python + optional numpy, fully deterministic,
and is the *same* code path used for both training and live inference - never
train on features the live miner cannot reproduce.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Sequence

_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_AGGRO = ("bet", "raise")
_PASSIVE = ("check", "call")
# Big-blind buckets the validator can emit (payload_view._VISIBLE_BB_BUCKETS).
_BB_BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)


def _entropy(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p + 1e-12)
    return h


def _stats(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, std, coefficient_of_variation)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(max(0.0, var))
    cv = std / (abs(mean) + 1e-9)
    return mean, std, cv


def _nearest_bucket_gap(value: float) -> float:
    """Distance from the nearest canonical BB bucket (bots snap to buckets)."""
    if value <= 0:
        return 0.0
    return min(abs(value - b) for b in _BB_BUCKETS)


# Ordered feature names. Keep this list and _vectorize() in lock-step.
FEATURE_NAMES: List[str] = [
    "n_hands",
    "mean_actions_per_hand",
    "std_actions_per_hand",
    "mean_streets_per_hand",
    "frac_reach_turn_plus",
    "mean_players",
    # table-level action-type fractions
    "frac_fold",
    "frac_check",
    "frac_call",
    "frac_bet",
    "frac_raise",
    "aggression_factor",
    "action_type_entropy",
    # bet/raise sizing regularity (the strongest bot tell)
    "size_cv_bb",
    "size_bucket_entropy",
    "size_modal_frac",
    "size_bucket_snap",
    # pot dynamics
    "mean_pot_growth",
    "std_pot_growth",
    # hero-centric behaviour (hero_seat is the focal/labelled player)
    "hero_action_frac",
    "hero_frac_fold",
    "hero_frac_aggro",
    "hero_size_cv_bb",
    "hero_aggression_factor",
    # consistency across hands (low variance == automated)
    "aggro_frac_cv_across_hands",
    "raise_after_raise_frac",
]


def _hand_features(hand: Dict[str, Any]) -> Dict[str, Any]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    hero_seat = 0
    meta = hand.get("metadata") or {}
    try:
        hero_seat = int(meta.get("hero_seat", 0) or 0)
    except (TypeError, ValueError):
        hero_seat = 0

    counts: Counter = Counter()
    hero_counts: Counter = Counter()
    sizes: List[float] = []
    hero_sizes: List[float] = []
    pot_growth: List[float] = []
    prev_type: str | None = None
    raise_after_raise = 0
    raise_total = 0

    for a in actions:
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type", "") or "").lower()
        if at not in _ACTION_TYPES:
            continue
        counts[at] += 1
        try:
            seat = int(a.get("actor_seat", 0) or 0)
        except (TypeError, ValueError):
            seat = 0
        is_hero = seat != 0 and seat == hero_seat
        if is_hero:
            hero_counts[at] += 1

        amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
        if at in _AGGRO and amt > 0:
            sizes.append(amt)
            if is_hero:
                hero_sizes.append(amt)

        pb = float(a.get("pot_before", 0.0) or 0.0)
        pa = float(a.get("pot_after", 0.0) or 0.0)
        if pb > 0:
            pot_growth.append(max(0.0, (pa - pb) / pb))

        if at == "raise":
            raise_total += 1
            if prev_type == "raise":
                raise_after_raise += 1
        prev_type = at

    return {
        "counts": counts,
        "hero_counts": hero_counts,
        "sizes": sizes,
        "hero_sizes": hero_sizes,
        "pot_growth": pot_growth,
        "n_actions": sum(counts.values()),
        "n_streets": len(streets),
        "n_players": len(players),
        "raise_after_raise": raise_after_raise,
        "raise_total": raise_total,
    }


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> List[float]:
    """Project one chunk (list of miner-visible hand dicts) to a feature vector."""
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    if not hands:
        return [0.0] * len(FEATURE_NAMES)

    hf = [_hand_features(h) for h in hands]

    total = Counter()
    hero_total = Counter()
    all_sizes: List[float] = []
    hero_sizes: List[float] = []
    all_pot_growth: List[float] = []
    actions_per_hand: List[float] = []
    streets_per_hand: List[float] = []
    players_per_hand: List[float] = []
    aggro_frac_per_hand: List[float] = []
    hero_actions = 0
    total_actions = 0
    raise_after_raise = 0
    raise_total = 0

    for f in hf:
        total.update(f["counts"])
        hero_total.update(f["hero_counts"])
        all_sizes.extend(f["sizes"])
        hero_sizes.extend(f["hero_sizes"])
        all_pot_growth.extend(f["pot_growth"])
        actions_per_hand.append(f["n_actions"])
        streets_per_hand.append(f["n_streets"])
        players_per_hand.append(f["n_players"])
        hero_actions += sum(f["hero_counts"].values())
        total_actions += f["n_actions"]
        raise_after_raise += f["raise_after_raise"]
        raise_total += f["raise_total"]
        na = max(1, f["n_actions"])
        aggro_frac_per_hand.append(
            sum(f["counts"].get(k, 0) for k in _AGGRO) / na
        )

    n_actions = max(1, sum(total.values()))
    fold = total.get("fold", 0)
    check = total.get("check", 0)
    call = total.get("call", 0)
    bet = total.get("bet", 0)
    raise_ = total.get("raise", 0)
    aggro = bet + raise_
    passive = check + call
    aggression_factor = aggro / (passive + 1e-9)

    size_mean, size_std, size_cv = _stats(all_sizes)
    size_buckets = Counter(min(range(len(_BB_BUCKETS)), key=lambda i: abs(_BB_BUCKETS[i] - s)) for s in all_sizes)
    size_bucket_entropy = _entropy(list(size_buckets.values()))
    size_modal_frac = (max(size_buckets.values()) / len(all_sizes)) if all_sizes else 0.0
    size_snap = (sum(_nearest_bucket_gap(s) for s in all_sizes) / len(all_sizes)) if all_sizes else 0.0

    pot_mean, pot_std, _ = _stats(all_pot_growth)
    _, _, aggro_cv_across = _stats(aggro_frac_per_hand)

    hero_n = max(1, sum(hero_total.values()))
    hero_aggro = hero_total.get("bet", 0) + hero_total.get("raise", 0)
    hero_passive = hero_total.get("check", 0) + hero_total.get("call", 0)
    _, _, hero_size_cv = _stats(hero_sizes)

    vec = [
        float(len(hands)),
        *(_stats(actions_per_hand)[:2]),
        sum(streets_per_hand) / len(hands),
        sum(1 for s in streets_per_hand if s >= 3) / len(hands),
        sum(players_per_hand) / len(hands),
        fold / n_actions,
        check / n_actions,
        call / n_actions,
        bet / n_actions,
        raise_ / n_actions,
        aggression_factor,
        _entropy([fold, check, call, bet, raise_]),
        size_cv,
        size_bucket_entropy,
        size_modal_frac,
        size_snap,
        pot_mean,
        pot_std,
        hero_actions / max(1, total_actions),
        hero_total.get("fold", 0) / hero_n,
        hero_aggro / hero_n,
        hero_size_cv,
        hero_aggro / (hero_passive + 1e-9),
        aggro_cv_across,
        raise_after_raise / max(1, raise_total),
    ]
    # Guard against any non-finite leakage so downstream models stay stable.
    return [float(v) if math.isfinite(v) else 0.0 for v in vec]


assert len(FEATURE_NAMES) == len(extract_chunk_features([{}])), (
    "FEATURE_NAMES and extract_chunk_features() are out of sync"
)
