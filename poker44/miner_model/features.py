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

Feature design (145 total):
  - 19 per-hand feature values × 7 statistics = 133 stat features
  - 8 chunk-level signature features
  - 4 global behavior rate features

All features are scale-invariant (fractions, CVs, entropies, bucket-based).
No absolute BB magnitude features are used — that would cause score saturation
on live data where the BB distribution differs from training data.

This module is pure-python + optional numpy, fully deterministic, and is the
*same* code path used for both training and live inference.
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

# Names for the 19 per-hand features (in order).
_PER_HAND_FEATURE_NAMES = [
    "frac_fold",            # 0
    "frac_check",           # 1
    "frac_call",            # 2
    "frac_bet",             # 3
    "frac_raise",           # 4
    "action_entropy",       # 5
    "aggression_factor",    # 6
    "size_cv",              # 7
    "size_bucket_entropy",  # 8
    "size_modal_frac",      # 9
    "size_bucket_snap",     # 10
    "hero_frac",            # 11
    "hero_fold_frac",       # 12
    "hero_aggro_frac",      # 13
    "hero_aggression_factor",  # 14
    "hero_size_cv",         # 15
    "pot_growth_mean",      # 16
    "reach_turn_plus",      # 17
    "raise_after_raise_frac",  # 18
]

_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _entropy(counts: Sequence[float]) -> float:
    """Shannon entropy of a count distribution."""
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


def _stats(values: Sequence[float]) -> tuple:
    """Return (mean, std, min, max, q10, q50, q90) for a sequence of values.

    For < 2 samples the std is 0; for percentiles with < 2 samples the single
    value (or 0) is used for all quantiles.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        v = float(values[0])
        return (v, 0.0, v, v, v, v, v)
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(max(0.0, var))
    vmin = float(min(values))
    vmax = float(max(values))

    def _percentile(sorted_vals: list, p: float) -> float:
        """Linear interpolation percentile on a sorted list."""
        nn = len(sorted_vals)
        if nn == 0:
            return 0.0
        if nn == 1:
            return float(sorted_vals[0])
        idx = p * (nn - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= nn:
            return float(sorted_vals[-1])
        frac = idx - lo
        return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac

    sv = sorted(values)
    q10 = _percentile(sv, 0.10)
    q50 = _percentile(sv, 0.50)
    q90 = _percentile(sv, 0.90)
    return (float(mean), float(std), vmin, vmax, q10, q50, q90)


def _nearest_bucket_gap(value: float) -> float:
    """Mean absolute distance from the nearest canonical BB bucket."""
    if value <= 0:
        return 0.0
    return min(abs(value - b) for b in _BB_BUCKETS)


def _bb_bucket_index(value: float) -> int:
    """Return the index of the nearest BB bucket for a given value."""
    return min(range(len(_BB_BUCKETS)), key=lambda i: abs(_BB_BUCKETS[i] - value))


def _lag1_autocorr(values: Sequence[float]) -> float:
    """Pearson lag-1 autocorrelation. Bots show high positive autocorr (consistent hand-to-hand)."""
    n = len(values)
    if n < 4:
        return 0.0
    x = [float(v) for v in values[:-1]]
    y = [float(v) for v in values[1:]]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x) + 1e-12)
    dy = math.sqrt(sum((b - my) ** 2 for b in y) + 1e-12)
    r = num / (dx * dy)
    return max(-1.0, min(1.0, r))


def _half_delta(values: Sequence[float]) -> float:
    """Abs difference of means between first and second half. Bots show near-zero drift."""
    n = len(values)
    if n < 4:
        return 0.0
    mid = n // 2
    first = sum(values[:mid]) / mid
    second = sum(values[mid:]) / max(1, n - mid)
    return abs(first - second)


# ---------------------------------------------------------------------------
# FEATURE_NAMES — exactly 151 entries
# 133 stat features (19 per-hand × 7 stats) + 8 chunk + 4 global + 6 temporal
# ---------------------------------------------------------------------------

# 133 stat features: for each of 19 per-hand features, 7 stats in order.
FEATURE_NAMES: List[str] = []
for _feat in _PER_HAND_FEATURE_NAMES:
    for _stat in _STATS:
        FEATURE_NAMES.append(f"{_stat}_{_feat}")

# 8 chunk-level signature features
FEATURE_NAMES += [
    "n_hands",
    "mean_n_players",
    "unique_action_types",
    "unique_bb_buckets",
    "top_action_share",
    "top_bb_bucket_share",
    "hero_participation_rate",
    "aggro_cv_across_hands",
]

# 4 global behavior rate features
FEATURE_NAMES += [
    "global_aggression_rate",
    "global_action_entropy",
    "global_actor_entropy",
    "long_hand_rate",
]

# 6 temporal consistency features (hand-to-hand patterns)
FEATURE_NAMES += [
    "lag1_autocorr_aggression",   # lag-1 autocorr of per-hand aggro fraction
    "lag1_autocorr_fold",         # lag-1 autocorr of per-hand fold fraction
    "lag1_autocorr_size_cv",      # lag-1 autocorr of per-hand size CV
    "aggro_half_delta",           # |first_half_aggro - second_half_aggro|
    "fold_half_delta",            # |first_half_fold - second_half_fold|
    "size_cv_half_delta",         # |first_half_size_cv - second_half_size_cv|
]


def _compute_per_hand_features(hand: Dict[str, Any]) -> List[float]:
    """Compute the 19 per-hand feature values for a single hand dict."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    meta = hand.get("metadata") or {}
    try:
        hero_seat = int(meta.get("hero_seat", 0) or 0)
    except (TypeError, ValueError):
        hero_seat = 0

    counts: Counter = Counter()
    hero_counts: Counter = Counter()
    sizes: List[float] = []
    hero_sizes: List[float] = []
    pot_growths: List[float] = []
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
            pot_growths.append(max(0.0, (pa - pb) / pb))

        if at == "raise":
            raise_total += 1
            if prev_type == "raise":
                raise_after_raise += 1
        prev_type = at

    n_actions = sum(counts.values())
    fold = counts.get("fold", 0)
    check = counts.get("check", 0)
    call = counts.get("call", 0)
    bet = counts.get("bet", 0)
    raise_ = counts.get("raise", 0)
    aggro = bet + raise_
    passive = check + call

    # 0: frac_fold
    feat0 = fold / max(1, n_actions)
    # 1: frac_check
    feat1 = check / max(1, n_actions)
    # 2: frac_call
    feat2 = call / max(1, n_actions)
    # 3: frac_bet
    feat3 = bet / max(1, n_actions)
    # 4: frac_raise
    feat4 = raise_ / max(1, n_actions)
    # 5: action_entropy
    feat5 = _entropy([fold, check, call, bet, raise_])
    # 6: aggression_factor
    feat6 = aggro / max(1e-9, passive)
    # 7: size_cv = CV of bet/raise normalized_amount_bb
    if sizes:
        s_mean = sum(sizes) / len(sizes)
        if len(sizes) > 1:
            s_var = sum((s - s_mean) ** 2 for s in sizes) / len(sizes)
            s_std = math.sqrt(max(0.0, s_var))
        else:
            s_std = 0.0
        feat7 = s_std / (abs(s_mean) + 1e-9)
    else:
        feat7 = 0.0
    # 8: size_bucket_entropy = entropy of BB bucket assignments for bet/raise
    if sizes:
        bucket_counts: Counter = Counter(_bb_bucket_index(s) for s in sizes)
        feat8 = _entropy(list(bucket_counts.values()))
    else:
        feat8 = 0.0
    # 9: size_modal_frac = fraction of bet/raise at the most common BB bucket
    if sizes:
        bucket_counts2: Counter = Counter(_bb_bucket_index(s) for s in sizes)
        feat9 = max(bucket_counts2.values()) / len(sizes)
    else:
        feat9 = 0.0
    # 10: size_bucket_snap = mean distance from each bet/raise to nearest bucket
    if sizes:
        feat10 = sum(_nearest_bucket_gap(s) for s in sizes) / len(sizes)
    else:
        feat10 = 0.0
    # 11: hero_frac = hero_actions / total_actions
    hero_n_actions = sum(hero_counts.values())
    feat11 = hero_n_actions / max(1, n_actions)
    # 12: hero_fold_frac = hero_folds / hero_actions
    feat12 = hero_counts.get("fold", 0) / max(1, hero_n_actions)
    # 13: hero_aggro_frac = hero(bet+raise) / hero_actions
    hero_aggro = hero_counts.get("bet", 0) + hero_counts.get("raise", 0)
    feat13 = hero_aggro / max(1, hero_n_actions)
    # 14: hero_aggression_factor = hero(bet+raise) / hero(check+call)
    hero_passive = hero_counts.get("check", 0) + hero_counts.get("call", 0)
    feat14 = hero_aggro / max(1e-9, hero_passive)
    # 15: hero_size_cv = CV of hero bet/raise sizes
    if hero_sizes:
        hs_mean = sum(hero_sizes) / len(hero_sizes)
        if len(hero_sizes) > 1:
            hs_var = sum((s - hs_mean) ** 2 for s in hero_sizes) / len(hero_sizes)
            hs_std = math.sqrt(max(0.0, hs_var))
        else:
            hs_std = 0.0
        feat15 = hs_std / (abs(hs_mean) + 1e-9)
    else:
        feat15 = 0.0
    # 16: pot_growth_mean = mean of (pot_after-pot_before)/pot_before
    feat16 = sum(pot_growths) / len(pot_growths) if pot_growths else 0.0
    # 17: reach_turn_plus = 1 if n_streets >= 3
    feat17 = 1.0 if len(streets) >= 3 else 0.0
    # 18: raise_after_raise_frac = raise_after_raise / raise_count
    feat18 = raise_after_raise / max(1, raise_total)

    return [
        feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9,
        feat10, feat11, feat12, feat13, feat14, feat15, feat16, feat17, feat18,
    ]


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> List[float]:
    """Project one chunk (list of miner-visible hand dicts) to a 145-float vector.

    Feature layout:
      [0:133]   per-hand statistics: 19 features × 7 stats (mean/std/min/max/q10/q50/q90)
      [133:141] chunk-level signature features (8)
      [141:145] global behavior rate features (4)
    """
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    if not hands:
        return [0.0] * len(FEATURE_NAMES)

    # --- Compute per-hand 19-vectors ---
    per_hand: List[List[float]] = [_compute_per_hand_features(h) for h in hands]
    n_per = len(_PER_HAND_FEATURE_NAMES)  # 19

    # --- 133 stat features: 19 × 7 ---
    stat_features: List[float] = []
    for fi in range(n_per):
        col = [per_hand[hi][fi] for hi in range(len(hands))]
        st = _stats(col)  # (mean, std, min, max, q10, q50, q90)
        stat_features.extend(st)

    # --- Chunk-level signature features (8) ---
    # For these we need raw hand data, not just per-hand vectors.
    n_hands = len(hands)

    # mean_n_players
    players_per_hand = [len(h.get("players") or []) for h in hands]
    mean_n_players = sum(players_per_hand) / n_hands

    # unique_action_types: distinct action types used across entire chunk
    all_action_type_set: set = set()
    chunk_action_counts: Counter = Counter()
    chunk_actor_counts: Counter = Counter()
    chunk_sizes_all: List[float] = []
    chunk_sizes_bucket_indices: List[int] = []
    hands_with_hero_action: int = 0

    for h in hands:
        actions = h.get("actions") or []
        meta = h.get("metadata") or {}
        try:
            hero_seat = int(meta.get("hero_seat", 0) or 0)
        except (TypeError, ValueError):
            hero_seat = 0
        hand_has_hero = False
        for a in actions:
            if not isinstance(a, dict):
                continue
            at = str(a.get("action_type", "") or "").lower()
            if at not in _ACTION_TYPES:
                continue
            all_action_type_set.add(at)
            chunk_action_counts[at] += 1
            try:
                seat = int(a.get("actor_seat", 0) or 0)
            except (TypeError, ValueError):
                seat = 0
            chunk_actor_counts[seat] += 1
            if seat != 0 and seat == hero_seat:
                hand_has_hero = True
            amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
            if at in _AGGRO and amt > 0:
                chunk_sizes_all.append(amt)
                chunk_sizes_bucket_indices.append(_bb_bucket_index(amt))
        if hand_has_hero:
            hands_with_hero_action += 1

    unique_action_types = float(len(all_action_type_set))

    # unique_bb_buckets: distinct bucket indices used across all bet/raise
    unique_bb_buckets = float(len(set(chunk_sizes_bucket_indices)))

    # top_action_share: fraction of all actions that are the most common type
    total_chunk_actions = sum(chunk_action_counts.values())
    if total_chunk_actions > 0:
        top_action_share = max(chunk_action_counts.values()) / total_chunk_actions
    else:
        top_action_share = 0.0

    # top_bb_bucket_share: fraction of bet/raise at the most common BB bucket
    if chunk_sizes_bucket_indices:
        bucket_idx_counts: Counter = Counter(chunk_sizes_bucket_indices)
        top_bb_bucket_share = max(bucket_idx_counts.values()) / len(chunk_sizes_bucket_indices)
    else:
        top_bb_bucket_share = 0.0

    # hero_participation_rate: fraction of hands where hero takes >= 1 action
    hero_participation_rate = hands_with_hero_action / n_hands

    # aggro_cv_across_hands: CV of per-hand aggression fraction across hands
    # per-hand aggression fraction is feature index 13 (hero_aggro_frac)?
    # No — use overall aggression fraction per hand: (bet+raise)/total_actions
    # This mirrors the original code's "aggro_frac_per_hand".
    aggro_frac_per_hand: List[float] = []
    for h in hands:
        actions = h.get("actions") or []
        n_act = 0
        n_aggro = 0
        for a in actions:
            if not isinstance(a, dict):
                continue
            at = str(a.get("action_type", "") or "").lower()
            if at not in _ACTION_TYPES:
                continue
            n_act += 1
            if at in _AGGRO:
                n_aggro += 1
        af = n_aggro / max(1, n_act)
        aggro_frac_per_hand.append(af)

    if len(aggro_frac_per_hand) >= 2:
        af_mean = sum(aggro_frac_per_hand) / len(aggro_frac_per_hand)
        af_var = sum((v - af_mean) ** 2 for v in aggro_frac_per_hand) / len(aggro_frac_per_hand)
        af_std = math.sqrt(max(0.0, af_var))
        aggro_cv_across_hands = af_std / (abs(af_mean) + 1e-9)
    else:
        aggro_cv_across_hands = 0.0

    chunk_sig_features = [
        float(n_hands),
        mean_n_players,
        unique_action_types,
        unique_bb_buckets,
        top_action_share,
        top_bb_bucket_share,
        hero_participation_rate,
        aggro_cv_across_hands,
    ]

    # --- Global behavior rate features (4) ---
    # global_aggression_rate: total(bet+raise) / total_all_actions
    total_all = sum(chunk_action_counts.values())
    global_aggro = chunk_action_counts.get("bet", 0) + chunk_action_counts.get("raise", 0)
    global_aggression_rate = global_aggro / max(1, total_all)

    # global_action_entropy: entropy of (fold,check,call,bet,raise) totals
    global_action_entropy = _entropy([
        chunk_action_counts.get(at, 0) for at in _ACTION_TYPES
    ])

    # global_actor_entropy: entropy of action counts by actor_seat
    if len(chunk_actor_counts) > 1:
        global_actor_entropy = _entropy(list(chunk_actor_counts.values()))
    else:
        global_actor_entropy = 0.0

    # long_hand_rate: fraction of hands with n_streets >= 3
    long_hand_rate = sum(
        1 for h in hands if len(h.get("streets") or []) >= 3
    ) / n_hands

    global_rate_features = [
        global_aggression_rate,
        global_action_entropy,
        global_actor_entropy,
        long_hand_rate,
    ]

    # --- Temporal consistency features (6) ---
    # Use per-hand aggression, fold, and size_cv sequences.
    aggro_frac_ph = [ph[3] + ph[4] for ph in per_hand]   # frac_bet + frac_raise
    fold_frac_ph  = [ph[0] for ph in per_hand]            # frac_fold
    size_cv_ph    = [ph[7] for ph in per_hand]            # size_cv

    temporal_features = [
        _lag1_autocorr(aggro_frac_ph),
        _lag1_autocorr(fold_frac_ph),
        _lag1_autocorr(size_cv_ph),
        _half_delta(aggro_frac_ph),
        _half_delta(fold_frac_ph),
        _half_delta(size_cv_ph),
    ]

    # --- Assemble final vector ---
    vec = stat_features + chunk_sig_features + global_rate_features + temporal_features

    # Guard against any non-finite leakage so downstream models stay stable.
    return [float(v) if math.isfinite(v) else 0.0 for v in vec]


assert len(FEATURE_NAMES) == len(extract_chunk_features([{}])), (
    "FEATURE_NAMES and extract_chunk_features() are out of sync"
)
