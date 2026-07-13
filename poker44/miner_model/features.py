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

Feature design (307 total):
  - 19 per-hand feature values × 7 statistics = 133 stat features
  - 10 schema per-hand features × 7 statistics = 70 stat features
  - 8 extra per-hand features × 7 statistics = 56 stat features (blind share,
    unique actor ratio, street count, player count, absolute BB sizing, stacks)
  - 8 chunk-level signature features (original)
  - 4 global behavior rate features
  - 12 temporal consistency features (lag-1/2 autocorr, half-delta, trend slope)
  - 12 chunk-level sequence-collision features (novel)
  - 6 pot-fraction and action-pattern features (bet/pot ratio CV, clustering, bigrams)
  - 6 intra-session consistency features (quartile std + Q4-Q1 drift)

The extra per-hand features use BB-normalized values (amount_bb, stacks/bb)
since the validator consistently normalizes these across all stake levels.

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

# Names for 10 additional schema per-hand features (indices 19-28).
_SCHEMA_PER_HAND_FEATURE_NAMES = [
    "schema_actor_switch_rate",     # 19: how often acting seat changes
    "schema_actor_run_max_share",   # 20: longest same-actor run / n_actions
    "schema_action_run_max_share",  # 21: longest same-action-type run / n_actions
    "schema_street_entropy",        # 22: entropy of street distribution
    "schema_preflop_share",         # 23: preflop actions / total actions
    "schema_postflop_share",        # 24: postflop actions / total actions
    "schema_pot_monotonic_rate",    # 25: fraction of consecutive pot pairs that increase
    "schema_raise_to_share",        # 26: fraction of actions with raise_to present
    "schema_nonzero_amount_share",  # 27: fraction of actions with non-zero amount
    "schema_starting_stack_iqr_bb", # 28: IQR of starting stacks in BB units
]

# Names for 8 supplemental per-hand features (indices 29-36).
# These capture blind actions, absolute bet sizing, stack depth, and player count.
_EXTRA_PER_HAND_FEATURE_NAMES = [
    "extra_blind_share",         # 29: (SB+BB) / all_actions_incl_blinds
    "extra_unique_actor_share",  # 30: unique_actors / n_actor_actions
    "extra_street_count",        # 31: number of distinct streets in the hand
    "extra_n_players",           # 32: number of players in the hand
    "extra_amount_mean_bb",      # 33: mean bet/raise amount in BB
    "extra_amount_q90_bb",       # 34: 90th pct bet/raise amount in BB
    "extra_stack_mean_bb",       # 35: mean starting stack in BB
    "extra_stack_std_bb",        # 36: std of starting stacks in BB
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


def _lagk_autocorr(values: Sequence[float], k: int = 1) -> float:
    """Pearson lag-k autocorrelation. Bots show consistent hand-to-hand patterns."""
    n = len(values)
    if n < k + 3:
        return 0.0
    x = [float(v) for v in values[:-k]]
    y = [float(v) for v in values[k:]]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x) + 1e-12)
    dy = math.sqrt(sum((b - my) ** 2 for b in y) + 1e-12)
    r = num / (dx * dy)
    return max(-1.0, min(1.0, r))


def _lag1_autocorr(values: Sequence[float]) -> float:
    return _lagk_autocorr(values, 1)


def _lag2_autocorr(values: Sequence[float]) -> float:
    return _lagk_autocorr(values, 2)


def _lag3_autocorr(values: Sequence[float]) -> float:
    return _lagk_autocorr(values, 3)


def _trend_slope(values: Sequence[float]) -> float:
    """Normalized linear regression slope over the session. Bots show near-zero drift."""
    n = len(values)
    if n < 4:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(values) / n
    num = sum((i - mx) * (values[i] - my) for i in range(n))
    denom = sum((i - mx) ** 2 for i in range(n))
    if denom < 1e-12:
        return 0.0
    return (num / denom) / (abs(my) + 1e-9)


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
# FEATURE_NAMES — 295 entries
# 133 stat features (19 per-hand × 7 stats)
# 70 schema stat features (10 schema per-hand × 7 stats)
# 56 extra stat features (8 extra per-hand × 7 stats)
# 8 chunk-level signature features (original)
# 4 global behavior rate features
# 12 temporal consistency features (lag-1/2 autocorr, half-delta, trend slope)
# 12 chunk-level sequence-collision features (novel)
# ---------------------------------------------------------------------------

# 133 stat features: for each of 19 per-hand features, 7 stats in order.
FEATURE_NAMES: List[str] = []
for _feat in _PER_HAND_FEATURE_NAMES:
    for _stat in _STATS:
        FEATURE_NAMES.append(f"{_stat}_{_feat}")

# 70 schema stat features: for each of 10 schema per-hand features, 7 stats.
for _feat in _SCHEMA_PER_HAND_FEATURE_NAMES:
    for _stat in _STATS:
        FEATURE_NAMES.append(f"{_stat}_{_feat}")

# 56 extra stat features: for each of 8 extra per-hand features, 7 stats.
for _feat in _EXTRA_PER_HAND_FEATURE_NAMES:
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

# 12 temporal consistency features (hand-to-hand patterns)
FEATURE_NAMES += [
    "lag1_autocorr_aggression",   # lag-1 autocorr of per-hand aggro fraction
    "lag1_autocorr_fold",         # lag-1 autocorr of per-hand fold fraction
    "lag1_autocorr_size_cv",      # lag-1 autocorr of per-hand size CV
    "aggro_half_delta",           # |first_half_aggro - second_half_aggro|
    "fold_half_delta",            # |first_half_fold - second_half_fold|
    "size_cv_half_delta",         # |first_half_size_cv - second_half_size_cv|
    "lag2_autocorr_aggression",   # lag-2 autocorr of per-hand aggro fraction
    "lag2_autocorr_fold",         # lag-2 autocorr of per-hand fold fraction
    "lag2_autocorr_size_cv",      # lag-2 autocorr of per-hand size CV
    "trend_slope_aggression",     # normalised linear slope of aggro fraction over session
    "trend_slope_fold",           # normalised linear slope of fold fraction over session
    "trend_slope_size_cv",        # normalised linear slope of size CV over session
    "lag3_autocorr_aggression",   # lag-3 autocorr of per-hand aggro fraction
    "lag3_autocorr_fold",         # lag-3 autocorr of per-hand fold fraction
    "lag3_autocorr_size_cv",      # lag-3 autocorr of per-hand size CV
]

# 12 chunk-level sequence-collision features
# These capture how repetitively a bot replays the same action sequences across hands.
FEATURE_NAMES += [
    "seq_action_sig_top_share",        # fraction of hands with the most common action-type sequence
    "seq_action_sig_unique_share",     # unique action-type sequences / n_hands
    "seq_actor_sig_top_share",         # fraction of hands with the most common actor-seat sequence
    "seq_actor_sig_unique_share",      # unique actor-seat sequences / n_hands
    "seq_street_sig_top_share",        # fraction of hands with the most common street sequence
    "seq_street_sig_unique_share",     # unique street sequences / n_hands
    "seq_amount_bucket_sig_top_share", # fraction of hands with the most common amount-bucket sequence
    "seq_amount_bucket_sig_unique_share", # unique amount-bucket sequences / n_hands
    "seq_high_aggression_hand_rate",   # fraction of hands with aggression_share >= 0.35
    "seq_low_action_entropy_hand_rate",# fraction of hands with action_entropy <= 0.35
    "seq_high_actor_entropy_hand_rate",# fraction of hands with actor_entropy >= 0.75
    "seq_long_action_hand_rate",       # fraction of hands with >= 12 actions
]

# 6 pot-fraction and action-pattern features (novel)
# Bots bet a fixed fraction of pot; humans vary. Action bigrams expose stereotyped play.
_STD_POT_FRACS = (0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.50, 2.00)
FEATURE_NAMES += [
    "bet_pot_ratio_cv",           # CV of all bet/raise (amount/pot_before) across chunk
    "bet_pot_ratio_cluster_frac", # fraction of bets within ±0.08 of a standard pot fraction
    "hero_bet_pot_ratio_cv",      # CV of hero-only bet/raise pot fractions
    "lag1_autocorr_hero_bet_pot", # lag-1 autocorr of per-hand avg hero bet/pot ratio
    "hero_fold_to_bet_rate",      # hero folds / (hero faces bet or raise)
    "action_bigram_entropy",      # entropy of consecutive action-type bigrams across all hands
]

# 6 intra-session consistency features (novel)
# Divide session into 4 quarters; compute how stable fold/aggression/size_cv are over the session.
# Bots maintain a fixed strategy throughout; humans adapt.
FEATURE_NAMES += [
    "fold_quartile_std",         # std of per-quartile mean fold rate (bots ≈ 0)
    "aggro_quartile_std",        # std of per-quartile mean aggression fraction
    "size_cv_quartile_std",      # std of per-quartile mean size CV
    "fold_q4_minus_q1",          # Q4 fold rate minus Q1 fold rate (session drift)
    "aggro_q4_minus_q1",         # Q4 aggression fraction minus Q1 (session drift)
    "size_cv_q4_minus_q1",       # Q4 size CV minus Q1 (session drift)
]

# 4 street action share features (audit top-2 discriminators)
# flop_action_share (inverse, AP=0.662) and river_action_share (direct, AP=0.622)
FEATURE_NAMES += [
    "chunk_flop_action_share",   # flop actions / total actions across chunk
    "chunk_river_action_share",  # river actions / total actions across chunk
    "hero_flop_action_share",    # hero flop actions / hero total actions
    "hero_river_action_share",   # hero river actions / hero total actions
    "flop_aggro_share",          # flop bet+raise / all bet+raise across chunk (bots fold preflop → less flop aggression)
    "turn_aggro_share",          # turn bet+raise / all bet+raise across chunk
    "river_aggro_share",         # river bet+raise / all bet+raise across chunk (bots commit on river → more river aggression)
]

# 4 raw amount + pot features (benchmark audit top-discriminators, AP 0.63-0.65 each)
FEATURE_NAMES += [
    "mean_normalized_amount_bb",  # mean of all non-zero bet amounts (in BB) across chunk
    "std_normalized_amount_bb",   # std of the same (bots use fixed sizes → lower std)
    "mean_pot_before",            # mean pot size before action across all actions in chunk
    "mean_pot_after",             # mean pot size after action across all actions in chunk
]


def _compute_schema_per_hand_features(hand: Dict[str, Any]) -> List[float]:
    """Compute 10 additional schema per-hand features (indices 19-28)."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    meta = hand.get("metadata") or {}

    action_types: List[str] = []
    actor_seats: List[int] = []
    street_names: List[str] = []
    amounts: List[float] = []
    pot_after_seq: List[float] = []
    raise_to_count = 0

    for a in actions:
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type", "") or "").lower()
        if not at:
            continue
        action_types.append(at)
        try:
            seat = int(a.get("actor_seat", 0) or 0)
        except (TypeError, ValueError):
            seat = 0
        if seat > 0:
            actor_seats.append(seat)
        street_names.append(str(a.get("street", "") or "").lower())
        amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
        amounts.append(max(0.0, amt))
        pa = float(a.get("pot_after", 0.0) or 0.0)
        pot_after_seq.append(max(0.0, pa))
        if a.get("raise_to") is not None:
            raise_to_count += 1

    n_actions = max(len(action_types), 1)
    n_actors = max(len(actor_seats), 1)

    # 19: actor_switch_rate — how often consecutive actors differ
    actor_switches = sum(1 for i in range(1, len(actor_seats)) if actor_seats[i] != actor_seats[i - 1])
    feat19 = actor_switches / max(len(actor_seats) - 1, 1)

    # 20: actor_run_max_share — longest run of same actor / n_actor_actions
    feat20 = _max_run_share_list(actor_seats)

    # 21: action_run_max_share — longest run of same action type / n_actions
    feat21 = _max_run_share_list(action_types)

    # 22: street_entropy — Shannon entropy over street distribution
    street_counts = Counter(s for s in street_names if s)
    feat22 = _entropy(list(street_counts.values()))

    # 23: preflop_share — preflop actions / total
    preflop_n = sum(1 for s in street_names if s == "preflop")
    feat23 = preflop_n / n_actions

    # 24: postflop_share — non-preflop, non-empty actions / total
    postflop_n = sum(1 for s in street_names if s and s != "preflop")
    feat24 = postflop_n / n_actions

    # 25: pot_monotonic_rate — fraction of consecutive pot_after pairs that increase
    if len(pot_after_seq) >= 2:
        mono = sum(1 for i in range(1, len(pot_after_seq)) if pot_after_seq[i] + 1e-9 >= pot_after_seq[i - 1])
        feat25 = mono / (len(pot_after_seq) - 1)
    else:
        feat25 = 0.0

    # 26: raise_to_share — fraction of actions that have a raise_to field
    feat26 = raise_to_count / n_actions

    # 27: nonzero_amount_share — fraction of actions with positive amount
    nonzero = sum(1 for v in amounts if v > 0)
    feat27 = nonzero / n_actions

    # 28: starting_stack_iqr_bb — IQR of starting stacks (normalized to BB units)
    # Use meta.bb if available, else assume $0.02
    try:
        bb_val = float(meta.get("bb", 0.02) or 0.02)
        if bb_val <= 0:
            bb_val = 0.02
    except (TypeError, ValueError):
        bb_val = 0.02
    stacks_bb = []
    for p in players:
        if not isinstance(p, dict):
            continue
        try:
            s = float(p.get("starting_stack", 0.0) or 0.0)
            stacks_bb.append(s / bb_val)
        except (TypeError, ValueError):
            pass
    if len(stacks_bb) >= 2:
        sv = sorted(stacks_bb)
        q75 = _stats(sv)[3]  # max used as upper bound in _stats
        # Compute proper IQR
        n_s = len(sv)
        q75_v = sv[int(round(0.75 * (n_s - 1)))]
        q25_v = sv[int(round(0.25 * (n_s - 1)))]
        feat28 = max(0.0, q75_v - q25_v)
    else:
        feat28 = 0.0

    return [feat19, feat20, feat21, feat22, feat23, feat24, feat25, feat26, feat27, feat28]


def _max_run_share_list(values: List) -> float:
    """Longest run of equal consecutive values / len(values). Returns 0 for empty."""
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest / len(values)


def _compute_extra_per_hand_features(hand: Dict[str, Any]) -> List[float]:
    """Compute 8 supplemental per-hand features (indices 29-36)."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    meta = hand.get("metadata") or {}

    try:
        bb_val = float(meta.get("bb", 0.02) or 0.02)
        if bb_val <= 0:
            bb_val = 0.02
    except (TypeError, ValueError):
        bb_val = 0.02

    all_types: List[str] = []
    actor_seats_all: List[int] = []
    streets_set: set = set()
    bet_raise_amounts: List[float] = []

    _BLIND_TYPES = {"small_blind", "big_blind"}
    _ALL_TYPES = {"fold", "check", "call", "bet", "raise", "small_blind", "big_blind"}

    for a in actions:
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type", "") or "").lower()
        if at not in _ALL_TYPES:
            continue
        all_types.append(at)

        try:
            seat = int(a.get("actor_seat", 0) or 0)
        except (TypeError, ValueError):
            seat = 0
        if seat > 0:
            actor_seats_all.append(seat)

        street = str(a.get("street", "") or "").lower()
        if street:
            streets_set.add(street)

        amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
        if at in ("bet", "raise") and amt > 0:
            bet_raise_amounts.append(amt)

    n_all = max(len(all_types), 1)
    n_actor_actions = max(len(actor_seats_all), 1)

    # 29: blind_share — (SB+BB) / all_actions_incl_blinds
    blind_n = sum(1 for t in all_types if t in _BLIND_TYPES)
    feat29 = blind_n / n_all

    # 30: unique_actor_share — distinct actors / actor-action count
    feat30 = len(set(actor_seats_all)) / n_actor_actions

    # 31: street_count — number of distinct streets played
    feat31 = float(len(streets_set))

    # 32: n_players — number of players in the hand
    feat32 = float(len([p for p in players if isinstance(p, dict)]))

    # 33: amount_mean_bb — mean bet/raise amount in BB (validator normalises)
    if bet_raise_amounts:
        feat33 = sum(bet_raise_amounts) / len(bet_raise_amounts)
    else:
        feat33 = 0.0

    # 34: amount_q90_bb — 90th pct bet/raise amount in BB
    if bet_raise_amounts:
        feat34 = _stats(bet_raise_amounts)[6]  # q90 is index 6
    else:
        feat34 = 0.0

    # 35: stack_mean_bb — mean starting stack in BB
    stacks_bb = []
    for p in players:
        if not isinstance(p, dict):
            continue
        try:
            s = float(p.get("starting_stack", 0.0) or 0.0)
            stacks_bb.append(s / bb_val)
        except (TypeError, ValueError):
            pass
    if stacks_bb:
        feat35 = sum(stacks_bb) / len(stacks_bb)
    else:
        feat35 = 0.0

    # 36: stack_std_bb — std of starting stacks in BB
    if len(stacks_bb) >= 2:
        mean_s = feat35
        var_s = sum((x - mean_s) ** 2 for x in stacks_bb) / len(stacks_bb)
        feat36 = math.sqrt(max(0.0, var_s))
    else:
        feat36 = 0.0

    return [feat29, feat30, feat31, feat32, feat33, feat34, feat35, feat36]


def _amount_bucket_label(value: float) -> str:
    """Coarse bucket label for a normalized_amount_bb value."""
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
    return "xl"


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
    # 6: aggression_factor — clamp denominator to 1 to avoid billion-scale values when passive=0
    feat6 = aggro / max(1, passive)
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
    feat14 = hero_aggro / max(1, hero_passive)
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
    """Project one chunk (list of miner-visible hand dicts) to a 321-float vector.

    Feature layout:
      [0:133]   per-hand statistics: 19 features × 7 stats
      [133:203] schema per-hand statistics: 10 features × 7 stats
      [203:259] extra per-hand statistics: 8 features × 7 stats
      [259:267] chunk-level signature features (8, original)
      [267:271] global behavior rate features (4)
      [271:278] street action share features (7)
      [278:293] temporal consistency features (15: lag1/2/3 autocorr, half-delta, trend slope)
      [293:305] sequence-collision features (12, novel)
      [305:311] pot-fraction + action-pattern features (6, novel)
      [311:317] intra-session consistency features (6, novel)
      [317:321] raw amount + pot features (4: mean/std amount_bb, mean pot_before/after)
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

    # --- 70 schema stat features: 10 × 7 ---
    schema_per_hand: List[List[float]] = [_compute_schema_per_hand_features(h) for h in hands]
    n_schema_per = len(_SCHEMA_PER_HAND_FEATURE_NAMES)  # 10
    for fi in range(n_schema_per):
        col = [schema_per_hand[hi][fi] for hi in range(len(hands))]
        st = _stats(col)
        stat_features.extend(st)

    # --- 56 extra stat features: 8 × 7 ---
    extra_per_hand: List[List[float]] = [_compute_extra_per_hand_features(h) for h in hands]
    n_extra_per = len(_EXTRA_PER_HAND_FEATURE_NAMES)  # 8
    for fi in range(n_extra_per):
        col = [extra_per_hand[hi][fi] for hi in range(len(hands))]
        st = _stats(col)
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
    chunk_street_action_counts: Counter = Counter()
    chunk_street_aggro_counts: Counter = Counter()
    chunk_sizes_all: List[float] = []
    chunk_sizes_bucket_indices: List[int] = []
    hands_with_hero_action: int = 0
    hero_flop_actions: int = 0
    hero_river_actions: int = 0
    hero_total_actions: int = 0
    all_amounts_nonzero: List[float] = []
    all_pot_before_vals: List[float] = []
    all_pot_after_vals: List[float] = []

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
            st_name = str(a.get("street", "") or "").lower()
            chunk_street_action_counts[st_name] += 1
            if at in _AGGRO:
                chunk_street_aggro_counts[st_name] += 1
            is_hero_act = (seat != 0 and seat == hero_seat)
            if is_hero_act:
                hero_total_actions += 1
                if st_name == "flop":
                    hero_flop_actions += 1
                elif st_name == "river":
                    hero_river_actions += 1
            if seat != 0 and seat == hero_seat:
                hand_has_hero = True
            amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
            if at in _AGGRO and amt > 0:
                chunk_sizes_all.append(amt)
                chunk_sizes_bucket_indices.append(_bb_bucket_index(amt))
            if amt > 0:
                all_amounts_nonzero.append(amt)
            pot_b_val = float(a.get("pot_before", 0.0) or 0.0)
            pot_a_val = float(a.get("pot_after", 0.0) or 0.0)
            if pot_b_val > 0:
                all_pot_before_vals.append(pot_b_val)
            if pot_a_val > 0:
                all_pot_after_vals.append(pot_a_val)
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

    # --- Street action share features (4) ---
    # flop/river share of total chunk actions.  Bots fold preflop more (lower flop share)
    # but commit to the river when they stay in (higher river share).
    _total_st = max(1, sum(chunk_street_action_counts.values()))
    chunk_flop_action_share  = chunk_street_action_counts.get("flop", 0)  / _total_st
    chunk_river_action_share = chunk_street_action_counts.get("river", 0) / _total_st
    hero_flop_action_share   = hero_flop_actions  / max(1, hero_total_actions)
    hero_river_action_share  = hero_river_actions / max(1, hero_total_actions)

    _total_aggro_st = max(1, sum(chunk_street_aggro_counts.values()))
    flop_aggro_share  = chunk_street_aggro_counts.get("flop",  0) / _total_aggro_st
    turn_aggro_share  = chunk_street_aggro_counts.get("turn",  0) / _total_aggro_st
    river_aggro_share = chunk_street_aggro_counts.get("river", 0) / _total_aggro_st

    street_features = [
        chunk_flop_action_share,
        chunk_river_action_share,
        hero_flop_action_share,
        hero_river_action_share,
        flop_aggro_share,
        turn_aggro_share,
        river_aggro_share,
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
        _lag2_autocorr(aggro_frac_ph),
        _lag2_autocorr(fold_frac_ph),
        _lag2_autocorr(size_cv_ph),
        _trend_slope(aggro_frac_ph),
        _trend_slope(fold_frac_ph),
        _trend_slope(size_cv_ph),
        _lag3_autocorr(aggro_frac_ph),
        _lag3_autocorr(fold_frac_ph),
        _lag3_autocorr(size_cv_ph),
    ]

    # --- Sequence-collision features (12) ---
    # Compute full-hand action/actor/street/amount-bucket sequence signatures
    # and measure how repetitive these are across all hands in the chunk.
    action_sigs: List[tuple] = []
    actor_sigs: List[tuple] = []
    street_sigs: List[tuple] = []
    amount_bucket_sigs: List[tuple] = []
    high_aggression_count = 0
    low_entropy_count = 0
    high_actor_entropy_count = 0
    long_action_count = 0

    for h, schema_ph in zip(hands, schema_per_hand):
        h_actions = h.get("actions") or []
        at_seq = tuple(
            str(a.get("action_type", "") or "").lower()
            for a in h_actions if isinstance(a, dict) and a.get("action_type")
        )
        actor_seq = tuple(
            int(a.get("actor_seat", 0) or 0)
            for a in h_actions if isinstance(a, dict) and (a.get("actor_seat") or 0) > 0
        )
        street_seq = tuple(
            str(a.get("street", "") or "").lower()
            for a in h_actions if isinstance(a, dict) and a.get("action_type")
        )
        amt_bucket_seq = tuple(
            _amount_bucket_label(float(a.get("normalized_amount_bb", 0.0) or 0.0))
            for a in h_actions if isinstance(a, dict) and a.get("action_type")
        )
        action_sigs.append(at_seq)
        actor_sigs.append(actor_seq)
        street_sigs.append(street_seq)
        amount_bucket_sigs.append(amt_bucket_seq)

        # Per-hand threshold rates (use schema features already computed)
        n_at = max(len(at_seq), 1)
        n_aggro_h = sum(1 for t in at_seq if t in _AGGRO)
        aggr_share = n_aggro_h / n_at
        at_entropy = _entropy(list(Counter(at_seq).values()))
        actor_ent = _entropy(list(Counter(actor_seq).values()))
        if aggr_share >= 0.35:
            high_aggression_count += 1
        if at_entropy <= 0.35:
            low_entropy_count += 1
        if actor_ent >= 0.75:
            high_actor_entropy_count += 1
        if len(at_seq) >= 12:
            long_action_count += 1

    n_hands_f = float(n_hands)
    action_sig_ctr = Counter(action_sigs)
    actor_sig_ctr = Counter(actor_sigs)
    street_sig_ctr = Counter(street_sigs)
    amount_sig_ctr = Counter(amount_bucket_sigs)

    collision_features = [
        max(action_sig_ctr.values()) / n_hands_f if action_sig_ctr else 0.0,
        len(action_sig_ctr) / n_hands_f,
        max(actor_sig_ctr.values()) / n_hands_f if actor_sig_ctr else 0.0,
        len(actor_sig_ctr) / n_hands_f,
        max(street_sig_ctr.values()) / n_hands_f if street_sig_ctr else 0.0,
        len(street_sig_ctr) / n_hands_f,
        max(amount_sig_ctr.values()) / n_hands_f if amount_sig_ctr else 0.0,
        len(amount_sig_ctr) / n_hands_f,
        high_aggression_count / n_hands_f,
        low_entropy_count / n_hands_f,
        high_actor_entropy_count / n_hands_f,
        long_action_count / n_hands_f,
    ]

    # --- Pot-fraction and action-pattern features (6) ---
    # bet_pot_ratio_cv: CV of (amount / pot_before) for all bet/raise actions.
    # Bots tend to use a fixed pot fraction; humans vary → bots have lower CV.
    all_pot_ratios: List[float] = []
    hero_pot_ratios_per_hand: List[float] = []  # per-hand avg hero bet/pot ratio
    hero_faces_aggr_count: int = 0    # times hero faces a bet/raise
    hero_fold_facing_aggr: int = 0    # times hero folds when facing a bet/raise
    bigram_ctr: Counter = Counter()

    for h in hands:
        actions = h.get("actions") or []
        meta = h.get("metadata") or {}
        try:
            hero_seat = int(meta.get("hero_seat", 0) or 0)
        except (TypeError, ValueError):
            hero_seat = 0

        hand_hero_pot_ratios: List[float] = []
        prev_at: str = ""
        last_non_hero_aggro_street: str = ""
        for a in actions:
            if not isinstance(a, dict):
                continue
            at = str(a.get("action_type", "") or "").lower()
            if at not in _ACTION_TYPES:
                continue
            try:
                seat = int(a.get("actor_seat", 0) or 0)
            except (TypeError, ValueError):
                seat = 0
            amt = float(a.get("amount", 0.0) or 0.0)
            pot_b = float(a.get("pot_before", 0.0) or 0.0)
            is_hero = seat != 0 and seat == hero_seat

            # All players bet/pot ratio
            if at in _AGGRO and pot_b > 0.01 and amt > 0:
                ratio = amt / pot_b
                all_pot_ratios.append(ratio)
                if is_hero:
                    hand_hero_pot_ratios.append(ratio)

            # Track hero facing opponent aggression
            if not is_hero and at in _AGGRO:
                last_non_hero_aggro_street = str(a.get("street", "") or "")
            if is_hero and at == "fold" and last_non_hero_aggro_street:
                hero_fold_facing_aggr += 1
                last_non_hero_aggro_street = ""
                hero_faces_aggr_count += 1
            elif is_hero and at in _PASSIVE and last_non_hero_aggro_street:
                hero_faces_aggr_count += 1
                last_non_hero_aggro_street = ""
            elif is_hero and at in _AGGRO:
                last_non_hero_aggro_street = ""

            # Action bigrams (any player, same hand)
            if prev_at and at:
                bigram_ctr[(prev_at, at)] += 1
            prev_at = at

        hero_pot_ratios_per_hand.append(
            sum(hand_hero_pot_ratios) / len(hand_hero_pot_ratios) if hand_hero_pot_ratios else 0.0
        )

    def _cv(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        if abs(m) < 1e-9:
            return 0.0
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        return s / abs(m)

    bet_pot_ratio_cv = _cv(all_pot_ratios)
    hero_bet_pot_ratio_cv = _cv([r for r in hero_pot_ratios_per_hand if r > 0])

    # Fraction of bets within ±0.08 of a standard pot fraction
    if all_pot_ratios:
        n_clustered = sum(
            1 for r in all_pot_ratios
            if any(abs(r - s) <= 0.08 for s in _STD_POT_FRACS)
        )
        bet_pot_ratio_cluster_frac = n_clustered / len(all_pot_ratios)
    else:
        bet_pot_ratio_cluster_frac = 0.0

    lag1_autocorr_hero_bet_pot = _lag1_autocorr(hero_pot_ratios_per_hand)

    hero_fold_to_bet_rate = (
        hero_fold_facing_aggr / hero_faces_aggr_count if hero_faces_aggr_count > 0 else 0.0
    )

    action_bigram_entropy = _entropy(list(bigram_ctr.values()))

    pot_frac_features = [
        bet_pot_ratio_cv,
        bet_pot_ratio_cluster_frac,
        hero_bet_pot_ratio_cv,
        lag1_autocorr_hero_bet_pot,
        hero_fold_to_bet_rate,
        action_bigram_entropy,
    ]

    # --- Intra-session consistency features (6) ---
    # Divide session into 4 equal quarters; compute mean fold/aggression/size_cv per quarter.
    # Bots maintain constant strategy (low std); humans adapt (higher std, larger drift).
    q_size = max(1, n_hands // 4)
    quarters: List[List[List[float]]] = [per_hand[i * q_size: (i + 1) * q_size] for i in range(4)]
    # Use last quarter to soak up any remainder
    remainder = per_hand[4 * q_size:]
    if remainder and quarters:
        quarters[-1] = quarters[-1] + remainder

    def _q_mean(q: List[List[float]], fi: int) -> float:
        if not q:
            return 0.0
        return sum(h[fi] for h in q) / len(q)

    fold_fi = 0     # frac_fold index in per-hand
    aggro_fi = 3    # frac_bet index (proxy for aggression)
    size_cv_fi = 7  # size_cv index

    q_fold = [_q_mean(q, fold_fi) for q in quarters]
    q_aggro = [_q_mean(q, aggro_fi) for q in quarters]
    q_size_cv = [_q_mean(q, size_cv_fi) for q in quarters]

    def _seq_std(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    intra_features = [
        _seq_std(q_fold),
        _seq_std(q_aggro),
        _seq_std(q_size_cv),
        q_fold[-1] - q_fold[0],
        q_aggro[-1] - q_aggro[0],
        q_size_cv[-1] - q_size_cv[0],
    ]

    # --- Raw amount + pot features (4) ---
    # These are chunk-level means/std of raw bet amounts and pot sizes.
    # Benchmark audit identifies these as the highest-AP individual features.
    def _smean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _sstd(vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _smean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    amount_pot_features = [
        _smean(all_amounts_nonzero),
        _sstd(all_amounts_nonzero),
        _smean(all_pot_before_vals),
        _smean(all_pot_after_vals),
    ]

    # --- Assemble final vector ---
    vec = (stat_features + chunk_sig_features + global_rate_features
           + street_features + temporal_features + collision_features
           + pot_frac_features + intra_features + amount_pot_features)

    # Guard against any non-finite leakage so downstream models stay stable.
    return [float(v) if math.isfinite(v) else 0.0 for v in vec]


assert len(FEATURE_NAMES) == len(extract_chunk_features([{}])), (
    "FEATURE_NAMES and extract_chunk_features() are out of sync"
)
