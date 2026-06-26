"""Score-shaping head that maximises Poker44 reward under the FPR cliff.

Validator reward (poker44/score/scoring.py):

    reward = (0.65 * AP + 0.35 * bot_recall) * (1 - FPR) ** 2,  zeroed if FPR >= 0.10

scored over a 20-sample window on the *raw* risk_scores (predictions bool is
ignored - the validator rounds risk_scores itself at 0.5).

Two structural facts:
  * AP (average_precision_score) is invariant under any strictly-monotonic
    transform of the scores - it only depends on ranking.
  * FPR / recall use a fixed 0.5 cutoff on those same scores. In a 20-window
    (~10 humans) a single human above 0.5 means FPR = 0.10 -> reward = 0.

So the optimal head takes a well-ordered probability ``p`` (good AP) and applies
a strictly-monotonic map that places the 0.5 crossover exactly at a calibrated
high-precision operating point ``t_star`` (where held-out human FPR ~= 0). AP is
untouched; only the location of the binary cutoff moves, protecting (1-FPR)**2.

``t_star`` is chosen at training time from the validation human-score
distribution with margin for the small-window variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


def shape_risk_score(p: float, t_star: float, sharpness: float = 14.0) -> float:
    """Strictly-monotonic map sending p==t_star -> 0.5, preserving ranking.

    A tiny linear term keeps the map strictly increasing (no saturation ties),
    so average-precision is preserved exactly.
    """
    p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else float(p)
    t_star = 1e-6 if t_star <= 0 else 1.0 - 1e-6 if t_star >= 1 else float(t_star)
    core = 1.0 / (1.0 + math.exp(-sharpness * (p - t_star)))
    shaped = 0.998 * core + 0.002 * p  # break ties at the rails, keep monotonic
    return 0.0 if shaped < 0.0 else 1.0 if shaped > 1.0 else shaped


def select_operating_point(
    human_scores: Sequence[float],
    bot_scores: Sequence[float],
    *,
    target_fpr: float = 0.02,
    window: int = 20,
) -> float:
    """Pick t_star maximising expected windowed reward contribution.

    The validator window is small (default 20) and the FPR cliff at 0.10 is
    catastrophic, so we keep the *validation* human FPR well below it
    (``target_fpr``) and, among safe thresholds, take the one that recalls the
    most bots. Falls back to the max human score (+margin) if nothing is safe.
    """
    humans = sorted(float(s) for s in human_scores)
    bots = [float(s) for s in bot_scores]
    if not humans:
        return 0.5
    if not bots:
        return min(1.0, humans[-1] + 1e-3)

    # Candidate thresholds: every distinct human score plus a hair above the max.
    candidates = sorted(set(humans + [humans[-1] + 1e-3, 1.0]))
    best_t = min(1.0, humans[-1] + 1e-3)
    best_reward = -1.0
    n_h = len(humans)
    for t in candidates:
        fpr = sum(1 for s in humans if s >= t) / n_h
        if fpr > target_fpr:
            continue
        recall = sum(1 for s in bots if s >= t) / len(bots)
        penalty = (1.0 - fpr) ** 2
        reward = (0.35 * recall) * penalty  # AP term is t-independent
        if reward > best_reward:
            best_reward = reward
            best_t = t
    return float(best_t)


@dataclass
class ScoringHead:
    """Maps a model's bot-probability to the final risk_score for the validator."""

    t_star: float = 0.5
    sharpness: float = 14.0

    def score(self, p: float) -> float:
        return shape_risk_score(p, self.t_star, self.sharpness)

    def score_many(self, ps: Sequence[float]) -> List[float]:
        return [round(self.score(p), 6) for p in ps]

    def to_dict(self) -> dict:
        return {"t_star": self.t_star, "sharpness": self.sharpness}

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringHead":
        return cls(
            t_star=float(d.get("t_star", 0.5)),
            sharpness=float(d.get("sharpness", 14.0)),
        )
