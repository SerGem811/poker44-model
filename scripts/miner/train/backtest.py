#!/usr/bin/env python3
"""Offline reward backtest for a trained Poker44 miner.

Replays a labelled dataset through the *exact* live scoring path
(model.predict_proba -> scoring head -> risk_scores) and then through the
validator's *exact* windowed reward (poker44.score.scoring.reward over random
20-sample windows, matching forward.py::_compute_windowed_rewards).

It answers the only question that matters before deploying:
  "What 20-window reward distribution will this model actually earn, and how
   often does it trip the FPR>=0.10 cliff to reward=0?"

Reports the reward distribution, the cliff-failure rate, and a t_star sweep so
you can tune --target-fpr against the real metric. With per-date data it also
prints a per-date breakdown (generalisation across benchmark days).

Usage:
    python scripts/miner/train/backtest.py --model models/poker44_gbdt.joblib \
        --data data/poker44_holdout.npz [--per-date] [--sweep]
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

import numpy as np
import joblib
from sklearn.metrics import average_precision_score

from poker44.miner_model.scoring_head import ScoringHead, shape_risk_score
from poker44.score.scoring import reward as validator_reward


def window_reward_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    window: int = 20,
    trials: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    rewards: List[float] = []
    fprs: List[float] = []
    if n < window:
        r, meta = validator_reward(scores, labels.astype(bool))
        return {
            "n": float(n), "mean": r, "p05": r, "p50": r, "p95": r,
            "cliff_zero_rate": 1.0 if meta["fpr"] >= 0.10 else 0.0,
            "mean_fpr": float(meta["fpr"]), "trials": 1.0,
        }
    for _ in range(trials):
        idx = rng.choice(n, size=window, replace=False)
        r, meta = validator_reward(scores[idx], labels[idx].astype(bool))
        rewards.append(r)
        fprs.append(meta["fpr"])
    rewards = np.asarray(rewards)
    return {
        "n": float(n),
        "mean": float(rewards.mean()),
        "p05": float(np.percentile(rewards, 5)),
        "p50": float(np.percentile(rewards, 50)),
        "p95": float(np.percentile(rewards, 95)),
        "cliff_zero_rate": float((np.asarray(fprs) >= 0.10).mean()),
        "mean_fpr": float(np.mean(fprs)),
        "trials": float(trials),
    }


def _fmt(d: Dict[str, float]) -> str:
    return (
        f"reward mean={d['mean']:.4f} p05={d['p05']:.4f} p50={d['p50']:.4f} "
        f"p95={d['p95']:.4f} | cliff_zero_rate={d['cliff_zero_rate']:.3f} "
        f"mean_fpr={d['mean_fpr']:.4f} (n={int(d['n'])})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--per-date", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="sweep the scoring-head t_star")
    args = ap.parse_args()

    blob = np.load(args.data, allow_pickle=True)
    X, y = blob["X"], blob["y"].astype(int)
    dates = blob["dates"] if "dates" in blob else None

    art = joblib.load(args.model)
    estimator = art["estimator"]
    head = ScoringHead.from_dict(art.get("scoring_head", {}))
    if "metrics" in art:
        print(f"[+] artifact train metrics: {art['metrics']}", file=sys.stderr)

    proba = estimator.predict_proba(X)[:, 1]
    scores = np.array([head.score(p) for p in proba])

    ap_score = average_precision_score(y, scores)
    overall = window_reward_distribution(scores, y, window=args.window, trials=args.trials)
    print(f"\n=== OVERALL (t_star={head.t_star:.4f}) ===")
    print(f"AP(shaped, rank-invariant) = {ap_score:.4f}")
    print(_fmt(overall))

    if args.per_date and dates is not None:
        print("\n=== PER-DATE (forward generalisation) ===")
        for d in sorted(set(dates.tolist())):
            m = dates == d
            if len(np.unique(y[m])) < 2:
                print(f"{d}: single-class, skipped")
                continue
            dist = window_reward_distribution(scores[m], y[m], window=args.window, trials=args.trials)
            print(f"{d}: AP={average_precision_score(y[m], scores[m]):.4f} | {_fmt(dist)}")

    if args.sweep:
        print("\n=== t_star SWEEP (pick the knee before cliff_zero_rate climbs) ===")
        for t in np.linspace(0.30, 0.90, 13):
            sc = np.array([shape_risk_score(p, float(t), head.sharpness) for p in proba])
            dist = window_reward_distribution(sc, y, window=args.window, trials=max(500, args.trials // 2))
            print(f"t_star={t:0.3f} -> {_fmt(dist)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
