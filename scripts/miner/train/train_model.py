#!/usr/bin/env python3
"""Train the Poker44 bot-detector and fit the FPR-safe scoring head.

Pipeline:
  1. load X,y from build_dataset.py;
  2. group-agnostic stratified split into train/validation;
  3. fit a gradient-boosted tree (LightGBM if available, else sklearn HGB);
  4. isotonic-calibrate probabilities for well-ordered, spread-out scores (AP);
  5. choose the scoring-head operating point t_star on validation humans so the
     0.5 cutoff sits where human FPR is ~0 (protects the (1-FPR)**2 cliff);
  6. report the simulated validator reward and persist the artifact.

Usage:
    python scripts/miner/train/train_model.py \
        --data data/poker44_train.npz --out models/poker44_gbdt.joblib
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from poker44.miner_model.scoring_head import ScoringHead, select_operating_point
from poker44.score.scoring import reward as validator_reward


def _make_estimator():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            reg_lambda=1.0,
            n_jobs=-1,
        ), "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            max_leaf_nodes=31, early_stopping=True,
        ), "sklearn-hgb"


def simulate_windowed_reward(scores: np.ndarray, labels: np.ndarray, window: int = 20, trials: int = 400) -> float:
    """Average validator reward over random 20-windows (matches forward.py)."""
    rng = np.random.default_rng(0)
    n = len(scores)
    if n < window:
        return float(validator_reward(scores, labels)[0])
    vals = []
    for _ in range(trials):
        idx = rng.choice(n, size=window, replace=False)
        vals.append(validator_reward(scores[idx], labels[idx].astype(bool))[0])
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument(
        "--fixed-t-star",
        type=float,
        default=None,
        help="Override the auto-selected operating point (use a value validated "
        "out-of-sample via backtest.py --sweep). Auto-selection on small "
        "validation splits is unstable; a sweep-validated constant is safer.",
    )
    ap.add_argument(
        "--holdout-latest-date",
        action="store_true",
        help="validate on the most recent source date instead of a random split "
        "(measures true forward generalisation, matching live deployment)",
    )
    args = ap.parse_args()

    blob = np.load(args.data, allow_pickle=True)
    X, y = blob["X"], blob["y"].astype(int)
    feature_names = list(blob["feature_names"]) if "feature_names" in blob else None
    dates = blob["dates"] if "dates" in blob else None
    print(f"[+] data X{X.shape} y{y.shape} bot_rate={y.mean():.3f}", file=sys.stderr)
    if len(np.unique(y)) < 2:
        print("Need both classes present.", file=sys.stderr)
        return 1

    if args.holdout_latest_date and dates is not None and len(set(dates.tolist())) > 1:
        latest = sorted(set(dates.tolist()))[-1]
        val_mask = dates == latest
        Xtr, Xval, ytr, yval = X[~val_mask], X[val_mask], y[~val_mask], y[val_mask]
        print(f"[+] date holdout: validating on latest date={latest} "
              f"(train={len(ytr)} val={len(yval)})", file=sys.stderr)
        if len(np.unique(yval)) < 2:
            print("[!] holdout date has one class; falling back to random split.", file=sys.stderr)
            Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=0)
    else:
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=0
        )
    base, kind = _make_estimator()
    print(f"[+] estimator={kind}", file=sys.stderr)
    # Isotonic calibration -> monotone, well-spread probabilities = strong AP.
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr)

    pval = clf.predict_proba(Xval)[:, 1]
    ap_score = average_precision_score(yval, pval)
    print(f"[=] validation AP (pre-shape, rank-invariant) = {ap_score:.4f}", file=sys.stderr)

    if args.fixed_t_star is not None:
        t_star = float(args.fixed_t_star)
        print(f"[+] using fixed t_star={t_star:.4f} (sweep-validated override)", file=sys.stderr)
    else:
        t_star = select_operating_point(
            human_scores=pval[yval == 0],
            bot_scores=pval[yval == 1],
            target_fpr=args.target_fpr,
        )
    head = ScoringHead(t_star=float(t_star), sharpness=14.0)
    shaped = np.array([head.score(p) for p in pval])

    val_fpr = float(((shaped >= 0.5) & (yval == 0)).sum() / max(1, (yval == 0).sum()))
    val_recall = float(((shaped >= 0.5) & (yval == 1)).sum() / max(1, (yval == 1).sum()))
    sim_reward = simulate_windowed_reward(shaped, yval)
    print(
        f"[=] t_star={t_star:.4f} | val FPR={val_fpr:.4f} recall={val_recall:.4f} "
        f"| AP(shaped)={average_precision_score(yval, shaped):.4f} "
        f"| simulated 20-window reward={sim_reward:.4f}",
        file=sys.stderr,
    )
    if val_fpr >= 0.10:
        print("[!] WARNING: validation FPR breaches the 0.10 cliff; lower --target-fpr.", file=sys.stderr)

    import joblib

    joblib.dump(
        {
            "estimator": clf,
            "scoring_head": head.to_dict(),
            "feature_names": feature_names,
            "metrics": {
                "ap": float(ap_score),
                "val_fpr": val_fpr,
                "val_recall": val_recall,
                "sim_reward": sim_reward,
            },
        },
        args.out,
    )
    print(f"[=] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
