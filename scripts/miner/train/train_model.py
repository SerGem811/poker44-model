#!/usr/bin/env python3
"""Train the Poker44 bot-detector: 5-way ensemble + BlendedIsotonicCalibrator.

Pipeline:
  1. Load X, y from data/train.npz and data/hold.npz (or a single --data file
     split internally when --hold is not provided).
  2. Train five base classifiers on the training split:
       lgbm1, lgbm2, lgbm3, ExtraTrees, RandomForest
  3. Average their bot-class probabilities on the validation set, then fit a
     BlendedIsotonicCalibrator (isotonic blended 50/50 with raw score) to
     de-saturate while preserving ranking.
  4. Wrap everything in an EnsembleClassifier that is callable via
       .predict_proba(X)[:, 1]
     and serialisable with joblib.
  5. Evaluate on the holdout: AP, val_fpr, val_recall.
  6. Sweep the scoring-head threshold t_star via shape_risk_score /
     simulate_windowed_reward (rank-first reward formula 0.75*AP + 0.25*recall).
  7. Save the bundle dict with keys: estimator, scoring_head, feature_names,
     metrics.

Usage:
    python scripts/miner/train/train_model.py \
        --train data/train.npz --hold data/hold.npz \
        --out models/poker44_gbdt.joblib

    # Single-file mode (internal stratified split):
    python scripts/miner/train/train_model.py \
        --data data/poker44_train.npz --out models/poker44_gbdt.joblib

    # Override threshold after manual backtest:
    python scripts/miner/train/train_model.py \
        --train data/train.npz --hold data/hold.npz \
        --out models/poker44_gbdt.joblib --fixed-t-star 0.91
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Optional heavy imports — fail loudly if missing so the user knows what to pip
# ---------------------------------------------------------------------------
try:
    from lightgbm import LGBMClassifier
except ImportError as _exc:  # noqa: F841
    raise ImportError(
        "lightgbm is required: pip install lightgbm"
    ) from _exc

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import joblib

# Poker44 repo may not be on sys.path when called directly; fix it.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      os.pardir, os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from poker44.miner_model.scoring_head import (  # noqa: E402
    ScoringHead,
    shape_risk_score,
    select_operating_point,
)
from poker44.score.scoring import reward as validator_reward  # noqa: E402
from poker44.miner_model.ensemble import (  # noqa: E402
    BlendedIsotonicCalibrator,
    EnsembleClassifier,
)


# BlendedIsotonicCalibrator and EnsembleClassifier live in the package so
# joblib can unpickle them without scripts/ on sys.path.
# Imported above: from poker44.miner_model.ensemble import ...


# ---------------------------------------------------------------------------
# Helper: build base model list
# ---------------------------------------------------------------------------

def _make_base_models() -> list:
    return [
        ("lgbm1", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
        )),
        ("lgbm2", LGBMClassifier(
            n_estimators=300, learning_rate=0.02, num_leaves=15,
            max_depth=4, reg_lambda=5.0, reg_alpha=2.0,
            random_state=123, n_jobs=-1, verbose=-1,
        )),
        ("lgbm3", LGBMClassifier(
            n_estimators=500, learning_rate=0.02, num_leaves=63,
            reg_lambda=2.0, random_state=7, n_jobs=-1, verbose=-1,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=5,
            max_features=0.5, random_state=42, n_jobs=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5,
            max_features=0.5, random_state=42, n_jobs=-1,
        )),
    ]


# ---------------------------------------------------------------------------
# Helper: kept for auto_retrain.py compatibility
# ---------------------------------------------------------------------------

def _make_estimator():
    """Backward-compat shim used by auto_retrain.py.

    Returns a single LGBMClassifier (first base model) for the legacy
    fit_calibrated() path in auto_retrain.  auto_retrain will be updated to
    call train_ensemble() directly; this shim avoids a hard break in the
    meantime.
    """
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
    ), "lightgbm"


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    blend: float = 0.5,
) -> EnsembleClassifier:
    """Fit all 5 base models then calibrate on the validation set.

    When X_val/y_val are omitted an internal 80/20 stratified split is used
    so callers (e.g. auto_retrain.py) can pass just (X, y).
    """
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=0
        )
    named_models = _make_base_models()
    fitted: list = []

    for name, clf in named_models:
        print(f"[+] training {name} ...", flush=True)
        clf.fit(X_train, y_train)
        fitted.append((name, clf))
        # Quick sanity AP on training predictions (not calibrated yet)
        p_tr = clf.predict_proba(X_train)[:, 1]
        ap_tr = average_precision_score(y_train, p_tr)
        print(f"    {name}: train AP = {ap_tr:.4f}", flush=True)

    print("[+] computing averaged validation probabilities ...", flush=True)
    proba_sum = None
    for name, clf in fitted:
        p = clf.predict_proba(X_val)[:, 1]
        if proba_sum is None:
            proba_sum = p.copy()
        else:
            proba_sum += p
    avg_val = proba_sum / len(fitted)

    ap_pre = average_precision_score(y_val, avg_val)
    print(f"[=] ensemble (pre-calibration) val AP = {ap_pre:.4f}", flush=True)

    print("[+] fitting BlendedIsotonicCalibrator (blend=0.5) ...", flush=True)
    calibrator = BlendedIsotonicCalibrator(blend=blend)
    calibrator.fit(avg_val, y_val)

    # AP after calibration (should be near-identical — isotonic is monotone)
    avg_cal = calibrator.transform(avg_val)
    ap_post = average_precision_score(y_val, avg_cal)
    print(f"[=] ensemble (post-calibration) val AP = {ap_post:.4f}", flush=True)

    ensemble = EnsembleClassifier(base_models=fitted, calibrator=calibrator)
    return ensemble


# ---------------------------------------------------------------------------
# Reward simulation (unchanged from original; reused by auto_retrain.py)
# ---------------------------------------------------------------------------

def simulate_windowed_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    window: int = 20,
    trials: int = 400,
) -> float:
    """Average validator reward over random 20-windows (mirrors forward.py)."""
    rng = np.random.default_rng(0)
    n = len(scores)
    if n < window:
        return float(validator_reward(scores, labels)[0])
    vals = []
    for _ in range(trials):
        idx = rng.choice(n, size=window, replace=False)
        vals.append(
            validator_reward(scores[idx], labels[idx].astype(bool))[0]
        )
    return float(np.mean(vals))


def best_safe_operating_point(
    proba_hold: np.ndarray,
    y_hold: np.ndarray,
    t_star_grid: list[float] | None = None,
) -> tuple[float | None, float, float]:
    """Pick t_star with zero cliff failures and max simulated reward.

    Returns (t_star, reward, cliff_rate).  t_star is None if nothing safe.
    """
    if t_star_grid is None:
        t_star_grid = [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95]

    best: tuple[float | None, float, float] = (None, -1.0, 1.0)
    rng = np.random.default_rng(0)
    n = len(proba_hold)

    for t in t_star_grid:
        shaped = np.array([shape_risk_score(p, t) for p in proba_hold])
        r = simulate_windowed_reward(shaped, y_hold, window=20, trials=1500)

        # Estimate cliff-breach rate (FPR >= 0.10 in a random 20-window)
        cliff = 0.0
        if n >= 20:
            fprs = []
            for _ in range(1500):
                idx = rng.choice(n, 20, replace=False)
                hh = y_hold[idx] == 0
                fp = ((shaped[idx] >= 0.5) & hh).sum()
                fprs.append(fp / max(1, hh.sum()))
            cliff = float((np.array(fprs) >= 0.10).mean())

        if cliff == 0.0 and r > best[1]:
            best = (t, r, cliff)

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Poker44 bot-detector: 5-way ensemble + BlendedIsotonicCalibrator"
    )
    # Two-file mode (preferred)
    parser.add_argument("--train", default=None,
                        help="Path to training .npz  (X, y, feature_names)")
    parser.add_argument("--hold", default=None,
                        help="Path to holdout .npz   (X, y, feature_names)")
    # Single-file fallback (backward compat with old --data flag)
    parser.add_argument("--data", default=None,
                        help="Single .npz; will be split internally (--test-size)")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Fraction for internal val split when using --data")
    # Output
    parser.add_argument("--out", default="models/poker44_gbdt.joblib")
    # Scoring head
    parser.add_argument("--fixed-t-star", type=float, default=None,
                        help="Override auto-selected t_star (sweep-validated constant)")
    parser.add_argument("--target-fpr", type=float, default=0.02,
                        help="Target FPR for select_operating_point fallback")
    # Calibrator
    parser.add_argument("--blend", type=float, default=0.5,
                        help="BlendedIsotonicCalibrator blend fraction (default 0.5)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.train is not None and args.hold is not None:
        # Two-file mode
        blob_tr = np.load(args.train, allow_pickle=True)
        blob_ho = np.load(args.hold, allow_pickle=True)
        X_train, y_train = blob_tr["X"], blob_tr["y"].astype(int)
        X_val,   y_val   = blob_ho["X"], blob_ho["y"].astype(int)
        feature_names = (
            list(blob_tr["feature_names"]) if "feature_names" in blob_tr
            else list(blob_ho["feature_names"]) if "feature_names" in blob_ho
            else None
        )
        print(f"[+] train X{X_train.shape} y{y_train.shape} "
              f"bot_rate={y_train.mean():.3f}", flush=True)
        print(f"[+] hold  X{X_val.shape}   y{y_val.shape}   "
              f"bot_rate={y_val.mean():.3f}", flush=True)
    elif args.data is not None:
        # Single-file mode
        blob = np.load(args.data, allow_pickle=True)
        X, y = blob["X"], blob["y"].astype(int)
        feature_names = list(blob["feature_names"]) if "feature_names" in blob else None
        print(f"[+] data X{X.shape} y{y.shape} bot_rate={y.mean():.3f}", flush=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=0
        )
        print(f"[+] internal split: train={len(y_train)} val={len(y_val)}", flush=True)
    else:
        # Try default paths
        default_train = "data/train.npz"
        default_hold  = "data/hold.npz"
        if os.path.exists(default_train) and os.path.exists(default_hold):
            blob_tr = np.load(default_train, allow_pickle=True)
            blob_ho = np.load(default_hold,  allow_pickle=True)
            X_train, y_train = blob_tr["X"], blob_tr["y"].astype(int)
            X_val,   y_val   = blob_ho["X"], blob_ho["y"].astype(int)
            feature_names = (
                list(blob_tr["feature_names"]) if "feature_names" in blob_tr else None
            )
            print(f"[+] train X{X_train.shape} (default path)", flush=True)
            print(f"[+] hold  X{X_val.shape}   (default path)", flush=True)
        else:
            print(
                "[!] Provide --train/--hold or --data (or place data/train.npz + data/hold.npz)",
                file=sys.stderr,
            )
            return 1

    # Sanity checks
    for split_name, yy in [("train", y_train), ("val/hold", y_val)]:
        if len(np.unique(yy)) < 2:
            print(f"[!] {split_name} set has only one class; cannot train.", file=sys.stderr)
            return 1

    print(f"[+] feature count: {X_train.shape[1]}", flush=True)

    # ------------------------------------------------------------------
    # 2. Train ensemble
    # ------------------------------------------------------------------
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, blend=args.blend)

    # ------------------------------------------------------------------
    # 3. Evaluate on hold/val set
    # ------------------------------------------------------------------
    p_val = ensemble.predict_proba(X_val)[:, 1]
    ap_score = average_precision_score(y_val, p_val)
    print(f"[=] holdout AP (calibrated ensemble) = {ap_score:.4f}", flush=True)

    # FPR and recall at 0.5 cutoff on raw calibrated scores
    val_fpr    = float(((p_val >= 0.5) & (y_val == 0)).sum() / max(1, (y_val == 0).sum()))
    val_recall = float(((p_val >= 0.5) & (y_val == 1)).sum() / max(1, (y_val == 1).sum()))
    print(f"[=] at 0.5 cutoff: FPR={val_fpr:.4f} recall={val_recall:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Scoring head t_star
    # ------------------------------------------------------------------
    if args.fixed_t_star is not None:
        t_star = float(args.fixed_t_star)
        print(f"[+] using fixed t_star={t_star:.4f} (sweep-validated override)", flush=True)
    else:
        print("[+] sweeping t_star grid ...", flush=True)
        t_star_best, r_best, cliff = best_safe_operating_point(p_val, y_val)

        if t_star_best is not None:
            t_star = t_star_best
            print(f"[=] auto t_star={t_star:.4f} "
                  f"(simulated reward={r_best:.4f} cliff_rate={cliff:.4f})",
                  flush=True)
        else:
            # Fallback: pick from human/bot score distributions
            print("[!] no cliff-free t_star found in grid; falling back to "
                  "select_operating_point()", flush=True)
            t_star = select_operating_point(
                human_scores=p_val[y_val == 0],
                bot_scores=p_val[y_val == 1],
                target_fpr=args.target_fpr,
            )
            print(f"[=] fallback t_star={t_star:.4f}", flush=True)

    head = ScoringHead(t_star=float(t_star), sharpness=14.0)

    # Apply scoring head and compute simulated reward
    shaped = np.array([head.score(p) for p in p_val])
    sim_reward = simulate_windowed_reward(shaped, y_val, window=20, trials=400)

    ap_shaped = average_precision_score(y_val, shaped)
    print(
        f"[=] t_star={t_star:.4f} | val FPR={val_fpr:.4f} recall={val_recall:.4f} "
        f"| AP(shaped)={ap_shaped:.4f} "
        f"| simulated 20-window reward={sim_reward:.4f}",
        flush=True,
    )

    if val_fpr >= 0.10:
        print("[!] WARNING: validation FPR is >= 0.10; consider a higher t_star "
              "or --target-fpr.", file=sys.stderr)

    # ------------------------------------------------------------------
    # 5. Save bundle
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bundle = {
        "estimator": ensemble,          # EnsembleClassifier with calibrator embedded
        "scoring_head": head.to_dict(), # {"t_star": float, "sharpness": float}
        "feature_names": feature_names,
        "metrics": {
            "ap": float(ap_score),
            "val_fpr": val_fpr,
            "val_recall": val_recall,
            "sim_reward": sim_reward,
        },
    }
    joblib.dump(bundle, args.out)
    print(f"[=] bundle saved to {args.out}", flush=True)
    print(f"[=] bundle keys: {list(bundle.keys())}", flush=True)
    print(f"[=] metrics: {bundle['metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
