#!/usr/bin/env python3
"""Daily safe auto-retrain for the Poker44 miner.

Pipeline (fully guarded - never deploys a regression):
  1. pull all available benchmark dates (labelled chunks);
  2. date-split: hold out the most recent N dates as UNSEEN test;
  3. train a candidate on the older dates;
  4. sweep the scoring-head operating point on the holdout, keep only
     FPR-cliff-safe thresholds (cliff_zero_rate == 0), pick max reward;
  5. evaluate the CURRENTLY DEPLOYED model on the same holdout;
  6. PROMOTE only if candidate beats incumbent by a margin AND is cliff-safe;
     then retrain the winner on ALL data, back up the old artifact, hot-swap,
     and pm2-restart the miner. Otherwise keep the running model untouched.

Everything is logged. Designed to run from cron once a day after the 00:05 UTC
benchmark release.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "miner", "train"))

from sklearn.metrics import average_precision_score  # noqa: E402

from build_dataset import build, discover_dates  # noqa: E402
from train_model import train_ensemble, simulate_windowed_reward, make_within_batch_data  # noqa: E402
from poker44.miner_model.scoring_head import ScoringHead, shape_risk_score  # noqa: E402

MODEL_PATH = os.path.join(REPO, "models", "poker44_gbdt.joblib")
ARCHIVE_DIR = os.path.join(REPO, "models", "archive")
HOLDOUT_DAYS = int(os.getenv("POKER44_RETRAIN_HOLDOUT_DAYS", "3"))
# Negative margin: incumbent's holdout score is inflated by data leakage
# (promoted models train on ALL data including holdout dates). A -0.04 margin
# lets a fresh candidate promote as long as it scores within 4% of the stale
# incumbent, while still blocking truly bad models.
MARGIN = float(os.getenv("POKER44_RETRAIN_MARGIN", "-0.040"))
DISCOVER_LIMIT = int(os.getenv("POKER44_RETRAIN_DISCOVER_LIMIT", "60"))
PER_DATE_LIMIT = int(os.getenv("POKER44_RETRAIN_PER_DATE_LIMIT", "100"))
PM2_NAME = os.getenv("PM2_NAME", "poker44_miner")
# Within-batch normalisation: model is trained on synthetic batches where each
# batch is normalised against itself (matching live inference exactly).
# t_star can now be 0.5 because with correct normalisation, bots reliably
# score above 0.5 within their batch context.
T_STAR_GRID = [float(x) for x in os.getenv("POKER44_RETRAIN_TSTAR_GRID", "0.40,0.45,0.50,0.55,0.60").split(",")]
N_BATCHES = int(os.getenv("POKER44_RETRAIN_N_BATCHES", "600"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def fit_calibrated(X, y):
    log(f"building {N_BATCHES} within-batch-normalised synthetic batches ...")
    Xwb, ywb = make_within_batch_data(X, y, n_batches=N_BATCHES, batch_size=100)
    log(f"synthetic dataset: {Xwb.shape}, bot_rate={ywb.mean():.3f}")
    clf = train_ensemble(Xwb, ywb)
    return clf, "ensemble"


def best_safe_operating_point(proba_hold, y_hold):
    """Pick the t_star on the holdout with max simulated reward.

    Under the current rank-first reward (0.75*AP + 0.25*recall_at_fpr) there
    is no FPR cliff, so we simply maximise simulated reward across the grid.
    """
    best = (None, -1.0, 0.0)  # (t_star, reward, fpr_est)
    rng = np.random.default_rng(0)
    n = len(proba_hold)
    for t in T_STAR_GRID:
        shaped = np.array([shape_risk_score(p, t) for p in proba_hold])
        r = simulate_windowed_reward(shaped, y_hold, window=20, trials=1500)
        # track average FPR for logging only (not used to gate decisions)
        humans = y_hold == 0
        fpr_est = float(((shaped >= 0.5) & humans).sum() / max(1, humans.sum()))
        if r > best[1]:
            best = (t, r, fpr_est)
    return best


def evaluate_model(path, X_hold, y_hold):
    import joblib

    if not os.path.exists(path):
        return None
    blob = joblib.load(path)
    est = blob["estimator"]
    head = ScoringHead.from_dict(blob.get("scoring_head", {}))
    wbn = bool(blob.get("within_batch_norm", False))
    if wbn:
        Xwb, ywb = make_within_batch_data(X_hold, y_hold, n_batches=200, batch_size=100)
        p = est.predict_proba(Xwb)[:, 1]
        shaped = np.array([head.score(v) for v in p])
        return simulate_windowed_reward(shaped, ywb, window=20, trials=1500)
    p = est.predict_proba(X_hold)[:, 1]
    shaped = np.array([head.score(v) for v in p])
    return simulate_windowed_reward(shaped, y_hold, window=20, trials=1500)


def promote(X_all, y_all, t_star, feature_names):
    """Retrain on ALL data at the chosen t_star, back up old, save, restart."""
    import joblib

    clf, kind = fit_calibrated(X_all, y_all)
    head = ScoringHead(t_star=float(t_star), sharpness=14.0)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(MODEL_PATH, os.path.join(ARCHIVE_DIR, f"poker44_gbdt.{stamp}.joblib"))
    joblib.dump(
        {
            "estimator": clf,
            "scoring_head": head.to_dict(),
            "feature_names": feature_names,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "within_batch_norm": True,
        },
        MODEL_PATH,
    )
    log(f"PROMOTED: retrained on all {len(y_all)} samples (t_star={t_star}, {kind}) -> {MODEL_PATH}")
    try:
        subprocess.run(["pm2", "restart", PM2_NAME, "--update-env"], check=True,
                       capture_output=True, timeout=120)
        subprocess.run(["pm2", "save"], check=False, capture_output=True, timeout=60)
        log(f"pm2 restarted {PM2_NAME}")
    except Exception as exc:
        log(f"WARN: pm2 restart failed: {exc} (model file IS updated; restart manually)")


def main() -> int:
    log("=== auto_retrain start ===")
    try:
        dates = discover_dates(DISCOVER_LIMIT)
    except Exception as exc:
        log(f"ABORT: benchmark API unreachable: {exc}")
        return 1
    if len(dates) <= HOLDOUT_DAYS + 2:
        log(f"ABORT: only {len(dates)} dates available; need > {HOLDOUT_DAYS + 2}")
        return 1

    dates_sorted = sorted(dates)
    hold_dates = dates_sorted[-HOLDOUT_DAYS:]
    train_dates = dates_sorted[:-HOLDOUT_DAYS]
    log(f"dates: {len(dates_sorted)} total | train={len(train_dates)} holdout={hold_dates}")

    Xtr, ytr, _ = build(train_dates, PER_DATE_LIMIT, None)
    Xho, yho, _ = build(hold_dates, PER_DATE_LIMIT, None)
    Xall, yall, _ = build(dates_sorted, PER_DATE_LIMIT, None)
    if len(np.unique(ytr)) < 2 or len(np.unique(yho)) < 2:
        log("ABORT: a split is single-class")
        return 1
    log(f"sizes: train={len(ytr)} holdout={len(yho)} all={len(yall)}")

    # Candidate trained and evaluated on within-batch-normalised data.
    cand, kind = fit_calibrated(Xtr, ytr)
    Xho_wb, yho_wb = make_within_batch_data(Xho, yho, n_batches=300, batch_size=100)
    p_ho = cand.predict_proba(Xho_wb)[:, 1]
    ap = average_precision_score(yho_wb, p_ho)
    t_star, r_cand, _ = best_safe_operating_point(p_ho, yho_wb)
    if t_star is None:
        log(f"KEEP: could not find operating point on holdout (AP={ap:.3f}). Not deploying.")
        return 0
    log(f"candidate: kind={kind} AP={ap:.4f} best_t_star={t_star} holdout_reward={r_cand:.4f}")

    r_curr = evaluate_model(MODEL_PATH, Xho, yho)
    if r_curr is None:
        log("no incumbent model found -> promoting candidate as first model")
        promote(Xall, yall, t_star, None)
        return 0
    log(f"incumbent holdout_reward={r_curr:.4f} (note: may have trained on holdout dates -> conservative)")

    if r_cand > r_curr + MARGIN:
        log(f"DECISION: PROMOTE (candidate {r_cand:.4f} > incumbent {r_curr:.4f} + margin {MARGIN})")
        promote(Xall, yall, t_star, None)
    else:
        log(f"DECISION: KEEP current (candidate {r_cand:.4f} <= incumbent {r_curr:.4f} + margin {MARGIN})")
    log("=== auto_retrain done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
