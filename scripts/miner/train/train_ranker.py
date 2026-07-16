#!/usr/bin/env python3
"""Train a LambdaMART bot-detection RANKER (our own implementation).

Why a ranker: the competition reward is rank-based (0.35*AP + 0.30*recall@FPR5%
+ ...). A LambdaMART ranker optimises within-group ranking directly, which is
exactly what is scored — unlike a binary classifier that optimises log-loss.

Two improvements over our previous classifier models:
  1. Objective = LightGBM `lambdarank` over within-batch groups. Each synthetic
     100-session batch is one ranking group; the model learns to rank bots above
     humans *inside a batch*, matching how the validator scores each batch.
  2. Live-sized chunks: benchmark sessions are ~40 hands but the live validator
     sends ~86-hand chunks (measured). We concatenate same-label benchmark
     sessions into ~80-hand chunks so the per-chunk feature statistics the model
     learns on match what it sees live.

Features are our own 317-vector (poker44.miner_model.features). Trained on the
public benchmark only (via the vendor/d0 raw cache or a fresh fetch).

Saves a bundle at models/poker44_ranker.joblib:
  {estimator, kind="ranker", within_batch_norm, feature_names, rank_calibration}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from poker44.miner_model.features import FEATURE_NAMES, extract_chunk_features  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

D0_CACHE = os.path.join(REPO, "vendor", "d0", "data_cache")
OUT = os.path.join(REPO, "models", "poker44_ranker.joblib")


def load_raw_groups():
    """Load (group_hands, label, date) from the vendor/d0 raw cache."""
    rows = []
    for path in sorted(glob.glob(os.path.join(D0_CACHE, "*.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        data = doc["data"]
        date = data["sourceDate"]
        for rec in data["chunks"]:
            groups = rec["chunks"]
            labels = rec["groundTruth"]
            for g, lab in zip(groups, labels):
                rows.append((g, int(lab), date))
    return rows


def build_livesize_chunks(rows, target_hands=80, rng=None):
    """Concatenate same-label sessions into ~target_hands chunks (match live ~86)."""
    rng = rng or np.random.default_rng(0)
    by_label = {0: [], 1: []}
    for g, lab, _ in rows:
        by_label[lab].append(g)
    X, y = [], []
    for lab in (0, 1):
        groups = by_label[lab]
        rng.shuffle(groups)
        buf, nhands = [], 0
        for g in groups:
            buf.extend(g)
            nhands += len(g)
            if nhands >= target_hands:
                canon = [prepare_hand_for_miner(h) for h in buf]
                X.append(extract_chunk_features(canon))
                y.append(lab)
                buf, nhands = [], 0
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def make_wbn_groups(X, y, n_batches=600, batch_size=100, bmin=0.30, bmax=0.70, seed=42):
    """Synthetic within-batch-normalised ranking groups. Returns X, y, group_sizes."""
    rng = np.random.default_rng(seed)
    bots = np.where(y == 1)[0]
    hums = np.where(y == 0)[0]
    Xo, yo, groups = [], [], []
    for _ in range(n_batches):
        nb = int(rng.integers(max(1, int(batch_size * bmin)),
                              min(batch_size - 1, int(batch_size * bmax)) + 1))
        nh = batch_size - nb
        idx = np.concatenate([rng.choice(bots, nb, replace=True),
                              rng.choice(hums, nh, replace=True)])
        Xb = X[idx].astype(float)
        m, s = Xb.mean(0), Xb.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        Xb = np.clip((Xb - m) / s, -5.0, 5.0)
        Xo.append(Xb)
        yo.append(y[idx])
        groups.append(batch_size)
    return np.vstack(Xo), np.concatenate(yo), np.array(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hands", type=int, default=80)
    ap.add_argument("--n-batches", type=int, default=700)
    ap.add_argument("--holdout-dates", type=int, default=4)
    args = ap.parse_args()

    rows = load_raw_groups()
    dates = sorted({d for _, _, d in rows})
    hold = set(dates[-args.holdout_dates:])
    tr_rows = [r for r in rows if r[2] not in hold]
    ho_rows = [r for r in rows if r[2] in hold]
    print(f"[+] {len(rows)} sessions | {len(dates)} dates | train {len(tr_rows)} holdout {len(ho_rows)}",
          flush=True)

    Xtr, ytr = build_livesize_chunks(tr_rows, args.target_hands)
    Xho, yho = build_livesize_chunks(ho_rows, args.target_hands)
    print(f"[+] live-sized chunks: train {Xtr.shape} holdout {Xho.shape} "
          f"(bot_rate {ytr.mean():.2f})", flush=True)

    Xg, yg, groups = make_wbn_groups(Xtr, ytr, n_batches=args.n_batches)
    print(f"[+] WBN ranking groups: {Xg.shape} in {len(groups)} groups", flush=True)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, label_gain=[0, 1], random_state=0, n_jobs=-1, verbose=-1,
    )
    ranker.fit(Xg, yg, group=groups)
    print("[+] ranker trained", flush=True)

    # Honest holdout: WBN the holdout as one batch and measure ranking AP.
    def wbn(X):
        m, s = X.mean(0), X.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        return np.clip((X - m) / s, -5.0, 5.0)

    if len(np.unique(yho)) == 2:
        raw = ranker.predict(wbn(Xho))
        ap = average_precision_score(yho, raw)
        print(f"[=] HOLDOUT ranking AP (live-sized, WBN) = {ap:.4f}", flush=True)

    import joblib
    from datetime import datetime, timezone
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump({
        "estimator": ranker,
        "kind": "ranker",
        "within_batch_norm": True,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rank_calibration": {"mode": "rank_gate", "bot_frac": 0.15,
                             "bot_low": 0.55, "bot_hi": 0.92,
                             "hum_low": 0.02, "hum_hi": 0.48},
    }, OUT)
    print(f"[=] saved ranker -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
