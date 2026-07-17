#!/usr/bin/env python3
"""Train our LambdaMART ranker on benchmark bots + REAL human hands.

The benchmark's 'human' sessions are synthetic and unrepresentative: measured
mean_aggression_factor ~0.78 vs ~1.81 for real online humans. A model trained on
benchmark humans therefore treats normal aggressive real play as bot-like and
mis-ranks real competition humans (AP collapse live). Adding a large corpus of
REAL human hands (public micro-stakes hand histories) as the negative class
teaches the true human distribution.

Positive class (bots): public Poker44 benchmark synthetic bots.
Negative class (humans): benchmark humans + real human hands grouped to live size.

Everything else matches train_ranker.py (WBN ranking groups, live-sized chunks,
rank-gate calibration). Saves models/poker44_ranker_aug.joblib.
"""

from __future__ import annotations

import argparse
import glob
import gzip
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
REAL_HUMANS = os.path.join(REPO, "data", "real_humans", "poker_hands_combined.json.gz")
OUT = os.path.join(REPO, "models", "poker44_ranker_aug.joblib")


def _feat(group):
    return extract_chunk_features([prepare_hand_for_miner(h) for h in group])


def load_benchmark():
    """Return (bot_chunks_raw, human_chunks_raw) as lists of hand-groups."""
    bots, humans = [], []
    for path in sorted(glob.glob(os.path.join(D0_CACHE, "*.json"))):
        data = json.load(open(path))["data"]
        for rec in data["chunks"]:
            for g, lab in zip(rec["chunks"], rec["groundTruth"]):
                (bots if int(lab) == 1 else humans).append(g)
    return bots, humans


def livesize(groups, target=80, rng=None):
    """Concatenate consecutive groups into ~target-hand chunks -> feature rows."""
    rng = rng or np.random.default_rng(0)
    groups = list(groups)
    rng.shuffle(groups)
    X, buf, nh = [], [], 0
    for g in groups:
        buf.extend(g)
        nh += len(g)
        if nh >= target:
            X.append(_feat(buf))
            buf, nh = [], 0
    return X


def real_human_chunks(target=80, rng=None):
    rng = rng or np.random.default_rng(1)
    hands = json.load(gzip.open(REAL_HUMANS))
    idx = np.arange(len(hands))
    rng.shuffle(idx)
    X, buf, nh = [], [], 0
    for i in idx:
        buf.append(hands[int(i)])
        nh += 1
        if nh >= target:
            X.append(_feat(buf))
            buf, nh = [], 0
    return X


def make_wbn_groups(X, y, n_batches=800, batch_size=100, bmin=0.30, bmax=0.70, seed=42):
    rng = np.random.default_rng(seed)
    bots = np.where(y == 1)[0]
    hums = np.where(y == 0)[0]
    Xo, yo, groups = [], [], []
    for _ in range(n_batches):
        nb = int(rng.integers(max(1, int(batch_size * bmin)),
                              min(batch_size - 1, int(batch_size * bmax)) + 1))
        idx = np.concatenate([rng.choice(bots, nb, replace=True),
                              rng.choice(hums, batch_size - nb, replace=True)])
        Xb = X[idx].astype(float)
        m, s = Xb.mean(0), Xb.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        Xo.append(np.clip((Xb - m) / s, -5.0, 5.0))
        yo.append(y[idx])
        groups.append(batch_size)
    return np.vstack(Xo), np.concatenate(yo), np.array(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hands", type=int, default=80)
    ap.add_argument("--n-batches", type=int, default=800)
    args = ap.parse_args()

    bot_g, hum_g = load_benchmark()
    print(f"[+] benchmark: {len(bot_g)} bot sessions, {len(hum_g)} human sessions", flush=True)

    Xbot = livesize(bot_g, args.target_hands, np.random.default_rng(2))
    Xhum_b = livesize(hum_g, args.target_hands, np.random.default_rng(3))
    Xhum_r = real_human_chunks(args.target_hands, np.random.default_rng(4))
    print(f"[+] chunks: bots={len(Xbot)} benchmark-humans={len(Xhum_b)} REAL-humans={len(Xhum_r)}",
          flush=True)

    X = np.asarray(Xbot + Xhum_b + Xhum_r, dtype=float)
    y = np.asarray([1] * len(Xbot) + [0] * (len(Xhum_b) + len(Xhum_r)), dtype=int)
    print(f"[+] training matrix {X.shape} bot_rate={y.mean():.2f}", flush=True)

    # Hold out 20% of each class for an honest ranking-AP read.
    rng = np.random.default_rng(7)
    def split(mask):
        idx = np.where(mask)[0]; rng.shuffle(idx)
        k = int(0.8 * len(idx)); return idx[:k], idx[k:]
    trb, hob = split(y == 1); trh, hoh = split(y == 0)
    tr = np.concatenate([trb, trh]); ho = np.concatenate([hob, hoh])

    Xg, yg, groups = make_wbn_groups(X[tr], y[tr], n_batches=args.n_batches)
    print(f"[+] WBN ranking groups {Xg.shape} in {len(groups)} groups", flush=True)

    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, label_gain=[0, 1], random_state=0, n_jobs=-1, verbose=-1,
    )
    ranker.fit(Xg, yg, group=groups)
    print("[+] ranker trained", flush=True)

    def wbn(A):
        m, s = A.mean(0), A.std(0); s = np.where(s < 1e-8, 1.0, s)
        return np.clip((A - m) / s, -5.0, 5.0)
    raw = ranker.predict(wbn(X[ho]))
    ap_all = average_precision_score(y[ho], raw)
    # AP measured ONLY against real humans as the negatives (closest to live)
    real_ho = ho[(y[ho] == 1) | np.isin(ho, np.where(y == 0)[0][-len(Xhum_r):])]
    print(f"[=] HOLDOUT ranking AP (all) = {ap_all:.4f}", flush=True)

    import joblib
    from datetime import datetime, timezone
    joblib.dump({
        "estimator": ranker, "kind": "ranker", "within_batch_norm": True,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rank_calibration": {"mode": "rank_gate", "bot_frac": 0.15,
                             "bot_low": 0.55, "bot_hi": 0.92,
                             "hum_low": 0.02, "hum_hi": 0.48},
    }, OUT)
    print(f"[=] saved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
