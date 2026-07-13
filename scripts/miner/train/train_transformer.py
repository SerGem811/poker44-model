#!/usr/bin/env python3
"""Train the ChunkTransformer bot-detector and add it to an existing model bundle.

Steps:
  1. Load session-sequence data (from build_sequences.py output).
  2. Split: hold out last HOLDOUT_FRAC of sessions (by date order).
  3. Train TransformerScorer on train set.
  4. Evaluate on holdout — AP, reward.
  5. If AP >= MIN_AP_TO_INCLUDE, add the transformer to the existing model bundle
     as an additional scorer. Otherwise keep the existing bundle unchanged.

The combined scorer: average(GBDT_prob, Transformer_prob).

Usage:
    python scripts/miner/train/train_transformer.py \
        --sequences data/sequences.pkl \
        --bundle models/poker44_gbdt.joblib \
        --out models/poker44_gbdt.joblib
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)

from sklearn.metrics import average_precision_score  # noqa: E402

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    print("[!] PyTorch not available — cannot train Transformer", file=sys.stderr)

import joblib  # noqa: E402
from poker44.miner_model.transformer import TransformerScorer  # noqa: E402
from poker44.score.scoring import reward as validator_reward  # noqa: E402
from poker44.miner_model.scoring_head import ScoringHead  # noqa: E402


HOLDOUT_FRAC = float(os.getenv("TRANSFORMER_HOLDOUT_FRAC", "0.20"))
MIN_AP_TO_INCLUDE = float(os.getenv("TRANSFORMER_MIN_AP", "0.70"))


def simulate_reward(probs: np.ndarray, y: np.ndarray, window: int = 20, trials: int = 800) -> float:
    rng = np.random.default_rng(42)
    n = len(probs)
    if n < window:
        return float(validator_reward(probs, y)[0])
    vals = [
        float(validator_reward(probs[rng.choice(n, window, replace=False)],
                               y[rng.choice(n, window, replace=False)].astype(bool))[0])
        for _ in range(trials)
    ]
    return float(np.mean(vals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="data/sequences.pkl")
    parser.add_argument("--bundle", default="models/poker44_gbdt.joblib")
    parser.add_argument("--out", default="models/poker44_gbdt.joblib")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not _TORCH_AVAILABLE:
        print("[!] PyTorch not available; skipping Transformer training.", file=sys.stderr)
        return 1

    # --- Load sequences ---
    if not os.path.exists(args.sequences):
        print(f"[!] {args.sequences} not found. Run build_sequences.py first.", file=sys.stderr)
        return 1
    with open(args.sequences, "rb") as f:
        data = pickle.load(f)
    sessions = [s for s, _ in data]
    labels   = np.array([y for _, y in data], dtype=int)
    print(f"[+] loaded {len(sessions)} sessions bot_rate={labels.mean():.3f}", flush=True)

    if len(np.unique(labels)) < 2:
        print("[!] single-class dataset; cannot train.", file=sys.stderr)
        return 1

    # --- Train/hold split (temporal: last HOLDOUT_FRAC as holdout) ---
    n = len(sessions)
    split = int(n * (1 - HOLDOUT_FRAC))
    if split < 10 or (n - split) < 5:
        print(f"[!] too few samples (n={n}) for {HOLDOUT_FRAC:.0%} holdout", file=sys.stderr)
        return 1

    s_train, y_train = sessions[:split], labels[:split]
    s_hold,  y_hold  = sessions[split:], labels[split:]
    print(f"[+] split: train={len(s_train)} hold={len(s_hold)}", flush=True)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_hold)) < 2:
        print("[!] split produced a single-class set; skipping.", file=sys.stderr)
        return 1

    # --- Train Transformer ---
    scorer = TransformerScorer(
        d_model=args.d_model,
        n_heads=4,
        n_hand_layers=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=1e-3,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"[+] training TransformerScorer ({args.epochs} epochs, d_model={args.d_model}) ...", flush=True)
    scorer.fit(s_train, y_train)

    # --- Evaluate on holdout ---
    p_hold = scorer.predict_proba(s_hold)[:, 1]
    ap = average_precision_score(y_hold, p_hold)
    reward_t = simulate_reward(p_hold, y_hold)
    print(f"[=] Transformer holdout: AP={ap:.4f}  sim_reward={reward_t:.4f}", flush=True)

    if ap < MIN_AP_TO_INCLUDE:
        print(f"[=] AP {ap:.4f} < threshold {MIN_AP_TO_INCLUDE}; NOT adding Transformer to bundle.", flush=True)
        return 0

    # --- Load existing GBDT bundle and evaluate it on same holdout ---
    if not os.path.exists(args.bundle):
        print(f"[!] bundle {args.bundle} not found; saving transformer-only bundle.", file=sys.stderr)
    else:
        bundle = joblib.load(args.bundle)
        est = bundle["estimator"]
        head_dict = bundle.get("scoring_head", {})

        # Need feature vectors for GBDT holdout evaluation — re-extract from raw sessions
        from poker44.miner_model.features import extract_chunk_features  # noqa: E402
        X_hold = np.array([extract_chunk_features(s) for s in s_hold])
        p_gbdt = est.predict_proba(X_hold)[:, 1]
        ap_gbdt = average_precision_score(y_hold, p_gbdt)
        print(f"[=] GBDT holdout: AP={ap_gbdt:.4f}", flush=True)

        # Combined (simple average)
        p_combined = 0.5 * p_gbdt + 0.5 * p_hold
        ap_combined = average_precision_score(y_hold, p_combined)
        reward_combined = simulate_reward(p_combined, y_hold)
        print(f"[=] Combined holdout: AP={ap_combined:.4f}  sim_reward={reward_combined:.4f}", flush=True)

        if ap_combined < ap_gbdt - 0.005:
            print(f"[=] Combined AP {ap_combined:.4f} worse than GBDT {ap_gbdt:.4f}; NOT including Transformer.", flush=True)
            return 0

        bundle["transformer"] = scorer
        bundle["transformer_weight"] = 0.5
        if "metrics" not in bundle:
            bundle["metrics"] = {}
        bundle["metrics"]["ap_transformer"] = float(ap)
        bundle["metrics"]["ap_combined"] = float(ap_combined)
        bundle["metrics"]["sim_reward_combined"] = float(reward_combined)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[=] saved combined bundle to {args.out}", flush=True)
    print(f"[=] metrics: {bundle['metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
