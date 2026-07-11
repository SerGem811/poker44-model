#!/usr/bin/env python3
"""Build a labelled training matrix from the public Poker44 benchmark.

Pulls chunk payloads + chunk-level labels from
``https://api.poker44.net/api/v1/benchmark`` and projects every hand through
``prepare_hand_for_miner`` (the same canonicaliser the live validator applies)
before feature extraction — eliminating train/serve skew.

Output: an .npz with X (n_chunks x n_features), y (0=human, 1=bot), and the
feature-name list.

Usage:
    python scripts/miner/train/build_dataset.py \
        --out data/poker44_train.npz --dates 2026-06-10,2026-06-11 --limit 500

If --dates is omitted it discovers recent releases via /benchmark/releases.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from poker44.miner_model.features import FEATURE_NAMES, extract_chunk_features
from poker44.validator.payload_view import prepare_hand_for_miner

API = "https://api.poker44.net/api/v1/benchmark"


def _get(path: str, params: Optional[dict] = None) -> Any:
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def discover_dates(limit: int) -> List[str]:
    rel = _get("/releases", {"limit": limit})
    data = rel.get("data", rel) if isinstance(rel, dict) else rel
    items = data.get("releases", []) if isinstance(data, dict) else (data or [])
    return [str(it.get("sourceDate")) for it in items if it.get("sourceDate")]


def _label_for(idx: int, ground_truth: Any, labels: Any) -> Optional[int]:
    """Resolve a 0/1 label for chunk-group ``idx`` from the benchmark payload."""
    seq = labels if isinstance(labels, list) else ground_truth
    if isinstance(seq, list) and idx < len(seq):
        v = seq[idx]
        if isinstance(v, dict):
            v = v.get("is_bot", v.get("label", v.get("bot")))
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v > 0)
        if isinstance(v, str):
            return 1 if v.strip().lower() in {"bot", "1", "true"} else 0
    return None


def iter_chunks(date: str, limit: int, split: Optional[str]):
    """Yield chunk *records* for a date. Response shape (v1.12):
    {"success": true, "data": {"chunks": [record, ...], "nextCursor": ...}}
    """
    cursor = None
    pulled = 0
    while pulled < limit:
        params: Dict[str, Any] = {"sourceDate": date, "limit": min(24, limit - pulled)}
        if cursor:
            params["cursor"] = cursor
        if split:
            params["split"] = split
        page = _get("/chunks", params)
        data = page.get("data", page) if isinstance(page, dict) else page
        records = data.get("chunks") if isinstance(data, dict) else data
        if not isinstance(records, list) or not records:
            break
        for rec in records:
            yield rec
            pulled += 1
            if pulled >= limit:
                break
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if not cursor:
            break


def build(dates: List[str], limit: int, split: Optional[str]):
    X: List[List[float]] = []
    y: List[int] = []
    src: List[str] = []  # per-chunk source date, for date-aware split/backtest
    skipped = 0
    for date in dates:
        print(f"[+] fetching {date} ...", file=sys.stderr)
        for rec in iter_chunks(date, limit, split):
            groups = rec.get("chunks") or []
            gt = rec.get("groundTruth")
            labels = rec.get("groundTruthLabels")
            for i, group in enumerate(groups):
                label = _label_for(i, gt, labels)
                if label is None:
                    skipped += 1
                    continue
                # Apply the same canonicaliser the live validator uses so
                # training features match inference features exactly.
                canonical = [prepare_hand_for_miner(h) for h in group]
                X.append(extract_chunk_features(canonical))
                y.append(label)
                src.append(date)
    print(f"[=] built {len(X)} labelled chunks ({skipped} unlabelled skipped)", file=sys.stderr)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dates", default="", help="comma-separated YYYY-MM-DD; empty=discover")
    ap.add_argument("--limit", type=int, default=500, help="max chunk records per date")
    ap.add_argument("--split", default="", help="train|validation (optional)")
    ap.add_argument("--discover-limit", type=int, default=10)
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        dates = discover_dates(args.discover_limit)
        print(f"[+] discovered dates: {dates}", file=sys.stderr)
    if not dates:
        print("No benchmark dates available.", file=sys.stderr)
        return 1

    X, y, src = build(dates, args.limit, args.split or None)
    if len(X) == 0:
        print("No labelled data built; check API shape.", file=sys.stderr)
        return 1
    np.savez_compressed(
        args.out, X=X, y=y, dates=src, feature_names=np.array(FEATURE_NAMES)
    )
    print(
        f"[=] wrote {args.out}: X{X.shape} y{y.shape} bot_rate={y.mean():.3f} "
        f"dates={sorted(set(src.tolist()))}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
