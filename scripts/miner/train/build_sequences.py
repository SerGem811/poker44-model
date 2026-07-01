#!/usr/bin/env python3
"""Build raw session-sequence dataset for Transformer training.

Pulls the same benchmark data as build_dataset.py but keeps raw action
sequences (not aggregated features) so the Transformer can learn from
action order and context.

Output: a .pkl file with list of (session: List[Dict], label: int) tuples.

Usage:
    python scripts/miner/train/build_sequences.py \
        --out data/sequences.pkl --discover-limit 60
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List, Optional, Tuple, Dict, Any

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.insert(0, REPO)

API = "https://api.poker44.net/api/v1/benchmark"


def _get(path: str, params: Optional[dict] = None) -> Any:
    r = requests.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def discover_dates(limit: int = 60) -> List[str]:
    rel = _get("/releases", {"limit": limit})
    data = rel.get("data", rel) if isinstance(rel, dict) else rel
    items = data.get("releases", []) if isinstance(data, dict) else (data or [])
    return [str(it.get("sourceDate")) for it in items if it.get("sourceDate")]


def _label_for(idx: int, ground_truth: Any, labels: Any) -> Optional[int]:
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


def build_sequences(
    dates: List[str],
    per_date_limit: int = 500,
) -> List[Tuple[List[Dict], int]]:
    """Pull labeled sessions and return (session, label) pairs."""
    results: List[Tuple[List[Dict], int]] = []
    skipped = 0

    for date in dates:
        print(f"[+] fetching {date} ...", flush=True)
        cursor = None
        pulled = 0
        while pulled < per_date_limit:
            params: dict = {
                "sourceDate": date,
                "limit": min(24, per_date_limit - pulled),
            }
            if cursor:
                params["cursor"] = cursor
            page = _get("/chunks", params)
            data = page.get("data", page) if isinstance(page, dict) else page
            records = data.get("chunks") if isinstance(data, dict) else data
            if not isinstance(records, list) or not records:
                break
            for rec in records:
                groups = rec.get("chunks") or []
                gt = rec.get("groundTruth")
                labels = rec.get("groundTruthLabels")
                for i, group in enumerate(groups):
                    label = _label_for(i, gt, labels)
                    if label is None:
                        skipped += 1
                        continue
                    # group is List[Dict] — list of hand dicts
                    session = [h for h in group if isinstance(h, dict)]
                    if session:
                        results.append((session, int(label)))
                pulled += 1
                if pulled >= per_date_limit:
                    break
            cursor = data.get("nextCursor") if isinstance(data, dict) else None
            if not cursor:
                break

    print(f"[=] built {len(results)} labeled sessions ({skipped} unlabelled skipped)", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dates", default="", help="comma-separated YYYY-MM-DD; empty=discover")
    parser.add_argument("--discover-limit", type=int, default=60)
    parser.add_argument("--per-date-limit", type=int, default=500)
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        dates = sorted(discover_dates(args.discover_limit))
        print(f"[+] discovered {len(dates)} dates", flush=True)
    if not dates:
        print("No benchmark dates available.", file=sys.stderr)
        return 1

    sessions = build_sequences(dates, args.per_date_limit)
    if not sessions:
        print("No labeled sessions built.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(sessions, f, protocol=4)
    labels = [y for _, y in sessions]
    bot_rate = sum(labels) / max(1, len(labels))
    print(f"[=] wrote {args.out}: {len(sessions)} sessions bot_rate={bot_rate:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
