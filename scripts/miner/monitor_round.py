#!/usr/bin/env python3
"""Append a Poker44 miner metrics snapshot to logs/round_metrics.csv.

Captures, per run: UTC timestamp, epoch id, rank, composite score, eligibility
source, latest live flag-rate (from pm2 logs), and the per-round window scores.
Run from cron every couple of hours to build a clean time series for the slow,
blind live-tuning loop (so threshold changes can be judged against real signal).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from urllib.request import urlopen

UID = int(os.getenv("POKER44_UID", "162"))
PM2_NAME = os.getenv("PM2_NAME", "poker44_miner")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO, "logs", "round_metrics.csv")
LB_URL = "https://api.poker44.net/api/v1/competition/leaderboard"


def latest_flag_rate() -> str:
    """Parse the most recent 'Scored N chunks | flagged=M' line from pm2 logs."""
    try:
        out = subprocess.run(
            ["pm2", "logs", PM2_NAME, "--lines", "400", "--nostream"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return ""
    last = ""
    for m in re.finditer(r"Scored (\d+) chunks \| flagged=(\d+)", out):
        total, flag = int(m.group(1)), int(m.group(2))
        last = f"{flag}/{total}={flag/total:.2f}" if total else ""
    return last


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with urlopen(LB_URL, timeout=20) as r:
            d = json.load(r)["data"]
    except Exception as exc:
        row = {"ts": ts, "error": f"leaderboard fetch failed: {exc}"}
        _append(row)
        return 0

    rows = d.get("rows", [])
    me = next((x for x in rows if x.get("uid") == UID), None)
    epoch = d.get("epoch", {})
    flag = latest_flag_rate()

    if me is None:
        _append({"ts": ts, "epoch": epoch.get("epochId"), "error": "uid not on leaderboard", "flag_rate": flag})
        return 0

    rounds = {f"R{w['roundIndex']}": w.get("compositeScore")
              for w in me.get("windowCompositeScores", [])}
    row = {
        "ts": ts,
        "epoch": epoch.get("epochId"),
        "secs_remaining": epoch.get("secondsRemaining"),
        "rank": me.get("rank"),
        "composite": me.get("compositeScore"),
        "score_source": me.get("scoreSource"),
        "manifest_failed": me.get("manifestReviewFailed"),
        "flag_rate": flag,
        **rounds,
    }
    _append(row)
    print(f"[{ts}] rank={row['rank']} composite={row['composite']} flag={flag} src={row['score_source']}")
    return 0


def _append(row: dict) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    # union of existing header + this row's keys, so new round columns are allowed
    fields = list(row.keys())
    existing = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            existing = next(csv.reader(f), [])
        fields = existing + [k for k in row if k not in existing]
    rewrite = fields != existing
    mode = "w" if rewrite else "a"
    prior = []
    if rewrite and os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            prior = list(csv.DictReader(f))
    with open(CSV_PATH, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader() if (rewrite or mode == "w") else None
        for p in prior:
            w.writerow(p)
        w.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
