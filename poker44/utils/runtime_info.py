"""Runtime metadata helpers for validator observability."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def collect_runtime_info() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    short_commit = _run_git("rev-parse", "--short", "HEAD")
    branch = _run_git("branch", "--show-current")
    dirty = bool(_run_git("status", "--porcelain"))
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": commit,
        "git_commit_short": short_commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "pid": os.getpid(),
        "started_at": time.time(),
    }


def write_runtime_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)
