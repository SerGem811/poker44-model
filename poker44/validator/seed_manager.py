from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Tuple

import bittensor as bt

UTC = timezone.utc


class SynchronizedSeedManager:
    """Generate deterministic window seeds shared by honest validators."""

    def __init__(self, secret_key: str, window_minutes: int = 10):
        if not secret_key:
            raise ValueError("secret_key must be a non-empty string")
        if window_minutes <= 0:
            raise ValueError("window_minutes must be > 0")

        self.secret_key = secret_key
        self.window_minutes = window_minutes
        self._last_window_start: datetime | None = None

    def get_window_for_time(
        self, current_time: datetime | None = None
    ) -> Tuple[datetime, datetime]:
        now = current_time or datetime.now(UTC)
        window_seconds = self.window_minutes * 60
        epoch_seconds = int(now.timestamp())
        window_start_epoch = (epoch_seconds // window_seconds) * window_seconds
        window_start = datetime.fromtimestamp(window_start_epoch, tz=UTC)
        window_end = window_start + timedelta(minutes=self.window_minutes)
        return window_start, window_end

    def generate_seed(
        self, current_time: datetime | None = None
    ) -> Tuple[int, datetime, datetime]:
        window_start, window_end = self.get_window_for_time(current_time=current_time)

        time_string = window_start.strftime("%Y-%m-%d-%H:%M")
        seed_input = f"{self.secret_key}{time_string}"
        hash_hex = hashlib.sha256(seed_input.encode("utf-8")).hexdigest()
        seed = int(hash_hex[:8], 16)

        if self._last_window_start != window_start:
            self._last_window_start = window_start
            bt.logging.info(
                f"New seed window: {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')} UTC | Seed: {seed}"
            )

        return seed, window_start, window_end
