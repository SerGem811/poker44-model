# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Poker44 validator entrypoint wired into the base Bittensor neuron."""
# neuron/validator.py

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__
from poker44.base.validator import BaseValidatorNeuron
from poker44.utils.config import config
from poker44.validator.forward import forward as forward_cycle
from hands_generator.mixed_dataset_provider import (
    DEFAULT_HUMAN_JSON_PATH,
    DEFAULT_OUTPUT_PATH,
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
)

load_dotenv()
os.makedirs("./logs", exist_ok=True)
bt.logging.set_trace()
bt.logging(debug=True, trace=False, logging_dir="./logs", record_log=True)


class Validator(BaseValidatorNeuron):
    """Poker44 validator neuron wired into the BaseValidator scaffold."""

    def __init__(self):
        cfg = config(Validator)
        super().__init__(config=cfg)
        bt.logging.info(f"🚀 Poker44 Validator v{__version__} started")

        self.forward_count = 0
        self.settings = cfg

        human_json_path = Path(
            os.getenv("POKER44_HUMAN_JSON_PATH", str(DEFAULT_HUMAN_JSON_PATH))
        ).expanduser().resolve()
        mixed_output_path = Path(
            os.getenv("POKER44_MIXED_DATASET_PATH", str(DEFAULT_OUTPUT_PATH))
        ).expanduser().resolve()
        refresh_seconds = int(
            os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(12 * 60 * 60))
        )
        chunk_count = int(os.getenv("POKER44_CHUNK_COUNT", "80"))
        min_hands_per_chunk = int(os.getenv("POKER44_MIN_HANDS_PER_CHUNK", "60"))
        max_hands_per_chunk = int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK", "120"))
        human_ratio = float(os.getenv("POKER44_HUMAN_RATIO", "0.5"))
        dataset_seed_env = os.getenv("POKER44_DATASET_SEED")
        dataset_seed = int(dataset_seed_env) if dataset_seed_env is not None else None

        self.chunk_batch_size = chunk_count
        self.dataset_cfg = MixedDatasetConfig(
            human_json_path=human_json_path,
            output_path=mixed_output_path,
            chunk_count=chunk_count,
            min_hands_per_chunk=min_hands_per_chunk,
            max_hands_per_chunk=max_hands_per_chunk,
            human_ratio=human_ratio,
            refresh_seconds=refresh_seconds,
            seed=dataset_seed,
        )
        self.provider = TimedMixedDatasetProvider(self.dataset_cfg)
        bt.logging.info(
            f"📁 Using mixed dataset provider | human_json={human_json_path} output={mixed_output_path} "
            f"chunks={chunk_count} hands_range=[{min_hands_per_chunk},{max_hands_per_chunk}] "
            f"ratio={human_ratio} refresh_s={refresh_seconds}"
        )
        self.poll_interval = self.settings.poll_interval_seconds
        self.reward_window = int(os.getenv("POKER44_REWARD_WINDOW", "50"))
        self.prediction_buffer = {}
        self.label_buffer = {}

    def resolve_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    async def forward(self, synapse=None):  # type: ignore[override]
        return await forward_cycle(self)


if __name__ == "__main__":  # pragma: no cover - manual execution
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(100)
