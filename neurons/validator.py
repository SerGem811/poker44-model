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
from pathlib import Path
from typing import Optional

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__, VALIDATOR_DEPLOY_VERSION
from poker44.base.validator import BaseValidatorNeuron
from poker44.utils.config import config
from poker44.utils.runtime_info import (
    collect_runtime_info,
    post_runtime_snapshot,
    write_runtime_snapshot,
)
from poker44.utils.wandb_helper import ValidatorWandbHelper
from poker44.validator.forward import forward as forward_cycle
from poker44.validator.integrity import (
    load_json_registry,
    normalize_uid_key_registry,
)
from hands_generator.mixed_dataset_provider import (
    DEFAULT_OUTPUT_PATH,
    MixedDatasetConfig,
    TimedMixedDatasetProvider,
)

load_dotenv()
os.makedirs("./logs", exist_ok=True)
bt.logging.set_trace()
bt.logging(debug=True, trace=False, logging_dir="./logs", record_log=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class Validator(BaseValidatorNeuron):
    """Poker44 validator neuron wired into the BaseValidator scaffold."""

    def __init__(self):
        cfg = config(Validator)
        super().__init__(config=cfg)
        bt.logging.info(f"🚀 Poker44 Validator v{__version__} started")

        self.forward_count = 0
        self.settings = cfg

        human_json_env = os.getenv("POKER44_HUMAN_JSON_PATH")
        if not human_json_env:
            raise RuntimeError(
                "POKER44_HUMAN_JSON_PATH must point to the private local human-hand JSON used by validators."
            )

        human_json_path = Path(human_json_env).expanduser().resolve()
        mixed_output_path = Path(
            os.getenv("POKER44_MIXED_DATASET_PATH", str(DEFAULT_OUTPUT_PATH))
        ).expanduser().resolve()
        refresh_seconds = int(
            os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(60 * 60))
        )
        chunk_count = int(os.getenv("POKER44_CHUNK_COUNT", "40"))
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
        bt.logging.info("🧭 Dataset generation is deterministic per refresh window.")
        configured_poll_interval = getattr(cfg, "poll_interval_seconds", refresh_seconds)
        self.poll_interval = int(
            os.getenv("POKER44_POLL_INTERVAL_SECONDS", str(configured_poll_interval))
        )
        self.reward_window = int(os.getenv("POKER44_REWARD_WINDOW", "40"))
        self.synced_window_mode = _env_bool("POKER44_SYNCED_WINDOW_MODE", True)
        self.sync_all_miners = _env_bool("POKER44_SYNC_ALL_MINERS", False)
        self.sync_direct_score_update = _env_bool(
            "POKER44_SYNC_DIRECT_SCORE_UPDATE",
            self.synced_window_mode,
        )
        self.sync_reset_buffers_on_window_change = _env_bool(
            "POKER44_SYNC_RESET_BUFFERS_ON_WINDOW_CHANGE",
            self.synced_window_mode,
        )
        self.current_eval_window_id: Optional[int] = None
        self.prediction_buffer = {}
        self.label_buffer = {}
        state_dir = Path(self.config.neuron.full_path)
        self.model_manifest_path = state_dir / "model_manifests.json"
        self.compliance_registry_path = state_dir / "compliance_registry.json"
        self.suspicion_registry_path = state_dir / "suspicion_registry.json"
        self.served_chunk_registry_path = state_dir / "served_chunk_registry.json"
        self.model_manifest_registry = load_json_registry(self.model_manifest_path)
        if self.model_manifest_registry:
            self.model_manifest_registry = normalize_uid_key_registry(
                self.model_manifest_registry
            )
        self.compliance_registry = load_json_registry(
            self.compliance_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.suspicion_registry = load_json_registry(
            self.suspicion_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.served_chunk_registry = load_json_registry(
            self.served_chunk_registry_path,
            default={"chunk_index": {}, "recent_cycles": [], "summary": {}},
        )
        self.wandb_helper = ValidatorWandbHelper(
            config=cfg,
            validator_uid=self.resolve_uid(self.wallet.hotkey.ss58_address),
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            netuid=cfg.netuid,
        )
        self.wandb_helper.log_validator_startup(
            dataset_cfg=self.dataset_cfg,
            poll_interval=self.poll_interval,
            reward_window=self.reward_window,
            runtime_info=self.runtime_info,
        )
        bt.logging.info(
            "🪟 Validator sync mode | "
            f"synced_window_mode={self.synced_window_mode} "
            f"sync_all_miners={self.sync_all_miners} "
            f"direct_score_update={self.sync_direct_score_update} "
            f"reset_buffers_on_window_change={self.sync_reset_buffers_on_window_change}"
        )
        bt.logging.info(
            "🧾 Validator runtime | "
            f"uid={self.resolve_uid(self.wallet.hotkey.ss58_address)} "
            f"hotkey={self.wallet.hotkey.ss58_address} "
            f"version={__version__} "
            f"deploy_version={VALIDATOR_DEPLOY_VERSION} "
            f"git_branch={self.runtime_info.get('git_branch', '')} "
            f"git_commit={self.runtime_info.get('git_commit_short', '')} "
            f"git_dirty={self.runtime_info.get('git_dirty', False)}"
        )
        self._write_runtime_snapshot(status="started")

    def resolve_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    @property
    def runtime_snapshot_path(self) -> Path:
        return Path(self.config.neuron.full_path) / "validator_runtime.json"

    @property
    def runtime_info(self) -> dict:
        info = getattr(self, "_runtime_info", None)
        if info is None:
            info = collect_runtime_info()
            self._runtime_info = info
        return info

    def _write_runtime_snapshot(self, *, status: str, extra: Optional[dict] = None) -> None:
        payload = {
            "status": status,
            "validator_uid": self.resolve_uid(self.wallet.hotkey.ss58_address),
            "hotkey": self.wallet.hotkey.ss58_address,
            "version": __version__,
            "deploy_version": VALIDATOR_DEPLOY_VERSION,
            "netuid": self.config.netuid,
            "poll_interval": self.poll_interval,
            "reward_window": self.reward_window,
            "synced_window_mode": self.synced_window_mode,
            "sync_all_miners": self.sync_all_miners,
            "sync_direct_score_update": self.sync_direct_score_update,
            "runtime": self.runtime_info,
        }
        if extra:
            payload.update(extra)
        write_runtime_snapshot(self.runtime_snapshot_path, payload)
        report_url = str(os.getenv("POKER44_VALIDATOR_RUNTIME_REPORT_URL", "")).strip()
        report_secret = str(os.getenv("POKER44_VALIDATOR_RUNTIME_SECRET", "")).strip()
        if report_url and report_secret:
            timeout_seconds = float(
                os.getenv("POKER44_VALIDATOR_RUNTIME_REPORT_TIMEOUT_SECONDS", "5")
            )
            ok, message = post_runtime_snapshot(
                url=report_url,
                secret=report_secret,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            if ok:
                bt.logging.debug(
                    f"Validator runtime snapshot reported successfully to collector: {report_url}"
                )
            else:
                bt.logging.warning(
                    "Validator runtime snapshot report failed | "
                    f"url={report_url} message={message}"
                )

    async def forward(self, synapse=None):  # type: ignore[override]
        return await forward_cycle(self)

    def __del__(self) -> None:
        wandb_helper = getattr(self, "wandb_helper", None)
        if wandb_helper is not None:
            try:
                wandb_helper.finish()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover - manual execution
    validator = Validator()
    validator.run()
