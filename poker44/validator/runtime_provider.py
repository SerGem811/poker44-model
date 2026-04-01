"""Provider-runtime bootstrap and dataset adapter for validator-side evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import bittensor as bt

from poker44.core.models import LabeledHandBatch


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _compute_batches_hash(batches: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(batches), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    api_base_url: str
    internal_secret: str
    validator_id: str
    min_eval_hands: int = 70
    max_eval_hands: int = 120
    require_mixed: bool = True
    attempt_publish_current: bool = True
    ensure_room: bool = True
    start_room: bool = False
    mark_evaluated: bool = True
    bootstrap_cmd: Optional[str] = None
    bootstrap_cwd: Optional[Path] = None
    bootstrap_retry_seconds: int = 30
    bootstrap_max_attempts: int = 10
    health_timeout_seconds: int = 90
    health_poll_interval_seconds: int = 5
    ensure_room_interval_seconds: int = 60
    request_timeout_seconds: int = 15

    @classmethod
    def from_env(cls, *, default_validator_id: str) -> "ProviderRuntimeConfig":
        api_base_url = _normalize_base_url(
            os.getenv("POKER44_PROVIDER_API_BASE_URL", "http://127.0.0.1:4001")
        )
        internal_secret = str(os.getenv("POKER44_PROVIDER_INTERNAL_SECRET", "")).strip()
        if not internal_secret:
            raise RuntimeError(
                "POKER44_PROVIDER_INTERNAL_SECRET is required when POKER44_RUNTIME_MODE=provider_runtime."
            )

        validator_id = (
            str(os.getenv("POKER44_PROVIDER_VALIDATOR_ID", default_validator_id)).strip()
            or default_validator_id
        )
        bootstrap_cwd_raw = str(os.getenv("POKER44_PROVIDER_BOOTSTRAP_CWD", "")).strip()
        bootstrap_cwd = Path(bootstrap_cwd_raw).expanduser().resolve() if bootstrap_cwd_raw else None
        return cls(
            api_base_url=api_base_url,
            internal_secret=internal_secret,
            validator_id=validator_id,
            min_eval_hands=max(0, int(os.getenv("POKER44_PROVIDER_MIN_EVAL_HANDS", "70"))),
            max_eval_hands=max(0, int(os.getenv("POKER44_PROVIDER_MAX_EVAL_HANDS", "120"))),
            require_mixed=_env_bool("POKER44_PROVIDER_REQUIRE_MIXED", True),
            attempt_publish_current=_env_bool("POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT", True),
            ensure_room=_env_bool("POKER44_PROVIDER_ENSURE_ROOM", True),
            start_room=_env_bool("POKER44_PROVIDER_START_ROOM", False),
            mark_evaluated=_env_bool("POKER44_PROVIDER_MARK_EVALUATED", True),
            bootstrap_cmd=str(os.getenv("POKER44_PROVIDER_BOOTSTRAP_CMD", "")).strip() or None,
            bootstrap_cwd=bootstrap_cwd,
            bootstrap_retry_seconds=int(os.getenv("POKER44_PROVIDER_BOOTSTRAP_RETRY_SECONDS", "30")),
            bootstrap_max_attempts=int(os.getenv("POKER44_PROVIDER_BOOTSTRAP_MAX_ATTEMPTS", "10")),
            health_timeout_seconds=int(os.getenv("POKER44_PROVIDER_HEALTH_TIMEOUT_SECONDS", "90")),
            health_poll_interval_seconds=int(
                os.getenv("POKER44_PROVIDER_HEALTH_POLL_INTERVAL_SECONDS", "5")
            ),
            ensure_room_interval_seconds=int(
                os.getenv("POKER44_PROVIDER_ENSURE_ROOM_INTERVAL_SECONDS", "60")
            ),
            request_timeout_seconds=int(os.getenv("POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS", "15")),
        )

    def public_summary(self) -> Dict[str, Any]:
        return {
            "mode": "provider_runtime",
            "api_base_url": self.api_base_url,
            "validator_id": self.validator_id,
            "min_eval_hands": self.min_eval_hands,
            "max_eval_hands": self.max_eval_hands,
            "require_mixed": self.require_mixed,
            "attempt_publish_current": self.attempt_publish_current,
            "ensure_room": self.ensure_room,
            "start_room": self.start_room,
            "mark_evaluated": self.mark_evaluated,
            "bootstrap_cmd_configured": bool(self.bootstrap_cmd),
        }


class _ProviderRuntimeHttpClient:
    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        url = f"{self.cfg.api_base_url}{path}"
        if query:
            url = f"{url}?{urlencode({k: v for k, v in query.items() if v is not None})}"

        headers = {
            "accept": "application/json",
            "x-eval-secret": self.cfg.internal_secret,
        }
        body_bytes = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        request = Request(url, data=body_bytes, headers=headers, method=method.upper())
        with urlopen(request, timeout=self.cfg.request_timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        decoded = json.loads(raw) if raw else None
        if isinstance(decoded, dict) and decoded.get("success") is True and "data" in decoded:
            return decoded["data"]
        return decoded

    def get(self, path: str, *, query: Optional[Mapping[str, Any]] = None) -> Any:
        return self._request("GET", path, query=query)

    def post(self, path: str, *, payload: Optional[Mapping[str, Any]] = None) -> Any:
        return self._request("POST", path, payload=payload)


class ProviderRuntimeManager:
    """Best-effort bootstrapper/health-checker for validator-local provider runtime."""

    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg
        self.client = _ProviderRuntimeHttpClient(cfg)
        self.last_room_code: Optional[str] = None
        self.last_ensure_room_ts: float = 0.0
        self.last_bootstrap_ts: float = 0.0
        self.bootstrap_attempts: int = 0
        self.status: Dict[str, Any] = {
            "runtime_ready": False,
            "room_code": None,
            "bootstrap_attempts": 0,
            "last_error": "",
        }

    def ensure_runtime_ready(self) -> bool:
        if self._runtime_healthy():
            self.status["runtime_ready"] = True
            self.status["last_error"] = ""
            self._ensure_room_if_due()
            return True

        if self._should_bootstrap():
            self._run_bootstrap()

        deadline = time.time() + max(1, self.cfg.health_timeout_seconds)
        while time.time() < deadline:
            if self._runtime_healthy():
                self.status["runtime_ready"] = True
                self.status["last_error"] = ""
                self._ensure_room_if_due(force=True)
                return True
            time.sleep(max(1, self.cfg.health_poll_interval_seconds))

        self.status["runtime_ready"] = False
        if not self.status.get("last_error"):
            self.status["last_error"] = "runtime healthchecks did not become ready in time"
        bt.logging.warning(
            f"Provider runtime not ready yet | base_url={self.cfg.api_base_url} error={self.status['last_error']}"
        )
        return False

    def _runtime_healthy(self) -> bool:
        try:
            eval_health = self.client.get("/internal/eval/health")
            room_health = self.client.get("/internal/rooms/health")
            eval_ok = isinstance(eval_health, dict) and bool(eval_health.get("ok"))
            room_ok = isinstance(room_health, dict) and bool(room_health.get("ok"))
            return eval_ok and room_ok
        except Exception as exc:
            self.status["last_error"] = str(exc)
            return False

    def _should_bootstrap(self) -> bool:
        if not self.cfg.bootstrap_cmd:
            return False
        if self.bootstrap_attempts >= max(1, self.cfg.bootstrap_max_attempts):
            return False
        if (time.time() - self.last_bootstrap_ts) < max(1, self.cfg.bootstrap_retry_seconds):
            return False
        return True

    def _run_bootstrap(self) -> None:
        assert self.cfg.bootstrap_cmd
        self.bootstrap_attempts += 1
        self.last_bootstrap_ts = time.time()
        self.status["bootstrap_attempts"] = self.bootstrap_attempts
        bt.logging.info(
            f"Provider runtime bootstrap attempt #{self.bootstrap_attempts} | cmd={self.cfg.bootstrap_cmd}"
        )
        try:
            result = subprocess.run(
                self.cfg.bootstrap_cmd,
                shell=True,
                cwd=str(self.cfg.bootstrap_cwd) if self.cfg.bootstrap_cwd else None,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.status["last_error"] = (
                    f"bootstrap failed rc={result.returncode}: {(result.stderr or result.stdout).strip()[:500]}"
                )
                bt.logging.warning(f"Provider runtime bootstrap failed: {self.status['last_error']}")
                return
            bt.logging.info("Provider runtime bootstrap command completed")
        except Exception as exc:
            self.status["last_error"] = f"bootstrap exception: {exc}"
            bt.logging.warning(self.status["last_error"])

    def _ensure_room_if_due(self, *, force: bool = False) -> None:
        if not self.cfg.ensure_room:
            return
        if not force and (time.time() - self.last_ensure_room_ts) < max(
            1, self.cfg.ensure_room_interval_seconds
        ):
            return
        try:
            result = self.client.post(
                "/internal/rooms/ensure",
                payload={"validatorId": self.cfg.validator_id},
            )
            room_code = None
            if isinstance(result, dict):
                room_code = str(result.get("roomCode") or "").strip().upper() or None
            if room_code:
                self.last_room_code = room_code
                self.status["room_code"] = room_code
                bt.logging.info(f"Provider runtime ensured room {room_code}")
                if self.cfg.start_room:
                    self.client.post(f"/internal/rooms/{room_code}/start", payload={})
            self.last_ensure_room_ts = time.time()
        except Exception as exc:
            self.status["last_error"] = f"ensure room failed: {exc}"
            bt.logging.warning(self.status["last_error"])


class ProviderRuntimeDatasetProvider:
    """Consumes live labeled eval batches from the provider runtime."""

    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg
        self.manager = ProviderRuntimeManager(cfg)
        self._dataset_hash: str = ""
        self._stats: Dict[str, Any] = cfg.public_summary()
        self._pending_hand_ids: List[str] = []
        self._active_chunk_id: str = ""

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @property
    def stats(self) -> Dict[str, Any]:
        merged = dict(self._stats)
        merged.update(self.manager.status)
        return merged

    def refresh_if_due(self) -> None:
        self.manager.ensure_runtime_ready()

    def fetch_hand_batch(
        self,
        *,
        limit: int = 80,
        include_integrity: bool = True,
    ) -> List[LabeledHandBatch]:
        _ = include_integrity
        self._pending_hand_ids = []
        self._active_chunk_id = ""

        if not self.manager.ensure_runtime_ready():
            self._stats.update(
                {
                    "batch_count": 0,
                    "last_fetch_status": "runtime_not_ready",
                    "last_fetch_at": int(time.time()),
                }
            )
            return []

        if self.cfg.min_eval_hands > 0:
            try:
                eval_health = self.manager.client.get(
                    "/internal/eval/health",
                    query={"minHands": self.cfg.min_eval_hands},
                )
                available_hands = 0
                ready_for_evaluation = False
                if isinstance(eval_health, dict):
                    available_hands = int(eval_health.get("availableHands") or 0)
                    ready_for_evaluation = bool(eval_health.get("readyForEvaluation"))
                self._stats.update(
                    {
                        "available_hands": available_hands,
                        "min_eval_hands": self.cfg.min_eval_hands,
                    }
                )
                if not ready_for_evaluation:
                    self._stats.update(
                        {
                            "batch_count": 0,
                            "last_fetch_status": "waiting_for_min_hands",
                            "last_fetch_at": int(time.time()),
                        }
                    )
                    bt.logging.info(
                        f"Provider runtime waiting for enough hands | have={available_hands} need={self.cfg.min_eval_hands}"
                    )
                    return []
            except Exception as exc:
                self._stats.update(
                    {
                        "batch_count": 0,
                        "last_fetch_status": f"eval_health_error:{exc}",
                        "last_fetch_at": int(time.time()),
                    }
                )
                bt.logging.warning(f"Provider runtime eval health check failed: {exc}")
                return []

        try:
            if self.cfg.attempt_publish_current:
                try:
                    publish_result = self.manager.client.post(
                        "/internal/eval/publish-current",
                        payload={
                            "validatorId": self.cfg.validator_id,
                            "handCount": max(self.cfg.min_eval_hands, min(self.cfg.max_eval_hands, limit or self.cfg.max_eval_hands)),
                            "requireMixed": self.cfg.require_mixed,
                        },
                    )
                    if isinstance(publish_result, dict):
                        self._stats.update(
                            {
                                "publish_reason": str(publish_result.get("reason") or ""),
                                "publish_chunk_id": str(publish_result.get("chunkId") or ""),
                                "publish_chunk_hash": str(publish_result.get("chunkHash") or ""),
                            }
                        )
                except Exception as exc:
                    bt.logging.warning(f"Provider runtime publish-current failed: {exc}")

            payload = self.manager.client.get("/internal/eval/current")
            batches_raw = payload.get("batches", []) if isinstance(payload, dict) else []
            if not isinstance(batches_raw, list):
                batches_raw = []
            if limit > 0:
                batches_raw = batches_raw[:limit]

            canonical_chunk_hash = (
                str(payload.get("chunkHash") or "").strip()
                if isinstance(payload, dict)
                else ""
            )
            self._dataset_hash = (
                canonical_chunk_hash
                if canonical_chunk_hash
                else (_compute_batches_hash(batches_raw) if batches_raw else "")
            )
            self._stats.update(
                {
                    "runtime_mode": "provider_runtime",
                    "batch_count": len(batches_raw),
                    "requested_limit": limit,
                    "last_fetch_status": "ok" if batches_raw else "waiting_for_active_chunk",
                    "last_fetch_at": int(time.time()),
                    "room_code": self.manager.last_room_code,
                    "active_chunk_id": str(payload.get("chunkId") or "") if isinstance(payload, dict) else "",
                    "active_chunk_hash": str(payload.get("chunkHash") or "") if isinstance(payload, dict) else "",
                    "active_chunk_producer": str(payload.get("producerValidatorId") or "") if isinstance(payload, dict) else "",
                    "active_window_start": str(payload.get("windowStart") or "") if isinstance(payload, dict) else "",
                    "active_window_end": str(payload.get("windowEnd") or "") if isinstance(payload, dict) else "",
                }
            )
            self._active_chunk_id = str(payload.get("chunkId") or "").strip() if isinstance(payload, dict) else ""

            if not batches_raw:
                return []

            batches: List[LabeledHandBatch] = []
            hand_ids: List[str] = []
            for entry in batches_raw:
                if not isinstance(entry, dict):
                    continue
                hands_raw = entry.get("hands")
                if not isinstance(hands_raw, list):
                    continue
                normalized_hands = [hand for hand in hands_raw if isinstance(hand, dict)]
                if not normalized_hands:
                    continue
                is_human = bool(entry.get("is_human", False))
                batches.append(LabeledHandBatch(hands=normalized_hands, is_human=is_human))  # type: ignore[arg-type]
                for hand in normalized_hands:
                    hand_id = str(hand.get("hand_id") or "").strip()
                    if hand_id:
                        hand_ids.append(hand_id)

            self._pending_hand_ids = sorted(set(hand_ids))
            return batches
        except Exception as exc:
            self._stats.update(
                {
                    "batch_count": 0,
                    "last_fetch_status": f"error:{exc}",
                    "last_fetch_at": int(time.time()),
                }
            )
            bt.logging.warning(f"Provider runtime fetch failed: {exc}")
            return []

    def mark_last_batch_evaluated(self) -> None:
        if not self.cfg.mark_evaluated:
            self._pending_hand_ids = []
            return
        if not self._pending_hand_ids:
            return
        try:
            result = self.manager.client.post(
                "/internal/eval/mark-evaluated",
                payload={
                    "hand_ids": list(self._pending_hand_ids),
                    "chunkId": self._active_chunk_id,
                    "validatorId": self.cfg.validator_id,
                },
            )
            updated = 0
            if isinstance(result, dict):
                updated = int(result.get("updated") or 0)
            bt.logging.info(
                f"Provider runtime marked evaluated hands | requested={len(self._pending_hand_ids)} updated={updated}"
            )
        except Exception as exc:
            bt.logging.warning(f"Provider runtime mark-evaluated failed: {exc}")
        finally:
            self._pending_hand_ids = []
