"""Validator audit lane with optional Verathos-backed verification."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import requests

from poker44.validator.integrity import load_json_registry, persist_json_registry

UTC = timezone.utc
DEFAULT_VERATHOS_BASE_URL = "https://api.verathos.ai/v1"
DEFAULT_RECENT_REPORTS = 32


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _truncate_text(value: Any, *, limit: int = 800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _normalize_json_content(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {"summary": "", "findings": [], "next_steps": []}

    try:
        payload = json.loads(text)
    except Exception:
        return {
            "summary": text,
            "findings": [],
            "next_steps": [],
        }

    if isinstance(payload, dict):
        return payload
    return {
        "summary": text,
        "findings": [],
        "next_steps": [],
    }


@dataclass(frozen=True)
class VerathosClientConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> Optional["VerathosClientConfig"]:
        api_key = str(os.getenv("POKER44_VERATHOS_API_KEY", "")).strip()
        model = str(os.getenv("POKER44_VERATHOS_MODEL", "")).strip()
        if not api_key or not model:
            return None

        base_url = str(
            os.getenv("POKER44_VERATHOS_BASE_URL", DEFAULT_VERATHOS_BASE_URL)
        ).strip().rstrip("/")
        timeout_seconds = float(
            os.getenv("POKER44_VERATHOS_TIMEOUT_SECONDS", "20")
        )
        return cls(
            base_url=base_url or DEFAULT_VERATHOS_BASE_URL,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )


class VerathosAuditClient:
    """Thin OpenAI-compatible client for Verathos audit calls."""

    def __init__(self, config: VerathosClientConfig) -> None:
        self.config = config

    def run_audit(self, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are producing a structured audit summary for Poker44, "
                        "an online poker bot-detection validator. Review the evidence "
                        "and return compact JSON with keys: summary, findings, next_steps, "
                        "confidence, and integrity_flags."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(evidence, sort_keys=True),
                },
            ],
        }
        started_at = time.time()
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        latency_seconds = time.time() - started_at
        response.raise_for_status()

        body = response.json()
        message_content = ""
        try:
            message_content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            message_content = ""

        parsed = _normalize_json_content(message_content)
        verification_metadata = {
            "proof_verified": body.get("proof_verified"),
            "verification": body.get("verification"),
            "proof": body.get("proof"),
            "attestation": body.get("attestation"),
        }
        # Drop empty keys to keep snapshots compact.
        verification_metadata = {
            key: value
            for key, value in verification_metadata.items()
            if value not in (None, "", {}, [])
        }

        proof_verified = bool(
            body.get("proof_verified")
            or (body.get("verification") or {}).get("verified")
            or (body.get("attestation") or {}).get("verified")
        )

        return {
            "provider": "verathos",
            "provider_request_id": body.get("id"),
            "provider_model": body.get("model") or self.config.model,
            "proof_verified": proof_verified,
            "verification_metadata": verification_metadata,
            "latency_seconds": latency_seconds,
            "usage": body.get("usage") or {},
            "audit_output": parsed,
            "raw_response_hash": hashlib.sha256(
                json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }


class ValidatorAuditLane:
    """Best-effort audit trail with optional external verification."""

    def __init__(
        self,
        *,
        path: str | Path | None,
        provider: str,
        mode: str,
        recent_limit: int = DEFAULT_RECENT_REPORTS,
        verathos_client: Optional[VerathosAuditClient] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.provider = provider
        self.mode = mode
        self.recent_limit = max(1, int(recent_limit))
        self.verathos_client = verathos_client
        self.registry = load_json_registry(
            self.path,
            default={"latest": {}, "recent_reports": [], "summary": {}},
        )
        self._recompute_summary()

    @classmethod
    def from_env(cls, *, path: str | Path | None) -> "ValidatorAuditLane":
        provider = str(os.getenv("POKER44_AUDIT_PROVIDER", "none")).strip().lower()
        mode = str(os.getenv("POKER44_AUDIT_MODE", "shadow")).strip().lower()
        recent_limit = int(
            os.getenv("POKER44_AUDIT_RECENT_REPORT_LIMIT", str(DEFAULT_RECENT_REPORTS))
        )

        verathos_client = None
        if provider == "verathos":
            client_cfg = VerathosClientConfig.from_env()
            if client_cfg is not None:
                verathos_client = VerathosAuditClient(client_cfg)

        return cls(
            path=path,
            provider=provider,
            mode=mode,
            recent_limit=recent_limit,
            verathos_client=verathos_client,
        )

    def public_summary(self) -> Dict[str, Any]:
        return dict(self.registry.get("summary", {}) or {})

    def latest_report(self) -> Dict[str, Any]:
        latest = self.registry.get("latest", {}) or {}
        if isinstance(latest, dict):
            return dict(latest)
        return {}

    def record_cycle(self, *, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        record = self._build_base_record(evidence=evidence)

        if self.provider == "none":
            record["status"] = "local_only"
            record["status_reason"] = "audit provider disabled"
        elif self.provider != "verathos":
            record["status"] = "provider_unsupported"
            record["status_reason"] = f"unsupported provider: {self.provider}"
        elif self.mode == "disabled":
            record["status"] = "provider_disabled"
            record["status_reason"] = "audit mode disabled"
        elif self.verathos_client is None:
            record["status"] = "provider_unconfigured"
            record["status_reason"] = "missing Verathos API configuration"
        else:
            try:
                result = self.verathos_client.run_audit(evidence)
                record.update(result)
                record["status"] = (
                    "verified" if bool(result.get("proof_verified")) else "provider_completed"
                )
                record["status_reason"] = ""
            except Exception as exc:
                record["status"] = "provider_failed"
                record["status_reason"] = _truncate_text(exc, limit=300)

        self._append_record(record)
        self._persist()
        return dict(record)

    def _build_base_record(self, *, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        latest = self.latest_report()
        provider_stats = evidence.get("provider_stats", {}) if isinstance(evidence, Mapping) else {}
        top_rows = evidence.get("top_rows", []) if isinstance(evidence, Mapping) else []
        return {
            "created_at": _now_iso(),
            "provider": self.provider,
            "mode": self.mode,
            "status": "pending",
            "status_reason": "",
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
            "dataset_hash": evidence.get("dataset_hash"),
            "validator_uid": evidence.get("validator_uid"),
            "validator_hotkey": evidence.get("validator_hotkey"),
            "forward_count": evidence.get("forward_count"),
            "top_uid": top_rows[0].get("uid") if top_rows else None,
            "top_reward": top_rows[0].get("reward") if top_rows else None,
            "input_hash": _sha256_json(dict(evidence)),
            "evidence": dict(evidence),
            "previous_status": latest.get("status"),
        }

    def _append_record(self, record: MutableMapping[str, Any]) -> None:
        recent_reports = self.registry.setdefault("recent_reports", [])
        latest_payload = {
            "created_at": record.get("created_at"),
            "provider": record.get("provider"),
            "mode": record.get("mode"),
            "status": record.get("status"),
            "status_reason": record.get("status_reason"),
            "competition_epoch_id": record.get("competition_epoch_id"),
            "active_chunk_id": record.get("active_chunk_id"),
            "active_chunk_hash": record.get("active_chunk_hash"),
            "dataset_hash": record.get("dataset_hash"),
            "top_uid": record.get("top_uid"),
            "top_reward": record.get("top_reward"),
            "proof_verified": bool(record.get("proof_verified")),
            "provider_model": record.get("provider_model"),
            "provider_request_id": record.get("provider_request_id"),
            "latency_seconds": record.get("latency_seconds"),
            "audit_output": record.get("audit_output"),
            "input_hash": record.get("input_hash"),
        }
        self.registry["latest"] = latest_payload
        recent_reports.append(dict(record))
        if len(recent_reports) > self.recent_limit:
            del recent_reports[:-self.recent_limit]
        self._recompute_summary()

    def _recompute_summary(self) -> None:
        recent_reports = list(self.registry.get("recent_reports", []) or [])
        verified_reports = sum(1 for item in recent_reports if item.get("proof_verified"))
        failed_reports = sum(
            1
            for item in recent_reports
            if str(item.get("status", "")).startswith("provider_")
            and item.get("status") not in {"provider_completed"}
        )
        latest = self.latest_report()
        self.registry["summary"] = {
            "provider": self.provider,
            "mode": self.mode,
            "recent_report_count": len(recent_reports),
            "verified_report_count": verified_reports,
            "provider_failure_count": failed_reports,
            "last_status": latest.get("status"),
            "last_created_at": latest.get("created_at"),
            "last_proof_verified": bool(latest.get("proof_verified")),
            "last_provider_model": latest.get("provider_model"),
            "latest": latest,
        }

    def _persist(self) -> None:
        persist_json_registry(self.path, self.registry)


def build_validator_audit_evidence(
    *,
    validator_uid: Optional[int],
    validator_hotkey: str,
    forward_count: int,
    dataset_hash: str,
    provider_stats: Mapping[str, Any],
    competition_rows: list[Mapping[str, Any]],
    chunk_count: int,
    total_hands: int,
    human_chunk_count: int,
    bot_chunk_count: int,
    suspicion_summary: Mapping[str, Any],
    compliance_summary: Mapping[str, Any],
    served_chunk_summary: Mapping[str, Any],
    max_rows: int,
) -> Dict[str, Any]:
    sorted_rows = sorted(
        [dict(row) for row in competition_rows],
        key=lambda row: (-float(row.get("reward", 0.0) or 0.0), int(row.get("uid", 0) or 0)),
    )
    trimmed_rows = []
    for row in sorted_rows[: max(1, int(max_rows))]:
        trimmed_rows.append(
            {
                "uid": row.get("uid"),
                "hotkey": row.get("hotkey"),
                "model_name": row.get("model_name"),
                "model_version": row.get("model_version"),
                "repo_url": row.get("repo_url"),
                "repo_commit": row.get("repo_commit"),
                "manifest_digest": row.get("manifest_digest"),
                "reward": row.get("reward"),
                "ap_score": row.get("ap_score"),
                "bot_recall": row.get("bot_recall"),
                "human_safety_penalty": row.get("human_safety_penalty"),
                "coverage_rate": row.get("coverage_rate"),
                "latency_mean_seconds": row.get("latency_mean_seconds"),
                "sample_count": row.get("sample_count"),
            }
        )

    evidence = {
        "validator_uid": validator_uid,
        "validator_hotkey": validator_hotkey,
        "forward_count": int(forward_count),
        "dataset_hash": dataset_hash,
        "provider_stats": {
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "competition_epoch_start": provider_stats.get("competition_epoch_start"),
            "competition_epoch_end": provider_stats.get("competition_epoch_end"),
            "competition_settlement_mode": provider_stats.get("competition_settlement_mode"),
            "active_window_start": provider_stats.get("active_window_start"),
            "active_window_end": provider_stats.get("active_window_end"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
        },
        "batch_summary": {
            "chunk_count": int(chunk_count),
            "total_hands": int(total_hands),
            "human_chunk_count": int(human_chunk_count),
            "bot_chunk_count": int(bot_chunk_count),
        },
        "integrity_summary": {
            "suspicion": dict(suspicion_summary or {}),
            "compliance": dict(compliance_summary or {}),
            "served_chunks": dict(served_chunk_summary or {}),
        },
        "top_rows": trimmed_rows,
    }
    evidence["evidence_hash"] = _sha256_json(evidence)
    return evidence
