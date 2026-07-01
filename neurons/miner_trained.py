"""Trained Poker44 miner: GBDT bot-detector + monotonic FPR-safe scoring head.

Drop-in replacement for ``neurons/miner.py``. It loads a model artifact produced
by ``scripts/miner/train/train_model.py`` and scores each chunk with:

    risk_score = scoring_head( model.predict_proba(features(chunk)) )

The scoring head positions the validator's fixed 0.5 cutoff at a calibrated
high-precision operating point so the FPR cliff (reward=0 at FPR>=0.10) is
protected while average-precision ranking is preserved (see scoring_head.py).

If no artifact is present it falls back to a deterministic heuristic over the
same features - meaningfully better than random, never worse than the reference
miner - so the axon always answers and keeps its 20-sample window full.

Run exactly like the reference miner, e.g.:

    POKER44_MODEL_PATH=models/poker44_gbdt.joblib \
    python neurons/miner_trained.py --netuid 126 --wallet.name <ck> \
        --wallet.hotkey <hk> --subtensor.network finney --axon.port 8091 \
        --blacklist.allowed_validator_hotkeys <validator_hotkey...>
"""

# from __future__ import annotations

import math
import os
import time
from pathlib import Path
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_model.features import FEATURE_NAMES, extract_chunk_features
from poker44.miner_model.scoring_head import ScoringHead, shape_risk_score
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

_DEFAULT_MODEL_PATH = os.getenv("POKER44_MODEL_PATH", "models/poker44_gbdt.joblib")


class TrainedModel:
    """Thin wrapper around a persisted sklearn/LightGBM estimator + scoring head."""

    def __init__(self, estimator, head: ScoringHead, feature_names: List[str]):
        self.estimator = estimator
        self.head = head
        self.feature_names = feature_names

    @classmethod
    def load(cls, path: str) -> Optional["TrainedModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            import joblib

            blob = joblib.load(p)
            return cls(
                estimator=blob["estimator"],
                head=ScoringHead.from_dict(blob.get("scoring_head", {})),
                feature_names=blob.get("feature_names", FEATURE_NAMES),
            )
        except Exception as exc:  # pragma: no cover - defensive load
            bt.logging.warning(f"Failed to load model from {path}: {exc}")
            return None

    def proba(self, feats: List[List[float]]) -> List[float]:
        import numpy as np

        x = np.asarray(feats, dtype=float)
        if hasattr(self.estimator, "predict_proba"):
            return [float(v) for v in self.estimator.predict_proba(x)[:, 1]]
        raw = self.estimator.predict(x)
        return [float(1.0 / (1.0 + math.exp(-float(v)))) for v in raw]


def _heuristic_proba(feats: List[float]) -> float:
    """Feature-driven fallback: emphasise sizing regularity + aggression tells."""
    f = dict(zip(FEATURE_NAMES, feats))
    z = (
        1.6 * (1.0 - min(1.0, f["size_bucket_entropy"] / 1.6))  # low entropy -> bot
        + 1.3 * min(1.0, f["size_modal_frac"])
        + 0.9 * (1.0 - min(1.0, f["size_cv_bb"] / 0.6))         # consistent sizing
        + 0.7 * min(1.0, f["aggression_factor"] / 1.5)
        + 0.6 * (1.0 - min(1.0, f["aggro_frac_cv_across_hands"] / 0.8))
        + 0.5 * min(1.0, f["frac_reach_turn_plus"])
        - 1.4
    )
    return 1.0 / (1.0 + math.exp(-z))


class TrainedMiner(BaseMinerNeuron):
    """Poker44 miner backed by a trained model with an FPR-safe scoring head."""

    def __init__(self, config=None):
        super(TrainedMiner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self.model = TrainedModel.load(_DEFAULT_MODEL_PATH)
        self.fallback_head = ScoringHead(t_star=0.62, sharpness=12.0)
        mode = "trained-gbdt" if self.model else "heuristic-fallback"
        bt.logging.info(f"🧠 Poker44 TrainedMiner started (mode={mode})")

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44" / "miner_model" / "features.py",
                repo_root / "poker44" / "miner_model" / "scoring_head.py",
            ],
            defaults={
                "model_name": "poker44-gbdt-behavioural",
                "model_version": "2",
                "framework": "lightgbm" if self.model else "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "open_source": True,
                "inference_mode": "remote",
                "notes": f"GBDT behavioural bot-detector ({mode}) with FPR-safe scoring head.",
                "training_data_statement": (
                    "Trained on the public Poker44 benchmark "
                    "(api.poker44.net/api/v1/benchmark) miner-visible chunk payloads "
                    "with chunk-level human/bot labels. No validator-only data used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data; "
                    "only the public benchmark is used."
                ),
                "data_attestation": "public-benchmark-only",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest status={self.manifest_compliance['status']} "
            f"missing={self.manifest_compliance['missing_fields']} "
            f"digest={self.manifest_digest}"
        )

    def _score_chunks(self, chunks: List[list]) -> List[float]:
        feats = [extract_chunk_features(chunk) for chunk in chunks]
        if self.model is not None:
            probs = self.model.proba(feats)
            return self.model.head.score_many(probs)
        probs = [_heuristic_proba(f) for f in feats]
        return [round(shape_risk_score(p, self.fallback_head.t_star, self.fallback_head.sharpness), 6) for p in probs]

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        scores = self._score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | flagged={sum(synapse.predictions)} "
            f"| score[min/max]={min(scores, default=0):.3f}/{max(scores, default=0):.3f}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with TrainedMiner() as miner:
        bt.logging.info("TrainedMiner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
