"""Trained-miner model package for Poker44.

Modules:
  features      - deterministic chunk -> feature-vector extractor (miner-visible payload).
  scoring_head  - monotonic score shaping that positions the 0.5 cutoff at a
                  high-precision operating point so the validator's FPR cliff is
                  protected without hurting average-precision (AP) ranking.
"""

from poker44.miner_model.features import FEATURE_NAMES, extract_chunk_features
from poker44.miner_model.scoring_head import ScoringHead, shape_risk_score

__all__ = [
    "FEATURE_NAMES",
    "extract_chunk_features",
    "ScoringHead",
    "shape_risk_score",
]
