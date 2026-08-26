"""DAPH PAV (Process-Advantage-Verifier) package.

PAV estimates step-level progress: did this action actually improve the
epistemic state enough to justify its cost? This is separate from Q_CAUSAL_V1,
which estimates long-horizon recoverable outcome.

Model ladder:
  PAV_B0 = StructuralPAV (wraps frozen PROGRESS_RULE_V1)
  PAV_V1 = GradientBoostingRegressor (only if B0 is insufficient)
  PAV_V2 = bootstrap ensemble (only if V1 is insufficient)
"""
from daph.pav.types import PAVPrediction, PAVScoreResult
from daph.pav.structural import StructuralPAV
from daph.pav.scorer import PAVScorer

__all__ = [
    "PAVPrediction",
    "PAVScoreResult",
    "StructuralPAV",
    "PAVScorer",
]
