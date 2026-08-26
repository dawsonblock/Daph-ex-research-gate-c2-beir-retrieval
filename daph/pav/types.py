"""PAV type definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PAVPrediction:
    """A single PAV prediction for one action.

    Attributes:
        action: The action being scored
        mean: Predicted step progress (PAV(s,a,s') ≈ V(s') - V(s) - C(a))
        std: Uncertainty estimate (0 for deterministic structural PAV)
        structural_score: Score from StructuralPAV (Progress V1), or None
        model_score: Score from learned model, or None
    """
    action: str
    mean: float
    std: float
    structural_score: float | None
    model_score: float | None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "structural_score": round(self.structural_score, 6) if self.structural_score is not None else None,
            "model_score": round(self.model_score, 6) if self.model_score is not None else None,
        }


@dataclass(frozen=True)
class PAVScoreResult:
    """Result of scoring a set of actions with PAV.

    Attributes:
        predictions: Per-action PAVPrediction
        selected: Actions that pass the PAV threshold (preferred set)
        abstained: Whether PAV declined to distinguish (return full set)
        config_sha: Hash of the PAV configuration used
        model_sha: Hash of the model used (or "structural" for PAV_B0)
        receipt: Provenance receipt data
    """
    predictions: tuple[PAVPrediction, ...]
    selected: tuple[str, ...]
    abstained: bool
    config_sha: str
    model_sha: str
    receipt: dict

    def as_dict(self) -> dict:
        return {
            "predictions": [p.as_dict() for p in self.predictions],
            "selected": list(self.selected),
            "abstained": self.abstained,
            "config_sha": self.config_sha,
            "model_sha": self.model_sha,
            "receipt": self.receipt,
        }


class PAVScorer(Protocol):
    """Protocol for PAV scorers.

    A PAV scorer evaluates candidate actions at a decision point.
    It sees the state and candidate actions but does NOT execute them.
    """

    def score_actions(
        self,
        checkpoint: Any,
        actions: tuple[str, ...],
        *,
        search_context: dict | None = None,
    ) -> PAVScoreResult:
        """Score candidate actions and return predictions + preferred set.

        Args:
            checkpoint: StateCheckpoint at the decision point
            actions: Candidate actions (typically the Q epsilon near-optimal set)
            search_context: Optional context from search (e.g. simulated branches)

        Returns:
            PAVScoreResult with per-action predictions and a preferred subset.
        """
        ...
