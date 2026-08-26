"""PAV scorer: unified scoring interface.

Wraps StructuralPAV, LearnedPAV, or EnsemblePAV behind a common interface.
Returns PAVScoreResult for use by the executive controller.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from daph.intervention.checkpoint import StateCheckpoint
from daph.intervention.restore import restore_runtime
from daph.pav.types import PAVPrediction, PAVScoreResult, PAVScorer
from daph.pav.structural import StructuralPAV
from daph.pav.model import LearnedPAV, extract_pav_features
from daph.pav.ensemble import EnsemblePAV


class StructuralPAVScorer:
    """Scorer wrapping StructuralPAV (PAV_B0)."""

    def __init__(self, structural: StructuralPAV):
        self.structural = structural

    def score_actions(
        self,
        checkpoint: StateCheckpoint,
        actions: tuple[str, ...],
        *,
        search_context: dict | None = None,
    ) -> PAVScoreResult:
        return self.structural.score_actions(checkpoint, actions, search_context=search_context)


class LearnedPAVScorer:
    """Scorer wrapping LearnedPAV (PAV_V1).

    Uses learned model for prediction. Does NOT execute actions.
    Falls back to structural score if model fails.
    """

    def __init__(
        self,
        learned: LearnedPAV,
        structural: StructuralPAV,
        epsilon_p: float = 0.05,
    ):
        self.learned = learned
        self.structural = structural
        self.epsilon_p = epsilon_p

    def score_actions(
        self,
        checkpoint: StateCheckpoint,
        actions: tuple[str, ...],
        *,
        search_context: dict | None = None,
    ) -> PAVScoreResult:
        timing_start = time.time()
        predictions = []

        for action_str in actions:
            try:
                learned_score = self.learned.predict(
                    checkpoint.state_features, action_str,
                )
            except Exception:
                learned_score = None

            # Also compute structural score for comparison
            struct_result = self.structural.score_actions(
                checkpoint, (action_str,), search_context=search_context,
            )
            struct_score = struct_result.predictions[0].mean if struct_result.predictions else None

            mean = learned_score if learned_score is not None else struct_score or -0.2
            predictions.append(PAVPrediction(
                action=action_str,
                mean=mean,
                std=0.0,
                structural_score=struct_score,
                model_score=learned_score,
            ))

        if not predictions:
            return PAVScoreResult(
                predictions=(),
                selected=actions,
                abstained=True,
                config_sha=self.structural.config_sha,
                model_sha="learned_pav_v1",
                receipt={"error": "no predictions"},
            )

        scores = {p.action: p.mean for p in predictions}
        max_score = max(scores.values())
        min_score = min(scores.values())
        gap = max_score - min_score

        if gap < self.epsilon_p:
            selected = tuple(actions)
            abstained = True
        else:
            selected = tuple(
                sorted(a for a, s in scores.items() if s >= max_score - self.epsilon_p)
            )
            abstained = False

        timing_ms = (time.time() - timing_start) * 1000
        receipt = {
            "scorer": "LearnedPAVScorer",
            "checkpoint_id": checkpoint.checkpoint_id,
            "actions": list(actions),
            "scores": {p.action: p.mean for p in predictions},
            "selected": list(selected),
            "abstained": abstained,
            "timing_ms": round(timing_ms, 2),
        }

        return PAVScoreResult(
            predictions=tuple(predictions),
            selected=selected,
            abstained=abstained,
            config_sha=self.structural.config_sha,
            model_sha="learned_pav_v1",
            receipt=receipt,
        )


class EnsemblePAVScorer:
    """Scorer wrapping EnsemblePAV (PAV_V2).

    Provides uncertainty estimates via bootstrap ensemble.
    """

    def __init__(
        self,
        ensemble: EnsemblePAV,
        structural: StructuralPAV,
        epsilon_p: float = 0.05,
    ):
        self.ensemble = ensemble
        self.structural = structural
        self.epsilon_p = epsilon_p

    def score_actions(
        self,
        checkpoint: StateCheckpoint,
        actions: tuple[str, ...],
        *,
        search_context: dict | None = None,
    ) -> PAVScoreResult:
        timing_start = time.time()
        predictions = []

        for action_str in actions:
            try:
                mean, std = self.ensemble.predict_with_uncertainty(
                    checkpoint.state_features, action_str,
                )
            except Exception:
                mean, std = -0.2, 0.0

            predictions.append(PAVPrediction(
                action=action_str,
                mean=mean,
                std=std,
                structural_score=None,
                model_score=mean,
            ))

        if not predictions:
            return PAVScoreResult(
                predictions=(),
                selected=actions,
                abstained=True,
                config_sha=self.structural.config_sha,
                model_sha="ensemble_pav_v2",
                receipt={"error": "no predictions"},
            )

        scores = {p.action: p.mean for p in predictions}
        max_score = max(scores.values())
        min_score = min(scores.values())
        gap = max_score - min_score

        if gap < self.epsilon_p:
            selected = tuple(actions)
            abstained = True
        else:
            selected = tuple(
                sorted(a for a, s in scores.items() if s >= max_score - self.epsilon_p)
            )
            abstained = False

        timing_ms = (time.time() - timing_start) * 1000
        receipt = {
            "scorer": "EnsemblePAVScorer",
            "checkpoint_id": checkpoint.checkpoint_id,
            "actions": list(actions),
            "scores": {p.action: p.mean for p in predictions},
            "uncertainties": {p.action: p.std for p in predictions},
            "selected": list(selected),
            "abstained": abstained,
            "timing_ms": round(timing_ms, 2),
        }

        return PAVScoreResult(
            predictions=tuple(predictions),
            selected=selected,
            abstained=abstained,
            config_sha=self.structural.config_sha,
            model_sha="ensemble_pav_v2",
            receipt=receipt,
        )


def make_pav_scorer(
    level: str = "B0",
    task: Any = None,
    utility: Any = None,
    learned_model: LearnedPAV | None = None,
    ensemble_model: EnsemblePAV | None = None,
    epsilon_p: float = 0.05,
) -> PAVScorer:
    """Factory for PAV scorers.

    Args:
        level: "B0" (structural), "V1" (learned GBT), or "V2" (ensemble)
        task: EvidenceTask (required for B0 and V1)
        utility: MetareasoningUtility (required for B0 and V1)
        learned_model: Pre-trained LearnedPAV (required for V1)
        ensemble_model: Pre-trained EnsemblePAV (required for V2)
        epsilon_p: PAV threshold for preferred set

    Returns:
        A PAVScorer instance.
    """
    if level == "B0":
        if task is None or utility is None:
            raise ValueError("StructuralPAV requires task and utility")
        structural = StructuralPAV(task, utility, epsilon_p)
        return StructuralPAVScorer(structural)

    elif level == "V1":
        if task is None or utility is None:
            raise ValueError("LearnedPAVScorer requires task and utility")
        if learned_model is None:
            raise ValueError("LearnedPAVScorer requires learned_model")
        structural = StructuralPAV(task, utility, epsilon_p)
        return LearnedPAVScorer(learned_model, structural, epsilon_p)

    elif level == "V2":
        if task is None or utility is None:
            raise ValueError("EnsemblePAVScorer requires task and utility")
        if ensemble_model is None:
            raise ValueError("EnsemblePAVScorer requires ensemble_model")
        structural = StructuralPAV(task, utility, epsilon_p)
        return EnsemblePAVScorer(ensemble_model, structural, epsilon_p)

    else:
        raise ValueError(f"Unknown PAV level: {level}")
