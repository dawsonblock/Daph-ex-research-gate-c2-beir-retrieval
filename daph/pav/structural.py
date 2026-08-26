"""StructuralPAV: Deterministic PAV baseline wrapping PROGRESS_RULE_V1.

This is PAV_B0. It does not learn — it delegates to the frozen deterministic
progress function. No retraining, no modification.

The structural PAV simulates each candidate action via the EvidenceExecutor,
computes Progress(s,a,s'), and returns the score. Actions that produce
no state change and incur cost get negative scores.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import EvidenceTask
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import StateCheckpoint
from daph.intervention.restore import restore_runtime
from daph.progress.progress_rule_v1 import compute_progress
from daph.pav.types import PAVPrediction, PAVScoreResult


# Frozen configuration — matches PROGRESS_RULE_V1 weights
_STRUCTURAL_PAV_CONFIG = {
    "epsilon_p": 0.05,
    "cost_normalization": 100.0,
    "weights": {
        "verification_coverage": 1.0,
        "evidence_novelty": 1.0,
        "hypothesis_resolution": 1.5,
        "terminal_readiness": 2.0,
        "contradiction_resolution": 0.5,
    },
}


def _config_sha() -> str:
    return hashlib.sha256(
        json.dumps(_STRUCTURAL_PAV_CONFIG, sort_keys=True).encode()
    ).hexdigest()


class StructuralPAV:
    """PAV_B0: deterministic structural progress as PAV.

    Wraps the frozen PROGRESS_RULE_V1. For each candidate action:
    1. Restore the checkpoint to get the pre-action runtime
    2. Execute the action via EvidenceExecutor (deterministic, no LLM)
    3. Compute Progress(s, a, s') using the frozen progress function
    4. Return the progress score as the PAV prediction

    This is the correct baseline. Do not retrain it.
    """

    def __init__(
        self,
        task: EvidenceTask,
        utility: MetareasoningUtility,
        epsilon_p: float = 0.05,
    ):
        self.task = task
        self.utility = utility
        self.epsilon_p = epsilon_p
        self._executor = EvidenceExecutor()
        self._config_sha = _config_sha()
        self._model_sha = "structural_pav_b0"

    def score_actions(
        self,
        checkpoint: StateCheckpoint,
        actions: tuple[str, ...],
        *,
        search_context: dict | None = None,
    ) -> PAVScoreResult:
        """Score candidate actions using deterministic structural progress.

        For each action:
        - Restore checkpoint -> runtime_before
        - Execute action -> runtime_after
        - Compute Progress(runtime_before, action_result, utility)
        - PAV mean = progress score

        For VERIFY, uses the first valid verify target if available.

        Returns preferred set: actions within epsilon_p of max progress.
        If all actions are within epsilon_p, abstains (returns full set).
        """
        predictions = []
        timing_start = time.time()

        for action_str in actions:
            action = DecisionAction(action_str)

            # Determine verify target if needed
            target_eid = None
            if action is DecisionAction.VERIFY:
                runtime_for_check = restore_runtime(checkpoint, self.task)
                valid_targets = valid_verify_targets(runtime_for_check)
                if valid_targets:
                    target_eid = valid_targets[0]
                else:
                    # No valid verify target — negative score
                    predictions.append(PAVPrediction(
                        action=action_str,
                        mean=-0.2,
                        std=0.0,
                        structural_score=-0.2,
                        model_score=None,
                    ))
                    continue

            try:
                runtime_before = restore_runtime(checkpoint, self.task)
                exec_result = self._executor.execute(
                    runtime_before, action, target_evidence_id=target_eid,
                )
                progress = compute_progress(
                    runtime_before, exec_result, self.utility,
                )
                score = progress.progress
            except Exception:
                score = -0.2

            predictions.append(PAVPrediction(
                action=action_str,
                mean=score,
                std=0.0,  # Deterministic — no uncertainty
                structural_score=score,
                model_score=None,
            ))

        # Compute preferred set
        if not predictions:
            return PAVScoreResult(
                predictions=(),
                selected=actions,
                abstained=True,
                config_sha=self._config_sha,
                model_sha=self._model_sha,
                receipt={"error": "no predictions", "timing_ms": 0},
            )

        scores = {p.action: p.mean for p in predictions}
        max_score = max(scores.values())
        min_score = min(scores.values())
        gap = max_score - min_score

        if gap < self.epsilon_p:
            # Cannot distinguish — abstain, return full set
            selected = tuple(actions)
            abstained = True
        else:
            selected = tuple(
                sorted(a for a, s in scores.items() if s >= max_score - self.epsilon_p)
            )
            abstained = False

        timing_ms = (time.time() - timing_start) * 1000

        receipt = {
            "scorer": "StructuralPAV",
            "checkpoint_id": checkpoint.checkpoint_id,
            "actions": list(actions),
            "scores": {p.action: p.mean for p in predictions},
            "selected": list(selected),
            "abstained": abstained,
            "epsilon_p": self.epsilon_p,
            "gap": gap,
            "timing_ms": round(timing_ms, 2),
        }

        return PAVScoreResult(
            predictions=tuple(predictions),
            selected=selected,
            abstained=abstained,
            config_sha=self._config_sha,
            model_sha=self._model_sha,
            receipt=receipt,
        )

    @property
    def config_sha(self) -> str:
        return self._config_sha

    @property
    def model_sha(self) -> str:
        return self._model_sha
