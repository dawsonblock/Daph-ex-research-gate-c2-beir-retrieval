"""DAPH PAV + Selective Search executive controller.

Challenger to frozen DAPH_PROGRESS_EXECUTIVE_V1.

Pipeline:
    legal = legal_actions(state)
    q = q_model.predict_actions(state, legal)
    q_set = epsilon_set(q, epsilon=3.0)
    pav = pav_scorer.score_actions(checkpoint, q_set)
    vp_guidance = refine_with_progress(q_set, pav)
    decision = search_trigger(...)
    if not decision.search:
        return vp_guidance
    result = planner.plan(checkpoint, vp_guidance.actions)
    if result.abstained:
        return vp_guidance
    return search_supported_guidance(result)

Hard invariants:
    - VP output unchanged when search disabled
    - Abstention output exactly equals VP
    - Illegal actions never enter tree
    - Node/call/time budgets never exceeded
    - Branch checkpoint restores to identical SHA
    - No future/hidden labels in packet
    - Same seed/state/config creates same tree
    - Search failure never fails the trajectory
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark.schema import EvidenceTask
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility

from daph.intervention.checkpoint import StateCheckpoint, compute_state_features
from daph.pav.types import PAVScoreResult, PAVScorer
from daph.search.types import SearchConfig, SearchResult, SearchTriggerResult
from daph.search.trigger import decide_search
from daph.search.planner import SearchPlanner


@dataclass(frozen=True)
class ExecutiveGuidance:
    """Model-facing executive guidance.

    Categorical output only — no raw Q values or PAV scores.
    """
    near_optimal_actions: tuple[str, ...]
    lower_value_actions: tuple[str, ...]
    guidance_confidence: str  # "clear", "ambiguous", "high"
    mode: str  # "VP", "SEARCH_SUPPORTED", "AMBIGUOUS"
    epistemic_phase: str
    search_triggered: bool
    search_abstained: bool
    search_winner: str | None
    search_reasons: tuple[str, ...]
    pav_selected: tuple[str, ...]
    pav_abstained: bool
    receipt: dict

    def as_packet(self) -> dict:
        """Build the model-facing packet (categorical only)."""
        if self.mode == "SEARCH_SUPPORTED" and self.search_winner:
            # Search found a winner — recommend it with alternatives
            recommended = [self.search_winner]
            alternatives = [a for a in self.near_optimal_actions
                           if a != self.search_winner]
            return {
                "metacognitive_guidance": {
                    "mode": "SEARCH_SUPPORTED",
                    "recommended_actions": recommended,
                    "alternatives": alternatives,
                    "confidence": "HIGH",
                }
            }
        elif self.guidance_confidence == "clear":
            return {
                "metacognitive_guidance": {
                    "mode": "VP",
                    "recommended_actions": list(self.near_optimal_actions),
                    "alternatives": list(self.lower_value_actions),
                    "confidence": "HIGH" if len(self.near_optimal_actions) == 1 else "MEDIUM",
                }
            }
        else:
            return {
                "metacognitive_guidance": {
                    "mode": "AMBIGUOUS",
                    "recommended_actions": list(self.near_optimal_actions),
                    "alternatives": [],
                    "confidence": "LOW",
                }
            }


class PAVSearchController:
    """Challenger executive: VP + PAV + Selective Search.

    Frozen DAPH_PROGRESS_EXECUTIVE_V1 is the control.
    This controller builds on top of it with optional PAV and search.
    """

    def __init__(
        self,
        task: EvidenceTask,
        utility: MetareasoningUtility,
        q_model: Any,  # QCAUSALModel from the runner
        pav_scorer: PAVScorer,
        search_config: SearchConfig | None = None,
        epsilon_q: float = 3.0,
        enable_search: bool = True,
        enable_pav: bool = True,
    ):
        self.task = task
        self.utility = utility
        self.q_model = q_model
        self.pav_scorer = pav_scorer
        self.search_config = search_config or SearchConfig()
        self.epsilon_q = epsilon_q
        self.enable_search = enable_search
        self.enable_pav = enable_pav

        self._planner = SearchPlanner(task, utility, self.search_config)

    def compute_guidance(
        self,
        checkpoint: StateCheckpoint,
        legal_actions: list[str],
        state_features: dict,
        phase: str,
    ) -> ExecutiveGuidance:
        """Compute executive guidance at a decision point.

        This is the main entry point. It:
        1. Predicts Q values for all legal actions
        2. Computes the Q epsilon near-optimal set
        3. Optionally runs PAV to refine the set
        4. Optionally triggers selective search
        5. Returns categorical guidance

        Args:
            checkpoint: State checkpoint at the decision point
            legal_actions: Legal actions at this state
            state_features: Controller-visible state features
            phase: Epistemic phase

        Returns:
            ExecutiveGuidance with categorical model-facing output.
        """
        # Step 1: Q values
        q_values = self.q_model.predict_q(state_features, legal_actions)

        # Step 2: Q epsilon near-optimal set
        q_max = max(q_values.values()) if q_values else 0.0
        near_optimal = sorted(
            a for a, q in q_values.items() if q >= q_max - self.epsilon_q
        )
        lower_value = sorted(
            a for a, q in q_values.items() if q < q_max - self.epsilon_q
        )

        # Step 3: PAV refinement (optional)
        pav_result: PAVScoreResult | None = None
        pav_selected = tuple(near_optimal)
        pav_abstained = True

        if self.enable_pav and near_optimal:
            pav_result = self.pav_scorer.score_actions(
                checkpoint, tuple(near_optimal),
            )
            if not pav_result.abstained:
                pav_selected = pav_result.selected
                pav_abstained = False
            else:
                pav_selected = tuple(near_optimal)
                pav_abstained = True

        # VP guidance (what frozen VP would produce)
        vp_confidence = "clear" if len(pav_selected) == 1 else "ambiguous"

        # Step 4: Search trigger (optional)
        search_triggered = False
        search_abstained = False
        search_winner = None
        search_reasons: tuple[str, ...] = ()
        search_result: SearchResult | None = None

        if self.enable_search and len(near_optimal) >= 2:
            trigger_result = decide_search(
                state_features=state_features,
                near_optimal_actions=tuple(near_optimal),
                pav_selected=pav_selected if not pav_abstained else None,
                pav_abstained=pav_abstained,
                q_values=q_values,
                config=self.search_config,
            )

            search_triggered = trigger_result.should_search
            search_reasons = trigger_result.reasons

            if search_triggered:
                # Run search over the near-optimal set
                search_result = self._planner.plan(
                    checkpoint=checkpoint,
                    candidate_actions=tuple(near_optimal),
                    q_values=q_values,
                    trigger_reasons=trigger_result.reasons,
                )

                search_abstained = search_result.abstained
                if not search_result.abstained:
                    search_winner = search_result.winner

        # Step 5: Build guidance
        if search_triggered and not search_abstained and search_winner:
            mode = "SEARCH_SUPPORTED"
            confidence = "high"
            # The search winner is the recommended action
            final_actions = tuple(near_optimal)  # Keep full set for model
        else:
            mode = "VP"
            confidence = vp_confidence
            final_actions = pav_selected

        receipt = {
            "controller": "PAVSearchController",
            "checkpoint_id": checkpoint.checkpoint_id,
            "legal_actions": legal_actions,
            "q_values": {a: round(q, 4) for a, q in q_values.items()},
            "near_optimal": near_optimal,
            "lower_value": lower_value,
            "pav_selected": list(pav_selected),
            "pav_abstained": pav_abstained,
            "search_triggered": search_triggered,
            "search_abstained": search_abstained,
            "search_winner": search_winner,
            "search_reasons": list(search_reasons),
            "mode": mode,
            "confidence": confidence,
        }

        if pav_result is not None:
            receipt["pav_receipt"] = pav_result.receipt

        if search_result is not None:
            receipt["search_receipt"] = search_result.receipt

        return ExecutiveGuidance(
            near_optimal_actions=final_actions,
            lower_value_actions=tuple(lower_value),
            guidance_confidence=confidence,
            mode=mode,
            epistemic_phase=phase,
            search_triggered=search_triggered,
            search_abstained=search_abstained,
            search_winner=search_winner,
            search_reasons=search_reasons,
            pav_selected=pav_selected,
            pav_abstained=pav_abstained,
            receipt=receipt,
        )
