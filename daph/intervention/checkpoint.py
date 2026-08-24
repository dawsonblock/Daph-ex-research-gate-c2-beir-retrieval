"""State checkpointing for causal intervention.

A checkpoint captures the complete observable state at a decision point,
allowing the same state to be restored and replayed with different forced actions.

Critical invariant:
    restore(checkpoint) must reconstruct the same visible state.
    hash(state) == checkpoint.state_sha256  # 100% required
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceItem, EvidenceRuntime, EvidenceSnapshot, EvidenceTask, EvidenceHypothesis,
    initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState


@dataclass(frozen=True)
class StateCheckpoint:
    """A frozen snapshot of observable state at a decision point.

    Attributes:
        checkpoint_id: SHA256 hash of the checkpoint content
        task_id: The task this checkpoint belongs to
        step: The step number within the trajectory
        phase: The epistemic phase at this checkpoint
        hypotheses: Hypothesis propositions and IDs
        evidence: Visible evidence items with their states
        state_features: Structured features derived from observable state
        resources: Resource state at checkpoint
        legal_actions: Actions that are legal from this state
        state_sha256: Hash of the core state content (for restore verification)
        prior_actions: Actions taken before this checkpoint
        prior_outcomes: Outcomes of prior actions
    """
    checkpoint_id: str
    task_id: str
    step: int
    phase: str
    hypotheses: tuple[dict, ...]
    evidence: tuple[dict, ...]
    state_features: dict
    resources: dict
    legal_actions: tuple[str, ...]
    state_sha256: str
    prior_actions: tuple[str, ...]
    prior_outcomes: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "step": self.step,
            "phase": self.phase,
            "hypotheses": list(self.hypotheses),
            "evidence": list(self.evidence),
            "state_features": self.state_features,
            "resources": self.resources,
            "legal_actions": list(self.legal_actions),
            "state_sha256": self.state_sha256,
            "prior_actions": list(self.prior_actions),
            "prior_outcomes": list(self.prior_outcomes),
        }


def _evidence_to_dict(ev: EvidenceItem) -> dict:
    return {
        "evidence_id": ev.evidence_id,
        "proposition": ev.proposition,
        "source_class": ev.source_class,
        "supports": list(ev.supports),
        "contradicts": list(ev.contradicts),
        "verification_state": ev.verification_state.value,
        "temporal_status": ev.temporal_status.value,
        "retrieved": ev.retrieved,
        "verify_result": ev.verify_result,
    }


def _dict_to_evidence(d: dict) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=d["evidence_id"],
        proposition=d["proposition"],
        source_class=d["source_class"],
        supports=tuple(d["supports"]),
        contradicts=tuple(d["contradicts"]),
        verification_state=VerificationState(d["verification_state"]),
        temporal_status=TemporalStatus(d["temporal_status"]),
        retrieved=d["retrieved"],
        verify_result=d["verify_result"],
    )


def _hypothesis_to_dict(h: EvidenceHypothesis) -> dict:
    return {
        "hypothesis_id": h.hypothesis_id,
        "proposition": h.proposition,
        "answer_action": h.answer_action.value,
        "answer_payload": h.answer_payload,
    }


def compute_state_features(
    runtime: EvidenceRuntime,
    prior_actions: tuple[str, ...] = (),
) -> dict:
    """Compute structured features from observable state.

    All features are functions of observable state only — no future data.
    """
    visible = runtime.visible_evidence
    hidden = runtime.hidden_evidence

    n_live = 0
    n_eliminated = 0
    n_untested = 0
    n_verified = 0
    n_supporting = 0
    n_contradicting = 0
    n_stale = 0

    for ev in visible:
        if ev.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED):
            n_verified += 1
        if ev.verification_state == VerificationState.SUFFICIENT and ev.supports:
            n_supporting += 1
        if ev.verification_state == VerificationState.FALSIFIED and ev.supports:
            n_contradicting += 1
        if ev.temporal_status == TemporalStatus.STALE:
            n_stale += 1

    # Hypothesis viability
    for h in runtime.task.hypotheses:
        has_support = False
        has_contradiction = False
        for ev in visible:
            if ev.verification_state == VerificationState.SUFFICIENT:
                if ev.temporal_status == TemporalStatus.STALE:
                    continue
                if h.hypothesis_id in ev.supports:
                    has_support = True
                if h.hypothesis_id in ev.contradicts:
                    has_contradiction = True
        if has_contradiction:
            n_eliminated += 1
        elif has_support:
            n_live += 1
        else:
            n_untested += 1

    rs = runtime.resources.as_dict()

    # Action history features
    last_action = prior_actions[-1] if prior_actions else None
    retrieval_count = sum(1 for a in prior_actions if a == "RETRIEVE")
    search_count = sum(1 for a in prior_actions if a == "SEARCH_MORE")
    verify_count = sum(1 for a in prior_actions if a == "VERIFY")

    # Same-action run length
    same_action_run = 0
    if prior_actions:
        last = prior_actions[-1]
        for a in reversed(prior_actions):
            if a == last:
                same_action_run += 1
            else:
                break

    return {
        "n_live": n_live,
        "n_eliminated": n_eliminated,
        "n_untested": n_untested,
        "n_total_hypotheses": len(runtime.task.hypotheses),
        "n_visible_evidence": len(visible),
        "n_hidden_evidence": len(hidden),
        "n_verified": n_verified,
        "n_supporting": n_supporting,
        "n_contradicting": n_contradicting,
        "n_stale": n_stale,
        "retrieval_remaining": rs.get("retrieval_calls_remaining", 0),
        "search_remaining": rs.get("search_calls_remaining", 0),
        "verify_remaining": rs.get("verification_calls_remaining", 0),
        "steps_remaining": rs.get("executive_steps_remaining", 0),
        "can_retrieve": rs.get("retrieval_calls_remaining", 0) > 0 and len(hidden) > 0,
        "can_search": rs.get("search_calls_remaining", 0) > 0,
        "can_verify": rs.get("verification_calls_remaining", 0) > 0 and any(
            ev.retrieved and ev.verification_state == VerificationState.UNVERIFIED
            for ev in visible
        ),
        "searched": runtime.searched,
        "reasoning_complete": runtime.reasoning_complete,
        "last_action": last_action,
        "same_action_run_length": same_action_run,
        "retrieval_count": retrieval_count,
        "search_count": search_count,
        "verify_count": verify_count,
    }


def compute_legal_actions(runtime: EvidenceRuntime) -> tuple[str, ...]:
    """Compute the set of legal actions from this state."""
    rs = runtime.resources.as_dict()
    visible = runtime.visible_evidence
    has_unverified = any(
        ev.retrieved and ev.verification_state == VerificationState.UNVERIFIED
        for ev in visible
    )
    has_hidden = len(runtime.hidden_evidence) > 0

    actions = []
    if rs.get("retrieval_calls_remaining", 0) > 0 and has_hidden:
        actions.append("RETRIEVE")
    if rs.get("search_calls_remaining", 0) > 0:
        actions.append("SEARCH_MORE")
    if rs.get("verification_calls_remaining", 0) > 0 and has_unverified:
        actions.append("VERIFY")
    actions.append("REASON_MORE")
    actions.append("ANSWER")
    actions.append("DEFER")
    return tuple(actions)


def _compute_state_sha256(
    task_id: str,
    step: int,
    evidence: tuple[dict, ...],
    resources: dict,
    prior_actions: tuple[str, ...],
) -> str:
    """Compute deterministic SHA256 of the core state content."""
    content = json.dumps({
        "task_id": task_id,
        "step": step,
        "evidence": evidence,
        "resources": resources,
        "prior_actions": list(prior_actions),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def _compute_checkpoint_id(
    task_id: str,
    step: int,
    state_sha256: str,
) -> str:
    """Compute the checkpoint ID from task, step, and state hash."""
    content = f"{task_id}:{step}:{state_sha256}"
    return hashlib.sha256(content.encode()).hexdigest()


def create_checkpoint(
    runtime: EvidenceRuntime,
    step: int,
    phase: str = "UNKNOWN",
    prior_actions: tuple[str, ...] = (),
    prior_outcomes: tuple[str, ...] = (),
) -> StateCheckpoint:
    """Create a checkpoint from the current runtime state.

    Args:
        runtime: The evidence runtime at the decision point
        step: The step number within the trajectory
        phase: The epistemic phase (from the phase classifier)
        prior_actions: Actions taken before this checkpoint
        prior_outcomes: Outcomes of prior actions

    Returns:
        A StateCheckpoint capturing the complete observable state.
    """
    visible = runtime.visible_evidence
    evidence_dicts = tuple(_evidence_to_dict(ev) for ev in visible)
    hyp_dicts = tuple(_hypothesis_to_dict(h) for h in runtime.task.hypotheses)
    resources_dict = runtime.resources.as_dict()
    features = compute_state_features(runtime, prior_actions)
    legal = compute_legal_actions(runtime)

    state_sha = _compute_state_sha256(
        runtime.task.task_id, step, evidence_dicts, resources_dict, prior_actions,
    )
    checkpoint_id = _compute_checkpoint_id(
        runtime.task.task_id, step, state_sha,
    )

    return StateCheckpoint(
        checkpoint_id=checkpoint_id,
        task_id=runtime.task.task_id,
        step=step,
        phase=phase,
        hypotheses=hyp_dicts,
        evidence=evidence_dicts,
        state_features=features,
        resources=resources_dict,
        legal_actions=legal,
        state_sha256=state_sha,
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
    )
