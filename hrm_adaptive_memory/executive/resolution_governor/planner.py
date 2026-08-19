"""Deterministic resolution governor planner.

Derives ResolutionAssistanceFrame and ResolutionContext strictly from
controller-visible state. No evaluator labels, oracle Q-values, gold
answers, or future outcomes.

The planner is deterministic: same state -> same frame.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.governor.state import GovernorState
from hrm_adaptive_memory.executive.governor.bottlenecks import (
    DecisionBottleneck, detect_bottlenecks,
)

from .schema import (
    RESOLUTION_SCHEMA, RESOLUTION_VERSION,
    Hypothesis, EvidenceAssessment, Discriminator, ResolutionStep,
    AnswerCondition, SearchSpecification, ResolutionAssistanceFrame,
    ResolutionContext, HypothesisUpdate, ResolutionReceipt,
    HYPOTHESIS_SUPPORTED, HYPOTHESIS_WEAK, HYPOTHESIS_CONTRADICTED,
    HYPOTHESIS_UNRESOLVED, HYPOTHESIS_ELIMINATED,
    VERIFICATION_SUFFICIENT, VERIFICATION_MISSING, VERIFICATION_UNVERIFIED,
    VERIFICATION_FALSIFIED, VERIFICATION_STALE,
    TEMPORAL_CURRENT, TEMPORAL_STALE, TEMPORAL_UNKNOWN,
    UPDATE_KEEP, UPDATE_DOWNWEIGHT, UPDATE_ELIMINATE,
)
from .hypotheses import build_hypotheses, build_evidence_map
from .decision_rule import (
    build_discriminators, build_answer_conditions, build_defer_condition,
)


def _state_sha256(
    task_id: str,
    task_summary: str,
    prior_actions: tuple[str, ...],
    prior_outcomes: tuple[str, ...],
    remaining_steps: int,
    snapshot: CognitiveStateSnapshot | None,
) -> str:
    """Compute a deterministic hash of the controller-visible state."""
    state = {
        "task_id": task_id,
        "task_summary": task_summary,
        "prior_actions": list(prior_actions),
        "prior_outcomes": list(prior_outcomes),
        "remaining_steps": remaining_steps,
    }
    if snapshot is not None:
        state["temporal_status"] = snapshot.temporal_status.value
        state["memories"] = [
            {"id": m.memory_id, "ver": m.verification_state.value,
             "temp": m.temporal_status.value, "conf": m.conflict_state}
            for m in snapshot.relevant_memories
        ]
        state["conflicts"] = [
            {"id": c.conflict_id, "rel": c.relation, "status": c.status}
            for c in snapshot.unresolved_conflicts
        ]
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_search_specification(
    discriminator: Discriminator,
    task_summary: str,
    snapshot: CognitiveStateSnapshot | None,
) -> SearchSpecification:
    """Build an executable search specification from a discriminator."""
    temporal = None
    if snapshot:
        if snapshot.temporal_status == TemporalStatus.STALE:
            temporal = "current"
        elif snapshot.temporal_status == TemporalStatus.UNKNOWN:
            temporal = "current"

    return SearchSpecification(
        subject=discriminator.evidence_target[:200],
        required_property=f"directly addresses: {discriminator.question[:150]}",
        temporal_constraint=temporal,
        source_constraint="primary source preferred" if snapshot and snapshot.provenance_summaries else None,
        must_confirm=(discriminator.if_true_supports,),
        must_disambiguate=(discriminator.if_true_supports, discriminator.if_false_supports),
        reject_if=("undated evidence", "secondary summaries", "evidence that does not distinguish hypotheses"),
    )


def _build_execution_plan(
    recommended_action: str,
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceAssessment, ...],
    discriminators: tuple[Discriminator, ...],
    answer_conditions: tuple[AnswerCondition, ...],
    max_additional_actions: int,
) -> tuple[ResolutionStep, ...]:
    """Build a bounded execution plan with decision consequences."""
    steps: list[ResolutionStep] = []

    if recommended_action == "SEARCH_MORE" and discriminators:
        disc = discriminators[0]
        steps.append(ResolutionStep(
            operation="search_for_discriminating_evidence",
            target=disc.evidence_target[:200],
            purpose=f"resolve: {disc.question[:150]}",
            decision_consequence=(
                f"if found and supports {disc.if_true_supports}: VERIFY then ANSWER; "
                f"if found and supports {disc.if_false_supports}: VERIFY then ANSWER; "
                f"if not found: DEFER"
            ),
            stop_condition="discriminating evidence found or search exhausted",
        ))
    elif recommended_action == "VERIFY" and evidence:
        unverified = [e for e in evidence if e.verification_state in (
            VERIFICATION_UNVERIFIED, VERIFICATION_MISSING)]
        if unverified:
            target_ev = unverified[0]
            steps.append(ResolutionStep(
                operation="verify_evidence",
                target=f"{target_ev.evidence_id}: {target_ev.claim[:150]}",
                purpose=f"establish verification state of {target_ev.evidence_id}",
                decision_consequence=(
                    f"SUFFICIENT: update hypotheses supported by {target_ev.evidence_id}; "
                    f"FALSIFIED: eliminate hypotheses supported by {target_ev.evidence_id}; "
                    f"INSUFFICIENT: search for additional evidence"
                ),
                stop_condition=f"verification state of {target_ev.evidence_id} changes",
            ))
    elif recommended_action == "RETRIEVE":
        steps.append(ResolutionStep(
            operation="retrieve_evidence",
            target="evidence relevant to the task and current hypotheses",
            purpose="gather evidence for hypothesis assessment",
            decision_consequence=(
                "new evidence: update evidence map and hypothesis statuses; "
                "no new evidence: consider DEFER"
            ),
            stop_condition="evidence retrieved or retrieval exhausted",
        ))
    elif recommended_action == "REASON_MORE":
        steps.append(ResolutionStep(
            operation="reason_about_hypotheses",
            target="current hypotheses and evidence relationships",
            purpose="determine if any answer condition is satisfied",
            decision_consequence=(
                "condition satisfied: ANSWER; "
                "condition not satisfied: identify next discriminator"
            ),
            stop_condition="answer condition evaluated",
        ))
    elif recommended_action == "ANSWER":
        # Check which answer condition is met
        for ac in answer_conditions:
            if ac.terminal_action == "ANSWER":
                steps.append(ResolutionStep(
                    operation="answer",
                    target=ac.answer_payload_reference[:200],
                    purpose=f"answer based on {ac.hypothesis_id}",
                    decision_consequence="terminal: task completed",
                    stop_condition="answer submitted",
                ))
                break
    elif recommended_action == "DEFER":
        steps.append(ResolutionStep(
            operation="defer",
            target="insufficient evidence to answer",
            purpose="formally defer due to unresolved discriminators",
            decision_consequence="terminal: task deferred",
            stop_condition="defer submitted",
        ))

    return tuple(steps[:max_additional_actions])


class ResolutionGovernor:
    """Deterministic resolution governor.

    Produces ResolutionAssistanceFrame from controller-visible state.
    Also manages persistent ResolutionContext across steps.
    """

    def __init__(self) -> None:
        self._general_governor = GeneralGovernor()

    def plan(
        self,
        observation: Any,
        remaining_steps: int,
        prior_actions: tuple[str, ...] = (),
        prior_outcomes: tuple[str, ...] = (),
    ) -> ResolutionAssistanceFrame | None:
        """Plan a resolution assistance frame from controller-visible state.

        Returns None if the recommended action is STOP (terminal, no scaffold).
        """
        # Extract cognitive state from observation
        snapshot = getattr(observation, "cognitive_state", None)
        task_summary = getattr(observation, "task_summary", "")
        task_id = getattr(observation, "task_id", "")

        # Use general governor for action recommendation
        gov_frame = self._general_governor.assess(
            observation=observation,
            remaining_steps=remaining_steps,
            prior_actions=prior_actions,
            prior_outcomes=prior_outcomes,
        )
        recommended_action = gov_frame.governor_top_action or "ANSWER"

        # STOP is terminal — no scaffold
        if recommended_action == "STOP":
            return None

        # Build hypotheses and evidence from cognitive state
        evidence = build_evidence_map(snapshot, task_summary)
        hypotheses = build_hypotheses(snapshot, task_summary, evidence)

        # Build discriminators and answer conditions
        discriminators = build_discriminators(hypotheses, evidence, snapshot, task_summary)
        answer_conditions = build_answer_conditions(hypotheses, evidence, task_summary)

        # Determine max additional actions (bounded 1-3)
        max_actions = min(3, max(1, remaining_steps))

        # Build defer condition
        defer_condition = build_defer_condition(hypotheses, evidence, max_actions)

        # Build search specification (for SEARCH_MORE)
        search_spec = None
        if recommended_action == "SEARCH_MORE" and discriminators:
            search_spec = _build_search_specification(discriminators[0], task_summary, snapshot)

        # Build execution plan
        execution_plan = _build_execution_plan(
            recommended_action, hypotheses, evidence,
            discriminators, answer_conditions, max_actions)

        # Compute state hash
        state_hash = _state_sha256(
            task_id, task_summary, prior_actions, prior_outcomes,
            remaining_steps, snapshot)

        # Determine unresolved question
        if discriminators:
            unresolved_question = discriminators[0].question
        elif any(h.current_status == HYPOTHESIS_UNRESOLVED for h in hypotheses):
            unresolved_question = "which hypothesis is correct?"
        else:
            unresolved_question = "is the current evidence sufficient to answer?"

        return ResolutionAssistanceFrame(
            schema=RESOLUTION_SCHEMA,
            version=RESOLUTION_VERSION,
            recommended_action=recommended_action,
            task_goal=task_summary[:200],
            candidate_hypotheses=hypotheses,
            current_evidence=evidence,
            unresolved_question=unresolved_question,
            discriminating_evidence=discriminators,
            execution_plan=execution_plan,
            answer_conditions=answer_conditions,
            defer_condition=defer_condition,
            search_specification=search_spec,
            max_additional_actions=max_actions,
            source_state_sha256=state_hash,
        )

    def init_context(
        self,
        task_id: str,
        observation: Any,
        remaining_steps: int,
        prior_actions: tuple[str, ...] = (),
        prior_outcomes: tuple[str, ...] = (),
    ) -> ResolutionContext:
        """Initialize a persistent resolution context."""
        snapshot = getattr(observation, "cognitive_state", None)
        task_summary = getattr(observation, "task_summary", "")

        evidence = build_evidence_map(snapshot, task_summary)
        hypotheses = build_hypotheses(snapshot, task_summary, evidence)
        discriminators = build_discriminators(hypotheses, evidence, snapshot, task_summary)

        # Determine best hypothesis
        best = None
        for h in hypotheses:
            if h.current_status == HYPOTHESIS_SUPPORTED:
                best = h.hypothesis_id
                break
        if best is None:
            for h in hypotheses:
                if h.current_status == HYPOTHESIS_WEAK:
                    best = h.hypothesis_id
                    break

        # Build pending steps from discriminators
        pending = tuple(d.question[:100] for d in discriminators)

        context_id = hashlib.sha256(
            f"{task_id}:{prior_actions}:{remaining_steps}".encode()
        ).hexdigest()[:16]

        return ResolutionContext(
            context_id=context_id,
            hypotheses=hypotheses,
            evidence=evidence,
            active_discriminator=discriminators[0] if discriminators else None,
            completed_steps=(),
            pending_steps=pending,
            current_best_hypothesis=best,
            termination_status="ACTIVE",
            hypothesis_updates=(),
            step_counter=0,
        )

    def update_context(
        self,
        context: ResolutionContext,
        action_taken: str,
        new_observation: Any,
        new_evidence_found: bool,
        evidence_verified: bool,
    ) -> tuple[ResolutionContext, ResolutionReceipt]:
        """Update the resolution context after an action.

        Returns the updated context and a receipt recording what changed.
        """
        snapshot = getattr(new_observation, "cognitive_state", None)
        task_summary = getattr(new_observation, "task_summary", "")

        # Rebuild evidence and hypotheses from new state
        new_evidence = build_evidence_map(snapshot, task_summary)
        new_hypotheses = build_hypotheses(snapshot, task_summary, new_evidence)

        # Track hypothesis updates
        updates: list[HypothesisUpdate] = []
        old_h_map = {h.hypothesis_id: h for h in context.hypotheses}
        for new_h in new_hypotheses:
            old_h = old_h_map.get(new_h.hypothesis_id)
            if old_h is None:
                continue
            if old_h.current_status != new_h.current_status:
                if new_h.current_status == HYPOTHESIS_ELIMINATED:
                    update_type = UPDATE_ELIMINATE
                elif new_h.current_status in (HYPOTHESIS_CONTRADICTED,):
                    update_type = UPDATE_DOWNWEIGHT
                else:
                    update_type = UPDATE_KEEP
                # Find triggering evidence
                trigger = "state_change"
                if new_evidence:
                    trigger = new_evidence[0].evidence_id
                updates.append(HypothesisUpdate(
                    hypothesis_id=new_h.hypothesis_id,
                    update=update_type,
                    evidence_id=trigger,
                    reason_code=f"status: {old_h.current_status} -> {new_h.current_status}",
                ))

        # Determine new best hypothesis
        new_best = None
        for h in new_hypotheses:
            if h.current_status == HYPOTHESIS_SUPPORTED:
                new_best = h.hypothesis_id
                break
        if new_best is None:
            for h in new_hypotheses:
                if h.current_status == HYPOTHESIS_WEAK:
                    new_best = h.hypothesis_id
                    break

        best_changed = new_best != context.current_best_hypothesis

        # Check if discriminator was resolved
        disc_resolved = False
        if context.active_discriminator:
            # Discriminator is resolved if one of its hypotheses is now SUPPORTED
            # and the other is CONTRADICTED/ELIMINATED
            h_true = context.active_discriminator.if_true_supports
            h_false = context.active_discriminator.if_false_supports
            h_map = {h.hypothesis_id: h for h in new_hypotheses}
            true_h = h_map.get(h_true)
            false_h = h_map.get(h_false)
            if true_h and true_h.current_status == HYPOTHESIS_SUPPORTED:
                if false_h and false_h.current_status in (HYPOTHESIS_CONTRADICTED, HYPOTHESIS_ELIMINATED):
                    disc_resolved = True

        # Check answer condition
        answer_cond_met = False
        if new_best:
            for h in new_hypotheses:
                if h.hypothesis_id == new_best and h.current_status == HYPOTHESIS_SUPPORTED:
                    answer_cond_met = True
                    break

        # Update termination status
        n_viable = sum(
            1 for h in new_hypotheses
            if h.current_status not in (HYPOTHESIS_ELIMINATED, HYPOTHESIS_CONTRADICTED)
        )
        if n_viable == 1 and answer_cond_met:
            term_status = "RESOLVED"
        elif context.step_counter + 1 >= 3:  # max_additional_actions
            term_status = "EXHAUSTED"
        else:
            term_status = "ACTIVE"

        # Update completed/pending steps
        completed = context.completed_steps + (context.active_discriminator.question[:100]
                                                if context.active_discriminator else action_taken,)
        new_discriminators = build_discriminators(
            new_hypotheses, new_evidence, snapshot, task_summary)
        pending = tuple(d.question[:100] for d in new_discriminators
                        if d.question[:100] not in completed)

        new_context = ResolutionContext(
            context_id=context.context_id,
            hypotheses=new_hypotheses,
            evidence=new_evidence,
            active_discriminator=new_discriminators[0] if new_discriminators else None,
            completed_steps=completed,
            pending_steps=pending,
            current_best_hypothesis=new_best,
            termination_status=term_status,
            hypothesis_updates=context.hypothesis_updates + tuple(updates),
            step_counter=context.step_counter + 1,
        )

        receipt = ResolutionReceipt(
            step_id=context.step_counter,
            action_taken=action_taken,
            hypotheses_before=len(context.hypotheses),
            hypotheses_after=len(new_hypotheses),
            discriminator_resolved=disc_resolved,
            new_evidence_found=new_evidence_found,
            evidence_verified=evidence_verified,
            best_hypothesis_changed=best_changed,
            answer_condition_satisfied=answer_cond_met,
            hypothesis_updates=tuple(updates),
            terminal_result=None,
        )

        return new_context, receipt
