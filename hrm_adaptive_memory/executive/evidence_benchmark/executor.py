"""Evidence-bearing executor with targeted action semantics.

RETRIEVE: exposes evidence items declared in task.retrieve_exposes
SEARCH_MORE: exposes evidence items declared in task.search_exposes
VERIFY: verifies a retrieved, currently-unverified evidence item
ANSWER: succeeds if the correct hypothesis has sufficient verified support
DEFER: succeeds if expected terminal is DEFER
STOP: succeeds if expected terminal is STOP

The executor resolves targeted actions against the resolution frame's
nominated target_evidence_id when available. If no target is nominated,
it applies to the most recently retrieved unverified item.

R12.9C: VERIFY on already-verified evidence is rejected with
INVALID_VERIFY_TARGET. The canonical valid_verify_targets() function
is the single source of truth for legal VERIFY targets.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceExhausted

from daph.epistemic import derive_hypothesis_topology, is_answer_ready, HypothesisState

from .schema import (
    EvidenceTask, EvidenceRuntime, EvidenceActionExecution,
    EvidenceItem, EvidenceSnapshot, EvidenceHypothesis,
    initial_evidence_runtime,
)


# ---------------------------------------------------------------------------
# R12.9C: Canonical valid VERIFY targets — single source of truth
# ---------------------------------------------------------------------------

# Terminal verification states that preclude re-verification
_TERMINAL_VERIFY_STATES = frozenset({
    VerificationState.SUFFICIENT,
    VerificationState.FALSIFIED,
    VerificationState.MISSING,
    VerificationState.STALE,
})


def valid_verify_targets(runtime: EvidenceRuntime) -> tuple[str, ...]:
    """Return the set of evidence IDs that are legal VERIFY targets.

    An evidence item is a legal VERIFY target if and only if:
      1. It has been retrieved (is visible to the controller)
      2. Its verification_state is UNVERIFIED (not already terminally verified)

    This is the single source of truth for VERIFY target validity.
    The executor, affordance system, and trajectory runners must all
    defer to this function.
    """
    return tuple(
        ev.evidence_id
        for ev in runtime.evidence
        if ev.retrieved and ev.verification_state == VerificationState.UNVERIFIED
    )


class EvidenceExecutor:
    """Executes actions on evidence-bearing tasks.

    Applies frozen task effects; never calls an LLM, retrieval service, or HTTP.
    """

    def execute(
        self,
        runtime: EvidenceRuntime,
        action: DecisionAction,
        target_evidence_id: str | None = None,
    ) -> EvidenceActionExecution:
        """Execute an action on the evidence runtime.

        Args:
            runtime: current evidence runtime state
            action: the action to execute
            target_evidence_id: optional evidence ID nominated by the
                resolution frame. If provided, VERIFY targets this specific
                evidence item. If None, VERIFY targets the most recently
                retrieved unverified item.
        """
        task = runtime.task

        # Consume resources
        try:
            next_resources = runtime.resources.consume(action)
        except ResourceExhausted:
            return EvidenceActionExecution(
                action, runtime, True, False, "RESOURCE_EXHAUSTED",
                (), ())

        next_evidence = list(runtime.evidence)
        exposed: list[str] = []
        verified: list[str] = []
        searched = runtime.searched
        reasoning_complete = runtime.reasoning_complete

        if action is DecisionAction.RETRIEVE:
            # Expose evidence items declared in retrieve_exposes
            for eid in task.retrieve_exposes:
                for i, ev in enumerate(next_evidence):
                    if ev.evidence_id == eid and not ev.retrieved:
                        next_evidence[i] = replace(ev, retrieved=True)
                        exposed.append(eid)
                        break

        elif action is DecisionAction.SEARCH_MORE:
            searched = True
            # Expose evidence items declared in search_exposes
            for eid in task.search_exposes:
                for i, ev in enumerate(next_evidence):
                    if ev.evidence_id == eid and not ev.retrieved:
                        next_evidence[i] = replace(ev, retrieved=True)
                        exposed.append(eid)
                        break

        elif action is DecisionAction.VERIFY:
            # R12.9C: Use canonical valid_verify_targets for target validation
            legal_targets = valid_verify_targets(runtime)
            target_id = target_evidence_id

            if target_id is None:
                # Find most recently retrieved unverified item
                retrieved_unverified = [
                    ev for ev in next_evidence
                    if ev.retrieved and ev.verification_state == VerificationState.UNVERIFIED
                ]
                if retrieved_unverified:
                    target_id = retrieved_unverified[-1].evidence_id

            # R12.9C: Reject VERIFY on already-verified or non-existent targets
            if target_id is not None and target_id not in legal_targets:
                # INVALID_VERIFY_TARGET: the target is either not retrieved,
                # already terminally verified, or doesn't exist.
                # Resources are consumed but evidence is not modified.
                next_runtime_invalid = EvidenceRuntime(
                    task=task,
                    resources=next_resources,
                    evidence=tuple(next_evidence),
                    retrieved_evidence_ids=runtime.retrieved_evidence_ids,
                    verified_evidence_ids=runtime.verified_evidence_ids,
                    searched=searched,
                    reasoning_complete=reasoning_complete,
                )
                return EvidenceActionExecution(
                    action, next_runtime_invalid, False, None,
                    "INVALID_VERIFY_TARGET",
                    tuple(exposed), ())

            if target_id is not None:
                for i, ev in enumerate(next_evidence):
                    if ev.evidence_id == target_id and ev.retrieved:
                        if ev.verify_result:
                            new_state = VerificationState(ev.verify_result)
                        else:
                            # Default: verifying makes it SUFFICIENT
                            new_state = VerificationState.SUFFICIENT
                        # If evidence is STALE, verification reveals it's STALE
                        if ev.temporal_status == TemporalStatus.STALE:
                            new_state = VerificationState.STALE
                        next_evidence[i] = replace(
                            ev, verification_state=new_state)
                        verified.append(target_id)
                        break

        elif action is DecisionAction.REASON_MORE:
            reasoning_complete = True

        next_retrieved = tuple(
            ev.evidence_id for ev in next_evidence if ev.retrieved)
        next_verified = tuple(
            ev.evidence_id for ev in next_evidence
            if ev.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED))

        next_runtime = EvidenceRuntime(
            task=task,
            resources=next_resources,
            evidence=tuple(next_evidence),
            retrieved_evidence_ids=next_retrieved,
            verified_evidence_ids=next_verified,
            searched=searched,
            reasoning_complete=reasoning_complete,
        )

        # Check terminal actions
        if action not in {DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}:
            return EvidenceActionExecution(
                action, next_runtime, False, None,
                f"{action.value}_COMPLETED",
                tuple(exposed), tuple(verified))

        # Terminal actions
        if action is DecisionAction.ANSWER:
            success = self._check_answer_success(next_runtime)
        elif action is DecisionAction.DEFER:
            success = self._check_defer_success(next_runtime)
        else:
            success = task.expected_terminal is action

        return EvidenceActionExecution(
            action, next_runtime, True, success,
            "TASK_SUCCESS" if success else "TASK_FAILURE",
            tuple(exposed), tuple(verified))

    def _check_answer_success(self, runtime: EvidenceRuntime) -> bool:
        """Check if ANSWER is correct.

        ANSWER succeeds if and only if:
          1. The task's expected terminal is ANSWER, AND
          2. The state is ANSWER_READY per EPISTEMIC_SEMANTICS_V1.md §6.1:
             exactly one hypothesis has verified support (SUFFICIENT, CURRENT)
             and no verified contradiction, AND
          3. That uniquely supported hypothesis is the correct hypothesis.

        Uses the canonical derive_hypothesis_topology() as the single source
        of truth for epistemic state — no duplicated topology logic.
        """
        task = runtime.task
        if task.expected_terminal is not DecisionAction.ANSWER:
            return False

        # Build observable evidence dicts for canonical topology
        visible_ev = []
        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            visible_ev.append({
                "evidence_id": ev.evidence_id,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "verification_state": ev.verification_state,
                "temporal_status": ev.temporal_status,
                "retrieved": ev.retrieved,
            })

        hyp_ids = [h.hypothesis_id for h in task.hypotheses]

        # Derive canonical topology — single source of truth
        topology = derive_hypothesis_topology(visible_ev, hyp_ids)

        # ANSWER_READY requires exactly one SUPPORTED hypothesis
        if not is_answer_ready(topology):
            return False

        # That uniquely supported hypothesis must be the correct one (evaluator-side check)
        return topology.unique_supported_hypothesis == task.correct_hypothesis_id

    def _check_defer_success(self, runtime: EvidenceRuntime) -> bool:
        """Check if DEFER is correct.

        DEFER succeeds if and only if:
          1. The task's expected terminal is DEFER, AND
          2. The state is NOT ANSWER_READY per canonical topology (no unique
             supported hypothesis), AND
          3. No admissible continuation can resolve the state.

        This unifies DEFER success with canonical topology per
        EPISTEMIC_SEMANTICS_V1.md §6.2. Previously DEFER only checked
        expected_terminal, which could disagree with the epistemic state.
        """
        task = runtime.task
        if task.expected_terminal is not DecisionAction.DEFER:
            return False

        # Build observable evidence dicts for canonical topology
        visible_ev = []
        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            visible_ev.append({
                "evidence_id": ev.evidence_id,
                "supports": list(ev.supports),
                "contradicts": list(ev.contradicts),
                "verification_state": ev.verification_state,
                "temporal_status": ev.temporal_status,
                "retrieved": ev.retrieved,
            })

        hyp_ids = [h.hypothesis_id for h in task.hypotheses]
        topology = derive_hypothesis_topology(visible_ev, hyp_ids)

        # DEFER is epistemically justified when ANSWER_READY is false.
        # Per EPISTEMIC_SEMANTICS_V1.md §6.1, ANSWER_READY requires the
        # uniquely supported hypothesis to have answer_action == ANSWER.
        # If the uniquely supported hypothesis has answer_action == DEFER,
        # the state is DEFER_READY, not ANSWER_READY.
        if is_answer_ready(topology):
            # Check if the supported hypothesis is actually an ANSWER hypothesis
            if topology.unique_supported_hypothesis is not None:
                supported_hyp = next(
                    (h for h in task.hypotheses
                     if h.hypothesis_id == topology.unique_supported_hypothesis),
                    None,
                )
                if supported_hyp is not None and supported_hyp.answer_action is DecisionAction.DEFER:
                    # Unique supported hypothesis maps to DEFER, not ANSWER.
                    # State is DEFER_READY (assuming no continuation can resolve).
                    pass  # Fall through to continuation check below
                else:
                    return False  # State is genuinely ANSWER_READY, DEFER is wrong
            else:
                return False  # Should not happen if is_answer_ready is True

        # Check if any continuation could resolve the state
        # A continuation is admissible if it could change the topology
        has_unverified_discriminating = topology.unverified_evidence_exists
        has_hidden = topology.hidden_evidence_count > 0

        rs = runtime.resources.as_dict()
        can_verify = rs.get("verification_calls_remaining", 0) > 0 and len(valid_verify_targets(runtime)) > 0
        can_retrieve = rs.get("retrieval_calls_remaining", 0) > 0 and has_hidden
        can_search = rs.get("search_calls_remaining", 0) > 0 and not runtime.searched

        # If any continuation exists that could resolve, DEFER is premature
        if can_verify and has_unverified_discriminating:
            return False
        if can_retrieve and has_hidden:
            return False
        if can_search:
            return False

        # No resolving continuation available — DEFER is justified
        return True


def build_evidence_snapshot(
    runtime: EvidenceRuntime,
    *,
    prior_actions: tuple[str, ...] = (),
    prior_outcomes: tuple[str, ...] = (),
) -> EvidenceSnapshot:
    """Build a controller-visible snapshot from the evidence runtime."""
    task = runtime.task
    visible = runtime.visible_evidence
    hidden_count = len(runtime.hidden_evidence)
    verified = [e for e in visible
                if e.verification_state in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED)]
    # Canonical semantics per EPISTEMIC_SEMANTICS_V1.md §3.2:
    # SUFFICIENT + supports(H) → verified support
    # SUFFICIENT + contradicts(H) → verified contradiction
    # FALSIFIED + any → no positive evidential force (claim failed)
    supporting = [e for e in verified
                  if e.verification_state == VerificationState.SUFFICIENT and e.supports]
    contradicting = [e for e in verified
                     if e.verification_state == VerificationState.SUFFICIENT and e.contradicts]

    # Clean affordances: whether operations are legally callable.
    # Derived exclusively from resource budgets and visible evidence state.
    # Does NOT inspect task.retrieve_exposes, task.search_exposes, or any
    # hidden/transition information.
    # R12.9C: can_verify defers to valid_verify_targets() as single source of truth
    rs = runtime.resources.as_dict()
    can_retrieve = rs.get("retrieval_calls_remaining", 0) > 0
    can_search = rs.get("search_calls_remaining", 0) > 0
    legal_verify_targets = valid_verify_targets(runtime)
    can_verify = (
        rs.get("verification_calls_remaining", 0) > 0
        and len(legal_verify_targets) > 0
    )

    return EvidenceSnapshot(
        task_id=task.task_id,
        task_summary=task.task_summary,
        visible_evidence=visible,
        hidden_evidence_count=hidden_count,
        hypotheses=task.hypotheses,
        verified_count=len(verified),
        supporting_count=len(supporting),
        contradicting_count=len(contradicting),
        searched=runtime.searched,
        reasoning_complete=runtime.reasoning_complete,
        resource_state=runtime.resources.as_dict(),
        prior_actions=prior_actions,
        prior_outcomes=prior_outcomes,
        can_retrieve=can_retrieve,
        can_search=can_search,
        can_verify=can_verify,
    )
