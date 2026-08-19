"""Evidence-bearing executor with targeted action semantics.

RETRIEVE: exposes evidence items declared in task.retrieve_exposes
SEARCH_MORE: exposes evidence items declared in task.search_exposes
VERIFY: verifies the most recently retrieved unverified evidence item
ANSWER: succeeds if the correct hypothesis has sufficient verified support
DEFER: succeeds if expected terminal is DEFER
STOP: succeeds if expected terminal is STOP

The executor resolves targeted actions against the resolution frame's
nominated target_evidence_id when available. If no target is nominated,
it applies to the most recently retrieved unverified item.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceExhausted

from .schema import (
    EvidenceTask, EvidenceRuntime, EvidenceActionExecution,
    EvidenceItem, EvidenceSnapshot, EvidenceHypothesis,
    initial_evidence_runtime,
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
            # Verify a specific evidence item
            target_id = target_evidence_id
            if target_id is None:
                # Find most recently retrieved unverified item
                retrieved_unverified = [
                    ev for ev in next_evidence
                    if ev.retrieved and ev.verification_state == VerificationState.UNVERIFIED
                ]
                if retrieved_unverified:
                    target_id = retrieved_unverified[-1].evidence_id

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
        else:
            success = task.expected_terminal is action

        return EvidenceActionExecution(
            action, next_runtime, True, success,
            "TASK_SUCCESS" if success else "TASK_FAILURE",
            tuple(exposed), tuple(verified))

    def _check_answer_success(self, runtime: EvidenceRuntime) -> bool:
        """Check if ANSWER is correct.

        ANSWER succeeds if:
          1. The task's expected terminal is ANSWER, AND
          2. The correct hypothesis has at least one SUFFICIENT, CURRENT
             evidence item supporting it, AND
          3. No contradicting evidence is SUFFICIENT for the correct hypothesis
        """
        task = runtime.task
        if task.expected_terminal is not DecisionAction.ANSWER:
            return False

        correct_h = task.correct_hypothesis_id

        # Check for sufficient supporting evidence for the correct hypothesis
        has_support = False
        has_contradiction = False

        for ev in runtime.evidence:
            if not ev.retrieved:
                continue
            if ev.verification_state != VerificationState.SUFFICIENT:
                continue
            if ev.temporal_status == TemporalStatus.STALE:
                continue
            if correct_h in ev.supports:
                has_support = True
            if correct_h in ev.contradicts:
                has_contradiction = True

        return has_support and not has_contradiction


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
    supporting = [e for e in verified
                  if e.verification_state == VerificationState.SUFFICIENT and e.supports]
    contradicting = [e for e in verified
                     if e.verification_state == VerificationState.FALSIFIED and e.supports]

    # Action-availability hints: whether RETRIEVE or SEARCH_MORE would
    # expose new evidence. Controller-visible: the model can try either
    # action and observe whether new evidence appears.
    retrieved_ids = {e.evidence_id for e in visible}
    retrieve_available = any(
        eid not in retrieved_ids for eid in task.retrieve_exposes
    )
    search_available = any(
        eid not in retrieved_ids for eid in task.search_exposes
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
        retrieve_available=retrieve_available,
        search_available=search_available,
    )
