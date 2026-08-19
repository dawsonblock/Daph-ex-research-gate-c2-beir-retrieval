"""Decision rules: discriminators and answer conditions.

Derives:
  - Discriminators: what information would change the decision
  - AnswerConditions: explicit hypothesis -> answer mappings
  - DeferCondition: when to give up

All derived strictly from controller-visible state.
"""
from __future__ import annotations

from typing import Any

from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, VerificationState, TemporalStatus,
)

from .schema import (
    Hypothesis, EvidenceAssessment, Discriminator, AnswerCondition,
    HYPOTHESIS_SUPPORTED, HYPOTHESIS_WEAK, HYPOTHESIS_CONTRADICTED,
    HYPOTHESIS_UNRESOLVED, HYPOTHESIS_ELIMINATED,
    VERIFICATION_SUFFICIENT, VERIFICATION_MISSING, VERIFICATION_UNVERIFIED,
    VERIFICATION_FALSIFIED, VERIFICATION_STALE,
    TEMPORAL_CURRENT, TEMPORAL_STALE, TEMPORAL_UNKNOWN,
)


def build_discriminators(
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceAssessment, ...],
    snapshot: CognitiveStateSnapshot | None,
    task_summary: str,
) -> tuple[Discriminator, ...]:
    """Build discriminators from hypotheses and evidence.

    A discriminator specifies:
      - A question whose answer would change the decision
      - Which hypothesis is supported if true vs false
      - What evidence target would resolve it
      - Whether verification is required

    Priority:
      1. If there are competing hypotheses (H1 vs H2), the discriminator
         is about which one is correct
      2. If temporal status is STALE/UNKNOWN, the discriminator is about
         whether evidence is still current
      3. If verification is missing, the discriminator is about whether
         the evidence can be verified
    """
    discriminators: list[Discriminator] = []

    viable = [h for h in hypotheses if h.current_status not in (
        HYPOTHESIS_ELIMINATED, HYPOTHESIS_CONTRADICTED)]

    if len(viable) >= 2:
        h1, h2 = viable[0], viable[1]
        # Discriminator between top two hypotheses
        discriminators.append(Discriminator(
            question=f"Does the evidence support {h1.hypothesis_id} or {h2.hypothesis_id}?",
            if_true_supports=h1.hypothesis_id,
            if_false_supports=h2.hypothesis_id,
            evidence_target=f"current verified evidence distinguishing {h1.hypothesis_id} from {h2.hypothesis_id}",
            verification_required=True,
        ))

    # Temporal discriminator
    if snapshot and snapshot.temporal_status in (TemporalStatus.STALE, TemporalStatus.UNKNOWN):
        h_primary = viable[0] if viable else hypotheses[0]
        h_alt = viable[1] if len(viable) > 1 else hypotheses[0]
        discriminators.append(Discriminator(
            question="Is the existing evidence still current and valid?",
            if_true_supports=h_primary.hypothesis_id,
            if_false_supports=h_alt.hypothesis_id,
            evidence_target="current timestamped evidence confirming or updating the existing evidence",
            verification_required=True,
        ))

    # Verification discriminator
    unverified = [e for e in evidence if e.verification_state in (
        VERIFICATION_UNVERIFIED, VERIFICATION_MISSING)]
    if unverified and len(discriminators) < 3:
        h_primary = viable[0] if viable else hypotheses[0]
        h_defer = next(
            (h for h in hypotheses if h.hypothesis_id == "H_defer"), None)
        h_alt = h_defer if h_defer else (viable[1] if len(viable) > 1 else hypotheses[0])
        discriminators.append(Discriminator(
            question=f"Can evidence {unverified[0].evidence_id} be verified as sufficient?",
            if_true_supports=h_primary.hypothesis_id,
            if_false_supports=h_alt.hypothesis_id,
            evidence_target=f"verification of {unverified[0].evidence_id} establishing its sufficiency",
            verification_required=True,
        ))

    # Conflict discriminator
    if snapshot and snapshot.unresolved_conflicts:
        h_primary = viable[0] if viable else hypotheses[0]
        h_alt = next(
            (h for h in hypotheses if h.hypothesis_id == "H2"), None)
        if h_alt is None and len(viable) > 1:
            h_alt = viable[1]
        elif h_alt is None:
            h_alt = hypotheses[0]
        discriminators.append(Discriminator(
            question=f"Can the conflict {snapshot.unresolved_conflicts[0].conflict_id} be resolved?",
            if_true_supports=h_primary.hypothesis_id,
            if_false_supports=h_alt.hypothesis_id,
            evidence_target=f"evidence resolving conflict {snapshot.unresolved_conflicts[0].conflict_id}",
            verification_required=True,
        ))

    return tuple(discriminators[:4])


def build_answer_conditions(
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceAssessment, ...],
    task_summary: str,
) -> tuple[AnswerCondition, ...]:
    """Build explicit hypothesis -> answer mappings.

    These close the gap between information acquired and decision changed.

    For each viable hypothesis, specify:
      - What condition must be met
      - What terminal action to take
      - What answer payload to use
    """
    conditions: list[AnswerCondition] = []

    for h in hypotheses:
        if h.current_status == HYPOTHESIS_SUPPORTED:
            # Already supported -> can answer
            conditions.append(AnswerCondition(
                hypothesis_id=h.hypothesis_id,
                condition=f"{h.hypothesis_id} has verified current support and no unresolved contradictions",
                terminal_action="ANSWER",
                answer_payload_reference=f"answer based on {h.hypothesis_id}: {h.proposition[:100]}",
            ))
        elif h.current_status == HYPOTHESIS_WEAK:
            # Weak support -> answer if verified
            conditions.append(AnswerCondition(
                hypothesis_id=h.hypothesis_id,
                condition=f"{h.hypothesis_id} gains verified current support and all contradictions are resolved",
                terminal_action="ANSWER",
                answer_payload_reference=f"answer based on {h.hypothesis_id}: {h.proposition[:100]}",
            ))
        elif h.current_status == HYPOTHESIS_UNRESOLVED:
            # Unresolved -> answer only if discriminating evidence is found
            conditions.append(AnswerCondition(
                hypothesis_id=h.hypothesis_id,
                condition=f"discriminating evidence confirms {h.hypothesis_id} and eliminates alternatives",
                terminal_action="ANSWER",
                answer_payload_reference=f"answer based on {h.hypothesis_id}: {h.proposition[:100]}",
            ))

    # Defer condition
    defer_h = next((h for h in hypotheses if h.hypothesis_id == "H_defer"), None)
    if defer_h:
        conditions.append(AnswerCondition(
            hypothesis_id=defer_h.hypothesis_id,
            condition="no hypothesis gains sufficient verified support after max_additional_actions",
            terminal_action="DEFER",
            answer_payload_reference="defer: insufficient evidence to answer",
        ))

    return tuple(conditions[:4])


def build_defer_condition(
    hypotheses: tuple[Hypothesis, ...],
    evidence: tuple[EvidenceAssessment, ...],
    max_additional_actions: int,
) -> str:
    """Build the defer condition string."""
    viable = [h for h in hypotheses if h.current_status not in (
        HYPOTHESIS_ELIMINATED, HYPOTHESIS_CONTRADICTED)]
    if len(viable) == 0:
        return "all hypotheses eliminated or contradicted; DEFER"
    if len(viable) == 1 and viable[0].current_status == HYPOTHESIS_SUPPORTED:
        return "single supported hypothesis remains; ANSWER rather than DEFER"
    return (
        f"after {max_additional_actions} additional actions, if no hypothesis "
        f"gains verified current support and all discriminators remain unresolved, DEFER"
    )
