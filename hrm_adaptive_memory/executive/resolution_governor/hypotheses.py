"""Hypothesis construction from controller-visible state.

Derives competing hypotheses strictly from:
  - task_summary (the task description)
  - relevant_memories (MemorySummary with verification_state, evidence_count)
  - verification_states (VerificationSummary)
  - unresolved_conflicts (ConflictSummary)
  - temporal_status
  - prior_actions and prior_outcomes

NO evaluator labels, oracle Q-values, gold answers, or latent task state.
"""
from __future__ import annotations

from typing import Any

from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, MemorySummary, VerificationState, TemporalStatus,
)

from .schema import (
    Hypothesis, EvidenceAssessment,
    HYPOTHESIS_SUPPORTED, HYPOTHESIS_WEAK, HYPOTHESIS_CONTRADICTED,
    HYPOTHESIS_UNRESOLVED, HYPOTHESIS_ELIMINATED,
    VERIFICATION_SUFFICIENT, VERIFICATION_MISSING, VERIFICATION_UNVERIFIED,
    VERIFICATION_FALSIFIED, VERIFICATION_STALE,
    TEMPORAL_CURRENT, TEMPORAL_STALE, TEMPORAL_UNKNOWN,
)


def _temporal_str(status: TemporalStatus) -> str:
    if status == TemporalStatus.CURRENT:
        return TEMPORAL_CURRENT
    if status == TemporalStatus.STALE:
        return TEMPORAL_STALE
    return TEMPORAL_UNKNOWN


def _verification_str(state: VerificationState) -> str:
    if state == VerificationState.SUFFICIENT:
        return VERIFICATION_SUFFICIENT
    if state == VerificationState.MISSING:
        return VERIFICATION_MISSING
    if state == VerificationState.UNVERIFIED:
        return VERIFICATION_UNVERIFIED
    if state == VerificationState.FALSIFIED:
        return VERIFICATION_FALSIFIED
    if state == VerificationState.STALE:
        return VERIFICATION_STALE
    return VERIFICATION_UNVERIFIED


def build_evidence_map(
    snapshot: CognitiveStateSnapshot | None,
    task_summary: str,
) -> tuple[EvidenceAssessment, ...]:
    """Build evidence assessments from cognitive state.

    Each relevant memory becomes an EvidenceAssessment with:
      - evidence_id from memory_id
      - claim derived from task context
      - verification_state from memory's verification_state
      - temporal_state from memory's temporal_status
      - supports/contradicts derived from conflict_state

    If no cognitive state (BLIND), returns empty tuple.
    """
    if snapshot is None:
        return ()

    evidence: list[EvidenceAssessment] = []
    for i, mem in enumerate(snapshot.relevant_memories):
        # Derive supports/contradicts from conflict_state
        supports: list[str] = []
        contradicts: list[str] = []
        conflict = mem.conflict_state.lower() if mem.conflict_state else ""
        if "support" in conflict or "confirm" in conflict:
            supports.append("H1")  # Default: supports primary hypothesis
        if "conflict" in conflict or "contradict" in conflict:
            contradicts.append("H1")
        if "stale" in conflict or "outdated" in conflict:
            # Stale evidence doesn't support anything actively
            pass

        # If there are conflicts, evidence might support H2
        if snapshot.unresolved_conflicts and "conflict" in conflict:
            supports.append("H2")

        evidence.append(EvidenceAssessment(
            evidence_id=mem.memory_id,
            claim=f"evidence item {i+1} for {task_summary[:80]}",
            source_type=f"lineage_{mem.source_lineage_count}",
            supports=tuple(supports) if supports else ("H1",),
            contradicts=tuple(contradicts),
            verification_state=_verification_str(mem.verification_state),
            temporal_state=_temporal_str(mem.temporal_status),
            relevance=f"evidence_count={mem.evidence_count}",
        ))

    return tuple(evidence)


def build_hypotheses(
    snapshot: CognitiveStateSnapshot | None,
    task_summary: str,
    evidence: tuple[EvidenceAssessment, ...],
) -> tuple[Hypothesis, ...]:
    """Build competing hypotheses from cognitive state and evidence.

    The hypothesis structure is derived from:
      1. The task itself implies a primary hypothesis (H1: answer the task)
      2. If there are unresolved conflicts, a competing hypothesis (H2)
      3. If temporal status is STALE/UNKNOWN, a temporal hypothesis (H_temp)
      4. If verification is missing, an insufficient-evidence hypothesis (H_insuf)

    Status is derived from evidence relationships:
      - SUPPORTED: has verified current supporting evidence, no contradictions
      - WEAK: has some support but unverified or stale
      - CONTRADICTED: has verified contradicting evidence
      - UNRESOLVED: mixed or insufficient evidence
      - ELIMINATED: falsified
    """
    if snapshot is None:
        # BLIND: single hypothesis with UNRESOLVED status
        return (
            Hypothesis(
                hypothesis_id="H1",
                proposition=f"answer the task: {task_summary[:150]}",
                supporting_evidence_ids=(),
                contradicting_evidence_ids=(),
                current_status=HYPOTHESIS_UNRESOLVED,
            ),
        )

    hypotheses: list[Hypothesis] = []

    # H1: primary answer hypothesis
    h1_support = tuple(
        e.evidence_id for e in evidence
        if "H1" in e.supports
        and e.verification_state == VERIFICATION_SUFFICIENT
        and e.temporal_state == TEMPORAL_CURRENT
    )
    h1_contradict = tuple(
        e.evidence_id for e in evidence
        if "H1" in e.contradicts
        and e.verification_state in (VERIFICATION_SUFFICIENT, VERIFICATION_FALSIFIED)
    )
    h1_weak_support = tuple(
        e.evidence_id for e in evidence
        if "H1" in e.supports
        and (e.verification_state in (VERIFICATION_UNVERIFIED, VERIFICATION_MISSING)
             or e.temporal_state in (TEMPORAL_STALE, TEMPORAL_UNKNOWN))
    )

    if h1_contradict:
        h1_status = HYPOTHESIS_CONTRADICTED
    elif h1_support:
        h1_status = HYPOTHESIS_SUPPORTED
    elif h1_weak_support:
        h1_status = HYPOTHESIS_WEAK
    else:
        h1_status = HYPOTHESIS_UNRESOLVED

    hypotheses.append(Hypothesis(
        hypothesis_id="H1",
        proposition=f"the primary answer to: {task_summary[:150]}",
        supporting_evidence_ids=h1_support + h1_weak_support,
        contradicting_evidence_ids=h1_contradict,
        current_status=h1_status,
    ))

    # H2: competing hypothesis (if conflicts exist)
    if snapshot.unresolved_conflicts:
        h2_support = tuple(
            e.evidence_id for e in evidence
            if "H2" in e.supports
            and e.verification_state == VERIFICATION_SUFFICIENT
            and e.temporal_state == TEMPORAL_CURRENT
        )
        h2_contradict = tuple(
            e.evidence_id for e in evidence
            if "H2" in e.contradicts
        )
        h2_weak = tuple(
            e.evidence_id for e in evidence
            if "H2" in e.supports
            and (e.verification_state in (VERIFICATION_UNVERIFIED, VERIFICATION_MISSING)
                 or e.temporal_state in (TEMPORAL_STALE, TEMPORAL_UNKNOWN))
        )

        if h2_contradict:
            h2_status = HYPOTHESIS_CONTRADICTED
        elif h2_support:
            h2_status = HYPOTHESIS_SUPPORTED
        elif h2_weak:
            h2_status = HYPOTHESIS_WEAK
        else:
            h2_status = HYPOTHESIS_UNRESOLVED

        conflict_desc = "; ".join(
            f"conflict {c.conflict_id} ({c.relation})"
            for c in snapshot.unresolved_conflicts[:2]
        )
        hypotheses.append(Hypothesis(
            hypothesis_id="H2",
            proposition=f"alternative answer due to: {conflict_desc[:150]}",
            supporting_evidence_ids=h2_support + h2_weak,
            contradicting_evidence_ids=h2_contradict,
            current_status=h2_status,
        ))

    # H_defer: insufficient evidence to answer (if verification missing)
    has_missing = any(
        e.verification_state == VERIFICATION_MISSING for e in evidence
    )
    has_unverified = any(
        e.verification_state == VERIFICATION_UNVERIFIED for e in evidence
    )
    if has_missing or has_unverified:
        defer_support = tuple(
            e.evidence_id for e in evidence
            if e.verification_state in (VERIFICATION_MISSING, VERIFICATION_UNVERIFIED)
        )
        hypotheses.append(Hypothesis(
            hypothesis_id="H_defer",
            proposition="insufficient verified evidence to answer confidently",
            supporting_evidence_ids=defer_support,
            contradicting_evidence_ids=(),
            current_status=HYPOTHESIS_WEAK if defer_support else HYPOTHESIS_UNRESOLVED,
        ))

    return tuple(hypotheses[:4])  # max 4
