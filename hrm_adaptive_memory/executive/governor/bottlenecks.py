"""Bottleneck detection: identify what is preventing task termination.

The governor detects the current decision bottleneck from controller-visible
state. It does NOT map bottlenecks directly to actions. Instead, it uses
them to score whether candidate actions can actually resolve the bottleneck.

Bottleneck classes:
    NO_EVIDENCE: insufficient evidence to answer
    UNVERIFIED_EVIDENCE: evidence present but not verified
    STALE_INFORMATION: evidence may be outdated
    UNRESOLVED_CONFLICT: conflicting evidence needs resolution
    INSUFFICIENT_REASONING: composition/reasoning incomplete
    RESOURCE_EXHAUSTION: no resources for useful actions
    REPEATED_NO_GAIN: same actions tried without progress
    READY_TO_ANSWER: no bottleneck detected
    IRREDUCIBLE_UNCERTAINTY: cannot resolve with available actions
"""
from __future__ import annotations

from dataclasses import dataclass
from hrm_adaptive_memory.executive.governor.state import GovernorState


BOTTLENECK_SCHEMA = "DAPH_V2B_I3_5_BOTTLENECK_V1"
BOTTLENECK_VERSION = 1

# Ordinal severity levels
NONE = "NONE"
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DecisionBottleneck:
    """A structured description of what prevents task completion."""
    kind: str
    severity: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    evidence: tuple[str, ...]
    targetable_by: tuple[str, ...]  # action names that CAN target this bottleneck

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "evidence": list(self.evidence),
            "targetable_by": list(self.targetable_by),
        }


def detect_bottlenecks(state: GovernorState) -> tuple[DecisionBottleneck, ...]:
    """Detect all current decision bottlenecks from controller-visible state.

    Returns a tuple of bottlenecks, ordered by severity (most severe first).
    If no bottlenecks are detected, returns a single READY_TO_ANSWER bottleneck.
    """
    bottlenecks: list[DecisionBottleneck] = []

    # Check for resource exhaustion
    resources = state.resource_state
    has_retrieval = resources.get("retrieval", 0) > 0
    has_verification = resources.get("verification", 0) > 0
    has_search = resources.get("search", 0) > 0
    has_reasoning = resources.get("reasoning", 0) > 0

    # Check for repeated no-gain
    if state.repeated_no_gain:
        bottlenecks.append(DecisionBottleneck(
            kind="REPEATED_NO_GAIN",
            severity=HIGH,
            evidence=(f"last_action={state.last_action}", "repeated_without_progress"),
            targetable_by=_actions_that_add_new_information(state),
        ))

    # If we have cognitive state (aware condition), use it
    cs = state.observation.cognitive_state
    if cs is not None:
        # Check verification status
        verif_states = cs.verification_states
        if verif_states:
            vs = verif_states[0].state
            vs_val = vs.value if hasattr(vs, "value") else str(vs)
            if vs_val in ("MISSING", "FALSIFIED"):
                sev = HIGH if not has_retrieval else MEDIUM
                bottlenecks.append(DecisionBottleneck(
                    kind="NO_EVIDENCE" if vs_val == "MISSING" else "FALSIFIED_EVIDENCE",
                    severity=sev,
                    evidence=(f"verification_state={vs_val}",),
                    targetable_by=_actions_for_evidence_gap(has_retrieval, has_search, has_verification),
                ))
            elif vs_val in ("UNVERIFIED", "STALE"):
                sev = HIGH if has_verification else MEDIUM
                bottlenecks.append(DecisionBottleneck(
                    kind="UNVERIFIED_EVIDENCE" if vs_val == "UNVERIFIED" else "STALE_INFORMATION",
                    severity=sev,
                    evidence=(f"verification_state={vs_val}",),
                    targetable_by=_actions_for_verification(has_verification, has_search),
                ))

        # Check temporal status
        temporal = cs.temporal_status
        ts_val = temporal.value if hasattr(temporal, "value") else str(temporal)
        if ts_val == "STALE":
            bottlenecks.append(DecisionBottleneck(
                kind="STALE_INFORMATION",
                severity=MEDIUM,
                evidence=(f"temporal_status={ts_val}",),
                targetable_by=_actions_for_temporal(has_verification, has_search, has_retrieval),
            ))

        # Check conflicts
        conflicts = cs.unresolved_conflicts
        if conflicts:
            resolvable = any(
                getattr(c, "status", "") == "RESOLVABLE"
                for c in conflicts)
            sev = HIGH if resolvable else CRITICAL
            bottlenecks.append(DecisionBottleneck(
                kind="UNRESOLVED_CONFLICT",
                severity=sev,
                evidence=(f"conflict_count={len(conflicts)}",
                          f"resolvable={resolvable}"),
                targetable_by=_actions_for_conflict(has_search, has_verification),
            ))

        # Check composition
        signals = cs.observation_signals
        if signals and "COMPOSITION_INCOMPLETE" in signals:
            bottlenecks.append(DecisionBottleneck(
                kind="INSUFFICIENT_REASONING",
                severity=MEDIUM,
                evidence=("composition_incomplete",),
                targetable_by=("REASON_MORE",) if has_reasoning else (),
            ))

        # Check prior outcomes for retrieval failures
        prior_outcomes = cs.prior_outcomes
        if prior_outcomes.count("RETRIEVE_FAILED") >= 2:
            bottlenecks.append(DecisionBottleneck(
                kind="REPEATED_NO_GAIN",
                severity=HIGH,
                evidence=(f"retrieve_failed_count={prior_outcomes.count('RETRIEVE_FAILED')}",),
                targetable_by=("SEARCH_MORE",) if has_search else (),
            ))
    else:
        # Blind condition: infer from action history
        if not state.action_was_executed("RETRIEVE") and ("RETRIEVE" in state.legal_actions):
            bottlenecks.append(DecisionBottleneck(
                kind="NO_EVIDENCE",
                severity=MEDIUM,
                evidence=("no_retrieval_yet", "blind_condition"),
                targetable_by=("RETRIEVE",) if has_retrieval else (),
            ))
        elif state.action_was_executed("RETRIEVE") and not state.action_was_executed("VERIFY"):
            bottlenecks.append(DecisionBottleneck(
                kind="UNVERIFIED_EVIDENCE",
                severity=MEDIUM,
                evidence=("retrieved_but_not_verified", "blind_condition"),
                targetable_by=("VERIFY",) if has_verification else (),
            ))

    # Check for resource exhaustion
    useful_actions_available = has_retrieval or has_verification or has_search or has_reasoning
    if not useful_actions_available and not bottlenecks:
        bottlenecks.append(DecisionBottleneck(
            kind="RESOURCE_EXHAUSTION",
            severity=HIGH,
            evidence=("no_resources_for_useful_actions",),
            targetable_by=("ANSWER", "DEFER", "STOP"),
        ))

    # If no bottlenecks, ready to answer
    if not bottlenecks:
        bottlenecks.append(DecisionBottleneck(
            kind="READY_TO_ANSWER",
            severity=NONE,
            evidence=("no_bottleneck_detected",),
            targetable_by=("ANSWER",),
        ))

    # Sort by severity (CRITICAL > HIGH > MEDIUM > LOW > NONE)
    severity_order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, NONE: 4}
    bottlenecks.sort(key=lambda b: severity_order.get(b.severity, 5))
    return tuple(bottlenecks)


def _actions_that_add_new_information(state: GovernorState) -> tuple[str, ...]:
    """Actions that can break a no-gain cycle by adding new information."""
    resources = state.resource_state
    result = []
    if resources.get("search", 0) > 0 and "SEARCH_MORE" in state.legal_actions:
        result.append("SEARCH_MORE")
    if resources.get("retrieval", 0) > 0 and "RETRIEVE" in state.legal_actions:
        result.append("RETRIEVE")
    return tuple(result)


def _actions_for_evidence_gap(has_retrieval: bool, has_search: bool, has_verification: bool) -> tuple[str, ...]:
    """Actions that can address an evidence gap."""
    result = []
    if has_retrieval:
        result.append("RETRIEVE")
    if has_search:
        result.append("SEARCH_MORE")
    if has_verification:
        result.append("VERIFY_ALTERNATE_SOURCE")
    return tuple(result)


def _actions_for_verification(has_verification: bool, has_search: bool) -> tuple[str, ...]:
    """Actions that can address unverified evidence."""
    result = []
    if has_verification:
        result.append("VERIFY")
    if has_search:
        result.append("SEARCH_MORE")
    return tuple(result)


def _actions_for_temporal(has_verification: bool, has_search: bool, has_retrieval: bool) -> tuple[str, ...]:
    """Actions that can address stale information."""
    result = []
    if has_verification:
        result.append("VERIFY")
    if has_search:
        result.append("SEARCH_MORE")
    if has_retrieval:
        result.append("RETRIEVE")
    return tuple(result)


def _actions_for_conflict(has_search: bool, has_verification: bool) -> tuple[str, ...]:
    """Actions that can address unresolved conflict."""
    result = []
    if has_search:
        result.append("SEARCH_MORE")
    if has_verification:
        result.append("VERIFY")
    return tuple(result)
