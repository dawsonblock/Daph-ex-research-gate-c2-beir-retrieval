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
    CHAIN_INCOMPLETE: V2 composition chain not started or not finished
    CHAIN_DISCOVERY: chain not started, need to discover starting action
    RESOURCE_EXHAUSTION: no resources for useful actions
    REPEATED_NO_GAIN: same actions tried without progress (outcome-based)
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

    Chain-aware: tracks V2_STAGE_N outcomes to detect composition chain
    progress and prevents premature READY_TO_ANSWER when the chain is
    incomplete or verification is missing.
    """
    bottlenecks: list[DecisionBottleneck] = []

    # Use typed resources
    res = state.resources
    has_retrieval = res.has_retrieval
    has_verification = res.has_verification
    has_search = res.has_search
    has_reasoning = res.has_reasoning

    # Extract cognitive state early so chain tracking can be gated on it.
    # Chain tracking only applies to the aware condition (which has
    # cognitive_state with prior_outcomes). The blind condition cannot
    # see V2_STAGE_N outcomes, so chain tracking would add noise.
    cs = state.observation.cognitive_state

    # Extract chain progress from prior outcomes (aware only)
    if cs is not None and cs.prior_outcomes:
        chain = state.chain_progress
    else:
        from hrm_adaptive_memory.executive.governor.chain_progress import ChainProgress
        chain = ChainProgress(
            stages_completed=0, stage_outcomes=(),
            actions_that_advanced=(), actions_that_failed=(),
            is_started=False, is_complete=False, is_poisoned=False,
            total_steps=len(state.prior_actions),
        )

    # Check for repeated no-gain (now outcome-based, not just action-based)
    if state.repeated_no_gain:
        bottlenecks.append(DecisionBottleneck(
            kind="REPEATED_NO_GAIN",
            severity=HIGH,
            evidence=(f"last_action={state.last_action}",
                      f"last_outcome={state.last_outcome}",
                      "repeated_same_action_and_outcome"),
            targetable_by=_actions_that_add_new_information(state),
        ))

    # Chain discovery and chain incomplete: aware condition only.
    # The blind condition cannot see V2_STAGE_N outcomes.
    if cs is not None:
        # Chain discovery: chain not started, need to find the starting action.
        # Only fire when the model has already tried actions but none advanced
        # the chain. On the very first step (no prior actions), the regular
        # NO_EVIDENCE / UNVERIFIED_EVIDENCE bottlenecks handle action selection.
        if chain.needs_discovery and chain.total_steps > 0:
            untried = state.untried_composable()
            if untried:
                bottlenecks.append(DecisionBottleneck(
                    kind="CHAIN_DISCOVERY",
                    severity=HIGH,
                    evidence=("chain_not_started",
                              f"stages_completed={chain.stages_completed}",
                              f"untried={untried}"),
                    targetable_by=untried,
                ))
            elif chain.is_poisoned:
                # Chain was poisoned — task likely unsolvable
                bottlenecks.append(DecisionBottleneck(
                    kind="IRREDUCIBLE_UNCERTAINTY",
                    severity=CRITICAL,
                    evidence=("chain_poisoned", "control_poisoned_in_outcomes"),
                    targetable_by=("DEFER", "STOP"),
                ))

        # Chain incomplete: started but not finished
        if chain.needs_continuation:
            # Recommend actions that haven't been tried yet (might advance chain)
            untried = state.untried_composable()
            # Also include actions that advanced before (might advance again)
            advanced = tuple(a for a in chain.actions_that_advanced
                             if a in state.legal_actions)
            targetable = tuple(dict.fromkeys(untried + advanced))  # dedup, preserve order
            if targetable:
                bottlenecks.append(DecisionBottleneck(
                    kind="CHAIN_INCOMPLETE",
                    severity=HIGH,
                    evidence=(f"stages_completed={chain.stages_completed}",
                              f"advanced_by={chain.actions_that_advanced}",
                              f"failed_actions={chain.actions_that_failed}"),
                    targetable_by=targetable,
                ))

    # If we have cognitive state (aware condition), use it
    vs_val: str | None = None
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
                    targetable_by=_actions_for_evidence_gap(has_retrieval, has_search),
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
    if not res.any_useful_remaining and not bottlenecks:
        bottlenecks.append(DecisionBottleneck(
            kind="RESOURCE_EXHAUSTION",
            severity=HIGH,
            evidence=("no_resources_for_useful_actions",),
            targetable_by=("ANSWER", "DEFER", "STOP"),
        ))

    # Premature-answer guard: never return READY_TO_ANSWER when:
    # 1. verification_state is MISSING (no evidence to answer with)
    # 2. chain is started but not complete (composition chain still needs work)
    # 3. chain is poisoned (task likely unsolvable, prefer DEFER)
    # Only applies to aware condition (where we can see verification_state
    # and chain progress). Blind condition uses its own heuristics.
    can_answer = True
    if cs is not None:
        if vs_val is not None and vs_val == "MISSING":
            can_answer = False
        if chain.is_started and not chain.is_complete and not chain.is_poisoned:
            can_answer = False
        if chain.is_poisoned:
            # Poisoned: prefer DEFER over ANSWER
            can_answer = False
            if not bottlenecks or all(b.kind == "READY_TO_ANSWER" for b in bottlenecks):
                bottlenecks.append(DecisionBottleneck(
                    kind="IRREDUCIBLE_UNCERTAINTY",
                    severity=CRITICAL,
                    evidence=("chain_poisoned", "cannot_answer"),
                    targetable_by=("DEFER", "STOP"),
                ))

    # If no bottlenecks and can answer, ready to answer
    if not bottlenecks and can_answer:
        bottlenecks.append(DecisionBottleneck(
            kind="READY_TO_ANSWER",
            severity=NONE,
            evidence=("no_bottleneck_detected",),
            targetable_by=("ANSWER",),
        ))
    elif not bottlenecks and not can_answer:
        # We have no detected bottleneck but can't answer safely.
        # This happens when verification shows SUFFICIENT but chain
        # tracking says incomplete. Add a chain-incomplete bottleneck.
        bottlenecks.append(DecisionBottleneck(
            kind="CHAIN_INCOMPLETE",
            severity=MEDIUM,
            evidence=("cannot_answer_safely", f"chain_complete={chain.is_complete}",
                      f"verification={vs_val}"),
            targetable_by=_actions_for_evidence_gap(has_retrieval, has_search),
        ))

    # Sort by severity (CRITICAL > HIGH > MEDIUM > LOW > NONE)
    severity_order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, NONE: 4}
    bottlenecks.sort(key=lambda b: severity_order.get(b.severity, 5))
    return tuple(bottlenecks)


def _actions_that_add_new_information(state: GovernorState) -> tuple[str, ...]:
    """Actions that can break a no-gain cycle by adding new information."""
    res = state.resources
    result = []
    if res.has_search and "SEARCH_MORE" in state.legal_actions:
        result.append("SEARCH_MORE")
    if res.has_retrieval and "RETRIEVE" in state.legal_actions:
        result.append("RETRIEVE")
    return tuple(result)


def _actions_for_evidence_gap(has_retrieval: bool, has_search: bool) -> tuple[str, ...]:
    """Actions that can address an evidence gap.

    Only actions in the current V1 seven-action vocabulary.
    """
    result = []
    if has_retrieval:
        result.append("RETRIEVE")
    if has_search:
        result.append("SEARCH_MORE")
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
