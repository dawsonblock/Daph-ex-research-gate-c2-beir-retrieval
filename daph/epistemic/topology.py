"""Canonical hypothesis topology derivation.

Implements derive_hypothesis_topology() per EPISTEMIC_SEMANTICS_V1.md §5.

This is the single function that all consumers must use to derive
epistemic state from observable evidence. No component may independently
re-derive hypothesis states from raw evidence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from daph.epistemic.types import (
    HypothesisState,
    HypothesisTopology,
    TerminalReadiness,
)


def derive_hypothesis_topology(
    evidence_items: Sequence[dict | Any],
    hypothesis_ids: Sequence[str],
    *,
    hidden_evidence_count: int = 0,
) -> HypothesisTopology:
    """Derive the canonical hypothesis topology from observable evidence.

    Args:
        evidence_items: Visible (retrieved) evidence items. Each must have:
            - evidence_id: str
            - supports: sequence of hypothesis IDs
            - contradicts: sequence of hypothesis IDs
            - verification_state: VerificationState or str (SUFFICIENT/FALSIFIED/UNVERIFIED/STALE/MISSING)
            - temporal_status: TemporalStatus or str (CURRENT/STALE/UNKNOWN)
            Can be EvidenceItem dataclass instances or dicts.
        hypothesis_ids: All hypothesis IDs in the task.
        hidden_evidence_count: Count of non-retrieved evidence items (not their content).

    Returns:
        HypothesisTopology: The canonical topology.

    Observability:
        Consumes ONLY observable fields. Does NOT inspect:
        - verify_result
        - correct_hypothesis_id
        - expected_terminal
        - oracle_resolution_path
        - hidden evidence content
        - future actions/outcomes
    """
    # Normalize evidence items to a common interface
    normalized = []
    for ev in evidence_items:
        if isinstance(ev, dict):
            ev_id = ev["evidence_id"]
            supports = tuple(ev.get("supports", ()))
            contradicts = tuple(ev.get("contradicts", ()))
            vstate_raw = ev.get("verification_state", "UNVERIFIED")
            tstatus_raw = ev.get("temporal_status", "CURRENT")
            retrieved = ev.get("retrieved", True)
        else:
            # Assume dataclass-like object (EvidenceItem)
            ev_id = ev.evidence_id
            supports = tuple(ev.supports)
            contradicts = tuple(ev.contradicts)
            vstate_raw = ev.verification_state
            tstatus_raw = ev.temporal_status
            retrieved = getattr(ev, "retrieved", True)

        # Normalize verification state
        if isinstance(vstate_raw, str):
            vstate = VerificationState(vstate_raw)
        else:
            vstate = vstate_raw

        # Normalize temporal status
        if isinstance(tstatus_raw, str):
            tstatus = TemporalStatus(tstatus_raw)
        else:
            tstatus = tstatus_raw

        # Skip non-retrieved evidence (should not be passed, but defense-in-depth)
        if not retrieved:
            continue

        normalized.append({
            "evidence_id": ev_id,
            "supports": supports,
            "contradicts": contradicts,
            "verification_state": vstate,
            "temporal_status": tstatus,
        })

    all_hyp_ids = list(hypothesis_ids)

    # Per-hypothesis evidence tracking
    verified_support: dict[str, list[str]] = defaultdict(list)
    verified_contradiction: dict[str, list[str]] = defaultdict(list)
    falsified_support: dict[str, list[str]] = defaultdict(list)
    falsified_contradiction: dict[str, list[str]] = defaultdict(list)
    unverified_support: dict[str, list[str]] = defaultdict(list)
    unverified_contradiction: dict[str, list[str]] = defaultdict(list)

    # Track which hypotheses have any evidence at all (for STALE vs UNTESTED)
    hyp_has_any_evidence: set[str] = set()
    hyp_has_current_evidence: set[str] = set()

    for ev in normalized:
        vstate = ev["verification_state"]
        tstatus = ev["temporal_status"]

        # Skip STALE temporal status entirely — no current evidential force
        is_current = (tstatus == TemporalStatus.CURRENT)

        for h_id in ev["supports"]:
            hyp_has_any_evidence.add(h_id)
            if is_current:
                hyp_has_current_evidence.add(h_id)
            if vstate == VerificationState.SUFFICIENT and is_current:
                verified_support[h_id].append(ev["evidence_id"])
            elif vstate == VerificationState.FALSIFIED and is_current:
                falsified_support[h_id].append(ev["evidence_id"])
            elif vstate == VerificationState.UNVERIFIED and is_current:
                unverified_support[h_id].append(ev["evidence_id"])
            # STALE/MISSING: no effect

        for h_id in ev["contradicts"]:
            hyp_has_any_evidence.add(h_id)
            if is_current:
                hyp_has_current_evidence.add(h_id)
            if vstate == VerificationState.SUFFICIENT and is_current:
                verified_contradiction[h_id].append(ev["evidence_id"])
            elif vstate == VerificationState.FALSIFIED and is_current:
                falsified_contradiction[h_id].append(ev["evidence_id"])
            elif vstate == VerificationState.UNVERIFIED and is_current:
                unverified_contradiction[h_id].append(ev["evidence_id"])
            # STALE/MISSING: no effect

    # Classify each hypothesis (§4)
    hypothesis_states: dict[str, HypothesisState] = {}
    n_supported = 0
    n_contradicted = 0
    n_weakened = 0
    n_untested = 0
    n_stale = 0

    n_hyp_with_verified_support = 0
    n_hyp_with_verified_contradiction = 0
    n_hyp_with_mixed_verified = 0

    for h_id in all_hyp_ids:
        has_vs = len(verified_support.get(h_id, [])) > 0
        has_vc = len(verified_contradiction.get(h_id, [])) > 0
        has_fs = len(falsified_support.get(h_id, [])) > 0
        has_fc = len(falsified_contradiction.get(h_id, [])) > 0

        if has_vs:
            n_hyp_with_verified_support += 1
        if has_vc:
            n_hyp_with_verified_contradiction += 1
        if has_vs and has_vc:
            n_hyp_with_mixed_verified += 1

        # Classification per §4 priority:
        # CONTRADICTED > SUPPORTED > WEAKENED > STALE > UNTESTED
        if has_vc:
            state = HypothesisState.CONTRADICTED
            n_contradicted += 1
        elif has_vs:
            state = HypothesisState.SUPPORTED
            n_supported += 1
        elif has_fs:
            state = HypothesisState.WEAKENED
            n_weakened += 1
        elif h_id in hyp_has_any_evidence and h_id not in hyp_has_current_evidence:
            # Has evidence but all of it is temporally stale
            state = HypothesisState.STALE
            n_stale += 1
        else:
            state = HypothesisState.UNTESTED
            n_untested += 1

        hypothesis_states[h_id] = state

    # Resolution state
    supported_hyps = [h_id for h_id, s in hypothesis_states.items()
                      if s == HypothesisState.SUPPORTED]
    unique_supported = supported_hyps[0] if len(supported_hyps) == 1 else None
    has_unresolved_competition = n_hyp_with_verified_support > 1
    has_unique_verified_supported = n_hyp_with_verified_support == 1

    # Evidence completeness
    verification_complete = all(
        ev["verification_state"] in (VerificationState.SUFFICIENT, VerificationState.FALSIFIED)
        for ev in normalized
    ) and len(normalized) > 0
    unverified_evidence_exists = any(
        ev["verification_state"] == VerificationState.UNVERIFIED
        for ev in normalized
    )

    return HypothesisTopology(
        hypothesis_states=dict(hypothesis_states),
        n_viable_hypotheses=n_supported,
        n_eliminated_hypotheses=n_contradicted,
        n_untested_hypotheses=n_untested,
        n_weakened_hypotheses=n_weakened,
        n_stale_hypotheses=n_stale,
        n_total_hypotheses=len(all_hyp_ids),
        n_hyp_with_verified_support=n_hyp_with_verified_support,
        n_hyp_with_verified_contradiction=n_hyp_with_verified_contradiction,
        n_hyp_with_mixed_verified=n_hyp_with_mixed_verified,
        unique_supported_hypothesis=unique_supported,
        has_verified_unresolved_competition=has_unresolved_competition,
        has_unique_verified_supported=has_unique_verified_supported,
        verification_complete=verification_complete,
        unverified_evidence_exists=unverified_evidence_exists,
        hidden_evidence_count=hidden_evidence_count,
        verified_support_by_hypothesis={h: tuple(v) for h, v in verified_support.items()},
        verified_contradiction_by_hypothesis={h: tuple(v) for h, v in verified_contradiction.items()},
        falsified_support_by_hypothesis={h: tuple(v) for h, v in falsified_support.items()},
        falsified_contradiction_by_hypothesis={h: tuple(v) for h, v in falsified_contradiction.items()},
        unverified_support_by_hypothesis={h: tuple(v) for h, v in unverified_support.items()},
        unverified_contradiction_by_hypothesis={h: tuple(v) for h, v in unverified_contradiction.items()},
    )


def classify_terminal_readiness(
    topology: HypothesisTopology,
    *,
    can_verify: bool,
    can_retrieve: bool,
    can_search: bool,
    has_unverified_discriminating_evidence: bool = False,
    has_hidden_evidence: bool = False,
    search_could_discriminate: bool = False,
) -> TerminalReadiness:
    """Classify terminal readiness from topology and action admissibility.

    Implements EPISTEMIC_SEMANTICS_V1 §6.

    Args:
        topology: The canonical hypothesis topology.
        can_verify: Whether VERIFY is admissible (budget remaining + valid targets).
        can_retrieve: Whether RETRIEVE is admissible (budget remaining + hidden evidence).
        can_search: Whether SEARCH_MORE is admissible (budget remaining + not already searched).
        has_unverified_discriminating_evidence: Whether unverified visible evidence
            exists that could discriminate between hypotheses (support or contradict
            a SUPPORTED or competing hypothesis).
        has_hidden_evidence: Whether hidden evidence exists (retrievable).
        search_could_discriminate: Whether search could produce discriminating evidence.

    Returns:
        TerminalReadiness: One of ANSWER_READY, DEFER_READY, CONTINUE_REQUIRED.
    """
    # §6.1: ANSWER_READY
    if is_answer_ready(topology):
        return TerminalReadiness.ANSWER_READY

    # §6.3: Check if any admissible continuation exists that could resolve
    continuation_could_resolve = (
        (can_verify and has_unverified_discriminating_evidence)
        or (can_retrieve and has_hidden_evidence)
        or (can_search and search_could_discriminate)
    )

    # §6.2: DEFER_READY
    if not continuation_could_resolve:
        return TerminalReadiness.DEFER_READY

    # §6.3: CONTINUE_REQUIRED
    return TerminalReadiness.CONTINUE_REQUIRED


def is_answer_ready(topology: HypothesisTopology) -> bool:
    """Check if the state is ANSWER_READY per §6.1."""
    return (
        topology.unique_supported_hypothesis is not None
        and not topology.has_verified_unresolved_competition
    )


def is_defer_ready(
    topology: HypothesisTopology,
    *,
    can_verify: bool,
    can_retrieve: bool,
    can_search: bool,
    has_unverified_discriminating_evidence: bool = False,
    has_hidden_evidence: bool = False,
    search_could_discriminate: bool = False,
) -> bool:
    """Check if the state is DEFER_READY per §6.2."""
    if is_answer_ready(topology):
        return False
    return not _continuation_could_resolve(
        can_verify, can_retrieve, can_search,
        has_unverified_discriminating_evidence,
        has_hidden_evidence, search_could_discriminate,
    )


def is_continue_required(
    topology: HypothesisTopology,
    *,
    can_verify: bool,
    can_retrieve: bool,
    can_search: bool,
    has_unverified_discriminating_evidence: bool = False,
    has_hidden_evidence: bool = False,
    search_could_discriminate: bool = False,
) -> bool:
    """Check if the state is CONTINUE_REQUIRED per §6.3."""
    if is_answer_ready(topology):
        return False
    return _continuation_could_resolve(
        can_verify, can_retrieve, can_search,
        has_unverified_discriminating_evidence,
        has_hidden_evidence, search_could_discriminate,
    )


def _continuation_could_resolve(
    can_verify: bool,
    can_retrieve: bool,
    can_search: bool,
    has_unverified_discriminating_evidence: bool,
    has_hidden_evidence: bool,
    search_could_discriminate: bool,
) -> bool:
    """Check whether any admissible continuation could materially resolve the state."""
    return (
        (can_verify and has_unverified_discriminating_evidence)
        or (can_retrieve and has_hidden_evidence)
        or (can_search and search_could_discriminate)
    )
