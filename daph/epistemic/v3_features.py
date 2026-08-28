"""I3.30R: V3 feature computation using canonical epistemic topology.

This replaces the buggy compute_v3_features() in run_i3_30_v3_coverage.py
which incorrectly treated FALSIFIED as a verified state.

The canonical topology from daph.epistemic implements the normative
semantics defined in EPISTEMIC_SEMANTICS_V1.md.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from hrm_adaptive_memory.cognitive_control.state import VerificationState

from daph.epistemic import derive_hypothesis_topology, HypothesisState


def compute_v3_features_canonical(
    evidence: Sequence[dict],
    hypotheses: Sequence[dict],
) -> dict:
    """Compute V3 structural features using canonical epistemic topology.

    This function implements EPISTEMIC_SEMANTICS_V1.md §3.2:
    - SUFFICIENT + supports(H) → verified support for H
    - SUFFICIENT + contradicts(H) → verified contradiction against H
    - FALSIFIED + supports(H) → no effect (support claim failed)
    - FALSIFIED + contradicts(H) → no effect (contradiction claim failed)

    Args:
        evidence: List of evidence dicts with keys:
            evidence_id, supports, contradicts, verification_state, temporal_status, retrieved
        hypotheses: List of hypothesis dicts with keys:
            hypothesis_id, answer_action

    Returns:
        dict with V3 structural features.
    """
    # Extract hypothesis IDs
    hyp_ids = [h["hypothesis_id"] if isinstance(h, dict) else h.hypothesis_id
               for h in hypotheses]

    # Derive canonical topology
    topo = derive_hypothesis_topology(evidence, hyp_ids)

    # Extract verified_hyp_action from the uniquely supported hypothesis
    verified_hyp_action = None
    if topo.unique_supported_hypothesis is not None:
        h_id = topo.unique_supported_hypothesis
        for h in hypotheses:
            if (h["hypothesis_id"] if isinstance(h, dict) else h.hypothesis_id) == h_id:
                verified_hyp_action = (h.get("answer_action") if isinstance(h, dict)
                                       else h.answer_action.value
                                       if hasattr(h.answer_action, "value")
                                       else str(h.answer_action))
                break

    # Also compute V2R features for backward compatibility
    # V2R features use the same canonical semantics for unverified evidence
    n_hyp_unverified_support = sum(
        1 for h_id in hyp_ids
        if len(topo.unverified_support_by_hypothesis.get(h_id, ())) > 0
    )
    n_hyp_unverified_contradiction = sum(
        1 for h_id in hyp_ids
        if len(topo.unverified_contradiction_by_hypothesis.get(h_id, ())) > 0
    )
    has_competing_unverified_support = n_hyp_unverified_support > 1

    return {
        # V3 canonical features (from topology)
        "n_hyp_with_verified_support": topo.n_hyp_with_verified_support,
        "n_hyp_with_verified_contradiction": topo.n_hyp_with_verified_contradiction,
        "n_hyp_with_mixed_verified": topo.n_hyp_with_mixed_verified,
        "n_viable_hypotheses": topo.n_viable_hypotheses,
        "n_eliminated_hypotheses": topo.n_eliminated_hypotheses,
        "has_unique_verified_supported_hypothesis": int(topo.has_unique_verified_supported),
        "has_verified_unresolved_competition": int(topo.has_verified_unresolved_competition),
        "verified_hyp_action": verified_hyp_action,
        "verified_hyp_action_is_answer": int(verified_hyp_action == "ANSWER"),
        "verified_hyp_action_is_defer": int(verified_hyp_action == "DEFER"),
        # V2R features (for backward compatibility)
        "n_hyp_unverified_support": n_hyp_unverified_support,
        "n_hyp_unverified_contradiction": n_hyp_unverified_contradiction,
        "has_competing_unverified_support": int(has_competing_unverified_support),
        # Additional topology features
        "n_untested_hypotheses": topo.n_untested_hypotheses,
        "n_weakened_hypotheses": topo.n_weakened_hypotheses,
        "verification_complete": int(topo.verification_complete),
        "unverified_evidence_exists": int(topo.unverified_evidence_exists),
        "hidden_evidence_count": topo.hidden_evidence_count,
    }
