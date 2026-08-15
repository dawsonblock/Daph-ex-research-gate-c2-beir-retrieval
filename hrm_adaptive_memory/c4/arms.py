"""C4 arm definitions — declarative configurations over shared stages.

Each arm differs from its predecessor in exactly one policy field, which is
what makes arm parity mechanically checkable.
"""
from __future__ import annotations

from .contracts import C4Arm, C4_PRIMARY_PACKET_BUDGET

ARMS: dict[str, C4Arm] = {
    "C4_0": C4Arm(
        arm_id="C4_0",
        description="current historical R1 pipeline baseline",
        query_policy="original",
        retrieval_policy="bm25_only",
        identity_policy="none",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        classification="BASELINE",
    ),
    "C4_1": C4Arm(
        arm_id="C4_1",
        description="C4-0 + subject-preserving information state + frozen follow-up query",
        query_policy="subject_preserving",
        retrieval_policy="bm25_only",
        identity_policy="none",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        classification="PRIMARY",
    ),
    "C4_2": C4Arm(
        arm_id="C4_2",
        description="C4-1 + BM25/BGE multi-signal candidate generation",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="none",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        classification="PRIMARY",
    ),
    "C4_3": C4Arm(
        arm_id="C4_3",
        description="C4-2 + I3 identity-record resolution + canonicalized state",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        classification="PRIMARY",
    ),
    "C4_4": C4Arm(
        arm_id="C4_4",
        description="C4-3 + S2c structure+relation selection with S0 fallback",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s2c_with_s0_fallback",
        evidence_policy="bounded_packet",
        classification="PRIMARY",
    ),
    "C4_4b": C4Arm(
        arm_id="C4_4b",
        description="diagnostic: if resolved -> S2c; else -> s_rel_only",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s2c_with_srel_fallback",
        evidence_policy="bounded_packet",
        classification="DIAGNOSTIC",
    ),
    "C4_5": C4Arm(
        arm_id="C4_5",
        description="same real pool + oracle selector (ceiling for selection)",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="oracle",
        evidence_policy="bounded_packet",
        classification="CEILING",
    ),
    "C4_6": C4Arm(
        arm_id="C4_6",
        description="oracle required evidence (R5-style reader/evidence ceiling)",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="oracle_evidence",
        evidence_policy="bounded_packet",
        classification="CEILING",
    ),
    # ── Diagnostic arms for ordering vs membership separation ──────────
    # C4_3o: S0 membership + deterministic packet ordering (ordering-only)
    # C4_4m: S2c membership + pool-order (membership-only, no deterministic order)
    # These let you decompose Q(C4_4) - Q(C4_3) into:
    #   ordering effect  = Q(C4_3o) - Q(C4_3)
    #   membership effect = Q(C4_4m) - Q(C4_3)
    #   combined effect   = Q(C4_4) - Q(C4_3)
    "C4_3o": C4Arm(
        arm_id="C4_3o",
        description="diagnostic: S0 membership + deterministic packet ordering (ordering-only effect)",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        classification="DIAGNOSTIC",
    ),
    "C4_4m": C4Arm(
        arm_id="C4_4m",
        description="diagnostic: S2c membership + pool-order (membership-only effect, no deterministic order)",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s2c_with_s0_fallback",
        evidence_policy="bounded_packet",
        classification="DIAGNOSTIC",
    ),
}

# Primary arm order for the development ladder
PRIMARY_ORDER = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]

# Diagnostic arm order (run after primary if desired)
DIAGNOSTIC_ORDER = ["C4_3o", "C4_4m"]

# Parity pairs: (arm_a, arm_b, expected_diff_field)
PARITY_PAIRS = [
    ("C4_2", "C4_3", "identity_policy"),
    ("C4_3", "C4_4", "selector_policy"),
    ("C4_4", "C4_5", "selector_policy"),
]
