#!/usr/bin/env python3
"""Tests for C4 identity stage canonical EXACT detection.

The C4 protocol requires that canonical subjects (entities that are already
in canonical form and need no identity resolution) produce
``IdentityResolution(status="EXACT")`` so that S2c selection activates on
canonical tasks. This was missing in the original implementation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hrm_adaptive_memory.c4.contracts import C4Arm, IdentityResolution, RetrievalResult
from hrm_adaptive_memory.c4.identity_stage import run_identity_stage


# Arm with I3 identity policy enabled
_ARM_I3 = C4Arm(
    arm_id="test", description="test arm",
    query_policy="subject_preserving",
    retrieval_policy="bm25_bge_fusion",
    identity_policy="i3_explicit_identity",
    selector_policy="s2c_with_s0_fallback",
    evidence_policy="bounded_packet",
    packet_budget=6,
)


def _retrieval(candidate_ids):
    return RetrievalResult(
        bm25_ranked=tuple((eid, 1.0) for eid in candidate_ids),
        bge_ranked=tuple((eid, 1.0) for eid in candidate_ids),
        fusion_ranked=tuple((eid, 1.0) for eid in candidate_ids),
        candidate_ids=tuple(candidate_ids),
        candidate_budget=len(candidate_ids),
        retrieval_policy="bm25_bge_fusion",
        bm25_backend="test",
        bge_model_id="test",
        bge_revision="test",
        rrf_k=60,
    )


def test_canonical_subject_produces_exact():
    """A subject that appears in non-identity evidence but has no identity
    record mapping it should be classified as EXACT."""
    texts = {
        "entity_attribute-0000/fact": "During setup, Nimbus sensor array was paired with 1424 for assigned category.",
        "entity_attribute-0001/identity": "QCM-4 is the short code for Quail control module.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Nimbus sensor array?",
        _ARM_I3,
        _retrieval(["entity_attribute-0000/fact", "entity_attribute-0001/identity"]),
        texts,
    )
    assert result.status == "EXACT"
    assert result.canonical == "Nimbus sensor array"
    assert result.surface == "Nimbus sensor array"
    assert result.resolution_needed is False
    assert result.resolution_changed_state is False


def test_abbreviation_subject_produces_resolved():
    """A subject that matches an identity record surface should be RESOLVED."""
    texts = {
        "entity_attribute-0001/fact": "Quail control module was assigned category Beta.",
        "entity_attribute-0001/identity": "QCM-4 is the short code for Quail control module.",
    }
    result = run_identity_stage(
        "Which assigned category applies to QCM-4?",
        _ARM_I3,
        _retrieval(["entity_attribute-0001/fact", "entity_attribute-0001/identity"]),
        texts,
    )
    assert result.status == "RESOLVED"
    assert result.canonical == "Quail control module"
    assert result.resolution_changed_state is True


def test_unknown_subject_produces_unresolved():
    """A subject that appears nowhere in the candidate pool should be UNRESOLVED."""
    texts = {
        "entity_attribute-0000/fact": "Some other entity was assigned category Alpha.",
        "entity_attribute-0001/identity": "QCM-4 is the short code for Quail control module.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Zephyr unknown device?",
        _ARM_I3,
        _retrieval(["entity_attribute-0000/fact", "entity_attribute-0001/identity"]),
        texts,
    )
    assert result.status == "UNRESOLVED"
    assert result.canonical is None


def test_exact_does_not_require_oracle_metadata():
    """EXACT detection uses only runtime-visible evidence content, no oracle."""
    texts = {
        "entity_attribute-0000/fact": "Nimbus sensor array has category 1424.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Nimbus sensor array?",
        _ARM_I3,
        _retrieval(["entity_attribute-0000/fact"]),
        texts,
    )
    assert result.status == "EXACT"
    # No evidence_ids needed for EXACT — no identity record was used
    assert result.evidence_ids == ()


def test_exact_activates_s2c_in_selection():
    """EXACT status should cause the selection stage to use S2c, not S0."""
    from hrm_adaptive_memory.c4.selection_stage import run_selection_stage

    identity = IdentityResolution(
        status="EXACT", surface="Nimbus sensor array",
        canonical="Nimbus sensor array",
        evidence_ids=(), candidate_mappings=(),
        resolution_needed=False, resolution_attempted=True,
        resolution_changed_state=False,
    )
    retrieval = _retrieval([
        "entity_attribute-0000/fact", "entity_attribute-0003/fact",
        "entity_attribute-0005/fact",
    ])
    texts = {
        "entity_attribute-0000/fact": "Nimbus sensor array was paired with 1424 for assigned category.",
        "entity_attribute-0003/fact": "Other entity has category Alpha.",
        "entity_attribute-0005/fact": "Third entity has category Beta.",
    }
    arm = C4Arm(
        arm_id="test", description="test arm",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="i3_explicit_identity",
        selector_policy="s2c_with_s0_fallback",
        evidence_policy="bounded_packet",
        packet_budget=6,
    )
    result = run_selection_stage(
        arm, "Which assigned category applies to Nimbus sensor array?",
        retrieval, identity, texts, required_evidence_ids=[],
    )
    assert result.selector == "s2c"


def test_none_policy_still_unresolved():
    """Arms with identity_policy='none' should never produce EXACT."""
    arm_none = C4Arm(
        arm_id="test", description="test arm",
        query_policy="subject_preserving",
        retrieval_policy="bm25_bge_fusion",
        identity_policy="none",
        selector_policy="s0",
        evidence_policy="bounded_packet",
        packet_budget=6,
    )
    texts = {
        "entity_attribute-0000/fact": "Nimbus sensor array has category 1424.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Nimbus sensor array?",
        arm_none,
        _retrieval(["entity_attribute-0000/fact"]),
        texts,
    )
    assert result.status == "UNRESOLVED"


def test_exact_rejects_substring_distractor_mention():
    """A subject that appears only as a substring inside a distractor entity
    should NOT trigger EXACT. The tightened detection uses parsed entity
    equality, not substring matching.

    Example: 'Falcon control' is a substring of 'Falcon control module', but
    if the evidence only mentions 'Falcon control module', the subject
    'Falcon control' should not be classified as EXACT.
    """
    texts = {
        "distractor-0000/fact": "Some record about Falcon control module was updated.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Falcon control?",
        _ARM_I3,
        _retrieval(["distractor-0000/fact"]),
        texts,
    )
    # 'Falcon control' is NOT a parsed V4 entity in the text —
    # 'Falcon control module' is. So EXACT should not fire.
    assert result.status == "UNRESOLVED"


def test_exact_requires_full_entity_match():
    """The subject must match a full parsed V4 entity, not just appear as
    a substring in the evidence text."""
    texts = {
        "entity_attribute-0000/fact": "During setup, Sparrow intake manifold was paired with 1424.",
    }
    result = run_identity_stage(
        "Which assigned category applies to Sparrow intake manifold?",
        _ARM_I3,
        _retrieval(["entity_attribute-0000/fact"]),
        texts,
    )
    # 'Sparrow intake manifold' IS a parsed V4 entity in the text
    assert result.status == "EXACT"
    assert result.canonical == "Sparrow intake manifold"
