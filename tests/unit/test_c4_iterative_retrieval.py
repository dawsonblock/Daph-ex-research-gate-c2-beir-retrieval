#!/usr/bin/env python3
"""Tests for C4 iterative (two-pass) retrieval.

Verifies that the subject_preserving query policy performs two-pass retrieval
when a bridge is discovered, and that the merged candidate pool includes
results from both passes.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hrm_adaptive_memory.c4.bridge_extraction import extract_bridge, extract_v4_entities


def test_bridge_extraction_finds_bridge_in_link_record():
    """Bridge extraction should find the bridge entity in a link record."""
    texts = {
        "task-0000/link": "The registered asset registry records that Sparrow intake manifold is assigned Finch control module.",
        "task-0000/value": "Finch control module: ownership tier now 3529",
    }
    bridge = extract_bridge(
        "Sparrow intake manifold",
        "Which ownership tier applies to Sparrow intake manifold?",
        tuple(texts.keys()), texts)
    assert bridge == "Finch control module"


def test_bridge_extraction_returns_none_for_single_hop():
    """Single-hop tasks (answer-bearing records) should not trigger bridge extraction."""
    texts = {
        "entity_attribute-0000/fact": "During setup, Nimbus sensor array was paired with 1424 for assigned category.",
    }
    bridge = extract_bridge(
        "Nimbus sensor array",
        "Which assigned category applies to Nimbus sensor array?",
        tuple(texts.keys()), texts)
    assert bridge is None


def test_bridge_extraction_excludes_subject():
    """The bridge should never be the subject itself."""
    texts = {
        "task-0000/link": "Sparrow intake manifold is assigned Finch control module.",
    }
    bridge = extract_bridge(
        "Sparrow intake manifold",
        "Which ownership tier applies to Sparrow intake manifold?",
        tuple(texts.keys()), texts)
    assert bridge is not None
    assert bridge.lower() != "sparrow intake manifold"


def test_v4_entity_extraction():
    """V4 entity extraction should find multi-word proper nouns."""
    ents = extract_v4_entities("Sparrow intake manifold is assigned Finch control module.")
    assert "Sparrow intake manifold" in ents
    assert "Finch control module" in ents


def test_v4_entity_extraction_excludes_non_entities():
    """V4 entity extraction should not match sentence-initial non-entities."""
    ents = extract_v4_entities("The registered asset registry records that X is assigned Y.")
    assert "The registered asset registry" not in ents


def test_information_state_with_bridge():
    """InformationState.with_bridge should accumulate, not replace."""
    from hrm_adaptive_memory.retrieval.information_state import (
        InformationState, formulate_followup, FOLLOWUP_FORMULATION)

    state = InformationState(subject="Sparrow intake manifold", target_relation="ownership tier")
    assert state.bridge is None

    state_with_bridge = state.with_bridge("Finch control module")
    assert state_with_bridge.subject == "Sparrow intake manifold"  # subject retained
    assert state_with_bridge.bridge == "Finch control module"
    assert state_with_bridge.target_relation == "ownership tier"  # relation retained
    assert state_with_bridge.hop == 1


def test_formulate_followup_with_bridge():
    """formulate_followup should produce 'subject bridge relation' when bridge exists."""
    from hrm_adaptive_memory.retrieval.information_state import (
        InformationState, formulate_followup, FOLLOWUP_FORMULATION)

    state = InformationState(subject="Sparrow intake manifold", target_relation="ownership tier")
    state = state.with_bridge("Finch control module")
    query = formulate_followup(state, formulation=FOLLOWUP_FORMULATION)
    assert "Sparrow intake manifold" in query  # subject retained
    assert "Finch control module" in query  # bridge included
    assert "ownership tier" in query  # relation retained


def test_formulate_followup_without_bridge():
    """formulate_followup without bridge should produce 'subject relation'."""
    from hrm_adaptive_memory.retrieval.information_state import (
        InformationState, formulate_followup, FOLLOWUP_FORMULATION)

    state = InformationState(subject="Nimbus sensor array", target_relation="assigned category")
    query = formulate_followup(state, formulation=FOLLOWUP_FORMULATION)
    assert "Nimbus sensor array" in query
    assert "assigned category" in query
    # No bridge term
    assert query == "Nimbus sensor array assigned category"
