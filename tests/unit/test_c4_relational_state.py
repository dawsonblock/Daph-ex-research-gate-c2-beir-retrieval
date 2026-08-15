"""Tests for C4 runtime relational state — relation parsing and bridge discovery."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hrm_adaptive_memory.c4.relational_state import (
    RelationFact, RelationalState, parse_relation_edges,
    build_relational_state, get_bridge, is_target_bound,
    _relation_matches, _normalize_relation,
)


# --- Relation edge parsing ---

def test_parse_semicolon_kv():
    """subject=X; RELATION=Y format."""
    edges = parse_relation_edges(
        "subject=Osprey relay unit; registered asset=Marlin pressure assembly",
        "test/link")
    assert len(edges) == 1
    assert edges[0].source_entity == "Osprey relay unit"
    assert edges[0].relation == "registered asset"
    assert edges[0].target_entity == "Marlin pressure assembly"


def test_parse_json_kv():
    """JSON format."""
    edges = parse_relation_edges(
        '{"subject": "Jacana drive cluster", "registered asset": "Heron intake manifold"}',
        "test/link")
    assert len(edges) == 1
    assert edges[0].source_entity == "Jacana drive cluster"
    assert edges[0].relation == "registered asset"
    assert edges[0].target_entity == "Heron intake manifold"


def test_parse_bracket_arrow():
    """[RELATION] X -> Y format."""
    edges = parse_relation_edges(
        "[registered asset] Pelican control module -> Bluebird sensor array",
        "test/link")
    assert len(edges) == 1
    assert edges[0].source_entity == "Pelican control module"
    assert edges[0].relation == "registered asset"
    assert edges[0].target_entity == "Bluebird sensor array"


def test_parse_changelog_set():
    """Changelog: RELATION for X set to Y format."""
    edges = parse_relation_edges(
        "Changelog: active configuration for Ibis relay unit set to Auk relay unit.",
        "test/link")
    assert len(edges) == 1
    assert edges[0].source_entity == "Ibis relay unit"
    assert edges[0].relation == "active configuration"
    assert edges[0].target_entity == "Auk relay unit"


def test_parse_engineering_notes():
    """Engineering notes indicate X uses Y as its RELATION format."""
    edges = parse_relation_edges(
        "Engineering notes indicate Curlew pressure assembly uses Bluebird sensor array as its registered asset.",
        "test/link")
    assert len(edges) == 1
    assert edges[0].source_entity == "Curlew pressure assembly"
    assert edges[0].relation == "registered asset"
    assert edges[0].target_entity == "Bluebird sensor array"


def test_parse_no_match_returns_empty():
    """Non-link content returns no edges."""
    edges = parse_relation_edges(
        "During setup, Nimbus sensor array was paired with 1424 for assigned category.",
        "test/fact")
    assert edges == []


# --- Relational state building ---

def test_build_state_extracts_facts():
    """Building state from candidate pool extracts relation facts."""
    texts = {
        "test/link-0": "subject=Sparrow intake manifold; registered asset=Finch control module",
        "test/fact-0": "Sparrow intake manifold has category 1424.",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "ownership tier",
        ("test/link-0", "test/fact-0"), texts,
        question="Which ownership tier applies to Sparrow intake manifold?")
    assert len(state.known_relations) >= 1
    assert any(f.source_entity == "Sparrow intake manifold" for f in state.known_relations)


def test_build_state_target_bound_when_answer_present():
    """Target is bound when a fact record contains the answer."""
    texts = {
        "test/fact-0": "Sparrow intake manifold has ownership tier 1424.",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "ownership tier",
        ("test/fact-0",), texts)
    assert state.target_bound is True


def test_build_state_target_not_bound_for_multi_hop():
    """Target is NOT bound when only link records exist (multi-hop)."""
    texts = {
        "test/link-0": "subject=Sparrow intake manifold; registered asset=Finch control module",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "ownership tier",
        ("test/link-0",), texts)
    assert state.target_bound is False


def test_get_bridge_returns_none_when_bound():
    """No bridge needed when target is already bound."""
    texts = {
        "test/fact-0": "Sparrow intake manifold has ownership tier 1424.",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "ownership tier",
        ("test/fact-0",), texts)
    assert get_bridge(state) is None


def test_get_bridge_returns_candidate_when_not_bound():
    """Bridge is returned when target is not bound and edges exist."""
    texts = {
        "test/link-0": "subject=Sparrow intake manifold; registered asset=Finch control module",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "ownership tier",
        ("test/link-0",), texts)
    bridge = get_bridge(state)
    assert bridge is not None
    assert bridge == "Finch control module"


def test_get_bridge_returns_none_for_single_hop():
    """No bridge for single-hop tasks (target already bound)."""
    texts = {
        "test/fact-0": "Sparrow intake manifold has registered asset Finch control module.",
    }
    state = build_relational_state(
        "Sparrow intake manifold", "registered asset",
        ("test/fact-0",), texts)
    assert get_bridge(state) is None


# --- Relation matching ---

def test_relation_matches_direct():
    assert _relation_matches("registered asset", "registered asset")


def test_relation_matches_partial():
    assert _relation_matches("registered asset", "registered")


def test_relation_matches_synonym():
    assert _relation_matches("assigned category", "category")


def test_relation_no_match_empty_target():
    assert not _relation_matches("registered asset", "")


def test_relation_no_match_different():
    assert not _relation_matches("registered asset", "ownership tier")


# --- RelationFact normalization ---

def test_relation_fact_normalizes_relation():
    fact = RelationFact(
        source_entity="X", relation="Registered Asset",
        target_entity="Y", evidence_id="test")
    assert fact.relation == "registered asset"


def test_relation_fact_strips_entities():
    fact = RelationFact(
        source_entity="  X  ", relation="rel",
        target_entity="  Y  ", evidence_id="test")
    assert fact.source_entity == "X"
    assert fact.target_entity == "Y"
