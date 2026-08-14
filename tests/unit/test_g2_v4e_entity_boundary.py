"""G2-v4E entity-boundary normalization.

The repair is a GRAMMAR rule, not a stopword list: the V4 generator's entity
grammar is closed (canonical = head + one of six two-word roles; alias = head +
that role's first word), so no legitimate entity name extends past a completed
role suffix. These tests pin both directions -- that real tails are stripped,
and that legitimate names are NOT over-stripped, which is the opposite bug a
stopword expansion would risk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.bridge_extraction import (
    extract_v4_entities, normalize_v4_entity_boundary)
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph

BOUNDARY_CORPUS = (Path(__file__).resolve().parents[2]
                   / "configs/g2_v4e_boundary_failures.json")


class TestBoundaryNormalization:
    @pytest.mark.parametrize("raw,expected", [
        # the four tails that dominate the frozen corpus audit
        ("Jacana control module resolves", "Jacana control module"),
        ("Osprey intake manifold uses", "Osprey intake manifold"),
        ("Gannet relay unit set", "Gannet relay unit"),
        ("Vireo pressure assembly value", "Vireo pressure assembly"),
        ("Kestrel sensor array as", "Kestrel sensor array"),
        # multi-token tails
        ("Dunlin drive cluster changed to", "Dunlin drive cluster"),
        ("Quail control module parameter grade", "Quail control module"),
    ])
    def test_strips_grammatical_tails(self, raw, expected):
        assert normalize_v4_entity_boundary(raw) == expected

    @pytest.mark.parametrize("legit", [
        "Jacana control module", "Osprey intake manifold", "Gannet relay unit",
        "Vireo pressure assembly", "Curlew sensor array", "Quail drive cluster",
    ])
    def test_does_not_overstrip_complete_canonical_names(self, legit):
        """BOUNDARY_OVERSTRIPPED guard: a complete canonical name must survive
        untouched. A stopword-expansion fix could easily truncate these."""
        assert normalize_v4_entity_boundary(legit) == legit

    @pytest.mark.parametrize("alias", [
        "Falcon control", "Nimbus sensor", "Pelican pressure",
        "Raven relay", "Bittern intake", "Teal drive",
    ])
    def test_preserves_the_alias_form(self, alias):
        """alias = head + role's FIRST word (two tokens) is also legitimate and
        must not be truncated to the bare head."""
        assert normalize_v4_entity_boundary(alias) == alias

    def test_leaves_phrases_without_a_role_suffix_untouched(self):
        """Non-entity text has no role suffix, so the rule must be a no-op --
        this is what keeps the repair from truncating arbitrary prose."""
        for phrase in ("Wren telemetry probe", "Abbreviation table", "Changelog entry"):
            assert normalize_v4_entity_boundary(phrase) == phrase

    def test_idempotent(self):
        once = normalize_v4_entity_boundary("Jacana control module resolves")
        assert normalize_v4_entity_boundary(once) == once


class TestExtractorPolicyArms:
    #: real corpus record whose extraction carries a trailing 'set'
    TEXT = "Changelog: assigned category for Dunlin pressure assembly set to GAMMA-BLUE."

    def test_E0_legacy_is_the_default_and_unchanged(self):
        assert extract_v4_entities(self.TEXT) == extract_v4_entities(
            self.TEXT, boundary_policy="legacy")

    def test_E0_legacy_still_exhibits_the_defect(self):
        """The control arm must reproduce the bug, or the comparison is void."""
        assert any(e.endswith(" set") for e in extract_v4_entities(self.TEXT))

    def test_E1_grammar_v4_repairs_it(self):
        entities = extract_v4_entities(self.TEXT, boundary_policy="grammar_v4")
        assert "Dunlin pressure assembly" in entities
        assert not any(e.endswith(" set") for e in entities)

    def test_unknown_policy_fails_closed(self):
        with pytest.raises(ValueError, match="boundary_policy"):
            extract_v4_entities(self.TEXT, boundary_policy="whatever")

    def test_policy_does_not_invent_or_drop_entities(self):
        """E1 changes boundaries only -- it must not change how MANY distinct
        entities a record yields (modulo dedup collapsing two spellings of the
        same name into one, which is the point)."""
        legacy = extract_v4_entities(self.TEXT)
        repaired = extract_v4_entities(self.TEXT, boundary_policy="grammar_v4")
        assert len(repaired) <= len(legacy)
        assert repaired


class TestGraphIntegration:
    #: link record and value record naming the SAME bridge entity, but the value
    #: record's extraction picks up a trailing 'resolves' under E0
    TEXTS = {
        "link": "Registry entry: Sparrow control module — registered asset — Finch relay unit.",
        "value": "Note: the service tier for Finch relay unit resolves to platinum.",
    }

    def test_E0_fragments_the_bridge_entity(self):
        graph = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                    relation="service tier")
        assert "finch relay unit resolves" in graph.records_by_entity

    def test_E1_unifies_the_bridge_entity_into_one_node(self):
        graph = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                    relation="service tier",
                                    boundary_policy="grammar_v4")
        assert "finch relay unit resolves" not in graph.records_by_entity
        assert {"link", "value"} <= graph.records_by_entity["finch relay unit"]

    def test_E1_reconnects_the_value_record_to_the_anchor(self):
        """The actual mechanism claim, stated precisely.

        The bridge ENTITY is adjacent to the subject under both arms, because
        the link record co-mentions them and extracts cleanly. What differs is
        which node the VALUE record attaches to: under E0 it hangs off the
        fragmented 'finch relay unit resolves' node, which nothing reaches, so
        traversal from the subject cannot discover the value record. Under E1 it
        attaches to the same 'finch relay unit' node the subject is adjacent to,
        making it discoverable -- that is the reachability recovery v4E predicts.
        """
        e0 = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                 relation="service tier")
        e1 = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                 relation="service tier",
                                 boundary_policy="grammar_v4")

        def records_reachable_from(graph, anchor):
            """Records hanging off the anchor or any of its neighbours."""
            reachable = set(graph.records_by_entity.get("sparrow control module", set()))
            for entity in graph.neighbours(anchor):
                reachable |= graph.records_by_entity.get(entity, set())
            return reachable

        assert "value" not in records_reachable_from(e0, "Sparrow control module")
        assert "value" in records_reachable_from(e1, "Sparrow control module")

    def test_default_graph_build_is_unchanged(self):
        a = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                relation="service tier")
        b = build_runtime_graph(record_ids=list(self.TEXTS), texts=self.TEXTS,
                                relation="service tier", boundary_policy="legacy")
        assert [e.as_dict() for e in a.edges] == [e.as_dict() for e in b.edges]


class TestProcessLevelPolicy:
    """Entity boundary parsing is a pipeline-wide treatment variable (it moves
    reachability by tens of points), and extraction happens in several modules,
    so a run sets it ONCE. These tests pin that it defaults to legacy, that it
    actually takes effect, and that it is always restored."""

    def test_default_is_legacy(self):
        from hrm_adaptive_memory.c4.bridge_extraction import get_default_boundary_policy
        assert get_default_boundary_policy() == "legacy"

    def test_setting_the_default_changes_extraction_without_an_explicit_arg(self):
        from hrm_adaptive_memory.c4.bridge_extraction import (
            get_default_boundary_policy, set_default_boundary_policy)
        text = "Changelog: assigned category for Dunlin pressure assembly set to GAMMA-BLUE."
        original = get_default_boundary_policy()
        try:
            assert any(e.endswith(" set") for e in extract_v4_entities(text))
            set_default_boundary_policy("grammar_v4")
            assert not any(e.endswith(" set") for e in extract_v4_entities(text))
        finally:
            set_default_boundary_policy(original)
        assert get_default_boundary_policy() == "legacy"

    def test_explicit_argument_overrides_the_default(self):
        from hrm_adaptive_memory.c4.bridge_extraction import set_default_boundary_policy
        text = "Changelog: assigned category for Dunlin pressure assembly set to GAMMA-BLUE."
        try:
            set_default_boundary_policy("grammar_v4")
            assert any(e.endswith(" set")
                       for e in extract_v4_entities(text, boundary_policy="legacy"))
        finally:
            set_default_boundary_policy("legacy")

    def test_unknown_default_policy_fails_closed(self):
        from hrm_adaptive_memory.c4.bridge_extraction import set_default_boundary_policy
        with pytest.raises(ValueError, match="boundary_policy"):
            set_default_boundary_policy("made_up")

    def test_config_hash_changes_with_policy(self):
        """The hash must distinguish arms, or receipts cannot prove which
        extractor produced a given graph."""
        from hrm_adaptive_memory.c4.bridge_extraction import (
            entity_extractor_config_hash, set_default_boundary_policy)
        try:
            set_default_boundary_policy("legacy")
            legacy_hash = entity_extractor_config_hash()
            set_default_boundary_policy("grammar_v4")
            grammar_hash = entity_extractor_config_hash()
        finally:
            set_default_boundary_policy("legacy")
        assert legacy_hash != grammar_hash
        assert len(legacy_hash) == 16 and len(grammar_hash) == 16

    def test_graph_builder_follows_the_process_default(self):
        from hrm_adaptive_memory.c4.bridge_extraction import set_default_boundary_policy
        texts = {"value": "Note: the service tier for Finch relay unit resolves to platinum."}
        try:
            set_default_boundary_policy("grammar_v4")
            graph = build_runtime_graph(record_ids=list(texts), texts=texts,
                                        relation="service tier")
            assert "finch relay unit" in graph.records_by_entity
            assert "finch relay unit resolves" not in graph.records_by_entity
        finally:
            set_default_boundary_policy("legacy")


class TestFrozenBoundaryCorpus:
    def test_corpus_artifact_exists_and_is_grounded(self):
        data = json.loads(BOUNDARY_CORPUS.read_text())
        assert data["total_extractions_scanned"] > 0
        assert data["distinct_boundary_failures"] > 0
        for entry in data["entries"][:20]:
            for field in ("raw_extraction", "expected_entity", "trailing_token",
                          "source_template_family", "count"):
                assert field in entry

    def test_every_frozen_failure_is_actually_repaired_by_E1(self):
        """The repair must cover the frozen corpus it was derived from -- a
        regression net against a future extractor change silently reopening it."""
        data = json.loads(BOUNDARY_CORPUS.read_text())
        for entry in data["entries"]:
            assert normalize_v4_entity_boundary(entry["raw_extraction"]) == \
                entry["expected_entity"], entry
