"""G1 leakage wall plus the structural guarantees of the runtime graph.

The leakage class is the important one. The runtime graph would look excellent
and mean nothing if any part of it could read the evaluator's proof graph, so
that separation is asserted mechanically rather than trusted.
"""
from __future__ import annotations

import inspect
import io
import json
import tokenize
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.runtime_graph import (
    EDGE_TYPES, MAX_HOPS, NODE_TYPES, bounded_neighborhood, build_runtime_graph)
from hrm_adaptive_memory.c4.typed_path import (
    TIER_DIRECT, TIER_ENDPOINT, TIER_FILLER, TIER_LINK, typed_path_prefilter)

PROTOCOL = Path(__file__).resolve().parents[2] / "configs/gate_g1_runtime_graph_v1.json"

SUBJECT = "Sparrow intake manifold"
RELATION = "service tier"

#: subject -> Finch (link) -> Finch has the target relation (endpoint).
#: Heron is co-mentioned with the subject but leads nowhere: it must NOT admit
#: records, which is exactly what B4's coarse prefilter got wrong.
TEXTS = {
    "link": "Sparrow intake manifold registered asset Finch control module.",
    "endpoint": "Finch control module service tier is platinum.",
    "decoy_link": "Sparrow intake manifold adjacent to Heron drive cluster.",
    "decoy": "Heron drive cluster inventory count is nine.",
    "unrelated": "Wren telemetry probe calibration log entry.",
    "direct": "Sparrow intake manifold service tier is bronze.",
}


def _executable_source(module) -> str:
    """Module source with docstrings and comments stripped.

    This module's prose legitimately NAMES the forbidden signals to explain why
    they are forbidden, so a naive substring scan over raw source would flag its
    own documentation. What matters is that no executable statement reads them.
    """
    source = inspect.getsource(module)
    kept: list[str] = []
    previous_type = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and previous_type in (
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
                tokenize.NL, tokenize.ENCODING):
            previous_type = token.type
            continue
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous_type = token.type
    return " ".join(kept)


def _graph_modules():
    import hrm_adaptive_memory.c4.runtime_graph as rg
    import hrm_adaptive_memory.c4.typed_path as tp
    return [rg, tp]


FORBIDDEN = ["required_evidence_ids", "proof_edges", "record_kind",
             "latent_bridge", "answer_node", "oracle_bridge", "gold_path",
             "oracle_evidence_ids", "_oracle_metadata"]


class TestLeakageWall:
    @pytest.mark.parametrize("forbidden", FORBIDDEN)
    def test_no_graph_module_references_an_oracle_field(self, forbidden):
        for module in _graph_modules():
            assert forbidden not in _executable_source(module), (
                f"{module.__name__} references {forbidden}; the runtime graph "
                "must never see the evaluator proof graph")

    def test_no_indirect_access_through_task_metadata(self):
        """A leak through ``task[...]`` or ``metadata[...]`` is the same leak."""
        for module in _graph_modules():
            stripped = _executable_source(module)
            for indirect in ("task [", "metadata [", "oracle"):
                assert indirect not in stripped.lower(), (
                    f"{module.__name__} reaches for {indirect!r}; graph "
                    "construction takes record text and a question only")

    def test_the_scan_would_actually_catch_a_violation(self):
        """Guards the guard: a scan matching nothing passes vacuously."""
        import hrm_adaptive_memory.c4.runtime_graph as rg
        import hrm_adaptive_memory.c4.typed_path as tp
        assert "build_runtime_graph" in _executable_source(rg)
        assert "bounded_neighborhood" in _executable_source(rg)
        assert "typed_path_prefilter" in _executable_source(tp)

    def test_protocol_lists_the_same_forbidden_signals(self):
        spec = json.loads(PROTOCOL.read_text())["forbidden_absolutely"]
        for name in ("required_evidence_ids", "proof_edges", "record_kind",
                     "latent_bridge", "answer_node", "oracle_bridge", "gold_path"):
            assert name in spec

    def test_builder_signature_accepts_no_task_object(self):
        params = set(inspect.signature(build_runtime_graph).parameters)
        assert params == {"record_ids", "texts", "relation"}


class TestGraphConstruction:
    def test_every_edge_carries_full_provenance(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        assert graph.edges
        for edge in graph.edges:
            data = edge.as_dict()
            for field_name in ("edge_id", "edge_type", "source_record_id",
                               "extraction_method", "source_span_hash",
                               "parser_version"):
                assert data[field_name], f"{field_name} missing on {edge.edge_type}"
            assert edge.edge_type in EDGE_TYPES

    def test_alias_edges_carry_real_record_provenance(self):
        """Non-vacuous provenance test for alias edges specifically.

        TestGraphConstruction's other provenance test uses a fixture with NO
        identity records, so it passed vacuously for ENTITY_ALIASES_ENTITY /
        ENTITY_RESOLVES_TO_ENTITY -- which is how a real bug survived: the
        builder read link.evidence_id while IdentityLink declares .record_id,
        emitting source_record_id='' on every alias edge.
        """
        texts = {
            "ident": "JPA-5 is the short code for Jacana pressure assembly.",
            "fact": "Changelog: assigned category for Jacana pressure assembly set to GAMMA-BLUE.",
        }
        graph = build_runtime_graph(record_ids=list(texts), texts=texts,
                                    relation="assigned category")
        alias_edges = [e for e in graph.edges
                       if e.edge_type in ("ENTITY_ALIASES_ENTITY",
                                          "ENTITY_RESOLVES_TO_ENTITY")]
        assert alias_edges, "fixture must actually produce alias edges"
        for edge in alias_edges:
            assert edge.source_record_id == "ident", (
                f"alias edge provenance lost: {edge.as_dict()}")

    def test_alias_links_connect_both_directions(self):
        """Connectivity (as distinct from provenance) must work: the identity
        record is the evidence that creates the alias<->canonical connection,
        so it must not require the alias to already be connected."""
        texts = {"ident": "JPA-5 is the short code for Jacana pressure assembly."}
        graph = build_runtime_graph(record_ids=list(texts), texts=texts, relation="")
        assert "jacana pressure assembly" in graph.neighbours("JPA-5")
        assert "jpa 5" in graph.neighbours("Jacana pressure assembly")

    def test_only_permitted_edge_types_are_emitted(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        assert {e.edge_type for e in graph.edges} <= EDGE_TYPES

    def test_relation_edges_come_from_visible_text(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        assert "endpoint" in graph.relation_records
        assert "unrelated" not in graph.relation_records

    def test_build_is_deterministic(self):
        ids = list(TEXTS)
        first = build_runtime_graph(record_ids=ids, texts=TEXTS, relation=RELATION)
        second = build_runtime_graph(record_ids=ids, texts=TEXTS, relation=RELATION)
        assert [e.as_dict() for e in first.edges] == [e.as_dict() for e in second.edges]

    def test_node_types_are_declared(self):
        assert {"ENTITY", "RECORD", "RELATION"} <= NODE_TYPES


class TestHopBound:
    def test_refuses_to_exceed_the_frozen_bound(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        with pytest.raises(ValueError, match="MAX_HOPS"):
            bounded_neighborhood(graph, [SUBJECT], hops=MAX_HOPS + 1)

    def test_two_hops_reaches_further_than_one(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        one, _ = bounded_neighborhood(graph, [SUBJECT], hops=1)
        two, _ = bounded_neighborhood(graph, [SUBJECT], hops=2)
        assert one <= two

    def test_traversal_counts_are_reported(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        _, stats = bounded_neighborhood(graph, [SUBJECT], hops=2)
        assert stats["nodes_visited"] > 0 and stats["edges_traversed"] > 0


class TestTypedPath:
    def _run(self, m: int = 3):
        return typed_path_prefilter(
            candidate_ids=list(TEXTS), texts=TEXTS, canonical_subject=SUBJECT,
            relation=RELATION, working_set_size=m,
            fusion_scores={k: 1.0 for k in TEXTS})

    def test_completes_the_subject_bridge_relation_path(self):
        result = self._run()
        assert result.typed_path_completed
        assert "endpoint" in result.kept

    def test_a_dead_end_bridge_admits_nothing(self):
        """Heron is one hop from the subject but its records never express the
        relation, so it must not admit records. This is the discrimination B4's
        coarse prefilter lacked: it kept anything near any co-mentioned entity."""
        result = self._run(m=6)
        tiers = {rid: t for rid, t in zip(result.kept, [None] * len(result.kept))}
        del tiers
        assert result.closing_bridge_entities < result.one_hop_bridge_candidates
        assert result.filler_count > 0

    def test_direct_answer_records_are_representable(self):
        result = self._run(m=1)
        assert result.kept == ["direct"], (
            "tier 0 must outrank bridged paths; a task whose answer sits on the "
            "subject has no bridge to find")

    def test_respects_the_working_set_bound(self):
        assert len(self._run(m=2).kept) == 2

    def test_is_deterministic(self):
        assert self._run().kept == self._run().kept

    def test_reports_path_hit_diagnostics(self):
        diagnostics = self._run().diagnostics()
        for key in ("canonical_subject_found", "target_relation_extracted",
                    "one_hop_bridge_candidates", "closing_bridge_entities",
                    "typed_paths_found", "typed_path_completed",
                    "working_set_size", "records_examined"):
            assert key in diagnostics

    def test_degrades_to_retrieval_rank_without_a_subject(self):
        result = typed_path_prefilter(
            candidate_ids=list(TEXTS), texts=TEXTS, canonical_subject=None,
            relation=RELATION, working_set_size=3)
        assert not result.canonical_subject_found
        assert len(result.kept) == 3

    def test_tier_constants_are_ordered(self):
        assert TIER_DIRECT < TIER_ENDPOINT < TIER_LINK < TIER_FILLER
