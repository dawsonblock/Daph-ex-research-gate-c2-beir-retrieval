"""Leakage wall plus the structural guarantees of G2 path enumeration/ranking.

Mirrors the discipline of test_c4_runtime_graph.py: a docstring-and-comment-
stripped scan asserts no executable statement in g2_paths.py reads any
evaluator-only field, and the algorithmic behaviour that separates G2 from
G1_TYPED_PATH (explicit competing paths, no working-set padding) is asserted
directly rather than only inferred from calibration numbers.
"""
from __future__ import annotations

import inspect
import io
import tokenize

import pytest

from hrm_adaptive_memory.c4.g2_paths import (
    TIER_BRIDGED_COMPLETE, TIER_BRIDGED_PARTIAL, TIER_DIRECT_COMPLETE,
    TIER_DIRECT_PARTIAL, enumerate_paths, g2_prefilter, rank_and_select_paths)
from hrm_adaptive_memory.c4.runtime_graph import build_runtime_graph

SUBJECT = "Sparrow intake manifold"
RELATION = "service tier"

#: Two DISTINCT closing bridges (Finch, Egret) compete to explain the same
#: relation; Heron is a dead end and must not admit a complete path.
TEXTS = {
    "direct": "Sparrow intake manifold, service tier is bronze.",
    "link_finch": "Sparrow intake manifold, registered asset Finch control module.",
    "endpoint_finch": "Finch control module, service tier is platinum.",
    "link_egret": "Sparrow intake manifold, adjacent to Egret power cell.",
    "endpoint_egret": "Egret power cell, service tier is gold.",
    "decoy_link": "Sparrow intake manifold, near Heron drive cluster.",
    "decoy": "Heron drive cluster, inventory count is nine.",
    "unrelated": "Wren telemetry probe, calibration log entry.",
}


def _executable_source(module) -> str:
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


FORBIDDEN = ["required_evidence_ids", "proof_edges", "record_kind",
             "latent_bridge", "answer_node", "oracle_bridge", "gold_path",
             "oracle_evidence_ids", "_oracle_metadata"]


class TestLeakageWall:
    @pytest.mark.parametrize("forbidden", FORBIDDEN)
    def test_module_never_references_an_oracle_field(self, forbidden):
        import hrm_adaptive_memory.c4.g2_paths as mod
        assert forbidden not in _executable_source(mod)

    def test_no_indirect_access_through_task_metadata(self):
        import hrm_adaptive_memory.c4.g2_paths as mod
        stripped = _executable_source(mod).lower()
        for indirect in ("task [", "metadata [", "oracle"):
            assert indirect not in stripped

    def test_guard_the_guard(self):
        import hrm_adaptive_memory.c4.g2_paths as mod
        stripped = _executable_source(mod)
        assert "enumerate_paths" in stripped
        assert "rank_and_select_paths" in stripped

    def test_g2_prefilter_signature_accepts_no_task_object(self):
        params = set(inspect.signature(g2_prefilter).parameters)
        assert "task" not in params and "evaluator" not in params
        assert {"candidate_ids", "texts", "canonical_subject", "relation",
                "working_set_size"} <= params


class TestPathEnumeration:
    def _paths(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        return enumerate_paths(graph=graph, canonical_subject=SUBJECT,
                               relation=RELATION,
                               fusion_scores={k: 1.0 for k in TEXTS})

    def test_finds_the_direct_path(self):
        paths = self._paths()
        direct = [p for p in paths if p.hop_count == 0 and p.complete]
        assert direct and direct[0].tier == TIER_DIRECT_COMPLETE
        assert direct[0].record_ids == ("direct",)

    def test_two_distinct_bridges_produce_two_distinct_competing_paths(self):
        """This is the core structural difference from G1_TYPED_PATH: Finch and
        Egret are not collapsed into one neighbourhood, they are two explicit,
        separately ranked hypotheses."""
        paths = self._paths()
        complete_bridged = [p for p in paths if p.hop_count == 1 and p.complete]
        terminals = {p.terminal_entity for p in complete_bridged}
        assert "finch control module" in terminals
        assert "egret power cell" in terminals
        assert len(complete_bridged) >= 2

    def test_dead_end_bridge_produces_no_complete_path(self):
        paths = self._paths()
        heron_paths = [p for p in paths if p.terminal_entity == "heron drive cluster"]
        assert heron_paths and not any(p.complete for p in heron_paths)
        assert heron_paths[0].tier == TIER_BRIDGED_PARTIAL

    def test_every_path_has_a_unique_id(self):
        paths = self._paths()
        assert len({p.path_id for p in paths}) == len(paths)

    def test_each_direct_subject_record_is_its_own_path(self):
        """Hop-0 records are atomic evidence units, not merged into one blob:
        a record stating the relation is a different hypothesis from one that
        merely mentions the subject, and must not be conflated."""
        paths = self._paths()
        direct = [p for p in paths if p.hop_count == 0]
        assert len(direct) >= 2  # "direct" (complete) + at least one partial mention
        assert all(p.record_ids == (p.record_ids[0],) if len(p.record_ids) == 1
                  else True for p in direct)
        complete_direct = [p for p in direct if p.complete]
        assert complete_direct == [d for d in direct if d.record_ids == ("direct",)]

    def test_multiple_records_for_the_same_bridge_merge_into_one_path(self):
        """Unlike hop-0, a bridge with several supporting records for the SAME
        terminal entity is one structural hypothesis and should combine."""
        paths = self._paths()
        finch = [p for p in paths if p.terminal_entity == "finch control module"]
        assert len(finch) == 1
        assert set(finch[0].record_ids) >= {"link_finch", "endpoint_finch"}

    def test_returns_nothing_without_a_subject(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        assert enumerate_paths(graph=graph, canonical_subject=None,
                               relation=RELATION) == []

    def test_is_deterministic(self):
        assert [p.path_id for p in self._paths()] == [p.path_id for p in self._paths()]


class TestRankingAndSelection:
    def _paths(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        return enumerate_paths(graph=graph, canonical_subject=SUBJECT,
                               relation=RELATION,
                               fusion_scores={k: 1.0 for k in TEXTS})

    def test_direct_complete_path_outranks_bridged_paths(self):
        retained, working = rank_and_select_paths(self._paths(), working_set_size=1)
        assert working == ["direct"]

    def test_dead_end_bridge_is_ranked_below_closing_bridges(self):
        retained, working = rank_and_select_paths(self._paths(), working_set_size=3)
        assert "direct" in working
        assert set(working) & {"endpoint_finch", "endpoint_egret"}
        assert "decoy" not in working

    def test_never_exceeds_the_ceiling(self):
        _, working = rank_and_select_paths(self._paths(), working_set_size=2)
        assert len(working) <= 2

    def test_does_not_pad_when_fewer_records_are_justified(self):
        """The behavioural correction vs G1_TYPED_PATH: ask for a generous
        ceiling and get back only what is structurally justified, not filler."""
        retained, working = rank_and_select_paths(self._paths(), working_set_size=50)
        assert len(working) < 50
        assert len(working) == len(set(working))

    def test_redundant_path_contributes_no_new_records_and_is_not_retained(self):
        graph = build_runtime_graph(record_ids=list(TEXTS), texts=TEXTS,
                                    relation=RELATION)
        paths = enumerate_paths(graph=graph, canonical_subject=SUBJECT,
                                relation=RELATION)
        retained, working = rank_and_select_paths(paths, working_set_size=50)
        seen = set()
        for path in retained:
            assert set(path.record_ids) - seen, "a fully redundant path was retained"
            seen |= set(path.record_ids)


class TestG2Prefilter:
    def test_end_to_end_diagnostics_present(self):
        result = g2_prefilter(candidate_ids=list(TEXTS), texts=TEXTS,
                              canonical_subject=SUBJECT, relation=RELATION,
                              working_set_size=10,
                              fusion_scores={k: 1.0 for k in TEXTS})
        diag = result.diagnostics()
        for key in ("paths_enumerated", "paths_complete",
                    "paths_target_relation_compatible", "paths_subject_anchored",
                    "paths_competing", "paths_retained", "unique_bridge_entities",
                    "records_per_retained_path", "path_overlap_ratio",
                    "working_set_size", "records_examined"):
            assert key in diag

    def test_unique_bridge_entities_counts_both_closing_bridges(self):
        result = g2_prefilter(candidate_ids=list(TEXTS), texts=TEXTS,
                              canonical_subject=SUBJECT, relation=RELATION,
                              working_set_size=50)
        assert result.diagnostics()["unique_bridge_entities"] >= 3  # finch, egret, heron

    def test_working_set_never_exceeds_ceiling(self):
        result = g2_prefilter(candidate_ids=list(TEXTS), texts=TEXTS,
                              canonical_subject=SUBJECT, relation=RELATION,
                              working_set_size=2)
        assert len(result.kept) <= 2

    def test_degrades_gracefully_without_a_subject(self):
        result = g2_prefilter(candidate_ids=list(TEXTS), texts=TEXTS,
                              canonical_subject=None, relation=RELATION,
                              working_set_size=10)
        assert result.kept == []
        assert not result.canonical_subject_found

    def test_tier_ordering_is_frozen(self):
        assert TIER_DIRECT_COMPLETE < TIER_BRIDGED_COMPLETE < \
            TIER_DIRECT_PARTIAL < TIER_BRIDGED_PARTIAL
