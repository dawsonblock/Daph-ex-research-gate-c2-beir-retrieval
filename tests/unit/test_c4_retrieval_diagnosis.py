"""Tests for scripts/diagnose_c4_retrieval.py.

The script's conclusion -- that the retrieval "collapse" is a fixed-absolute
candidate budget meeting a 4.15x larger corpus, not a representation failure
-- rests entirely on two classifications being right: which evidence role a
required record plays, and why it missed the pool. Both are pinned here.

The cause taxonomy in particular carries weight: FUSION_DISPLACEMENT vs
BELOW_BUDGET is the difference between a fix that costs nothing (change the
fusion rule, same k) and one that needs a new protocol version (change the
budget, and then re-measure the selector against a bigger pool).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_diagnose_c4_retrieval", ROOT / "scripts/diagnose_c4_retrieval.py")
diag = importlib.util.module_from_spec(_spec)
sys.modules["_diagnose_c4_retrieval"] = diag
_spec.loader.exec_module(diag)

BUDGET = 50


def _task(*, answer_node="T#value", bridge=None, edges=()):
    return {
        "task_id": "t-1",
        "_oracle_metadata": {
            "answer_node": answer_node,
            "latent_bridge": bridge,
            "proof_edges": list(edges),
        },
    }


class TestRoleDerivation:
    """Roles come from the task's own proof graph, not from id text."""

    def test_edge_terminating_at_answer_node_is_terminal_answer(self):
        task = _task(edges=[{"record_id": "T/value", "source": "T#subject",
                             "target": "T#value"}])
        assert diag.role_of("T/value", task, {}) == "TERMINAL_ANSWER"

    def test_edge_terminating_at_bridge_is_bridge(self):
        task = _task(bridge="T#bridge",
                     edges=[{"record_id": "T/link", "source": "T#subject",
                             "target": "T#bridge"}])
        assert diag.role_of("T/link", task, {}) == "BRIDGE"

    def test_record_kind_marks_identity(self):
        assert diag.role_of("T/identity", _task(),
                            {"T/identity": "required_identity"}) == "IDENTITY"

    def test_record_kind_marks_temporal_current(self):
        assert diag.role_of("T/current", _task(),
                            {"T/current": "required_current"}) == "TEMPORAL_CURRENT"

    def test_identity_kind_wins_over_proof_edge_position(self):
        """An identity record can also appear as an edge; the kind is
        authoritative, otherwise identity recall would be undercounted."""
        task = _task(edges=[{"record_id": "T/identity", "source": "T#subject",
                             "target": "T#value"}])
        assert diag.role_of("T/identity", task,
                            {"T/identity": "required_identity"}) == "IDENTITY"

    def test_unmatched_record_falls_back_to_supporting(self):
        assert diag.role_of("T/other", _task(), {}) == "SUPPORTING"

    def test_no_oracle_metadata_does_not_crash(self):
        assert diag.role_of("T/x", {"task_id": "t"}, {}) == "SUPPORTING"


class TestCauseClassification:
    def test_in_pool_is_not_a_failure(self):
        """Guards against silently classifying a record that actually made it."""
        with pytest.raises(AssertionError):
            diag.cause_of(10, 10, 10, BUDGET)

    def test_ranked_just_past_the_cutoff_is_below_budget(self):
        assert diag.cause_of(51, 60, 70, BUDGET) == "BELOW_BUDGET"

    def test_bm25_had_it_in_budget_but_fusion_dropped_it(self):
        """The cheap-fix case: RRF discarded what a retriever already had."""
        assert diag.cause_of(80, 37, 900, BUDGET) == "FUSION_DISPLACEMENT"

    def test_dense_had_it_in_budget_but_fusion_dropped_it(self):
        assert diag.cause_of(80, 900, 42, BUDGET) == "FUSION_DISPLACEMENT"

    def test_displacement_wins_over_below_budget(self):
        """Both descriptions can be literally true; displacement is the more
        specific and actionable one, so it must take precedence."""
        assert diag.cause_of(51, 50, 800, BUDGET) == "FUSION_DISPLACEMENT"

    def test_absent_from_fusion_but_ranked_by_one_retriever(self):
        assert diag.cause_of(None, 120, None, BUDGET) == "DENSE_MISS"
        assert diag.cause_of(None, None, 120, BUDGET) == "LEXICAL_MISS"

    def test_both_retrievers_rank_it_far_away_is_representation_failure(self):
        assert diag.cause_of(None, 900, 900, BUDGET) == "BOTH_RETRIEVERS_MISS"

    def test_entirely_unranked_is_unranked(self):
        assert diag.cause_of(None, None, None, BUDGET) == "UNRANKED"

    def test_every_cause_is_declared(self):
        """A cause the report cannot name would vanish from the summary."""
        produced = {
            diag.cause_of(51, 60, 70, BUDGET),
            diag.cause_of(80, 37, 900, BUDGET),
            diag.cause_of(None, 120, None, BUDGET),
            diag.cause_of(None, None, 120, BUDGET),
            diag.cause_of(None, 900, 900, BUDGET),
            diag.cause_of(None, None, None, BUDGET),
        }
        assert produced <= set(diag.CAUSES)


class TestRankOf:
    def test_rank_is_one_indexed(self):
        assert diag.rank_of(["a", "b", "c"], "a") == 1
        assert diag.rank_of(["a", "b", "c"], "c") == 3

    def test_absent_is_none_not_zero(self):
        """None must not be confusable with rank 0 in comparisons."""
        assert diag.rank_of(["a"], "zzz") is None


class TestReusesFrozenPipelineComponents:
    """The diagnosis is only valid if it observes the SAME retrieval a real run
    performs. A local reimplementation of corpus loading already drifted once
    (it dropped token_count), which would have changed what the backends
    indexed, so these are imported from the runner rather than copied."""

    def test_corpus_loading_is_imported_from_the_runner(self):
        from scripts.run_gate_c4 import _load_split, _to_index_records
        assert diag.load_split is _load_split
        assert diag.to_index_records is _to_index_records

    def test_fusion_is_the_frozen_rrf(self):
        from hrm_adaptive_memory.c4.retrieval_stage import _rrf
        assert diag._rrf is _rrf

    def test_budget_and_rrf_k_come_from_frozen_contracts(self):
        from hrm_adaptive_memory.c4.contracts import (
            C4_CANDIDATE_BUDGET, C4_RRF_K)
        assert diag.C4_CANDIDATE_BUDGET == C4_CANDIDATE_BUDGET
        assert diag.C4_RRF_K == C4_RRF_K
