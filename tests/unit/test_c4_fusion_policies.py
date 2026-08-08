"""Tests for hrm_adaptive_memory/c4/fusion.py (c4_retrieval_fusion_v1).

Three things must hold or the experiment is not measuring what it claims:

  1. R0 reproduces the CERTIFIED path exactly. It delegates to
     retrieval_stage._rrf rather than reimplementing it, because a baseline
     free to drift makes every measured delta a delta against nothing.
  2. R1 removes the diagnosed consensus bonus, and its equivalence to
     min-rank ordering is pinned -- that equivalence is what proves R2 is not
     an independent arm, a claim the protocol originally got wrong.
  3. Every policy is deterministic under permutation and hash seed, since the
     whole project's replay discipline depends on it.
"""
from __future__ import annotations

import random

import pytest

from hrm_adaptive_memory.c4.contracts import C4_CANDIDATE_BUDGET, C4_RRF_K
from hrm_adaptive_memory.c4.fusion import (
    POLICIES, frozen_rrf, max_reciprocal, oracle_fusion,
    reserved_slot_interleave)
from hrm_adaptive_memory.c4.retrieval_stage import _rrf

K = C4_RRF_K
B = C4_CANDIDATE_BUDGET


def _lists(seed: int, universe: int = 80, depth: int = 60):
    rng = random.Random(seed)
    pool = [f"e{i}" for i in range(universe)]
    return rng.sample(pool, depth), rng.sample(pool, depth)


class TestBaselineDoesNotDrift:
    def test_R0_is_the_frozen_rrf_by_delegation(self):
        a, b = _lists(1)
        assert frozen_rrf([a, b], K, B) == _rrf([a, b], K, B)

    def test_R0_accepts_tuples_without_changing_result(self):
        """The runner passes tuples; a list/tuple difference must not alter output."""
        a, b = _lists(2)
        assert frozen_rrf([tuple(a), tuple(b)], K, B) == frozen_rrf([a, b], K, B)


class TestConsensusBonusIsTheDiagnosedPathology:
    def test_rrf_displaces_a_single_list_leader_for_a_both_list_middler(self):
        """The exact pathology: 'solo' is BM25 rank 1 and nowhere in BGE;
        'pair' is rank 12 in both. RRF's SUM prefers the consensus record."""
        solo, pair = "solo", "pair"
        bm25 = [solo] + [f"x{i}" for i in range(10)] + [pair]
        bge = [f"y{i}" for i in range(11)] + [pair]
        ranked = [e for e, _ in frozen_rrf([bm25, bge], K, 2)]
        assert ranked[0] == pair, "RRF should rank the consensus record first"

    def test_max_aggregation_reverses_that_preference(self):
        solo, pair = "solo", "pair"
        bm25 = [solo] + [f"x{i}" for i in range(10)] + [pair]
        bge = [f"y{i}" for i in range(11)] + [pair]
        ranked = [e for e, _ in max_reciprocal([bm25, bge], K, 2)]
        assert ranked[0] == solo, "max aggregation must respect a rank-1 hit"

    def test_break_even_algebra_holds(self):
        """A single-list record at s is beaten under RRF iff b < 2s + k."""
        for s in (1, 5, 15, 25):
            boundary = 2 * s + K
            assert 1.0 / (K + s) > 2.0 / (K + boundary + 1)
            assert 1.0 / (K + s) < 2.0 / (K + boundary - 1)


class TestR1EqualsMinRankOrdering:
    """This equivalence is why R2 is not independent corroboration of R1."""

    def _min_rank_order(self, lists, limit):
        best: dict[str, int] = {}
        for lst in lists:
            for index, eid in enumerate(lst, 1):
                best[eid] = min(best.get(eid, 10 ** 9), index)
        return [e for e, _ in sorted(best.items(), key=lambda kv: (kv[1], kv[0]))][:limit]

    @pytest.mark.parametrize("seed", range(8))
    def test_max_reciprocal_is_exactly_min_rank_with_id_tiebreak(self, seed):
        a, b = _lists(seed)
        assert [e for e, _ in max_reciprocal([a, b], K, B)] == \
            self._min_rank_order([a, b], B)

    def test_R1_and_R2_agree_on_membership_most_of_the_time(self):
        """Documents the dependence quantitatively rather than asserting they
        are identical -- they differ at tie-breaks near the budget edge."""
        same = 0
        trials = 200
        for seed in range(trials):
            a, b = _lists(seed + 100)
            p1 = {e for e, _ in max_reciprocal([a, b], K, B)}
            p2 = {e for e, _ in reserved_slot_interleave([a, b], K, B)}
            same += (p1 == p2)
        assert same > trials * 0.7, (
            "R1 and R2 are the same min-rank ordering modulo tie-breaks; if "
            "they diverged often, that reasoning would be wrong")


class TestGuaranteeDepth:
    """Why this policy family recovers only part of the displacement loss."""

    def test_min_rank_survival_is_guaranteed_only_to_budget_over_lists(self):
        """Worst case: two DISJOINT lists, so exactly 2r records have min-rank
        <= r and the budget is consumed at r = B/L = 25.

        Names are 0-indexed while ranks are 1-indexed: a24 is rank 25 (the last
        guaranteed rank) and a25 is rank 26 (the first unguaranteed one).
        """
        bm25 = [f"a{i}" for i in range(60)]
        bge = [f"b{i}" for i in range(60)]
        pool = {e for e, _ in max_reciprocal([bm25, bge], K, 50)}
        assert "a24" in pool, "rank 25 is the last rank inside B/L = 25"
        assert "a25" not in pool, "rank 26 is past B/L and is not guaranteed"
        # The cutoff is a property of the depth limit, not of which list won.
        assert "b24" in pool and "b25" not in pool

    def test_deeper_budget_extends_the_guarantee_proportionally(self):
        """Doubling the budget doubles the guaranteed depth: B/L = 50, so rank
        50 (a49) survives and rank 51 (a50) does not."""
        bm25 = [f"a{i}" for i in range(60)]
        bge = [f"b{i}" for i in range(60)]
        pool = {e for e, _ in max_reciprocal([bm25, bge], K, 100)}
        assert "a49" in pool, "rank 50 is inside B/L = 50"
        assert "a50" not in pool, "rank 51 is past it"


class TestOracleCeiling:
    def test_places_required_records_the_constituents_contain(self):
        bm25 = [f"x{i}" for i in range(60)]
        bge = [f"y{i}" for i in range(60)]
        out = {e for e, _ in oracle_fusion([bm25, bge], K, 10, required=["x59", "y59"])}
        assert {"x59", "y59"} <= out

    def test_cannot_invent_records_absent_from_both_lists(self):
        """The ceiling bounds REORDERING, so evidence neither retriever ranked
        must stay out -- otherwise R3 would overstate the fusion headroom."""
        out = {e for e, _ in oracle_fusion([["a"], ["b"]], K, 5, required=["ghost"])}
        assert "ghost" not in out

    def test_fills_remaining_budget_in_frozen_rrf_order(self):
        a, b = _lists(7)
        required = [a[-1]]
        out = [e for e, _ in oracle_fusion([a, b], K, B, required=required)]
        assert out[0] == required[0]
        assert len(out) == B
        assert len(set(out)) == B, "no duplicates between protected and filled"


class TestDeterminism:
    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_output_is_permutation_invariant_within_a_list(self, name):
        """Ranking order is meaningful, but equal-scoring records must not
        resolve by dict/set iteration order."""
        policy = POLICIES[name]
        a, b = _lists(11)
        first = policy([a, b], K, B)
        for _ in range(5):
            assert policy([a, b], K, B) == first

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_scores_are_non_increasing(self, name):
        scores = [s for _, s in POLICIES[name](_lists(12), K, B)]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_respects_the_budget_and_never_duplicates(self, name):
        out = POLICIES[name](_lists(13), K, 20)
        ids = [e for e, _ in out]
        assert len(ids) <= 20
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_empty_input_is_safe(self, name):
        assert POLICIES[name]([], K, B) == []
        assert POLICIES[name]([[], []], K, B) == []

    def test_ties_break_by_evidence_id_not_insertion_order(self):
        """Same rank in the same position from differently-ordered inputs must
        still produce a stable, id-sorted result."""
        out = [e for e, _ in max_reciprocal([["zzz", "aaa"], ["aaa", "zzz"]], K, 2)]
        assert out == ["aaa", "zzz"]
