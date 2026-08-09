"""S4 path-coherent packet composition: the frozen contract, pinned.

Atomicity is the load-bearing property -- a path either survives whole or is
not admitted as a path -- so it is asserted directly rather than inferred from
aggregate CES.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from hrm_adaptive_memory.c4.packet_composition import (
    NOT_COMPUTABLE, complete_path_packet, complete_paths_represented,
    compose_path_coherent_packet, cross_path_fragmentation,
    packet_coherence_ratio, path_order_key)


@dataclass
class FakePath:
    path_id: str
    record_ids: tuple
    tier: int = 1
    hop_count: int = 1
    retrieval_score: float = 1.0
    complete: bool = True


class TestFrozenOrdering:
    def test_orders_by_tier_then_hop_then_score(self):
        a = FakePath("a", ("r1",), tier=0, hop_count=1, retrieval_score=0.5)
        b = FakePath("b", ("r2",), tier=1, hop_count=0, retrieval_score=9.0)
        assert path_order_key(a) < path_order_key(b), "tier dominates"

    def test_path_id_is_the_final_tie_break(self):
        a = FakePath("aaa", ("r1",))
        b = FakePath("bbb", ("r1",))
        assert path_order_key(a) < path_order_key(b)

    def test_ordering_is_total_and_deterministic(self):
        paths = [FakePath(f"p{i}", (f"r{i}",)) for i in range(5)]
        assert ([p.path_id for p in sorted(paths, key=path_order_key)]
                == [p.path_id for p in sorted(reversed(paths), key=path_order_key)])


class TestAtomicAdmission:
    def test_admits_the_top_path_whole(self):
        paths = [FakePath("p1", ("a", "b"), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=["z", "y"],
            working_set=["a", "b", "z", "y"], packet_budget=6)
        assert r.path_admitted_atomically
        assert r.packet[:2] == ["a", "b"]
        assert r.anchor_path_id == "p1"

    def test_never_cherry_picks_across_paths_in_the_anchor_stage(self):
        """The exact S3 failure mode: two records from path A, two from path B.
        The anchor stage must contribute records from ONE path only."""
        paths = [FakePath("pA", ("a1", "a2"), tier=0),
                 FakePath("pB", ("b1", "b2"), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=[], working_set=["a1","a2","b1","b2"],
            packet_budget=6)
        assert set(r.anchor_path_records) == {"a1", "a2"}
        assert not ({"b1", "b2"} & set(r.anchor_path_records))

    def test_skips_a_path_too_large_for_the_budget(self):
        """All-or-nothing: an oversized path is not partially admitted."""
        big = FakePath("big", tuple(f"x{i}" for i in range(9)), tier=0)
        small = FakePath("small", ("s1", "s2"), tier=1)
        r = compose_path_coherent_packet(
            complete_paths=[big, small], s2_ordering=[],
            working_set=[f"x{i}" for i in range(9)] + ["s1", "s2"], packet_budget=6)
        assert r.anchor_path_id == "small"
        assert not (set(r.packet) & {f"x{i}" for i in range(9)})

    def test_no_complete_path_degrades_to_pure_s2_ordering(self):
        r = compose_path_coherent_packet(
            complete_paths=[], s2_ordering=["a", "b", "c"],
            working_set=["a", "b", "c"], packet_budget=6)
        assert r.packet == ["a", "b", "c"]
        assert not r.path_admitted_atomically

    def test_ignores_path_records_absent_from_the_working_set(self):
        paths = [FakePath("p1", ("a", "missing"), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=["z"], working_set=["a", "z"],
            packet_budget=6)
        assert "missing" not in r.packet

    def test_never_exceeds_the_budget(self):
        paths = [FakePath("p1", ("a", "b"), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=[f"f{i}" for i in range(20)],
            working_set=["a","b"] + [f"f{i}" for i in range(20)], packet_budget=6)
        assert len(r.packet) == 6

    def test_no_duplicates(self):
        paths = [FakePath("p1", ("a", "b"), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=["a", "b", "c"],
            working_set=["a", "b", "c"], packet_budget=6)
        assert len(r.packet) == len(set(r.packet))

    def test_fill_uses_unchanged_s2_order(self):
        paths = [FakePath("p1", ("a",), tier=0)]
        r = compose_path_coherent_packet(
            complete_paths=paths, s2_ordering=["z", "y", "x"],
            working_set=["a", "z", "y", "x"], packet_budget=4)
        assert r.fill_records == ["z", "y", "x"]

    def test_is_deterministic(self):
        paths = [FakePath("pA", ("a1","a2")), FakePath("pB", ("b1","b2"))]
        kw = dict(complete_paths=paths, s2_ordering=["z"],
                  working_set=["a1","a2","b1","b2","z"], packet_budget=6)
        assert (compose_path_coherent_packet(**kw).packet
                == compose_path_coherent_packet(**kw).packet)


class TestCoherenceMetrics:
    def test_complete_path_packet_binary(self):
        paths = [FakePath("p1", ("a", "b"))]
        assert complete_path_packet(["a", "b", "z"], paths) == 1
        assert complete_path_packet(["a", "z"], paths) == 0

    def test_pcr_is_one_for_a_single_coherent_path(self):
        paths = [FakePath("p1", ("a", "b"))]
        assert packet_coherence_ratio(["a", "b"], paths) == 1.0

    def test_pcr_detects_cross_path_fragmentation(self):
        paths = [FakePath("pA", ("a1", "a2")), FakePath("pB", ("b1", "b2"))]
        # two from each path: best single path covers 2 of 4 path-relevant records
        assert packet_coherence_ratio(["a1", "a2", "b1", "b2"], paths) == 0.5
        assert cross_path_fragmentation(["a1", "a2", "b1", "b2"], paths) == 0.5

    def test_empty_denominator_is_not_computable_not_zero(self):
        paths = [FakePath("p1", ("a", "b"))]
        result = packet_coherence_ratio(["z", "y"], paths)
        assert result == NOT_COMPUTABLE
        assert result != 0.0

    def test_paths_represented_counts_distinct_paths(self):
        paths = [FakePath("pA", ("a1",)), FakePath("pB", ("b1",))]
        assert complete_paths_represented(["a1", "b1"], paths) == 2
        assert complete_paths_represented(["a1"], paths) == 1
