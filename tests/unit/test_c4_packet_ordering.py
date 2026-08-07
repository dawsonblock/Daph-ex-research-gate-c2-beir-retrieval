"""Tests for C4 packet ordering — membership vs ordering separation.

Phase 3-4 of the C4 determinism repair.
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.c4.packet_ordering import (
    order_packet,
    canonical_membership_hash,
    canonical_order_hash,
    packet_receipt,
    ORDERING_POLICY_ID,
)


class TestOrderPacket:
    """The frozen packet-ordering policy."""

    def test_identity_before_value(self):
        """Identity records come before value records."""
        ids = ["task/value", "task/identity", "task/link"]
        ordered = order_packet(ids)
        assert ordered[0] == "task/identity"
        assert ordered[1] == "task/link"
        assert ordered[2] == "task/value"

    def test_distractor_last(self):
        """Distractors come last."""
        ids = ["task/distractor", "task/value", "task/identity"]
        ordered = order_packet(ids)
        assert ordered[-1] == "task/distractor"

    def test_same_role_sorted_by_selector_score(self):
        """Within the same role, higher selector score comes first."""
        ids = ["task/value", "task-2/value"]
        scores = {"task/value": 5.0, "task-2/value": 3.0}
        ordered = order_packet(ids, selector_scores=scores)
        assert ordered[0] == "task/value"
        assert ordered[1] == "task-2/value"

    def test_same_role_same_score_sorted_by_record_id(self):
        """Within the same role and score, record_id breaks ties."""
        ids = ["task-2/value", "task-1/value"]
        ordered = order_packet(ids)
        assert ordered[0] == "task-1/value"
        assert ordered[1] == "task-2/value"

    def test_deterministic_across_permutations(self):
        """Same input in different order produces same output."""
        ids = ["task/value", "task/identity", "task/link", "task/dead-end-0"]
        import random
        for seed in range(100):
            shuffled = ids[:]
            random.Random(seed).shuffle(shuffled)
            ordered = order_packet(shuffled)
            assert ordered == order_packet(ids)

    def test_total_order_no_ties(self):
        """No two different records compare equal."""
        ids = [f"task-{i}/value" for i in range(100)]
        ordered = order_packet(ids)
        assert len(set(ordered)) == len(ordered)
        assert ordered == sorted(ids)  # all same role, same score → record_id order


class TestMembershipHash:
    def test_order_independent(self):
        """Membership hash is the same regardless of input order."""
        h1 = canonical_membership_hash(["a", "b", "c"])
        h2 = canonical_membership_hash(["c", "a", "b"])
        assert h1 == h2

    def test_different_members_different_hash(self):
        h1 = canonical_membership_hash(["a", "b"])
        h2 = canonical_membership_hash(["a", "c"])
        assert h1 != h2


class TestOrderHash:
    def test_order_sensitive(self):
        """Order hash differs for different orderings."""
        h1 = canonical_order_hash(["a", "b", "c"])
        h2 = canonical_order_hash(["c", "a", "b"])
        assert h1 != h2

    def test_same_order_same_hash(self):
        h1 = canonical_order_hash(["a", "b", "c"])
        h2 = canonical_order_hash(["a", "b", "c"])
        assert h1 == h2


class TestPacketReceipt:
    def test_contains_all_hashes(self):
        receipt = packet_receipt(
            task_id="task-001",
            selected_ids=["task/value", "task/identity"],
            ordered_ids=["task/identity", "task/value"],
            query_hash="abc123",
            candidate_pool_hash="def456",
            selector_policy_id="s2c_v1",
        )
        assert "membership_hash" in receipt
        assert "order_hash" in receipt
        assert receipt["ordering_policy_id"] == ORDERING_POLICY_ID
        assert receipt["selected_set_ids"] == ["task/identity", "task/value"]
        assert receipt["ordered_selected_ids"] == ["task/identity", "task/value"]

    def test_membership_vs_order_distinction(self):
        """Same members, different order → same membership hash, different order hash."""
        r1 = packet_receipt("t", ["a", "b"], ["a", "b"])
        r2 = packet_receipt("t", ["a", "b"], ["b", "a"])
        assert r1["membership_hash"] == r2["membership_hash"]
        assert r1["order_hash"] != r2["order_hash"]
