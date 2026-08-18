"""Tests for I3.5.1 factorial block scheduler."""
import pytest
from hrm_adaptive_memory.executive.i3_5_1.factorial_scheduler import (
    schedule_block, schedule_many_blocks, check_balance,
    compute_permutation_index, BlockSchedule,
)
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, all_condition_ids


class TestFactorialScheduler:
    def test_schedule_returns_four_conditions(self):
        s = schedule_block("test_task_001")
        assert len(s.condition_order) == 4

    def test_schedule_is_deterministic(self):
        s1 = schedule_block("test_task_001")
        s2 = schedule_block("test_task_001")
        assert s1.condition_order == s2.condition_order
        assert s1.permutation_index == s2.permutation_index

    def test_different_tasks_may_differ(self):
        """Not all tasks should have the same order."""
        orders = set()
        for i in range(100):
            s = schedule_block(f"task_{i:04d}")
            orders.add(s.permutation_index)
        assert len(orders) > 1, "All tasks got the same permutation"

    def test_permutation_index_in_range(self):
        for i in range(100):
            idx = compute_permutation_index("seed", f"task_{i}")
            assert 0 <= idx < 24

    def test_all_four_condition_ids_present(self):
        s = schedule_block("test_task_001")
        ids = set(s.condition_order)
        assert len(ids) == 4
        for cid in all_condition_ids():
            assert cid in ids

    def test_schedule_metadata(self):
        s = schedule_block("test_task_001")
        d = s.as_dict()
        assert d["task_id"] == "test_task_001"
        assert "schedule_seed" in d
        assert "permutation_index" in d
        assert len(d["condition_order"]) == 4

    def test_balance_check(self):
        schedules = schedule_many_blocks([f"task_{i:04d}" for i in range(300)])
        balance = check_balance(schedules)
        assert balance["total_blocks"] == 300
        # Each position should have all 4 conditions represented
        for pos_str, counts in balance["position_distribution"].items():
            assert sum(counts.values()) == 300

    def test_many_blocks_deterministic(self):
        ids = [f"task_{i:04d}" for i in range(10)]
        s1 = schedule_many_blocks(ids)
        s2 = schedule_many_blocks(ids)
        for a, b in zip(s1, s2):
            assert a.condition_order == b.condition_order
