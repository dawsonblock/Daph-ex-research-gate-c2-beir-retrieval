"""Factorial block scheduler for I3.5.1.

For each task, counterbalances the four conditions using a frozen
permutation seed. There are 24 possible orderings of four arms.

permutation_index = HMAC_SHA256(schedule_seed, task_id) % 24

The schedule_seed, permutation_index, and condition_order are stored
in evaluator receipts, never in model packets.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from .conditions import ConditionID, ExperimentalCondition, FROZEN_CONDITIONS

SCHEDULER_SCHEMA = "DAPH_V2B_I3_5_1_FACTORIAL_SCHEDULER_V1"
SCHEDULER_VERSION = 1
DEFAULT_SCHEDULE_SEED = "v2b_i3_5_1_factorial_schedule_v1"

# Precompute all 24 permutations of the four condition IDs
_ALL_PERMUTATIONS: tuple[tuple[ConditionID, ...], ...] = tuple(
    permutations(c.condition_id for c in FROZEN_CONDITIONS)
)
assert len(_ALL_PERMUTATIONS) == 24


@dataclass(frozen=True)
class BlockSchedule:
    """Schedule for one task block (four conditions)."""
    task_id: str
    schedule_seed: str
    permutation_index: int
    condition_order: tuple[ConditionID, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "schedule_seed": self.schedule_seed,
            "permutation_index": self.permutation_index,
            "condition_order": [c.value for c in self.condition_order],
        }


def compute_permutation_index(
    schedule_seed: str,
    task_id: str,
) -> int:
    """Deterministic permutation index in [0, 24)."""
    key = schedule_seed.encode("utf-8")
    msg = task_id.encode("utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    # Use first 8 bytes as unsigned integer
    val = int.from_bytes(digest[:8], "big")
    return val % 24


def schedule_block(
    task_id: str,
    schedule_seed: str = DEFAULT_SCHEDULE_SEED,
) -> BlockSchedule:
    """Compute the frozen condition order for one task block."""
    idx = compute_permutation_index(schedule_seed, task_id)
    order = _ALL_PERMUTATIONS[idx]
    return BlockSchedule(
        task_id=task_id,
        schedule_seed=schedule_seed,
        permutation_index=idx,
        condition_order=order,
    )


def schedule_many_blocks(
    task_ids: list[str],
    schedule_seed: str = DEFAULT_SCHEDULE_SEED,
) -> list[BlockSchedule]:
    """Compute schedules for many task blocks."""
    return [schedule_block(tid, schedule_seed) for tid in task_ids]


def check_balance(
    schedules: list[BlockSchedule],
) -> dict[str, Any]:
    """Check that condition orderings are roughly balanced across tasks."""
    from collections import Counter
    perm_counts = Counter(s.permutation_index for s in schedules)
    # Check each condition appears in each position roughly equally
    position_counts: dict[int, Counter] = {i: Counter() for i in range(4)}
    for s in schedules:
        for pos, cid in enumerate(s.condition_order):
            position_counts[pos][cid] += 1
    return {
        "schema": SCHEDULER_SCHEMA,
        "schema_version": SCHEDULER_VERSION,
        "total_blocks": len(schedules),
        "permutation_distribution": dict(perm_counts),
        "position_distribution": {
            str(pos): dict(counts)
            for pos, counts in position_counts.items()
        },
    }


def block_valid(
    schedules: list[BlockSchedule],
    completed_condition_ids: set[tuple[str, ConditionID]],
    task_id: str,
) -> bool:
    """Check whether a task block is valid (all 4 conditions completed)."""
    expected = {ConditionID.BLIND_NO_GOVERNOR, ConditionID.BLIND_GOVERNOR,
                ConditionID.AWARE_NO_GOVERNOR, ConditionID.AWARE_GOVERNOR}
    completed = {cid for (tid, cid) in completed_condition_ids if tid == task_id}
    return expected.issubset(completed)
