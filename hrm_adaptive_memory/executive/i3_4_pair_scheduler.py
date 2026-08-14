"""Counterbalanced paired call scheduler for I3.4.1.

For each task, derive a deterministic order from:
    h_i = SHA256(experiment_id || task_id)

If h_i is even: BLIND → AWARE
If h_i is odd:  AWARE → BLIND

The two calls for each task are kept adjacent to protect against provider
drift.  Every pair gets a pair_id, pair_order, and fingerprint tracking.

Schema identity: ``DAPH_V2B_I3_4_PAIR_SCHEDULER_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

SCHEDULER_SCHEMA = "DAPH_V2B_I3_4_PAIR_SCHEDULER_V1"
SCHEDULER_VERSION = 1

BLIND_FIRST = "BLIND_FIRST"
AWARE_FIRST = "AWARE_FIRST"


@dataclass(frozen=True)
class PairSchedule:
    """Schedule for one blind/aware task pair."""

    pair_id: str
    task_id: str
    pair_order: str           # BLIND_FIRST or AWARE_FIRST
    first_condition: str      # STATE_BLIND_CONTROLLER or STATE_AWARE_CONTROLLER
    second_condition: str     # the other one
    schedule_hash: str        # SHA-256 of (experiment_id, task_id)

    def as_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "task_id": self.task_id,
            "pair_order": self.pair_order,
            "first_condition": self.first_condition,
            "second_condition": self.second_condition,
            "schedule_hash": self.schedule_hash,
        }


@dataclass(frozen=True)
class PairFingerprintRecord:
    """Fingerprint tracking for a completed pair.

    If the model fingerprint changes within a pair, the preregistered
    invalidation rule applies: the pair is invalid and not scored.
    """

    pair_id: str
    first_call_fingerprint: str | None
    second_call_fingerprint: str | None
    first_call_timestamp: str | None
    second_call_timestamp: str | None
    fingerprint_match: bool
    pair_valid: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "first_call_fingerprint": self.first_call_fingerprint,
            "second_call_fingerprint": self.second_call_fingerprint,
            "first_call_timestamp": self.first_call_timestamp,
            "second_call_timestamp": self.second_call_timestamp,
            "fingerprint_match": self.fingerprint_match,
            "pair_valid": self.pair_valid,
        }


def compute_pair_hash(experiment_id: str, task_id: str) -> str:
    """Compute the deterministic pair hash for a task."""
    return hashlib.sha256(f"{experiment_id}:{task_id}".encode()).hexdigest()


def is_blind_first(experiment_id: str, task_id: str) -> bool:
    """Determine if blind should go first for this task.

    Derived from h_i = SHA256(experiment_id || task_id).
    Even hash → BLIND → AWARE.
    Odd hash → AWARE → BLIND.
    """
    h = compute_pair_hash(experiment_id, task_id)
    # Use the last hex digit to determine even/odd.
    return int(h[-1], 16) % 2 == 0


def build_pair_schedule(
    *,
    experiment_id: str,
    task_ids: list[str],
    blind_condition: str = "STATE_BLIND_CONTROLLER",
    aware_condition: str = "STATE_AWARE_CONTROLLER",
) -> list[PairSchedule]:
    """Build the counterbalanced pair schedule for all tasks.

    The schedule is deterministic given the experiment_id and task list.
    """
    schedules: list[PairSchedule] = []
    for task_id in sorted(task_ids):
        h = compute_pair_hash(experiment_id, task_id)
        blind_first = int(h[-1], 16) % 2 == 0
        if blind_first:
            order = BLIND_FIRST
            first, second = blind_condition, aware_condition
        else:
            order = AWARE_FIRST
            first, second = aware_condition, blind_condition
        pair_id = f"pair_{h[:16]}"
        schedules.append(PairSchedule(
            pair_id=pair_id, task_id=task_id,
            pair_order=order, first_condition=first,
            second_condition=second, schedule_hash=h,
        ))
    return schedules


def check_pair_fingerprints(
    *,
    pair_id: str,
    first_call_fingerprint: str | None,
    second_call_fingerprint: str | None,
    first_call_timestamp: str | None = None,
    second_call_timestamp: str | None = None,
) -> PairFingerprintRecord:
    """Check if fingerprints match within a pair.

    Preregistered invalidation rule:
    - If either fingerprint is None, the pair is valid (no evidence of drift).
    - If both fingerprints are present and differ, the pair is INVALID.
    - If both fingerprints are present and match, the pair is valid.

    If fingerprint changes occur repeatedly, the phase should be stopped.
    """
    if first_call_fingerprint is None or second_call_fingerprint is None:
        match = True  # No evidence of drift; cannot invalidate
        valid = True
    else:
        match = first_call_fingerprint == second_call_fingerprint
        valid = match  # Within-pair provider identity change → pair invalid
    return PairFingerprintRecord(
        pair_id=pair_id,
        first_call_fingerprint=first_call_fingerprint,
        second_call_fingerprint=second_call_fingerprint,
        first_call_timestamp=first_call_timestamp,
        second_call_timestamp=second_call_timestamp,
        fingerprint_match=match,
        pair_valid=valid,
    )


def scheduler_module_sha256() -> str:
    """Canonical SHA-256 of this module's source code."""
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
