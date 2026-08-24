"""DAPH I3.4 — Transition dataset loading and task-level splitting.

Splits by task_id, not randomly, to prevent leakage of task identity
across steps. No trajectory from one task can appear in multiple splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_transitions(path: str | Path) -> list[dict[str, Any]]:
    """Load transition records from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_by_task(
    transitions: list[dict],
    *,
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split transitions by task_id into train/dev/test.

    No trajectory from one task can appear in multiple splits.
    """
    # Group by task_id
    task_groups: dict[str, list[dict]] = defaultdict(list)
    for t in transitions:
        task_groups[t["task_id"]].append(t)

    task_ids = sorted(task_groups.keys())
    n_tasks = len(task_ids)

    # Deterministic shuffle
    import random
    rng = random.Random(seed)
    rng.shuffle(task_ids)

    n_train = int(n_tasks * train_frac)
    n_dev = int(n_tasks * dev_frac)

    train_ids = set(task_ids[:n_train])
    dev_ids = set(task_ids[n_train:n_train + n_dev])
    test_ids = set(task_ids[n_train + n_dev:])

    train = []
    dev = []
    test = []
    for tid in task_ids:
        if tid in train_ids:
            train.extend(task_groups[tid])
        elif tid in dev_ids:
            dev.extend(task_groups[tid])
        else:
            test.extend(task_groups[tid])

    return train, dev, test


def get_action_value_target(t: dict) -> float:
    """Extract the action-value target from a transition.

    Uses utility_to_go as the primary target. This captures:
    - step costs
    - success/failure
    - resource consumption
    - terminal penalties/rewards
    """
    return float(t.get("utility_to_go", 0.0))


def get_epistemic_target(t: dict) -> float:
    """Extract the epistemic progress target."""
    return float(t.get("delta_epistemic_utility", 0.0))


def get_success_target(t: dict) -> int:
    """Extract the binary success target."""
    return 1 if t.get("success", False) else 0


def get_feature_vector(t: dict) -> list[float]:
    """Extract the numeric feature vector from a transition's before-state."""
    features = t.get("features_before", {})
    return [
        float(features.get("n_live", 0)),
        float(features.get("n_eliminated", 0)),
        float(features.get("n_total", 0)),
        float(features.get("n_visible", 0)),
        float(features.get("n_hidden", 0)),
        float(features.get("n_verified", 0)),
        float(features.get("n_supporting", 0)),
        float(features.get("n_contradicting", 0)),
        float(features.get("retrieval_remaining", 0)),
        float(features.get("search_remaining", 0)),
        float(features.get("verify_remaining", 0)),
        float(features.get("steps_remaining", 0)),
        1.0 if features.get("can_retrieve", False) else 0.0,
        1.0 if features.get("can_search", False) else 0.0,
        1.0 if features.get("can_verify", False) else 0.0,
        1.0 if features.get("t2", False) else 0.0,
        float(features.get("step", 0)),
    ]


FEATURE_NAMES = (
    "n_live", "n_eliminated", "n_total",
    "n_visible", "n_hidden", "n_verified",
    "n_supporting", "n_contradicting",
    "retrieval_remaining", "search_remaining", "verify_remaining",
    "steps_remaining",
    "can_retrieve", "can_search", "can_verify",
    "t2", "step",
)


def get_phase(t: dict) -> str:
    """Extract the phase from a transition."""
    return t.get("phase_before", "UNKNOWN")


def get_action(t: dict) -> str:
    """Extract the action from a transition."""
    return t.get("action", "UNKNOWN")


def get_legal_actions(t: dict) -> list[str]:
    """Extract the legal actions from a transition."""
    return t.get("legal_actions", [])


def get_dataset_hash(transitions: list[dict]) -> str:
    """Compute a hash of the transition dataset for provenance."""
    content = json.dumps(transitions, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
