"""Evaluation-specific observable-oracle views for I3.4.1.

The observable optimum V_O^M(B_M(s)) depends on the prior over latent states.
The evaluation distribution differs between development, validation,
held_out_instance, held_out_surface, and held_out_structure.

This module computes per-task observable oracle values by:
1. Loading the frozen I3.3.2 sequential oracle tables.
2. Reading the ``members`` field of each table's initial information state
   to determine which tasks belong to which information class.
3. Looking up ``belief_values[initial_state_id]`` for each class to get
   V_O^*(B) for that class.
4. Mapping each task to its information class's V_O^*(B).
5. Producing per-task, per-class, and per-split view artifacts.

Schema identity: ``DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V2`` (frozen).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

VIEW_SCHEMA = "DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V2"
VIEW_VERSION = 2

# Condition names matching the I3.3.2 sequential oracle sets.
CONDITIONS = (
    "STATE_AWARE_CONTROLLER",
    "STATE_BLIND_CONTROLLER",
    "NO_TEMPORAL",
    "NO_PROVENANCE",
    "NO_VERIFICATION",
    "NO_HISTORY",
    "NO_CONFLICT",
)

# Oracle file paths (relative to repo root).
ORACLE_DIR = "experiments/v2b_i3_3/oracle_tables"
ORACLE_FILES = {
    "STATE_AWARE_CONTROLLER": f"{ORACLE_DIR}/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz",
    "STATE_BLIND_CONTROLLER": f"{ORACLE_DIR}/v2b_i3_3_sequential_state_blind_controller_v1.jsonl.gz",
    "NO_TEMPORAL": f"{ORACLE_DIR}/v2b_i3_3_sequential_no_temporal_v1.jsonl.gz",
    "NO_PROVENANCE": f"{ORACLE_DIR}/v2b_i3_3_sequential_no_provenance_v1.jsonl.gz",
    "NO_VERIFICATION": f"{ORACLE_DIR}/v2b_i3_3_sequential_no_verification_v1.jsonl.gz",
    "NO_HISTORY": f"{ORACLE_DIR}/v2b_i3_3_sequential_no_history_v1.jsonl.gz",
    "NO_CONFLICT": f"{ORACLE_DIR}/v2b_i3_3_sequential_no_conflict_v1.jsonl.gz",
}

CONTROLLER_PACKETS_PATH = "experiments/v2b_i3_3/controller_packets/v2b_i3_3_controller_packets_v1.json"
SPLITS_PATH = "experiments/v2b_i3_3/splits/v2b_i3_3_splits_v1.json"
LATENT_ORACLE_PATH = f"{ORACLE_DIR}/v2b_i3_3_latent_oracles_v1.jsonl.gz"


@dataclass(frozen=True)
class InformationClass:
    """One information class from the sequential observable oracle.

    All tasks in the same class share the same initial public observation
    packet under the given observation mask, and therefore share the same
    V_O^*(B).
    """

    class_id: str  # initial_information_state_id
    observable_optimal_value: float  # belief_values[initial_id]
    member_task_ids: tuple[str, ...]
    posterior_weights: tuple[str, ...]
    table_identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "observable_optimal_value": self.observable_optimal_value,
            "member_task_ids": list(self.member_task_ids),
            "posterior_weights": list(self.posterior_weights),
            "table_identity_sha256": self.table_identity_sha256,
        }


@dataclass(frozen=True)
class TaskObservableEntry:
    """Per-task observable oracle value for one condition."""

    task_id: str
    condition: str
    information_class_id: str
    observable_optimal_value: float
    table_identity_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "information_class_id": self.information_class_id,
            "observable_optimal_value": self.observable_optimal_value,
            "table_identity_sha256": self.table_identity_sha256,
        }


@dataclass(frozen=True)
class ObservableOracleView:
    """Evaluation-specific observable-oracle view for one split/condition.

    Contains per-task V_O^*(B_i) values, per-class structure, and split-level
    summary statistics.
    """

    split_name: str
    condition: str
    task_ids: tuple[str, ...]
    task_count: int
    # Per-task entries
    task_entries: tuple[TaskObservableEntry, ...]
    # Per-class structure
    information_classes: tuple[InformationClass, ...]
    # Split-level summary (task-uniform mean of per-task V_O)
    observable_optimal_value: float
    # Provenance
    observable_oracle_set_sha256: str
    latent_oracle_table_sha256: str
    view_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": VIEW_SCHEMA,
            "schema_version": VIEW_VERSION,
            "split_name": self.split_name,
            "condition": self.condition,
            "task_ids": list(self.task_ids),
            "task_count": self.task_count,
            "task_entries": [e.as_dict() for e in self.task_entries],
            "information_classes": [c.as_dict() for c in self.information_classes],
            "observable_optimal_value": self.observable_optimal_value,
            "observable_oracle_set_sha256": self.observable_oracle_set_sha256,
            "latent_oracle_table_sha256": self.latent_oracle_table_sha256,
            "view_sha256": self.view_sha256,
        }


def _load_split_task_ids(root: str | Path) -> dict[str, list[str]]:
    """Load task_ids for each split from the frozen split definitions."""
    path = Path(root) / SPLITS_PATH
    data = json.loads(path.read_text())
    splits: dict[str, list[str]] = {}
    for split_name, split_data in data["splits"].items():
        task_ids = [entry["task_id"] for entry in split_data]
        splits[split_name] = task_ids
    return splits


def load_task_to_observable(
    root: str | Path, condition: str
) -> dict[str, TaskObservableEntry]:
    """Load per-task V_O^*(B_i) for one condition by reading information-class
    membership from the sequential oracle tables.

    This is the correct mapping: each table line in the JSONL.gz file is one
    information class. The ``members`` field of the initial information state
    lists which tasks belong to that class. The ``belief_values`` at the
    initial state ID gives V_O^*(B) for that class.

    Returns a dict mapping task_id -> TaskObservableEntry.
    """
    root = Path(root)
    path = root / ORACLE_FILES[condition]
    task_map: dict[str, TaskObservableEntry] = {}

    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            table = entry["table"]
            init_id = entry["initial_information_state_id"]
            vo = table["belief_values"].get(init_id)
            if vo is None:
                continue
            table_identity = table.get("identity_sha256", "")
            init_state = table["information_states"].get(init_id, {})
            members = init_state.get("members", [])
            for member in members:
                task_id = member["task_id"]
                task_map[task_id] = TaskObservableEntry(
                    task_id=task_id,
                    condition=condition,
                    information_class_id=init_id,
                    observable_optimal_value=vo,
                    table_identity_sha256=table_identity,
                )

    return task_map


def load_information_classes(
    root: str | Path, condition: str
) -> list[InformationClass]:
    """Load all information classes for one condition."""
    root = Path(root)
    path = root / ORACLE_FILES[condition]
    classes: list[InformationClass] = []

    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            table = entry["table"]
            init_id = entry["initial_information_state_id"]
            vo = table["belief_values"].get(init_id)
            if vo is None:
                continue
            table_identity = table.get("identity_sha256", "")
            init_state = table["information_states"].get(init_id, {})
            members = init_state.get("members", [])
            member_task_ids = tuple(
                sorted(m["task_id"] for m in members)
            )
            posterior_weights = tuple(
                m["posterior_weight"] for m in sorted(members, key=lambda x: x["task_id"])
            )
            classes.append(InformationClass(
                class_id=init_id,
                observable_optimal_value=vo,
                member_task_ids=member_task_ids,
                posterior_weights=posterior_weights,
                table_identity_sha256=table_identity,
            ))

    return classes


def _compute_view_sha256(
    *,
    split_name: str,
    condition: str,
    task_entries: tuple[TaskObservableEntry, ...],
    information_classes: tuple[InformationClass, ...],
    observable_optimal_value: float,
    observable_oracle_set_sha256: str,
    latent_oracle_table_sha256: str,
) -> str:
    """Compute the canonical SHA-256 of the view."""
    payload = json.dumps({
        "schema": VIEW_SCHEMA,
        "split_name": split_name,
        "condition": condition,
        "task_entries": [e.as_dict() for e in task_entries],
        "information_classes": [c.as_dict() for c in information_classes],
        "observable_optimal_value": observable_optimal_value,
        "observable_oracle_set_sha256": observable_oracle_set_sha256,
        "latent_oracle_table_sha256": latent_oracle_table_sha256,
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_observable_oracle_views(
    root: str | Path = ".",
    *,
    splits: tuple[str, ...] = ("development", "validation",
                               "held_out_instance", "held_out_surface",
                               "held_out_structure"),
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[ObservableOracleView]:
    """Build evaluation-specific observable-oracle views with per-task V_O.

    For each split and condition:
    1. Load the sequential oracle tables for that condition.
    2. Read information-class membership to map each task to its V_O^*(B_i).
    3. Filter to tasks in the given split.
    4. Produce per-task entries, per-class structure, and split-level summary.
    """
    root = Path(root)
    split_task_ids = _load_split_task_ids(root)

    # Load the I3.3.2 baseline for oracle set hashes.
    baseline = json.loads((root / "configs/v2b_i3_3_3_baseline.json").read_text())
    oracle_set_hashes = {
        cond: baseline["sequential_observable_oracle_sets"][cond]["set_sha256"]
        for cond in CONDITIONS
    }
    latent_table_sha = baseline["latent_oracle_set"]["table_set_sha256"]

    views: list[ObservableOracleView] = []
    for condition in conditions:
        # Load per-task V_O via information-class membership.
        task_map = load_task_to_observable(root, condition)
        all_classes = load_information_classes(root, condition)

        for split_name in splits:
            if split_name not in split_task_ids:
                continue
            split_tasks = split_task_ids[split_name]

            # Per-task entries for this split
            task_entries: list[TaskObservableEntry] = []
            for task_id in sorted(split_tasks):
                if task_id in task_map:
                    task_entries.append(task_map[task_id])

            if not task_entries:
                continue

            # Per-class structure: only classes that have members in this split
            split_task_set = set(split_tasks)
            split_classes = [
                cls for cls in all_classes
                if any(tid in split_task_set for tid in cls.member_task_ids)
            ]

            # Split-level summary: task-uniform mean of per-task V_O
            mean_observable = sum(e.observable_optimal_value for e in task_entries) / len(task_entries)

            view_sha = _compute_view_sha256(
                split_name=split_name,
                condition=condition,
                task_entries=tuple(task_entries),
                information_classes=tuple(split_classes),
                observable_optimal_value=mean_observable,
                observable_oracle_set_sha256=oracle_set_hashes[condition],
                latent_oracle_table_sha256=latent_table_sha,
            )
            views.append(ObservableOracleView(
                split_name=split_name,
                condition=condition,
                task_ids=tuple(sorted(split_tasks)),
                task_count=len(task_entries),
                task_entries=tuple(task_entries),
                information_classes=tuple(split_classes),
                observable_optimal_value=mean_observable,
                observable_oracle_set_sha256=oracle_set_hashes[condition],
                latent_oracle_table_sha256=latent_table_sha,
                view_sha256=view_sha,
            ))
    return views


def save_views(views: list[ObservableOracleView], path: str | Path) -> str:
    """Save views to a JSON file and return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": VIEW_SCHEMA,
        "schema_version": VIEW_VERSION,
        "views": [v.as_dict() for v in views],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def views_module_sha256() -> str:
    """Canonical SHA-256 of this module's source code."""
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
