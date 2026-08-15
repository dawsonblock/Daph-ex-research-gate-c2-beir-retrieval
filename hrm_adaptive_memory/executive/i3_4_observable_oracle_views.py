"""Evaluation-specific observable-oracle views for I3.4.1.

The observable optimum V_O^M(B_M(s)) depends on the prior over latent states.
The evaluation distribution differs between development, validation,
held_out_instance, held_out_surface, and held_out_structure.

This module computes per-split observable oracle values by:
1. Loading the frozen I3.3.2 sequential oracle tables.
2. Mapping each oracle entry to its task_id via the controller packets.
3. Filtering to the tasks in each evaluation split.
4. Computing the task-uniform mean observable value for each split/condition.
5. Producing a canonical, SHA-256-identified view artifact.

Schema identity: ``DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V1`` (frozen).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

VIEW_SCHEMA = "DAPH_V2B_I3_4_OBSERVABLE_ORACLE_VIEW_V1"
VIEW_VERSION = 1

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
class ObservableOracleView:
    """One evaluation-specific observable-oracle view for one condition.

    The observable_optimal_value is the task-uniform mean of V_O^M(B_M(s))
    over the tasks in this split under this condition.
    """

    split_name: str
    condition: str
    task_ids: tuple[str, ...]
    task_count: int
    observable_optimal_value: float  # E[V_O^M] over this split
    information_class_hash: str
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
            "observable_optimal_value": self.observable_optimal_value,
            "information_class_hash": self.information_class_hash,
            "observable_oracle_set_sha256": self.observable_oracle_set_sha256,
            "latent_oracle_table_sha256": self.latent_oracle_table_sha256,
            "view_sha256": self.view_sha256,
        }


def _load_task_order(root: str | Path) -> list[str]:
    """Load the 750 task_ids in canonical order from controller packets."""
    path = Path(root) / CONTROLLER_PACKETS_PATH
    data = json.loads(path.read_text())
    return [p["task_id"] for p in data["packets"]]


def _load_split_task_ids(root: str | Path) -> dict[str, list[str]]:
    """Load task_ids for each split from the frozen split definitions."""
    path = Path(root) / SPLITS_PATH
    data = json.loads(path.read_text())
    splits: dict[str, list[str]] = {}
    for split_name, split_data in data["splits"].items():
        task_ids = [entry["task_id"] for entry in split_data]
        splits[split_name] = task_ids
    return splits


def _load_observable_values(root: str | Path, condition: str) -> list[float | None]:
    """Load the initial-state belief values (V_O^M) for all 750 tasks.

    Returns a list of 750 values (or None for tasks without a reachable
    information state under this condition).
    """
    path = Path(root) / ORACLE_FILES[condition]
    values: list[float | None] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            table = entry["table"]
            init_id = entry["initial_information_state_id"]
            belief_value = table["belief_values"].get(init_id)
            values.append(belief_value)
    return values


def _load_latent_optimal_values(root: str | Path) -> list[float | None]:
    """Load the latent optimal values (V_L^*) for all 750 tasks."""
    path = Path(root) / LATENT_ORACLE_PATH
    values: list[float | None] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            # The latent oracle has state_values; the initial state value
            # is V_L^*(s).
            table = entry.get("table", entry)
            state_values = table.get("state_values", {})
            init_id = entry.get("initial_state_id") or table.get("initial_state_id")
            if init_id and init_id in state_values:
                values.append(state_values[init_id])
            else:
                # Fallback: use the first state value
                if state_values:
                    values.append(next(iter(state_values.values())))
                else:
                    values.append(None)
    return values


def _compute_view_sha256(
    *,
    split_name: str,
    condition: str,
    task_ids: tuple[str, ...],
    observable_optimal_value: float,
    information_class_hash: str,
    observable_oracle_set_sha256: str,
    latent_oracle_table_sha256: str,
) -> str:
    """Compute the canonical SHA-256 of the view."""
    payload = json.dumps({
        "schema": VIEW_SCHEMA,
        "split_name": split_name,
        "condition": condition,
        "task_ids": list(task_ids),
        "observable_optimal_value": observable_optimal_value,
        "information_class_hash": information_class_hash,
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
    """Build evaluation-specific observable-oracle views.

    For each split and condition, computes the task-uniform mean of
    V_O^M(B_M(s)) over the tasks in that split.

    This is the correct way to feed observable_optimal_value into the
    IG/DG/TR decomposition for each evaluation population.
    """
    root = Path(root)
    task_order = _load_task_order(root)
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
        # Load observable values for all 750 tasks (positional).
        all_observable = _load_observable_values(root, condition)
        # Map task_id -> observable value.
        task_to_observable = {}
        for i, task_id in enumerate(task_order):
            if i < len(all_observable) and all_observable[i] is not None:
                task_to_observable[task_id] = all_observable[i]

        for split_name in splits:
            if split_name not in split_task_ids:
                continue
            split_tasks = split_task_ids[split_name]
            # Filter to tasks that have observable values.
            valid_observables = [
                task_to_observable[t] for t in split_tasks
                if t in task_to_observable
            ]
            if not valid_observables:
                continue
            # Task-uniform mean observable value.
            mean_observable = sum(valid_observables) / len(valid_observables)
            # Information class hash: hash of the sorted task_ids in this split.
            info_class_hash = hashlib.sha256(
                json.dumps(sorted(split_tasks), separators=(",", ":")).encode()
            ).hexdigest()
            view_sha = _compute_view_sha256(
                split_name=split_name, condition=condition,
                task_ids=tuple(sorted(split_tasks)),
                observable_optimal_value=mean_observable,
                information_class_hash=info_class_hash,
                observable_oracle_set_sha256=oracle_set_hashes[condition],
                latent_oracle_table_sha256=latent_table_sha,
            )
            views.append(ObservableOracleView(
                split_name=split_name,
                condition=condition,
                task_ids=tuple(sorted(split_tasks)),
                task_count=len(split_tasks),
                observable_optimal_value=mean_observable,
                information_class_hash=info_class_hash,
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
