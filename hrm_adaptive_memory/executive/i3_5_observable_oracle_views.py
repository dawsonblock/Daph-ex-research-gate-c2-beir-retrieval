"""V2 observable-oracle views for I3.5 scoring.

Adapts the I3.4 observable oracle views for V2 structural tasks.
Reads from V2 oracle tables and V2 split definitions.

Schema identity: ``DAPH_V2B_I3_5_OBSERVABLE_ORACLE_VIEW_V2`` (frozen).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIEW_SCHEMA = "DAPH_V2B_I3_5_OBSERVABLE_ORACLE_VIEW_V2"
VIEW_VERSION = 2

CONDITIONS = (
    "STATE_AWARE_CONTROLLER",
    "STATE_BLIND_CONTROLLER",
)

ORACLE_DIR = "experiments/v2b_i3_5/oracle_tables"
ORACLE_FILES = {
    "STATE_AWARE_CONTROLLER": f"{ORACLE_DIR}/v2b_i3_5_sequential_state_aware_controller_v1.jsonl.gz",
    "STATE_BLIND_CONTROLLER": f"{ORACLE_DIR}/v2b_i3_5_sequential_state_blind_controller_v1.jsonl.gz",
}

SPLITS_PATH = "experiments/v2b_i3_5/splits/v2b_i3_5_splits_v2.json"
LATENT_ORACLE_PATH = f"{ORACLE_DIR}/v2b_i3_5_latent_oracles_v1.jsonl.gz"


@dataclass(frozen=True)
class V2TaskObservableEntry:
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


def load_v2_task_to_observable(
    root: str | Path, condition: str
) -> dict[str, V2TaskObservableEntry]:
    """Load per-task V_O^*(B_i) for one condition from V2 oracle tables."""
    root = Path(root)
    path = root / ORACLE_FILES[condition]
    task_map: dict[str, V2TaskObservableEntry] = {}

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
                task_map[task_id] = V2TaskObservableEntry(
                    task_id=task_id,
                    condition=condition,
                    information_class_id=init_id,
                    observable_optimal_value=vo,
                    table_identity_sha256=table_identity,
                )

    return task_map


def _load_v2_split_task_ids(root: str | Path) -> dict[str, list[str]]:
    """Load task_ids for each V2 split."""
    path = Path(root) / SPLITS_PATH
    data = json.loads(path.read_text())
    splits: dict[str, list[str]] = {}
    for split_name, split_data in data["splits"].items():
        task_ids = [entry["task_id"] for entry in split_data]
        splits[split_name] = task_ids
    return splits


def build_v2_observable_oracle_views(
    root: str | Path = ".",
    *,
    splits: tuple[str, ...] = ("structure_dev_v2", "structure_validation_v2",
                               "structure_held_out_v2"),
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[dict[str, Any]]:
    """Build V2 observable-oracle views with per-task V_O for each split/condition."""
    root = Path(root)
    split_task_ids = _load_v2_split_task_ids(root)

    # Load oracle cache manifest for hashes
    cache_manifest = json.loads(
        (root / f"{ORACLE_DIR}/v2b_i3_5_oracle_cache_manifest_v1.json").read_text())
    oracle_set_hashes = {
        cond: cache_manifest["sequential_observable_oracles"][cond]["set_sha256"]
        for cond in CONDITIONS
    }
    latent_table_sha = cache_manifest["latent_oracles"]["table_set_sha256"]

    views = []
    for condition in conditions:
        task_map = load_v2_task_to_observable(root, condition)

        for split_name in splits:
            if split_name not in split_task_ids:
                continue
            split_tasks = split_task_ids[split_name]

            task_entries = []
            for task_id in sorted(split_tasks):
                if task_id in task_map:
                    task_entries.append(task_map[task_id])

            if not task_entries:
                continue

            mean_observable = sum(
                e.observable_optimal_value for e in task_entries) / len(task_entries)

            view_sha = hashlib.sha256(json.dumps({
                "schema": VIEW_SCHEMA,
                "split_name": split_name,
                "condition": condition,
                "task_entries": [e.as_dict() for e in task_entries],
                "observable_optimal_value": mean_observable,
                "observable_oracle_set_sha256": oracle_set_hashes[condition],
                "latent_oracle_table_sha256": latent_table_sha,
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

            views.append({
                "schema": VIEW_SCHEMA,
                "schema_version": VIEW_VERSION,
                "split_name": split_name,
                "condition": condition,
                "task_ids": sorted(split_tasks),
                "task_count": len(task_entries),
                "task_entries": [e.as_dict() for e in task_entries],
                "observable_optimal_value": mean_observable,
                "observable_oracle_set_sha256": oracle_set_hashes[condition],
                "latent_oracle_table_sha256": latent_table_sha,
                "view_sha256": view_sha,
            })

    return views


def save_v2_views(views: list[dict[str, Any]], path: str | Path) -> str:
    """Save V2 views to a JSON file and return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": VIEW_SCHEMA,
        "schema_version": VIEW_VERSION,
        "views": views,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
