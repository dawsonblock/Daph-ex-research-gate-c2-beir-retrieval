"""PAV dataset collection and storage.

Collects transition-level data for training learned PAV models.
Each record includes state-before/after features, action, progress components,
Q value, costs, and terminal outcome.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PAVTransitionRecord:
    """A single transition record for PAV training.

    All fields are controller-visible — no hidden task labels.
    """
    transition_id: str
    task_id: str
    step: int
    action: str
    state_sha_before: str
    state_sha_after: str
    features_before: dict
    features_after: dict
    progress_components: dict
    q_value: float
    action_cost: float
    resource_cost: float
    terminal_utility: float | None
    terminal_success: bool | None
    provenance: dict

    def as_dict(self) -> dict:
        return {
            "transition_id": self.transition_id,
            "task_id": self.task_id,
            "step": self.step,
            "action": self.action,
            "state_sha_before": self.state_sha_before,
            "state_sha_after": self.state_sha_after,
            "features_before": self.features_before,
            "features_after": self.features_after,
            "progress_components": self.progress_components,
            "q_value": self.q_value,
            "action_cost": self.action_cost,
            "resource_cost": self.resource_cost,
            "terminal_utility": self.terminal_utility,
            "terminal_success": self.terminal_success,
            "provenance": self.provenance,
        }


def save_transitions(
    records: list[PAVTransitionRecord],
    path: str | Path,
    manifest: dict,
) -> None:
    """Save transitions to JSONL with a manifest."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")

    # Compute dataset hash
    content_hash = hashlib.sha256()
    for r in records:
        content_hash.update(json.dumps(r.as_dict(), sort_keys=True).encode())

    manifest_path = p.parent / "manifest.json"
    full_manifest = {
        **manifest,
        "n_records": len(records),
        "dataset_sha256": content_hash.hexdigest(),
        "filename": p.name,
    }
    with open(manifest_path, "w") as f:
        json.dump(full_manifest, f, indent=2, sort_keys=True)


def load_transitions(path: str | Path) -> tuple[list[dict], dict]:
    """Load transitions and manifest."""
    p = Path(path)
    records = []
    if p.exists():
        with open(p) as f:
            for line in f:
                records.append(json.loads(line))

    manifest_path = p.parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    return records, manifest
