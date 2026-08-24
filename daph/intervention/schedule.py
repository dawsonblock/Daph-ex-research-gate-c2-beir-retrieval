"""Frozen intervention schedules.

An intervention schedule specifies which actions to force from each checkpoint.
The schedule is frozen before execution and hashed for provenance.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Intervention:
    """A single forced-action intervention from a checkpoint.

    Attributes:
        checkpoint_id: The checkpoint to intervene from
        action: The action to force
        intervention_type: "CAUSAL_DETERMINISTIC" or "FORCED_ACTION_ROLLOUT"
        target_evidence_id: Optional evidence target for VERIFY actions
    """
    checkpoint_id: str
    action: str
    intervention_type: str  # "CAUSAL_DETERMINISTIC" or "FORCED_ACTION_ROLLOUT"
    target_evidence_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "action": self.action,
            "intervention_type": self.intervention_type,
            "target_evidence_id": self.target_evidence_id,
        }


@dataclass(frozen=True)
class InterventionSchedule:
    """A frozen schedule of interventions.

    Attributes:
        schedule_id: SHA256 hash of the schedule content
        interventions: Tuple of Intervention objects
        created_at: ISO timestamp
        description: Human-readable description
    """
    schedule_id: str
    interventions: tuple[Intervention, ...]
    created_at: str
    description: str

    def as_dict(self) -> dict:
        return {
            "schedule_id": self.schedule_id,
            "interventions": [i.as_dict() for i in self.interventions],
            "created_at": self.created_at,
            "description": self.description,
            "n_interventions": len(self.interventions),
        }


def classify_intervention_type(action: str) -> str:
    """Classify an action as deterministic or forced-rollout.

    CAUSAL_DETERMINISTIC: actions that can be replayed exactly
      - DEFER, ANSWER, STOP (terminal, no downstream model calls)
      - VERIFY (deterministic evidence state change)

    FORCED_ACTION_ROLLOUT: actions where downstream behavior includes model calls
      - RETRIEVE, SEARCH_MORE, REASON_MORE
    """
    if action in ("DEFER", "ANSWER", "STOP", "VERIFY"):
        return "CAUSAL_DETERMINISTIC"
    else:
        return "FORCED_ACTION_ROLLOUT"


def build_intervention_schedule(
    checkpoint_ids: list[str],
    actions_per_checkpoint: Mapping[str, list[str]],
    description: str = "",
    created_at: str = "",
) -> InterventionSchedule:
    """Build a frozen intervention schedule.

    Args:
        checkpoint_ids: List of checkpoint IDs to intervene from
        actions_per_checkpoint: Mapping from checkpoint_id to list of actions
        description: Human-readable description
        created_at: ISO timestamp

    Returns:
        A frozen InterventionSchedule with a deterministic SHA256 ID.
    """
    interventions: list[Intervention] = []
    for cp_id in checkpoint_ids:
        actions = actions_per_checkpoint.get(cp_id, [])
        for action in actions:
            # Parse target evidence ID if present (e.g. "VERIFY:E1")
            target = None
            action_name = action
            if ":" in action:
                action_name, target = action.split(":")

            itype = classify_intervention_type(action_name)
            interventions.append(Intervention(
                checkpoint_id=cp_id,
                action=action_name,
                intervention_type=itype,
                target_evidence_id=target,
            ))

    # Compute schedule ID
    content = json.dumps({
        "interventions": [i.as_dict() for i in interventions],
        "description": description,
    }, sort_keys=True)
    schedule_id = hashlib.sha256(content.encode()).hexdigest()

    return InterventionSchedule(
        schedule_id=schedule_id,
        interventions=tuple(interventions),
        created_at=created_at,
        description=description,
    )


def save_schedule(schedule: InterventionSchedule, path: Path) -> None:
    """Save a schedule to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schedule.as_dict(), f, indent=2, sort_keys=True)


def load_schedule(path: Path) -> InterventionSchedule:
    """Load a schedule from JSON."""
    with open(path) as f:
        data = json.load(f)
    interventions = tuple(
        Intervention(
            checkpoint_id=i["checkpoint_id"],
            action=i["action"],
            intervention_type=i["intervention_type"],
            target_evidence_id=i.get("target_evidence_id"),
        )
        for i in data["interventions"]
    )
    return InterventionSchedule(
        schedule_id=data["schedule_id"],
        interventions=interventions,
        created_at=data["created_at"],
        description=data["description"],
    )
