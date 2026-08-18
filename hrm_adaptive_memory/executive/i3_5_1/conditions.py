"""Experimental condition definitions for the I3.5.1 factorial design.

Two independent factors:
  S in {BLIND, AWARE}  — cognitive-state visibility
  G in {OFF, ON}       — governor availability

Four conditions with stable machine IDs. The model-visible packet
must never contain condition_id, BLIND/AWARE labels, GOVERNOR_ON/OFF,
or treatment_group metadata. Those are evaluator-only fields.

Schema identity: DAPH_V2B_I3_5_1_CONDITIONS_V1 (frozen).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ObservationMode(Enum):
    """Cognitive-state visibility factor S."""
    BLIND = "BLIND"
    AWARE = "AWARE"


class GovernorMode(Enum):
    """Governor availability factor G."""
    OFF = "OFF"
    ON = "ON"


class ConditionID(Enum):
    """Stable machine IDs for the four factorial conditions."""
    BLIND_NO_GOVERNOR = "BLIND_NO_GOVERNOR"   # C00
    BLIND_GOVERNOR = "BLIND_GOVERNOR"          # C01
    AWARE_NO_GOVERNOR = "AWARE_NO_GOVERNOR"   # C10
    AWARE_GOVERNOR = "AWARE_GOVERNOR"          # C11


CONDITIONS_SCHEMA = "DAPH_V2B_I3_5_1_CONDITIONS_V1"
CONDITIONS_VERSION = 1


@dataclass(frozen=True)
class ExperimentalCondition:
    """One cell of the 2x2 factorial design.

    The model-visible packet is built from observation_mode and
    governor_enabled. The condition_id is evaluator metadata and
    must never appear in the packet.
    """
    condition_id: ConditionID
    observation_mode: ObservationMode
    governor_enabled: bool

    @property
    def short_id(self) -> str:
        """Short label for logging (not for model packets)."""
        return self.condition_id.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id.value,
            "observation_mode": self.observation_mode.value,
            "governor_enabled": self.governor_enabled,
        }


# Frozen condition registry — the four cells of the factorial design.
FROZEN_CONDITIONS: tuple[ExperimentalCondition, ...] = (
    ExperimentalCondition(
        condition_id=ConditionID.BLIND_NO_GOVERNOR,
        observation_mode=ObservationMode.BLIND,
        governor_enabled=False,
    ),
    ExperimentalCondition(
        condition_id=ConditionID.BLIND_GOVERNOR,
        observation_mode=ObservationMode.BLIND,
        governor_enabled=True,
    ),
    ExperimentalCondition(
        condition_id=ConditionID.AWARE_NO_GOVERNOR,
        observation_mode=ObservationMode.AWARE,
        governor_enabled=False,
    ),
    ExperimentalCondition(
        condition_id=ConditionID.AWARE_GOVERNOR,
        observation_mode=ObservationMode.AWARE,
        governor_enabled=True,
    ),
)

CONDITION_BY_ID: dict[ConditionID, ExperimentalCondition] = {
    c.condition_id: c for c in FROZEN_CONDITIONS
}


def get_condition(condition_id: ConditionID) -> ExperimentalCondition:
    """Look up a frozen condition by ID."""
    return CONDITION_BY_ID[condition_id]


def all_condition_ids() -> tuple[ConditionID, ...]:
    """Return all four condition IDs in canonical order."""
    return tuple(c.condition_id for c in FROZEN_CONDITIONS)
