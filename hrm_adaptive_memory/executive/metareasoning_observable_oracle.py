"""Information-bound oracle for the V2B-I3.1 initial decision state.

The latent table asks what an omniscient controller could do.  This module
groups *initial* latent states that emit the same frozen controller packet and
chooses the best shared proposal under a frozen uniform prior.  It therefore
measures representation loss separately from controller decision quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .metareasoning_controller import ObservationMask, apply_observation_mask
from .metareasoning_executor import build_observable_snapshot
from .metareasoning_state import runtime_from_oracle_state
from .metareasoning_transition_table import OraclePolicyTable


OBSERVABLE_ORACLE_SCHEMA = "DAPH_V2B_OBSERVABLE_ORACLE_TABLE_V1"
OBSERVABLE_ORACLE_REVISION = "v2b-i3.1-observable-initial-oracle-v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observation_packet(runtime, table: OraclePolicyTable, mask: ObservationMask) -> dict[str, object]:
    """The exact non-private fields used for I3.1 observation equivalence.

    I3.1 evaluates the opening decision only.  That avoids treating a merged
    Markov state as if it retained a unique, path-specific action history.
    Subsequent step regret remains evaluated against the latent O(1) table.
    """
    state = table.states[table.initial_state_id]
    initial = runtime_from_oracle_state(runtime, state)
    snapshot = apply_observation_mask(
        build_observable_snapshot(initial, prior_decisions=(), prior_outcomes=()), mask)
    return {
        "instance_id": initial.task.controller_instance_id or "opaque-instance",
        "task_summary": initial.task.task_summary,
        "resource_state": initial.resources.as_dict(),
        "allowed_actions": [action.value for action in V2B_ACTIONS if initial.resources.can_execute(action)],
        "cognitive_state": None if snapshot is None else {
            "verification_states": [item.state.value for item in snapshot.verification_states],
            "provenance_summaries": list(snapshot.provenance_summaries),
            "temporal_status": snapshot.temporal_status.value,
            "conflicts": [item.conflict_id for item in snapshot.unresolved_conflicts],
            "prior_outcomes": list(snapshot.prior_outcomes),
            "observation_signals": list(snapshot.observation_signals),
        },
    }


@dataclass(frozen=True)
class ObservationClass:
    observation_hash: str
    latent_state_refs: tuple[str, ...]
    common_proposals: tuple[DecisionAction, ...]
    optimal_proposals: tuple[DecisionAction, ...]
    value: float
    q_values: Mapping[str, float]
    latent_optimal_disagreement: bool


@dataclass(frozen=True)
class ObservableOraclePolicyTable:
    mask_sha256: str
    identity_sha256: str
    classes: Mapping[str, ObservationClass]
    state_to_observation: Mapping[str, str]

    def observation_for(self, table: OraclePolicyTable) -> ObservationClass:
        key = f"{table.identity_sha256}:{table.initial_state_id}"
        return self.classes[self.state_to_observation[key]]

    def information_gap(self, table: OraclePolicyTable) -> float:
        return table.initial_value - self.observation_for(table).value

    @property
    def ambiguity_count(self) -> int:
        return sum(item.latent_optimal_disagreement for item in self.classes.values())

    def serializable(self) -> dict[str, object]:
        return {
            "schema": OBSERVABLE_ORACLE_SCHEMA,
            "mask_sha256": self.mask_sha256,
            "identity_sha256": self.identity_sha256,
            "class_count": len(self.classes),
            "classes": {
                key: {"latent_state_refs": list(item.latent_state_refs),
                      "common_proposals": [action.value for action in item.common_proposals],
                      "optimal_proposals": [action.value for action in item.optimal_proposals],
                      "value": item.value, "q_values": dict(item.q_values),
                      "latent_optimal_disagreement": item.latent_optimal_disagreement}
                for key, item in sorted(self.classes.items())
            },
        }

    @property
    def table_sha256(self) -> str:
        return _canonical_hash(self.serializable())


def build_observable_oracle(*, runtime_tables: Iterable[tuple[object, OraclePolicyTable]],
                            mask: ObservationMask) -> ObservableOraclePolicyTable:
    """Build one uniform-prior information-bound table for an observation mask."""
    runtime_tables = tuple(runtime_tables)
    grouped: dict[str, list[tuple[str, OraclePolicyTable]]] = {}
    for runtime, table in runtime_tables:
        packet_hash = _canonical_hash(observation_packet(runtime, table, mask))
        reference = f"{table.identity_sha256}:{table.initial_state_id}"
        grouped.setdefault(packet_hash, []).append((reference, table))
    classes: dict[str, ObservationClass] = {}
    state_to_observation: dict[str, str] = {}
    for packet_hash, members in sorted(grouped.items()):
        proposal_sets = [set(action for origin, action in table.proposal_q_values
                             if origin == table.initial_state_id) for _, table in members]
        common = tuple(sorted(set.intersection(*proposal_sets) if proposal_sets else set(),
                              key=lambda action: action.value))
        q_values = {
            action.value: sum(table.proposal_q_values[(table.initial_state_id, action)]
                              for _, table in members) / len(members)
            for action in common
        }
        best = max(q_values.values(), default=float("-inf"))
        optimal = tuple(sorted((DecisionAction(action) for action, value in q_values.items()
                                if abs(value - best) <= 1e-12), key=lambda action: action.value))
        latent_actions = {tuple(action.value for action in table.optimal_actions[table.initial_state_id])
                          for _, table in members}
        item = ObservationClass(packet_hash, tuple(sorted(ref for ref, _ in members)), common,
                                optimal, best, q_values, len(latent_actions) > 1)
        classes[packet_hash] = item
        for reference, _ in members:
            state_to_observation[reference] = packet_hash
    identities = sorted(table.identity_sha256 for _, table in runtime_tables)
    identity = _canonical_hash({"latent_tables": identities, "mask_sha256": mask.sha256(),
                                "prior": "UNIFORM_BY_INITIAL_OBSERVATION_CLASS_V1",
                                "revision": OBSERVABLE_ORACLE_REVISION})
    return ObservableOraclePolicyTable(mask.sha256(), identity, classes, state_to_observation)
