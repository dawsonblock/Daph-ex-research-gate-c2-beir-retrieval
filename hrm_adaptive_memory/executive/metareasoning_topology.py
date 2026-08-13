"""Behavior-derived control-topology identities for V2B-I3.3.2.

The topology deliberately excludes task ids, surface text, generator labels,
state values, and budget-profile names.  It commits only the executable
proposal/policy/transition graph reachable from the frozen initial state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
from math import inf
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .metareasoning_transition_table import OraclePolicyTable, TransitionResult


TOPOLOGY_SCHEMA = "DAPH_V2B_I3_3_2_TRANSITION_TOPOLOGY_V1"
TOPOLOGY_IMPLEMENTATION_REVISION = "v2b-i3.3.2-behavior-derived-v1"


def _sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _edge_label(action: DecisionAction, transition: TransitionResult) -> dict[str, object]:
    return {
        "proposed_action": action.value,
        "policy_effect": transition.policy_effect,
        "resolved_action": (
            transition.resolved_action.value if transition.resolved_action is not None else None),
        "terminal": transition.terminal,
        "terminal_result": transition.terminal_result,
        "task_success": transition.task_success,
    }


@dataclass(frozen=True)
class TransitionTopology:
    sha256: str
    canonical_graph: Mapping[str, object]
    minimum_optimal_trajectory_depth: int | None
    maximum_relevant_trajectory_depth: int
    decision_branch_points: int
    policy_intervention_edges: int

    @property
    def depth_band(self) -> str:
        depth = self.minimum_optimal_trajectory_depth
        if depth is None:
            return "NO_SUCCESS_PATH"
        if depth <= 1:
            return "DEPTH_1"
        if depth == 2:
            return "DEPTH_2"
        if depth == 3:
            return "DEPTH_3"
        return "DEPTH_4_PLUS"


def transition_topology(table: OraclePolicyTable) -> TransitionTopology:
    """Canonicalize the reachable control graph by recursive bisimulation hash."""
    state_fingerprints: dict[str, str] = {}
    state_material: dict[str, object] = {}
    minimum_success_depth: dict[str, float] = {}
    maximum_depth: dict[str, int] = {}
    branch_points = 0
    intervention_edges = 0
    outgoing: dict[str, list[tuple[DecisionAction, TransitionResult]]] = defaultdict(list)
    for (origin, action), transition in table.proposal_transitions.items():
        outgoing[origin].append((action, transition))
    for edges in outgoing.values():
        edges.sort(key=lambda item: item[0].value)

    ordered_states = sorted(
        table.states,
        key=lambda state_id: (table.states[state_id].steps_remaining, state_id),
    )
    for state_id in ordered_states:
        edges: list[dict[str, object]] = []
        successor_kinds: set[str] = set()
        state_minimum = inf
        state_maximum = 0
        for action, transition in outgoing.get(state_id, ()):
            label = _edge_label(action, transition)
            optimal_edge = (
                transition.resolved_action is not None
                and transition.resolved_action in table.optimal_actions[state_id]
            )
            if transition.policy_effect != "ALLOW":
                intervention_edges += 1
            if transition.terminal:
                successor = {
                    "kind": "terminal",
                    "terminal_result": transition.terminal_result,
                    "task_success": transition.task_success,
                }
                successor_key = _sha256(successor)
                if transition.task_success and optimal_edge:
                    state_minimum = min(state_minimum, 1)
                state_maximum = max(state_maximum, 1)
            else:
                if transition.next_state_id is None:
                    raise RuntimeError("nonterminal topology edge lacks successor")
                child = transition.next_state_id
                successor = {"kind": "state", "fingerprint": state_fingerprints[child]}
                successor_key = state_fingerprints[child]
                child_minimum = minimum_success_depth[child]
                if optimal_edge and child_minimum != inf:
                    state_minimum = min(state_minimum, 1 + child_minimum)
                state_maximum = max(state_maximum, 1 + maximum_depth[child])
            successor_kinds.add(successor_key)
            edges.append({"edge": label, "successor": successor})
        material = {"edges": edges}
        state_material[state_id] = material
        state_fingerprints[state_id] = _sha256(material)
        minimum_success_depth[state_id] = state_minimum
        maximum_depth[state_id] = state_maximum
        if len(successor_kinds) > 1:
            branch_points += 1

    root = table.initial_state_id
    # A multiset of reachable node fingerprints prevents two roots with the
    # same immediate view but different reachable graph multiplicity from
    # sharing an identity.
    fingerprint_counts: dict[str, int] = {}
    for fingerprint in state_fingerprints.values():
        fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
    canonical_graph = {
        "schema": TOPOLOGY_SCHEMA,
        "implementation_revision": TOPOLOGY_IMPLEMENTATION_REVISION,
        "root_fingerprint": state_fingerprints[root],
        "reachable_fingerprint_multiset": dict(sorted(fingerprint_counts.items())),
    }
    minimum = minimum_success_depth[root]
    return TransitionTopology(
        sha256=_sha256(canonical_graph),
        canonical_graph=canonical_graph,
        minimum_optimal_trajectory_depth=None if minimum == inf else int(minimum),
        maximum_relevant_trajectory_depth=maximum_depth[root],
        decision_branch_points=branch_points,
        policy_intervention_edges=intervention_edges,
    )
