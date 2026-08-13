"""Compatibility facade over the V2B-I3.1 reachability-indexed oracle table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .metareasoning_benchmark import I3BenchmarkTask
from .metareasoning_executor import I3Runtime, initial_i3_runtime
from .metareasoning_state import runtime_from_oracle_state
from .metareasoning_transition_table import OraclePolicyTable, OracleTableCache
from .metareasoning_utility import MetareasoningUtility
from .policy import FrozenPolicy
from .resources import ResourceState


@dataclass(frozen=True)
class OracleDecision:
    state_hash: str
    action: DecisionAction | None
    utility: float
    q_values: Mapping[str, float]
    minimum_remaining_cost: float


class ExactOptimalPolicyOracle:
    """Exact oracle backed by one persistent O(1)-lookup policy table.

    The public API remains compatible with I3 callers while avoiding the old
    recursive solve/action_regret recomputation path.
    """

    _default_cache = OracleTableCache()

    def __init__(self, *, task: I3BenchmarkTask, policy: FrozenPolicy,
                 utility_weights: Mapping[str, float],
                 table_cache: OracleTableCache | None = None,
                 utility: MetareasoningUtility | None = None):
        self.task = task
        self.policy = policy
        self.utility = utility or MetareasoningUtility.from_i3_weights(utility_weights)
        self._cache = table_cache or self._default_cache
        self._table: OraclePolicyTable | None = None

    def _get_table(self, runtime: I3Runtime) -> OraclePolicyTable:
        if self._table is None:
            self._table = self._cache.get_or_build(initial_runtime=runtime, policy=self.policy,
                                                    utility=self.utility)
        return self._table

    def solve(self, runtime: I3Runtime) -> OracleDecision:
        table = self._get_table(runtime)
        state_id = table.state_id_for(runtime)
        actions = table.optimal_actions[state_id]
        q_values = {action.value: value for (origin, action), value in table.q_values.items()
                    if origin == state_id}
        return OracleDecision(state_id, actions[0] if actions else None, table.state_values[state_id],
                              q_values, table.minimum_remaining_cost[state_id])

    def action_value(self, runtime: I3Runtime, action: DecisionAction) -> float:
        table = self._get_table(runtime)
        return table.q_values.get((table.state_id_for(runtime), action), float("-inf"))

    def action_regret(self, runtime: I3Runtime, action: DecisionAction) -> float:
        table = self._get_table(runtime)
        return table.action_regret(table.state_id_for(runtime), action)

    def action_cost(self, before: I3Runtime, after: I3Runtime) -> float:
        return self.utility.action_utility(before.resources, after.resources)

    def terminal_utility(self, task_success: bool,
                         action: DecisionAction = DecisionAction.ANSWER) -> float:
        return self.utility.terminal_reward(action, task_success)

    def legal_actions(self, runtime: I3Runtime) -> tuple[DecisionAction, ...]:
        table = self._get_table(runtime)
        state = table.state_id_for(runtime)
        return tuple(sorted((action for origin, action in table.transitions if origin == state),
                            key=lambda action: action.value))

    def reachable_states(self, runtime: I3Runtime) -> tuple[I3Runtime, ...]:
        table = self._get_table(runtime)
        return tuple(runtime_from_oracle_state(runtime, table.states[state])
                     for state in sorted(table.states))

    @property
    def table(self) -> OraclePolicyTable | None:
        return self._table
