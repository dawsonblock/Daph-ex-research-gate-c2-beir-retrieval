"""Exact finite-horizon dynamic-programming oracle for V2B-I3.

The oracle is an evaluator for the deterministic benchmark, never an input to
either controller condition.  It sees latent state solely to establish a
frozen optimal-policy reference and action/trajectory regret.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction, PolicyEffect

from typing import Mapping

from .metareasoning_benchmark import I3BenchmarkTask
from .metareasoning_executor import (
    DeterministicMetareasoningExecutor, I3Runtime, policy_facts, runtime_state_hash)
from .policy import FrozenPolicy
from .resources import ResourceExhausted, ResourceState


@dataclass(frozen=True)
class OracleDecision:
    state_hash: str
    action: DecisionAction | None
    utility: float
    q_values: Mapping[str, float]
    minimum_remaining_cost: float


class ExactOptimalPolicyOracle:
    """Compute optimal utility with the same frozen policy and resource rules."""

    def __init__(self, *, task: I3BenchmarkTask, policy: FrozenPolicy,
                 utility_weights: Mapping[str, float]):
        self.task = task
        self.policy = policy
        self.executor = DeterministicMetareasoningExecutor()
        self.weights = utility_weights

    @staticmethod
    def _key(runtime: I3Runtime) -> tuple[object, ...]:
        r = runtime.resources
        return (
            runtime.verification_state.value, runtime.temporal_status.value,
            runtime.unresolved_conflict, runtime.composition_complete, runtime.retrieved,
            runtime.searched, r.executive_steps_used, r.reasoning_tokens_used,
            r.retrieval_calls_used, r.verification_calls_used, r.search_calls_used,
            r.elapsed_ms, r.monetary_cost_microusd,
        )

    def _runtime(self, initial: I3Runtime, key: tuple[object, ...]) -> I3Runtime:
        (verification, temporal, conflict, composition, retrieved, searched, steps, tokens,
         retrievals, verifications, searches, elapsed, monetary) = key
        resources = ResourceState(
            initial.resources.budget, executive_steps_used=int(steps), reasoning_tokens_used=int(tokens),
            retrieval_calls_used=int(retrievals), verification_calls_used=int(verifications),
            search_calls_used=int(searches), elapsed_ms=int(elapsed),
            monetary_cost_microusd=int(monetary),
        )
        return replace(initial, verification_state=type(initial.verification_state)(verification),
                       temporal_status=type(initial.temporal_status)(temporal),
                       unresolved_conflict=bool(conflict), composition_complete=bool(composition),
                       retrieved=bool(retrieved), searched=bool(searched), resources=resources)

    def action_cost(self, before: I3Runtime, after: I3Runtime) -> float:
        used_before, used_after = before.resources, after.resources
        return -(
            self.weights["executive_step"] * (used_after.executive_steps_used - used_before.executive_steps_used)
            + self.weights["retrieval"] * (used_after.retrieval_calls_used - used_before.retrieval_calls_used)
            + self.weights["verification"] * (used_after.verification_calls_used - used_before.verification_calls_used)
            + self.weights["search"] * (used_after.search_calls_used - used_before.search_calls_used)
            + self.weights["reasoning_128_tokens"] *
            ((used_after.reasoning_tokens_used - used_before.reasoning_tokens_used) / 128)
            + self.weights["logical_ms"] * (used_after.elapsed_ms - used_before.elapsed_ms)
        )

    def terminal_utility(self, task_success: bool) -> float:
        return self.weights["success_reward"] if task_success else -self.weights["failure_penalty"]

    def _resolve(self, runtime: I3Runtime, proposed: DecisionAction) -> DecisionAction | None:
        decision = self.policy.gate.evaluate(runtime.task.task_id, proposed, policy_facts(runtime))
        if decision.effect is PolicyEffect.DENY:
            return None
        action = decision.required_action if decision.effect is PolicyEffect.REQUIRE else proposed
        assert action is not None
        return action if runtime.resources.can_execute(action) else None

    def legal_actions(self, runtime: I3Runtime) -> tuple[DecisionAction, ...]:
        return tuple(sorted({resolved for proposed in V2B_ACTIONS
                             if (resolved := self._resolve(runtime, proposed)) is not None},
                            key=lambda action: action.value))

    def _execute_value(self, runtime: I3Runtime, action: DecisionAction,
                       continuation) -> float:
        try:
            result = self.executor.execute(runtime, action)
        except ResourceExhausted:
            return float("-inf")
        utility = self.action_cost(runtime, result.runtime)
        if result.terminal:
            assert result.task_success is not None
            return utility + self.terminal_utility(result.task_success)
        return utility + continuation(self._key(result.runtime))

    def solve(self, initial: I3Runtime) -> OracleDecision:
        @lru_cache(maxsize=None)
        def value(key: tuple[object, ...]) -> float:
            runtime = self._runtime(initial, key)
            candidates = []
            for proposed in V2B_ACTIONS:
                action = self._resolve(runtime, proposed)
                if action is not None:
                    candidates.append(self._execute_value(runtime, action, value))
            return max(candidates, default=-self.weights["failure_penalty"])

        key = self._key(initial)
        q_values: dict[str, float] = {}
        for action in self.legal_actions(initial):
            q_values[action.value] = self._execute_value(initial, action, value)
        best_action = max(q_values, key=q_values.get) if q_values else None

        @lru_cache(maxsize=None)
        def minimum_success_cost(key: tuple[object, ...]) -> float:
            runtime = self._runtime(initial, key)
            costs: list[float] = []
            for action in self.legal_actions(runtime):
                result = self.executor.execute(runtime, action)
                immediate = -self.action_cost(runtime, result.runtime)
                if result.terminal:
                    if result.task_success:
                        costs.append(immediate)
                else:
                    next_cost = minimum_success_cost(self._key(result.runtime))
                    if next_cost != float("inf"):
                        costs.append(immediate + next_cost)
            return min(costs, default=float("inf"))

        return OracleDecision(runtime_state_hash(initial),
                              None if best_action is None else DecisionAction(best_action), value(key),
                              q_values, minimum_success_cost(key))

    def action_value(self, runtime: I3Runtime, action: DecisionAction) -> float:
        """Optimal continuation value if an already-authorized action executes now."""
        @lru_cache(maxsize=None)
        def value(key: tuple[object, ...]) -> float:
            current = self._runtime(runtime, key)
            candidates = []
            for proposed in V2B_ACTIONS:
                resolved = self._resolve(current, proposed)
                if resolved is not None:
                    candidates.append(self._execute_value(current, resolved, value))
            return max(candidates, default=-self.weights["failure_penalty"])
        return self._execute_value(runtime, action, value)

    def action_regret(self, runtime: I3Runtime, action: DecisionAction) -> float:
        optimum = self.solve(runtime).utility
        return max(0.0, optimum - self.action_value(runtime, action))

    def reachable_states(self, initial: I3Runtime) -> tuple[I3Runtime, ...]:
        """Enumerate finite policy-legal states for consistency/replay checks."""
        queue = [initial]
        seen: set[tuple[object, ...]] = set()
        states: list[I3Runtime] = []
        while queue:
            runtime = queue.pop()
            key = self._key(runtime)
            if key in seen:
                continue
            seen.add(key); states.append(runtime)
            for action in self.legal_actions(runtime):
                result = self.executor.execute(runtime, action)
                if not result.terminal:
                    queue.append(result.runtime)
        return tuple(states)
