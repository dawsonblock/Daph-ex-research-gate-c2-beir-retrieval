"""Reachability-indexed, backwards-DP latent oracle for V2B-I3.1.

The table performs policy resolution and deterministic execution once per
reachable state/action.  Evaluation then consists of dictionary lookups; it
never recursively replays the environment for each trajectory step.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import resource
from time import perf_counter
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction, PolicyEffect

from .metareasoning_benchmark import I3BenchmarkTask
from .metareasoning_executor import (
    DeterministicMetareasoningExecutor, I3Runtime, initial_i3_runtime, policy_facts)
from .metareasoning_state import OracleState, canonicalize_runtime_state, runtime_from_oracle_state
from .metareasoning_utility import MetareasoningUtility, frozen_action_cost_hash
from .policy import FrozenPolicy
from .resources import ResourceBudget, ResourceExhausted, ResourceState


ORACLE_TABLE_SCHEMA = "DAPH_V2B_ORACLE_TABLE_V1"
ORACLE_IMPLEMENTATION_REVISION = "v2b-i3.2.2-cost-reward-separated-v1"
DEFAULT_MAX_STATES = 20_000
DEFAULT_MAX_TRANSITIONS = 140_000


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _task_hash(task: I3BenchmarkTask) -> str:
    return _hash(asdict(task))


def _budget_hash(task: I3BenchmarkTask, runtime: I3Runtime) -> str:
    return _hash(asdict(runtime.resources.budget))


@dataclass(frozen=True)
class TransitionResult:
    next_state_id: str | None
    terminal: bool
    terminal_result: str | None
    task_success: bool | None
    action_cost: float
    immediate_reward: float
    policy_effect: str
    resolved_action: DecisionAction | None


@dataclass(frozen=True)
class OraclePolicyTable:
    task_id: str
    budget_profile: str
    identity_sha256: str
    initial_state_id: str
    states: Mapping[str, OracleState]
    transitions: Mapping[tuple[str, DecisionAction], TransitionResult]
    proposal_transitions: Mapping[tuple[str, DecisionAction], TransitionResult]
    state_values: Mapping[str, float]
    q_values: Mapping[tuple[str, DecisionAction], float]
    proposal_q_values: Mapping[tuple[str, DecisionAction], float]
    optimal_actions: Mapping[str, tuple[DecisionAction, ...]]
    minimum_remaining_cost: Mapping[str, float]
    build_metrics: Mapping[str, float | int]

    @property
    def initial_value(self) -> float:
        return self.state_values[self.initial_state_id]

    def state_id_for(self, runtime: I3Runtime) -> str:
        return canonicalize_runtime_state(runtime).state_id()

    def action_regret(self, state_id: str, action: DecisionAction) -> float:
        value = self.q_values.get((state_id, action))
        if value is None:
            return float("inf")
        return max(0.0, self.state_values[state_id] - value)

    def proposal_regret(self, state_id: str, action: DecisionAction) -> float:
        value = self.proposal_q_values.get((state_id, action))
        if value is None:
            return float("inf")
        return max(0.0, self.state_values[state_id] - value)

    def serializable(self) -> dict[str, object]:
        return {
            "schema": ORACLE_TABLE_SCHEMA,
            "task_id": self.task_id,
            "budget_profile": self.budget_profile,
            "identity_sha256": self.identity_sha256,
            "state_count": len(self.states),
            "transition_count": len(self.transitions),
            "initial_state_id": self.initial_state_id,
            "initial_value": self.initial_value,
            "states": {key: state.as_dict() for key, state in sorted(self.states.items())},
            "state_values": dict(sorted(self.state_values.items())),
            "q_values": {f"{state}:{action.value}": value for (state, action), value
                         in sorted(self.q_values.items(), key=lambda item: (item[0][0], item[0][1].value))},
            "optimal_actions": {state: [action.value for action in actions]
                                for state, actions in sorted(self.optimal_actions.items())},
            "build_metrics": dict(self.build_metrics),
        }

    @property
    def table_sha256(self) -> str:
        # Build timing/RSS are operational telemetry, not oracle semantics.
        # Excluding them makes the table identity deterministic across runs.
        material = self.serializable()
        material.pop("build_metrics", None)
        return _hash(material)


def _resolve(policy: FrozenPolicy, runtime: I3Runtime,
             proposed: DecisionAction) -> tuple[PolicyEffect, DecisionAction | None]:
    decision = policy.gate.evaluate(runtime.task.task_id, proposed, policy_facts(runtime))
    if decision.effect is PolicyEffect.DENY:
        return decision.effect, None
    resolved = decision.required_action if decision.effect is PolicyEffect.REQUIRE else proposed
    return decision.effect, resolved


def build_oracle_policy_table(*, task: I3BenchmarkTask, policy: FrozenPolicy,
                              utility: MetareasoningUtility,
                              budget: ResourceBudget,
                              include_policy_feedback: bool = False,
                              max_states: int = DEFAULT_MAX_STATES,
                              max_transitions: int = DEFAULT_MAX_TRANSITIONS) -> OraclePolicyTable:
    """Enumerate a finite latent graph once, then solve it backwards exactly."""
    return build_oracle_policy_table_for_runtime(
        initial_runtime=initial_i3_runtime(task, ResourceState(budget)), policy=policy,
        utility=utility, include_policy_feedback=include_policy_feedback,
        max_states=max_states, max_transitions=max_transitions)


def build_oracle_policy_table_for_runtime(*, initial_runtime: I3Runtime, policy: FrozenPolicy,
                                          utility: MetareasoningUtility,
                                          include_policy_feedback: bool = False,
                                          max_states: int = DEFAULT_MAX_STATES,
                                          max_transitions: int = DEFAULT_MAX_TRANSITIONS) -> OraclePolicyTable:
    """Build a table from an explicitly budgeted initial runtime."""
    started = perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    task = initial_runtime.task
    executor = DeterministicMetareasoningExecutor()
    initial_state = canonicalize_runtime_state(initial_runtime)
    initial_id = initial_state.state_id()
    states: dict[str, OracleState] = {initial_id: initial_state}
    runtimes: dict[str, I3Runtime] = {initial_id: initial_runtime}
    transitions: dict[tuple[str, DecisionAction], TransitionResult] = {}
    proposal_transitions: dict[tuple[str, DecisionAction], TransitionResult] = {}
    queue: deque[str] = deque([initial_id])

    while queue:
        state_id = queue.popleft()
        runtime = runtimes[state_id]
        for proposed in V2B_ACTIONS:
            effect, resolved = _resolve(policy, runtime, proposed)
            if resolved is None or not runtime.resources.can_execute(resolved):
                if not include_policy_feedback:
                    continue
                try:
                    rejected_runtime = I3Runtime(
                        task=runtime.task, resources=runtime.resources.consume_policy_rejection(),
                        verification_state=runtime.verification_state, temporal_status=runtime.temporal_status,
                        unresolved_conflict=runtime.unresolved_conflict,
                        composition_complete=runtime.composition_complete,
                        provenance_count=runtime.provenance_count,
                        conflict_resolvable=runtime.conflict_resolvable,
                        prior_outcomes=runtime.prior_outcomes,
                        retrieved=runtime.retrieved, searched=runtime.searched)
                except ResourceExhausted:
                    continue
                next_state = canonicalize_runtime_state(rejected_runtime)
                next_id = next_state.state_id()
                if next_state.steps_remaining >= states[state_id].steps_remaining:
                    raise RuntimeError("ORACLE_ZERO_COST_CYCLE")
                if next_id not in states:
                    if len(states) >= max_states:
                        raise RuntimeError("ORACLE_STATE_SPACE_LIMIT")
                    states[next_id] = next_state
                    runtimes[next_id] = rejected_runtime
                    queue.append(next_id)
                result = TransitionResult(
                    next_state_id=next_id, terminal=False, terminal_result=None,
                    task_success=None,
                    action_cost=utility.action_cost(runtime.resources, rejected_runtime.resources),
                    immediate_reward=utility.immediate_reward(
                        before=runtime.resources, after=rejected_runtime.resources),
                    policy_effect=effect.value, resolved_action=None)
                proposal_transitions[(state_id, proposed)] = result
                continue
            execution = executor.execute(runtime, resolved)
            cost = utility.action_cost(runtime.resources, execution.runtime.resources)
            immediate_reward = utility.immediate_reward(
                before=runtime.resources, after=execution.runtime.resources)
            next_id: str | None = None
            if not execution.terminal:
                next_state = canonicalize_runtime_state(execution.runtime)
                next_id = next_state.state_id()
                # Every transition consumes an executive step.  This catches
                # zero-cost/zero-budget cycles before dynamic programming.
                if next_state.steps_remaining >= states[state_id].steps_remaining:
                    raise RuntimeError("ORACLE_ZERO_COST_CYCLE")
                if next_id not in states:
                    if len(states) >= max_states:
                        raise RuntimeError("ORACLE_STATE_SPACE_LIMIT")
                    states[next_id] = next_state
                    runtimes[next_id] = execution.runtime
                    queue.append(next_id)
            result = TransitionResult(
                next_state_id=next_id, terminal=execution.terminal,
                terminal_result=execution.outcome_code if execution.terminal else None,
                task_success=execution.task_success, action_cost=cost,
                immediate_reward=immediate_reward,
                policy_effect=effect.value, resolved_action=resolved)
            proposal_transitions[(state_id, proposed)] = result
            transitions.setdefault((state_id, resolved), result)
            if len(transitions) > max_transitions:
                raise RuntimeError("ORACLE_STATE_SPACE_LIMIT")

    # Backward dynamic programming.  Strictly lower step count successors make
    # this a finite DAG even when other resource dimensions are unchanged.
    values: dict[str, float] = {}
    q_values: dict[tuple[str, DecisionAction], float] = {}
    optimal: dict[str, tuple[DecisionAction, ...]] = {}
    min_cost: dict[str, float] = {}
    failure_value = min(utility.incorrect_answer, utility.incorrect_defer, utility.incorrect_stop)
    for state_id in sorted(states, key=lambda item: (states[item].steps_remaining, item)):
        candidates: dict[DecisionAction, float] = {}
        costs: list[float] = []
        for (origin, action), transition in transitions.items():
            if origin != state_id:
                continue
            if transition.terminal:
                assert transition.task_success is not None
                value = (transition.immediate_reward - transition.action_cost
                         + utility.terminal_reward(action, transition.task_success))
                if transition.task_success:
                    costs.append(transition.action_cost)
            else:
                assert transition.next_state_id is not None
                value = transition.immediate_reward - transition.action_cost + values[transition.next_state_id]
                later = min_cost[transition.next_state_id]
                if later != float("inf"):
                    costs.append(transition.action_cost + later)
            candidates[action] = value
            q_values[(state_id, action)] = value
        if not candidates:
            values[state_id] = failure_value
            optimal[state_id] = ()
            min_cost[state_id] = float("inf")
            continue
        best = max(candidates.values())
        values[state_id] = best
        optimal[state_id] = tuple(sorted((action for action, value in candidates.items()
                                          if abs(value - best) <= 1e-12), key=lambda action: action.value))
        min_cost[state_id] = min(costs, default=float("inf"))

    proposal_q: dict[tuple[str, DecisionAction], float] = {}
    for key, transition in proposal_transitions.items():
        if transition.resolved_action is None:
            assert transition.next_state_id is not None
            proposal_q[key] = (transition.immediate_reward - transition.action_cost
                               + values[transition.next_state_id])
        else:
            proposal_q[key] = q_values[(key[0], transition.resolved_action)]
    identity = _hash({
        "task_sha256": _task_hash(task), "budget_sha256": _budget_hash(task, initial_runtime),
        "policy_sha256": policy.sha256, "utility_sha256": utility.sha256,
        "action_cost_sha256": frozen_action_cost_hash(),
        "include_policy_feedback": include_policy_feedback,
        "oracle_implementation_revision": ORACLE_IMPLEMENTATION_REVISION,
    })
    elapsed = perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    table = OraclePolicyTable(
        task_id=task.task_id, budget_profile=task.budget_profile, identity_sha256=identity,
        initial_state_id=initial_id, states=states, transitions=transitions,
        proposal_transitions=proposal_transitions, state_values=values, q_values=q_values,
        proposal_q_values=proposal_q, optimal_actions=optimal, minimum_remaining_cost=min_cost,
        build_metrics={"reachable_states": len(states), "reachable_transitions": len(transitions),
                       "oracle_build_seconds": elapsed,
                       "peak_resident_memory_delta_kib": max(0, rss_after - rss_before),
                       "table_bytes": len(json.dumps({"states": {key: state.as_dict() for key, state in states.items()},
                                                        "values": values}, sort_keys=True)),
                       "cache_hit_rate": 0.0})
    return table


class OracleTableCache:
    """Persistent per-process table cache shared by every observation condition."""

    def __init__(self) -> None:
        self._tables: dict[str, OraclePolicyTable] = {}
        self.hits = 0
        self.misses = 0

    def get_or_build(self, *, initial_runtime: I3Runtime, policy: FrozenPolicy,
                     utility: MetareasoningUtility,
                     include_policy_feedback: bool = False) -> OraclePolicyTable:
        task = initial_runtime.task
        key = _hash({"task": _task_hash(task), "budget": _budget_hash(task, initial_runtime),
                     "policy": policy.sha256, "utility": utility.sha256,
                     "costs": frozen_action_cost_hash(), "include_policy_feedback": include_policy_feedback,
                     "revision": ORACLE_IMPLEMENTATION_REVISION})
        if key in self._tables:
            self.hits += 1
            return self._tables[key]
        self.misses += 1
        table = build_oracle_policy_table_for_runtime(
            initial_runtime=initial_runtime, policy=policy, utility=utility,
            include_policy_feedback=include_policy_feedback)
        self._tables[key] = table
        return table

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if not total else self.hits / total
