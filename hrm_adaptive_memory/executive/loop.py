"""Deterministic V2B-I2 orchestration shell with policy and resource gates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import (
    CognitiveControlStore, DecisionAction, PolicyEffect)
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from .actions import ActionProposal
from .benchmark import BenchmarkTask, FrozenBenchmark
from .controller import (ControlController, ControlObservation,
                         StateAwareController)
from .executor import (ActionExecution, DeterministicActionExecutor, TaskRuntime,
                       build_cognitive_state, initial_runtime)
from .policy import FrozenPolicy
from .resources import ResourceBudget, ResourceExhausted, ResourceState


@dataclass(frozen=True)
class ActionTrace:
    proposed_action: DecisionAction
    executed_action: DecisionAction
    policy_effect: PolicyEffect
    policy_reasons: tuple[str, ...]
    outcome_code: str
    task_success: bool | None


@dataclass(frozen=True)
class TaskRun:
    task_id: str
    condition: str
    traces: tuple[ActionTrace, ...]
    task_success: bool
    resources: Mapping[str, int]


@dataclass(frozen=True)
class ConditionRun:
    condition: str
    controller_id: str
    tasks: tuple[TaskRun, ...]
    metrics: Mapping[str, float | int]


class V2BExperimentLoop:
    """A model never executes tools: proposal → policy → resource → executor."""

    def __init__(self, *, policy: FrozenPolicy, executor: DeterministicActionExecutor | None = None,
                 budget: ResourceBudget = ResourceBudget()):
        self.policy = policy
        self.executor = executor or DeterministicActionExecutor()
        self.budget = budget
        self._timestamp_index = 0

    def _timestamp(self) -> str:
        value = datetime(2026, 8, 12, tzinfo=timezone.utc) + timedelta(microseconds=self._timestamp_index)
        self._timestamp_index += 1
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _control_observation(runtime: TaskRuntime) -> ControlObservation:
        resources = runtime.resources.as_dict()
        return ControlObservation(runtime.task.task_id, runtime.task.task_summary,
                                  resources["executive_steps_used"],
                                  resources["executive_steps_remaining"])

    @staticmethod
    def _resolve_action(proposal: ActionProposal, runtime: TaskRuntime, policy_result
                        ) -> tuple[DecisionAction, str]:
        if policy_result.effect is PolicyEffect.DENY:
            return DecisionAction.DEFER, "POLICY_DENY"
        if policy_result.effect is PolicyEffect.REQUIRE:
            assert policy_result.required_action is not None
            return policy_result.required_action, "POLICY_REQUIRE"
        if runtime.resources.can_execute(proposal.action):
            return proposal.action, "POLICY_ALLOW"
        return DecisionAction.DEFER, "RESOURCE_EXHAUSTED"

    def _record(self, store: CognitiveControlStore, runtime: TaskRuntime, action: DecisionAction,
                proposal: ActionProposal, resolution: str, parent: str | None):
        reason = proposal.reason_code if resolution == "POLICY_ALLOW" else resolution
        return store.record_decision(
            task_id=runtime.task.task_id, selected_action=action,
            alternatives_considered=tuple(item for item in V2B_ACTIONS if item is not action),
            observations=(runtime.verification_state.value, runtime.temporal_status.value),
            evidence_used=(f"benchmark-evidence-{runtime.task.task_id}",),
            memory_used=(f"benchmark-memory-{runtime.task.task_id}",),
            policy_id=self.policy.policy_id, reason_code=reason,
            resource_state=runtime.resources.as_dict(), expected_utility=None,
            uncertainty="DETERMINISTIC_BENCHMARK", timestamp=self._timestamp(),
            operation_id=f"{runtime.task.task_id}:{store.root.name}:decision:{len(store.decisions)}",
            parent_decision_id=parent)

    def run_task(self, task: BenchmarkTask, *, condition: str,
                 controller: ControlController | StateAwareController,
                 store: CognitiveControlStore) -> TaskRun:
        runtime = initial_runtime(task, ResourceState(self.budget))
        traces: list[ActionTrace] = []
        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []
        parent_id: str | None = None
        while True:
            if condition == "CONTROL":
                proposal = controller.choose(self._control_observation(runtime))  # type: ignore[attr-defined]
            elif condition == "V2B":
                snapshot = build_cognitive_state(runtime, prior_decisions=tuple(prior_decisions),
                                                 prior_outcomes=tuple(prior_outcomes))
                proposal = controller.choose(snapshot)  # type: ignore[attr-defined]
            else:
                raise ValueError("V2B-I2 condition must be CONTROL or V2B")
            policy_result = self.policy.gate.evaluate(task.task_id, proposal.action,
                                                      build_cognitive_state(
                                                          runtime, prior_decisions=tuple(prior_decisions),
                                                          prior_outcomes=tuple(prior_outcomes)).policy_facts)
            action, resolution = self._resolve_action(proposal, runtime, policy_result)
            if not runtime.resources.can_execute(action):
                action, resolution = DecisionAction.DEFER, "RESOURCE_EXHAUSTED"
            decision = self._record(store, runtime, action, proposal, resolution, parent_id)
            try:
                execution = self.executor.execute(runtime, action)
            except ResourceExhausted:
                # No further work can be done; record a terminal failure without attempting a tool.
                execution = ActionExecution(DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")
            outcome = {"outcome_code": execution.outcome_code, "task_success": execution.task_success}
            completed = store.record_outcome(
                decision.decision_id, outcome=outcome, at=self._timestamp(),
                operation_id=f"{task.task_id}:{store.root.name}:outcome:{len(store.decisions)}")
            prior_decisions.append(DecisionSummary(completed.decision_id, action.value,
                                                    completed.reason_code, execution.outcome_code))
            prior_outcomes.append(execution.outcome_code)
            traces.append(ActionTrace(proposal.action, action, policy_result.effect,
                                      policy_result.reason_codes, execution.outcome_code,
                                      execution.task_success))
            runtime, parent_id = execution.runtime, completed.decision_id
            if execution.terminal:
                return TaskRun(task.task_id, condition, tuple(traces), bool(execution.task_success),
                               runtime.resources.as_dict())

    def run_condition(self, benchmark: FrozenBenchmark, *, condition: str,
                      controller: ControlController | StateAwareController,
                      store_root: str | Path) -> ConditionRun:
        store = CognitiveControlStore(Path(store_root) / condition.lower())
        tasks = tuple(self.run_task(task, condition=condition, controller=controller, store=store)
                      for task in benchmark.tasks)
        return ConditionRun(condition, controller.controller_id, tasks, self._metrics(tasks))

    @staticmethod
    def _metrics(tasks: Iterable[TaskRun]) -> Mapping[str, float | int]:
        tasks = tuple(tasks)
        traces = tuple(trace for task in tasks for trace in task.traces)
        answers = tuple(trace for trace in traces if trace.executed_action is DecisionAction.ANSWER)
        deferrals = tuple(trace for trace in traces if trace.executed_action is DecisionAction.DEFER)
        useful_retrieval = sum(1 for task in tasks if task.task_success and any(
            trace.executed_action is DecisionAction.RETRIEVE for trace in task.traces))
        useful_verification = sum(1 for task in tasks if task.task_success and any(
            trace.executed_action is DecisionAction.VERIFY for trace in task.traces))
        retrievals = sum(task.resources["retrieval_calls_used"] for task in tasks)
        verifications = sum(task.resources["verification_calls_used"] for task in tasks)
        success = sum(task.task_success for task in tasks)
        return {
            "tasks": len(tasks), "task_successes": success,
            "task_success_rate": success / len(tasks) if tasks else 0.0,
            "unsupported_assertion_rate": (sum(trace.task_success is False for trace in answers) / len(answers)
                                            if answers else 0.0),
            "correct_deferrals": sum(trace.task_success is True for trace in deferrals),
            "premature_stop_rate": 0.0,
            "failure_to_stop_rate": 0.0,
            "retrieval_calls": retrievals, "verification_calls": verifications,
            "reasoning_tokens": sum(task.resources["reasoning_tokens_used"] for task in tasks),
            "logical_latency_ms": sum(task.resources["elapsed_ms"] for task in tasks),
            "policy_interventions": sum(trace.policy_effect is not PolicyEffect.ALLOW for trace in traces),
            "useful_retrieval_rate": useful_retrieval / retrievals if retrievals else 0.0,
            "useful_verification_rate": useful_verification / verifications if verifications else 0.0,
        }
