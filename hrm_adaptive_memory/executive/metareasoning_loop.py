"""V2B-I3 protocol loop: masking, replanning, deltas, and oracle evaluation."""
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
from .metareasoning_benchmark import I3BenchmarkTask, MetareasoningBenchmark
from .metareasoning_controller import ControllerObservation, MatchedMetareasoningController
from .metareasoning_executor import (
    DeterministicMetareasoningExecutor, I3Runtime, answerable, build_observable_snapshot,
    initial_i3_runtime, policy_facts, runtime_state_hash, state_delta)
from .metareasoning_oracle import ExactOptimalPolicyOracle
from .policy import FrozenPolicy
from .resources import ResourceExhausted, ResourceState


STATE_BLIND = "STATE_BLIND_CONTROLLER"
STATE_AWARE = "STATE_AWARE_CONTROLLER"


@dataclass(frozen=True)
class I3ActionTrace:
    proposed_action: DecisionAction
    policy_effect: PolicyEffect
    policy_reasons: tuple[str, ...]
    policy_resolved_action: DecisionAction | None
    execution_status: str
    executed_action: DecisionAction | None
    pre_state_hash: str
    post_state_hash: str | None
    state_delta: Mapping[str, object] | None
    outcome_code: str
    task_success: bool | None
    action_regret: float | None
    answerable_before: bool
    success_reachable_before: bool


@dataclass(frozen=True)
class I3TaskRun:
    task_id: str
    condition: str
    traces: tuple[I3ActionTrace, ...]
    task_success: bool
    resources: Mapping[str, int]
    optimal_utility: float
    realized_utility: float
    trajectory_regret: float


@dataclass(frozen=True)
class I3ConditionRun:
    condition: str
    controller_id: str
    controller_algorithm_id: str
    tasks: tuple[I3TaskRun, ...]
    metrics: Mapping[str, float | int]


class V2BMetareasoningExperiment:
    """Execute state-blind/state-aware conditions over an identical MDP substrate."""

    def __init__(self, *, benchmark: MetareasoningBenchmark, policy: FrozenPolicy,
                 executor: DeterministicMetareasoningExecutor | None = None):
        self.benchmark = benchmark
        self.policy = policy
        self.executor = executor or DeterministicMetareasoningExecutor()
        self._timestamp_index = 0

    def _timestamp(self) -> str:
        value = datetime(2026, 8, 12, tzinfo=timezone.utc) + timedelta(microseconds=self._timestamp_index)
        self._timestamp_index += 1
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _observation(runtime: I3Runtime, *, state_visible: bool,
                     traces: list[I3ActionTrace], decisions: tuple[DecisionSummary, ...],
                     outcomes: tuple[str, ...]) -> ControllerObservation:
        snapshot = (build_observable_snapshot(runtime, prior_decisions=decisions, prior_outcomes=outcomes)
                    if state_visible else None)
        return ControllerObservation(
            task_id=runtime.task.task_id, task_summary=runtime.task.task_summary,
            resource_state=runtime.resources.as_dict(),
            executed_actions=tuple(trace.executed_action for trace in traces
                                   if trace.executed_action is not None),
            rejected_actions=tuple(trace.policy_resolved_action or trace.proposed_action for trace in traces
                                   if trace.execution_status in {"POLICY_REJECTED", "RESOURCE_REJECTED"}),
            cognitive_state=snapshot,
        )

    def _record(self, store: CognitiveControlStore, runtime: I3Runtime, *, selected: DecisionAction,
                proposal: ActionProposal, reason: str, parent: str | None):
        return store.record_decision(
            task_id=runtime.task.task_id, selected_action=selected,
            alternatives_considered=tuple(item for item in V2B_ACTIONS if item is not selected),
            observations=(runtime.verification_state.value, runtime.temporal_status.value,
                          "CONFLICT" if runtime.unresolved_conflict else "NO_CONFLICT"),
            evidence_used=(f"benchmark-evidence-{runtime.task.task_id}",),
            memory_used=(f"benchmark-memory-{runtime.task.task_id}",),
            policy_id=self.policy.policy_id, reason_code=reason,
            resource_state=runtime.resources.as_dict(), expected_utility=None,
            uncertainty="DETERMINISTIC_METAREASONING_DEVELOPMENT", timestamp=self._timestamp(),
            operation_id=f"{runtime.task.task_id}:{store.root.name}:decision:{len(store.decisions)}",
            parent_decision_id=parent)

    def _complete_record(self, store: CognitiveControlStore, decision, *, task: I3BenchmarkTask,
                         status: str, outcome_code: str, task_success: bool | None):
        return store.record_outcome(
            decision.decision_id,
            outcome={"execution_status": status, "outcome_code": outcome_code,
                     "task_success": task_success},
            at=self._timestamp(),
            operation_id=f"{task.task_id}:{store.root.name}:outcome:{len(store.decisions)}")

    def _run_task(self, task: I3BenchmarkTask, *, condition: str,
                  controller: MatchedMetareasoningController,
                  store: CognitiveControlStore) -> I3TaskRun:
        state_visible = condition == STATE_AWARE
        runtime = initial_i3_runtime(task, ResourceState(self.benchmark.budget_for(task)))
        oracle = ExactOptimalPolicyOracle(task=task, policy=self.policy,
                                          utility_weights=self.benchmark.utility_weights)
        initial_oracle = oracle.solve(runtime)
        traces: list[I3ActionTrace] = []
        decisions: list[DecisionSummary] = []
        outcomes: list[str] = []
        parent_id: str | None = None
        realized_utility = 0.0
        # Rejected proposals are cost-free and must replan; cap only protects a buggy controller.
        proposal_limit = runtime.resources.budget.max_executive_steps * 3 + len(V2B_ACTIONS)
        for _ in range(proposal_limit):
            observation = self._observation(runtime, state_visible=state_visible, traces=traces,
                                            decisions=tuple(decisions), outcomes=tuple(outcomes))
            proposal = controller.choose(observation)
            pre_hash = runtime_state_hash(runtime)
            pre_answerable = answerable(runtime)
            reachable = oracle.solve(runtime).utility > 0
            policy_result = self.policy.gate.evaluate(task.task_id, proposal.action, policy_facts(runtime))
            selected = (policy_result.required_action if policy_result.effect is PolicyEffect.REQUIRE
                        else proposal.action)
            if policy_result.effect is PolicyEffect.DENY:
                decision = self._record(store, runtime, selected=proposal.action, proposal=proposal,
                                        reason="POLICY_REJECTED", parent=parent_id)
                completed = self._complete_record(
                    store, decision, task=task, status="POLICY_REJECTED",
                    outcome_code="POLICY_REJECTED", task_success=None)
                decisions.append(DecisionSummary(completed.decision_id, proposal.action.value,
                                                 completed.reason_code, "POLICY_REJECTED"))
                outcomes.append("POLICY_REJECTED")
                parent_id = completed.decision_id
                traces.append(I3ActionTrace(
                    proposal.action, policy_result.effect, policy_result.reason_codes, None,
                    "POLICY_REJECTED", None, pre_hash, None, None, "POLICY_REJECTED", None,
                    None, pre_answerable, reachable))
                continue
            assert selected is not None
            if not runtime.resources.can_execute(selected):
                decision = self._record(store, runtime, selected=selected, proposal=proposal,
                                        reason="RESOURCE_REJECTED", parent=parent_id)
                completed = self._complete_record(
                    store, decision, task=task, status="RESOURCE_REJECTED",
                    outcome_code="RESOURCE_REJECTED", task_success=None)
                decisions.append(DecisionSummary(completed.decision_id, selected.value,
                                                 completed.reason_code, "RESOURCE_REJECTED"))
                outcomes.append("RESOURCE_REJECTED")
                parent_id = completed.decision_id
                traces.append(I3ActionTrace(
                    proposal.action, policy_result.effect, policy_result.reason_codes, selected,
                    "RESOURCE_REJECTED", None, pre_hash, None, None, "RESOURCE_REJECTED", None,
                    None, pre_answerable, reachable))
                continue
            decision = self._record(
                store, runtime, selected=selected, proposal=proposal,
                reason=(proposal.reason_code if policy_result.effect is PolicyEffect.ALLOW
                        else "POLICY_REQUIRED"), parent=parent_id)
            try:
                execution = self.executor.execute(runtime, selected)
            except ResourceExhausted:  # Defensive: resource acceptance and execution must agree.
                raise RuntimeError("executor rejected an action after resource acceptance") from None
            delta = state_delta(runtime, execution.runtime)
            regret = oracle.action_regret(runtime, selected)
            realized_utility += oracle.action_cost(runtime, execution.runtime)
            if execution.terminal:
                assert execution.task_success is not None
                realized_utility += oracle.terminal_utility(execution.task_success)
            completed = self._complete_record(
                store, decision, task=task, status="EXECUTED", outcome_code=execution.outcome_code,
                task_success=execution.task_success)
            decisions.append(DecisionSummary(completed.decision_id, selected.value,
                                             completed.reason_code, execution.outcome_code))
            outcomes.append(execution.outcome_code)
            parent_id = completed.decision_id
            traces.append(I3ActionTrace(
                proposal.action, policy_result.effect, policy_result.reason_codes, selected,
                "EXECUTED", selected, pre_hash, runtime_state_hash(execution.runtime), delta,
                execution.outcome_code, execution.task_success, regret, pre_answerable, reachable))
            runtime = execution.runtime
            if execution.terminal:
                success = bool(execution.task_success)
                return I3TaskRun(task.task_id, condition, tuple(traces), success,
                                 runtime.resources.as_dict(), initial_oracle.utility,
                                 realized_utility, max(0.0, initial_oracle.utility - realized_utility))
        # A controller that ignores rejection feedback cannot be credited with execution.
        realized_utility -= self.benchmark.utility_weights["failure_penalty"]
        return I3TaskRun(task.task_id, condition, tuple(traces), False, runtime.resources.as_dict(),
                         initial_oracle.utility, realized_utility,
                         max(0.0, initial_oracle.utility - realized_utility))

    def run_condition(self, *, condition: str, controller: MatchedMetareasoningController,
                      store_root: str | Path) -> I3ConditionRun:
        if condition not in {STATE_BLIND, STATE_AWARE}:
            raise ValueError("V2B-I3 condition must be STATE_BLIND_CONTROLLER or STATE_AWARE_CONTROLLER")
        store = CognitiveControlStore(Path(store_root) / condition.lower())
        tasks = tuple(self._run_task(task, condition=condition, controller=controller, store=store)
                      for task in self.benchmark.tasks)
        return I3ConditionRun(condition, controller.controller_id, controller.algorithm_id,
                              tasks, self._metrics(tasks))

    @staticmethod
    def _metrics(tasks: Iterable[I3TaskRun]) -> Mapping[str, float | int]:
        tasks = tuple(tasks)
        traces = tuple(trace for task in tasks for trace in task.traces)
        executed = tuple(trace for trace in traces if trace.execution_status == "EXECUTED")
        retrievals = tuple(trace for trace in executed if trace.executed_action is DecisionAction.RETRIEVE)
        verifications = tuple(trace for trace in executed if trace.executed_action is DecisionAction.VERIFY)
        answers = tuple(trace for trace in executed if trace.executed_action is DecisionAction.ANSWER)
        deferrals = tuple(trace for trace in executed if trace.executed_action is DecisionAction.DEFER)
        reached_sufficient = sum(any(trace.answerable_before for trace in task.traces) for task in tasks)
        avoidable_after_sufficient = sum(
            trace.answerable_before and trace.executed_action not in {
                DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}
            for trace in executed)
        success_reachable = sum(bool(task.traces and task.traces[0].success_reachable_before) for task in tasks)
        premature = sum(
            trace.execution_status == "EXECUTED" and trace.executed_action in {
                DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}
            and trace.task_success is False and trace.success_reachable_before
            for trace in traces)
        total_oracle_utility = sum(abs(task.optimal_utility) for task in tasks)
        return {
            "tasks": len(tasks),
            "task_successes": sum(task.task_success for task in tasks),
            "task_success_rate": (sum(task.task_success for task in tasks) / len(tasks) if tasks else 0.0),
            "unsupported_assertion_rate": (
                sum(trace.task_success is False for trace in answers) / len(answers) if answers else 0.0),
            "correct_deferrals": sum(trace.task_success is True for trace in deferrals),
            "policy_rejection_count": sum(trace.execution_status == "POLICY_REJECTED" for trace in traces),
            "resource_rejection_count": sum(trace.execution_status == "RESOURCE_REJECTED" for trace in traces),
            "premature_stop_rate": premature / success_reachable if success_reachable else 0.0,
            "failure_to_stop_rate": avoidable_after_sufficient / reached_sufficient if reached_sufficient else 0.0,
            "retrieval_calls": len(retrievals), "verification_calls": len(verifications),
            "search_calls": sum(trace.executed_action is DecisionAction.SEARCH_MORE for trace in executed),
            "reasoning_tokens": sum(task.resources["reasoning_tokens_used"] for task in tasks),
            "logical_latency_ms": sum(task.resources["elapsed_ms"] for task in tasks),
            "useful_retrieval_rate": (
                sum(bool(trace.state_delta and trace.state_delta["decision_relevant_improvement"])
                    for trace in retrievals) / len(retrievals) if retrievals else 0.0),
            "useful_verification_rate": (
                sum(bool(trace.state_delta and trace.state_delta["decision_relevant_improvement"])
                    for trace in verifications) / len(verifications) if verifications else 0.0),
            "mean_action_regret": (
                sum(trace.action_regret or 0.0 for trace in executed) / len(executed) if executed else 0.0),
            "trajectory_regret": sum(task.trajectory_regret for task in tasks),
            "normalized_executive_regret": (
                sum(task.trajectory_regret for task in tasks) / (total_oracle_utility + 1e-9)),
        }
