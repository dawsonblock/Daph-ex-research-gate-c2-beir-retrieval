"""V2B-I3 protocol loop: masking, replanning, deltas, and oracle evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping
from statistics import median

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import (
    CognitiveControlStore, DecisionAction, PolicyEffect)
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from .actions import ActionProposal
from .metareasoning_benchmark import I3BenchmarkTask, MetareasoningBenchmark
from .metareasoning_controller import (
    STATE_AWARE_MASK, STATE_BLIND_MASK, ControllerObservation, MatchedMetareasoningController,
    ObservationMask, apply_observation_mask)
from .metareasoning_executor import (
    DeterministicMetareasoningExecutor, I3Runtime, answerable, build_observable_snapshot,
    initial_i3_runtime, policy_facts, runtime_state_hash, state_delta)
from .metareasoning_oracle import ExactOptimalPolicyOracle
from .metareasoning_transition_table import OracleTableCache
from .metareasoning_utility import MetareasoningUtility
from .policy import FrozenPolicy
from .resources import ResourceExhausted, ResourceState


STATE_BLIND = "STATE_BLIND_CONTROLLER"
STATE_AWARE = "STATE_AWARE_CONTROLLER"
NO_VERIFICATION = "NO_VERIFICATION"
NO_PROVENANCE = "NO_PROVENANCE"
NO_TEMPORAL = "NO_TEMPORAL"
NO_CONFLICT = "NO_CONFLICT"
NO_HISTORY = "NO_HISTORY"

OBSERVATION_MASKS: Mapping[str, ObservationMask] = {
    STATE_BLIND: STATE_BLIND_MASK,
    STATE_AWARE: STATE_AWARE_MASK,
    NO_VERIFICATION: ObservationMask(provenance=True, temporal=True, conflicts=True,
                                     prior_outcomes=True, composition=True),
    NO_PROVENANCE: ObservationMask(verification=True, temporal=True, conflicts=True,
                                   prior_outcomes=True, composition=True),
    NO_TEMPORAL: ObservationMask(verification=True, provenance=True, conflicts=True,
                                 prior_outcomes=True, composition=True),
    NO_CONFLICT: ObservationMask(verification=True, provenance=True, temporal=True,
                                 prior_outcomes=True, composition=True),
    NO_HISTORY: ObservationMask(verification=True, provenance=True, temporal=True,
                                conflicts=True, composition=True),
}


@dataclass(frozen=True)
class I3ActionTrace:
    step_id: int
    observation_hash: str
    proposed_action: DecisionAction
    proposal_reason_code: str
    policy_effect: PolicyEffect
    policy_reasons: tuple[str, ...]
    policy_resolved_action: DecisionAction | None
    execution_status: str
    executed_action: DecisionAction | None
    pre_state_hash: str
    post_state_hash: str | None
    resources_before: Mapping[str, int]
    resources_after: Mapping[str, int]
    state_delta: Mapping[str, object] | None
    outcome_code: str
    task_success: bool | None
    action_regret: float | None
    action_cost: float | None
    answerable_before: bool
    success_reachable_before: bool


@dataclass(frozen=True)
class RejectedAction:
    proposed_action: DecisionAction
    policy_reason_codes: tuple[str, ...]
    step_id: int


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
    rejected_actions: tuple[RejectedAction, ...]
    terminal_result: str


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
                 executor: DeterministicMetareasoningExecutor | None = None,
                 utility: MetareasoningUtility | None = None,
                 oracle_table_cache: OracleTableCache | None = None):
        self.benchmark = benchmark
        self.policy = policy
        self.executor = executor or DeterministicMetareasoningExecutor()
        self.utility = utility or MetareasoningUtility.from_i3_weights(benchmark.utility_weights)
        self.oracle_table_cache = oracle_table_cache or OracleTableCache()
        self._timestamp_index = 0

    def _timestamp(self) -> str:
        value = datetime(2026, 8, 12, tzinfo=timezone.utc) + timedelta(microseconds=self._timestamp_index)
        self._timestamp_index += 1
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _observation(runtime: I3Runtime, *, mask: ObservationMask,
                     traces: list[I3ActionTrace], decisions: tuple[DecisionSummary, ...],
                     outcomes: tuple[str, ...]) -> ControllerObservation:
        snapshot = apply_observation_mask(
            build_observable_snapshot(runtime, prior_decisions=decisions, prior_outcomes=outcomes), mask)
        return ControllerObservation(
            # `task_id` is the opaque public instance id here; the private task
            # id stays within the runtime, policy, and receipt layers.
            task_id=(runtime.task.controller_instance_id or runtime.task.task_id),
            task_summary=runtime.task.task_summary,
            resource_state=runtime.resources.as_dict(),
            allowed_actions=tuple(action for action in V2B_ACTIONS if runtime.resources.can_execute(action)),
            executed_actions=tuple(trace.executed_action for trace in traces
                                   if trace.executed_action is not None),
            rejected_actions=tuple(trace.policy_resolved_action or trace.proposed_action for trace in traces
                                   if trace.execution_status in {"POLICY_REJECTED", "RESOURCE_REJECTED"}),
            cognitive_state=snapshot,
        )

    @staticmethod
    def _observation_hash(observation: ControllerObservation, mask: ObservationMask) -> str:
        import hashlib
        import json
        snapshot = observation.cognitive_state
        material = {
            "task_id": observation.task_id, "task_summary": observation.task_summary,
            "resource_state": observation.resource_state,
            "allowed_actions": [action.value for action in observation.allowed_actions],
            "executed_actions": [action.value for action in observation.executed_actions],
            "rejected_actions": [action.value for action in observation.rejected_actions],
            "observation_mask_sha256": mask.sha256(),
            "cognitive_state": None if snapshot is None else {
                "verification_states": [item.state.value for item in snapshot.verification_states],
                "provenance_summaries": list(snapshot.provenance_summaries),
                "temporal_status": snapshot.temporal_status.value,
                "conflicts": [item.conflict_id for item in snapshot.unresolved_conflicts],
                "prior_decisions": [item.decision_id for item in snapshot.prior_decisions],
                "prior_outcomes": list(snapshot.prior_outcomes),
                "observation_signals": list(snapshot.observation_signals),
            },
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

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
                  store: CognitiveControlStore, mask: ObservationMask) -> I3TaskRun:
        runtime = initial_i3_runtime(task, ResourceState(self.benchmark.budget_for(task)))
        oracle = ExactOptimalPolicyOracle(task=task, policy=self.policy,
                                          utility_weights=self.benchmark.utility_weights,
                                          utility=self.utility,
                                          table_cache=self.oracle_table_cache)
        initial_oracle = oracle.solve(runtime)
        traces: list[I3ActionTrace] = []
        decisions: list[DecisionSummary] = []
        outcomes: list[str] = []
        parent_id: str | None = None
        realized_utility = 0.0
        rejected_actions: list[RejectedAction] = []
        policy_rejections = 0
        max_policy_rejections = 3
        proposal_limit = runtime.resources.budget.max_executive_steps * 2 + max_policy_rejections
        for _ in range(proposal_limit):
            observation = self._observation(runtime, mask=mask, traces=traces,
                                            decisions=tuple(decisions), outcomes=tuple(outcomes))
            proposal = controller.choose(observation)
            pre_hash = runtime_state_hash(runtime)
            observation_hash = self._observation_hash(observation, mask)
            resources_before = runtime.resources.as_dict()
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
                policy_rejections += 1
                rejected_actions.append(RejectedAction(proposal.action, policy_result.reason_codes, len(traces)))
                traces.append(I3ActionTrace(
                    step_id=len(traces), observation_hash=observation_hash, proposed_action=proposal.action,
                    proposal_reason_code=proposal.reason_code, policy_effect=policy_result.effect,
                    policy_reasons=policy_result.reason_codes, policy_resolved_action=None,
                    execution_status="POLICY_REJECTED", executed_action=None, pre_state_hash=pre_hash,
                    post_state_hash=None, resources_before=resources_before,
                    resources_after=runtime.resources.as_dict(), state_delta=None,
                    outcome_code="POLICY_REJECTED", task_success=None, action_regret=None,
                    action_cost=None,
                    answerable_before=pre_answerable, success_reachable_before=reachable))
                if policy_rejections >= max_policy_rejections:
                    realized_utility += self.utility.incorrect_defer
                    return I3TaskRun(
                        task.task_id, condition, tuple(traces), False, runtime.resources.as_dict(),
                        initial_oracle.utility, realized_utility,
                        max(0.0, initial_oracle.utility - realized_utility), tuple(rejected_actions),
                        "POLICY_REJECTION_LIMIT")
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
                rejected_actions.append(RejectedAction(selected, policy_result.reason_codes, len(traces)))
                traces.append(I3ActionTrace(
                    step_id=len(traces), observation_hash=observation_hash, proposed_action=proposal.action,
                    proposal_reason_code=proposal.reason_code, policy_effect=policy_result.effect,
                    policy_reasons=policy_result.reason_codes, policy_resolved_action=selected,
                    execution_status="RESOURCE_REJECTED", executed_action=None, pre_state_hash=pre_hash,
                    post_state_hash=None, resources_before=resources_before,
                    resources_after=runtime.resources.as_dict(), state_delta=None,
                    outcome_code="RESOURCE_REJECTED", task_success=None, action_regret=None,
                    action_cost=None,
                    answerable_before=pre_answerable, success_reachable_before=reachable))
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
            action_cost = oracle.action_cost(runtime, execution.runtime)
            realized_utility -= action_cost
            if execution.terminal:
                assert execution.task_success is not None
                realized_utility += oracle.terminal_utility(execution.task_success, selected)
            completed = self._complete_record(
                store, decision, task=task, status="EXECUTED", outcome_code=execution.outcome_code,
                task_success=execution.task_success)
            decisions.append(DecisionSummary(completed.decision_id, selected.value,
                                             completed.reason_code, execution.outcome_code))
            outcomes.append(execution.outcome_code)
            parent_id = completed.decision_id
            traces.append(I3ActionTrace(
                step_id=len(traces), observation_hash=observation_hash, proposed_action=proposal.action,
                proposal_reason_code=proposal.reason_code, policy_effect=policy_result.effect,
                policy_reasons=policy_result.reason_codes, policy_resolved_action=selected,
                execution_status="EXECUTED", executed_action=selected, pre_state_hash=pre_hash,
                post_state_hash=runtime_state_hash(execution.runtime), resources_before=resources_before,
                resources_after=execution.runtime.resources.as_dict(), state_delta=delta,
                outcome_code=execution.outcome_code, task_success=execution.task_success,
                action_regret=regret, action_cost=action_cost, answerable_before=pre_answerable,
                success_reachable_before=reachable))
            runtime = execution.runtime
            if execution.terminal:
                success = bool(execution.task_success)
                return I3TaskRun(task.task_id, condition, tuple(traces), success,
                                 runtime.resources.as_dict(), initial_oracle.utility,
                                 realized_utility, max(0.0, initial_oracle.utility - realized_utility),
                                 tuple(rejected_actions), execution.outcome_code)
        # A controller that ignores rejection feedback cannot be credited with execution.
        realized_utility -= self.benchmark.utility_weights["failure_penalty"]
        return I3TaskRun(task.task_id, condition, tuple(traces), False, runtime.resources.as_dict(),
                         initial_oracle.utility, realized_utility,
                         max(0.0, initial_oracle.utility - realized_utility), tuple(rejected_actions),
                         "PROPOSAL_LIMIT")

    def run_condition(self, *, condition: str, controller: MatchedMetareasoningController,
                      store_root: str | Path, mask: ObservationMask | None = None) -> I3ConditionRun:
        mask = OBSERVATION_MASKS.get(condition) if mask is None else mask
        if mask is None:
            raise ValueError("V2B-I3 condition needs an explicit observation mask")
        store = CognitiveControlStore(Path(store_root) / condition.lower())
        tasks = tuple(self._run_task(task, condition=condition, controller=controller, store=store, mask=mask)
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
        policy_violations = sum(
            trace.execution_status == "EXECUTED" and (
                (trace.policy_effect is PolicyEffect.DENY)
                or (trace.policy_effect is PolicyEffect.REQUIRE
                    and trace.executed_action is not trace.policy_resolved_action))
            for trace in traces)
        normalized_regrets = sorted(
            task.trajectory_regret / (abs(task.optimal_utility) + 1e-9) for task in tasks)
        p90_index = max(0, min(len(normalized_regrets) - 1,
                               int((len(normalized_regrets) - 1) * 0.9)))
        costs_after_sufficiency = [
            trace.action_cost for task in tasks for trace in task.traces
            if trace.answerable_before and trace.execution_status == "EXECUTED"
            and trace.action_cost is not None
        ]
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
            "mean_redundant_actions_after_sufficiency": (
                avoidable_after_sufficient / reached_sufficient if reached_sufficient else 0.0),
            "max_redundant_actions_after_sufficiency": max((sum(
                trace.answerable_before and trace.executed_action not in {
                    DecisionAction.ANSWER, DecisionAction.DEFER, DecisionAction.STOP}
                for trace in task.traces) for task in tasks), default=0),
            "mean_cost_after_sufficiency": (
                sum(costs_after_sufficiency) / len(costs_after_sufficiency)
                if costs_after_sufficiency else 0.0),
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
            "mean_normalized_executive_regret": (
                sum(normalized_regrets) / len(normalized_regrets) if normalized_regrets else 0.0),
            "median_normalized_executive_regret": median(normalized_regrets) if normalized_regrets else 0.0,
            "p90_normalized_executive_regret": normalized_regrets[p90_index] if normalized_regrets else 0.0,
            "zero_regret_task_rate": (
                sum(regret <= 1e-9 for regret in normalized_regrets) / len(normalized_regrets)
                if normalized_regrets else 0.0),
            "worst_case_regret": max((task.trajectory_regret for task in tasks), default=0.0),
            "answer_accuracy": (
                sum(trace.task_success is True for trace in answers) / len(answers) if answers else 0.0),
            "policy_violation_rate": policy_violations / len(executed) if executed else 0.0,
            "executive_steps": sum(task.resources["executive_steps_used"] for task in tasks),
            "total_action_cost": sum(
                trace.action_cost or 0.0 for trace in executed),
        }
