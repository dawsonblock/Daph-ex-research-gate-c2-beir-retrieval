"""Runtime evaluation and receipts for V2B-I3.2 sequential information states."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from statistics import median
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .metareasoning_benchmark import I3BenchmarkTask, MetareasoningBenchmark
from .metareasoning_controller import MatchedMetareasoningController, ObservationMask
from .metareasoning_executor import I3Runtime, initial_i3_runtime, runtime_state_hash
from .metareasoning_sequential_oracle import (
    InformationHistoryEvent, LatentMember, SequentialObservableOracleSet,
    SequentialObservablePolicyTable, _apply_proposal, canonical_packet, controller_observation)
from .metareasoning_transition_table import OraclePolicyTable
from .metareasoning_utility import MetareasoningUtility
from .policy import FrozenPolicy
from .resources import ResourceState


TRAJECTORY_RECEIPT_SCHEMA = "DAPH_V2B_I3_2_TRAJECTORY_RECEIPT_V1"
DEVELOPMENT_RECEIPT_SCHEMA = "DAPH_V2B_I3_2_DEVELOPMENT_RECEIPT_V1"


@dataclass(frozen=True)
class I3_2Trace:
    step_id: int
    information_state_before: str
    observation_hash: str
    proposed_action: DecisionAction
    policy_effect: str
    policy_resolved_action: DecisionAction | None
    policy_reason_class: str
    execution_status: str
    state_before_hash: str
    state_after_hash: str | None
    executed_action: DecisionAction | None
    immediate_utility: float
    information_gain_bits: float
    action_regret: float


@dataclass(frozen=True)
class I3_2TaskRun:
    task_id: str
    condition: str
    initial_information_state_id: str
    traces: tuple[I3_2Trace, ...]
    terminal_result: str
    task_success: bool
    realized_utility: float
    final_state_hash: str | None
    resources: Mapping[str, int]


def _member_for(runtime: I3Runtime, latent_table: OraclePolicyTable) -> LatentMember:
    return LatentMember(runtime.task.task_id, latent_table.identity_sha256,
                        latent_table.state_id_for(runtime), 1)


def _outcome_for_member(table: SequentialObservablePolicyTable, information_state_id: str,
                        action: DecisionAction, member: LatentMember):
    transition = table.transitions[(information_state_id, action)]
    return next(item for item in transition.outcomes if member.key in item.member_keys)


def run_task_with_runtime(*, initial_runtime: I3Runtime, condition: str,
                          controller: MatchedMetareasoningController, mask: ObservationMask,
                          policy: FrozenPolicy, utility: MetareasoningUtility,
                          latent_table: OraclePolicyTable,
                          oracle_set: SequentialObservableOracleSet,
                          max_steps: int = 24) -> I3_2TaskRun:
    runtime = initial_runtime; history: tuple[InformationHistoryEvent, ...] = ()
    member = _member_for(runtime, latent_table)
    table = oracle_set.table_for_member(member)
    information_id = table.initial_information_state_id
    traces: list[I3_2Trace] = []; realized = 0.0
    for step in range(max_steps):
        observation = controller_observation(runtime=runtime, history=history, mask=mask)
        if not observation.allowed_actions:
            return I3_2TaskRun(runtime.task.task_id, condition, table.initial_information_state_id,
                               tuple(traces), "RESOURCE_EXHAUSTED", False, realized,
                               runtime_state_hash(runtime), runtime.resources.as_dict())
        proposal = controller.choose(observation)
        transition = table.transitions.get((information_id, proposal.action))
        if transition is None:
            raise RuntimeError(
                f"runtime proposed action absent from sequential oracle: {runtime.task.task_id} "
                f"{information_id} {proposal.action.value}")
        outcome = _apply_proposal(runtime=runtime, proposed=proposal.action, policy=policy, utility=utility)
        oracle_member = table.member_transitions[(information_id, proposal.action, member.key)]
        if outcome.history_event != oracle_member.history_event:
            raise RuntimeError("runtime/oracle policy-feedback parity failure")
        selected = outcome.history_event.resolved_action
        next_hash = None if outcome.runtime is None else runtime_state_hash(outcome.runtime)
        if next_hash != oracle_member.next_runtime_state_hash:
            raise RuntimeError(
                f"runtime/oracle state-transition parity failure: {runtime.task.task_id} "
                f"{proposal.action.value} {next_hash} != {oracle_member.next_runtime_state_hash}")
        outcome_class = _outcome_for_member(table, information_id, proposal.action, member)
        traces.append(I3_2Trace(
            step_id=step, information_state_before=information_id,
            observation_hash=table.information_states[information_id].observation_hash,
            proposed_action=proposal.action, policy_effect=outcome.feedback.effect,
            policy_resolved_action=selected, policy_reason_class=outcome.feedback.reason_class,
            execution_status=outcome.history_event.execution_status,
            state_before_hash=runtime_state_hash(runtime), state_after_hash=next_hash,
            executed_action=selected if outcome.history_event.execution_status == "EXECUTED" else None,
            immediate_utility=outcome.immediate_utility,
            information_gain_bits=transition.expected_information_gain_bits,
            action_regret=table.action_regret(information_id, proposal.action)))
        realized += outcome.immediate_utility
        if outcome.terminal:
            assert outcome.terminal_utility is not None
            realized += outcome.terminal_utility
            return I3_2TaskRun(runtime.task.task_id, condition, table.initial_information_state_id,
                               tuple(traces), outcome.history_event.execution_status,
                               bool(outcome.task_success), realized, next_hash,
                               runtime.resources.as_dict() if outcome.runtime is None
                               else outcome.runtime.resources.as_dict())
        assert outcome.runtime is not None and outcome_class.next_information_state_id is not None
        runtime = outcome.runtime; history = tuple(table.information_states[
            outcome_class.next_information_state_id].history)
        assert oracle_member.next_member is not None
        member = oracle_member.next_member; information_id = outcome_class.next_information_state_id
    return I3_2TaskRun(runtime.task.task_id, condition, table.initial_information_state_id,
                       tuple(traces), "CONTROLLER_STEP_LIMIT", False, realized,
                       runtime_state_hash(runtime), runtime.resources.as_dict())


def run_condition(*, benchmark: MetareasoningBenchmark, condition: str,
                  controller: MatchedMetareasoningController, mask: ObservationMask,
                  policy: FrozenPolicy, utility: MetareasoningUtility,
                  latent_tables: Mapping[str, OraclePolicyTable],
                  oracle_set: SequentialObservableOracleSet) -> tuple[I3_2TaskRun, ...]:
    return tuple(run_task_with_runtime(
        initial_runtime=initial_i3_runtime(task, ResourceState(benchmark.budget_for(task))),
        condition=condition, controller=controller, mask=mask, policy=policy, utility=utility,
        latent_table=latent_tables[task.task_id], oracle_set=oracle_set)
        for task in benchmark.tasks)


def class_decomposition(*, runs: tuple[I3_2TaskRun, ...],
                        oracle_set: SequentialObservableOracleSet,
                        latent_tables: Mapping[str, OraclePolicyTable]) -> dict[str, dict[str, float | int]]:
    by_task = {item.task_id: item for item in runs}; result = {}
    for table_id, table in oracle_set.tables.items():
        root = table.information_states[table.initial_information_state_id]
        actual = sum(float(member.posterior_weight) * by_task[member.task_id].realized_utility
                     for member in root.members)
        latent = table.expected_latent_values[table.initial_information_state_id]
        observable = table.belief_values[table.initial_information_state_id]
        information_gap = latent - observable; decision_gap = observable - actual
        total = latent - actual
        if information_gap < -1e-9 or decision_gap < -1e-9 or abs(total - information_gap - decision_gap) > 1e-8:
            raise RuntimeError("I3.2 regret-decomposition invariant failed")
        result[table_id] = {"member_count": len(root.members), "latent_upper_bound": latent,
                            "observable_upper_bound": observable, "controller_value": actual,
                            "information_gap": information_gap, "decision_gap": decision_gap,
                            "total_regret": total,
                            "belief_entropy_bits": root.entropy_bits}
    return result


def aggregate_metrics(*, runs: tuple[I3_2TaskRun, ...],
                      decomposition: Mapping[str, Mapping[str, float | int]],
                      oracle_set: SequentialObservableOracleSet,
                      benchmark: MetareasoningBenchmark, mask: ObservationMask) -> dict[str, float | int]:
    groups = tuple(decomposition.values())
    def avg(name: str) -> float:
        return sum(float(item[name]) for item in groups) / len(groups)
    all_traces = tuple(trace for run in runs for trace in run.traces)
    successful = sum(run.task_success for run in runs)
    by_task = {task.task_id: task for task in benchmark.tasks}
    packets = [canonical_packet(controller_observation(
        runtime=initial_i3_runtime(by_task[run.task_id], ResourceState(benchmark.budget_for(by_task[run.task_id]))),
        history=(), mask=mask)) for run in runs]
    observation_sizes = [len(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()) for packet in packets]
    cognitive_fact_counts = [0 if packet["cognitive_state"] is None else sum(
        len(value) if isinstance(value, list) else 1
        for value in packet["cognitive_state"].values()) for packet in packets]  # type: ignore[index,union-attr]
    return {
        "task_count": len(runs), "task_success_rate": successful / len(runs),
        "mean_information_gap": avg("information_gap"), "mean_decision_gap": avg("decision_gap"),
        "mean_total_regret": avg("total_regret"),
        "mean_latent_upper_bound": avg("latent_upper_bound"),
        "mean_observable_upper_bound": avg("observable_upper_bound"),
        "mean_controller_value": avg("controller_value"),
        "mean_belief_cardinality": sum(int(item["member_count"]) for item in groups) / len(groups),
        "max_belief_cardinality": max(int(item["member_count"]) for item in groups),
        "policy_probe_count": sum(trace.policy_effect != "ALLOW" for trace in all_traces),
        "policy_probe_information_gain_bits": sum(trace.information_gain_bits for trace in all_traces
                                                     if trace.policy_effect != "ALLOW"),
        "rejected_proposal_cost": -sum(trace.immediate_utility for trace in all_traces
                                         if trace.execution_status in {"POLICY_REJECTED", "RESOURCE_REJECTED"}),
        "mean_action_information_gain_bits": (sum(trace.information_gain_bits for trace in all_traces)
                                               / max(1, len(all_traces))),
        "total_action_utility_cost": -sum(trace.immediate_utility for trace in all_traces),
        "mean_observation_bytes": sum(observation_sizes) / len(observation_sizes),
        "mean_cognitive_fact_count": sum(cognitive_fact_counts) / len(cognitive_fact_counts),
    }


def trajectory_payload(*, run: I3_2TaskRun, table: SequentialObservablePolicyTable,
                       condition: str, policy_sha256: str, utility_sha256: str,
                       controller_revision: str) -> dict[str, object]:
    return {"schema": TRAJECTORY_RECEIPT_SCHEMA, "task_id": run.task_id, "condition": condition,
            "initial_information_state_id": run.initial_information_state_id,
            "sequential_observable_table_sha256": table.table_sha256,
            "observation_mask_sha256": table.observation_mask_hash,
            "policy_sha256": policy_sha256, "utility_sha256": utility_sha256,
            "controller_revision": controller_revision,
            "steps": [asdict(item) for item in run.traces], "terminal_result": run.terminal_result,
            "task_success": run.task_success, "trajectory_utility": run.realized_utility,
            "final_state_hash": run.final_state_hash, "resources": dict(run.resources)}


def replay_trajectory(*, benchmark: MetareasoningBenchmark, task_id: str,
                      traces: list[Mapping[str, object]], policy: FrozenPolicy,
                      utility: MetareasoningUtility) -> dict[str, object]:
    """Replay a receipt without a controller and require exact state/cost parity."""
    task = next(item for item in benchmark.tasks if item.task_id == task_id)
    runtime = initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
    utility_value = 0.0
    for trace in traces:
        if trace["state_before_hash"] != runtime_state_hash(runtime):
            raise RuntimeError("I3.2 replay pre-state mismatch")
        raw = trace["proposed_action"]
        action = raw if isinstance(raw, DecisionAction) else DecisionAction(str(raw).removeprefix("DecisionAction."))
        outcome = _apply_proposal(runtime=runtime, proposed=action, policy=policy, utility=utility)
        if trace["execution_status"] != outcome.history_event.execution_status:
            raise RuntimeError("I3.2 replay execution-status mismatch")
        after = None if outcome.runtime is None else runtime_state_hash(outcome.runtime)
        if trace["state_after_hash"] != after:
            raise RuntimeError("I3.2 replay post-state mismatch")
        if float(trace["immediate_utility"]) != outcome.immediate_utility:
            raise RuntimeError("I3.2 replay action-cost mismatch")
        utility_value += outcome.immediate_utility
        if outcome.terminal:
            assert outcome.terminal_utility is not None
            utility_value += outcome.terminal_utility
            runtime = outcome.runtime or runtime
            break
        assert outcome.runtime is not None
        runtime = outcome.runtime
    return {"state_hash": runtime_state_hash(runtime), "resources": runtime.resources.as_dict(),
            "trajectory_utility": utility_value}
