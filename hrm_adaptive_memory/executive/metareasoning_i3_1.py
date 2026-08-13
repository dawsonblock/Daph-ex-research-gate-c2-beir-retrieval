"""I3.1 regret decomposition and deterministic trajectory replay helpers."""
from __future__ import annotations

from dataclasses import asdict
from statistics import median
from typing import Mapping

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .metareasoning_benchmark import MetareasoningBenchmark
from .metareasoning_executor import DeterministicMetareasoningExecutor, initial_i3_runtime, runtime_state_hash
from .metareasoning_loop import I3TaskRun
from .metareasoning_observable_oracle import ObservableOraclePolicyTable
from .metareasoning_transition_table import OraclePolicyTable
from .metareasoning_utility import MetareasoningUtility
from .resources import ResourceState


TRAJECTORY_RECEIPT_SCHEMA = "DAPH_V2B_I3_1_TRAJECTORY_RECEIPT_V1"
AGGREGATE_RECEIPT_SCHEMA = "DAPH_V2B_I3_1_DEVELOPMENT_RECEIPT_V1"


def _quantile(values: list[float], numerator: int, denominator: int) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, (len(values) * numerator) // denominator)]


def regret_decomposition(*, run: I3TaskRun, table: OraclePolicyTable,
                         observable: ObservableOraclePolicyTable) -> dict[str, float | str]:
    item = observable.observation_for(table)
    latent = table.initial_value
    information_gap = latent - item.value
    decision_gap = item.value - run.realized_utility
    total_regret = latent - run.realized_utility
    epsilon = 1e-9
    return {
        "latent_oracle_value": latent,
        "observable_oracle_value": item.value,
        "controller_utility": run.realized_utility,
        "information_gap": information_gap,
        "decision_gap": decision_gap,
        "total_regret": total_regret,
        "normalized_information_gap": information_gap / (abs(latent) + epsilon),
        "normalized_decision_gap": decision_gap / (abs(item.value) + epsilon),
        "normalized_total_regret": total_regret / (abs(latent) + epsilon),
        "observation_class_sha256": item.observation_hash,
    }


def aggregate_metrics(*, run_tasks: tuple[I3TaskRun, ...], tables: Mapping[str, OraclePolicyTable],
                      observable: ObservableOraclePolicyTable) -> dict[str, float | int]:
    decompositions = [regret_decomposition(run=run, table=tables[run.task_id], observable=observable)
                      for run in run_tasks]
    values = lambda name: [float(item[name]) for item in decompositions]
    latent = values("latent_oracle_value")
    info = values("information_gap")
    decision = values("decision_gap")
    total = values("total_regret")
    return {
        "mean_latent_oracle_value": sum(latent) / len(latent),
        "mean_information_gap": sum(info) / len(info),
        "median_information_gap": median(info),
        "p90_information_gap": _quantile(info, 9, 10),
        "mean_decision_gap": sum(decision) / len(decision),
        "median_decision_gap": median(decision),
        "p90_decision_gap": _quantile(decision, 9, 10),
        "mean_total_regret": sum(total) / len(total),
        "mean_information_efficiency": 1.0 - sum(
            float(item["normalized_information_gap"]) for item in decompositions) / len(decompositions),
        "zero_decision_gap_task_rate": sum(abs(value) <= 1e-12 for value in decision) / len(decision),
        "observable_oracle_ambiguity_count": observable.ambiguity_count,
        "observable_oracle_class_count": len(observable.classes),
    }


def trajectory_payload(*, run: I3TaskRun, table: OraclePolicyTable,
                       observable: ObservableOraclePolicyTable, condition: str,
                       observation_mask_sha256: str, controller_revision: str,
                       policy_sha256: str, utility_sha256: str, budget_sha256: str) -> dict[str, object]:
    decomposition = regret_decomposition(run=run, table=table, observable=observable)
    return {
        "schema": TRAJECTORY_RECEIPT_SCHEMA,
        "task_id": run.task_id,
        "condition": condition,
        "initial_state_id": table.initial_state_id,
        "observation_mask_sha256": observation_mask_sha256,
        "latent_oracle_table_sha256": table.table_sha256,
        "observable_oracle_table_sha256": observable.table_sha256,
        "controller_revision": controller_revision,
        "policy_sha256": policy_sha256,
        "utility_sha256": utility_sha256,
        "budget_sha256": budget_sha256,
        "steps": [asdict(item) for item in run.traces],
        "terminal_result": run.terminal_result,
        "resources": dict(run.resources),
        "trajectory_utility": run.realized_utility,
        "decomposition": decomposition,
    }


def replay_trajectory(*, benchmark: MetareasoningBenchmark, task_id: str,
                      traces: list[Mapping[str, object]], utility: MetareasoningUtility) -> dict[str, object]:
    """Replay executed trace actions without a controller and verify state/cost parity."""
    task = next(task for task in benchmark.tasks if task.task_id == task_id)
    runtime = initial_i3_runtime(task, ResourceState(benchmark.budget_for(task)))
    executor = DeterministicMetareasoningExecutor()
    utility_value = 0.0
    for trace in traces:
        if trace["execution_status"] != "EXECUTED":
            continue
        if trace["pre_state_hash"] != runtime_state_hash(runtime):
            raise RuntimeError("trajectory replay pre-state mismatch")
        raw_action = trace["executed_action"]
        if isinstance(raw_action, DecisionAction):
            action = raw_action
        else:
            value = str(raw_action).removeprefix("DecisionAction.")
            action = DecisionAction(value)
        execution = executor.execute(runtime, action)
        if trace["post_state_hash"] != runtime_state_hash(execution.runtime):
            raise RuntimeError("trajectory replay post-state mismatch")
        utility_value += utility.action_utility(runtime.resources, execution.runtime.resources)
        if execution.terminal:
            assert execution.task_success is not None
            utility_value += utility.terminal_reward(action, execution.task_success)
        runtime = execution.runtime
    return {"state_hash": runtime_state_hash(runtime), "resources": runtime.resources.as_dict(),
            "trajectory_utility": utility_value}
