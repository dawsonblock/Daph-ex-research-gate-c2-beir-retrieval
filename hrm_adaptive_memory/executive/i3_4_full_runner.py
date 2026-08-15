"""Full I3.4.1 experiment runner with multi-step trajectories.

Integrates:
- Frozen I3.3.2 benchmark (per-split task loading)
- ObservationMask for blind/aware condition construction
- PinnedModelController with DeepSeek backend (multi-step trajectories)
- DeterministicActionExecutor for action effects
- Observable oracle views for per-split V_O^M
- Scientific scoring (IG/DG/TR decomposition)
- Paired deltas and statistical analysis
- Append-only receipt and result persistence

Schema identity: ``DAPH_V2B_I3_4_FULL_RUNNER_V1`` (frozen).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    CognitiveStateSnapshot, DecisionSummary, TemporalStatus, VerificationState)

from .actions import ActionProposal
from .executor import (
    ActionExecution, DeterministicActionExecutor, TaskRuntime,
    build_cognitive_state, initial_runtime)
from .metareasoning_benchmark import I3BenchmarkTask, MetareasoningBenchmark
from .metareasoning_controller import (
    ControllerObservation, ObservationMask,
    STATE_BLIND_MASK, STATE_AWARE_MASK,
    apply_observation_mask)
from .model_backend import DeepSeekBackend, ModelBackend, ModelCallResult
from .model_decoder import decode_output
from .model_packet import (
    assert_no_condition_leakage, packet_json, packet_sha256, serialize_packet)
from .model_prompt import SYSTEM_PROMPT
from .pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, CONTROLLER_ID, FAIL_CLOSED_PROPOSAL)
from .resources import ResourceBudget, ResourceState, ResourceExhausted
from .i3_4_scientific_scoring import (
    I34ScientificTaskContribution, I34PairedDelta,
    compute_task_contribution, compute_aggregate,
    compute_paired_deltas, mean_delta_dg, verify_all_identities)
from .i3_4_statistical_analysis import (
    paired_bootstrap, topology_cluster_bootstrap, BootstrapResult)
from .i3_4_pair_scheduler import (
    compute_pair_hash, is_blind_first, check_pair_fingerprints)
from .i3_4_model_identity_policy import FROZEN_IDENTITY_POLICY
from .i3_4_generation_config import FROZEN_CONFIG

RUNNER_SCHEMA = "DAPH_V2B_I3_4_FULL_RUNNER_V1"
RUNNER_VERSION = 1
MAX_STEPS = 24


@dataclass(frozen=True)
class TrajectoryStep:
    """One step in a multi-step trajectory."""
    step_id: int
    proposed_action: str
    reason_code: str
    executed_action: str
    outcome_code: str
    task_success: bool | None
    terminal: bool


@dataclass(frozen=True)
class ConditionTrajectory:
    """Full trajectory for one task under one condition."""
    task_id: str
    condition: str  # "BLIND" or "AWARE"
    steps: tuple[TrajectoryStep, ...]
    terminal_result: str
    task_success: bool
    realized_utility: float
    resources: Mapping[str, int]
    model_calls: int
    decoder_failures: int
    backend_errors: int
    system_fingerprint: str | None
    model_name: str | None


@dataclass(frozen=True)
class PairedTrajectoryResult:
    """Paired blind/aware trajectory result for one task."""
    task_id: str
    pair_id: str
    blind: ConditionTrajectory
    aware: ConditionTrajectory
    fingerprint_match: bool
    pair_valid: bool


@dataclass
class FullExperimentRunner:
    """Runs the complete I3.4.1 experiment with multi-step trajectories.

    For each task:
    1. The pair scheduler determines call order (BLIND→AWARE or AWARE→BLIND).
    2. A full multi-step trajectory is run for the first condition.
    3. A full multi-step trajectory is run for the second condition.
    4. Each step makes one DeepSeek API call with receipt.
    5. Fingerprints are checked within the pair.
    """

    backend: DeepSeekBackend
    executor: DeterministicActionExecutor = field(default_factory=DeterministicActionExecutor)
    experiment_id: str = "v2b_i3_4_experiment_v1"
    max_steps: int = MAX_STEPS
    strict_json: bool = True
    temperature: float = FROZEN_CONFIG.temperature
    max_tokens: int = FROZEN_CONFIG.max_tokens
    results: list[PairedTrajectoryResult] = field(default_factory=list, repr=False)

    def _make_controller_observation(
        self,
        runtime: TaskRuntime,
        task: I3BenchmarkTask,
        mask: ObservationMask,
        prior_decisions: tuple[DecisionSummary, ...],
        prior_outcomes: tuple[str, ...],
    ) -> ControllerObservation:
        """Build a ControllerObservation with the mask applied."""
        snapshot = build_cognitive_state(
            runtime, prior_decisions=prior_decisions, prior_outcomes=prior_outcomes)
        masked_state = apply_observation_mask(snapshot, mask)
        resources = runtime.resources.as_dict()
        allowed = tuple(action for action in V2B_ACTIONS if runtime.resources.can_execute(action))
        executed = tuple(
            DecisionSummary(d.decision_id, d.selected_action, d.reason_code, d.outcome)
            for d in prior_decisions)
        return ControllerObservation(
            task_id=task.controller_instance_id or task.task_id,
            task_summary=task.task_summary,
            resource_state=resources,
            allowed_actions=allowed,
            executed_actions=tuple(
                DecisionAction(d.selected_action) if isinstance(d.selected_action, str)
                else d.selected_action for d in prior_decisions),
            rejected_actions=(),
            cognitive_state=masked_state,
        )

    def _run_trajectory(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
        mask: ObservationMask,
        condition: str,
    ) -> ConditionTrajectory:
        """Run a full multi-step trajectory for one task under one condition."""
        resources = ResourceState(budget)
        runtime = initial_runtime(
            _I3TaskAdapter(task), resources)
        steps: list[TrajectoryStep] = []
        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []
        realized = 0.0
        model_calls = 0
        decoder_failures = 0
        backend_errors = 0
        last_fingerprint: str | None = None
        last_model: str | None = None

        for step_id in range(self.max_steps):
            observation = self._make_controller_observation(
                runtime, task, mask,
                tuple(prior_decisions), tuple(prior_outcomes))

            # Set backend metadata for receipts
            self.backend.task_id = task.task_id
            self.backend.condition = condition
            self.backend.pair_id = f"{self.experiment_id}:{task.task_id}"

            # Serialize and call the model
            packet = serialize_packet(observation)
            assert_no_condition_leakage(packet)
            user_prompt = packet_json(packet)

            model_calls += 1
            try:
                call_result = self.backend.generate(
                    system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                    temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception:
                backend_errors += 1
                # Fail closed: DEFER on backend error
                proposal = BACKEND_ERROR_PROPOSAL
                call_result = None
            else:
                last_fingerprint = call_result.system_fingerprint
                last_model = call_result.model_name
                outcome = decode_output(call_result.raw_output, strict=self.strict_json)
                if outcome.valid and outcome.proposal:
                    proposal = outcome.proposal
                else:
                    decoder_failures += 1
                    proposal = FAIL_CLOSED_PROPOSAL

            # Execute the action through the deterministic executor
            action = proposal.action
            try:
                execution = self.executor.execute(runtime, action)
            except ResourceExhausted:
                execution = ActionExecution(
                    DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")

            steps.append(TrajectoryStep(
                step_id=step_id,
                proposed_action=proposal.action.value,
                reason_code=proposal.reason_code,
                executed_action=execution.action.value,
                outcome_code=execution.outcome_code,
                task_success=execution.task_success,
                terminal=execution.terminal))

            prior_decisions.append(DecisionSummary(
                f"{task.task_id}:step:{step_id}", action.value,
                proposal.reason_code, execution.outcome_code))
            prior_outcomes.append(execution.outcome_code)

            runtime = execution.runtime
            if execution.terminal:
                return ConditionTrajectory(
                    task_id=task.task_id, condition=condition,
                    steps=tuple(steps), terminal_result=execution.outcome_code,
                    task_success=bool(execution.task_success),
                    realized_utility=realized,
                    resources=runtime.resources.as_dict(),
                    model_calls=model_calls, decoder_failures=decoder_failures,
                    backend_errors=backend_errors,
                    system_fingerprint=last_fingerprint,
                    model_name=last_model)

        # Step limit reached
        return ConditionTrajectory(
            task_id=task.task_id, condition=condition,
            steps=tuple(steps), terminal_result="STEP_LIMIT",
            task_success=False, realized_utility=realized,
            resources=runtime.resources.as_dict(),
            model_calls=model_calls, decoder_failures=decoder_failures,
            backend_errors=backend_errors,
            system_fingerprint=last_fingerprint,
            model_name=last_model)

    def run_pair(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
    ) -> PairedTrajectoryResult:
        """Run one counterbalanced pair (blind + aware) for one task."""
        pair_id = f"{self.experiment_id}:{task.task_id}"
        blind_first = is_blind_first(self.experiment_id, task.task_id)

        if blind_first:
            blind = self._run_trajectory(task, budget, STATE_BLIND_MASK, "BLIND")
            aware = self._run_trajectory(task, budget, STATE_AWARE_MASK, "AWARE")
        else:
            aware = self._run_trajectory(task, budget, STATE_AWARE_MASK, "AWARE")
            blind = self._run_trajectory(task, budget, STATE_BLIND_MASK, "BLIND")

        # Check fingerprints within the pair
        fp_record = check_pair_fingerprints(
            pair_id=pair_id,
            first_call_fingerprint=blind.system_fingerprint,
            second_call_fingerprint=aware.system_fingerprint,
            require_fingerprint=FROZEN_IDENTITY_POLICY.require_fingerprint,
        )

        result = PairedTrajectoryResult(
            task_id=task.task_id, pair_id=pair_id,
            blind=blind, aware=aware,
            fingerprint_match=fp_record.fingerprint_match,
            pair_valid=fp_record.pair_valid)
        self.results.append(result)
        return result

    def run_split(
        self,
        benchmark: MetareasoningBenchmark,
        split: str,
        *,
        max_tasks: int | None = None,
        progress_every: int = 10,
    ) -> list[PairedTrajectoryResult]:
        """Run all tasks in a split."""
        split_benchmark = benchmark.for_split(split)
        tasks = split_benchmark.tasks
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        results = []
        for i, task in enumerate(tasks):
            result = self.run_pair(task, split_benchmark.budget_for(task))
            results.append(result)
            if (i + 1) % progress_every == 0:
                print(f"  [{i+1}/{len(tasks)}] {task.task_id}: "
                      f"blind_success={result.blind.task_success}, "
                      f"aware_success={result.aware.task_success}")
        return results

    def all_receipts(self) -> list:
        return self.backend.call_receipts

    def runner_summary(self) -> dict[str, Any]:
        return {
            "schema": RUNNER_SCHEMA,
            "schema_version": RUNNER_VERSION,
            "experiment_id": self.experiment_id,
            "controller_id": CONTROLLER_ID,
            "pairs_completed": len(self.results),
            "total_receipts": len(self.backend.call_receipts),
            "total_model_calls": sum(
                r.blind.model_calls + r.aware.model_calls for r in self.results),
            "total_decoder_failures": sum(
                r.blind.decoder_failures + r.aware.decoder_failures for r in self.results),
            "total_backend_errors": sum(
                r.blind.backend_errors + r.aware.backend_errors for r in self.results),
            "strict_json": self.strict_json,
        }


class _I3TaskAdapter:
    """Adapter to make I3BenchmarkTask compatible with the I2 executor."""

    def __init__(self, task: I3BenchmarkTask):
        self.task_id = task.task_id
        self.category = task.category
        self.task_summary = task.task_summary
        self.high_stakes = task.high_stakes
        self.initial_verification_state = task.latent.verification_state
        self.initial_temporal_status = task.latent.temporal_status
        self.unresolved_conflict = task.latent.unresolved_conflict
        self.reasoning_required = not task.latent.composition_complete
        self.expected_terminal = task.latent.expected_terminal
        self.action_effects = task.action_effects


def save_receipts(receipts: list, path: str | Path) -> str:
    """Save receipts to an append-only JSONL file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in receipts:
            f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_results(
    results: list[PairedTrajectoryResult],
    path: str | Path,
) -> str:
    """Save paired results to a JSON file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RUNNER_SCHEMA,
        "schema_version": RUNNER_VERSION,
        "results": [
            {
                "task_id": r.task_id,
                "pair_id": r.pair_id,
                "fingerprint_match": r.fingerprint_match,
                "pair_valid": r.pair_valid,
                "blind": {
                    "condition": r.blind.condition,
                    "terminal_result": r.blind.terminal_result,
                    "task_success": r.blind.task_success,
                    "realized_utility": r.blind.realized_utility,
                    "model_calls": r.blind.model_calls,
                    "decoder_failures": r.blind.decoder_failures,
                    "backend_errors": r.blind.backend_errors,
                    "system_fingerprint": r.blind.system_fingerprint,
                    "model_name": r.blind.model_name,
                    "resources": dict(r.blind.resources),
                    "steps": [
                        {"step_id": s.step_id, "proposed_action": s.proposed_action,
                         "reason_code": s.reason_code, "executed_action": s.executed_action,
                         "outcome_code": s.outcome_code, "task_success": s.task_success,
                         "terminal": s.terminal}
                        for s in r.blind.steps
                    ],
                },
                "aware": {
                    "condition": r.aware.condition,
                    "terminal_result": r.aware.terminal_result,
                    "task_success": r.aware.task_success,
                    "realized_utility": r.aware.realized_utility,
                    "model_calls": r.aware.model_calls,
                    "decoder_failures": r.aware.decoder_failures,
                    "backend_errors": r.aware.backend_errors,
                    "system_fingerprint": r.aware.system_fingerprint,
                    "model_name": r.aware.model_name,
                    "resources": dict(r.aware.resources),
                    "steps": [
                        {"step_id": s.step_id, "proposed_action": s.proposed_action,
                         "reason_code": s.reason_code, "executed_action": s.executed_action,
                         "outcome_code": s.outcome_code, "task_success": s.task_success,
                         "terminal": s.terminal}
                        for s in r.aware.steps
                    ],
                },
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_results(
    results: list[PairedTrajectoryResult],
    benchmark: MetareasoningBenchmark,
    oracle_views_path: str | Path,
    latent_oracle_path: str | Path,
    utility_weights: Mapping[str, float],
) -> tuple[list[I34ScientificTaskContribution], list[I34PairedDelta]]:
    """Score paired results using the observable oracle views and latent oracle.

    Returns (all_contributions, paired_deltas).
    """
    # Load observable oracle views
    views_data = json.loads(Path(oracle_views_path).read_text())
    views_by_split_cond: dict[tuple[str, str], float] = {}
    for v in views_data["views"]:
        views_by_split_cond[(v["split_name"], v["condition"])] = v["observable_optimal_value"]

    # Load latent oracle values (per-task V_L^*)
    latent_values: dict[str, float] = {}
    with gzip.open(latent_oracle_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry.get("task_id", "")
            table = entry.get("table", entry)
            state_values = table.get("state_values", {})
            init_id = entry.get("initial_state_id") or table.get("initial_state_id")
            if init_id and init_id in state_values:
                latent_values[task_id] = float(state_values[init_id])

    # Build task lookup
    task_by_id = {t.task_id: t for t in benchmark.tasks}

    # Compute controller values from trajectory utility
    # V_π^M(s) = realized_utility (from the deterministic executor)
    blind_contributions: dict[str, I34ScientificTaskContribution] = {}
    aware_contributions: dict[str, I34ScientificTaskContribution] = {}
    all_contributions: list[I34ScientificTaskContribution] = []

    for result in results:
        task = task_by_id.get(result.task_id)
        if task is None:
            continue
        split = task.split

        # Get observable values for this split
        v_o_blind = views_by_split_cond.get((split, "STATE_BLIND_CONTROLLER"), 0.0)
        v_o_aware = views_by_split_cond.get((split, "STATE_AWARE_CONTROLLER"), 0.0)

        # Get latent value
        v_l = latent_values.get(result.task_id, 0.0)

        # Controller values (realized utility from executor)
        v_pi_blind = result.blind.realized_utility
        v_pi_aware = result.aware.realized_utility

        # Compute contributions
        blind_contrib = compute_task_contribution(
            task_id=result.task_id, condition="STATE_BLIND_CONTROLLER",
            latent_optimal_value=v_l, observable_optimal_value=v_o_blind,
            controller_value=v_pi_blind,
            information_class_hash="",
            observable_oracle_set_sha256="",
            latent_oracle_table_sha256="")
        aware_contrib = compute_task_contribution(
            task_id=result.task_id, condition="STATE_AWARE_CONTROLLER",
            latent_optimal_value=v_l, observable_optimal_value=v_o_aware,
            controller_value=v_pi_aware,
            information_class_hash="",
            observable_oracle_set_sha256="",
            latent_oracle_table_sha256="")

        blind_contributions[result.task_id] = blind_contrib
        aware_contributions[result.task_id] = aware_contrib
        all_contributions.extend([blind_contrib, aware_contrib])

    # Compute paired deltas
    deltas = compute_paired_deltas(
        blind_contributions=blind_contributions,
        aware_contributions=aware_contributions)

    return all_contributions, deltas


def run_statistical_analysis(
    deltas: list[I34PairedDelta],
    cluster_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the full statistical analysis on paired deltas."""
    if not deltas:
        return {"error": "no deltas to analyze"}

    # Task-level paired bootstrap
    task_result = paired_bootstrap(deltas)

    # Topology-cluster bootstrap (if cluster map provided)
    topo_result_uniform = None
    topo_result_population = None
    if cluster_map:
        topo_result_uniform = topology_cluster_bootstrap(
            deltas, cluster_map=cluster_map, estimand="topology_uniform")
        topo_result_population = topology_cluster_bootstrap(
            deltas, cluster_map=cluster_map, estimand="task_population")

    # Mean deltas
    mean_ddg = mean_delta_dg(deltas)

    return {
        "n_paired_tasks": len(deltas),
        "mean_delta_dg": mean_ddg,
        "task_level_bootstrap": {
            "point_estimate": task_result.point_estimate,
            "ci_lower": task_result.lower_bound,
            "ci_upper": task_result.upper_bound,
            "ci_level": task_result.ci_level,
            "iterations": task_result.iterations,
        },
        "topology_uniform_bootstrap": {
            "point_estimate": topo_result_uniform.point_estimate if topo_result_uniform else None,
            "ci_lower": topo_result_uniform.lower_bound if topo_result_uniform else None,
            "ci_upper": topo_result_uniform.upper_bound if topo_result_uniform else None,
        } if topo_result_uniform else None,
        "topology_population_bootstrap": {
            "point_estimate": topo_result_population.point_estimate if topo_result_population else None,
            "ci_lower": topo_result_population.lower_bound if topo_result_population else None,
            "ci_upper": topo_result_population.upper_bound if topo_result_population else None,
        } if topo_result_population else None,
    }
