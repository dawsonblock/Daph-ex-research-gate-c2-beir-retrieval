"""Full I3.5 experiment runner with governor-enhanced multi-step trajectories.

Extends the I3.4 runner with the General Governor layer:
- Governor assesses each step and builds a decision frame
- The frame is injected into the model packet
- DeepSeek chooses from the frame (governor recommends, model decides)
- Governor top action and model action are both recorded for diagnostics

Schema identity: ``DAPH_V2B_I3_5_FULL_RUNNER_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
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
from .metareasoning_utility import MetareasoningUtility

from .i3_5_model_packet import (
    serialize_governor_packet, governor_packet_json,
    governor_packet_sha256, assert_no_governor_leakage,
    PACKET_SCHEMA as I35_PACKET_SCHEMA)
from .i3_5_model_prompt import SYSTEM_PROMPT as I35_SYSTEM_PROMPT
from .governor.assessor import GeneralGovernor, GovernorDecisionFrame
from .governor.serializer import serialize_frame_dict, frame_sha256
from .governor.identity import compute_governor_identity

import gzip

RUNNER_SCHEMA = "DAPH_V2B_I3_5_FULL_RUNNER_V1"
RUNNER_VERSION = 1
MAX_STEPS = 24


@dataclass(frozen=True)
class GovernorTrajectoryStep:
    """One step in a governor-enhanced trajectory."""
    step_id: int
    proposed_action: str
    reason_code: str
    executed_action: str
    outcome_code: str
    task_success: bool | None
    terminal: bool
    # Governor diagnostics
    governor_top_action: str
    governor_reason_code: str
    governor_frame_sha256: str
    governor_agrees: bool  # whether governor top == model choice


@dataclass(frozen=True)
class GovernorConditionTrajectory:
    """Full trajectory for one task under one condition with governor data."""
    task_id: str
    condition: str  # "BLIND" or "AWARE"
    steps: tuple[GovernorTrajectoryStep, ...]
    terminal_result: str
    task_success: bool
    realized_utility: float
    resources: Mapping[str, int]
    model_calls: int
    decoder_failures: int
    backend_errors: int
    system_fingerprint: str | None
    model_name: str | None
    governor_agreement_rate: float  # fraction of steps where governor == model


@dataclass(frozen=True)
class GovernorPairedResult:
    """Paired blind/aware trajectory result with governor diagnostics."""
    task_id: str
    pair_id: str
    blind: GovernorConditionTrajectory
    aware: GovernorConditionTrajectory
    fingerprint_match: bool
    pair_valid: bool


@dataclass
class I35FullExperimentRunner:
    """Runs the complete I3.5 experiment with governor-enhanced trajectories.

    For each task:
    1. The pair scheduler determines call order (BLIND→AWARE or AWARE→BLIND).
    2. A full multi-step trajectory is run for the first condition.
    3. A full multi-step trajectory is run for the second condition.
    4. Each step: governor assesses → frame injected → DeepSeek chooses → executor runs.
    5. Governor top action and model action are both recorded.
    """

    backend: DeepSeekBackend
    executor: DeterministicActionExecutor = field(default_factory=DeterministicActionExecutor)
    governor: GeneralGovernor = field(default_factory=GeneralGovernor)
    utility: MetareasoningUtility | None = None
    experiment_id: str = "v2b_i3_5_experiment_v1"
    max_steps: int = MAX_STEPS
    strict_json: bool = True
    temperature: float = FROZEN_CONFIG.temperature
    max_tokens: int = FROZEN_CONFIG.max_tokens
    results: list[GovernorPairedResult] = field(default_factory=list, repr=False)

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
    ) -> GovernorConditionTrajectory:
        """Run a full multi-step trajectory with governor enhancement."""
        resources = ResourceState(budget)
        runtime = initial_runtime(
            _I3TaskAdapter(task), resources)
        steps: list[GovernorTrajectoryStep] = []
        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []
        realized = 0.0
        model_calls = 0
        decoder_failures = 0
        backend_errors = 0
        last_fingerprint: str | None = None
        last_model: str | None = None
        governor_agreements = 0

        for step_id in range(self.max_steps):
            observation = self._make_controller_observation(
                runtime, task, mask,
                tuple(prior_decisions), tuple(prior_outcomes))

            # Governor assesses the current state
            prior_action_strs = tuple(
                d.selected_action if isinstance(d.selected_action, str)
                else d.selected_action.value for d in prior_decisions)
            governor_frame = self.governor.assess(
                observation=observation,
                remaining_steps=self.max_steps - step_id,
                prior_actions=prior_action_strs,
                prior_outcomes=tuple(prior_outcomes),
            )

            # Set backend metadata for receipts
            self.backend.task_id = task.task_id
            self.backend.condition = condition
            self.backend.pair_id = f"{self.experiment_id}:{task.task_id}"

            # Serialize and call the model with governor frame
            packet = serialize_governor_packet(observation, governor_frame)
            assert_no_governor_leakage(packet)
            user_prompt = governor_packet_json(packet)

            model_calls += 1
            try:
                call_result = self.backend.generate(
                    system_prompt=I35_SYSTEM_PROMPT, user_prompt=user_prompt,
                    temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception:
                backend_errors += 1
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
            resources_before = runtime.resources
            try:
                execution = self.executor.execute(runtime, action)
            except ResourceExhausted:
                execution = ActionExecution(
                    DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")

            # Compute utility for this step
            if self.utility is not None:
                resources_after = execution.runtime.resources
                step_cost = self.utility.action_cost(resources_before, resources_after)
                realized -= step_cost
                if execution.terminal:
                    realized += self.utility.terminal_reward(
                        execution.action, bool(execution.task_success))

            # Governor diagnostics
            gov_top = governor_frame.governor_top_action
            gov_reason = governor_frame.governor_reason_code
            gov_sha = frame_sha256(governor_frame)
            gov_agrees = (gov_top == action.value)
            if gov_agrees:
                governor_agreements += 1

            steps.append(GovernorTrajectoryStep(
                step_id=step_id,
                proposed_action=proposal.action.value,
                reason_code=proposal.reason_code,
                executed_action=execution.action.value,
                outcome_code=execution.outcome_code,
                task_success=execution.task_success,
                terminal=execution.terminal,
                governor_top_action=gov_top,
                governor_reason_code=gov_reason,
                governor_frame_sha256=gov_sha,
                governor_agrees=gov_agrees))

            prior_decisions.append(DecisionSummary(
                f"{task.task_id}:step:{step_id}", action.value,
                proposal.reason_code, execution.outcome_code))
            prior_outcomes.append(execution.outcome_code)

            runtime = execution.runtime
            if execution.terminal:
                total_steps = len(steps)
                agreement_rate = governor_agreements / total_steps if total_steps > 0 else 0.0
                return GovernorConditionTrajectory(
                    task_id=task.task_id, condition=condition,
                    steps=tuple(steps), terminal_result=execution.outcome_code,
                    task_success=bool(execution.task_success),
                    realized_utility=realized,
                    resources=runtime.resources.as_dict(),
                    model_calls=model_calls, decoder_failures=decoder_failures,
                    backend_errors=backend_errors,
                    system_fingerprint=last_fingerprint,
                    model_name=last_model,
                    governor_agreement_rate=agreement_rate)

        # Step limit reached
        total_steps = len(steps)
        agreement_rate = governor_agreements / total_steps if total_steps > 0 else 0.0
        return GovernorConditionTrajectory(
            task_id=task.task_id, condition=condition,
            steps=tuple(steps), terminal_result="STEP_LIMIT",
            task_success=False, realized_utility=realized,
            resources=runtime.resources.as_dict(),
            model_calls=model_calls, decoder_failures=decoder_failures,
            backend_errors=backend_errors,
            system_fingerprint=last_fingerprint,
            model_name=last_model,
            governor_agreement_rate=agreement_rate)

    def run_pair(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
    ) -> GovernorPairedResult:
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

        result = GovernorPairedResult(
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
    ) -> list[GovernorPairedResult]:
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
                      f"aware_success={result.aware.task_success}, "
                      f"gov_agree_blind={result.blind.governor_agreement_rate:.2f}, "
                      f"gov_agree_aware={result.aware.governor_agreement_rate:.2f}")
        return results

    def all_receipts(self) -> list:
        return self.backend.call_receipts

    def runner_summary(self) -> dict[str, Any]:
        gov_id = compute_governor_identity()
        return {
            "schema": RUNNER_SCHEMA,
            "schema_version": RUNNER_VERSION,
            "experiment_id": self.experiment_id,
            "controller_id": CONTROLLER_ID,
            "governor_sha256": gov_id["governor_sha256"],
            "action_semantics_sha256": gov_id["action_semantics_sha256"],
            "pairs_completed": len(self.results),
            "total_receipts": len(self.backend.call_receipts),
            "total_model_calls": sum(
                r.blind.model_calls + r.aware.model_calls for r in self.results),
            "total_decoder_failures": sum(
                r.blind.decoder_failures + r.aware.decoder_failures for r in self.results),
            "total_backend_errors": sum(
                r.blind.backend_errors + r.aware.backend_errors for r in self.results),
            "strict_json": self.strict_json,
            "mean_governor_agreement_blind": (
                sum(r.blind.governor_agreement_rate for r in self.results) / len(self.results)
                if self.results else 0.0),
            "mean_governor_agreement_aware": (
                sum(r.aware.governor_agreement_rate for r in self.results) / len(self.results)
                if self.results else 0.0),
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


def save_governor_receipts(receipts: list, path: str | Path) -> str:
    """Save receipts to an append-only JSONL file and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in receipts:
            f.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_governor_results(
    results: list[GovernorPairedResult],
    path: str | Path,
) -> str:
    """Save paired governor results to a JSON file and return its SHA-256."""
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
                "blind": _serialize_condition(r.blind),
                "aware": _serialize_condition(r.aware),
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialize_condition(trajectory: GovernorConditionTrajectory) -> dict[str, Any]:
    """Serialize a single condition trajectory."""
    return {
        "condition": trajectory.condition,
        "terminal_result": trajectory.terminal_result,
        "task_success": trajectory.task_success,
        "realized_utility": trajectory.realized_utility,
        "model_calls": trajectory.model_calls,
        "decoder_failures": trajectory.decoder_failures,
        "backend_errors": trajectory.backend_errors,
        "system_fingerprint": trajectory.system_fingerprint,
        "model_name": trajectory.model_name,
        "resources": dict(trajectory.resources),
        "governor_agreement_rate": trajectory.governor_agreement_rate,
        "steps": [
            {
                "step_id": s.step_id,
                "proposed_action": s.proposed_action,
                "reason_code": s.reason_code,
                "executed_action": s.executed_action,
                "outcome_code": s.outcome_code,
                "task_success": s.task_success,
                "terminal": s.terminal,
                "governor_top_action": s.governor_top_action,
                "governor_reason_code": s.governor_reason_code,
                "governor_frame_sha256": s.governor_frame_sha256,
                "governor_agrees": s.governor_agrees,
            }
            for s in trajectory.steps
        ],
    }


def score_governor_results(
    results: list[GovernorPairedResult],
    benchmark: MetareasoningBenchmark,
    oracle_views_path: str | Path,
    latent_oracle_path: str | Path,
    utility_weights: Mapping[str, float],
) -> tuple[list[I34ScientificTaskContribution], list[I34PairedDelta]]:
    """Score paired governor results using the same V2 scoring as I3.4.

    The scoring is identical — only the trajectories differ.
    """
    # Load observable oracle views — per-task V_O from V2 view structure
    views_data = json.loads(Path(oracle_views_path).read_text())
    task_vo: dict[tuple[str, str], tuple[float, str, str]] = {}
    for v in views_data["views"]:
        condition = v["condition"]
        oracle_set_sha = v.get("observable_oracle_set_sha256", "")
        for entry in v.get("task_entries", []):
            tid = entry["task_id"]
            task_vo[(tid, condition)] = (
                entry["observable_optimal_value"],
                entry.get("information_class_id", ""),
                oracle_set_sha,
            )

    # Load latent oracle values
    latent_values: dict[str, float] = {}
    latent_table_shas: dict[str, str] = {}
    with gzip.open(latent_oracle_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            task_id = entry.get("task_id", "")
            table = entry.get("table", entry)
            state_values = table.get("state_values", {})
            init_id = entry.get("initial_state_id") or table.get("initial_state_id")
            if init_id and init_id in state_values:
                latent_values[task_id] = float(state_values[init_id])
            latent_table_shas[task_id] = table.get("identity_sha256", "")

    task_by_id = {t.task_id: t for t in benchmark.tasks}

    blind_contributions: dict[str, I34ScientificTaskContribution] = {}
    aware_contributions: dict[str, I34ScientificTaskContribution] = {}
    all_contributions: list[I34ScientificTaskContribution] = []

    for result in results:
        task = task_by_id.get(result.task_id)
        if task is None:
            continue

        vo_blind_entry = task_vo.get((result.task_id, "STATE_BLIND_CONTROLLER"))
        vo_aware_entry = task_vo.get((result.task_id, "STATE_AWARE_CONTROLLER"))
        if vo_blind_entry is None or vo_aware_entry is None:
            continue
        v_o_blind, blind_class_id, blind_oracle_sha = vo_blind_entry
        v_o_aware, aware_class_id, aware_oracle_sha = vo_aware_entry

        v_l = latent_values.get(result.task_id, 0.0)
        latent_sha = latent_table_shas.get(result.task_id, "")

        v_pi_blind = result.blind.realized_utility
        v_pi_aware = result.aware.realized_utility

        blind_contrib = compute_task_contribution(
            task_id=result.task_id, condition="STATE_BLIND_CONTROLLER",
            latent_optimal_value=v_l, observable_optimal_value=v_o_blind,
            controller_value=v_pi_blind,
            information_class_hash=blind_class_id,
            observable_oracle_set_sha256=blind_oracle_sha,
            latent_oracle_table_sha256=latent_sha)
        aware_contrib = compute_task_contribution(
            task_id=result.task_id, condition="STATE_AWARE_CONTROLLER",
            latent_optimal_value=v_l, observable_optimal_value=v_o_aware,
            controller_value=v_pi_aware,
            information_class_hash=aware_class_id,
            observable_oracle_set_sha256=aware_oracle_sha,
            latent_oracle_table_sha256=latent_sha)

        blind_contributions[result.task_id] = blind_contrib
        aware_contributions[result.task_id] = aware_contrib
        all_contributions.extend([blind_contrib, aware_contrib])

    deltas = compute_paired_deltas(
        blind_contributions=blind_contributions,
        aware_contributions=aware_contributions)

    return all_contributions, deltas
