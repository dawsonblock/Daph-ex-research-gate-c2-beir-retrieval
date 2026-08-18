"""Trajectory runner for I3.5.1 factorial experiment.

Runs a full multi-step trajectory for one task under one condition.
The critical control path:

  observation = build_observation(runtime, task, condition, ...)
  if condition.governor_enabled:
      governor_frame = governor.assess(observation, ...)
      packet = build_governor_packet(observation, governor_frame)
  else:
      governor_frame = None
      packet = build_base_packet(observation)

No-governor arms never invoke governor.assess().
No-governor packets contain zero governor structure.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from ..actions import ActionProposal
from ..executor import (
    ActionExecution, DeterministicActionExecutor, TaskRuntime,
    initial_runtime,
)
from ..metareasoning_benchmark import I3BenchmarkTask
from ..model_backend import DeepSeekBackend, ModelBackend, ModelCallResult
from ..model_decoder import decode_output
from ..pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)
from ..resources import ResourceBudget, ResourceState, ResourceExhausted
from ..metareasoning_utility import MetareasoningUtility
from ..governor.assessor import GeneralGovernor, GovernorDecisionFrame
from ..governor.serializer import frame_sha256

from .conditions import ExperimentalCondition, ConditionID
from .observation_builder import build_observation
from .packet_builder import (
    build_base_packet, build_governor_packet,
    packet_json, packet_sha256, assert_no_evaluator_leakage,
)
from .model_prompt import SYSTEM_PROMPT
from .receipts import ReceiptLedger, make_receipt

RUNNER_SCHEMA = "DAPH_V2B_I3_5_1_TRAJECTORY_RUNNER_V1"
RUNNER_VERSION = 1


@dataclass(frozen=True)
class TrajectoryStep:
    """One step in a trajectory."""
    step_id: int
    condition_id: str
    proposed_action: str
    reason_code: str
    executed_action: str
    outcome_code: str
    task_success: bool | None
    terminal: bool
    # Governor diagnostics (None for no-governor conditions)
    governor_top_action: str | None
    governor_reason_code: str | None
    governor_frame_sha256: str | None
    governor_agrees: bool | None
    # Packet info
    packet_sha256: str
    packet_schema: str


@dataclass(frozen=True)
class ConditionTrajectory:
    """Full trajectory for one task under one condition."""
    task_id: str
    condition_id: str
    observation_mode: str
    governor_enabled: bool
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
    governor_agreement_rate: float | None  # None for no-governor


@dataclass
class FactorialExperimentRunner:
    """Runs the complete I3.5.1 2x2 factorial experiment.

    For each task block:
    1. The factorial scheduler determines the condition order.
    2. A full multi-step trajectory is run for each of the four conditions.
    3. Each step: observation built → governor (if enabled) → packet → model → executor.
    4. Receipts are recorded in an append-only hash chain ledger.
    """

    backend: DeepSeekBackend
    executor: DeterministicActionExecutor = field(default_factory=DeterministicActionExecutor)
    governor: GeneralGovernor = field(default_factory=GeneralGovernor)
    utility: MetareasoningUtility | None = None
    experiment_id: str = "v2b_i3_5_1_experiment_v1"
    experiment_identity_sha256: str = ""
    max_steps: int = 24
    strict_json: bool = True
    temperature: float = 0.0
    max_tokens: int = 2048
    receipt_ledger: ReceiptLedger = field(default_factory=lambda: ReceiptLedger())
    results: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def _run_trajectory(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
        condition: ExperimentalCondition,
    ) -> ConditionTrajectory:
        """Run a full multi-step trajectory for one condition."""
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
        governor_agreements = 0

        trajectory_id = f"{self.experiment_id}:{task.task_id}:{condition.condition_id.value}"

        for step_id in range(self.max_steps):
            observation = build_observation(
                runtime, task, condition,
                tuple(prior_decisions), tuple(prior_outcomes))

            prior_action_strs = tuple(
                d.selected_action if isinstance(d.selected_action, str)
                else d.selected_action.value for d in prior_decisions)

            # Governor: only assess if enabled
            governor_frame: GovernorDecisionFrame | None = None
            if condition.governor_enabled:
                governor_frame = self.governor.assess(
                    observation=observation,
                    remaining_steps=self.max_steps - step_id,
                    prior_actions=prior_action_strs,
                    prior_outcomes=tuple(prior_outcomes),
                )
                packet = build_governor_packet(observation, governor_frame)
                packet_schema = packet["schema"]
                gov_top = governor_frame.governor_top_action
                gov_reason = governor_frame.governor_reason_code
                gov_sha = frame_sha256(governor_frame)
            else:
                packet = build_base_packet(observation)
                packet_schema = packet["schema"]
                gov_top = None
                gov_reason = None
                gov_sha = None

            assert_no_evaluator_leakage(packet)
            user_prompt = packet_json(packet)
            pkt_sha = packet_sha256(packet)

            # Build receipt metadata
            condition_sha = hashlib.sha256(
                condition.condition_id.value.encode()).hexdigest()

            # Set backend metadata
            self.backend.task_id = task.task_id
            self.backend.condition = condition.condition_id.value
            self.backend.pair_id = trajectory_id

            ts_start = datetime.now(timezone.utc).isoformat()
            t_start = time.monotonic()

            model_calls += 1
            try:
                call_result = self.backend.generate(
                    system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                    temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception:
                backend_errors += 1
                proposal = BACKEND_ERROR_PROPOSAL
                call_result = None
                http_status = 500
                raw_output = None
                reported_model = None
                fingerprint = None
            else:
                last_fingerprint = call_result.system_fingerprint
                last_model = call_result.model_name
                raw_output = call_result.raw_output
                http_status = 200
                reported_model = call_result.model_name
                fingerprint = call_result.system_fingerprint
                outcome = decode_output(call_result.raw_output, strict=self.strict_json)
                if outcome.valid and outcome.proposal:
                    proposal = outcome.proposal
                else:
                    decoder_failures += 1
                    proposal = FAIL_CLOSED_PROPOSAL

            t_end = time.monotonic()
            ts_end = datetime.now(timezone.utc).isoformat()
            latency_ms = (t_end - t_start) * 1000.0

            # Record receipt
            receipt = make_receipt(
                run_id=self.receipt_ledger.run_id,
                experiment_identity_sha256=self.experiment_identity_sha256,
                condition_identity_sha256=condition_sha,
                task_id=task.task_id,
                pair_or_block_id=f"{self.experiment_id}:{task.task_id}",
                trajectory_id=trajectory_id,
                step_id=step_id,
                attempt_index=0,
                input_packet=packet,
                system_prompt=SYSTEM_PROMPT,
                generation_config={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "strict_json": self.strict_json,
                },
                provider="deepseek",
                requested_model="deepseek-chat",
                reported_model=reported_model,
                system_fingerprint=fingerprint,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                latency_ms=latency_ms,
                http_status=http_status,
                result_class=(
                    "OK" if call_result and proposal is not BACKEND_ERROR_PROPOSAL
                    else "BACKEND_ERROR" if call_result is None
                    else "DECODER_FAILURE" if proposal is FAIL_CLOSED_PROPOSAL
                    else "OK"
                ),
                raw_output=raw_output,
                parsed_output=(
                    {"action": proposal.action.value, "reason_code": proposal.reason_code}
                    if proposal is not None else None
                ),
                decoder_status=(
                    "VALID" if proposal is not None and proposal is not FAIL_CLOSED_PROPOSAL
                    else "INVALID" if proposal is FAIL_CLOSED_PROPOSAL and call_result is not None
                    else "NOT_RUN"
                ),
                previous_receipt_sha256=self.receipt_ledger.receipt_chain_root,
            )
            self.receipt_ledger.add(receipt)

            # Execute the action
            action = proposal.action
            resources_before = runtime.resources
            try:
                execution = self.executor.execute(runtime, action)
            except ResourceExhausted:
                execution = ActionExecution(
                    DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")

            # Compute utility
            if self.utility is not None:
                resources_after = execution.runtime.resources
                step_cost = self.utility.action_cost(resources_before, resources_after)
                realized -= step_cost
                if execution.terminal:
                    realized += self.utility.terminal_reward(
                        execution.action, bool(execution.task_success))

            # Governor diagnostics
            gov_agrees: bool | None = None
            if condition.governor_enabled and gov_top is not None:
                gov_agrees = (gov_top == action.value)
                if gov_agrees:
                    governor_agreements += 1

            steps.append(TrajectoryStep(
                step_id=step_id,
                condition_id=condition.condition_id.value,
                proposed_action=proposal.action.value,
                reason_code=proposal.reason_code,
                executed_action=execution.action.value,
                outcome_code=execution.outcome_code,
                task_success=execution.task_success,
                terminal=execution.terminal,
                governor_top_action=gov_top,
                governor_reason_code=gov_reason,
                governor_frame_sha256=gov_sha,
                governor_agrees=gov_agrees,
                packet_sha256=pkt_sha,
                packet_schema=packet_schema,
            ))

            prior_decisions.append(DecisionSummary(
                f"{task.task_id}:step:{step_id}", action.value,
                proposal.reason_code, execution.outcome_code))
            prior_outcomes.append(execution.outcome_code)

            runtime = execution.runtime
            if execution.terminal:
                total_steps = len(steps)
                agreement_rate = (
                    governor_agreements / total_steps
                    if condition.governor_enabled and total_steps > 0
                    else None
                )
                return ConditionTrajectory(
                    task_id=task.task_id,
                    condition_id=condition.condition_id.value,
                    observation_mode=condition.observation_mode.value,
                    governor_enabled=condition.governor_enabled,
                    steps=tuple(steps),
                    terminal_result=execution.outcome_code,
                    task_success=bool(execution.task_success),
                    realized_utility=realized,
                    resources=runtime.resources.as_dict(),
                    model_calls=model_calls,
                    decoder_failures=decoder_failures,
                    backend_errors=backend_errors,
                    system_fingerprint=last_fingerprint,
                    model_name=last_model,
                    governor_agreement_rate=agreement_rate,
                )

        # Step limit reached
        total_steps = len(steps)
        agreement_rate = (
            governor_agreements / total_steps
            if condition.governor_enabled and total_steps > 0
            else None
        )
        return ConditionTrajectory(
            task_id=task.task_id,
            condition_id=condition.condition_id.value,
            observation_mode=condition.observation_mode.value,
            governor_enabled=condition.governor_enabled,
            steps=tuple(steps),
            terminal_result="STEP_LIMIT",
            task_success=False,
            realized_utility=realized,
            resources=runtime.resources.as_dict(),
            model_calls=model_calls,
            decoder_failures=decoder_failures,
            backend_errors=backend_errors,
            system_fingerprint=last_fingerprint,
            model_name=last_model,
            governor_agreement_rate=agreement_rate,
        )

    def run_block(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
        condition_order: tuple[ConditionID, ...],
    ) -> dict[str, Any]:
        """Run all four conditions for one task block."""
        from .conditions import get_condition
        trajectories: dict[str, ConditionTrajectory] = {}
        for cid in condition_order:
            cond = get_condition(cid)
            traj = self._run_trajectory(task, budget, cond)
            trajectories[cid.value] = traj

        block_result = {
            "task_id": task.task_id,
            "block_id": f"{self.experiment_id}:{task.task_id}",
            "trajectories": {
                cid.value: _serialize_trajectory(traj)
                for cid, traj in zip(condition_order,
                                     [trajectories[c.value] for c in condition_order])
            },
        }
        self.results.append(block_result)
        return block_result

    def runner_summary(self) -> dict[str, Any]:
        from ..governor.identity import compute_governor_identity
        gov_id = compute_governor_identity()
        return {
            "schema": RUNNER_SCHEMA,
            "schema_version": RUNNER_VERSION,
            "experiment_id": self.experiment_id,
            "experiment_identity_sha256": self.experiment_identity_sha256,
            "governor_sha256": gov_id["governor_sha256"],
            "action_semantics_sha256": gov_id["action_semantics_sha256"],
            "blocks_completed": len(self.results),
            "total_receipts": self.receipt_ledger.receipt_count,
            "receipt_chain_root": self.receipt_ledger.receipt_chain_root,
            "total_model_calls": sum(
                sum(t["model_calls"]
                    for t in block["trajectories"].values())
                for block in self.results),
            "total_decoder_failures": sum(
                sum(t["decoder_failures"]
                    for t in block["trajectories"].values())
                for block in self.results),
            "total_backend_errors": sum(
                sum(t["backend_errors"]
                    for t in block["trajectories"].values())
                for block in self.results),
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


def _serialize_trajectory(traj: ConditionTrajectory) -> dict[str, Any]:
    """Serialize a single condition trajectory."""
    return {
        "condition_id": traj.condition_id,
        "observation_mode": traj.observation_mode,
        "governor_enabled": traj.governor_enabled,
        "terminal_result": traj.terminal_result,
        "task_success": traj.task_success,
        "realized_utility": traj.realized_utility,
        "model_calls": traj.model_calls,
        "decoder_failures": traj.decoder_failures,
        "backend_errors": traj.backend_errors,
        "system_fingerprint": traj.system_fingerprint,
        "model_name": traj.model_name,
        "resources": dict(traj.resources),
        "governor_agreement_rate": traj.governor_agreement_rate,
        "steps": [
            {
                "step_id": s.step_id,
                "condition_id": s.condition_id,
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
                "packet_sha256": s.packet_sha256,
                "packet_schema": s.packet_schema,
            }
            for s in traj.steps
        ],
    }


def save_results(
    results: list[dict[str, Any]],
    path: str | Path,
    *,
    experiment_identity_sha256: str,
    receipt_chain_root: str,
    source_receipts_sha256: str,
) -> str:
    """Save results with provenance metadata. Return file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "DAPH_V2B_I3_5_1_RESULTS_V1",
        "schema_version": 1,
        "experiment_identity_sha256": experiment_identity_sha256,
        "receipt_chain_root": receipt_chain_root,
        "source_receipts_sha256": source_receipts_sha256,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
