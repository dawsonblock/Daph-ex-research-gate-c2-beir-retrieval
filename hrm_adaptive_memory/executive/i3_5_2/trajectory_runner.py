"""Trajectory runner for V2B-I3.5.2 Selective Governor Intervention.

Supports four governor modes:
  - OFF: Base packet only, governor never invoked.
  - ALWAYS_ON: Governor always invoked and injected into model packet.
  - SELECTIVE: SelectiveGovernorGate decides whether governor is injected.
  - SHADOW_SELECTIVE: Gate evaluates silently, base packet sent to model.
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

from ..i3_5_1.conditions import ExperimentalCondition, ConditionID, get_condition
from ..i3_5_1.observation_builder import build_observation
from ..i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    packet_json, packet_sha256, assert_no_evaluator_leakage,
)
from ..i3_5_1.model_prompt import SYSTEM_PROMPT
from ..i3_5_1.receipts import ReceiptLedger, make_receipt

from ..selective_governor import (
    SelectiveGovernorGate,
    InterventionDecision,
    serialize_decision,
)
from .modes import GovernorMode

RUNNER_SCHEMA = "DAPH_V2B_I3_5_2_TRAJECTORY_RUNNER_V1"
RUNNER_VERSION = 1


@dataclass(frozen=True)
class I352TrajectoryStep:
    """One step in an I3.5.2 trajectory with selective gating diagnostics."""
    step_id: int
    condition_id: str
    governor_mode: str
    proposed_action: str
    reason_code: str
    executed_action: str
    outcome_code: str
    task_success: bool | None
    terminal: bool
    # Selective gate diagnostics
    gate_decision: dict[str, Any] | None
    gate_intervened: bool
    # Governor diagnostics
    governor_top_action: str | None
    governor_reason_code: str | None
    governor_frame_sha256: str | None
    governor_agrees: bool | None
    # Packet info
    packet_sha256: str
    packet_schema: str


@dataclass(frozen=True)
class I352ConditionTrajectory:
    """Full trajectory for one task under an I3.5.2 condition."""
    task_id: str
    condition_id: str
    observation_mode: str
    governor_mode: str
    steps: tuple[I352TrajectoryStep, ...]
    terminal_result: str
    task_success: bool
    realized_utility: float
    resources: Mapping[str, int]
    model_calls: int
    decoder_failures: int
    backend_errors: int
    system_fingerprint: str | None
    model_name: str | None
    governor_agreement_rate: float | None
    interventions_approved: int
    total_decisions: int


@dataclass
class I352FactorialRunner:
    """Runs I3.5.2 trajectories with selective governor gating."""

    backend: DeepSeekBackend
    executor: DeterministicActionExecutor = field(default_factory=DeterministicActionExecutor)
    governor: GeneralGovernor = field(default_factory=GeneralGovernor)
    gate: SelectiveGovernorGate = field(default_factory=SelectiveGovernorGate)
    utility: MetareasoningUtility | None = None
    experiment_id: str = "v2b_i3_5_2_experiment_v1"
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
        governor_mode: GovernorMode = GovernorMode.SELECTIVE,
    ) -> I352ConditionTrajectory:
        """Run a full multi-step trajectory with the specified governor mode."""
        resources = ResourceState(budget)
        runtime = initial_runtime(_I3TaskAdapter(task), resources)
        steps: list[I352TrajectoryStep] = []
        prior_decisions: list[DecisionSummary] = []
        prior_outcomes: list[str] = []
        realized = 0.0
        model_calls = 0
        decoder_failures = 0
        backend_errors = 0
        last_fingerprint: str | None = None
        last_model: str | None = None
        governor_agreements = 0
        interventions_approved = 0

        trajectory_id = f"{self.experiment_id}:{task.task_id}:{condition.condition_id.value}:{governor_mode.value}"

        for step_id in range(self.max_steps):
            observation = build_observation(
                runtime, task, condition,
                tuple(prior_decisions), tuple(prior_outcomes))

            prior_action_strs = tuple(
                d.selected_action if isinstance(d.selected_action, str)
                else d.selected_action.value for d in prior_decisions)

            # Routing according to GovernorMode
            gate_decision: InterventionDecision | None = None
            governor_frame: GovernorDecisionFrame | None = None
            actual_intervened = False

            if governor_mode == GovernorMode.OFF or not condition.governor_enabled:
                packet = build_base_packet(observation)
                packet_schema = packet["schema"]
                gov_top, gov_reason, gov_sha = None, None, None

            elif governor_mode == GovernorMode.ALWAYS_ON:
                actual_intervened = True
                interventions_approved += 1
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

            elif governor_mode == GovernorMode.SELECTIVE:
                gate_decision = self.gate.assess(
                    observation=observation,
                    remaining_steps=self.max_steps - step_id,
                    prior_actions=prior_action_strs,
                    prior_outcomes=tuple(prior_outcomes),
                )
                if gate_decision.intervene:
                    actual_intervened = True
                    interventions_approved += 1
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
                    gov_top, gov_reason, gov_sha = None, None, None

            elif governor_mode == GovernorMode.SHADOW_SELECTIVE:
                gate_decision = self.gate.assess(
                    observation=observation,
                    remaining_steps=self.max_steps - step_id,
                    prior_actions=prior_action_strs,
                    prior_outcomes=tuple(prior_outcomes),
                )
                # Shadow mode evaluates silently, sends base packet
                packet = build_base_packet(observation)
                packet_schema = packet["schema"]
                gov_top, gov_reason, gov_sha = None, None, None

            assert_no_evaluator_leakage(packet)
            user_prompt = packet_json(packet)
            pkt_sha = packet_sha256(packet)

            condition_sha = hashlib.sha256(
                f"{condition.condition_id.value}:{governor_mode.value}".encode()).hexdigest()

            self.backend.task_id = task.task_id
            self.backend.condition = f"{condition.condition_id.value}_{governor_mode.value}"
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

            # Execute action
            action = proposal.action
            resources_before = runtime.resources
            try:
                execution = self.executor.execute(runtime, action)
            except ResourceExhausted:
                execution = ActionExecution(
                    DecisionAction.DEFER, runtime, True, False, "RESOURCE_EXHAUSTED")

            if self.utility is not None:
                resources_after = execution.runtime.resources
                step_cost = self.utility.action_cost(resources_before, resources_after)
                realized -= step_cost
                if execution.terminal:
                    realized += self.utility.terminal_reward(
                        execution.action, bool(execution.task_success))

            # Governor agreement
            gov_agrees: bool | None = None
            if actual_intervened and gov_top is not None:
                gov_agrees = (gov_top == action.value)
                if gov_agrees:
                    governor_agreements += 1

            steps.append(I352TrajectoryStep(
                step_id=step_id,
                condition_id=condition.condition_id.value,
                governor_mode=governor_mode.value,
                proposed_action=proposal.action.value,
                reason_code=proposal.reason_code,
                executed_action=execution.action.value,
                outcome_code=execution.outcome_code,
                task_success=execution.task_success,
                terminal=execution.terminal,
                gate_decision=serialize_decision(gate_decision) if gate_decision else None,
                gate_intervened=actual_intervened,
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
                    governor_agreements / interventions_approved
                    if interventions_approved > 0
                    else None
                )
                return I352ConditionTrajectory(
                    task_id=task.task_id,
                    condition_id=condition.condition_id.value,
                    observation_mode=condition.observation_mode.value,
                    governor_mode=governor_mode.value,
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
                    interventions_approved=interventions_approved,
                    total_decisions=total_steps,
                )

        total_steps = len(steps)
        agreement_rate = (
            governor_agreements / interventions_approved
            if interventions_approved > 0
            else None
        )
        return I352ConditionTrajectory(
            task_id=task.task_id,
            condition_id=condition.condition_id.value,
            observation_mode=condition.observation_mode.value,
            governor_mode=governor_mode.value,
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
            interventions_approved=interventions_approved,
            total_decisions=total_steps,
        )

    def run_comparison_block_standalone(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
        modes: tuple[GovernorMode, ...] = (
            GovernorMode.OFF,
            GovernorMode.ALWAYS_ON,
            GovernorMode.SELECTIVE,
        ),
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run comparison arms (AWARE under OFF / ALWAYS_ON / SELECTIVE) for one task block."""
        saved_ledger = self.receipt_ledger
        self.receipt_ledger = ReceiptLedger(run_id=saved_ledger.run_id)

        cond = get_condition(ConditionID.AWARE_GOVERNOR)
        trajectories: dict[str, Any] = {}

        for mode in modes:
            traj = self._run_trajectory(task, budget, cond, governor_mode=mode)
            trajectories[mode.value] = _serialize_i352_trajectory(traj)

        receipts = [r.as_dict() for r in self.receipt_ledger.receipts]
        self.receipt_ledger = saved_ledger

        block_result = {
            "task_id": task.task_id,
            "block_id": f"{self.experiment_id}:{task.task_id}",
            "trajectories": trajectories,
        }
        return block_result, receipts


class _I3TaskAdapter:
    """Adapter to make I3BenchmarkTask compatible with the executor."""
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


def _serialize_i352_trajectory(traj: I352ConditionTrajectory) -> dict[str, Any]:
    return {
        "condition_id": traj.condition_id,
        "observation_mode": traj.observation_mode,
        "governor_mode": traj.governor_mode,
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
        "interventions_approved": traj.interventions_approved,
        "total_decisions": traj.total_decisions,
        "steps": [
            {
                "step_id": s.step_id,
                "condition_id": s.condition_id,
                "governor_mode": s.governor_mode,
                "proposed_action": s.proposed_action,
                "reason_code": s.reason_code,
                "executed_action": s.executed_action,
                "outcome_code": s.outcome_code,
                "task_success": s.task_success,
                "terminal": s.terminal,
                "gate_decision": s.gate_decision,
                "gate_intervened": s.gate_intervened,
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
