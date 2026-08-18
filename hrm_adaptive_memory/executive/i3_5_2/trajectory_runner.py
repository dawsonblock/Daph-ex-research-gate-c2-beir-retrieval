"""Trajectory runner for V2B-I3.5.2 Selective Governor Intervention.

Supports five governor modes:
  - OFF: Base packet only, governor never invoked.
  - ALWAYS_ON: Governor always invoked and injected into model packet.
  - SELECTIVE: SelectiveGovernorGate decides whether governor is injected.
  - SELECTIVE_FRAME: Same as SELECTIVE — gate approves → governor advisory
    packet → model chooses. Explicit name for the advisory architecture.
  - SHADOW_SELECTIVE: Gate evaluates silently, base packet sent to model.

I3.5.2c additions:
  - Token and latency tracking per step
  - Detailed per-intervention instrumentation
  - Counterbalancing support via deterministic arm ordering
  - Cascade (consecutive intervention) tracking
"""
from __future__ import annotations

import hashlib
import hmac
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
from ..selective_governor.features import extract_features

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

# The six permutations for counterbalancing
ARM_PERMUTATIONS: tuple[tuple[GovernorMode, ...], ...] = (
    (GovernorMode.OFF, GovernorMode.ALWAYS_ON, GovernorMode.SELECTIVE_FRAME),
    (GovernorMode.OFF, GovernorMode.SELECTIVE_FRAME, GovernorMode.ALWAYS_ON),
    (GovernorMode.ALWAYS_ON, GovernorMode.OFF, GovernorMode.SELECTIVE_FRAME),
    (GovernorMode.ALWAYS_ON, GovernorMode.SELECTIVE_FRAME, GovernorMode.OFF),
    (GovernorMode.SELECTIVE_FRAME, GovernorMode.OFF, GovernorMode.ALWAYS_ON),
    (GovernorMode.SELECTIVE_FRAME, GovernorMode.ALWAYS_ON, GovernorMode.OFF),
)


def counterbalanced_order(seed: str, task_id: str) -> tuple[GovernorMode, ...]:
    """Deterministically select arm ordering from HMAC(seed, task_id) % 6."""
    key = seed.encode() if isinstance(seed, str) else seed
    msg = task_id.encode()
    h = hmac.new(key, msg, hashlib.sha256).digest()
    idx = int.from_bytes(h[:4], "big") % 6
    return ARM_PERMUTATIONS[idx]


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
    # Cost tracking (I3.5.2c)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    packet_byte_length: int


@dataclass(frozen=True)
class InterventionRecord:
    """Detailed record of a single intervention event."""
    task_id: str
    step_id: int
    gate_decision: str
    gate_reason: str
    expected_delta_q: float
    predicted_harm_probability: float
    predicted_confidence: float
    model_action: str
    governor_top_action: str
    governor_model_agreement: bool
    outcome: str
    packet_schema: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class I352ConditionTrajectory:
    """Full trajectory for one task under an I3.5.2 condition."""
    task_id: str
    condition_id: str
    observation_mode: str
    governor_mode: str
    steps: tuple[I352TrajectoryStep, ...]
    interventions: tuple[InterventionRecord, ...]
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
    # Cost totals (I3.5.2c)
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_latency_ms: float
    # Cascade diagnostics (I3.5.2c)
    max_consecutive_interventions: int
    intervention_chain_lengths: tuple[int, ...]


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
        governor_mode: GovernorMode = GovernorMode.SELECTIVE_FRAME,
    ) -> I352ConditionTrajectory:
        """Run a full multi-step trajectory with the specified governor mode."""
        resources = ResourceState(budget)
        runtime = initial_runtime(_I3TaskAdapter(task), resources)
        steps: list[I352TrajectoryStep] = []
        interventions: list[InterventionRecord] = []
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

        # Cost totals
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_latency_ms = 0.0

        # Cascade tracking
        current_consecutive = 0
        max_consecutive = 0
        intervention_chain_lengths: list[int] = []

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

            elif governor_mode in (GovernorMode.SELECTIVE, GovernorMode.SELECTIVE_FRAME):
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
                packet = build_base_packet(observation)
                packet_schema = packet["schema"]
                gov_top, gov_reason, gov_sha = None, None, None

            assert_no_evaluator_leakage(packet)
            user_prompt = packet_json(packet)
            pkt_sha = packet_sha256(packet)
            packet_bytes = len(user_prompt.encode())

            condition_sha = hashlib.sha256(
                f"{condition.condition_id.value}:{governor_mode.value}".encode()).hexdigest()

            self.backend.task_id = task.task_id
            self.backend.condition = f"{condition.condition_id.value}_{governor_mode.value}"
            self.backend.pair_id = trajectory_id

            ts_start = datetime.now(timezone.utc).isoformat()
            t_start = time.monotonic()

            model_calls += 1
            prompt_tokens = 0
            completion_tokens = 0
            call_total_tokens = 0
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
                prompt_tokens = call_result.prompt_tokens
                completion_tokens = call_result.completion_tokens
                call_total_tokens = prompt_tokens + completion_tokens
                outcome = decode_output(call_result.raw_output, strict=self.strict_json)
                if outcome.valid and outcome.proposal:
                    proposal = outcome.proposal
                else:
                    decoder_failures += 1
                    proposal = FAIL_CLOSED_PROPOSAL

            t_end = time.monotonic()
            ts_end = datetime.now(timezone.utc).isoformat()
            latency_ms = (t_end - t_start) * 1000.0
            total_latency_ms += latency_ms
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += call_total_tokens

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

            # Cascade tracking
            if actual_intervened:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                if current_consecutive > 0:
                    intervention_chain_lengths.append(current_consecutive)
                current_consecutive = 0

            # Record intervention detail
            if actual_intervened and gate_decision is not None:
                interventions.append(InterventionRecord(
                    task_id=task.task_id,
                    step_id=step_id,
                    gate_decision="INTERVENE",
                    gate_reason=gate_decision.reason_code,
                    expected_delta_q=gate_decision.expected_delta_utility,
                    predicted_harm_probability=gate_decision.harm_probability,
                    predicted_confidence=gate_decision.confidence,
                    model_action=action.value,
                    governor_top_action=gov_top or "",
                    governor_model_agreement=gov_agrees or False,
                    outcome=execution.outcome_code,
                    packet_schema=packet_schema,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                ))
            elif actual_intervened and governor_mode == GovernorMode.ALWAYS_ON:
                # Always-on interventions don't have gate_decision but still record
                interventions.append(InterventionRecord(
                    task_id=task.task_id,
                    step_id=step_id,
                    gate_decision="ALWAYS_ON",
                    gate_reason=gov_reason or "ALWAYS_ON",
                    expected_delta_q=0.0,
                    predicted_harm_probability=0.0,
                    predicted_confidence=1.0,
                    model_action=action.value,
                    governor_top_action=gov_top or "",
                    governor_model_agreement=gov_agrees or False,
                    outcome=execution.outcome_code,
                    packet_schema=packet_schema,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                ))

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
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=call_total_tokens,
                latency_ms=latency_ms,
                packet_byte_length=packet_bytes,
            ))

            prior_decisions.append(DecisionSummary(
                f"{task.task_id}:step:{step_id}", action.value,
                proposal.reason_code, execution.outcome_code))
            prior_outcomes.append(execution.outcome_code)

            runtime = execution.runtime
            if execution.terminal:
                # Close any open cascade
                if current_consecutive > 0:
                    intervention_chain_lengths.append(current_consecutive)
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
                    interventions=tuple(interventions),
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
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency_ms,
                    max_consecutive_interventions=max_consecutive,
                    intervention_chain_lengths=tuple(intervention_chain_lengths),
                )

        # Close any open cascade
        if current_consecutive > 0:
            intervention_chain_lengths.append(current_consecutive)
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
            interventions=tuple(interventions),
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
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            max_consecutive_interventions=max_consecutive,
            intervention_chain_lengths=tuple(intervention_chain_lengths),
        )

    def run_comparison_block_standalone(
        self,
        task: I3BenchmarkTask,
        budget: ResourceBudget,
        modes: tuple[GovernorMode, ...] = (
            GovernorMode.OFF,
            GovernorMode.ALWAYS_ON,
            GovernorMode.SELECTIVE_FRAME,
        ),
        counterbalance_seed: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run comparison arms for one task block with optional counterbalancing.

        If counterbalance_seed is non-empty, uses HMAC(seed, task_id) % 6
        to deterministically select arm ordering from the 6 permutations.
        """
        saved_ledger = self.receipt_ledger
        self.receipt_ledger = ReceiptLedger(run_id=saved_ledger.run_id)

        cond = get_condition(ConditionID.AWARE_GOVERNOR)
        trajectories: dict[str, Any] = {}

        # Determine execution order
        if counterbalance_seed:
            execution_order = counterbalanced_order(counterbalance_seed, task.task_id)
        else:
            execution_order = modes

        for mode in execution_order:
            traj = self._run_trajectory(task, budget, cond, governor_mode=mode)
            trajectories[mode.value] = _serialize_i352_trajectory(traj)

        receipts = [r.as_dict() for r in self.receipt_ledger.receipts]
        self.receipt_ledger = saved_ledger

        block_result = {
            "task_id": task.task_id,
            "block_id": f"{self.experiment_id}:{task.task_id}",
            "execution_order": [m.value for m in execution_order],
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
        # Cost totals (I3.5.2c)
        "total_prompt_tokens": traj.total_prompt_tokens,
        "total_completion_tokens": traj.total_completion_tokens,
        "total_tokens": traj.total_tokens,
        "total_latency_ms": round(traj.total_latency_ms, 2),
        # Cascade diagnostics (I3.5.2c)
        "max_consecutive_interventions": traj.max_consecutive_interventions,
        "intervention_chain_lengths": list(traj.intervention_chain_lengths),
        # Detailed intervention records (I3.5.2c)
        "interventions": [
            {
                "task_id": iv.task_id,
                "step_id": iv.step_id,
                "gate_decision": iv.gate_decision,
                "gate_reason": iv.gate_reason,
                "expected_delta_q": iv.expected_delta_q,
                "predicted_harm_probability": iv.predicted_harm_probability,
                "predicted_confidence": iv.predicted_confidence,
                "model_action": iv.model_action,
                "governor_top_action": iv.governor_top_action,
                "governor_model_agreement": iv.governor_model_agreement,
                "outcome": iv.outcome,
                "packet_schema": iv.packet_schema,
                "prompt_tokens": iv.prompt_tokens,
                "completion_tokens": iv.completion_tokens,
                "latency_ms": round(iv.latency_ms, 2),
            }
            for iv in traj.interventions
        ],
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
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "total_tokens": s.total_tokens,
                "latency_ms": round(s.latency_ms, 2),
                "packet_byte_length": s.packet_byte_length,
            }
            for s in traj.steps
        ],
    }
