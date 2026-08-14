"""I3.4.1 paired experiment runner.

Integrates:
- PinnedModelController (with strict JSON and frozen generation config)
- PairScheduler (deterministic AB/BA call ordering)
- CallReceipts (one per backend attempt)
- Model identity verification

This is the runtime that guarantees call 1 is immediately followed by
call 2 within each counterbalanced pair, and persists their receipts.

Schema identity: ``DAPH_V2B_I3_4_PAIRED_RUNNER_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .i3_4_model_identity_policy import FROZEN_IDENTITY_POLICY, ModelIdentityPolicy
from .i3_4_pair_scheduler import (
    PairFingerprintRecord, PairSchedule, compute_pair_hash,
    is_blind_first, check_pair_fingerprints)
from .model_backend import DeepSeekBackend, ModelBackend, ModelCallResult
from .model_decoder import DecoderOutcome, decode_output
from .model_packet import (
    assert_no_condition_leakage, packet_json, packet_sha256, serialize_packet)
from .model_prompt import SYSTEM_PROMPT
from .pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, CONTROLLER_ID, FAIL_CLOSED_PROPOSAL)
from .actions import ActionProposal
from .metareasoning_controller import ControllerObservation

RUNNER_SCHEMA = "DAPH_V2B_I3_4_PAIRED_RUNNER_V1"
RUNNER_VERSION = 1


@dataclass(frozen=True)
class PairedCallResult:
    """Result of one call within a pair."""

    task_id: str
    condition: str  # "BLIND" or "AWARE"
    call_order: int  # 1 or 2 (within the pair)
    proposal: ActionProposal
    decoder_outcome: DecoderOutcome | None
    call_result: ModelCallResult | None
    backend_error: str | None
    receipt_index: int  # index into backend.call_receipts


@dataclass(frozen=True)
class PairedTaskResult:
    """Result of one counterbalanced pair (blind + aware) for one task."""

    task_id: str
    pair_id: str
    schedule: PairSchedule
    blind_result: PairedCallResult
    aware_result: PairedCallResult
    fingerprint_record: PairFingerprintRecord
    identity_valid: bool
    identity_reason: str


@dataclass
class PairedExperimentRunner:
    """Runs the I3.4.1 paired experiment with full provenance.

    For each task:
    1. The PairScheduler determines whether the order is BLIND→AWARE or AWARE→BLIND.
    2. The first call is made immediately, followed by the second call.
    3. Each call produces a CallReceipt in the backend.
    4. Fingerprints are checked within the pair.
    5. Model identity is verified on every call.
    """

    backend: DeepSeekBackend
    identity_policy: ModelIdentityPolicy = field(
        default_factory=lambda: FROZEN_IDENTITY_POLICY)
    experiment_id: str = "v2b_i3_4_experiment_v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    strict_json: bool = True
    results: list[PairedTaskResult] = field(default_factory=list, repr=False)

    def _make_call(
        self,
        *,
        task_id: str,
        condition: str,
        call_order: int,
        observation: ControllerObservation,
    ) -> PairedCallResult:
        """Make one model call and record the result."""
        # Set backend metadata for the receipt.
        self.backend.task_id = task_id
        self.backend.condition = condition
        self.backend.pair_id = f"{self.experiment_id}:{task_id}"

        # Serialize the observation into a packet.
        packet = serialize_packet(observation)
        assert_no_condition_leakage(packet)
        user_prompt = packet_json(packet)

        receipt_index_before = len(self.backend.call_receipts)

        try:
            call_result = self.backend.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as exc:
            return PairedCallResult(
                task_id=task_id, condition=condition, call_order=call_order,
                proposal=BACKEND_ERROR_PROPOSAL,
                decoder_outcome=None, call_result=None,
                backend_error=str(exc),
                receipt_index=receipt_index_before)

        outcome = decode_output(call_result.raw_output, strict=self.strict_json)
        proposal = outcome.proposal if outcome.valid and outcome.proposal else FAIL_CLOSED_PROPOSAL

        return PairedCallResult(
            task_id=task_id, condition=condition, call_order=call_order,
            proposal=proposal, decoder_outcome=outcome,
            call_result=call_result, backend_error=None,
            receipt_index=receipt_index_before)

    def run_pair(
        self,
        task_id: str,
        blind_observation: ControllerObservation,
        aware_observation: ControllerObservation,
    ) -> PairedTaskResult:
        """Run one counterbalanced pair for one task.

        The scheduler determines the call order (BLIND→AWARE or AWARE→BLIND).
        The second call is made immediately after the first.
        """
        pair_id = f"{self.experiment_id}:{task_id}"
        schedule_hash = compute_pair_hash(self.experiment_id, task_id)
        blind_first = is_blind_first(self.experiment_id, task_id)
        if blind_first:
            schedule = PairSchedule(
                pair_id=pair_id, task_id=task_id,
                pair_order="BLIND_FIRST",
                first_condition="STATE_BLIND_CONTROLLER",
                second_condition="STATE_AWARE_CONTROLLER",
                schedule_hash=schedule_hash)
        else:
            schedule = PairSchedule(
                pair_id=pair_id, task_id=task_id,
                pair_order="AWARE_FIRST",
                first_condition="STATE_AWARE_CONTROLLER",
                second_condition="STATE_BLIND_CONTROLLER",
                schedule_hash=schedule_hash)

        # Execute calls in the scheduled order.
        if schedule.first_condition == "STATE_BLIND_CONTROLLER":
            first = self._make_call(
                task_id=task_id, condition="BLIND", call_order=1,
                observation=blind_observation)
            second = self._make_call(
                task_id=task_id, condition="AWARE", call_order=2,
                observation=aware_observation)
            blind_result = first
            aware_result = second
        else:
            first = self._make_call(
                task_id=task_id, condition="AWARE", call_order=1,
                observation=aware_observation)
            second = self._make_call(
                task_id=task_id, condition="BLIND", call_order=2,
                observation=blind_observation)
            aware_result = first
            blind_result = second

        # Check fingerprints within the pair.
        first_fp = (first.call_result.system_fingerprint
                    if first.call_result else None)
        second_fp = (second.call_result.system_fingerprint
                     if second.call_result else None)
        fp_record = check_pair_fingerprints(
            pair_id=pair_id,
            first_call_fingerprint=first_fp,
            second_call_fingerprint=second_fp,
            require_fingerprint=self.identity_policy.require_fingerprint,
        )

        # Verify model identity on both calls.
        first_model = (first.call_result.model_name
                       if first.call_result else None)
        second_model = (second.call_result.model_name
                        if second.call_result else None)
        id1_valid, id1_reason = self.identity_policy.verify_call(
            first_model, first_fp)
        id2_valid, id2_reason = self.identity_policy.verify_call(
            second_model, second_fp)
        identity_valid = id1_valid and id2_valid and fp_record.pair_valid
        identity_reason = (
            f"first: {id1_reason}; second: {id2_reason}; "
            f"pair: {fp_record.pair_valid}")

        result = PairedTaskResult(
            task_id=task_id, pair_id=pair_id, schedule=schedule,
            blind_result=blind_result, aware_result=aware_result,
            fingerprint_record=fp_record,
            identity_valid=identity_valid, identity_reason=identity_reason)
        self.results.append(result)
        return result

    def all_receipts(self) -> list:
        """Return all call receipts accumulated by the backend."""
        return self.backend.call_receipts

    def runner_summary(self) -> dict[str, Any]:
        """Return a summary of the runner state."""
        return {
            "schema": RUNNER_SCHEMA,
            "schema_version": RUNNER_VERSION,
            "experiment_id": self.experiment_id,
            "controller_id": CONTROLLER_ID,
            "pairs_completed": len(self.results),
            "total_receipts": len(self.backend.call_receipts),
            "identity_policy_sha256": self.identity_policy.sha256(),
            "strict_json": self.strict_json,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
