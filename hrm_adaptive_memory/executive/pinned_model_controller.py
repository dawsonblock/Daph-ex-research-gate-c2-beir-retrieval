"""I3.4 pinned-model executive controller.

``PinnedModelController`` is a condition-agnostic executive controller backed
by a pinned language model.  It contains no ``if/aware/else`` logic and no
condition-specific branching.  The same code path, prompt, serializer, and
decoder run under every observation mask.  Differences between conditions arise
solely from the masked content of the input packet.

The controller:
1. Serializes the ``ControllerObservation`` into a canonical packet.
2. Sends the packet (as JSON) to the pinned model via a ``ModelBackend``.
3. Decodes the model output through the fail-closed ``decode_output`` decoder.
4. Returns a schema-validated ``ActionProposal``.

When the model output is malformed or invalid, the controller returns a
fail-closed ``DEFER`` proposal with reason code ``MODEL_OUTPUT_INVALID`` and
records the rejection in ``last_decoder_outcome`` for development metrics.

Controller identity: ``v2b_i3_4_pinned_model_controller_v1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction

from .actions import ActionProposal
from .metareasoning_controller import ControllerObservation
from .model_backend import ModelBackend, ModelCallResult, StubBackend
from .model_decoder import DecoderOutcome, decode_output
from .model_packet import (
    assert_no_condition_leakage, packet_json, packet_sha256, serialize_packet)
from .model_prompt import SYSTEM_PROMPT

CONTROLLER_ID = "v2b_i3_4_pinned_model_controller_v1"
ALGORITHM_ID = "v2b_i3_4_pinned_model_v1"

# Fail-closed proposal for malformed model output.
FAIL_CLOSED_PROPOSAL = ActionProposal(
    action=DecisionAction.DEFER, reason_code="MODEL_OUTPUT_INVALID", target_id=None)

# Fail-closed proposal for backend/API errors (network, timeout, HTTP error).
BACKEND_ERROR_PROPOSAL = ActionProposal(
    action=DecisionAction.DEFER, reason_code="MODEL_BACKEND_ERROR", target_id=None)


@dataclass
class PinnedModelController:
    """Condition-agnostic pinned-model executive controller.

    The controller is constructed once per run and reused across all tasks and
    conditions.  The ``ModelBackend`` is injected so the same controller code
    works with the real DeepSeek API or a deterministic stub for testing.
    """

    backend: ModelBackend = field(default_factory=StubBackend)
    temperature: float = 0.0
    max_tokens: int = 2048
    # Scientific mode: strict whole-response JSON (no prose extraction).
    strict_json: bool = True
    # Development tracking (mutable, not part of frozen identity)
    last_decoder_outcome: DecoderOutcome | None = field(default=None, repr=False, init=False)
    last_call_result: ModelCallResult | None = field(default=None, repr=False, init=False)
    last_packet_sha256: str | None = field(default=None, repr=False, init=False)
    last_backend_error: str | None = field(default=None, repr=False, init=False)
    _call_count: int = field(default=0, repr=False, init=False)

    controller_id: str = field(default=CONTROLLER_ID, init=False)
    algorithm_id: str = field(default=ALGORITHM_ID, init=False)

    def choose(self, observation: ControllerObservation) -> ActionProposal:
        """Serialize → model → decode → validate.  No condition branching."""
        packet = serialize_packet(observation)
        assert_no_condition_leakage(packet)
        self.last_packet_sha256 = packet_sha256(packet)
        user_prompt = packet_json(packet)
        try:
            call_result = self.backend.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception as exc:
            # Fail closed on any backend error (network, timeout, HTTP, parse).
            # The error is recorded for development metrics; the loop continues.
            self.last_backend_error = str(exc)
            self.last_call_result = None
            self.last_decoder_outcome = None
            self._call_count += 1
            return BACKEND_ERROR_PROPOSAL
        self.last_backend_error = None
        self.last_call_result = call_result
        self._call_count += 1
        outcome = decode_output(call_result.raw_output, strict=self.strict_json)
        self.last_decoder_outcome = outcome
        if outcome.valid and outcome.proposal is not None:
            return outcome.proposal
        # Fail closed: return DEFER so the loop can record the rejection and
        # continue.  The malformed output is preserved in last_decoder_outcome.
        return FAIL_CLOSED_PROPOSAL

    @property
    def call_count(self) -> int:
        return self._call_count

    def development_metrics(self) -> dict[str, Any]:
        """Return development metrics accumulated since construction."""
        outcome = self.last_decoder_outcome
        call = self.last_call_result
        return {
            "controller_id": self.controller_id,
            "call_count": self._call_count,
            "last_packet_sha256": self.last_packet_sha256,
            "last_valid": outcome.valid if outcome else None,
            "last_rejection_code": outcome.rejection_code if outcome else None,
            "last_backend_error": self.last_backend_error,
            "last_prompt_tokens": call.prompt_tokens if call else None,
            "last_completion_tokens": call.completion_tokens if call else None,
            "last_reasoning_tokens": call.reasoning_tokens if call else None,
            "last_latency_ms": call.latency_ms if call else None,
            "last_model_name": call.model_name if call else None,
            "last_system_fingerprint": call.system_fingerprint if call else None,
            "last_finish_reason": call.finish_reason if call else None,
        }
