"""Controller identity binding for I3.4 pinned-model executive.

Generates and verifies ``controller_identity.json``, which binds every
component that affects model-output reproducibility and evaluation
reproducibility:

- model name / provider / revision (system_fingerprint when available)
- system prompt text and hash
- input packet schema / version / hash
- output schema / version / hash
- serializer implementation / hash
- decoder implementation / hash
- controller implementation / hash
- generation settings
- policy / utility / observation-mask / runtime / corpus references

Model-output reproducibility (does the same prompt produce the same output?)
is distinguished from evaluation reproducibility (does the same output produce
the same trajectory and score?).  The former depends on the model and
generation settings; the latter depends on the deterministic
policy/executor/oracle infrastructure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .model_decoder import OUTPUT_SCHEMA, OUTPUT_SCHEMA_VERSION
from .model_packet import PACKET_SCHEMA, PACKET_SCHEMA_VERSION
from .model_prompt import PROMPT_ID, PROMPT_VERSION, SYSTEM_PROMPT, prompt_sha256
from .pinned_model_controller import ALGORITHM_ID, CONTROLLER_ID

IDENTITY_SCHEMA = "DAPH_V2B_I3_4_CONTROLLER_IDENTITY_V1"


def _source_hash(*path_parts: str) -> str:
    """SHA-256 of a source file's bytes, read from the repo root."""
    path = Path(__file__).parents[2] / Path(*path_parts)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ControllerIdentity:
    """Frozen identity binding for the I3.4 pinned-model controller."""

    schema: str = IDENTITY_SCHEMA
    controller_id: str = CONTROLLER_ID
    algorithm_id: str = ALGORITHM_ID

    # Model-output reproducibility
    model: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: Mapping[str, Any] = field(default_factory=dict)
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    serializer: Mapping[str, Any] = field(default_factory=dict)
    decoder: Mapping[str, Any] = field(default_factory=dict)
    controller_code: Mapping[str, Any] = field(default_factory=dict)
    model_backend: Mapping[str, Any] = field(default_factory=dict)
    generation_settings: Mapping[str, Any] = field(default_factory=dict)

    # Evaluation reproducibility (references to frozen I3.3.2 infrastructure)
    policy_reference: Mapping[str, Any] = field(default_factory=dict)
    utility_reference: Mapping[str, Any] = field(default_factory=dict)
    observation_mask_reference: Mapping[str, Any] = field(default_factory=dict)
    runtime_environment: Mapping[str, Any] = field(default_factory=dict)
    test_corpus_reference: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "controller_id": self.controller_id,
            "algorithm_id": self.algorithm_id,
            "model": dict(self.model),
            "system_prompt": dict(self.system_prompt),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "serializer": dict(self.serializer),
            "decoder": dict(self.decoder),
            "controller_code": dict(self.controller_code),
            "model_backend": dict(self.model_backend),
            "generation_settings": dict(self.generation_settings),
            "policy_reference": dict(self.policy_reference),
            "utility_reference": dict(self.utility_reference),
            "observation_mask_reference": dict(self.observation_mask_reference),
            "runtime_environment": dict(self.runtime_environment),
            "test_corpus_reference": dict(self.test_corpus_reference),
        }

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_identity(
    *,
    model_name: str,
    model_provider: str,
    model_revision: str | None = None,
    system_fingerprint: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    policy_path: str,
    policy_sha256: str,
    utility_path: str,
    utility_sha256: str,
    observation_masks_path: str,
    observation_masks_sha256: str,
    benchmark_manifest_path: str,
    benchmark_manifest_sha256: str,
    python_version: str,
    platform: str,
) -> ControllerIdentity:
    """Build a frozen controller identity binding all reproducibility inputs."""
    return ControllerIdentity(
        model={
            "name": model_name,
            "provider": model_provider,
            "revision": model_revision,
            "system_fingerprint": system_fingerprint,
        },
        system_prompt={
            "prompt_id": PROMPT_ID,
            "prompt_version": PROMPT_VERSION,
            "sha256": prompt_sha256(),
            "char_count": len(SYSTEM_PROMPT),
        },
        input_schema={
            "schema": PACKET_SCHEMA,
            "schema_version": PACKET_SCHEMA_VERSION,
            "serializer_module": "hrm_adaptive_memory.executive.model_packet",
            "serializer_sha256": _source_hash("hrm_adaptive_memory", "executive", "model_packet.py"),
        },
        output_schema={
            "schema": OUTPUT_SCHEMA,
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "decoder_module": "hrm_adaptive_memory.executive.model_decoder",
            "decoder_sha256": _source_hash("hrm_adaptive_memory", "executive", "model_decoder.py"),
        },
        serializer={
            "module": "hrm_adaptive_memory.executive.model_packet",
            "function": "serialize_packet",
            "source_sha256": _source_hash("hrm_adaptive_memory", "executive", "model_packet.py"),
        },
        decoder={
            "module": "hrm_adaptive_memory.executive.model_decoder",
            "function": "decode_output",
            "source_sha256": _source_hash("hrm_adaptive_memory", "executive", "model_decoder.py"),
        },
        controller_code={
            "module": "hrm_adaptive_memory.executive.pinned_model_controller",
            "class": "PinnedModelController",
            "source_sha256": _source_hash("hrm_adaptive_memory", "executive", "pinned_model_controller.py"),
        },
        model_backend={
            "module": "hrm_adaptive_memory.executive.model_backend",
            "deepseek_class": "DeepSeekBackend",
            "source_sha256": _source_hash("hrm_adaptive_memory", "executive", "model_backend.py"),
        },
        generation_settings={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        policy_reference={
            "path": policy_path,
            "sha256": policy_sha256,
        },
        utility_reference={
            "path": utility_path,
            "sha256": utility_sha256,
        },
        observation_mask_reference={
            "path": observation_masks_path,
            "sha256": observation_masks_sha256,
        },
        runtime_environment={
            "python_version": python_version,
            "platform": platform,
        },
        test_corpus_reference={
            "benchmark_manifest_path": benchmark_manifest_path,
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
        },
    )


def save_identity(identity: ControllerIdentity, path: str | Path) -> str:
    """Write the identity to *path* and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.to_dict()
    payload["identity_sha256"] = identity.sha256()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return identity.sha256()


def load_identity(path: str | Path) -> dict[str, Any]:
    """Load and validate a controller identity file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != IDENTITY_SCHEMA:
        raise ValueError(f"unexpected identity schema: {payload.get('schema')}")
    stored_hash = payload.get("identity_sha256")
    if stored_hash is None:
        raise ValueError("identity file missing identity_sha256")
    # Recompute hash from the payload without the stored hash field.
    check_payload = {k: v for k, v in payload.items() if k != "identity_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(check_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed != stored_hash:
        raise ValueError(
            f"identity hash mismatch: stored={stored_hash} recomputed={recomputed}")
    return payload
