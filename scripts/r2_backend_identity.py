#!/usr/bin/env python3
"""
R2_POLICY_BACKEND_V2 — Pinned Backend Identity.

Records the exact backend identity for R2-DEV-V2:
    - GGUF filename and full SHA256
    - llama.cpp/llama-cpp-python version or commit
    - CUDA/runtime configuration
    - context size, temperature, max tokens
    - schema-builder SHA

This module is frozen after qualification. Any change to the backend
identity requires a new qualification run and a new version number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class PinnedBackendIdentity:
    """Frozen backend identity for R2-DEV-V2."""

    version: str  # "R2_POLICY_BACKEND_V2"
    gguf_filename: str
    gguf_sha256: str
    gguf_repository: str
    gguf_size_bytes: int
    runtime: str  # "llama-cpp-python" or "llama.cpp"
    runtime_version: str
    cuda_enabled: bool
    gpu_model: str
    context_size: int
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    repeat_penalty: float
    seed: int
    schema_builder_sha256: str  # SHA of r2_schema.py
    decoder_mode: str  # "strict"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def identity_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


# Schema builder SHA — computed from r2_schema.py at qualification time.
# This is set after the qualification run and frozen.
# To compute: hashlib.sha256(Path("scripts/r2_schema.py").read_bytes()).hexdigest()
SCHEMA_BUILDER_SHA256 = (
    "c20cd3a5adf976ddce2296ded11e21d1b2d9c972cd94c205404e5d6b410a3b0e"
)


# Default pinned identity for R2-DEV-V2.
# Fields marked "PLACEHOLDER" must be filled after the live qualification run.
R2_POLICY_BACKEND_V2 = PinnedBackendIdentity(
    version="R2_POLICY_BACKEND_V2",
    gguf_filename="gemma-3-12b-it-qat-q4_0.gguf",
    gguf_sha256="PLACEHOLDER_FILL_AFTER_QUALIFICATION",
    gguf_repository="google/gemma-3-12b-it-qat-q4_0-gguf",
    gguf_size_bytes=0,  # fill after qualification
    runtime="llama-cpp-python",
    runtime_version="PLACEHOLDER_FILL_AFTER_QUALIFICATION",
    cuda_enabled=True,
    gpu_model="Tesla T4",
    context_size=4096,
    temperature=0.0,
    max_tokens=128,
    top_p=1.0,
    top_k=40,
    repeat_penalty=1.0,
    seed=42,
    schema_builder_sha256=SCHEMA_BUILDER_SHA256,
    decoder_mode="strict",
)


def compute_schema_builder_sha() -> str:
    """Compute SHA256 of r2_schema.py."""
    schema_path = Path(__file__).resolve().parent / "r2_schema.py"
    return hashlib.sha256(schema_path.read_bytes()).hexdigest()


def fill_pinned_identity(
    gguf_sha256: str,
    gguf_size_bytes: int,
    runtime_version: str,
    schema_builder_sha256: str | None = None,
) -> PinnedBackendIdentity:
    """Fill in the placeholder fields after live qualification.

    Returns a new PinnedBackendIdentity with the actual values.
    """
    if schema_builder_sha256 is None:
        schema_builder_sha256 = compute_schema_builder_sha()

    return PinnedBackendIdentity(
        version="R2_POLICY_BACKEND_V2",
        gguf_filename="gemma-3-12b-it-qat-q4_0.gguf",
        gguf_sha256=gguf_sha256,
        gguf_repository="google/gemma-3-12b-it-qat-q4_0-gguf",
        gguf_size_bytes=gguf_size_bytes,
        runtime="llama-cpp-python",
        runtime_version=runtime_version,
        cuda_enabled=True,
        gpu_model="Tesla T4",
        context_size=4096,
        temperature=0.0,
        max_tokens=128,
        top_p=1.0,
        top_k=40,
        repeat_penalty=1.0,
        seed=42,
        schema_builder_sha256=schema_builder_sha256,
        decoder_mode="strict",
    )
