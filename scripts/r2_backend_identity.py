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

R2-DEV-V2 uses Qwen2.5-7B-Instruct as the policy model. This is a NEW
backend development line, NOT a continuation of the R13 Gemma lineage.
Absolute utilities from R2-DEV-V2 must NOT be compared directly to R13
Gemma results. Within-model contrasts (D vs C0, E vs C0) are valid.

Classification: R2_DECODER_MECHANISM_QUAL_001 + R2_POLICY_BACKEND_V2
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
    grammar_enforcement: str  # "LlamaGrammar.from_json_schema"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def identity_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def has_placeholders(self) -> bool:
        """Check if any field contains a PLACEHOLDER value."""
        for v in asdict(self).values():
            if isinstance(v, str) and "PLACEHOLDER" in v:
                return True
            if isinstance(v, int) and v == 0 and self.gguf_size_bytes == 0:
                return True
        return False


# Schema builder SHA — computed from r2_schema.py at qualification time.
SCHEMA_BUILDER_SHA256 = (
    "c20cd3a5adf976ddce2296ded11e21d1b2d9c972cd94c205404e5d6b410a3b0e"
)


# Frozen pinned identity for R2-DEV-V2.
# Qualified on 2026-08-23 with Qwen2.5-7B-Instruct Q4_K_M on Colab T4.
# All fields are actual observed values — no placeholders.
R2_POLICY_BACKEND_V2 = PinnedBackendIdentity(
    version="R2_POLICY_BACKEND_V2",
    gguf_filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    gguf_sha256="65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423",
    gguf_repository="bartowski/Qwen2.5-7B-Instruct-GGUF",
    gguf_size_bytes=4683074240,
    runtime="llama-cpp-python",
    runtime_version="llama-cpp-python 0.3.35",
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
    grammar_enforcement="LlamaGrammar.from_json_schema",
)


def compute_schema_builder_sha() -> str:
    """Compute SHA256 of r2_schema.py at runtime."""
    schema_path = Path(__file__).resolve().parent / "r2_schema.py"
    return hashlib.sha256(schema_path.read_bytes()).hexdigest()


def compute_gguf_sha256(gguf_path: str) -> tuple[str, int]:
    """Compute SHA256 and size of a GGUF file at runtime.

    Returns (sha256_hex, size_bytes).
    """
    sha = hashlib.sha256()
    size = 0
    with open(gguf_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), size


def get_runtime_version() -> str:
    """Get the llama-cpp-python runtime version at runtime."""
    try:
        import llama_cpp
        return f"llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')}"
    except ImportError:
        return "llama-cpp-python (not installed)"


def verify_pinned_identity(
    gguf_path: str | None = None,
) -> dict:
    """Verify that the runtime environment matches the pinned identity.

    Checks:
      - GGUF SHA matches (if gguf_path provided)
      - GGUF size matches (if gguf_path provided)
      - Schema builder SHA matches
      - Runtime version matches
      - No placeholders in identity

    Returns a dict with per-check pass/fail and overall status.
    """
    pinned = R2_POLICY_BACKEND_V2
    results = {
        "pinned_identity_sha256": pinned.identity_sha256(),
        "has_placeholders": pinned.has_placeholders(),
        "checks": {},
        "overall_passed": True,
    }

    # Q8: No placeholders
    q8_passed = not pinned.has_placeholders()
    results["checks"]["q8_no_placeholders"] = q8_passed
    results["overall_passed"] &= q8_passed

    # Q10: Schema builder SHA
    actual_schema_sha = compute_schema_builder_sha()
    q10_passed = actual_schema_sha == pinned.schema_builder_sha256
    results["checks"]["q10_schema_builder_sha"] = {
        "passed": q10_passed,
        "expected": pinned.schema_builder_sha256,
        "actual": actual_schema_sha,
    }
    results["overall_passed"] &= q10_passed

    # Q11: Runtime version
    actual_runtime = get_runtime_version()
    q11_passed = actual_runtime == pinned.runtime_version
    results["checks"]["q11_runtime_version"] = {
        "passed": q11_passed,
        "expected": pinned.runtime_version,
        "actual": actual_runtime,
    }
    results["overall_passed"] &= q11_passed

    # Q9: GGUF SHA and size (if path provided)
    if gguf_path:
        actual_sha, actual_size = compute_gguf_sha256(gguf_path)
        q9_sha_passed = actual_sha == pinned.gguf_sha256
        q9_size_passed = actual_size == pinned.gguf_size_bytes
        results["checks"]["q9_gguf_sha256"] = {
            "passed": q9_sha_passed,
            "expected": pinned.gguf_sha256,
            "actual": actual_sha,
        }
        results["checks"]["q9_gguf_size"] = {
            "passed": q9_size_passed,
            "expected": pinned.gguf_size_bytes,
            "actual": actual_size,
        }
        results["overall_passed"] &= q9_sha_passed and q9_size_passed
    else:
        results["checks"]["q9_gguf_sha256"] = {"skipped": True, "passed": True}
        results["checks"]["q9_gguf_size"] = {"skipped": True, "passed": True}

    return results
