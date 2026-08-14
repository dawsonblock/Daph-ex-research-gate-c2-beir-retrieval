"""Frozen generation configuration for the I3.4.1 hosted-model experiment.

The experiment must explicitly select either thinking or non-thinking mode.
For the first I3.4 experiment, thinking is DISABLED because the scientific
variable is the structured DAPH cognitive state, not uncontrolled hidden
reasoning budget.  This produces a cleaner comparison:

    cognitive-state effect

rather than:

    cognitive-state effect + variable hidden reasoning

Schema identity: ``DAPH_V2B_I3_4_FROZEN_GENERATION_CONFIG_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GENERATION_CONFIG_SCHEMA = "DAPH_V2B_I3_4_FROZEN_GENERATION_CONFIG_V1"
GENERATION_CONFIG_VERSION = 1


@dataclass(frozen=True)
class FrozenGenerationConfig:
    """Explicitly frozen generation configuration.

    The provider default must never be implicit.  Every generation parameter
    that affects model output is explicitly bound here.
    """

    model: str
    thinking_mode: str           # "disabled" or "enabled"
    reasoning_effort: str | None # None when thinking is disabled
    temperature: float | None
    max_tokens: int
    response_format: str         # "json_object" for strict JSON mode
    timeout_seconds: int
    max_retries: int
    retry_policy_id: str         # references the frozen retry policy

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": GENERATION_CONFIG_SCHEMA,
            "schema_version": GENERATION_CONFIG_VERSION,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_policy_id": self.retry_policy_id,
        }

    def sha256(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True,
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


# Frozen configuration for the first I3.4 experiment.
# Thinking is DISABLED to isolate the cognitive-state effect.
FROZEN_CONFIG = FrozenGenerationConfig(
    model="deepseek-v4-flash",
    thinking_mode="disabled",
    reasoning_effort=None,
    temperature=0.0,
    max_tokens=2048,
    response_format="json_object",
    timeout_seconds=120,
    max_retries=3,
    retry_policy_id="v2b_i3_4_retry_policy_v1",
)


def config_sha256() -> str:
    return FROZEN_CONFIG.sha256()


def save_config(path: str | Path) -> str:
    """Write the frozen generation config to *path* and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = FROZEN_CONFIG.as_dict()
    payload["config_sha256"] = FROZEN_CONFIG.sha256()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["config_sha256"]
