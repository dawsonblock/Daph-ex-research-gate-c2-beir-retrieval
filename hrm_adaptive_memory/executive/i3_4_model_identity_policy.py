"""Hosted-model identity policy for I3.4.1.

A pinned model is not just a model name.  It is a tuple of:
- provider
- model name
- model revision / fingerprint (if available)
- thinking mode
- generation config hash

The experiment must record the provider's reported model and system
fingerprint on every call.  If the provider does not expose a stable
revision, the experiment must use the system_fingerprint as the revision
proxy and state this explicitly.

If the fingerprint changes within a pair, the pair is invalid (see
i3_4_pair_scheduler.check_pair_fingerprints).

If the fingerprint changes across phases, the experiment identity is
broken and the run is VOID.

Schema identity: ``DAPH_V2B_I3_4_MODEL_IDENTITY_POLICY_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .i3_4_generation_config import FROZEN_CONFIG

IDENTITY_POLICY_SCHEMA = "DAPH_V2B_I3_4_MODEL_IDENTITY_POLICY_V1"
IDENTITY_POLICY_VERSION = 1


@dataclass(frozen=True)
class ModelIdentityPolicy:
    """Policy for verifying hosted-model identity across calls.

    The model is considered pinned if:
    1. The provider-reported model name matches the frozen model name.
    2. The system fingerprint is present on every call.
    3. The fingerprint does not change within a counterbalanced pair.
    4. The fingerprint does not change across phases of the same experiment.
    """

    frozen_model: str
    frozen_provider: str
    thinking_mode: str
    generation_config_sha256: str
    revision_source: str  # "system_fingerprint" or "model_revision"
    require_fingerprint: bool = True
    allow_fingerprint_change_across_pairs: bool = True
    allow_fingerprint_change_within_pair: bool = False
    allow_fingerprint_change_across_phases: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": IDENTITY_POLICY_SCHEMA,
            "schema_version": IDENTITY_POLICY_VERSION,
            "frozen_model": self.frozen_model,
            "frozen_provider": self.frozen_provider,
            "thinking_mode": self.thinking_mode,
            "generation_config_sha256": self.generation_config_sha256,
            "revision_source": self.revision_source,
            "require_fingerprint": self.require_fingerprint,
            "allow_fingerprint_change_across_pairs": self.allow_fingerprint_change_across_pairs,
            "allow_fingerprint_change_within_pair": self.allow_fingerprint_change_within_pair,
            "allow_fingerprint_change_across_phases": self.allow_fingerprint_change_across_phases,
        }

    def verify_call(
        self,
        reported_model: str | None,
        system_fingerprint: str | None,
    ) -> tuple[bool, str]:
        """Verify one call against the identity policy.

        Returns (valid, reason).
        """
        if reported_model is not None and reported_model != self.frozen_model:
            return False, f"Model mismatch: expected {self.frozen_model}, got {reported_model}"
        if self.require_fingerprint and system_fingerprint is None:
            return False, "Fingerprint required but not provided"
        return True, "OK"

    def verify_pair(
        self,
        first_fingerprint: str | None,
        second_fingerprint: str | None,
    ) -> tuple[bool, str]:
        """Verify that fingerprints match within a pair.

        If require_fingerprint is True, a missing fingerprint invalidates
        the pair.  If require_fingerprint is False, missing fingerprints
        are acceptable (no evidence of drift).
        """
        if self.require_fingerprint:
            if first_fingerprint is None:
                return False, "Fingerprint required but missing on first call"
            if second_fingerprint is None:
                return False, "Fingerprint required but missing on second call"
        if not self.allow_fingerprint_change_within_pair:
            if first_fingerprint is not None and second_fingerprint is not None:
                if first_fingerprint != second_fingerprint:
                    return False, "Fingerprint changed within pair"
        return True, "OK"

    def verify_phase(
        self,
        phase_fingerprints: list[str | None],
    ) -> tuple[bool, str]:
        """Verify that fingerprints are consistent across phases."""
        if not self.allow_fingerprint_change_across_phases:
            unique = set(f for f in phase_fingerprints if f is not None)
            if len(unique) > 1:
                return False, f"Fingerprint changed across phases: {len(unique)} distinct"
        return True, "OK"


# Frozen policy for the first I3.4 experiment.
# The generation_config_sha256 is bound from the actual FrozenGenerationConfig.
FROZEN_IDENTITY_POLICY = ModelIdentityPolicy(
    frozen_model="deepseek-v4-flash",
    frozen_provider="deepseek",
    thinking_mode="disabled",
    generation_config_sha256=FROZEN_CONFIG.sha256(),
    revision_source="system_fingerprint",
    require_fingerprint=True,
    allow_fingerprint_change_across_pairs=True,
    allow_fingerprint_change_within_pair=False,
    allow_fingerprint_change_across_phases=False,
)


def identity_policy_sha256() -> str:
    """Canonical SHA-256 of the frozen identity policy."""
    return hashlib.sha256(
        json.dumps(FROZEN_IDENTITY_POLICY.as_dict(), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
