"""Tests for the I3.4.1 hosted-model identity policy."""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.executive.i3_4_model_identity_policy import (
    FROZEN_IDENTITY_POLICY, IDENTITY_POLICY_SCHEMA,
    ModelIdentityPolicy, identity_policy_sha256)


def test_identity_policy_schema():
    assert IDENTITY_POLICY_SCHEMA == "DAPH_V2B_I3_4_MODEL_IDENTITY_POLICY_V1"


def test_frozen_policy_uses_system_fingerprint():
    assert FROZEN_IDENTITY_POLICY.revision_source == "system_fingerprint"
    assert FROZEN_IDENTITY_POLICY.require_fingerprint is True


def test_frozen_policy_disallows_within_pair_change():
    assert FROZEN_IDENTITY_POLICY.allow_fingerprint_change_within_pair is False


def test_frozen_policy_disallows_across_phase_change():
    assert FROZEN_IDENTITY_POLICY.allow_fingerprint_change_across_phases is False


def test_frozen_policy_binds_generation_config_hash():
    """The identity policy must bind the actual generation config hash, not empty string."""
    from hrm_adaptive_memory.executive.i3_4_generation_config import FROZEN_CONFIG
    assert FROZEN_IDENTITY_POLICY.generation_config_sha256 != ""
    assert FROZEN_IDENTITY_POLICY.generation_config_sha256 == FROZEN_CONFIG.sha256()


def test_identity_policy_has_sha256():
    h = identity_policy_sha256()
    assert len(h) == 64


def test_verify_call_model_match():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_call(
        reported_model="deepseek-v4-flash",
        system_fingerprint="fp_abc")
    assert valid
    assert reason == "OK"


def test_verify_call_model_mismatch():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_call(
        reported_model="deepseek-v4-pro",
        system_fingerprint="fp_abc")
    assert not valid
    assert "mismatch" in reason.lower()


def test_verify_call_missing_fingerprint():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_call(
        reported_model="deepseek-v4-flash",
        system_fingerprint=None)
    assert not valid
    assert "fingerprint" in reason.lower()


def test_verify_pair_match():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_pair("fp_abc", "fp_abc")
    assert valid


def test_verify_pair_mismatch():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_pair("fp_abc", "fp_xyz")
    assert not valid
    assert "within pair" in reason.lower()


def test_verify_pair_none_is_invalid_when_required():
    """If require_fingerprint is True, a missing fingerprint invalidates the pair."""
    valid, reason = FROZEN_IDENTITY_POLICY.verify_pair(None, "fp_abc")
    assert not valid
    assert "missing" in reason.lower()


def test_verify_phase_consistent():
    valid, _ = FROZEN_IDENTITY_POLICY.verify_phase(["fp_abc", "fp_abc", "fp_abc"])
    assert valid


def test_verify_phase_inconsistent():
    valid, reason = FROZEN_IDENTITY_POLICY.verify_phase(["fp_abc", "fp_xyz"])
    assert not valid
    assert "across phases" in reason.lower()


def test_verify_phase_ignores_none():
    valid, _ = FROZEN_IDENTITY_POLICY.verify_phase(["fp_abc", None, "fp_abc"])
    assert valid


def test_policy_is_immutable():
    with pytest.raises(Exception):
        FROZEN_IDENTITY_POLICY.frozen_model = "other"  # type: ignore
