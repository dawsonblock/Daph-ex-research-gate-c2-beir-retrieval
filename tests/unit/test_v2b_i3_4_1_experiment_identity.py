"""Tests for the I3.4.1 full experiment identity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.executive.i3_4_experiment_identity import (
    IDENTITY_SCHEMA, IDENTITY_VERSION, ExperimentIdentity,
    build_experiment_identity, save_experiment_identity)


ROOT = Path(__file__).resolve().parents[2]


def test_identity_schema_is_frozen():
    assert IDENTITY_SCHEMA == "DAPH_V2B_I3_4_EXPERIMENT_IDENTITY_V1"
    assert IDENTITY_VERSION == 1


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_build_experiment_identity():
    identity = build_experiment_identity(
        observable_oracle_views_sha256="test_views_sha",
        observable_oracle_view_count=34,
        root=ROOT)
    assert identity.experiment_id == "v2b_i3_4_experiment_v1"
    assert identity.schema == IDENTITY_SCHEMA
    assert identity.benchmark_id == "v2b_i3_3_2_scientific_split_v1"
    assert identity.scientific_criteria_version == "V2"
    assert identity.controller_id == "v2b_i3_4_pinned_model_controller_v1"
    assert identity.frozen_model == "deepseek-v4-flash"
    assert identity.frozen_provider == "deepseek"
    assert identity.generation_config_sha256 != ""
    assert identity.model_identity_policy_sha256 != ""
    assert identity.evaluation_task_count == 750


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_identity_sha256_is_deterministic():
    """The identity SHA-256 must be deterministic (excluding timestamp)."""
    id1 = build_experiment_identity(
        observable_oracle_views_sha256="test_sha",
        observable_oracle_view_count=34,
        root=ROOT)
    id2 = build_experiment_identity(
        observable_oracle_views_sha256="test_sha",
        observable_oracle_view_count=34,
        root=ROOT)
    assert id1.sha256() == id2.sha256()


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_identity_sha256_changes_with_different_views():
    """Different oracle views must produce a different identity hash."""
    id1 = build_experiment_identity(
        observable_oracle_views_sha256="sha_A",
        observable_oracle_view_count=34,
        root=ROOT)
    id2 = build_experiment_identity(
        observable_oracle_views_sha256="sha_B",
        observable_oracle_view_count=34,
        root=ROOT)
    assert id1.sha256() != id2.sha256()


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_identity_as_dict_is_serializable():
    identity = build_experiment_identity(
        observable_oracle_views_sha256="test_sha",
        observable_oracle_view_count=34,
        root=ROOT)
    d = identity.as_dict()
    json.dumps(d)  # should not raise


@pytest.mark.skipif(
    not (ROOT / "experiments/v2b_i3_3/oracle_tables/v2b_i3_3_sequential_state_aware_controller_v1.jsonl.gz").exists(),
    reason="Oracle tables not available in lite archive")
def test_identity_binds_all_components():
    """The identity must bind all frozen components."""
    identity = build_experiment_identity(
        observable_oracle_views_sha256="test_sha",
        observable_oracle_view_count=34,
        root=ROOT)
    d = identity.as_dict()
    # All required components must be present
    assert "benchmark" in d
    assert "scientific_criteria" in d
    assert "evaluation_subset" in d
    assert "observable_oracle_views" in d
    assert "controller" in d
    assert "provider_model_policy" in d
    assert "generation_config" in d
    assert "retry_policy" in d
    assert "paired_scheduler" in d
    assert "statistical_implementation" in d
    assert "runtime_environment" in d
    # No component should have an empty hash
    assert d["benchmark"]["benchmark_closure_sha256"] != ""
    assert d["generation_config"]["sha256"] != ""
    assert d["provider_model_policy"]["identity_policy_sha256"] != ""
