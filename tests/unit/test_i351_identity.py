"""Tests for I3.5.1 single canonical identity."""
import pytest
import json
from pathlib import Path
from hrm_adaptive_memory.executive.i3_5_1.experiment_identity import (
    ExperimentIdentity, assert_same_experiment_identity,
    IDENTITY_SCHEMA, IDENTITY_VERSION,
)


class TestCanonicalIdentity:
    def test_identity_schema(self):
        assert IDENTITY_SCHEMA == "DAPH_V2B_I3_5_1_EXPERIMENT_IDENTITY_V1"
        assert IDENTITY_VERSION == 1

    def test_identity_sha256_excludes_timestamp(self):
        """The identity hash must not depend on creation time."""
        from hrm_adaptive_memory.executive.i3_5_1.experiment_identity import ExperimentIdentity
        import dataclasses
        # Create two identities that differ only in timestamp
        fields = {
            "experiment_id": "test", "schema": "test", "schema_version": 1,
            "source_commit": "abc", "source_tree_sha256": "def",
            "benchmark_identity": "b", "split_identity": "s",
            "task_corpus_sha256": "t", "scientific_criteria_sha256": "sc",
            "generation_config_sha256": "gc", "model_policy_sha256": "mp",
            "prompt_sha256": "p", "decoder_sha256": "d",
            "runner_sha256": "r", "condition_scheduler_sha256": "cs",
            "packet_serializer_sha256": "ps", "governor_sha256": "g",
            "governor_config_sha256": "gcs", "action_semantics_sha256": "as",
            "executor_sha256": "e", "scoring_sha256": "sc2",
            "statistics_sha256": "st", "oracle_manifest_sha256": "om",
            "observable_oracle_views_sha256": "oov",
            "runtime_environment_identity": "env",
            "dependency_lock_sha256": "dl",
        }
        id1 = ExperimentIdentity(**fields, created_before_first_model_call="2026-01-01T00:00:00Z")
        id2 = ExperimentIdentity(**fields, created_before_first_model_call="2026-02-01T12:00:00Z")
        assert id1.sha256() == id2.sha256(), "Identity hash must exclude timestamp"

    def test_assert_same_identity_passes(self):
        a = {"experiment_identity_sha256": "abc123"}
        b = {"experiment_identity_sha256": "abc123"}
        assert_same_experiment_identity(a, b)  # Should not raise

    def test_assert_same_identity_fails_on_mismatch(self):
        a = {"experiment_identity_sha256": "abc123"}
        b = {"experiment_identity_sha256": "def456"}
        with pytest.raises(ValueError, match="identity mismatch"):
            assert_same_experiment_identity(a, b)

    def test_assert_same_identity_fails_on_missing(self):
        a = {"experiment_identity_sha256": "abc123"}
        b = {"no_sha": "missing"}
        with pytest.raises(ValueError, match="missing experiment_identity_sha256"):
            assert_same_experiment_identity(a, b)

    def test_identity_file_exists(self):
        path = Path("experiments/v2b_i3_5_1/manifests/v2b_i3_5_1_experiment_identity_v1.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "experiment_identity_sha256" in data
        assert data["experiment_identity_sha256"] != ""
        assert data["schema"] == IDENTITY_SCHEMA

    def test_only_one_identity_file(self):
        """There must be exactly one canonical identity in root configs/manifests."""
        # The old I3.5 had two conflicting root identities: configs/ and manifests/.
        # I3.5.1 must have exactly one root canonical identity in manifests/.
        root_identity_files = [
            f for f in Path("experiments/v2b_i3_5_1").rglob("*experiment_identity*.json")
            if "manifests" in f.parts or "configs" in f.parts
        ]
        assert len(root_identity_files) == 1, (
            f"Expected exactly 1 root identity file, found {len(root_identity_files)}: "
            f"{[str(f) for f in root_identity_files]}"
        )
