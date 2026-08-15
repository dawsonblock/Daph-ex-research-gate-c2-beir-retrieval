"""Tests for C4 evaluator provenance — cryptographic binding of outputs to sources.

Phase 9 of the C4 determinism repair.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.evaluator_manifest import (
    build_evaluator_manifest,
    finalize_evaluator_manifest,
    verify_evaluator_manifest,
    load_and_verify_evaluator_manifest,
    write_evaluator_manifest,
    SCHEMA_VERSION,
)


class TestEvaluatorManifest:
    """Every evaluator output must declare its exact input."""

    def test_manifest_contains_source_hash(self, tmp_path):
        # Create a fake source JSONL
        source = tmp_path / "C4_4.jsonl"
        source.write_text('{"task_id": "t1"}\n')

        manifest = build_evaluator_manifest(
            repo=tmp_path,
            run_id="test-run",
            arm_id="C4_4",
            jsonl_path=source,
            output_dir=tmp_path / "eval_output",
        )

        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["source"]["arm_id"] == "C4_4"
        assert manifest["source"]["jsonl_sha256"] is not None
        assert len(manifest["source"]["jsonl_sha256"]) == 64  # SHA-256 hex

    def test_verify_matching_hash(self, tmp_path):
        source = tmp_path / "C4_4.jsonl"
        source.write_text('{"task_id": "t1"}\n')

        manifest = build_evaluator_manifest(
            repo=tmp_path, run_id="r", arm_id="C4_4",
            jsonl_path=source, output_dir=tmp_path)

        # Should not raise
        verify_evaluator_manifest(manifest, source)

    def test_verify_mismatched_hash_raises(self, tmp_path):
        source_a = tmp_path / "C4_4_a.jsonl"
        source_a.write_text('{"task_id": "t1"}\n')
        source_b = tmp_path / "C4_4_b.jsonl"
        source_b.write_text('{"task_id": "t2"}\n')

        manifest = build_evaluator_manifest(
            repo=tmp_path, run_id="r", arm_id="C4_4",
            jsonl_path=source_a, output_dir=tmp_path)

        # Verifying against the WRONG source must raise
        with pytest.raises(AssertionError, match="provenance mismatch"):
            verify_evaluator_manifest(manifest, source_b)

    def test_load_and_verify_missing_manifest_raises(self, tmp_path):
        source = tmp_path / "C4_4.jsonl"
        source.write_text('{"task_id": "t1"}\n')

        with pytest.raises(AssertionError, match="missing"):
            load_and_verify_evaluator_manifest(
                tmp_path / "nonexistent_manifest.json", source)

    def test_load_and_verify_valid_manifest(self, tmp_path):
        source = tmp_path / "C4_4.jsonl"
        source.write_text('{"task_id": "t1"}\n')

        manifest = build_evaluator_manifest(
            repo=tmp_path, run_id="r", arm_id="C4_4",
            jsonl_path=source, output_dir=tmp_path)

        manifest_path = write_evaluator_manifest(manifest, tmp_path)

        # Should load and verify without raising
        loaded = load_and_verify_evaluator_manifest(manifest_path, source)
        assert loaded["source"]["arm_id"] == "C4_4"

    def test_finalize_adds_output_hashes(self, tmp_path):
        source = tmp_path / "C4_4.jsonl"
        source.write_text('{"task_id": "t1"}\n')
        output = tmp_path / "rescored.jsonl"
        output.write_text('{"task_id": "t1", "quality": 1.0}\n')
        analysis = tmp_path / "analysis.json"
        analysis.write_text('{"mean_quality": 1.0}\n')

        manifest = build_evaluator_manifest(
            repo=tmp_path, run_id="r", arm_id="C4_4",
            jsonl_path=source, output_dir=tmp_path)

        manifest = finalize_evaluator_manifest(
            manifest, output_jsonl_path=output, analysis_path=analysis)

        assert "jsonl_sha256" in manifest["output"]
        assert "analysis_sha256" in manifest["output"]
        assert len(manifest["output"]["jsonl_sha256"]) == 64

    def test_two_different_sources_produce_different_manifests(self, tmp_path):
        """This is the test that would have caught the original bug."""
        source_a = tmp_path / "run_A_C4_4.jsonl"
        source_a.write_text('{"task_id": "t1", "output": "A"}\n')
        source_b = tmp_path / "run_B_C4_4.jsonl"
        source_b.write_text('{"task_id": "t1", "output": "B"}\n')

        manifest_a = build_evaluator_manifest(
            repo=tmp_path, run_id="A", arm_id="C4_4",
            jsonl_path=source_a, output_dir=tmp_path)
        manifest_b = build_evaluator_manifest(
            repo=tmp_path, run_id="B", arm_id="C4_4",
            jsonl_path=source_b, output_dir=tmp_path)

        assert manifest_a["source"]["jsonl_sha256"] != manifest_b["source"]["jsonl_sha256"]

        # An evaluator result bound to source A must NOT verify against source B
        with pytest.raises(AssertionError):
            verify_evaluator_manifest(manifest_a, source_b)
