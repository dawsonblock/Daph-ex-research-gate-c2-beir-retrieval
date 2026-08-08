"""Tests for C4 result provenance — manifest and RESULTS.sha256."""
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hrm_adaptive_memory.c4.provenance import (
    build_manifest, write_results_hash, verify_results_hash,
    compute_results_hash, sha256_corpus, _sha256_file,
    write_bundle_hash, verify_bundle_hash, compute_bundle_hash,
    BUNDLE_HASH_FILENAME,
)


def test_build_manifest_contains_all_required_fields():
    """Manifest must include all fields specified in Phase 2."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        manifest = build_manifest(
            repo=repo, mode="full", split="development",
            arm_ids=["C4_0", "C4_1"], task_count=120,
            protocol_sha256="abc123",
            task_corpus_sha256="def456",
            evidence_corpus_sha256="ghi789",
        )
    required = {
        "mode", "split", "arm_ids", "task_count",
        "git_commit", "protocol_sha256",
        "task_corpus_sha256", "evidence_corpus_sha256",
        "hrm_model_id", "hrm_model_revision", "hrm_max_new_tokens",
        "bge_model_id", "bge_revision",
        "pooling", "normalization", "query_prefix",
        "candidate_budget", "packet_budget", "rrf_k",
        "query_policy_version", "bridge_policy_version",
        "identity_policy_version",
        "selector_version", "selector_config_hash",
        "torch_version", "transformers_version",
        "device", "dtype", "batch_size",
        "generation_condition", "python_version", "created_utc",
    }
    assert required.issubset(set(manifest.keys())), (
        f"Missing fields: {required - set(manifest.keys())}"
    )


def test_results_hash_is_deterministic():
    """RESULTS.sha256 must be deterministic for the same files."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.json").write_text(json.dumps({"a": 1}, sort_keys=True))
        (d / "C4_0.jsonl").write_text('{"task_id": "t1"}\n{"task_id": "t2"}\n')
        h1 = compute_results_hash(d)
        h2 = compute_results_hash(d)
        assert h1 == h2


def test_results_hash_changes_when_files_change():
    """RESULTS.sha256 must change when content changes."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.json").write_text(json.dumps({"a": 1}))
        h1 = compute_results_hash(d)
        (d / "manifest.json").write_text(json.dumps({"a": 2}))
        h2 = compute_results_hash(d)
        assert h1 != h2


def test_write_and_verify_results_hash():
    """write_results_hash then verify_results_hash should succeed."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.json").write_text(json.dumps({"a": 1}))
        (d / "C4_0.jsonl").write_text('{"task_id": "t1"}\n')
        write_results_hash(d)
        assert (d / "RESULTS.sha256").exists()
        assert verify_results_hash(d) is True


def test_verify_results_hash_detects_tampering():
    """verify_results_hash should fail after tampering."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.json").write_text(json.dumps({"a": 1}))
        (d / "C4_0.jsonl").write_text('{"task_id": "t1"}\n')
        write_results_hash(d)
        # Tamper
        (d / "manifest.json").write_text(json.dumps({"a": 999}))
        assert verify_results_hash(d) is False


def test_results_hash_excludes_itself():
    """RESULTS.sha256 should not hash itself."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "manifest.json").write_text(json.dumps({"a": 1}))
        write_results_hash(d)
        # Writing again should produce the same hash (RESULTS.sha256 excluded)
        h1 = (d / "RESULTS.sha256").read_text()
        write_results_hash(d)
        h2 = (d / "RESULTS.sha256").read_text()
        assert h1 == h2


def test_sha256_corpus_returns_empty_for_missing_file():
    assert sha256_corpus(Path("/nonexistent/file.jsonl")) == ""


# --- BUNDLE.sha256: the root hash covering everything, including
# certification/, which RESULTS.sha256 was never scoped to cover. ---

def _certified_bundle(d: Path) -> None:
    """A bundle shaped like a real certified one: results + certification/."""
    (d / "manifest.json").write_text(json.dumps({"a": 1}))
    (d / "C4_0.jsonl").write_text('{"task_id": "t1"}\n')
    write_results_hash(d)
    cert = d / "certification"
    cert.mkdir()
    (cert / "CERTIFICATION.json").write_text(json.dumps({"VALID_RUN": True}))
    (cert / "ENVIRONMENT.lock").write_text(json.dumps({"python": "3.12"}))
    (cert / "source_snapshot.zip").write_bytes(b"not a real zip, just bytes")


def test_bundle_hash_covers_certification_subdirectory():
    """The gap RESULTS.sha256 leaves: certification/ must be covered."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        content = compute_bundle_hash(d)
        assert "certification/CERTIFICATION.json" in content
        assert "certification/ENVIRONMENT.lock" in content
        assert "certification/source_snapshot.zip" in content
        assert "RESULTS.sha256" in content
        assert "C4_0.jsonl" in content


def test_bundle_hash_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        assert compute_bundle_hash(d) == compute_bundle_hash(d)


def test_write_and_verify_bundle_hash():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        write_bundle_hash(d)
        assert (d / BUNDLE_HASH_FILENAME).exists()
        assert verify_bundle_hash(d) is True


def test_bundle_hash_detects_tampering_in_certification_dir():
    """The exact gap being closed: a change under certification/ must be caught."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        write_bundle_hash(d)
        assert verify_bundle_hash(d) is True

        (d / "certification" / "CERTIFICATION.json").write_text(
            json.dumps({"VALID_RUN": False}))
        assert verify_bundle_hash(d) is False, (
            "RESULTS.sha256 alone would miss this -- it never covered "
            "certification/, which is exactly the gap BUNDLE.sha256 closes")


def test_bundle_hash_excludes_itself():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        write_bundle_hash(d)
        h1 = (d / BUNDLE_HASH_FILENAME).read_text()
        write_bundle_hash(d)
        h2 = (d / BUNDLE_HASH_FILENAME).read_text()
        assert h1 == h2

    assert BUNDLE_HASH_FILENAME not in h1


def test_missing_bundle_hash_file_fails_verification():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        assert verify_bundle_hash(d) is False


def test_bundle_hash_written_after_results_hash_still_covers_it():
    """Writing order matters: BUNDLE.sha256 must be written LAST so it can
    cover RESULTS.sha256 and CERTIFICATION.json, which is exactly why
    certify_c4_run.py calls write_bundle_hash() after every other artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _certified_bundle(d)
        content = compute_bundle_hash(d)
        assert "RESULTS.sha256" in content
        assert "certification/CERTIFICATION.json" in content
