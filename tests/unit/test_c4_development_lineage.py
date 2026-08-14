"""Direct tests for hrm_adaptive_memory.c4.development_lineage.

This module is shared between two call sites that must never diverge:
scripts/colab_c4_requalify.py (checked EARLY, before GPU work) and
scripts/certify_c4_run.py's development_lineage gate (checked again,
authoritatively, at certification time). test_c4_development_lineage_gate.py
already exercises this logic thoroughly through the Gate wrapper; these tests
cover the underlying function and its LineageCheck result directly, since it
is a shared library now, not private to the certifier.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.development_lineage import (
    LineageCheck, check_development_lineage)

PROTOCOL_SHA = "b4e22a55" + "0" * 56


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"


@pytest.fixture()
def repo(tmp_path: Path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "protocol.json").write_text("v1\n")
    _git(r, "init", "--quiet")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "--quiet", "-m", "dev commit")
    dev_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r,
                                capture_output=True, text=True).stdout.strip()
    return r, dev_commit


def _cert(repo: Path, dev_commit: str, *, protocol_sha=PROTOCOL_SHA,
         valid_run=True) -> Path:
    cert_dir = repo / "certification"
    cert_dir.mkdir(exist_ok=True)
    path = cert_dir / "CERTIFICATION.json"
    path.write_text(json.dumps({
        "VALID_RUN": valid_run,
        "protocol_sha256": protocol_sha,
        "gates": {"source_lineage": {"detail": {"git": {"head": dev_commit}}}},
    }))
    return path


class TestReturnType:
    def test_result_is_a_lineage_check(self, repo):
        r, dev_commit = repo
        cert_path = _cert(r, dev_commit)
        result = check_development_lineage(
            dev_certification_path=cert_path,
            this_protocol_sha256=PROTOCOL_SHA, repo=r)
        assert isinstance(result, LineageCheck)
        assert result.ok
        assert result.dev_commit == dev_commit
        assert result.dev_protocol_sha256 == PROTOCOL_SHA
        assert result.this_commit == dev_commit

    def test_summary_is_json_serializable(self, repo):
        r, dev_commit = repo
        cert_path = _cert(r, dev_commit)
        result = check_development_lineage(
            dev_certification_path=cert_path,
            this_protocol_sha256=PROTOCOL_SHA, repo=r)
        json.dumps(result.summary())  # must not raise

    def test_summary_has_the_expected_keys(self, repo):
        r, dev_commit = repo
        cert_path = _cert(r, dev_commit)
        result = check_development_lineage(
            dev_certification_path=cert_path,
            this_protocol_sha256=PROTOCOL_SHA, repo=r)
        assert set(result.summary()) == {
            "ok", "violations", "development_protocol_sha256",
            "development_commit", "this_run_protocol_sha256",
            "this_run_commit", "source_drift_from_development",
        }


class TestBothCallSitesGetTheSameAnswer:
    """The property the shared module exists to guarantee."""

    def test_gate_wrapper_and_direct_call_agree(self, repo):
        import importlib.util
        import sys
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_certifier_agree_check", root / "scripts/certify_c4_run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_certifier_agree_check"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("_certifier_agree_check", None)

        r, dev_commit = repo
        cert_path = _cert(r, dev_commit)
        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}

        direct = check_development_lineage(
            dev_certification_path=cert_path,
            this_protocol_sha256=PROTOCOL_SHA, repo=r)
        via_gate = module.gate_development_lineage(manifest, cert_path, repo=r)

        assert direct.ok == via_gate.passed
        assert direct.violations == via_gate.violations
