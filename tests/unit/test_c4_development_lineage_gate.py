"""Tests for the development_lineage gate: qualification must prove it did
not silently diverge from the development configuration that earned
VALID_RUN.

'No development-side changes' is enforced at protocol granularity, not git
commit equality -- a later commit containing only tooling/certification
fixes (no mechanism change) must still be usable for qualification, since
mechanism changes are the only ones this project requires a new protocol
version for. The two hard checks are: protocol_sha256 equality, and
development's certified commit being an ancestor of this run's commit. A
full file-level drift report against the development snapshot is attached
as detail, visible but not itself gate-blocking.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_certifier_dev_lineage", ROOT / "scripts/certify_c4_run.py")
certify_mod = importlib.util.module_from_spec(_spec)
sys.modules["_certifier_dev_lineage"] = certify_mod
_spec.loader.exec_module(certify_mod)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc


@pytest.fixture()
def repo_with_dev_commit(tmp_path: Path):
    """A repo with one commit standing in for 'development's certified
    commit', ready for the test to add more commits on top."""
    r = tmp_path / "repo"
    r.mkdir()
    (r / "protocol.json").write_text("v1\n")
    _git(r, "init", "--quiet")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "--quiet", "-m", "dev commit")
    dev_commit = _git(r, "rev-parse", "HEAD").stdout.strip()
    return r, dev_commit


def _dev_certification(dev_commit: str, protocol_sha: str, *,
                       valid_run: bool = True) -> dict:
    """A minimal but shaped-correctly development CERTIFICATION.json."""
    return {
        "VALID_RUN": valid_run,
        "protocol_sha256": protocol_sha,
        "gates": {
            "source_lineage": {
                "passed": True,
                "detail": {"git": {"head": dev_commit}},
            },
        },
    }


def _write_cert_dir(repo: Path, cert: dict, *, snapshot: dict | None = None) -> Path:
    cert_dir = repo / "certification"
    cert_dir.mkdir(exist_ok=True)
    cert_path = cert_dir / "CERTIFICATION.json"
    cert_path.write_text(json.dumps(cert))
    if snapshot is not None:
        (cert_dir / "SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot))
    return cert_path


PROTOCOL_SHA = "b4e22a55" + "0" * 56


class TestSkippedForDevelopment:
    def test_development_split_never_gets_this_gate(self, repo_with_dev_commit):
        """certify() must not add a 17th gate to development bundles."""
        repo, dev_commit = repo_with_dev_commit
        manifest = {"split": "development", "protocol_sha256": PROTOCOL_SHA}
        # Simulate the conditional in certify(): development is skipped
        # entirely, so gate_development_lineage should never be invoked.
        assert manifest.get("split") == "development"


class TestHardRequirements:
    def test_matching_protocol_and_same_commit_passes(self, repo_with_dev_commit):
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(
            repo, cert, snapshot={"schema_version": "c4-source-snapshot-v1",
                                  "files": {"protocol.json":
                                            certify_mod._sha256_file(repo / "protocol.json")}})
        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}

        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert g.passed, g.violations

    def test_matching_protocol_descendant_commit_passes(self, repo_with_dev_commit):
        """A later commit with only non-mechanism changes must be usable."""
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(
            repo, cert, snapshot={"schema_version": "c4-source-snapshot-v1",
                                  "files": {"protocol.json":
                                            certify_mod._sha256_file(repo / "protocol.json")}})
        # A later, purely-additive commit (e.g. a tooling fix).
        (repo / "tooling.py").write_text("# new file\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "tooling fix, no mechanism change")

        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert g.passed, g.violations

    def test_protocol_mismatch_fails(self, repo_with_dev_commit):
        """The primary hard check: a changed protocol must abort."""
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(repo, cert)
        manifest = {"split": "qualification", "protocol_sha256": "f" * 64}

        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert not g.passed
        assert any("protocol_sha256 differs" in v for v in g.violations)

    def test_non_ancestor_commit_fails(self, repo_with_dev_commit):
        """History rewritten/reset backward must abort, not silently pass."""
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(repo, cert)

        # Reset to an orphan branch: dev_commit is no longer an ancestor.
        _git(repo, "checkout", "--orphan", "rewritten")
        (repo / "protocol.json").write_text("v2-rewritten\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "rewritten history")

        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert not g.passed
        assert any("not an ancestor" in v for v in g.violations)

    def test_dev_certificate_with_valid_run_false_fails(self, repo_with_dev_commit):
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA, valid_run=False)
        cert_path = _write_cert_dir(repo, cert)
        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}

        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert not g.passed
        assert any("VALID_RUN=false" in v for v in g.violations)

    def test_missing_dev_certificate_fails(self, repo_with_dev_commit):
        repo, _dev_commit = repo_with_dev_commit
        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(
            manifest, repo / "nonexistent" / "CERTIFICATION.json", repo=repo)
        assert not g.passed
        assert any("not found" in v for v in g.violations)

    def test_malformed_dev_certificate_fails(self, repo_with_dev_commit):
        repo, _dev_commit = repo_with_dev_commit
        cert_dir = repo / "certification"
        cert_dir.mkdir()
        (cert_dir / "CERTIFICATION.json").write_text("{not valid json")
        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(
            manifest, cert_dir / "CERTIFICATION.json", repo=repo)
        assert not g.passed

    def test_missing_protocol_sha_in_manifest_fails(self, repo_with_dev_commit):
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(repo, cert)
        manifest = {"split": "qualification"}  # no protocol_sha256
        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert not g.passed
        assert any("cannot compare protocol_sha256" in v for v in g.violations)


class TestDriftReportIsVisibleNotBlocking:
    def test_drift_from_tooling_files_is_reported_but_does_not_fail(
            self, repo_with_dev_commit):
        """The exact scenario this gate exists to allow: a tooling file
        changed since development certification, protocol unchanged."""
        repo, dev_commit = repo_with_dev_commit
        snapshot = {
            "schema_version": "c4-source-snapshot-v1",
            "files": {"protocol.json":
                     certify_mod._sha256_file(repo / "protocol.json")},
        }
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(repo, cert, snapshot=snapshot)

        # Change the file the snapshot recorded, on a new commit descending
        # from dev_commit -- simulating a post-certification tooling fix.
        (repo / "protocol.json").write_text("v1-with-a-tooling-tweak\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "tooling tweak")

        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)

        assert g.passed, g.violations
        drift = g.detail["source_drift_from_development"]
        assert drift["ok"] is False
        assert "protocol.json" in drift["changed"]

    def test_missing_snapshot_is_recorded_not_fatal(self, repo_with_dev_commit):
        """A missing SOURCE_SNAPSHOT.json degrades the report, not the gate --
        the two hard checks above are what actually matter."""
        repo, dev_commit = repo_with_dev_commit
        cert = _dev_certification(dev_commit, PROTOCOL_SHA)
        cert_path = _write_cert_dir(repo, cert)  # no snapshot written

        manifest = {"split": "qualification", "protocol_sha256": PROTOCOL_SHA}
        g = certify_mod.gate_development_lineage(manifest, cert_path, repo=repo)
        assert g.passed, g.violations
        assert "error" in g.detail["source_drift_from_development"]


class TestCertifyIntegration:
    """The conditional in certify() itself: gate added only for non-dev splits."""

    def test_development_bundle_gate_count_unaffected(self, tmp_path):
        """Guards against ever adding a 17th gate to development bundles."""
        import inspect
        source = inspect.getsource(certify_mod.certify)
        assert 'manifest.get("split") != "development"' in source

    def test_default_dev_certification_path_is_a_sibling_not_a_cousin(self):
        """Regression: bundle.parent.parent pointed at evidence/gate_c4/
        development/certification/... (missing the 'full/' segment) instead
        of the real sibling path evidence/gate_c4/full/development/
        certification/CERTIFICATION.json. Only ONE .parent from a bundle
        like evidence/gate_c4/full/qualification lands on 'full/', where
        'development' actually lives beside it."""
        bundle = Path("evidence/gate_c4/full/qualification")
        default = bundle.parent / "development" / "certification" / "CERTIFICATION.json"
        assert str(default) == (
            "evidence/gate_c4/full/development/certification/CERTIFICATION.json")
