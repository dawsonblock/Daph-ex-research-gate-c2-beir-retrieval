"""Source-lineage checks: what counts as dirty, and what does not.

The bug these tests exist to prevent is subtle and total: a certifying run
rewrites *tracked* files under ``evidence/`` (961 frozen packets, the
determinism receipt, dry-run and smoke receipts), so an unscoped
``git status --porcelain`` is guaranteed non-empty by the time certification
runs. Treating that as a dirty tree makes ``VALID_RUN: true`` unreachable by
construction, no matter how good the science is.

Cleanliness must therefore be scoped to the paths that define the revision.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4 import git_state


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, timeout=30)
    assert proc.returncode == 0, f"git {args}: {proc.stderr}"
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A tiny repo shaped like this project: source paths plus evidence/."""
    r = tmp_path / "repo"
    (r / "hrm_adaptive_memory" / "c4").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "configs").mkdir()
    (r / "tests").mkdir()
    (r / "evidence" / "gate_c4").mkdir(parents=True)

    (r / "hrm_adaptive_memory" / "c4" / "mod.py").write_text("x = 1\n")
    (r / "scripts" / "run.py").write_text("print(1)\n")
    (r / "configs" / "protocol.json").write_text("{}\n")
    (r / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (r / "pyproject.toml").write_text("[project]\nname='x'\n")
    (r / "evidence" / "gate_c4" / "receipts.jsonl").write_text('{"a":1}\n')

    _git(r, "init", "--quiet")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "--quiet", "-m", "initial")
    return r


class TestCleanRepo:
    def test_clean_tree_is_clean(self, repo):
        state = git_state.inspect(repo)
        assert state.is_repo
        assert state.source_clean
        assert state.source_changes == []
        assert state.output_changes == []
        assert state.other_changes == []

    def test_head_is_resolved(self, repo):
        state = git_state.inspect(repo)
        assert state.head == _git(repo, "rev-parse", "HEAD")
        assert len(state.head) == 40


class TestEvidenceDirtIsNotFatal:
    """The whole point of the scoping."""

    def test_modified_tracked_evidence_stays_clean(self, repo):
        (repo / "evidence" / "gate_c4" / "receipts.jsonl").write_text('{"a":2}\n')
        state = git_state.inspect(repo)
        assert state.source_clean, state.output_changes
        assert len(state.output_changes) == 1

    def test_new_untracked_evidence_stays_clean(self, repo):
        (repo / "evidence" / "gate_c4" / "analysis.json").write_text("{}\n")
        state = git_state.inspect(repo)
        assert state.source_clean
        assert len(state.output_changes) == 1

    def test_many_evidence_changes_stay_clean(self, repo):
        """A real run rewrites hundreds of frozen packets."""
        packets = repo / "evidence" / "gate_c4" / "frozen_packets"
        packets.mkdir()
        for i in range(50):
            (packets / f"p{i}.json").write_text("{}\n")
        state = git_state.inspect(repo)
        assert state.source_clean
        assert len(state.output_changes) == 50


class TestSourceDirtIsFatal:
    @pytest.mark.parametrize("rel", [
        "hrm_adaptive_memory/c4/mod.py",
        "scripts/run.py",
        "configs/protocol.json",
        "tests/test_x.py",
        "pyproject.toml",
    ])
    def test_modified_source_is_dirty(self, repo, rel):
        (repo / rel).write_text("# changed\n")
        state = git_state.inspect(repo)
        assert not state.source_clean
        assert any(rel in c for c in state.source_changes)

    def test_new_untracked_source_is_dirty(self, repo):
        (repo / "scripts" / "patch.py").write_text("# hotfix\n")
        state = git_state.inspect(repo)
        assert not state.source_clean
        assert any("scripts/patch.py" in c for c in state.source_changes)

    def test_deleted_source_is_dirty(self, repo):
        (repo / "scripts" / "run.py").unlink()
        state = git_state.inspect(repo)
        assert not state.source_clean

    def test_unclassified_path_is_dirty(self, repo):
        """Anything outside source and evidence still breaks HEAD equivalence."""
        (repo / "README.md").write_text("# hi\n")
        state = git_state.inspect(repo)
        assert not state.source_clean
        assert any("README.md" in c for c in state.other_changes)


class TestPathParsing:
    def test_first_entry_is_not_shifted(self, repo):
        """Regression: stripping whole stdout ate the leading status column.

        ``git status --porcelain`` encodes state in columns 1-2, so a global
        strip() turned ' M scripts/run.py' into 'M scripts/run.py' and every
        path lost its first character -- 'scripts' became 'cripts', which then
        classified as unknown instead of source.
        """
        (repo / "scripts" / "run.py").write_text("# changed\n")
        state = git_state.inspect(repo)
        assert any(c.endswith("scripts/run.py") for c in state.source_changes)
        assert not any("cripts" in c and "scripts" not in c
                       for c in state.source_changes + state.other_changes)

    def test_renamed_source_is_classified_by_destination(self, repo):
        _git(repo, "mv", "scripts/run.py", "scripts/run2.py")
        state = git_state.inspect(repo)
        assert not state.source_clean

    def test_source_and_evidence_are_separated_together(self, repo):
        (repo / "scripts" / "run.py").write_text("# changed\n")
        (repo / "evidence" / "gate_c4" / "receipts.jsonl").write_text('{"a":3}\n')
        state = git_state.inspect(repo)
        assert len(state.source_changes) == 1
        assert len(state.output_changes) == 1
        assert not state.source_clean


class TestNonRepo:
    def test_plain_directory_is_not_a_repo(self, tmp_path):
        state = git_state.inspect(tmp_path)
        assert not state.is_repo
        assert not state.source_clean
        assert "not a git repository" in state.error


class TestRevisionMatching:
    HEAD = "db0e9b335ba0a1b2c3d4e5f60718293a4b5c6d7e"

    def test_full_sha_matches(self):
        assert git_state.revision_matches(self.HEAD, self.HEAD)

    def test_abbreviated_sha_matches(self):
        assert git_state.revision_matches(self.HEAD, "db0e9b3")
        assert git_state.revision_matches(self.HEAD, "db0e9b335ba0")

    def test_case_insensitive(self):
        assert git_state.revision_matches(self.HEAD, "DB0E9B3")

    def test_different_sha_does_not_match(self):
        assert not git_state.revision_matches(self.HEAD, "dd3f0fd")

    def test_too_short_is_rejected(self):
        """A 4-char prefix would match many commits; refuse to guess."""
        assert not git_state.revision_matches(self.HEAD, "db0e")

    def test_empty_inputs_do_not_match(self):
        assert not git_state.revision_matches(None, self.HEAD)
        assert not git_state.revision_matches(self.HEAD, "")

    def test_whitespace_tolerated(self):
        assert git_state.revision_matches(self.HEAD, f"  {self.HEAD}\n")


class TestSharedDefinition:
    def test_certifier_uses_this_modules_source_paths(self):
        """One definition, so the snapshot and the dirty check cannot disagree."""
        import importlib.util
        import sys
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_certifier_paths", root / "scripts/certify_c4_run.py")
        module = importlib.util.module_from_spec(spec)
        # Register before exec: the module defines a @dataclass, and annotation
        # resolution looks the defining module up in sys.modules.
        sys.modules["_certifier_paths"] = module
        try:
            spec.loader.exec_module(module)
            assert module.SOURCE_PATHS is git_state.SOURCE_PATHS
        finally:
            sys.modules.pop("_certifier_paths", None)

    def test_evidence_is_not_a_source_path(self):
        assert "evidence" not in git_state.SOURCE_PATHS
        assert "evidence" in git_state.OUTPUT_PATHS
