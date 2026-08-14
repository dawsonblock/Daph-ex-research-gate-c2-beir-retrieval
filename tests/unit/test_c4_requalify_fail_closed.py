"""The certifying runner must abort on every declared abort condition.

Two classes of defect are pinned here.

**Fail-open steps.** The runner documented itself as fail-closed while
installing dependencies with ``check=False`` and printing "Continuing..." after
an environment mismatch. ``dependency version mismatch`` is a declared abort
condition, and a mismatched environment cannot produce a certifiable run, so
continuing only spends ~35 minutes of GPU time to reach ``VALID_RUN: false``.

**Self-replacing source.** The runner used to clone the repository's default
branch into its own directory, so a session could hold two checkouts at once and
debugging would drift between them. A certification script must not silently
replace its own source tree halfway through execution.

**Scratch files landing in the working tree.** On a non-Colab host (no
``/content``), the requirements-file and archive-output paths fell back to
``Path.cwd()`` -- the repository root, since this script always runs from
there. That plants an untracked file that sits through the entire run and is
only noticed when ``certify_c4_run.py``'s ``source_lineage`` gate re-checks
cleanliness at step 17, failing certification after the GPU work is done
rather than before it starts. Found running against a RunPod pod: step 2
passed on a clean tree, step 3 wrote ``.c4_requirements.txt`` into the repo
root before its pip install failed on an unrelated PEP 668 error, and the
leftover file then made every later run's own step 2 abort on a tree it had
polluted itself.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/colab_c4_requalify.py"


def _load():
    spec = importlib.util.spec_from_file_location("_requalify", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_requalify"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_requalify", None)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def source() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def body() -> str:
    """Executable source, with the module docstring removed.

    The docstring quotes the old fail-open message on purpose, so tests that
    assert the behaviour is gone must not read it as evidence the behaviour
    remains.
    """
    text = SCRIPT.read_text()
    module = ast.parse(text)
    doc = ast.get_docstring(module, clean=False)
    if doc:
        text = text.replace(doc, "", 1)
    return text


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(SCRIPT.read_text())


class TestNoSelfReplacingSource:
    def test_script_does_not_clone(self, body):
        assert '"clone"' not in body
        assert "git clone" not in body

    def test_no_repo_url_or_dir_constants(self, mod):
        assert not hasattr(mod, "REPO_URL")
        assert not hasattr(mod, "REPO_DIR")

    def test_does_not_chdir_away(self, body):
        """It must certify the tree it was invoked in."""
        assert "os.chdir" not in body

    def test_expected_commit_is_required(self, mod):
        with pytest.raises(SystemExit):
            mod.parse_args([])

    def test_expected_commit_is_parsed(self, mod):
        args = mod.parse_args(["--expected-commit", "db0e9b3"])
        assert args.expected_commit == "db0e9b3"


class TestScratchFilesStayOutOfTree:
    """No scratch file may land in the working tree on a non-Colab host."""

    def test_scratch_dir_is_never_cwd(self, mod, tmp_path, monkeypatch):
        """The exact regression: on a host with no /content, the previous
        fallback was Path.cwd() -- the repository root when this script runs,
        which is exactly the tree step 2 (and later, certify_c4_run.py's
        source_lineage gate) must find clean."""
        monkeypatch.chdir(tmp_path)
        result = mod.scratch_dir()
        assert result != tmp_path
        assert result != Path.cwd()

    def test_scratch_dir_is_writable(self, mod):
        result = mod.scratch_dir()
        probe = result / "c4_scratch_dir_probe.tmp"
        probe.write_text("x")
        assert probe.read_text() == "x"
        probe.unlink()

    def test_scratch_dir_prefers_content_when_present(self, mod, tmp_path,
                                                       monkeypatch):
        """Real /content (Colab's convention) takes priority over tempdir."""
        real_path = mod.Path
        fake_content = tmp_path / "content"
        fake_content.mkdir()

        def fake_path_ctor(p="/content"):
            return fake_content if str(p) == "/content" else real_path(p)

        monkeypatch.setattr(mod, "Path", fake_path_ctor)
        assert mod.scratch_dir() == fake_content

    def test_no_raw_content_path_used_for_requirements_or_archive(self, body):
        """Both write sites must go through scratch_dir(), not a literal path."""
        assert 'Path("/content/c4_requirements.txt")' not in body
        assert 'f"/content/{stem}"' not in body

    def test_requirements_and_archive_use_scratch_dir(self, body):
        assert "requirements = scratch_dir()" in body
        assert "zip_base = str(scratch_dir()" in body


class TestDependencyStepsFailClosed:
    def test_locked_dependency_install_aborts_on_failure(self, tree):
        """`pip install -r <lock>` must not run with check=False."""
        installs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "run"
            and any(isinstance(a, ast.Constant)
                    and "Install locked dependencies" == a.value
                    for a in node.args)
        ]
        assert installs, "could not find the locked-dependency install call"
        for call in installs:
            checks = [kw.value for kw in call.keywords if kw.arg == "check"]
            # Either check is absent (defaults True) or explicitly True.
            for value in checks:
                assert isinstance(value, ast.Constant) and value.value is True, \
                    "locked dependency install must abort on failure"

    def test_editable_install_aborts_on_failure(self, tree):
        installs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "run"
            and any(isinstance(a, ast.Constant)
                    and "Install repository (editable)" == a.value
                    for a in node.args)
        ]
        assert installs
        for call in installs:
            for kw in call.keywords:
                if kw.arg == "check":
                    assert kw.value.value is True

    def test_no_continuing_language_remains(self, body):
        for phrase in ("Continuing anyway", "Continuing (the run is still useful"):
            assert phrase not in body

    def test_environment_mismatch_aborts(self, source):
        """Step 4 must abort, not warn."""
        step4 = source.split("Step 4: Verify the environment", 1)[1]
        step4 = step4.split("Step 5", 1)[0]
        assert 'abort("Step 4"' in step4
        assert "VALID_RUN=false until" not in step4

    def test_null_pins_abort_before_gpu_work(self, source):
        step3 = source.split("Step 3: Install dependencies", 1)[1]
        step3 = step3.split("Step 4", 1)[0]
        assert 'abort("Step 3"' in step3
        assert "MUST_RECORD" in step3


class TestRevisionStepFailsClosed:
    def test_dirty_source_aborts(self, source):
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3", 1)[0]
        assert "source_changes" in step2
        assert 'abort("Step 2"' in step2
        # The original fail-open bug: printing this exact WARNING phrase and
        # continuing past a dirty tree instead of aborting. A genuinely
        # different, benign WARNING now also lives in step 2 (a best-effort
        # early protocol read that gracefully defers to step 5's
        # authoritative check on failure) -- that one is fine; this specific
        # phrase, on a dirty tree, is not.
        assert "working tree is dirty. Certification gate" not in step2

    def test_revision_mismatch_aborts(self, source):
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3", 1)[0]
        assert "revision_matches" in step2

    def test_evidence_dirt_is_not_treated_as_fatal(self, source):
        """Steps 7-11 rewrite tracked evidence; that must stay allowed."""
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3", 1)[0]
        assert "output_changes" in step2
        assert "expected" in step2.lower()


class TestAbortSemantics:
    def test_abort_exits_nonzero(self, mod, capsys):
        with pytest.raises(SystemExit) as exc:
            mod.abort("Step X", "because")
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "ABORT at Step X" in out

    def test_run_aborts_by_default(self, mod, capsys):
        with pytest.raises(SystemExit):
            mod.run([sys.executable, "-c", "import sys; sys.exit(3)"],
                    "Failing Step", timeout=60)

    def test_run_can_be_explicitly_nonfatal(self, mod):
        code = mod.run([sys.executable, "-c", "import sys; sys.exit(3)"],
                       "Informational Step", timeout=60, check=False)
        assert code == 3


class TestDeclaredAbortSteps:
    """The steps the protocol names must be the steps that abort."""

    def test_test_suite_aborts(self, source):
        block = source.split("Step 6: Run test suite", 1)[1].split("Step 7", 1)[0]
        assert "check=True" in block

    def test_determinism_aborts(self, source):
        block = source.split("Step 7: Determinism", 1)[1].split("Step 8", 1)[0]
        assert "check=True" in block

    def test_dry_run_aborts(self, source):
        block = source.split("Step 9: CPU-only dry run", 1)[1].split("Step 10", 1)[0]
        assert "check=True" in block

    def test_full_run_aborts(self, source):
        block = source.split("Step 12: Full HRM", 1)[1]
        block = block.split("Step 13", 1)[0]
        assert "check=True" in block

    def test_diagnostic_arms_abort(self, source):
        block = source.split("Step 13: Diagnostic arms", 1)[1].split("Step 14", 1)[0]
        assert "check=True" in block

    def test_analyzer_aborts(self, source):
        block = source.split("Step 14: Analyzer", 1)[1].split("Step 15", 1)[0]
        assert "check=True" in block

    def test_bridge_gate_is_informational(self, source):
        """C4-BRIDGE is a known negative result; it must not abort."""
        block = source.split("Step 10: C4-BRIDGE", 1)[1].split("Step 11", 1)[0]
        assert "check=False" in block
