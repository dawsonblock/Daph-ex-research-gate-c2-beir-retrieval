"""Notebooks must be launchers, not second implementations.

The C4 notebooks were independent implementations of the run. They drifted from
the tested scripts, were fail-open where the protocol requires an abort, and
self-asserted their own certification. None of that could be caught by pytest,
because notebook cells were not under test.

These tests put the rule under test: active notebooks may invoke scripts, but
they may not contain the logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
SUPERSEDED = NOTEBOOKS / "superseded"

# Symbols that mean scientific logic has leaked into a notebook.
FORBIDDEN_SYMBOLS = (
    "compose_evidence_prompt",
    "run_pre_hrm_stages",
    "run_packet_stage",
    "order_packet",
    "compute_quality",
    "arm_quality",
    "oracle_gap_capture",
    "selector_gap_capture",
    "build_manifest",
    "write_results_hash",
)

# Phrases that indicate a fail-open step.
FAIL_OPEN_PHRASES = (
    "Continuing anyway",
    "continuing anyway",
)


def _active_notebooks() -> list[Path]:
    return sorted(p for p in NOTEBOOKS.glob("*.ipynb"))


def _code_source(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code")


def _all_source(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in nb.get("cells", []))


class TestActiveNotebooks:
    def test_at_least_one_active_notebook(self):
        assert _active_notebooks()

    @pytest.mark.parametrize("name", [p.name for p in _active_notebooks()])
    def test_is_valid_notebook_json(self, name):
        nb = json.loads((NOTEBOOKS / name).read_text())
        assert nb["nbformat"] == 4
        assert isinstance(nb["cells"], list)

    def test_c4_launcher_exists(self):
        assert (NOTEBOOKS / "colab_c4_requalify.ipynb").is_file()

    def test_c4_launcher_invokes_the_authoritative_script(self):
        source = _code_source(NOTEBOOKS / "colab_c4_requalify.ipynb")
        assert "scripts/colab_c4_requalify.py" in source
        assert "scripts/c4_freeze_environment.py" in source

    def test_c4_launcher_has_no_scientific_logic(self):
        source = _code_source(NOTEBOOKS / "colab_c4_requalify.ipynb")
        leaked = [s for s in FORBIDDEN_SYMBOLS if s in source]
        assert not leaked, (
            f"scientific logic leaked into the launcher: {leaked}. Put it in "
            f"scripts/ or hrm_adaptive_memory/ and add a test.")

    def test_c4_launcher_is_not_fail_open(self):
        source = _all_source(NOTEBOOKS / "colab_c4_requalify.ipynb")
        for phrase in FAIL_OPEN_PHRASES:
            assert phrase not in source

    def test_c4_launcher_reads_valid_run_from_the_certificate(self):
        """The notebook must report VALID_RUN, not decide it."""
        source = _code_source(NOTEBOOKS / "colab_c4_requalify.ipynb")
        assert "CERTIFICATION.json" in source
        assert 'cert["VALID_RUN"]' in source
        # It must not compute a verdict itself.
        assert "valid_run_status" not in source
        assert "== expected_protocol_sha" not in source


class TestSupersededNotebooks:
    """Retired paths are preserved, and clearly marked."""

    def test_superseded_directory_exists(self):
        assert SUPERSEDED.is_dir()

    def test_old_notebooks_are_not_in_scripts(self):
        """A second executable path must not sit beside the tested one."""
        stale = sorted((ROOT / "scripts").glob("colab_*.ipynb"))
        assert not stale, f"notebooks still in scripts/: {stale}"
        assert not (ROOT / "scripts/colab_c4_conformant_run.py").exists()

    def test_superseded_files_are_preserved(self):
        """Provenance is part of the evidence chain; do not delete."""
        assert (SUPERSEDED / "colab_c4_requalify_pre_fail_closed.ipynb").is_file()
        assert (SUPERSEDED / "colab_c4_conformant_run_pre_fail_closed.ipynb").is_file()
        assert (SUPERSEDED / "colab_c4_conformant_run_pre_fail_closed.py").is_file()

    @pytest.mark.parametrize("name", [
        "colab_c4_requalify_pre_fail_closed.ipynb",
        "colab_c4_conformant_run_pre_fail_closed.ipynb",
    ])
    def test_superseded_notebook_carries_the_notice(self, name):
        nb = json.loads((SUPERSEDED / name).read_text())
        first = "".join(nb["cells"][0]["source"])
        assert first.startswith("# SUPERSEDED")
        assert "DO NOT USE FOR QUALIFICATION" in first
        assert "scripts/colab_c4_requalify.py" in first

    def test_superseded_script_carries_the_notice(self):
        text = (SUPERSEDED / "colab_c4_conformant_run_pre_fail_closed.py").read_text()
        assert "SUPERSEDED — DO NOT USE FOR QUALIFICATION" in text
        assert "scripts/colab_c4_requalify.py" in text

    def test_superseded_files_keep_their_defects_documented(self):
        """The notice must say what was wrong, not merely that it is old."""
        nb = json.loads(
            (SUPERSEDED / "colab_c4_requalify_pre_fail_closed.ipynb").read_text())
        notice = "".join(nb["cells"][0]["source"])
        for topic in ("Fail-open", "Unpinned", "certification", "Prompt-order"):
            assert topic in notice, topic


class TestNotebookDocumentation:
    def test_readme_exists(self):
        assert (NOTEBOOKS / "README.md").is_file()

    def test_readme_states_the_rule(self):
        text = (NOTEBOOKS / "README.md").read_text()
        assert "no scientific logic" in text.lower()
        assert "scripts/colab_c4_requalify.py" in text

    def test_readme_lists_every_superseded_file(self):
        text = (NOTEBOOKS / "README.md").read_text()
        for path in sorted(SUPERSEDED.iterdir()):
            if path.is_file():
                assert path.name in text, path.name
