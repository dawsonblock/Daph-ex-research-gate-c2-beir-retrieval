"""Tests for --split generalization in scripts/colab_c4_requalify.py.

The launcher used to hardcode "development" and "120" throughout: the split
name in nine separate subprocess calls, the task count in print statements
and the determinism check's --tasks value, and the diagnostic-arms step
unconditionally. This generalizes it to development/qualification/ood while
adding the one new requirement qualification and ood actually need: proof
that the run traces back to a certified development configuration, checked
EARLY (before any GPU work) as well as authoritatively at certification.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/colab_c4_requalify.py"


def _load():
    spec = importlib.util.spec_from_file_location("_requalify_split", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_requalify_split"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_requalify_split", None)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def source() -> str:
    return SCRIPT.read_text()


class TestArgParsing:
    def test_split_defaults_to_development(self, mod):
        args = mod.parse_args(["--expected-commit", "abc1234"])
        assert args.split == "development"

    def test_split_accepts_qualification(self, mod):
        args = mod.parse_args(
            ["--expected-commit", "abc1234", "--split", "qualification"])
        assert args.split == "qualification"

    def test_split_accepts_ood(self, mod):
        args = mod.parse_args(
            ["--expected-commit", "abc1234", "--split", "ood"])
        assert args.split == "ood"

    def test_unknown_split_rejected(self, mod):
        with pytest.raises(SystemExit):
            mod.parse_args(
                ["--expected-commit", "abc1234", "--split", "not_a_split"])

    def test_dev_certification_defaults_to_none(self, mod):
        args = mod.parse_args(["--expected-commit", "abc1234"])
        assert args.dev_certification is None

    def test_dev_certification_can_be_overridden(self, mod):
        args = mod.parse_args(
            ["--expected-commit", "abc1234",
             "--dev-certification", "/some/path.json"])
        assert str(args.dev_certification) == "/some/path.json"


class TestTaskCount:
    def test_development_task_count_matches_corpus(self, mod):
        """120 was hardcoded everywhere; this must now be read from disk."""
        path = ROOT / "data/hrm/controlled_gate_a_v4/development/oracle_tasks.jsonl"
        if not path.is_file():
            pytest.skip("development corpus not present")
        expected = sum(1 for l in path.read_text().splitlines() if l.strip())
        assert mod._task_count("development") == expected

    def test_qualification_task_count_differs_from_development(self, mod):
        dev_path = ROOT / "data/hrm/controlled_gate_a_v4/development/oracle_tasks.jsonl"
        qual_path = ROOT / "data/hrm/controlled_gate_a_v4/qualification/oracle_tasks.jsonl"
        if not (dev_path.is_file() and qual_path.is_file()):
            pytest.skip("corpora not present")
        assert mod._task_count("qualification") != mod._task_count("development")
        assert mod._task_count("qualification") > mod._task_count("development")

    def test_missing_split_corpus_returns_zero_not_a_crash(self, mod):
        assert mod._task_count("nonexistent_split") == 0


class TestNoHardcoded120Remains:
    def test_determinism_step_uses_task_count_variable(self, source):
        step7 = source.split("Step 7: Determinism", 1)[1]
        step7 = step7.split("Step 8", 1)[0]
        assert "str(task_count)" in step7
        assert '"--tasks", "120"' not in step7

    def test_full_run_step_passes_split_variable_not_literal(self, source):
        step12 = source.split("Step 12: Full HRM", 1)[1].split("Step 13", 1)[0]
        assert '"--split", split' in step12
        assert '"--split", "development"' not in step12

    def test_freeze_packets_uses_split_variable(self, source):
        assert '"scripts/c4_freeze_packets.py", "--split", split' in source

    def test_dry_run_and_smoke_pass_split(self, source):
        assert '"dry-run", "--split", split' in source
        assert '"smoke", "--split", split' in source


class TestDiagnosticArmsOnlyOnDevelopment:
    def test_step_13_is_conditional_on_is_development(self, source):
        step13 = source.split("Step 13: Diagnostic arms", 1)[1]
        step13 = step13.split("Step 14", 1)[0]
        assert "if not is_development:" in step13
        assert "SKIPPED for --split" in step13

    def test_skip_message_cites_the_runbook_rule(self, source):
        step13 = source.split("Step 13: Diagnostic arms", 1)[1]
        step13 = step13.split("Step 14", 1)[0]
        # Split across two adjacent f-string literals in source, so checked
        # as two fragments rather than one contiguous phrase.
        assert "must not be used to choose between C4_4" in step13
        assert "and C4_4m -- that decision is made on development alone" in step13

    def test_summary_arm_list_excludes_diagnostics_when_not_development(self, source):
        assert "summary_arms = ARMS + (DIAGNOSTIC_ARMS if is_development else [])" in source


class TestEarlyLineageCheck:
    """The fail-fast optimization: check before GPU work, not just at the end."""

    def test_early_check_lives_in_step_2_before_step_3(self, source):
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3: Install dependencies", 1)[0]
        assert "check_development_lineage" in step2
        assert "if not is_development:" in step2

    def test_early_check_aborts_on_lineage_failure(self, source):
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3: Install dependencies", 1)[0]
        assert "lineage.ok" in step2
        assert 'abort("Step 2"' in step2

    def test_early_check_reports_drift_without_failing_on_it(self, source):
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3: Install dependencies", 1)[0]
        assert "informational, not" in step2

    def test_development_split_never_runs_the_lineage_check(self, source):
        """Development has nothing prior to compare itself against."""
        # The check is entirely gated behind `if not is_development:`, which
        # test_early_check_lives_in_step_2_before_step_3 already confirms;
        # this test pins that the gate variable itself is what's used, not
        # e.g. an inverted or unconditional call.
        step2 = source.split("Step 2: Verify the source revision", 1)[1]
        step2 = step2.split("Step 3: Install dependencies", 1)[0]
        guarded = step2.split("if not is_development:", 1)[1]
        assert "check_development_lineage(" in guarded


class TestCertificationGetsDevCertificationFlag:
    def test_dev_certification_passed_when_not_development(self, source):
        step17 = source.split("Step 17: Certification", 1)[1]
        step17 = step17.split("Step 18", 1)[0]
        assert "if not is_development:" in step17
        assert '"--dev-certification", str(dev_cert_path)' in step17


class TestArchiveNamingReflectsSplit:
    def test_stem_includes_split_not_hardcoded_development(self, source):
        step18 = source.split("Step 18: Package", 1)[1]
        assert "split_upper" in step18
        assert "VALID_CONFORMANT_C4_V2_DEVELOPMENT_RESULT\" if valid_run" not in step18


class TestFinalBannerDoesNotOverclaim:
    def test_non_development_banner_forbids_choosing_between_arms(self, source):
        done_section = source.rsplit("# -- Done", 1)[1]
        assert "Do NOT use this split's numbers to choose between" in done_section
