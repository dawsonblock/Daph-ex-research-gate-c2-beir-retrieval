"""Tests for manifest arm_ids merging across repeated run_full() invocations.

The bug: colab_c4_requalify.py calls ``run_gate_c4.py full`` twice against the
same output directory -- once for the primary seven-arm ladder, once for the
diagnostic arms C4_3o/C4_4m -- and run_full() wrote manifest.json
unconditionally each time. The second call's write clobbered the first's
entirely, so the final on-disk manifest recorded ``arm_ids: ["C4_3o",
"C4_4m"]`` even though the bundle directory held receipts for all nine arms.

This never affected a certification verdict: certify_c4_run.py reconstructs
arm completeness from the actual ``C4_*.jsonl`` files present in the bundle,
never from manifest.json's arm_ids field. But the manifest is supposed to be a
readable record of what the bundle contains, and after the second call it
wasn't one -- confirmed against the certified 2026-08-08 RunPod bundle, whose
committed manifest.json shows exactly this clobbered state.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_run_gate_c4():
    spec = importlib.util.spec_from_file_location(
        "_run_gate_c4_manifest_test", ROOT / "scripts/run_gate_c4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_gate_c4_manifest_test"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_run_gate_c4_manifest_test", None)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_run_gate_c4()


class TestMergedArmManifestFields:
    def test_no_prior_manifest_returns_just_this_call(self, mod, tmp_path):
        fields = mod._merged_arm_manifest_fields(tmp_path, ["C4_0", "C4_1"])
        assert fields["arm_ids"] == ["C4_0", "C4_1"]
        assert fields["primary_arm_ids"] == ["C4_0", "C4_1"]
        assert fields["diagnostic_arm_ids"] == []

    def test_second_call_merges_with_first(self, mod, tmp_path):
        """The exact regression: primary ladder, then diagnostic arms."""
        first = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)
        (tmp_path / "manifest.json").write_text(json.dumps(first))

        second = mod._merged_arm_manifest_fields(tmp_path, ["C4_3o", "C4_4m"])

        assert second["primary_arm_ids"] == mod.PRIMARY_ORDER
        assert second["diagnostic_arm_ids"] == ["C4_3o", "C4_4m"]
        assert set(second["arm_ids"]) == set(mod.PRIMARY_ORDER) | {"C4_3o", "C4_4m"}
        # The bug this pins: arm_ids must NOT collapse to just the second call.
        assert second["arm_ids"] != ["C4_3o", "C4_4m"]

    def test_reversed_call_order_still_merges(self, mod, tmp_path):
        """Diagnostic arms run first, then primary -- union must not care."""
        first = mod._merged_arm_manifest_fields(tmp_path, ["C4_3o", "C4_4m"])
        (tmp_path / "manifest.json").write_text(json.dumps(first))

        second = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)

        assert second["primary_arm_ids"] == mod.PRIMARY_ORDER
        assert second["diagnostic_arm_ids"] == ["C4_3o", "C4_4m"]

    def test_arm_ids_ordering_is_canonical_not_insertion_order(self, mod, tmp_path):
        """Output order follows PRIMARY_ORDER/DIAGNOSTIC_ORDER, not call order."""
        first = mod._merged_arm_manifest_fields(
            tmp_path, ["C4_4m", "C4_6", "C4_0"])
        assert first["arm_ids"] == ["C4_0", "C4_6", "C4_4m"]

    def test_repeated_identical_call_is_idempotent(self, mod, tmp_path):
        first = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)
        (tmp_path / "manifest.json").write_text(json.dumps(first))
        second = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)
        assert second == first

    def test_missing_manifest_file_does_not_crash(self, mod, tmp_path):
        assert not (tmp_path / "manifest.json").exists()
        fields = mod._merged_arm_manifest_fields(tmp_path, ["C4_0"])
        assert fields["arm_ids"] == ["C4_0"]

    def test_malformed_prior_manifest_does_not_crash(self, mod, tmp_path):
        (tmp_path / "manifest.json").write_text("not valid json{{{")
        fields = mod._merged_arm_manifest_fields(tmp_path, ["C4_0"])
        assert fields["arm_ids"] == ["C4_0"]

    def test_three_sequential_calls_each_persisted_keep_merging(self, mod, tmp_path):
        """Three calls, each persisted before the next -- matching how
        run_full() actually uses this (compute, write, next invocation reads
        what was written). A partial re-run of just one diagnostic arm at a
        time must not lose what came before it."""
        first = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)
        (tmp_path / "manifest.json").write_text(json.dumps(first))

        second = mod._merged_arm_manifest_fields(tmp_path, ["C4_3o"])
        (tmp_path / "manifest.json").write_text(json.dumps(second))
        assert second["diagnostic_arm_ids"] == ["C4_3o"]
        assert second["primary_arm_ids"] == mod.PRIMARY_ORDER

        third = mod._merged_arm_manifest_fields(tmp_path, ["C4_4m"])
        assert third["diagnostic_arm_ids"] == ["C4_3o", "C4_4m"]
        assert third["primary_arm_ids"] == mod.PRIMARY_ORDER


class TestAgainstTheCommittedRegressionCase:
    """Reproduce the exact clobbered state found in the certified bundle."""

    def test_clobbered_manifest_is_what_the_bug_produced(self, mod, tmp_path):
        # Simulate the OLD (buggy) behavior: plain overwrite, no merge.
        (tmp_path / "manifest.json").write_text(
            json.dumps({"arm_ids": mod.PRIMARY_ORDER}))
        (tmp_path / "manifest.json").write_text(
            json.dumps({"arm_ids": ["C4_3o", "C4_4m"]}))
        old_buggy_result = json.loads((tmp_path / "manifest.json").read_text())
        assert old_buggy_result["arm_ids"] == ["C4_3o", "C4_4m"], (
            "this reproduces the exact state found in the committed "
            "evidence/gate_c4/full/development/manifest.json before the fix")

    def test_fixed_path_recovers_full_arm_list_from_the_same_inputs(
            self, mod, tmp_path):
        first = mod._merged_arm_manifest_fields(tmp_path, mod.PRIMARY_ORDER)
        (tmp_path / "manifest.json").write_text(json.dumps(first))
        second = mod._merged_arm_manifest_fields(tmp_path, ["C4_3o", "C4_4m"])
        (tmp_path / "manifest.json").write_text(json.dumps(second))

        final = json.loads((tmp_path / "manifest.json").read_text())
        assert set(final["arm_ids"]) == set(mod.PRIMARY_ORDER) | {"C4_3o", "C4_4m"}
        assert len(final["arm_ids"]) == 9


class TestCertifierIsUnaffectedByTheField:
    """The reason this bug never invalidated a certificate."""

    def test_gate_arms_complete_ignores_manifest_arm_ids(self):
        """certify_c4_run.py's gate_arms_complete reconstructs arm presence
        from PRIMARY_ORDER against the receipts dict it was handed -- built
        by globbing C4_*.jsonl in the bundle -- never from manifest.json."""
        import inspect
        spec = importlib.util.spec_from_file_location(
            "_certifier_manifest_check", ROOT / "scripts/certify_c4_run.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_certifier_manifest_check"] = module
        try:
            spec.loader.exec_module(module)
            source = inspect.getsource(module.gate_arms_complete)
        finally:
            sys.modules.pop("_certifier_manifest_check", None)
        assert "manifest" not in source
