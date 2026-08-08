"""Tests for D5 (family regression): analyze_gate_c4.family_regression().

D5's protocol wording is "no catastrophic collapse in any important family."
Frozen before any qualification run: task-structure family (not entity_regime),
-0.10 absolute Q point-estimate delta, not CI-based -- see RESEARCH_STATUS.json
for the decision record and why (development has too few tasks per family,
~12, for a per-family CI to mean much).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_analyzer_family_regression", ROOT / "scripts/analyze_gate_c4.py")
analyzer = importlib.util.module_from_spec(_spec)
sys.modules["_analyzer_family_regression"] = analyzer
_spec.loader.exec_module(analyzer)


def _receipt(task_id: str, family: str, quality: float) -> dict:
    return {
        "task_id": task_id,
        "evaluator_annotation": {
            "quality": quality,
            "correct": quality >= 0.5,
            "family": family,
            "source_cluster_id": "cluster_a",
        },
    }


def _arms(deltas: dict[str, list[float]], base: float = 0.2) -> dict:
    """Build C4_0/C4_4 receipts: one family per key, one task per delta."""
    c4_0, c4_4 = [], []
    for fam, fam_deltas in deltas.items():
        for i, d in enumerate(fam_deltas):
            tid = f"{fam}-{i}"
            c4_0.append(_receipt(tid, fam, base))
            c4_4.append(_receipt(tid, fam, base + d))
    return {"C4_0": c4_0, "C4_4": c4_4}


class TestDefaultThreshold:
    def test_default_threshold_is_the_frozen_value(self):
        assert analyzer.D5_FAMILY_REGRESSION_THRESHOLD == -0.10


class TestSafeCases:
    def test_all_families_improve_is_safe(self):
        arms = _arms({"fam_a": [0.2, 0.2], "fam_b": [0.3, 0.1]})
        result = analyzer.family_regression(arms)
        assert result["safe"]
        assert result["regressed_families"] == []

    def test_small_regression_within_threshold_is_safe(self):
        """The actual observed development regression (-0.029) must pass."""
        arms = _arms({"canonical_like": [-0.029, -0.029]})
        result = analyzer.family_regression(arms)
        assert result["safe"], result

    def test_regression_exactly_at_threshold_boundary(self):
        """threshold is a strict '<' comparison: exactly -0.10 must NOT fail."""
        arms = _arms({"fam_a": [-0.10, -0.10]})
        result = analyzer.family_regression(arms)
        assert result["per_family"]["fam_a"]["mean_delta"] == pytest.approx(-0.10)
        assert not result["per_family"]["fam_a"]["regressed"]
        assert result["safe"]


class TestUnsafeCases:
    def test_one_family_below_threshold_is_unsafe(self):
        arms = _arms({"fam_a": [0.2, 0.2], "fam_b": [-0.15, -0.15]})
        result = analyzer.family_regression(arms)
        assert not result["safe"]
        assert result["regressed_families"] == ["fam_b"]
        assert result["per_family"]["fam_b"]["mean_delta"] == pytest.approx(-0.15)

    def test_worst_family_is_the_most_negative(self):
        arms = _arms({"fam_a": [-0.05], "fam_b": [-0.20], "fam_c": [0.1]})
        result = analyzer.family_regression(arms)
        assert result["worst_family"] == "fam_b"
        assert result["worst_delta"] == pytest.approx(-0.20)

    def test_multiple_regressed_families_all_reported(self):
        arms = _arms({"fam_a": [-0.15], "fam_b": [-0.20], "fam_c": [0.1]})
        result = analyzer.family_regression(arms)
        assert set(result["regressed_families"]) == {"fam_a", "fam_b"}


class TestCustomThreshold:
    def test_custom_threshold_is_respected(self):
        arms = _arms({"fam_a": [-0.06]})
        assert analyzer.family_regression(arms, threshold=-0.10)["safe"]
        assert not analyzer.family_regression(arms, threshold=-0.05)["safe"]


class TestMissingArms:
    def test_missing_c4_0_returns_none(self):
        result = analyzer.family_regression({"C4_4": [_receipt("t", "f", 0.2)]})
        assert result is None

    def test_missing_c4_4_returns_none(self):
        result = analyzer.family_regression({"C4_0": [_receipt("t", "f", 0.2)]})
        assert result is None

    def test_none_is_never_confused_with_a_safe_empty_result(self):
        """A missing-arms condition must never look like 'zero families
        regressed' -- that would silently pass a gate that should abort."""
        result = analyzer.family_regression({})
        assert result is None
        assert result != {"safe": True, "regressed_families": []}


class TestPerTaskPairing:
    def test_only_common_tasks_are_compared(self):
        c4_0 = [_receipt("t1", "fam_a", 0.2), _receipt("t2", "fam_a", 0.2)]
        c4_4 = [_receipt("t1", "fam_a", 0.4)]  # t2 missing from C4_4
        result = analyzer.family_regression({"C4_0": c4_0, "C4_4": c4_4})
        assert result["per_family"]["fam_a"]["n"] == 1

    def test_family_is_read_from_c4_0_not_c4_4(self):
        """Family membership shouldn't shift with the arm; using C4_0's
        label consistently matters if a receipt were ever mislabeled."""
        c4_0 = [_receipt("t1", "fam_a", 0.2)]
        c4_4 = [dict(_receipt("t1", "fam_a", 0.4),
                    evaluator_annotation={**_receipt("t1", "fam_a", 0.4)
                                          ["evaluator_annotation"],
                                          "family": "fam_b"})]
        result = analyzer.family_regression({"C4_0": c4_0, "C4_4": c4_4})
        assert "fam_a" in result["per_family"]
        assert "fam_b" not in result["per_family"]


class TestAgainstTheRealCertifiedBundle:
    def test_real_development_bundle_passes_d5(self):
        bundle = ROOT / "evidence/gate_c4/full/development"
        if not (bundle / "C4_0.jsonl").is_file():
            pytest.skip("development bundle not present")
        arms = analyzer.load_all(bundle)
        result = analyzer.family_regression(arms)
        assert result is not None
        assert result["safe"], (
            f"worst family {result['worst_family']} at "
            f"{result['worst_delta']:+.4f}")

    def test_real_bundle_has_ten_families(self):
        bundle = ROOT / "evidence/gate_c4/full/development"
        if not (bundle / "C4_0.jsonl").is_file():
            pytest.skip("development bundle not present")
        arms = analyzer.load_all(bundle)
        result = analyzer.family_regression(arms)
        assert len(result["per_family"]) == 10
