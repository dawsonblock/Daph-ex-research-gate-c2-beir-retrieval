"""Tests for C4 run certification.

The property under test is that VALID_RUN cannot be true unless every gate was
actually evaluated and passed. Each gate is exercised on artifacts that should
fail it, because the defect being fixed was a certificate whose prerequisites
were string literals:

    "determinism_gate": "PASSED (100% Parity)",
    "all_arms_complete": True,
    "result_hashes_verified": True,
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.packet_ordering import ORDERING_POLICY_ID
from hrm_adaptive_memory.c4.provenance import write_results_hash

ROOT = Path(__file__).resolve().parents[2]

# scripts/ is not a package; load the certifier by path.
_spec = importlib.util.spec_from_file_location(
    "certify_c4_run", ROOT / "scripts/certify_c4_run.py")
certify_mod = importlib.util.module_from_spec(_spec)
sys.modules["certify_c4_run"] = certify_mod
_spec.loader.exec_module(certify_mod)

PRIMARY = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]


# --- Synthetic bundle fixtures ---------------------------------------------

def _receipt(arm_id: str, task_id: str, *, quality: float = 1.0,
             correct: bool = True, selected=("t/identity", "t/value"),
             required=("t/identity", "t/value"), ordering_applied=None,
             prompt_hash: str = "p" * 64, hrm_prompt_hash: str | None = None,
             second_pass: bool = False,
             candidate_pool_hash: str = "c" * 64,
             membership_hash: str = "m" * 64,
             order_hash: str = "o" * 64) -> dict:
    if ordering_applied is None:
        ordering_applied = arm_id in ("C4_4", "C4_3o")
    return {
        "task_id": task_id,
        "arm_id": arm_id,
        "split": "development",
        "runtime_payload": {
            "task_id": task_id,
            "arm_id": arm_id,
            "query": {"second_pass_performed": second_pass},
            "retrieval": {"candidate_ids": list(selected)},
            "identity": {"status": "EXACT"},
            "selection": {"selector": "s2c", "selected_ids": list(selected)},
            "packet": {
                "packet_ids": list(selected),
                "packet_hash": "h" * 64,
                "packet_budget": 6,
                "candidate_pool_hash": candidate_pool_hash,
                "membership_hash": membership_hash,
                "order_hash": order_hash,
                "prompt_hash": prompt_hash,
                "ordering_policy_id": (ORDERING_POLICY_ID if ordering_applied
                                       else "pool_order"),
                "ordering_applied": ordering_applied,
                "selected_set_ids": sorted(selected),
                "ordered_selected_ids": list(selected),
            },
            "hrm": {"prompt_hash": hrm_prompt_hash or prompt_hash,
                    "output": "answer"},
        },
        "evaluator_annotation": {
            "correct": correct,
            "quality": quality,
            "csr": 1.0 if set(required) <= set(selected) else 0.0,
            "required_evidence_ids": list(required),
            "family": "fam_a",
            "source_cluster_id": "cluster_a",
        },
    }


def _arms(n_tasks: int = 4, **kwargs) -> dict[str, list[dict]]:
    return {
        arm: [_receipt(arm, f"task-{i}", **kwargs) for i in range(n_tasks)]
        for arm in PRIMARY
    }


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """A bundle whose files hash correctly."""
    d = tmp_path / "development"
    d.mkdir()
    for arm, recs in _arms().items():
        (d / f"{arm}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    (d / "manifest.json").write_text(json.dumps(
        {"task_count": 4, "protocol_sha256": "x" * 64, "git_commit": "a" * 40},
        indent=2, sort_keys=True))
    (d / "analysis.json").write_text(json.dumps({"parity": {}}, indent=2))
    write_results_hash(d)
    return d


# --- Gate plumbing ----------------------------------------------------------

class TestGatePlumbing:
    def test_gate_with_violation_cannot_pass(self):
        g = certify_mod.Gate("x")
        g.fail("something")
        assert g.finalize().passed is False

    def test_clean_gate_passes(self):
        assert certify_mod.Gate("x").finalize().passed is True


class TestTestSuiteGate:
    def test_not_running_tests_is_a_failure_not_a_skip(self):
        """--no-tests must FAIL the gate; skipping cannot clear an abort."""
        g = certify_mod.gate_tests(run_tests=False, tests_path="tests")
        assert g.passed is False
        assert g.detail["skipped"] is True
        assert any("not run" in v for v in g.violations)


class TestArmsComplete:
    def test_full_bundle_passes(self):
        assert certify_mod.gate_arms_complete(_arms(4), 4).passed

    def test_missing_arm_fails(self):
        arms = _arms(4)
        del arms["C4_5"]
        g = certify_mod.gate_arms_complete(arms, 4)
        assert not g.passed
        assert any("C4_5" in v for v in g.violations)

    def test_short_arm_fails(self):
        arms = _arms(4)
        arms["C4_4"] = arms["C4_4"][:2]
        g = certify_mod.gate_arms_complete(arms, 4)
        assert not g.passed
        assert any("2/4" in v for v in g.violations)

    def test_duplicate_task_ids_fail(self):
        arms = _arms(4)
        arms["C4_0"][1] = arms["C4_0"][0]
        g = certify_mod.gate_arms_complete(arms, 4)
        assert not g.passed
        assert any("duplicate" in v for v in g.violations)


class TestOracleLeakage:
    def test_clean_payload_passes(self):
        assert certify_mod.gate_no_oracle_leakage(_arms(2)).passed

    def test_leaked_required_evidence_ids_fail(self):
        arms = _arms(2)
        arms["C4_4"][0]["runtime_payload"]["required_evidence_ids"] = ["t/value"]
        g = certify_mod.gate_no_oracle_leakage(arms)
        assert not g.passed
        assert any("required_evidence_ids" in v for v in g.violations)

    def test_nested_oracle_metadata_fails(self):
        arms = _arms(2)
        arms["C4_0"][0]["runtime_payload"]["selection"]["_oracle_metadata"] = {}
        assert not certify_mod.gate_no_oracle_leakage(arms).passed


class TestReceiptHashFields:
    def test_v2_receipts_pass(self):
        assert certify_mod.gate_receipt_hash_fields(_arms(2)).passed

    def test_pre_v2_receipts_fail(self):
        """Receipts carrying only packet_hash must not certify."""
        arms = _arms(2)
        for recs in arms.values():
            for r in recs:
                for k in ("candidate_pool_hash", "membership_hash",
                          "order_hash", "prompt_hash"):
                    r["runtime_payload"]["packet"].pop(k)
        g = certify_mod.gate_receipt_hash_fields(arms)
        assert not g.passed
        assert any("predate" in v for v in g.violations)


class TestPromptBinding:
    def test_bound_prompt_passes(self):
        assert certify_mod.gate_prompt_binding(_arms(2)).passed

    def test_recomposed_prompt_fails(self):
        """The exact defect: HRM generating from a re-derived prompt."""
        arms = _arms(2)
        arms["C4_4"][0]["runtime_payload"]["hrm"]["prompt_hash"] = "z" * 64
        g = certify_mod.gate_prompt_binding(arms)
        assert not g.passed
        assert g.detail["mismatches"] == 1
        assert any("not the frozen packet" in v for v in g.violations)

    def test_missing_prompt_hash_fails(self):
        arms = _arms(2)
        arms["C4_0"][0]["runtime_payload"]["packet"]["prompt_hash"] = ""
        assert not certify_mod.gate_prompt_binding(arms).passed


class TestOrderingConformance:
    def test_correct_assignment_passes(self):
        assert certify_mod.gate_ordering_conformance(_arms(2)).passed

    def test_c4_4_without_ordering_fails(self):
        arms = _arms(2)
        for r in arms["C4_4"]:
            r["runtime_payload"]["packet"]["ordering_applied"] = False
            r["runtime_payload"]["packet"]["ordering_policy_id"] = "pool_order"
        g = certify_mod.gate_ordering_conformance(arms)
        assert not g.passed
        assert any("ordering_applied should be True" in v for v in g.violations)

    def test_c4_3_with_ordering_fails(self):
        arms = _arms(2)
        for r in arms["C4_3"]:
            r["runtime_payload"]["packet"]["ordering_applied"] = True
            r["runtime_payload"]["packet"]["ordering_policy_id"] = ORDERING_POLICY_ID
        assert not certify_mod.gate_ordering_conformance(arms).passed

    def test_inconsistent_within_arm_fails(self):
        arms = _arms(2)
        arms["C4_4"][0]["runtime_payload"]["packet"]["ordering_applied"] = False
        g = certify_mod.gate_ordering_conformance(arms)
        assert not g.passed
        assert any("inconsistent" in v for v in g.violations)


class TestIterativeExcluded:
    def test_one_pass_passes(self):
        assert certify_mod.gate_iterative_excluded(_arms(2)).passed

    def test_second_pass_fails(self):
        arms = _arms(2)
        arms["C4_2"][0]["runtime_payload"]["query"]["second_pass_performed"] = True
        g = certify_mod.gate_iterative_excluded(arms)
        assert not g.passed
        assert g.detail["second_pass_counts"] == {"C4_2": 1}


class TestCausalParity:
    def test_shared_pool_passes(self):
        arms = _arms(2)
        analysis = {"parity": {"all_arms_same_tasks": True}}
        assert certify_mod.gate_parity(arms, analysis).passed

    def test_divergent_pool_between_c4_3_and_c4_4_fails(self):
        """A selector contrast is only causal if the pool is held fixed."""
        arms = _arms(2)
        for r in arms["C4_4"]:
            r["runtime_payload"]["packet"]["candidate_pool_hash"] = "d" * 64
        g = certify_mod.gate_parity(arms, {"parity": {"all_arms_same_tasks": True}})
        assert not g.passed
        assert any("confounded" in v for v in g.violations)

    def test_analysis_parity_false_fails(self):
        g = certify_mod.gate_parity(_arms(2),
                                   {"parity": {"all_arms_same_tasks": False}})
        assert not g.passed

    def test_c4_4m_must_share_membership_with_c4_4(self):
        arms = _arms(2)
        arms["C4_4m"] = [_receipt("C4_4m", f"task-{i}", ordering_applied=False,
                                  membership_hash="different" * 8)
                         for i in range(2)]
        g = certify_mod.gate_parity(arms, {"parity": {"all_arms_same_tasks": True}})
        assert not g.passed
        assert any("isolate ordering only" in v for v in g.violations)


class TestMetricCorrectness:
    def _analysis(self, arms) -> dict:
        q = {a: sum(r["evaluator_annotation"]["quality"] for r in recs) / len(recs)
             for a, recs in arms.items()}
        from hrm_adaptive_memory.c4.metrics import (
            oracle_gap_capture, selector_gap_capture)
        return {
            "arm_summary": {a: {"quality": v} for a, v in q.items()},
            "selector_gap_capture": selector_gap_capture(q),
            "oracle_gap_capture": oracle_gap_capture(q),
        }

    def test_consistent_metrics_pass(self):
        arms = _arms(4)
        # A monotone, non-degenerate ladder: Q(C4_3) < Q(C4_4) < Q(C4_5), so
        # both gap-capture denominators are nonzero and actually get checked.
        complete_counts = {"C4_0": 0, "C4_1": 0, "C4_2": 1, "C4_3": 1,
                           "C4_4": 2, "C4_5": 3, "C4_6": 4}
        for arm, recs in arms.items():
            for i, r in enumerate(recs):
                complete = i < complete_counts[arm]
                r["evaluator_annotation"]["correct"] = True
                r["runtime_payload"]["selection"]["selected_ids"] = (
                    ["t/identity", "t/value"] if complete else ["t/identity"])
                r["evaluator_annotation"]["quality"] = 1.0 if complete else 0.25
                r["evaluator_annotation"]["csr"] = 1.0 if complete else 0.0
        g = certify_mod.gate_metric_correctness(arms, self._analysis(arms))
        assert g.passed, g.violations
        assert g.detail["recomputed_selector_gap_capture"] is not None
        assert g.detail["recomputed_oracle_gap_capture"] is not None

    def test_tampered_quality_fails(self):
        arms = _arms(2)
        analysis = self._analysis(arms)
        arms["C4_4"][0]["evaluator_annotation"]["quality"] = 0.5
        g = certify_mod.gate_metric_correctness(arms, analysis)
        assert not g.passed
        assert any("stored quality" in v for v in g.violations)

    def test_quality_overloaded_with_accuracy_fails(self):
        """quality = float(correct) is the historical error; it must not pass."""
        arms = _arms(2)
        for recs in arms.values():
            for r in recs:
                r["runtime_payload"]["selection"]["selected_ids"] = ["t/identity"]
                r["evaluator_annotation"]["correct"] = True
                r["evaluator_annotation"]["quality"] = 1.0  # should be 0.25
                r["evaluator_annotation"]["csr"] = 0.0
        g = certify_mod.gate_metric_correctness(arms, self._analysis(arms))
        assert not g.passed

    def test_reported_gap_capture_mismatch_fails(self):
        """Catches an analysis.json whose OGC used C4_5 in the numerator."""
        arms = _arms(2)
        analysis = self._analysis(arms)
        analysis["oracle_gap_capture"] = 0.7947
        g = certify_mod.gate_metric_correctness(arms, analysis)
        assert not g.passed
        assert any("oracle_gap_capture" in v for v in g.violations)

    def test_arm_summary_mismatch_fails(self):
        arms = _arms(2)
        analysis = self._analysis(arms)
        analysis["arm_summary"]["C4_4"]["quality"] = 0.9999
        g = certify_mod.gate_metric_correctness(arms, analysis)
        assert not g.passed
        assert any("arm_summary" in v for v in g.violations)


class TestDerivedMetricAgreement:
    """analysis.json is never trusted because it is present."""

    def _consistent(self, n: int = 6) -> tuple[dict, dict]:
        """Arms plus the analysis the analyzer actually produces for them."""
        arms = _arms(n)
        complete_counts = {"C4_0": 0, "C4_1": 1, "C4_2": 2, "C4_3": 2,
                           "C4_4": 4, "C4_5": 5, "C4_6": 6}
        for arm, recs in arms.items():
            for i, r in enumerate(recs):
                complete = i < complete_counts[arm]
                r["evaluator_annotation"]["correct"] = complete
                r["runtime_payload"]["selection"]["selected_ids"] = (
                    ["t/identity", "t/value"] if complete else ["t/identity"])
                r["evaluator_annotation"]["quality"] = 1.0 if complete else 0.0
                r["evaluator_annotation"]["csr"] = 1.0 if complete else 0.0
                # Two families so the grouped bootstrap has something to resample.
                r["evaluator_annotation"]["family"] = f"fam_{i % 2}"
                r["evaluator_annotation"]["source_cluster_id"] = f"cluster_{i % 2}"
        analyzer = certify_mod._load_analyzer()
        primary = {a: recs for a, recs in arms.items() if a in PRIMARY}
        from hrm_adaptive_memory.c4.metrics import (
            oracle_gap_capture, selector_gap_capture)
        q = {a: analyzer.arm_quality(recs) for a, recs in primary.items()}
        analysis = {
            "parity": {"all_arms_same_tasks": True},
            "arm_summary": {
                a: {"n": len(recs), "quality": analyzer.arm_quality(recs),
                    "correct_rate": analyzer.arm_correct_rate(recs)}
                for a, recs in primary.items()},
            "csr_stats": {
                a: {"mean_csr": sum(r["evaluator_annotation"]["csr"]
                                    for r in recs) / len(recs)}
                for a, recs in primary.items()},
            "primary_delta": analyzer.paired_deltas(primary["C4_0"], primary["C4_4"]),
            "selector_gap_capture": selector_gap_capture(q),
            "oracle_gap_capture": oracle_gap_capture(q),
        }
        return arms, analysis

    def test_analyzer_is_importable(self):
        assert certify_mod._load_analyzer() is not None

    def test_consistent_analysis_passes(self):
        arms, analysis = self._consistent()
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert g.passed, g.violations

    def test_tampered_arm_quality_fails(self):
        arms, analysis = self._consistent()
        analysis["arm_summary"]["C4_4"]["quality"] += 0.01
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("arm_quality.C4_4" in v for v in g.violations)

    def test_tampered_accuracy_fails(self):
        arms, analysis = self._consistent()
        analysis["arm_summary"]["C4_0"]["correct_rate"] = 0.9
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("correct_rate.C4_0" in v for v in g.violations)

    def test_tampered_csr_fails(self):
        arms, analysis = self._consistent()
        analysis["csr_stats"]["C4_4"]["mean_csr"] = 1.0
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("mean_csr.C4_4" in v for v in g.violations)

    def test_tampered_arm_count_fails(self):
        arms, analysis = self._consistent()
        analysis["arm_summary"]["C4_2"]["n"] = 120
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("arm_summary.C4_2.n" in v for v in g.violations)

    def test_false_task_set_equality_claim_fails(self):
        arms, analysis = self._consistent()
        arms["C4_5"] = arms["C4_5"][:-1]
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("all_arms_same_tasks" in v for v in g.violations)

    def test_tampered_primary_delta_fails(self):
        arms, analysis = self._consistent()
        analysis["primary_delta"]["mean_delta"] = 0.99
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("primary_delta.mean_delta" in v for v in g.violations)

    def test_tampered_family_ci_fails(self):
        arms, analysis = self._consistent()
        analysis["primary_delta"]["ci_lower"] = 0.5
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("primary_delta.ci_lower" in v for v in g.violations)

    def test_tampered_cluster_ci_fails(self):
        arms, analysis = self._consistent()
        analysis["primary_delta"]["ci_upper_cluster"] = 9.0
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("ci_upper_cluster" in v for v in g.violations)

    def test_missing_cluster_ci_fails(self):
        arms, analysis = self._consistent()
        del analysis["primary_delta"]["ci_lower_cluster"]
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("absent from analysis.json" in v for v in g.violations)

    def test_tampered_flip_counts_fail(self):
        arms, analysis = self._consistent()
        analysis["primary_delta"]["flips"]["regress"] += 5
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("flips.regress" in v for v in g.violations)

    def test_ogc_using_c4_5_numerator_fails(self):
        """The exact historical discrepancy: 0.7947 reported vs 0.2263 real."""
        arms, analysis = self._consistent()
        primary = {a: recs for a, recs in arms.items() if a in PRIMARY}
        analyzer = certify_mod._load_analyzer()
        q = {a: analyzer.arm_quality(recs) for a, recs in primary.items()}
        # Wrong formula: (C4_5 - C4_0) / (C4_6 - C4_0)
        analysis["oracle_gap_capture"] = (q["C4_5"] - q["C4_0"]) / (q["C4_6"] - q["C4_0"])
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("oracle_gap_capture" in v for v in g.violations)

    def test_missing_sgc_fails(self):
        arms, analysis = self._consistent()
        analysis["selector_gap_capture"] = None
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed

    def test_diagnostic_arms_do_not_affect_primary_recomputation(self):
        """Adding C4_3o/C4_4m must not change any primary metric."""
        arms, analysis = self._consistent()
        baseline = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert baseline.passed, baseline.violations

        with_diagnostics = dict(arms)
        with_diagnostics["C4_3o"] = [
            _receipt("C4_3o", f"task-{i}", ordering_applied=True) for i in range(6)]
        with_diagnostics["C4_4m"] = [
            _receipt("C4_4m", f"task-{i}", ordering_applied=False) for i in range(6)]
        g = certify_mod.gate_derived_metric_agreement(with_diagnostics, analysis)
        assert g.passed, g.violations
        assert g.detail["recomputed_arm_counts"].keys() == set(PRIMARY)
        assert g.detail["selector_gap_capture"] == \
            baseline.detail["selector_gap_capture"]
        assert g.detail["oracle_gap_capture"] == \
            baseline.detail["oracle_gap_capture"]

    def test_missing_primary_arm_fails(self):
        arms, analysis = self._consistent()
        del arms["C4_4"]
        g = certify_mod.gate_derived_metric_agreement(arms, analysis)
        assert not g.passed
        assert any("primary delta" in v for v in g.violations)


class TestDiagnosticArmIsolationInCertification:
    """Diagnostic arms explain C4_4; they never gate promotion."""

    def test_arm_count_gate_ignores_diagnostic_arms(self):
        """A short diagnostic arm must not fail arms_complete."""
        arms = _arms(4)
        arms["C4_4m"] = [_receipt("C4_4m", "task-0", ordering_applied=False)]
        g = certify_mod.gate_arms_complete(arms, 4)
        assert g.passed, g.violations
        assert set(g.detail["per_arm_counts"]) == set(PRIMARY)

    def test_arm_count_gate_still_requires_all_primary_arms(self):
        arms = _arms(4)
        del arms["C4_6"]
        arms["C4_3o"] = [_receipt("C4_3o", f"task-{i}") for i in range(4)]
        g = certify_mod.gate_arms_complete(arms, 4)
        assert not g.passed
        assert any("C4_6" in v for v in g.violations)

    def test_statistical_gate_uses_only_the_primary_delta(self):
        """The promotion threshold is C4_4 vs C4_0, never a diagnostic arm."""
        import inspect
        source = inspect.getsource(certify_mod.gate_statistical)
        assert "C4_3o" not in source
        assert "C4_4m" not in source
        assert "primary_delta" in source


class TestStatisticalGate:
    def _delta(self, **kw) -> dict:
        base = {"mean_delta": 0.21, "ci_lower": 0.09, "ci_upper": 0.33,
                "ci_lower_cluster": 0.07, "ci_upper_cluster": 0.35,
                "ci_lower_template": 0.05, "ci_upper_template": 0.37,
                "flips": {"improve": 30, "regress": 2, "unchanged": 88}}
        base.update(kw)
        return base

    def test_passing_delta_and_cis(self):
        assert certify_mod.gate_statistical({"primary_delta": self._delta()}).passed

    def test_missing_primary_delta_fails(self):
        assert not certify_mod.gate_statistical({}).passed

    def test_below_threshold_fails(self):
        g = certify_mod.gate_statistical({"primary_delta": self._delta(mean_delta=0.10)})
        assert not g.passed
        assert any("below the predeclared threshold" in v for v in g.violations)

    def test_family_ci_touching_zero_fails(self):
        g = certify_mod.gate_statistical(
            {"primary_delta": self._delta(ci_lower=-0.01)})
        assert not g.passed
        assert any("family-grouped" in v for v in g.violations)

    def test_missing_cluster_ci_fails(self):
        delta = self._delta()
        del delta["ci_lower_cluster"]
        g = certify_mod.gate_statistical({"primary_delta": delta})
        assert not g.passed
        assert any("cluster-grouped" in v for v in g.violations)


class TestDeterminismGate:
    def _receipt_v2(self, tmp_path: Path, **kw) -> Path:
        payload = {
            "schema_version": "c4-determinism-qualification-v2",
            "tasks": 120, "seeds": [0, 42, 12345], "arms": ["C4_4"],
            "result": "PASS",
            "compared_fields": ["candidate_pool_hash", "membership_hash",
                                "order_hash", "packet_hash", "prompt_hash",
                                "query_text"],
            "per_arm": {"C4_4": {"result": "PASS",
                                 "diffs": {"42": {"packet_hash": 0}}}},
        }
        payload.update(kw)
        p = tmp_path / "det.json"
        p.write_text(json.dumps(payload))
        return p

    def test_valid_receipt_passes(self, tmp_path):
        assert certify_mod.gate_determinism(self._receipt_v2(tmp_path), 120).passed

    def test_missing_receipt_fails(self, tmp_path):
        g = certify_mod.gate_determinism(tmp_path / "absent.json", 120)
        assert not g.passed

    def test_v1_schema_is_rejected(self, tmp_path):
        """v1 claimed pool/order invariance it never measured."""
        p = self._receipt_v2(
            tmp_path, schema_version="c4-determinism-qualification-v1")
        g = certify_mod.gate_determinism(p, 120)
        assert not g.passed
        assert any("never" in v and "measured" in v for v in g.violations)

    def test_unmeasured_fields_rejected(self, tmp_path):
        p = self._receipt_v2(tmp_path,
                             compared_fields=["packet_hash", "query_text"])
        g = certify_mod.gate_determinism(p, 120)
        assert not g.passed
        assert any("did not compare" in v for v in g.violations)

    def test_fewer_tasks_than_run_fails(self, tmp_path):
        g = certify_mod.gate_determinism(self._receipt_v2(tmp_path, tasks=6), 120)
        assert not g.passed
        assert any("6 tasks" in v for v in g.violations)

    def test_single_seed_fails(self, tmp_path):
        g = certify_mod.gate_determinism(self._receipt_v2(tmp_path, seeds=[0]), 120)
        assert not g.passed

    def test_missing_primary_arm_fails(self, tmp_path):
        g = certify_mod.gate_determinism(self._receipt_v2(tmp_path, arms=["C4_0"]), 120)
        assert not g.passed
        assert any("C4_4" in v for v in g.violations)

    def test_nonzero_diffs_fail(self, tmp_path):
        p = self._receipt_v2(tmp_path, per_arm={
            "C4_4": {"result": "PASS", "diffs": {"42": {"order_hash": 3}}}})
        g = certify_mod.gate_determinism(p, 120)
        assert not g.passed
        assert any("order_hash" in v for v in g.violations)

    def test_fail_result_fails(self, tmp_path):
        g = certify_mod.gate_determinism(self._receipt_v2(tmp_path, result="FAIL"), 120)
        assert not g.passed


class TestResultsHashGate:
    def test_intact_bundle_passes(self, bundle):
        assert certify_mod.gate_results_hash(bundle).passed

    def test_tampered_receipt_fails(self, bundle):
        path = bundle / "C4_4.jsonl"
        path.write_text(path.read_text() + json.dumps({"task_id": "extra"}) + "\n")
        g = certify_mod.gate_results_hash(bundle)
        assert not g.passed
        assert any("C4_4.jsonl" in v for v in g.violations)

    def test_missing_hash_file_fails(self, bundle):
        (bundle / "RESULTS.sha256").unlink()
        assert not certify_mod.gate_results_hash(bundle).passed

    def test_deleted_listed_file_fails(self, bundle):
        (bundle / "C4_1.jsonl").unlink()
        g = certify_mod.gate_results_hash(bundle)
        assert not g.passed
        assert any("missing" in v for v in g.violations)


class TestValidRunConjunction:
    def test_one_failing_gate_makes_valid_run_false(self, bundle, monkeypatch):
        """VALID_RUN is a conjunction, not the protocol hash alone."""
        certificate, _snapshot, _lock = certify_mod.certify(
            bundle=bundle,
            protocol_path=ROOT / "configs/gate_c4_protocol_v2.json",
            lock_path=ROOT / "configs/c4_requirements.lock",
            determinism_receipt=bundle / "absent.json",
            run_tests=False,
            tests_path="tests",
            expect_source_sha=None,
        )
        assert certificate["VALID_RUN"] is False
        assert certificate["verdict"] == "NOT_CERTIFIED"
        assert "test_suite" in certificate["gates_failed"]
        assert "determinism" in certificate["gates_failed"]

    def test_certificate_records_every_gate(self, bundle):
        certificate, _s, _l = certify_mod.certify(
            bundle=bundle,
            protocol_path=ROOT / "configs/gate_c4_protocol_v2.json",
            lock_path=ROOT / "configs/c4_requirements.lock",
            determinism_receipt=bundle / "absent.json",
            run_tests=False, tests_path="tests", expect_source_sha=None,
        )
        assert certificate["gates_total"] == len(certificate["gates"])
        assert certificate["gates_total"] >= 15
        for name, result in certificate["gates"].items():
            assert set(result) == {"passed", "violations", "detail"}, name

    def test_performance_numbers_do_not_influence_valid_run(self, bundle):
        """A strong delta must not compensate for a failed reproducibility gate."""
        (bundle / "analysis.json").write_text(json.dumps({
            "parity": {"all_arms_same_tasks": True},
            "primary_delta": {"mean_delta": 0.99, "ci_lower": 0.9,
                              "ci_upper": 1.0, "ci_lower_cluster": 0.9,
                              "ci_upper_cluster": 1.0, "ci_lower_template": 0.9,
                              "ci_upper_template": 1.0, "flips": {}},
        }))
        write_results_hash(bundle)
        certificate, _s, _l = certify_mod.certify(
            bundle=bundle,
            protocol_path=ROOT / "configs/gate_c4_protocol_v2.json",
            lock_path=ROOT / "configs/c4_requirements.lock",
            determinism_receipt=bundle / "absent.json",
            run_tests=False, tests_path="tests", expect_source_sha=None,
        )
        assert certificate["VALID_RUN"] is False
