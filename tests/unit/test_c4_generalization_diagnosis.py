"""Tests for scripts/diagnose_c4_generalization.py.

The decomposition's whole value is that its buckets are mutually exclusive
and exhaustive along the pipeline's own stages -- if a task could land in two
buckets, or fall through all of them, the "retrieval is 2.6x the selector
lane" conclusion drawn from it would be meaningless. These tests pin the
classification boundaries with hand-built receipts, then check the real
qualification bundle for the invariants that must hold on actual data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_diagnose_c4", ROOT / "scripts/diagnose_c4_generalization.py")
diag = importlib.util.module_from_spec(_spec)
sys.modules["_diagnose_c4"] = diag
_spec.loader.exec_module(diag)

QUAL = ROOT / "evidence/gate_c4/full/qualification"


def _receipt(*, task_id="t-1", required=("e1",), pool=("e1",), selected=("e1",),
             correct=True, quality=1.0, identity="EXACT", family="entity_attribute",
             regime="canonical", fusion=None):
    ranked = [[e, 0.5] for e in (fusion if fusion is not None else pool)]
    return {
        "task_id": task_id,
        "arm_id": "C4_4",
        "runtime_payload": {
            "retrieval": {"candidate_ids": list(pool), "fusion_ranked": ranked},
            "identity": {"status": identity},
            "selection": {"selected_ids": list(selected)},
        },
        "evaluator_annotation": {
            "required_evidence_ids": list(required),
            "correct": correct,
            "quality": quality,
            "family": family,
            "metadata": {"entity_regime": regime},
        },
    }


class TestClassificationBoundaries:
    def test_evidence_never_retrieved_is_retrieval_failure(self):
        assert diag.classify(_receipt(required=("e1",), pool=("e2",),
                                      selected=("e2",))) == "A_RETRIEVAL"

    def test_retrieval_failure_wins_even_if_answer_happens_to_be_correct(self):
        """Guessing right does not mean the pipeline supplied the evidence --
        it must still count against retrieval, or A would understate."""
        assert diag.classify(_receipt(required=("e1",), pool=("e2",),
                                      selected=("e2",), correct=True)) == "A_RETRIEVAL"

    def test_available_but_dropped_is_selector_failure(self):
        assert diag.classify(_receipt(required=("e1",), pool=("e1", "e2"),
                                      selected=("e2",))) == "B_SELECTOR"

    def test_partial_evidence_counts_as_dropped(self):
        """Required sets are all-or-nothing: half a chain answers nothing."""
        assert diag.classify(_receipt(required=("e1", "e2"), pool=("e1", "e2"),
                                      selected=("e1",))) == "B_SELECTOR"

    def test_survived_but_wrong_is_reader_failure(self):
        assert diag.classify(_receipt(selected=("e1",), correct=False)) == "C_READER"

    def test_survived_and_correct_is_success(self):
        assert diag.classify(_receipt(selected=("e1",), correct=True)) == "SUCCESS"

    def test_buckets_are_exhaustive_and_exclusive(self):
        receipts = [
            _receipt(task_id="a", required=("e1",), pool=("e2",), selected=("e2",)),
            _receipt(task_id="b", required=("e1",), pool=("e1",), selected=("e2",)),
            _receipt(task_id="c", selected=("e1",), correct=False, quality=0.5),
            _receipt(task_id="d", selected=("e1",), correct=True),
        ]
        report = diag.decompose(receipts)
        assert report["task_count"] == 4
        assert sum(b["n"] for b in report["buckets"].values()) == 4
        assert [report["buckets"][k]["n"] for k in diag.BUCKETS] == [1, 1, 1, 1]


class TestKeepRateAsymmetry:
    def test_unavailable_evidence_excluded_from_keep_rate(self):
        """Keep rate must measure only what the selector could have kept;
        counting unavailable evidence would blame the selector for retrieval."""
        receipts = [
            _receipt(task_id="a", required=("e1",), pool=("e2",), selected=("e2",),
                     identity="EXACT"),
            _receipt(task_id="b", required=("e1",), pool=("e1",), selected=("e1",),
                     identity="EXACT"),
        ]
        keep = diag.decompose(receipts)["D_exact_over_expansion"][
            "available_evidence_keep_rate_by_identity_status"]
        assert keep["EXACT"]["available"] == 1
        assert keep["EXACT"]["keep_rate"] == 1.0

    def test_rank_1_drop_is_flagged(self):
        """Dropping the retriever's own top hit is the strongest form of the
        over-expansion claim, so it gets counted separately."""
        receipts = [_receipt(required=("e1",), pool=("e1", "e2"), selected=("e2",),
                             fusion=("e1", "e2"), identity="EXACT")]
        d = diag.decompose(receipts)["D_exact_over_expansion"]
        assert d["selector_drops_by_identity_status"]["EXACT"][
            "required_was_fusion_rank_1"] == 1
        assert d["rank_1_drop_examples"][0]["task_id"] == "t-1"

    def test_rank_1_not_flagged_when_evidence_ranked_lower(self):
        receipts = [_receipt(required=("e1",), pool=("e1", "e2"), selected=("e2",),
                             fusion=("e2", "e1"), identity="EXACT")]
        d = diag.decompose(receipts)["D_exact_over_expansion"]
        assert d["selector_drops_by_identity_status"]["EXACT"][
            "required_was_fusion_rank_1"] == 0


class TestPerFamilyBottleneck:
    def test_retrieval_deficit_vs_selector_headroom_decides_label(self):
        primary = [
            # thin pool, selector kept what little there was -> RETRIEVAL
            _receipt(task_id="r1", family="thin", required=("e1",), pool=("e2",),
                     selected=("e2",)),
            # evidence there, selector dropped it -> SELECTOR
            _receipt(task_id="s1", family="fat", required=("e1",),
                     pool=("e1", "e2"), selected=("e2",)),
        ]
        oracle = [
            _receipt(task_id="r1", family="thin", required=("e1",), pool=("e2",),
                     selected=("e2",)),
            _receipt(task_id="s1", family="fat", required=("e1",),
                     pool=("e1", "e2"), selected=("e1",)),
        ]
        out = diag.per_family(primary, oracle)
        assert out["thin"]["bottleneck"] == "RETRIEVAL"
        assert out["fat"]["bottleneck"] == "SELECTOR"
        assert out["fat"]["selector_headroom"] == 1.0


class TestSubgroupDeltas:
    def test_pairs_by_task_id_not_position(self):
        """Arms must be joined on task_id; positional zip would silently
        compare different tasks if either file were ordered differently."""
        baseline = [_receipt(task_id="a", quality=0.0, regime="canonical"),
                    _receipt(task_id="b", quality=0.0, regime="abbreviation")]
        primary = [_receipt(task_id="b", quality=1.0, regime="abbreviation"),
                   _receipt(task_id="a", quality=0.25, regime="canonical")]
        out = diag.subgroup_deltas(baseline, primary, "entity_regime")
        assert out["canonical"]["mean_delta"] == 0.25
        assert out["abbreviation"]["mean_delta"] == 1.0


class TestNeverWritesIntoACertifiedBundle:
    """BUNDLE.sha256 is a recursive hash over the whole bundle directory,
    including certification/. Dropping a diagnosis file inside therefore
    breaks verification of an already-certified result -- which is exactly
    what happened on this script's first run against the real qualification
    bundle, caught by verify_bundle_hash returning False."""

    def test_default_output_is_outside_the_bundle(self):
        bundle = ROOT / "evidence/gate_c4/full/qualification"
        assert not diag.default_out_path(bundle).is_relative_to(bundle)

    def test_default_output_is_a_sibling_diagnosis_directory(self):
        bundle = Path("evidence/gate_c4/full/qualification")
        assert diag.default_out_path(bundle) == Path(
            "evidence/gate_c4/diagnosis/qualification_generalization.json")

    @pytest.mark.parametrize("style", ["relative", "absolute"])
    def test_explicit_out_inside_the_bundle_is_refused(self, style, tmp_path):
        """A relative --out must be caught too: is_relative_to() compares path
        text, so an unresolved relative path silently escaped the check in the
        first version of this guard."""
        import subprocess
        target = "evidence/gate_c4/full/qualification/should_not_appear.json"
        if style == "absolute":
            target = str(ROOT / target)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/diagnose_c4_generalization.py"),
             "--out", target],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0, proc.stdout[-500:]
        assert "refusing to write inside the bundle" in (proc.stdout + proc.stderr)
        assert not (ROOT / "evidence/gate_c4/full/qualification"
                    / "should_not_appear.json").exists()

    @pytest.mark.skipif(not (QUAL / "BUNDLE.sha256").is_file(),
                        reason="qualification bundle not present")
    def test_committed_qualification_bundle_hash_still_verifies(self):
        """Guards the committed artifact itself, not just the script."""
        from hrm_adaptive_memory.c4.provenance import verify_bundle_hash
        assert verify_bundle_hash(QUAL) is True


@pytest.mark.skipif(not (QUAL / "C4_4.jsonl").is_file(),
                    reason="qualification bundle not present")
class TestAgainstTheRealQualificationBundle:
    """Invariants that must hold on the committed 500-task receipts."""

    @pytest.fixture(scope="class")
    def report(self):
        return {
            "decomposition": diag.decompose(diag.load_arm(QUAL, "C4_4")),
            "families": diag.per_family(diag.load_arm(QUAL, "C4_4"),
                                        diag.load_arm(QUAL, "C4_5")),
        }

    def test_all_500_tasks_are_classified(self, report):
        d = report["decomposition"]
        assert d["task_count"] == 500
        assert sum(b["n"] for b in d["buckets"].values()) == 500

    def test_success_bucket_is_perfect_quality_by_construction(self, report):
        assert report["decomposition"]["buckets"]["SUCCESS"]["mean_quality"] == 1.0

    def test_oracle_selector_never_below_candidate_pool(self, report):
        """C4_5 selects from the same pool, so its CES is that pool's ceiling;
        oracle CES above candidate CES would mean leakage."""
        for family, row in report["families"].items():
            assert row["oracle_selected_ces"] <= row["candidate_ces"] + 1e-9, family

    def test_selector_never_beats_the_oracle_it_is_measured_against(self, report):
        for family, row in report["families"].items():
            assert row["selected_ces"] <= row["oracle_selected_ces"] + 1e-9, family

    def test_exact_identity_keep_rate_is_far_below_resolved(self, report):
        """The finding that motivates information-state routing. Pinned so a
        future selector change cannot quietly erase the evidence for it."""
        keep = report["decomposition"]["D_exact_over_expansion"][
            "available_evidence_keep_rate_by_identity_status"]
        assert keep["EXACT"]["keep_rate"] < 0.40
        assert keep["RESOLVED"]["keep_rate"] > 0.85
