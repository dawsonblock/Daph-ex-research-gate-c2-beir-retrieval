"""Tests for the frozen confirmation split and its builder.

This split is the one-shot test for J1 (frozen_rrf + S2). If it is
contaminated, or if it quietly avoids the corpus-scaling condition that broke
v2.1, then passing it would prove nothing -- so both properties are pinned
here against the committed data, not just against the builder's own report.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/hrm/controlled_gate_a_v4"
CONFIRMATION = DATASET / "confirmation"

# Cycle-breaking import, mirroring the builder: importing experiments.* first
# hits a pre-existing experiments <-> evaluation circular import.
import hrm_adaptive_memory.evaluation  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location(
    "_build_confirmation", ROOT / "scripts/build_c5_confirmation_split.py")
builder = importlib.util.module_from_spec(_spec)
sys.modules["_build_confirmation"] = builder
_spec.loader.exec_module(builder)


def _load(split: str):
    base = DATASET / split
    tasks = [json.loads(l) for l in (base / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (base / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


pytestmark = pytest.mark.skipif(
    not (CONFIRMATION / "oracle_tasks.jsonl").is_file(),
    reason="confirmation split not built")


@pytest.fixture(scope="module")
def confirmation():
    return _load("confirmation")


@pytest.fixture(scope="module")
def frozen():
    return {name: _load(name) for name in ("development", "qualification", "ood")}


class TestSeedAndSizing:
    def test_seed_differs_from_every_pre_existing_split(self):
        """A reused seed would regenerate an existing split's content verbatim."""
        assert builder.check_seed_is_unused() == []

    def test_final_size_meets_the_protocol_minimum(self, confirmation):
        tasks, _ = confirmation
        assert len(tasks) >= 500

    def test_families_are_balanced(self, confirmation):
        tasks, _ = confirmation
        counts = Counter(t["family"] for t in tasks)
        assert len(counts) == 10
        assert set(counts.values()) == {builder.TASKS_PER_FAMILY}

    def test_overgeneration_exceeds_the_target(self):
        """Collision filtering removes tasks, so generating exactly the target
        would leave families short."""
        assert builder.OVERGENERATE_PER_FAMILY > builder.TASKS_PER_FAMILY


class TestContentIsolationFromEveryFrozenSplit:
    """The property that makes this a valid test set."""

    def test_no_question_is_reused(self, confirmation, frozen):
        detail, problems = builder.check_cross_split_isolation(confirmation, frozen)
        assert problems == []
        for split, counts in detail.items():
            assert counts["shared_questions"] == 0, split

    def test_no_question_answer_pair_is_reused(self, confirmation, frozen):
        detail, _ = builder.check_cross_split_isolation(confirmation, frozen)
        for split, counts in detail.items():
            assert counts["shared_question_answer_pairs"] == 0, split

    def test_no_evidence_text_is_reused(self, confirmation, frozen):
        detail, _ = builder.check_cross_split_isolation(confirmation, frozen)
        for split, counts in detail.items():
            assert counts["shared_evidence_content"] == 0, split

    def test_ids_are_expected_to_overlap_and_that_is_not_contamination(
            self, confirmation, frozen):
        """Guards against a future 'fix' that mistakes positional ids for
        identities. Ids are {family}-{ordinal} labels scoped by directory;
        development and qualification already share all of development's."""
        tasks, _ = confirmation
        dev_ids = {t["task_id"] for t in frozen["development"][0]}
        qual_ids = {t["task_id"] for t in frozen["qualification"][0]}
        assert dev_ids <= qual_ids, "premise: frozen splits already share ids"
        assert {t["task_id"] for t in tasks} & dev_ids, (
            "confirmation shares ids too, by construction -- content is what "
            "must differ, and the other tests in this class check that")


class TestReproducesTheCorpusScalingCondition:
    """v2.1 failed because a fixed k=50 met a 4.15x larger corpus. A
    confirmation split at development's scale would dodge that condition."""

    def test_corpus_is_close_to_qualification_scale(self, confirmation, frozen):
        _, evidence = confirmation
        _, qual_evidence = frozen["qualification"]
        assert abs(len(evidence) - len(qual_evidence)) / len(qual_evidence) <= 0.10

    def test_corpus_is_several_times_development(self, confirmation, frozen):
        _, evidence = confirmation
        _, dev_evidence = frozen["development"]
        assert len(evidence) / len(dev_evidence) > 4.0


class TestStructurePreserved:
    def test_both_identity_regimes_present_and_balanced(self, confirmation):
        tasks, _ = confirmation
        counts = Counter(t["metadata"]["entity_regime"] for t in tasks)
        assert set(counts) == {"canonical", "abbreviation"}
        assert min(counts.values()) / len(tasks) > 0.4

    def test_both_bridged_and_unbridged_tasks_present(self, confirmation):
        """The EXACT-and-bridged population is exactly what S2 repairs, so a
        split without bridged tasks could not test the mechanism."""
        tasks, _ = confirmation
        bridged = sum(1 for t in tasks
                      if (t.get("_oracle_metadata") or {}).get("latent_bridge"))
        assert bridged > 0 and bridged < len(tasks)
        assert bridged / len(tasks) > 0.5

    def test_temporal_families_present(self, confirmation):
        families = {t["family"] for t in confirmation[0]}
        assert {"temporal_chain", "temporal_update"} <= families

    def test_structural_diversity_meets_the_same_minimums_as_qualification(
            self, confirmation):
        structural, problems = builder.check_structural_diversity(confirmation[0])
        assert problems == []
        assert structural["families"] >= 8


class TestFrozenRecordAndAdditiveWrite:
    def test_audit_record_declares_it_frozen_before_evaluation(self):
        audit = json.loads((CONFIRMATION / "CONFIRMATION_AUDIT.json").read_text())
        assert audit["frozen_before_evaluation"] is True
        assert "run_exactly_once" in audit
        assert audit["seed"] == builder.CONFIRMATION_SEED

    def test_audit_records_zero_content_overlap(self):
        audit = json.loads((CONFIRMATION / "CONFIRMATION_AUDIT.json").read_text())
        detail = audit["cross_split_content_isolation"]["detail"]
        for split, counts in detail.items():
            assert counts["shared_questions"] == 0, split
            assert counts["shared_evidence_content"] == 0, split

    def test_committed_digests_match_the_files_on_disk(self):
        """Catches post-freeze edits to a split that must not change."""
        import hashlib
        audit = json.loads((CONFIRMATION / "CONFIRMATION_AUDIT.json").read_text())
        for name, digest in audit["digests"].items():
            actual = hashlib.sha256((CONFIRMATION / name).read_bytes()).hexdigest()
            assert actual == digest, f"{name} changed after freezing"

    def test_builder_refuses_to_overwrite_the_frozen_split(self):
        """Re-running must not silently regenerate it."""
        assert builder.check_no_existing_file_would_change() != []

    def test_certified_bundle_corpus_hashes_are_unaffected(self):
        """The write had to be additive: certified bundles hash the development
        and qualification corpora by content."""
        from hrm_adaptive_memory.c4.provenance import sha256_corpus
        for split in ("development", "qualification"):
            manifest_path = ROOT / f"evidence/gate_c4/full/{split}/manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            base = DATASET / split
            assert sha256_corpus(base / "oracle_tasks.jsonl") == \
                manifest["task_corpus_sha256"], split
            assert sha256_corpus(base / "evidence.jsonl") == \
                manifest["evidence_corpus_sha256"], split
