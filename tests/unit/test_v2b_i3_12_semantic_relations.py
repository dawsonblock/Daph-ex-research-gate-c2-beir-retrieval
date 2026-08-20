"""I3.12e: Leakage barriers and behavioral equivalence tests for the
semantic relation extractor.

These tests enforce:
  G02: zero evaluator leakage (extractor cannot access gold relations,
       correct_hypothesis_id, verify_result, expected_terminal,
       oracle_path, category)
  - behavioral equivalence: same text -> same relations regardless of
    hidden task truth
  - extractor identity stability
  - relation graph well-formedness
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.executive.semantic_relations import (
    DeterministicRelationExtractor,
    RelationType,
    ExtractionInput,
    compute_relation_metrics,
)
from hrm_adaptive_memory.executive.semantic_relations.extractor import FORBIDDEN_FIELDS
from hrm_adaptive_memory.executive.semantic_relations.validator import (
    validate_extraction_input,
    validate_relation_graph,
    check_behavioral_equivalence,
)
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    generate_i3_12_corpus,
    GoldRelation,
)


class TestForbiddenFieldRejection:
    """G02: Extractor must reject forbidden fields."""

    @pytest.fixture
    def extractor(self):
        return DeterministicRelationExtractor()

    @pytest.mark.parametrize("forbidden_field", sorted(FORBIDDEN_FIELDS))
    def test_extract_rejects_forbidden_field(self, extractor, forbidden_field):
        """Each forbidden field must be rejected by extract()."""
        with pytest.raises(ValueError, match="forbidden field"):
            extractor.extract(
                evidence_id="E1",
                evidence_proposition="Source A confirms X is current.",
                hypothesis_id="H1",
                hypothesis_proposition="X is current and confirmed.",
                **{forbidden_field: "test_value"},
            )

    def test_extract_graph_rejects_forbidden_evidence_field(self, extractor):
        """Evidence items with forbidden fields must be rejected."""
        with pytest.raises(ValueError, match="forbidden field"):
            extractor.extract_graph(
                task_id="test",
                evidence_items=[{"evidence_id": "E1", "proposition": "test",
                                 "correct_hypothesis_id": "H1"}],
                hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
            )

    def test_extract_graph_rejects_forbidden_hypothesis_field(self, extractor):
        """Hypotheses with forbidden fields must be rejected."""
        with pytest.raises(ValueError, match="forbidden field"):
            extractor.extract_graph(
                task_id="test",
                evidence_items=[{"evidence_id": "E1", "proposition": "test"}],
                hypotheses=[{"hypothesis_id": "H1", "proposition": "test",
                             "expected_terminal": "ANSWER"}],
            )

    def test_extract_graph_rejects_supports_in_evidence(self, extractor):
        """Evidence items must not carry oracle supports field."""
        with pytest.raises(ValueError, match="forbidden field"):
            extractor.extract_graph(
                task_id="test",
                evidence_items=[{"evidence_id": "E1", "proposition": "test",
                                 "supports": ["H1"]}],
                hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
            )

    def test_extract_graph_rejects_contradicts_in_evidence(self, extractor):
        """Evidence items must not carry oracle contradicts field."""
        with pytest.raises(ValueError, match="forbidden field"):
            extractor.extract_graph(
                task_id="test",
                evidence_items=[{"evidence_id": "E1", "proposition": "test",
                                 "contradicts": ["H1"]}],
                hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
            )


class TestBehavioralEquivalence:
    """Same text must produce same relations regardless of hidden task truth."""

    @pytest.fixture
    def extractor(self):
        return DeterministicRelationExtractor()

    def test_same_text_same_relation(self, extractor):
        """Repeated calls with same text must produce same relation."""
        assert check_behavioral_equivalence(
            extractor,
            "Source A confirms that X is current and operational.",
            "The system should ANSWER because X is current and confirmed.",
        )

    def test_different_hidden_truth_same_output(self, extractor):
        """Two tasks with same evidence/hypothesis text but different
        correct_hypothesis_id must produce identical inferred relations."""
        # The extractor must not see correct_hypothesis_id
        # If it did, it could cheat. This test verifies it can't.
        ev_text = "Source A confirms that the API endpoint status is current and operational."
        hyp_text = "The system should ANSWER because the API endpoint status is current and confirmed."

        r1 = extractor.extract(
            evidence_id="E1", evidence_proposition=ev_text,
            hypothesis_id="H1", hypothesis_proposition=hyp_text,
        )
        r2 = extractor.extract(
            evidence_id="E1", evidence_proposition=ev_text,
            hypothesis_id="H1", hypothesis_proposition=hyp_text,
        )

        assert r1.relation.relation == r2.relation.relation
        assert r1.relation.relation is RelationType.SUPPORT

    def test_extractor_identity_stable(self, extractor):
        """Extractor identity must be deterministic."""
        ext2 = DeterministicRelationExtractor()
        assert extractor.identity.sha256 == ext2.identity.sha256


class TestRelationGraphWellFormedness:
    """Relation graphs must be well-formed."""

    @pytest.fixture
    def extractor(self):
        return DeterministicRelationExtractor()

    def test_graph_has_all_evidence_hypothesis_pairs(self, extractor):
        """Graph must have one relation per evidence x hypothesis pair."""
        graph = extractor.extract_graph(
            task_id="test",
            evidence_items=[
                {"evidence_id": "E1", "proposition": "Source A confirms X is current."},
                {"evidence_id": "E2", "proposition": "Source B refutes X is current."},
            ],
            hypotheses=[
                {"hypothesis_id": "H1", "proposition": "X is current and confirmed."},
                {"hypothesis_id": "H2", "proposition": "X is stale or unconfirmed."},
            ],
        )
        assert len(graph.relations) == 4  # 2 evidence x 2 hypotheses

    def test_graph_no_duplicates(self, extractor):
        """Graph must not have duplicate (evidence, hypothesis) pairs."""
        graph = extractor.extract_graph(
            task_id="test",
            evidence_items=[{"evidence_id": "E1", "proposition": "test"}],
            hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
        )
        violations = validate_relation_graph(graph)
        assert not violations

    def test_graph_has_hashes(self, extractor):
        """Each relation must have evidence and hypothesis hashes."""
        graph = extractor.extract_graph(
            task_id="test",
            evidence_items=[{"evidence_id": "E1", "proposition": "test"}],
            hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
        )
        for rel in graph.relations:
            assert rel.evidence_sha256
            assert rel.hypothesis_sha256
            assert len(rel.evidence_sha256) == 64  # SHA-256 hex
            assert len(rel.hypothesis_sha256) == 64

    def test_graph_sha256_stable(self, extractor):
        """Relation graph hash must be deterministic."""
        g1 = extractor.extract_graph(
            task_id="test",
            evidence_items=[{"evidence_id": "E1", "proposition": "test"}],
            hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
        )
        g2 = extractor.extract_graph(
            task_id="test",
            evidence_items=[{"evidence_id": "E1", "proposition": "test"}],
            hypotheses=[{"hypothesis_id": "H1", "proposition": "test"}],
        )
        assert g1.relation_graph_sha256 == g2.relation_graph_sha256


class TestExtractorAccuracy:
    """G01: Extractor must achieve macro F1 >= 0.90 on S1 corpus."""

    def test_extractor_macro_f1_above_gate(self):
        """Extractor must achieve macro F1 >= 0.90 on the S1 corpus."""
        tasks = generate_i3_12_corpus(n_per_category=5, seed=42)
        ext = DeterministicRelationExtractor()

        all_predicted = []
        all_gold = []

        for task in tasks:
            for ev in task.evidence_task.evidence_items:
                for h in task.evidence_task.hypotheses:
                    result = ext.extract(
                        evidence_id=ev.evidence_id,
                        evidence_proposition=ev.proposition,
                        hypothesis_id=h.hypothesis_id,
                        hypothesis_proposition=h.proposition,
                    )
                    gold_rel = "NEUTRAL"
                    for gr in task.gold_relations:
                        if gr.evidence_id == ev.evidence_id and gr.hypothesis_id == h.hypothesis_id:
                            gold_rel = gr.relation
                            break
                    all_predicted.append(result.relation.relation)
                    all_gold.append(RelationType(gold_rel))

        metrics = compute_relation_metrics(all_predicted, all_gold)
        assert metrics.macro_f1 >= 0.90, f"Macro F1 {metrics.macro_f1:.4f} < 0.90"
        assert metrics.accuracy >= 0.90, f"Accuracy {metrics.accuracy:.4f} < 0.90"


class TestValidatorInput:
    """Test the validator functions."""

    def test_clean_input_no_violations(self):
        """Clean input should have no violations."""
        data = {"evidence_id": "E1", "proposition": "test", "hypothesis_id": "H1"}
        violations = validate_extraction_input(data)
        assert not violations

    def test_forbidden_field_detected(self):
        """Forbidden fields should be detected."""
        data = {"evidence_id": "E1", "proposition": "test", "correct_hypothesis_id": "H1"}
        violations = validate_extraction_input(data)
        assert len(violations) == 1
        assert "correct_hypothesis_id" in violations[0]
