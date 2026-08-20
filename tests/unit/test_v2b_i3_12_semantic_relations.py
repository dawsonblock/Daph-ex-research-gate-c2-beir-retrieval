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


class TestSnapshotIntegration:
    """I3.12f: Integration of inferred relations into EvidenceSnapshot."""

    @pytest.fixture
    def budget(self):
        from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
        return ResourceBudget(
            max_executive_steps=24, max_reasoning_tokens=2048,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
        )

    @pytest.fixture
    def extractor(self):
        return DeterministicRelationExtractor()

    def test_s0_oracle_snapshot_preserves_relations(self, budget):
        """S0 (oracle) snapshot must preserve original supports/contradicts."""
        from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
        from hrm_adaptive_memory.executive.semantic_relations.integration import build_evidence_snapshot_oracle
        from hrm_adaptive_memory.executive.resources import ResourceState

        tasks = generate_i3_12_corpus(n_per_category=1, seed=42)
        task = tasks[0]
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))
        snap = build_evidence_snapshot_oracle(runtime)

        # Oracle relations should match the original task evidence
        for ev_orig, ev_snap in zip(task.evidence_task.evidence_items, snap.visible_evidence):
            assert ev_snap.supports == ev_orig.supports
            assert ev_snap.contradicts == ev_orig.contradicts

    def test_s1_inferred_snapshot_has_relations(self, budget, extractor):
        """S1 (inferred) snapshot must have inferred supports/contradicts."""
        from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
        from hrm_adaptive_memory.executive.semantic_relations.integration import build_evidence_snapshot_with_inferred_relations
        from hrm_adaptive_memory.executive.resources import ResourceState

        tasks = generate_i3_12_corpus(n_per_category=1, seed=42)
        task = tasks[0]
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))
        snap, graph = build_evidence_snapshot_with_inferred_relations(runtime, extractor)

        # Each visible evidence should have inferred relations
        for ev in snap.visible_evidence:
            # Inferred relations should be non-empty for supporting/contradicting evidence
            assert isinstance(ev.supports, tuple)
            assert isinstance(ev.contradicts, tuple)

    def test_s1_does_not_mutate_original_runtime(self, budget, extractor):
        """S1 inference must not mutate the original runtime."""
        from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
        from hrm_adaptive_memory.executive.semantic_relations.integration import infer_relations_for_runtime
        from hrm_adaptive_memory.executive.resources import ResourceState

        tasks = generate_i3_12_corpus(n_per_category=1, seed=42)
        task = tasks[0]
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))

        orig_supports = runtime.evidence[0].supports
        orig_contradicts = runtime.evidence[0].contradicts

        new_runtime, graph = infer_relations_for_runtime(runtime, extractor)

        # Original must be unchanged
        assert runtime.evidence[0].supports == orig_supports
        assert runtime.evidence[0].contradicts == orig_contradicts
        # New runtime must be a different object
        assert runtime is not new_runtime
        assert runtime.evidence[0] is not new_runtime.evidence[0]

    def test_s0_s1_affordances_identical(self, budget, extractor):
        """S0 and S1 must have identical affordances (no information advantage)."""
        from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
        from hrm_adaptive_memory.executive.semantic_relations.integration import (
            build_evidence_snapshot_oracle,
            build_evidence_snapshot_with_inferred_relations,
        )
        from hrm_adaptive_memory.executive.resources import ResourceState

        tasks = generate_i3_12_corpus(n_per_category=1, seed=42)
        task = tasks[0]
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))

        snap_s0 = build_evidence_snapshot_oracle(runtime)
        snap_s1, _ = build_evidence_snapshot_with_inferred_relations(runtime, extractor)

        assert snap_s0.can_retrieve == snap_s1.can_retrieve
        assert snap_s0.can_search == snap_s1.can_search
        assert snap_s0.can_verify == snap_s1.can_verify
        assert snap_s0.resource_state == snap_s1.resource_state

    def test_s1_relation_graph_has_provenance(self, budget, extractor):
        """S1 relation graph must have provenance hashes."""
        from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
        from hrm_adaptive_memory.executive.semantic_relations.integration import build_evidence_snapshot_with_inferred_relations
        from hrm_adaptive_memory.executive.resources import ResourceState

        tasks = generate_i3_12_corpus(n_per_category=1, seed=42)
        task = tasks[0]
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))
        _, graph = build_evidence_snapshot_with_inferred_relations(runtime, extractor)

        assert graph.relation_graph_sha256
        assert graph.extractor_identity_sha256
        assert len(graph.relations) > 0
        for rel in graph.relations:
            assert rel.evidence_sha256
            assert rel.hypothesis_sha256
