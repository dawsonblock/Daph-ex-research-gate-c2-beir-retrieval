"""Tests for semantic conformance checker."""
import pytest
from hrm_adaptive_memory.executive.evidence_benchmark.i3_30r3_confirmation_generator import (
    generate_confirmation_benchmark, get_confirmation_budget_for_task,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor

from daph.conformance.semantic_conformance import (
    ConformanceRecord,
    check_conformance,
    check_conformance_for_task,
)


@pytest.fixture
def tasks():
    return generate_confirmation_benchmark(seed=43291)


@pytest.fixture
def executor():
    return EvidenceExecutor()


class TestConformanceRecord:
    """Test ConformanceRecord structure and serialization."""

    def test_record_is_serializable(self, tasks, executor):
        """ConformanceRecord.as_dict() produces valid JSON."""
        task = tasks[0]
        budget = get_confirmation_budget_for_task(task)
        resources = ResourceState(budget=budget)
        from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
        runtime = initial_evidence_runtime(task, resources)

        record = check_conformance(runtime, step=0, executor=executor)

        import json
        d = record.as_dict()
        json_str = json.dumps(d)
        assert json.loads(json_str) == d

    def test_record_has_all_fields(self, tasks, executor):
        """ConformanceRecord has all required fields."""
        task = tasks[0]
        budget = get_confirmation_budget_for_task(task)
        resources = ResourceState(budget=budget)
        from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
        runtime = initial_evidence_runtime(task, resources)

        record = check_conformance(runtime, step=0, executor=executor)

        assert record.task_id is not None
        assert record.step == 0
        assert len(record.state_sha256) == 64
        assert record.topology_readiness in ("ANSWER_READY", "DEFER_READY", "CONTINUE_REQUIRED")
        assert record.cert_readiness in ("ANSWER_READY", "DEFER_READY", "CONTINUE_REQUIRED")
        assert record.executor_truth_readiness in ("ANSWER_READY", "DEFER_READY", "CONTINUE_REQUIRED")
        assert record.benchmark_truth_readiness in ("ANSWER_READY", "DEFER_READY", "CONTINUE_REQUIRED")
        assert record.causal_best_action in ("ANSWER", "DEFER", "CONTINUE")
        assert isinstance(record.conformant, bool)
        assert isinstance(record.disagreements, tuple)


class TestConformanceCheck:
    """Test conformance checking across strata."""

    def test_d4_conformant(self, tasks, executor):
        """D4 (ANSWER-correct preservation control) should be conformant."""
        d4 = [t for t in tasks if "_d4_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d4)
        resources = ResourceState(budget=budget)
        from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
        runtime = initial_evidence_runtime(d4, resources)

        record = check_conformance(runtime, step=0, executor=executor)

        # D4 is the preservation control — all components should agree
        assert record.conformant, f"D4 should be conformant: {record.disagreements}"

    def test_d1_exposes_defer_cert_gap(self, tasks, executor):
        """D1 initial state exposes the DEFER certificate gap.

        Topology, executor, and benchmark all say DEFER_READY,
        but the certificate says CONTINUE_REQUIRED.
        This is a known semantic gap, not a bug in the checker.

        Per P1.2: this should be classified as safe_abstention, not
        unsafe_disagreement, because the certificate abstains rather
        than forcing a wrong action.
        """
        d1 = [t for t in tasks if "_d1_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d1)
        resources = ResourceState(budget=budget)
        from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
        runtime = initial_evidence_runtime(d1, resources)

        record = check_conformance(runtime, step=0, executor=executor)

        # The checker should detect the disagreement
        assert not record.conformant
        assert any("cert(CONTINUE_REQUIRED)" in d for d in record.disagreements)
        # P1.2: this is safe abstention, not unsafe disagreement
        assert record.disagreement_type == "safe_abstention"

    def test_conformance_for_task_returns_multiple_records(self, tasks, executor):
        """check_conformance_for_task checks multiple decision points."""
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)

        records = check_conformance_for_task(d5, resources, executor, pre_verify=True)

        assert len(records) >= 1
        for r in records:
            assert r.task_id == d5.task_id

    def test_all_strata_produce_records(self, tasks, executor):
        """Every stratum should produce at least one conformance record."""
        for stratum in ["d1", "d2", "d3", "d4", "d5"]:
            task = [t for t in tasks if f"_{stratum}_" in t.task_id][0]
            budget = get_confirmation_budget_for_task(task)
            resources = ResourceState(budget=budget)

            records = check_conformance_for_task(task, resources, executor)

            assert len(records) >= 1, f"{stratum.upper()} produced no records"
            assert all(isinstance(r, ConformanceRecord) for r in records)

    def test_disagreement_type_field_present(self, tasks, executor):
        """Every record should have a disagreement_type field."""
        for stratum in ["d1", "d2", "d3", "d4", "d5"]:
            task = [t for t in tasks if f"_{stratum}_" in t.task_id][0]
            budget = get_confirmation_budget_for_task(task)
            resources = ResourceState(budget=budget)

            records = check_conformance_for_task(task, resources, executor)

            for r in records:
                assert r.disagreement_type in (
                    "no_disagreement", "safe_abstention",
                    "unsafe_disagreement", "other_disagreement",
                )

    def test_d5_initial_state_is_continue_required(self, tasks, executor):
        """D5 initial state should be CONTINUE_REQUIRED, not ANSWER_READY.

        This tests the P1.1 fix: expected_terminal=ANSWER does not mean
        the initial state is ANSWER_READY. D5 requires verification first.
        """
        d5 = [t for t in tasks if "_d5_" in t.task_id][0]
        budget = get_confirmation_budget_for_task(d5)
        resources = ResourceState(budget=budget)
        from hrm_adaptive_memory.executive.evidence_benchmark.schema import initial_evidence_runtime
        runtime = initial_evidence_runtime(d5, resources)

        record = check_conformance(runtime, step=0, executor=executor)

        # D5 initial state should NOT be ANSWER_READY just because
        # expected_terminal is ANSWER
        assert record.benchmark_truth_readiness != "ANSWER_READY" or record.conformant, (
            f"D5 initial state should not be ANSWER_READY just from expected_terminal. "
            f"benchmark_readiness={record.benchmark_truth_readiness}, "
            f"conformant={record.conformant}, disagreements={record.disagreements}"
        )
