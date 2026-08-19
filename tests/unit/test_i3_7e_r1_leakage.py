"""Tests for I3.7e-r1 leakage-clean minimal decision state.

Source-level tests:
  - build_minimal_decision_state_packet must not accept EvidenceRuntime
  - must not reference hidden_evidence, verify_result, expected_terminal,
    correct_hypothesis_id, oracle_resolution_path

Behavioral test:
  - Construct two runtimes with identical controller-visible snapshots
    but different hidden evidence identities/relationships
  - Require M(O_1) == M(O_2)
"""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)

from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
    EvidenceItem, EvidenceHypothesis, EvidenceTask, EvidenceSnapshot,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    build_evidence_snapshot,
)

# Import the script as a module
script_path = ROOT / "scripts" / "run_i3_7e_compact_governor.py"
import importlib.util
spec = importlib.util.spec_from_file_location("i3_7e_script", script_path)
i3_7e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i3_7e)


# ---------------------------------------------------------------------------
# Source-level tests
# ---------------------------------------------------------------------------

class TestSourceLevelLeakage:
    """Verify the M builder signature and source do not reference latent state."""

    def test_signature_does_not_accept_evidence_runtime(self):
        """build_minimal_decision_state_packet must not take EvidenceRuntime."""
        sig = inspect.signature(i3_7e.build_minimal_decision_state_packet)
        param_types = []
        for name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types.append(str(param.annotation))
        # Must not contain EvidenceRuntime
        for pt in param_types:
            assert "EvidenceRuntime" not in pt, \
                f"build_minimal_decision_state_packet accepts EvidenceRuntime: {pt}"
            assert "EvidenceTask" not in pt, \
                f"build_minimal_decision_state_packet accepts EvidenceTask: {pt}"

    def test_source_does_not_reference_hidden_evidence(self):
        """The M builder source must not access runtime.hidden_evidence."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        # Check for runtime access to hidden_evidence (the leak pattern)
        assert "runtime.hidden_evidence" not in source, \
            "build_minimal_decision_state_packet accesses runtime.hidden_evidence"
        # Also check for direct hidden_evidence access on any variable
        # (but snapshot.hidden_evidence_count is allowed — it's the count, not the items)
        import re
        # Disallow patterns like "X.hidden_evidence" where X is not "snapshot"
        # or "hidden_evidence[" (direct list access)
        assert not re.search(r'(?<!snapshot\.)hidden_evidence(?!\w)', source), \
            "build_minimal_decision_state_packet accesses hidden_evidence directly"

    def test_source_does_not_reference_verify_result(self):
        """The M builder source must not reference verify_result."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        assert "verify_result" not in source, \
            "build_minimal_decision_state_packet references verify_result"

    def test_source_does_not_reference_expected_terminal(self):
        """The M builder source must not reference expected_terminal."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        assert "expected_terminal" not in source, \
            "build_minimal_decision_state_packet references expected_terminal"

    def test_source_does_not_reference_correct_hypothesis_id(self):
        """The M builder source must not reference correct_hypothesis_id."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        assert "correct_hypothesis_id" not in source, \
            "build_minimal_decision_state_packet references correct_hypothesis_id"

    def test_source_does_not_reference_oracle(self):
        """The M builder source must not reference oracle paths."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        assert "oracle" not in source.lower(), \
            "build_minimal_decision_state_packet references oracle"

    def test_source_does_not_access_runtime(self):
        """The M builder source must not access a runtime variable."""
        source = inspect.getsource(i3_7e.build_minimal_decision_state_packet)
        # Check for runtime variable access (runtime. or runtime, or runtime))
        # but allow the word "runtime" in docstrings
        import re
        # Remove docstrings before checking
        source_no_doc = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
        source_no_doc = re.sub(r"'''.*?'''", '', source_no_doc, flags=re.DOTALL)
        # Check for runtime as a variable access
        assert not re.search(r'\bruntime\b', source_no_doc), \
            "build_minimal_decision_state_packet accesses a runtime variable"

    def test_snapshot_only_helper_does_not_reference_runtime(self):
        """_classify_from_snapshot must not reference runtime."""
        source = inspect.getsource(i3_7e._classify_from_snapshot)
        import re
        source_no_doc = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
        assert not re.search(r'\bruntime\b', source_no_doc), \
            "_classify_from_snapshot references runtime"
        assert not re.search(r'\bEvidenceRuntime\b', source_no_doc), \
            "_classify_from_snapshot references EvidenceRuntime"

    def test_snapshot_only_acs_does_not_reference_runtime(self):
        """_answer_condition_from_snapshot must not reference runtime."""
        source = inspect.getsource(i3_7e._answer_condition_from_snapshot)
        assert "runtime" not in source, \
            "_answer_condition_from_snapshot references runtime"


# ---------------------------------------------------------------------------
# Behavioral test: identical snapshots, different hidden evidence → identical M
# ---------------------------------------------------------------------------

def _make_hypotheses():
    return (
        EvidenceHypothesis(
            hypothesis_id="H1",
            proposition="claim is true",
            answer_action=DecisionAction.ANSWER,
            answer_payload="yes",
        ),
        EvidenceHypothesis(
            hypothesis_id="H2",
            proposition="claim is false",
            answer_action=DecisionAction.DEFER,
            answer_payload="no",
        ),
    )


def _make_visible_evidence():
    """Two visible evidence items, both unverified."""
    return (
        EvidenceItem(
            evidence_id="E1",
            proposition="Source A supports H1",
            source_class="initial",
            supports=("H1",),
            contradicts=(),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="SUFFICIENT",
        ),
        EvidenceItem(
            evidence_id="E2",
            proposition="Source B contradicts H1",
            source_class="initial",
            supports=("H2",),
            contradicts=("H1",),
            verification_state=VerificationState.UNVERIFIED,
            temporal_status=TemporalStatus.CURRENT,
            retrieved=True,
            verify_result="FALSIFIED",
        ),
    )


def _make_task_with_hidden(hidden_items):
    """Create a task with given hidden evidence items."""
    visible = _make_visible_evidence()
    all_evidence = visible + hidden_items
    return EvidenceTask(
        task_id="test_behavioral_leakage_v1",
        split="test",
        category="evidence_conflict",
        task_summary="Test task for leakage",
        high_stakes=False,
        budget_profile="STANDARD",
        hypotheses=_make_hypotheses(),
        evidence_items=all_evidence,
        retrieve_exposes=tuple(e.evidence_id for e in hidden_items),
        search_exposes=(),
        oracle_resolution_path=("RETRIEVE:E3", "VERIFY:E3", "ANSWER"),
        expected_terminal=DecisionAction.ANSWER,
        correct_hypothesis_id="H1",
    )


class TestBehavioralLeakage:
    """Identical visible state + different hidden evidence → identical M packets."""

    def test_identical_snapshots_different_hidden_evidence(self):
        """Two runtimes with same visible state but different hidden evidence
        must produce identical M packets."""
        from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

        budget = ResourceBudget(
            max_executive_steps=24, max_reasoning_tokens=2048,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
        )

        # Hidden evidence set 1: E3 supports H1, E4 contradicts H2
        hidden_1 = (
            EvidenceItem(
                evidence_id="E3",
                proposition="Hidden source confirms H1",
                source_class="search",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
            EvidenceItem(
                evidence_id="E4",
                proposition="Hidden source refutes H2",
                source_class="search",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        # Hidden evidence set 2: completely different IDs and relationships
        hidden_2 = (
            EvidenceItem(
                evidence_id="E99",
                proposition="Completely different hidden evidence",
                source_class="search",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.STALE,
                retrieved=False,
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E100",
                proposition="Another different hidden item",
                source_class="search",
                supports=("H2",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        task_1 = _make_task_with_hidden(hidden_1)
        task_2 = _make_task_with_hidden(hidden_2)

        runtime_1 = initial_evidence_runtime(task_1, ResourceState(budget))
        runtime_2 = initial_evidence_runtime(task_2, ResourceState(budget))

        # Build snapshots — should be identical (same visible evidence)
        snapshot_1 = build_evidence_snapshot(runtime_1)
        snapshot_2 = build_evidence_snapshot(runtime_2)

        # Verify snapshots are identical in visible content
        assert snapshot_1.hidden_evidence_count == snapshot_2.hidden_evidence_count
        assert len(snapshot_1.visible_evidence) == len(snapshot_2.visible_evidence)
        for e1, e2 in zip(snapshot_1.visible_evidence, snapshot_2.visible_evidence):
            assert e1.evidence_id == e2.evidence_id
            assert e1.verification_state == e2.verification_state

        # Build M packets from snapshots only
        packet_1 = i3_7e.build_minimal_decision_state_packet(snapshot_1)
        packet_2 = i3_7e.build_minimal_decision_state_packet(snapshot_2)

        # The decision_state_summary must be identical
        assert packet_1["decision_state_summary"] == packet_2["decision_state_summary"], \
            f"M packets differ despite identical visible state:\n" \
            f"  packet_1: {json.dumps(packet_1['decision_state_summary'], indent=2)}\n" \
            f"  packet_2: {json.dumps(packet_2['decision_state_summary'], indent=2)}"

    def test_identical_snapshots_different_hidden_count(self):
        """Different number of hidden items but same visible state
        must produce M packets that differ only in blocker rationale,
        not in decision_state or hypothesis classification."""
        from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

        budget = ResourceBudget(
            max_executive_steps=24, max_reasoning_tokens=2048,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
        )

        # Task with 1 hidden item
        hidden_few = (
            EvidenceItem(
                evidence_id="E3",
                proposition="Hidden 1",
                source_class="search",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        # Task with 3 hidden items (different IDs, different relationships)
        hidden_many = (
            EvidenceItem(
                evidence_id="E50",
                proposition="Hidden A",
                source_class="search",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="FALSIFIED",
            ),
            EvidenceItem(
                evidence_id="E51",
                proposition="Hidden B",
                source_class="search",
                supports=("H1",),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.STALE,
                retrieved=False,
                verify_result="STALE",
            ),
            EvidenceItem(
                evidence_id="E52",
                proposition="Hidden C",
                source_class="search",
                supports=("H2",),
                contradicts=("H1",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        task_few = _make_task_with_hidden(hidden_few)
        task_many = _make_task_with_hidden(hidden_many)

        runtime_few = initial_evidence_runtime(task_few, ResourceState(budget))
        runtime_many = initial_evidence_runtime(task_many, ResourceState(budget))

        snapshot_few = build_evidence_snapshot(runtime_few)
        snapshot_many = build_evidence_snapshot(runtime_many)

        # Hidden counts differ
        assert snapshot_few.hidden_evidence_count == 1
        assert snapshot_many.hidden_evidence_count == 3

        # Both have hidden evidence, so both should say NEEDS_EVIDENCE
        # with evidence_id=null (cannot name hidden items)
        packet_few = i3_7e.build_minimal_decision_state_packet(snapshot_few)
        packet_many = i3_7e.build_minimal_decision_state_packet(snapshot_many)

        # decision_state must be the same (both have unverified visible evidence)
        assert packet_few["decision_state_summary"]["decision_state"] == \
               packet_many["decision_state_summary"]["decision_state"], \
            "Decision state differs despite same visible evidence"

        # If both are NEEDS_EVIDENCE, remaining_blocker must not name hidden IDs
        for packet, label in [(packet_few, "few"), (packet_many, "many")]:
            blocker = packet["decision_state_summary"]["remaining_blocker"]
            if blocker and blocker.get("evidence_id"):
                # If evidence_id is set, it must be a visible evidence ID
                visible_ids = {e.evidence_id for e in snapshot_few.visible_evidence}
                assert blocker["evidence_id"] in visible_ids, \
                    f"M packet ({label}) names hidden evidence ID: {blocker['evidence_id']}"

    def test_no_hidden_evidence_ids_in_packet(self):
        """M packet must never contain hidden evidence IDs in any field."""
        from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

        budget = ResourceBudget(
            max_executive_steps=24, max_reasoning_tokens=2048,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
        )

        hidden = (
            EvidenceItem(
                evidence_id="E_SECRET_42",
                proposition="Hidden evidence with distinctive ID",
                source_class="search",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        task = _make_task_with_hidden(hidden)
        runtime = initial_evidence_runtime(task, ResourceState(budget))
        snapshot = build_evidence_snapshot(runtime)

        packet = i3_7e.build_minimal_decision_state_packet(snapshot)
        packet_str = json.dumps(packet)

        # The hidden evidence ID must never appear in the packet
        assert "E_SECRET_42" not in packet_str, \
            "Hidden evidence ID leaked into M packet"

    def test_ready_to_answer_after_verification(self):
        """After verifying visible evidence, M should say READY_TO_ANSWER
        without needing any hidden evidence information."""
        from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState

        budget = ResourceBudget(
            max_executive_steps=24, max_reasoning_tokens=2048,
            max_retrieval_calls=5, max_verification_calls=5,
            max_search_calls=5, max_elapsed_ms=10000,
        )

        hidden = (
            EvidenceItem(
                evidence_id="E3",
                proposition="Hidden evidence",
                source_class="search",
                supports=("H1",),
                contradicts=("H2",),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=False,
                verify_result="SUFFICIENT",
            ),
        )

        task = _make_task_with_hidden(hidden)
        runtime = initial_evidence_runtime(task, ResourceState(budget))
        executor = EvidenceExecutor()

        # VERIFY E2 (contradicts H1, verify_result=FALSIFIED → H1's contradiction eliminated)
        res1 = executor.execute(runtime, DecisionAction.VERIFY)
        # VERIFY E1 (supports H1, verify_result=SUFFICIENT → H1 has verified support)
        res2 = executor.execute(res1.runtime, DecisionAction.VERIFY)

        snapshot = build_evidence_snapshot(
            res2.runtime,
            prior_actions=("VERIFY", "VERIFY"),
            prior_outcomes=("VERIFY_COMPLETED", "VERIFY_COMPLETED"),
        )

        packet = i3_7e.build_minimal_decision_state_packet(snapshot)
        ds = packet["decision_state_summary"]

        assert ds["decision_state"] == "READY_TO_ANSWER", \
            f"Expected READY_TO_ANSWER, got {ds['decision_state']}: {json.dumps(ds, indent=2)}"
        assert ds["remaining_blocker"] is None
        assert "H1" in ds["live_hypotheses"]
        # H2's support (E2) was FALSIFIED, so H2 is weakened, not eliminated
        # (eliminated would require a SUFFICIENT contradiction against H2)
        assert "H2" not in ds["live_hypotheses"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
