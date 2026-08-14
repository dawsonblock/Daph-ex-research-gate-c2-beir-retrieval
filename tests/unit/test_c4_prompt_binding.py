"""Tests that the ordered packet is what HRM actually consumes.

Under protocol v1 the packet stage computed a deterministic order and hashed it,
but the runner then rebuilt the prompt from ``selection.selected_ids``:

    full_prompt = compose_evidence_prompt(
        task["question"],
        [texts.get(eid) for eid in pre_hrm.selection.selected_ids ...])

So the ordering policy never reached the model. Every cell of the
membership x ordering 2x2 produced a byte-identical prompt, which would have
reported an ordering effect of exactly 0.0000 for structural reasons rather
than empirical ones. These tests pin the fix.
"""
from __future__ import annotations

import pytest

from hrm_adaptive_memory.c4.arms import ARMS
from hrm_adaptive_memory.c4.contracts import RetrievalResult, SelectionResult
from hrm_adaptive_memory.c4.packet_ordering import ORDERING_POLICY_ID
from hrm_adaptive_memory.c4.packet_stage import (
    _DETERMINISTIC_ORDER_ARMS, run_packet_stage)

QUESTION = "Who owns the Northwind account?"

# Deliberately in an order that the role policy must change: a distractor and a
# value ahead of the identity record.
POOL_ORDER = ("t/distractor", "t/value", "t/identity", "t/link")

TEXTS = {
    "t/identity": "Northwind Corp is also known as NWC.",
    "t/link": "NWC is administered by the Atlas group.",
    "t/value": "The Atlas group owner is Dana Reed.",
    "t/distractor": "Southwind Corp is owned by Ravi Patel.",
}


def _selection(selector: str = "s2c") -> SelectionResult:
    return SelectionResult(
        selector=selector,
        selected_ids=POOL_ORDER,
        selector_policy="s2c_with_s0_fallback",
        identity_status="EXACT",
    )


def _retrieval(candidate_ids=POOL_ORDER, scores=None) -> RetrievalResult:
    fusion = tuple((eid, (scores or {}).get(eid, 0.0)) for eid in candidate_ids)
    return RetrievalResult(
        bm25_ranked=fusion,
        bge_ranked=(),
        fusion_ranked=fusion,
        candidate_ids=tuple(candidate_ids),
        candidate_budget=50,
        retrieval_policy="bm25_bge_fusion",
        bm25_backend="local",
        bge_model_id="bge-small",
        bge_revision="rev",
        rrf_k=10,
    )


class TestOrderingReachesThePrompt:
    def test_c4_4_reorders_the_packet(self):
        prompt, packet = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS, retrieval=_retrieval())
        assert packet.ordering_applied is True
        assert packet.ordering_policy_id == ORDERING_POLICY_ID
        assert packet.packet_ids[0] == "t/identity"
        assert packet.packet_ids[-1] == "t/distractor"
        # The reordering must be visible in the prompt text, not just the hash.
        assert prompt.index(TEXTS["t/identity"]) < prompt.index(TEXTS["t/value"])
        assert prompt.index(TEXTS["t/distractor"]) > prompt.index(TEXTS["t/value"])

    def test_c4_4m_keeps_pool_order(self):
        prompt, packet = run_packet_stage(
            ARMS["C4_4m"], QUESTION, _selection(), TEXTS, retrieval=_retrieval())
        assert packet.ordering_applied is False
        assert packet.ordering_policy_id == "pool_order"
        assert packet.packet_ids == POOL_ORDER
        assert prompt.index(TEXTS["t/distractor"]) < prompt.index(TEXTS["t/identity"])

    def test_prompt_hash_binds_the_prompt_text(self):
        from hrm_adaptive_memory.c4.packet_ordering import canonical_prompt_hash
        prompt, packet = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS, retrieval=_retrieval())
        assert packet.prompt_hash == canonical_prompt_hash(prompt)


class TestTwoByTwoIsNonDegenerate:
    """Each cell of the 2x2 must be distinguishable from the others."""

    def _cell(self, arm_id: str):
        return run_packet_stage(ARMS[arm_id], QUESTION, _selection(), TEXTS,
                                retrieval=_retrieval())

    def test_c4_4_and_c4_4m_share_membership(self):
        _, a = self._cell("C4_4")
        _, b = self._cell("C4_4m")
        assert a.membership_hash == b.membership_hash

    def test_c4_4_and_c4_4m_differ_in_order(self):
        _, a = self._cell("C4_4")
        _, b = self._cell("C4_4m")
        assert a.order_hash != b.order_hash

    def test_c4_4_and_c4_4m_differ_in_prompt(self):
        """The regression guard: identical prompts made the contrast vacuous."""
        pa, a = self._cell("C4_4")
        pb, b = self._cell("C4_4m")
        assert pa != pb
        assert a.prompt_hash != b.prompt_hash

    def test_c4_3_and_c4_3o_differ_in_order_only(self):
        _, a = self._cell("C4_3")
        _, b = self._cell("C4_3o")
        assert a.membership_hash == b.membership_hash
        assert a.order_hash != b.order_hash
        assert a.prompt_hash != b.prompt_hash

    def test_exactly_two_arms_apply_ordering(self):
        assert _DETERMINISTIC_ORDER_ARMS == {"C4_4", "C4_3o"}


class TestBoundaryHashes:
    def test_all_five_hashes_populated(self):
        _, packet = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS, retrieval=_retrieval())
        for field in ("candidate_pool_hash", "membership_hash", "order_hash",
                      "packet_hash", "prompt_hash"):
            value = getattr(packet, field)
            assert value and len(value) == 64, field

    def test_membership_hash_is_order_independent(self):
        _, a = run_packet_stage(ARMS["C4_4m"], QUESTION, _selection(), TEXTS,
                                retrieval=_retrieval())
        reversed_selection = SelectionResult(
            selector="s2c", selected_ids=tuple(reversed(POOL_ORDER)),
            selector_policy="s2c_with_s0_fallback", identity_status="EXACT")
        _, b = run_packet_stage(ARMS["C4_4m"], QUESTION, reversed_selection,
                                TEXTS, retrieval=_retrieval())
        assert a.membership_hash == b.membership_hash
        assert a.order_hash != b.order_hash

    def test_candidate_pool_hash_tracks_the_pool(self):
        _, a = run_packet_stage(ARMS["C4_4"], QUESTION, _selection(), TEXTS,
                                retrieval=_retrieval())
        _, b = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS,
            retrieval=_retrieval(candidate_ids=tuple(reversed(POOL_ORDER))))
        assert a.candidate_pool_hash != b.candidate_pool_hash
        # Membership and order are unchanged: the boundary hashes localize it.
        assert a.membership_hash == b.membership_hash
        assert a.order_hash == b.order_hash

    def test_score_sources_records_what_was_available(self):
        """An inert tie-break term must be visible, not assumed active."""
        _, with_retrieval = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS, retrieval=_retrieval(
                scores={"t/value": 0.9}))
        assert with_retrieval.score_sources["retrieval_scores"] is True
        # Protocol v2_1 removes selector score from packet ordering entirely,
        # so there is no selector_scores key to report as inert.
        assert "selector_scores" not in with_retrieval.score_sources

        _, without = run_packet_stage(ARMS["C4_4"], QUESTION, _selection(), TEXTS)
        assert without.score_sources["retrieval_scores"] is False


class TestRunnerBinding:
    """The runner must fail closed if HRM is fed anything else."""

    def test_assert_prompt_binding_rejects_mismatch(self):
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "run_gate_c4_binding", root / "scripts/run_gate_c4.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        prompt, packet = run_packet_stage(
            ARMS["C4_4"], QUESTION, _selection(), TEXTS, retrieval=_retrieval())
        pre = module.PreHRMResult(
            task_id="t", arm_id="C4_4", split="development",
            query=None, retrieval=None, identity=None, selection=_selection(),
            packet=packet, information_state_before={},
            information_state_after={}, prompt=prompt)

        good = module.HRMResult(
            output="x", prompt_hash=packet.prompt_hash, prompt_tokens=1,
            completion_tokens=1, model_id="m", model_revision="r",
            latency_seconds=0.0)
        module._assert_prompt_binding(pre, good)  # must not raise

        bad = module.HRMResult(
            output="x", prompt_hash="0" * 64, prompt_tokens=1,
            completion_tokens=1, model_id="m", model_revision="r",
            latency_seconds=0.0)
        with pytest.raises(AssertionError, match="Prompt binding violated"):
            module._assert_prompt_binding(pre, bad)

    def test_resume_key_changes_with_pipeline_version(self):
        """Old receipts must not be reused after the prompt path changed."""
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "run_gate_c4_resume", root / "scripts/run_gate_c4.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        task = {"task_id": "task-1"}
        key_now = module._resume_key(task, ARMS["C4_4"], "abc")
        assert module.PIPELINE_VERSION == "c4_pipeline_v2_ordered_packet"
        module.PIPELINE_VERSION = "c4_pipeline_v1"
        assert module._resume_key(task, ARMS["C4_4"], "abc") != key_now
