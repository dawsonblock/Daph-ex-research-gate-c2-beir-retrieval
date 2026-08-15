"""Tests for hrm_adaptive_memory/c4/selector_v2.py (Sprint B2, c4_selector_v1).

The load-bearing properties, in order of how badly a bug would mislead:

  1. No forbidden signal is read. record_kind values here are generator
     answer-key labels, so a selector consulting them would be reading the
     answer key rather than selecting.
  2. One-hop reachability stays ONE hop. Unbounded expansion is the obvious
     mechanism-creep failure and would silently turn S1 into graph search.
  3. The packet budget never grows. Otherwise a measured gain could come from
     handing the reader more evidence instead of better selection.
  4. RESOLVED is untouched, since its keep-rate is already ~92% and the arm
     exists to repair EXACT without disturbing that.
  5. Tie-breaks are exactly the frozen order, so replay is deterministic.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.c4.selector_v2 import (
    DIRECT_SUBJECT_TARGET, ONE_HOP_BRIDGE_TARGET, connectivity_status,
    find_protected_record, is_current, is_stale, one_hop_bridge_entities,
    select_s1, _candidates)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/gate_c4_selector_v1.json"
QUESTION = "Which ownership tier applies to Sparrow intake manifold?"
SUBJECT = "Sparrow intake manifold"


def _frozen(order):
    """Stand-in for the certified selector: returns its first `budget` items."""
    def select(budget):
        return list(order[:budget])
    return select


def _executable_source(module) -> str:
    """Module source with docstrings and comments stripped.

    The prose in this module legitimately NAMES the forbidden signals in order
    to explain why they are forbidden, so a naive substring scan over the raw
    source would flag its own documentation. What matters is that no executable
    statement reads them.
    """
    import io
    import tokenize

    source = inspect.getsource(module)
    kept: list[str] = []
    previous_type = tokenize.INDENT
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        # A STRING that stands alone as a statement is a docstring.
        if token.type == tokenize.STRING and previous_type in (
                tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
                tokenize.NL, tokenize.ENCODING):
            previous_type = token.type
            continue
        kept.append(token.string)
        if token.type not in (tokenize.NL, tokenize.NEWLINE):
            previous_type = token.type
    return " ".join(kept)


class TestNoForbiddenSignals:
    @pytest.mark.parametrize("forbidden", [
        "record_kind", "_oracle_metadata", "proof_edges", "latent_bridge",
        "answer_node", "required_evidence_ids", "oracle_evidence_ids"])
    def test_module_never_references_an_oracle_field(self, forbidden):
        import hrm_adaptive_memory.c4.selector_v2 as mod
        assert forbidden not in _executable_source(mod), (
            f"{forbidden} is an answer-key signal and must not reach the selector")

    def test_the_scan_would_actually_catch_a_violation(self):
        """Guards the guard: a stripped-source scan that silently matched
        nothing would pass this whole class vacuously."""
        import hrm_adaptive_memory.c4.selector_v2 as mod
        stripped = _executable_source(mod)
        assert "find_protected_record" in stripped, "scan lost real code"
        assert "one_hop_bridge_entities" in stripped

    def test_protocol_lists_the_same_forbidden_signals(self):
        audit = json.loads(PROTOCOL.read_text())[
            "LEAKAGE_CORRECTION_APPLIED"]["runtime_signal_audit"]
        assert "record_kind" in audit["forbidden"]


class TestOneHopBoundedness:
    def test_finds_an_entity_one_hop_from_the_subject(self):
        rows = _candidates(
            ["r1"], {"r1": "Sparrow intake manifold registered asset "
                           "Finch control module."}, None)
        assert "Finch control module" in one_hop_bridge_entities(rows, SUBJECT)

    def test_does_not_expand_transitively(self):
        """subject -> A must NOT yield B reachable only from A. Two hops is a
        different mechanism and belongs in a successor protocol."""
        texts = {
            "r1": "Sparrow intake manifold registered asset Finch control module.",
            "r2": "Finch control module registered asset Heron drive cluster.",
        }
        rows = _candidates(["r1", "r2"], texts, None)
        found = one_hop_bridge_entities(rows, SUBJECT)
        assert "Finch control module" in found
        assert "Heron drive cluster" not in found, "reachability must stay one hop"

    def test_subject_itself_is_not_reported_as_a_bridge(self):
        rows = _candidates(["r1"], {"r1": f"{SUBJECT} has ownership tier 1."}, None)
        assert SUBJECT not in one_hop_bridge_entities(rows, SUBJECT)


class TestEligibilityAndAnchoring:
    def test_direct_subject_anchor(self):
        texts = {"r1": f"{SUBJECT} has ownership tier 3529."}
        receipt = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["r1"], texts=texts)
        assert receipt.protection_reason == DIRECT_SUBJECT_TARGET
        assert receipt.bridge_entity is None

    def test_bridge_anchor_reaches_the_defect_population(self):
        """The case the original subject+relation rule could not reach."""
        texts = {
            "r1": f"{SUBJECT} registered asset Finch control module.",
            "r2": "- updated Finch control module: ownership tier now 3529",
        }
        receipt = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["r1", "r2"], texts=texts)
        assert receipt.protection_reason == ONE_HOP_BRIDGE_TARGET
        assert receipt.bridge_entity == "Finch control module"
        assert receipt.protected_record_id == "r2"

    def test_relation_is_required(self):
        texts = {"r1": f"{SUBJECT} was commissioned in March."}
        assert find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["r1"], texts=texts) is None

    def test_no_canonical_subject_declines(self):
        texts = {"r1": f"{SUBJECT} has ownership tier 3529."}
        assert find_protected_record(
            question=QUESTION, canonical_subject=None,
            candidate_ids=["r1"], texts=texts) is None

    def test_receipt_records_the_full_anchoring_chain(self):
        """Auditability requirement: every protection must be explainable."""
        texts = {"r1": f"{SUBJECT} registered asset Finch control module.",
                 "r2": "Finch control module ownership tier 3529"}
        summary = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["r1", "r2"], texts=texts).summary()
        for field in ("protection_reason", "anchor_subject", "bridge_entity",
                      "target_relation", "protected_record_id"):
            assert field in summary


class TestFrozenTieBreakOrder:
    def test_direct_anchor_preferred_over_bridge_anchor(self):
        texts = {
            "bridge_rec": f"{SUBJECT} registered asset Finch control module.",
            "z_direct": f"{SUBJECT} ownership tier 1.",
            "a_bridge": "Finch control module ownership tier 2.",
        }
        receipt = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["bridge_rec", "a_bridge", "z_direct"], texts=texts)
        assert receipt.protection_reason == DIRECT_SUBJECT_TARGET

    def test_higher_fusion_score_wins_within_the_same_anchor_class(self):
        texts = {"lo": f"{SUBJECT} ownership tier 1.",
                 "hi": f"{SUBJECT} ownership tier 2."}
        receipt = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["lo", "hi"], texts=texts,
            fusion_scores={"lo": 0.1, "hi": 0.9})
        assert receipt.protected_record_id == "hi"

    def test_rank_breaks_ties_when_scores_are_equal(self):
        texts = {"first": f"{SUBJECT} ownership tier 1.",
                 "second": f"{SUBJECT} ownership tier 2."}
        receipt = find_protected_record(
            question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["first", "second"], texts=texts,
            fusion_scores={"first": 0.5, "second": 0.5})
        assert receipt.protected_record_id == "first"

    def test_is_deterministic_across_repeated_calls(self):
        texts = {f"r{i}": f"{SUBJECT} ownership tier {i}." for i in range(6)}
        ids = list(texts)
        first = find_protected_record(question=QUESTION, canonical_subject=SUBJECT,
                                      candidate_ids=ids, texts=texts).summary()
        for _ in range(5):
            assert find_protected_record(
                question=QUESTION, canonical_subject=SUBJECT,
                candidate_ids=ids, texts=texts).summary() == first


class TestPacketBudgetInvariant:
    def test_protection_consumes_a_slot_and_never_grows_the_packet(self):
        texts = {f"f{i}": f"filler {i}" for i in range(10)}
        texts["p"] = f"{SUBJECT} ownership tier 3529."
        order = [f"f{i}" for i in range(10)]
        selected, receipt = select_s1(
            identity_status="EXACT", question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=["p"] + order, texts=texts, budget=6,
            frozen_select=_frozen(order))
        assert receipt is not None
        assert len(selected) == 6, "packet must not grow"
        assert selected[0] == "p"
        assert len(set(selected)) == 6, "no duplicates"

    def test_protected_record_is_not_double_counted(self):
        """If the frozen selector also picks the protected record, the packet
        must not contain it twice nor silently shrink."""
        texts = {"p": f"{SUBJECT} ownership tier 1.",
                 **{f"f{i}": f"filler {i}" for i in range(8)}}
        order = ["p"] + [f"f{i}" for i in range(8)]
        selected, _ = select_s1(
            identity_status="EXACT", question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=order, texts=texts, budget=6,
            frozen_select=_frozen(order))
        assert selected.count("p") == 1
        assert len(selected) == 6


class TestFallbackBehavior:
    def test_resolved_identity_is_left_completely_unchanged(self):
        """RESOLVED keep-rate is already ~92%; the arm must not disturb it."""
        texts = {"p": f"{SUBJECT} ownership tier 1.", "f": "filler"}
        order = ["f", "p"]
        selected, receipt = select_s1(
            identity_status="RESOLVED", question=QUESTION,
            canonical_subject=SUBJECT, candidate_ids=order, texts=texts,
            budget=2, frozen_select=_frozen(order))
        assert receipt is None
        assert selected == order

    def test_exact_with_nothing_eligible_falls_back_unchanged(self):
        texts = {"f1": "unrelated", "f2": "also unrelated"}
        order = ["f1", "f2"]
        selected, receipt = select_s1(
            identity_status="EXACT", question=QUESTION, canonical_subject=SUBJECT,
            candidate_ids=order, texts=texts, budget=2,
            frozen_select=_frozen(order))
        assert receipt is None
        assert selected == order

    def test_unresolved_identity_falls_back_unchanged(self):
        texts = {"p": f"{SUBJECT} ownership tier 1."}
        selected, receipt = select_s1(
            identity_status="UNRESOLVED", question=QUESTION,
            canonical_subject=None, candidate_ids=["p"], texts=texts,
            budget=1, frozen_select=_frozen(["p"]))
        assert receipt is None and selected == ["p"]


class TestConnectivityAndTemporalSignals:
    def test_disconnected_records_are_identified(self):
        assert connectivity_status("Totally unrelated text.", SUBJECT, ()) == \
            "DISCONNECTED"

    def test_subject_and_bridge_connections_are_distinguished(self):
        assert connectivity_status(f"{SUBJECT} x", SUBJECT, ()) == "CONNECTED_SUBJECT"
        assert connectivity_status("Finch control module x", SUBJECT,
                                   ("Finch control module",)) == "CONNECTED_BRIDGE"

    def test_identity_records_are_exempt_from_connectivity(self):
        """A surface->canonical mapping mentions neither subject nor bridge but
        is legitimately required, so it must not be rejected as disconnected."""
        assert connectivity_status("QCM-4 is the short code for Quail control "
                                   "module.", SUBJECT, ()) == "IDENTITY"

    def test_temporal_markers_are_read_from_content(self):
        assert is_stale("Revision 1 (effective 2030-01-01, since superseded) "
                        "recorded: ...")
        assert is_current("Revision 2 (effective 2031-06-01) supersedes "
                          "revision 1: ...")
        assert not is_stale("A plain record with no revision marker.")
