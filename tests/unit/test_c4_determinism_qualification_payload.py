"""Tests for scripts/c4_determinism_qualification.py's blank-field guard.

_assert_payload_complete() exists to fail closed when a compared field was
never actually computed (a blank hash would trivially compare equal across
seeds and manufacture a false PASS). But identity_canonical has a legitimate
blank state -- identity resolution abstains (status UNRESOLVED or AMBIGUOUS)
rather than guess, per identity_stage.py's "never guess" rule -- and treating
that as a payload defect turned a correct abstain into a false ABORT the
first time the qualification split (500 tasks) contained an AMBIGUOUS case
the 120-task development split never happened to exercise.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_determinism_qualification", ROOT / "scripts/c4_determinism_qualification.py")
dq_mod = importlib.util.module_from_spec(_spec)
sys.modules["_determinism_qualification"] = dq_mod
_spec.loader.exec_module(dq_mod)


def _record(**overrides) -> dict:
    base = {
        "task_id": "t-1",
        "query_text": "q",
        "candidate_ids": ["a", "b"],
        "candidate_pool_hash": "h1",
        "identity_status": "AMBIGUOUS",
        "identity_canonical": "",
        "selected_ids": ["a"],
        "membership_hash": "h2",
        "order_hash": "h3",
        "ordered_selected_ids": ["a"],
        "packet_hash": "h4",
        "prompt_hash": "h5",
    }
    base.update(overrides)
    return base


class TestBlankIdentityCanonicalIsAllowed:
    def test_blank_canonical_with_ambiguous_status_passes(self):
        """The exact real-world case: AMBIGUOUS identity, blank canonical."""
        dq_mod._assert_payload_complete([_record()])

    def test_blank_canonical_with_unresolved_status_passes(self):
        dq_mod._assert_payload_complete(
            [_record(identity_status="UNRESOLVED", identity_canonical=None)])

    def test_non_blank_canonical_still_works(self):
        dq_mod._assert_payload_complete(
            [_record(identity_status="RESOLVED", identity_canonical="Some Entity")])


class TestOtherBlankFieldsStillFailClosed:
    """Every other field has no legitimate blank state -- a blank there still
    means the computation was skipped and must still abort."""

    @pytest.mark.parametrize("key,blank_value", [
        ("query_text", ""),
        ("candidate_ids", []),
        ("candidate_pool_hash", ""),
        ("identity_status", ""),
        ("selected_ids", []),
        ("membership_hash", ""),
        ("order_hash", ""),
        ("ordered_selected_ids", []),
        ("packet_hash", ""),
        ("prompt_hash", ""),
    ])
    def test_blank_field_still_raises(self, key, blank_value):
        with pytest.raises(AssertionError, match=f"Field {key!r} is empty"):
            dq_mod._assert_payload_complete([_record(**{key: blank_value})])


class TestStructuralGuardsUnaffected:
    def test_missing_field_in_payload_still_raises(self):
        record = _record()
        del record["packet_hash"]
        with pytest.raises(AssertionError, match="missing compared fields"):
            dq_mod._assert_payload_complete([record])

    def test_empty_records_list_still_raises(self):
        with pytest.raises(AssertionError, match="no records"):
            dq_mod._assert_payload_complete([])
