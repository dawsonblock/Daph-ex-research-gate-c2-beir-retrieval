"""Regression tests for exact-build V2A qualification binding."""
from __future__ import annotations

import copy

import pytest

from hrm_adaptive_memory.external_verification import qualification as subject


def _identity() -> dict:
    return {
        "identity_version": subject.IDENTITY_VERSION,
        "source_commit": "a" * 40,
        "source_tree_hash": "b" * 40,
        "component_hashes": {"runtime": {"path": "runtime.py", "sha256": "c" * 64}},
        "test_corpus_sha256": "d" * 64,
        "qualification_lock": {"path": "lock", "sha256": "e" * 64},
        "dependency_environment": {"sha256": "f" * 64},
    }


def _receipt(identity: dict) -> dict:
    return {
        "run_valid": True,
        "source_commit": identity["source_commit"],
        "source_tree_hash": identity["source_tree_hash"],
        "qualification_identity": copy.deepcopy(identity),
    }


def test_receipts_must_all_bind_to_one_complete_identity(monkeypatch, tmp_path):
    identity = _identity()
    monkeypatch.setattr(subject, "qualification_identity", lambda _root, _commit: identity)
    receipts = {name: _receipt(identity) for name in ("pressure", "adversarial", "network", "full")}
    assert subject.validate_receipt_set(tmp_path, receipts) == identity


def test_receipt_without_identity_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="lacks qualification_identity"):
        subject.validate_receipt_set(tmp_path, {"pressure": {"run_valid": True}})


def test_cross_commit_or_component_receipt_cannot_be_mixed(monkeypatch, tmp_path):
    identity = _identity()
    monkeypatch.setattr(subject, "qualification_identity", lambda _root, _commit: identity)
    receipts = {"pressure": _receipt(identity), "full": _receipt(identity)}
    receipts["full"]["qualification_identity"]["component_hashes"]["runtime"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="common qualification identity"):
        subject.validate_receipt_set(tmp_path, receipts)


def test_receipt_identity_must_match_committed_bytes(monkeypatch, tmp_path):
    identity = _identity()
    committed = copy.deepcopy(identity)
    committed["test_corpus_sha256"] = "0" * 64
    monkeypatch.setattr(subject, "qualification_identity", lambda _root, _commit: committed)
    with pytest.raises(RuntimeError, match="committed qualification inputs"):
        subject.validate_receipt_set(tmp_path, {"full": _receipt(identity)})
