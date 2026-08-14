"""Fail-closed matrix for the V2A qualified boundary.

Every tampering / drift scenario must raise BoundaryError and never warn-and-
continue. Covers the cases required by I3.3.3:

  * wrong commit (absent from history)
  * wrong tree (commit resolves but to a different tree)
  * missing evidence receipt
  * mutated evidence receipt (on-disk bytes differ from recorded hash)
  * receipt with wrong source identity (binds a different commit/tree)
  * modified component hash (recorded component hash disagrees with the blob)
  * missing qualification component (recorded path absent at the qualified commit)
  * wrong schema / empty receipt set
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_v2a_qualified_boundary import BoundaryError, check  # noqa: E402

REAL_BOUNDARY = json.loads((ROOT / "configs" / "v2a_qualified_boundary.json").read_text())
COMMIT = REAL_BOUNDARY["commit"]
TREE = REAL_BOUNDARY["tree"]
# Use the full_suite receipt as the mutable template (it carries component_hashes).
TEMPLATE_RECEIPT_PATH = REAL_BOUNDARY["required_receipts"]["full_suite"]["path"]
TEMPLATE_RECEIPT = json.loads((ROOT / TEMPLATE_RECEIPT_PATH).read_text())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_receipt(tmp_path: Path, name: str, receipt: dict) -> tuple[str, str]:
    """Write a receipt dict to tmp_path/name and return (abs_path, sha256)."""
    rel = f"receipts/{name}.json"
    out = tmp_path / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True, indent=2).encode()
    out.write_bytes(payload)
    return str(out), _sha256_bytes(payload)


def _base_config(tmp_path: Path) -> dict:
    """A passing boundary config using a real (unmodified) receipt copy."""
    path, digest = _write_receipt(tmp_path, "full_suite", TEMPLATE_RECEIPT)
    cfg = {
        "schema": "DAPH_V2A_QUALIFIED_BOUNDARY_V2",
        "commit": COMMIT,
        "tree": TREE,
        "required_receipts": {"full_suite": {"path": path, "sha256": digest}},
        "qualification_inputs": [],
    }
    return cfg


def _run(tmp_path: Path, cfg: dict) -> None:
    boundary_file = tmp_path / "boundary.json"
    boundary_file.write_text(json.dumps(cfg, indent=2))
    check(boundary_path=boundary_file, root=ROOT)


def test_clean_copy_of_real_receipt_passes(tmp_path):
    # Sanity: an unmodified receipt copy against the real commit/tree verifies.
    _run(tmp_path, _base_config(tmp_path))


def test_wrong_commit_absent_from_history_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["commit"] = "0" * 40  # not present in any clone
    with pytest.raises(BoundaryError, match="absent from the cloned history"):
        _run(tmp_path, cfg)


def test_wrong_tree_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["tree"] = "1" * 40  # commit is real but maps to a different tree
    with pytest.raises(BoundaryError, match="maps to tree"):
        _run(tmp_path, cfg)


def test_missing_evidence_receipt_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["required_receipts"]["full_suite"]["path"] = str(tmp_path / "does_not_exist.json")
    with pytest.raises(BoundaryError, match="receipt is absent"):
        _run(tmp_path, cfg)


def test_mutated_evidence_receipt_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    # Flip a byte in the on-disk receipt without updating the recorded hash.
    path = cfg["required_receipts"]["full_suite"]["path"]
    data = Path(path).read_bytes()
    mutated = bytearray(data)
    mutated[-1] ^= 0x01
    Path(path).write_bytes(bytes(mutated))
    with pytest.raises(BoundaryError, match="SHA-256 mismatch"):
        _run(tmp_path, cfg)


def test_receipt_with_wrong_source_identity_fails_closed(tmp_path):
    receipt = json.loads(json.dumps(TEMPLATE_RECEIPT))
    receipt["source_commit"] = "2" * 40
    path, digest = _write_receipt(tmp_path, "wrong_identity", receipt)
    cfg = {
        "schema": "DAPH_V2A_QUALIFIED_BOUNDARY_V2",
        "commit": COMMIT,
        "tree": TREE,
        "required_receipts": {"full_suite": {"path": path, "sha256": digest}},
        "qualification_inputs": [],
    }
    with pytest.raises(BoundaryError, match="binds source_commit"):
        _run(tmp_path, cfg)


def test_receipt_qi_wrong_source_identity_fails_closed(tmp_path):
    receipt = json.loads(json.dumps(TEMPLATE_RECEIPT))
    # Top-level identity correct, qualification_identity wrong.
    receipt["qualification_identity"]["source_commit"] = "3" * 40
    path, digest = _write_receipt(tmp_path, "wrong_qi", receipt)
    cfg = {
        "schema": "DAPH_V2A_QUALIFIED_BOUNDARY_V2",
        "commit": COMMIT,
        "tree": TREE,
        "required_receipts": {"full_suite": {"path": path, "sha256": digest}},
        "qualification_inputs": [],
    }
    with pytest.raises(BoundaryError, match="qualification_identity.source_commit"):
        _run(tmp_path, cfg)


def test_modified_component_hash_fails_closed(tmp_path):
    receipt = json.loads(json.dumps(TEMPLATE_RECEIPT))
    comp = next(iter(receipt["qualification_identity"]["component_hashes"]))
    receipt["qualification_identity"]["component_hashes"][comp]["sha256"] = "4" * 64
    path, digest = _write_receipt(tmp_path, "bad_component", receipt)
    cfg = {
        "schema": "DAPH_V2A_QUALIFIED_BOUNDARY_V2",
        "commit": COMMIT,
        "tree": TREE,
        "required_receipts": {"full_suite": {"path": path, "sha256": digest}},
        "qualification_inputs": [],
    }
    with pytest.raises(BoundaryError, match="disagrees with qualified commit blob"):
        _run(tmp_path, cfg)


def test_missing_qualification_component_fails_closed(tmp_path):
    receipt = json.loads(json.dumps(TEMPLATE_RECEIPT))
    comp = next(iter(receipt["qualification_identity"]["component_hashes"]))
    receipt["qualification_identity"]["component_hashes"][comp]["path"] = "nonexistent/path.py"
    path, digest = _write_receipt(tmp_path, "missing_component", receipt)
    cfg = {
        "schema": "DAPH_V2A_QUALIFIED_BOUNDARY_V2",
        "commit": COMMIT,
        "tree": TREE,
        "required_receipts": {"full_suite": {"path": path, "sha256": digest}},
        "qualification_inputs": [],
    }
    with pytest.raises(BoundaryError, match="absent at qualified commit"):
        _run(tmp_path, cfg)


def test_wrong_schema_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["schema"] = "DAPH_V2A_QUALIFIED_BOUNDARY_V1"
    with pytest.raises(BoundaryError, match="Unexpected V2A boundary schema"):
        _run(tmp_path, cfg)


def test_empty_receipt_set_fails_closed(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["required_receipts"] = {}
    with pytest.raises(BoundaryError, match="no required receipts"):
        _run(tmp_path, cfg)
