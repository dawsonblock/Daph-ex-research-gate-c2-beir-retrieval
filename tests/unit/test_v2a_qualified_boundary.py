"""Regression tests for the immutable V2A baseline reference."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_v2a_qualified_tag_and_boundary_metadata_are_exact():
    boundary = json.loads((ROOT / "configs/v2a_qualified_boundary.json").read_text())
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", f"{boundary['required_tag']}^{{commit}}"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{tag_commit}^{{tree}}"], cwd=ROOT, text=True).strip()
    assert tag_commit == boundary["qualified_commit"]
    assert tree == boundary["qualified_tree"]
    for receipt in boundary["qualification_receipts"]:
        assert (ROOT / receipt).is_file()
