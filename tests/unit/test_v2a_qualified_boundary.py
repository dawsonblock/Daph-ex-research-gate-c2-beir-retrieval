"""Regression tests for the immutable V2A baseline reference (boundary V2).

The V2 boundary derives the qualified baseline from the qualified commit/tree
and the committed qualification receipts -- no local git tag, no dist/ archive --
so it reproduces on a clean checkout. See test_v2a_boundary_fail_closed.py for
the exhaustive fail-closed matrix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_v2a_qualified_boundary import check  # noqa: E402


def test_v2a_boundary_metadata_is_v2_and_check_passes():
    boundary = json.loads((ROOT / "configs" / "v2a_qualified_boundary.json").read_text())
    assert boundary["schema"] == "DAPH_V2A_QUALIFIED_BOUNDARY_V2"
    assert boundary["commit"] == "77f348325352c0cd76d08514a60196fba61e4749"
    assert boundary["tree"] == "50e4bf0b2b17b06447c1c1d0dc72207834648d50"
    for name, meta in boundary["required_receipts"].items():
        receipt = ROOT / meta["path"]
        assert receipt.is_file(), f"missing receipt {name}: {meta['path']}"
    # The real boundary must verify cleanly on the current checkout.
    check()
