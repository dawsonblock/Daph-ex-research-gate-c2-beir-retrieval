"""Shared test hygiene for the unit suite.

WHY THIS EXISTS
---------------
``pin_certified_memory_v1_boundary_policy()`` mutates a PROCESS-GLOBAL
default in hrm_adaptive_memory.c4.bridge_extraction
(``_DEFAULT_BOUNDARY_POLICY``, "legacy" -> "grammar_v4"). Five test modules
call it, none of them restored it, and nothing failed only because the
modules that pin it happened to sort AFTER
tests/unit/test_g2_v4e_entity_boundary.py, which asserts the default is
"legacy".

That is an ordering-dependent landmine, not a stable pass: adding any test
module whose name sorts earlier detonates it. Adding
test_background_verification.py did exactly that and broke seven unrelated
tests in test_g2_v4e_entity_boundary.py -- tests which still passed in
isolation, which is the signature of leaked global state rather than a real
regression.

The fixture below snapshots and restores that global around EVERY test, so
the suite stops depending on filename collation for correctness.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_boundary_policy_global():
    """Snapshot/restore the process-global entity-boundary policy.

    Autouse and unconditional: a test that pins the policy is doing something
    legitimate, so the fix is to contain the blast radius rather than to
    forbid pinning.
    """
    try:
        from hrm_adaptive_memory.c4 import bridge_extraction
    except Exception:  # pragma: no cover - suite runnable without the module
        yield
        return
    before = bridge_extraction.get_default_boundary_policy()
    try:
        yield
    finally:
        if bridge_extraction.get_default_boundary_policy() != before:
            bridge_extraction.set_default_boundary_policy(before)
