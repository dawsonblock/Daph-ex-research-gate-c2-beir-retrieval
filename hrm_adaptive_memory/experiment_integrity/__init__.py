"""Shared fail-closed integrity infrastructure for every experimental gate.

Extracted from patterns hardened in the C4 certification pipeline
(scripts/certify_c4_run.py, hrm_adaptive_memory/c4/development_lineage.py,
snapshot_drift.py) so that each new gate (C5, B3, B4, G1, G2, G2-v2, and
whatever comes after) does not have to rediscover and repair the same class of
provenance bug -- a stale resume key, an unvalidated NaN silently passing a
threshold check, a hung subprocess on paid GPU infrastructure, an empty
denominator quietly reported as a passing rate.

Each module here is standalone and independently testable. Nothing in this
package is C4-specific; C4/C5/B3/B4/G1/G2 modules should import from here
rather than re-implement.
"""
from __future__ import annotations
