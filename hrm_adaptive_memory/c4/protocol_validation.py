"""Fail-closed semantic validation of the frozen C4 v2 protocol.

Computing the protocol SHA256 proves only that the *file* is the expected one.
It proves nothing about whether the code that is about to run implements what
the file declares. A validator that reads optimistically with ``.get(k, "N/A")``
and keeps going is worse than no validator: it prints reassurance for fields it
never found.

Every check here is an exact semantic invariant across two sources of truth:

    configs/gate_c4_protocol_v2.json   what the protocol declares
    hrm_adaptive_memory.c4.*          what the code will actually execute

``validate_c4_protocol`` raises :class:`ProtocolViolation` on the first
mismatch. There is no permissive mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

# v2 specified packet ordering twice, inconsistently (selector-score-first in
# determinism_requirements.tie_break_policy vs role-tier-first in
# packet_ordering_policy.within_role_tier). Reproducibility cannot be certified
# against an internally inconsistent contract, so v2 is no longer a certifiable
# active protocol; it is retained for lineage only.
EXPECTED_PROTOCOL_ID = "c4_v2_1_reproducible_and_qualified"
EXPECTED_PROTOCOL_VERSION = "v2_1_frozen"
SUPERSEDED_PROTOCOL_FILE = "gate_c4_protocol_v2.json"
EXPECTED_METRIC_MODULE = "hrm_adaptive_memory.c4.metrics"
PRIMARY_ARMS = ("C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6")
DIAGNOSTIC_ARMS = ("C4_3o", "C4_4m")

# The one canonical packet order. Each entry must appear, in this position, in
# packet_ordering_policy.canonical_sort_key.
CANONICAL_SORT_KEY_TERMS = (
    ("role", "role priority"),
    ("retrieval", "retrieval fusion score"),
    ("record_id", "record_id lexical"),
)

# Metrics certification must recompute from raw receipts rather than trust.
REQUIRED_RECOMPUTED_METRICS = frozenset({
    "arm quality", "binary accuracy", "complete-set retention (CSR)",
    "primary delta", "selector_gap_capture", "oracle_gap_capture",
    "family-grouped CI", "cluster-grouped CI", "arm receipt counts",
    "task-set equality",
})

# Arm policy fields the protocol pins. Checked against the executable registry
# in hrm_adaptive_memory.c4.arms, not against prose in the protocol file.
REQUIRED_ARM_POLICIES: dict[str, dict[str, str]] = {
    "C4_0": {"query_policy": "original", "retrieval_policy": "bm25_only",
             "identity_policy": "none", "selector_policy": "s0"},
    "C4_1": {"query_policy": "subject_preserving", "retrieval_policy": "bm25_only",
             "identity_policy": "none", "selector_policy": "s0"},
    "C4_2": {"query_policy": "subject_preserving",
             "retrieval_policy": "bm25_bge_fusion",
             "identity_policy": "none", "selector_policy": "s0"},
    "C4_3": {"query_policy": "subject_preserving",
             "retrieval_policy": "bm25_bge_fusion",
             "identity_policy": "i3_explicit_identity", "selector_policy": "s0"},
    "C4_4": {"query_policy": "subject_preserving",
             "retrieval_policy": "bm25_bge_fusion",
             "identity_policy": "i3_explicit_identity",
             "selector_policy": "s2c_with_s0_fallback"},
    "C4_5": {"query_policy": "subject_preserving",
             "retrieval_policy": "bm25_bge_fusion",
             "identity_policy": "i3_explicit_identity",
             "selector_policy": "oracle"},
    "C4_6": {"query_policy": "subject_preserving",
             "retrieval_policy": "bm25_bge_fusion",
             "identity_policy": "i3_explicit_identity",
             "selector_policy": "oracle_evidence"},
    # Diagnostic 2x2 completion: membership and ordering varied independently.
    "C4_3o": {"identity_policy": "i3_explicit_identity", "selector_policy": "s0"},
    "C4_4m": {"identity_policy": "i3_explicit_identity",
              "selector_policy": "s2c_with_s0_fallback"},
}

# Arms that must apply the deterministic packet-ordering policy. The other
# arms must keep retrieval pool order, or the 2x2 decomposition is degenerate.
REQUIRED_DETERMINISTIC_ORDER_ARMS = frozenset({"C4_4", "C4_3o"})


class ProtocolViolation(AssertionError):
    """Raised when the protocol and the executable code disagree."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolation(message)


def _get(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Fetch a required key. Missing keys abort — they never become 'N/A'."""
    if key not in mapping:
        raise ProtocolViolation(
            f"Required protocol field {context}.{key} is missing. "
            f"Present keys: {sorted(mapping)}")
    return mapping[key]


def _normalize_role(role: str) -> str:
    """'rejected-0' -> 'rejected' so protocol role names match code keys."""
    return re.sub(r"-\d+$", "", role)


def validate_c4_protocol(protocol: Mapping[str, Any]) -> dict:
    """Validate the frozen protocol against the code that will execute it.

    Returns a dict of the checks that passed, for inclusion in receipts.
    Raises :class:`ProtocolViolation` on the first mismatch.
    """
    from .arms import ARMS, PARITY_PAIRS, PRIMARY_ORDER
    from .contracts import C4_PRIMARY_PACKET_BUDGET
    from .packet_ordering import ORDERING_POLICY_ID
    from .packet_stage import _DETERMINISTIC_ORDER_ARMS
    from ..retrieval_bench.selectors.chain import ROLE_PRIORITY

    checks: dict[str, Any] = {}

    # --- 1. Protocol identity ------------------------------------------------
    _require(_get(protocol, "protocol_id", "protocol") == EXPECTED_PROTOCOL_ID,
             f"protocol_id must be {EXPECTED_PROTOCOL_ID!r}, got "
             f"{protocol.get('protocol_id')!r}")
    _require(_get(protocol, "protocol_version", "protocol") == EXPECTED_PROTOCOL_VERSION,
             f"protocol_version must be {EXPECTED_PROTOCOL_VERSION!r}, got "
             f"{protocol.get('protocol_version')!r}")
    _require(_get(protocol, "frozen_before_any_c4_v2_measurement", "protocol") is True,
             "protocol must declare frozen_before_any_c4_v2_measurement = true")
    checks["protocol_identity"] = True

    # --- 2. One-pass primary pipeline ---------------------------------------
    arch = _get(protocol, "architecture", "protocol")
    one_pass = str(_get(arch, "one_pass_only", "architecture")).lower()
    _require("one-pass" in one_pass,
             f"architecture.one_pass_only must declare a one-pass pipeline, got {one_pass!r}")
    iterative = _get(protocol, "iterative_retrieval_status", "protocol")
    _require(_get(iterative, "classification", "iterative_retrieval_status")
             == "OUTSIDE_PRIMARY_C4",
             "iterative retrieval must be classified OUTSIDE_PRIMARY_C4")
    checks["one_pass_primary"] = True

    # --- 3. Arms present in both protocol and code ---------------------------
    protocol_arms = _get(protocol, "arms", "protocol")
    missing_protocol = [a for a in PRIMARY_ARMS if a not in protocol_arms]
    _require(not missing_protocol,
             f"protocol.arms is missing primary arms: {missing_protocol}")
    missing_code = [a for a in REQUIRED_ARM_POLICIES if a not in ARMS]
    _require(not missing_code,
             f"code arm registry is missing arms: {missing_code}")
    checks["arms_present"] = sorted(REQUIRED_ARM_POLICIES)

    # --- 4. Per-arm executable policy fields --------------------------------
    for arm_id, required in REQUIRED_ARM_POLICIES.items():
        arm = ARMS[arm_id]
        for field, expected in required.items():
            actual = getattr(arm, field)
            _require(actual == expected,
                     f"{arm_id}.{field} must be {expected!r}, got {actual!r}")
        _require(arm.packet_budget == C4_PRIMARY_PACKET_BUDGET,
                 f"{arm_id}.packet_budget must be the frozen "
                 f"{C4_PRIMARY_PACKET_BUDGET}, got {arm.packet_budget}")
    checks["arm_policies"] = True

    # --- 5. Membership vs ordering separated, and the 2x2 is non-degenerate --
    _require(set(_DETERMINISTIC_ORDER_ARMS) == set(REQUIRED_DETERMINISTIC_ORDER_ARMS),
             f"deterministic-order arms must be "
             f"{sorted(REQUIRED_DETERMINISTIC_ORDER_ARMS)}, got "
             f"{sorted(_DETERMINISTIC_ORDER_ARMS)}")
    # C4_3 vs C4_3o: same membership policy, different ordering.
    _require(ARMS["C4_3"].selector_policy == ARMS["C4_3o"].selector_policy,
             "C4_3o must share C4_3's membership policy (ordering-only contrast)")
    _require(("C4_3" in _DETERMINISTIC_ORDER_ARMS) !=
             ("C4_3o" in _DETERMINISTIC_ORDER_ARMS),
             "C4_3 and C4_3o must differ in ordering, else the ordering effect "
             "is zero by construction")
    # C4_4 vs C4_4m: same membership policy, different ordering.
    _require(ARMS["C4_4"].selector_policy == ARMS["C4_4m"].selector_policy,
             "C4_4m must share C4_4's membership policy (ordering-only contrast)")
    _require(("C4_4" in _DETERMINISTIC_ORDER_ARMS) !=
             ("C4_4m" in _DETERMINISTIC_ORDER_ARMS),
             "C4_4 and C4_4m must differ in ordering, else the membership "
             "effect absorbs the ordering effect")
    mvo = _get(protocol, "membership_vs_ordering", "protocol")
    _require("membership_hash (order-independent)" in _get(mvo, "hashes", "membership_vs_ordering"),
             "protocol must declare an order-independent membership_hash")
    _require("order_hash (order-sensitive)" in mvo["hashes"],
             "protocol must declare an order-sensitive order_hash")
    checks["membership_ordering_separated"] = True

    # --- 6. Ordering policy: exactly one definition, matching the code ------
    ordering = _get(protocol, "packet_ordering_policy", "protocol")
    _require(_get(ordering, "policy_id", "packet_ordering_policy") == ORDERING_POLICY_ID,
             f"packet_ordering_policy.policy_id must equal code "
             f"ORDERING_POLICY_ID={ORDERING_POLICY_ID!r}, got "
             f"{ordering.get('policy_id')!r}")
    _validate_single_ordering_definition(protocol, ordering)
    policy_versions = _get(protocol, "policy_versions", "protocol")
    _require(_get(policy_versions, "ordering_policy_version", "policy_versions")
             == ORDERING_POLICY_ID,
             "policy_versions.ordering_policy_version must name the frozen "
             f"ordering policy {ORDERING_POLICY_ID!r}")
    # The declared role tier order must be a monotone non-decreasing walk
    # through the code's priority table.
    chain_order = _get(ordering, "chain_order", "packet_ordering_policy")
    normalized = {_normalize_role(k): v for k, v in ROLE_PRIORITY.items()}
    priorities = []
    for role in chain_order:
        _require(role in normalized,
                 f"protocol chain_order role {role!r} has no priority in "
                 f"ROLE_PRIORITY; ordering would silently fall back to default")
        priorities.append(normalized[role])
    _require(priorities == sorted(priorities),
             f"protocol chain_order disagrees with code ROLE_PRIORITY: "
             f"declared order maps to priorities {priorities}")
    checks["ordering_policy"] = ORDERING_POLICY_ID

    # --- 7. Selector / identity policy versions -----------------------------
    _require(_get(policy_versions, "identity_policy_version", "policy_versions")
             == "i3_v1",
             "identity_policy_version must be i3_v1")
    _require(_get(policy_versions, "selector_policy_version", "policy_versions")
             == "s2c_deterministic_v1",
             "selector_policy_version must be s2c_deterministic_v1")
    checks["policy_versions"] = dict(policy_versions)

    # --- 8. Authoritative metric module and quality states ------------------
    metric_defs = _get(protocol, "metric_definitions", "protocol")
    for name in ("quality", "oracle_gap_capture", "selector_gap_capture"):
        section = _get(metric_defs, name, "metric_definitions")
        _require(_get(section, "authoritative_module", f"metric_definitions.{name}")
                 == EXPECTED_METRIC_MODULE,
                 f"metric_definitions.{name}.authoritative_module must be "
                 f"{EXPECTED_METRIC_MODULE!r}")
    _require(_get(metric_defs["oracle_gap_capture"], "numerator_uses",
                  "metric_definitions.oracle_gap_capture").startswith("C4_4"),
             "OGC numerator must use C4_4, not C4_5")
    _validate_metric_implementation(metric_defs)
    checks["metric_module"] = EXPECTED_METRIC_MODULE

    # --- 9. Parity pairs isolate one mechanism each -------------------------
    for arm_a, arm_b, expected_field in PARITY_PAIRS:
        a, b = ARMS[arm_a], ARMS[arm_b]
        differing = [f for f in ("query_policy", "retrieval_policy",
                                 "identity_policy", "selector_policy")
                     if getattr(a, f) != getattr(b, f)]
        _require(differing == [expected_field],
                 f"parity pair {arm_a}->{arm_b} must differ only in "
                 f"{expected_field}, differs in {differing}")
    checks["parity_pairs"] = True

    # --- 10. Fail-closed runner abort conditions --------------------------
    runner = _get(protocol, "fail_closed_runner", "protocol")
    aborts = _get(runner, "abort_conditions", "fail_closed_runner")
    for required_abort in ("test suite fails", "protocol hash mismatch",
                           "dependency version mismatch"):
        _require(required_abort in aborts,
                 f"fail_closed_runner.abort_conditions must include "
                 f"{required_abort!r}")
    checks["abort_conditions"] = list(aborts)

    # --- 11. Lineage: v2_1 must declare what it supersedes and why ---------
    lineage = _get(protocol, "lineage", "protocol")
    _require(_get(lineage, "supersedes", "lineage") == SUPERSEDED_PROTOCOL_FILE,
             f"lineage.supersedes must be {SUPERSEDED_PROTOCOL_FILE!r}")
    _require(_get(lineage, "mechanism_change", "lineage") is False,
             "lineage.mechanism_change must be false: v2_1 is a conformance "
             "repair, not a mechanism change. A mechanism change requires a "
             "new arm ladder, not a protocol clarification.")
    _require(_get(lineage, "scientific_change", "lineage")
             == "clarification/conformance repair",
             "lineage.scientific_change must be "
             "'clarification/conformance repair'")
    for field in ("reason", "conformance_defect_repaired",
                  "affected_prior_results"):
        _require(bool(_get(lineage, field, "lineage")),
                 f"lineage.{field} must be non-empty")
    checks["lineage"] = {
        "supersedes": lineage["supersedes"],
        "mechanism_change": lineage["mechanism_change"],
    }

    # --- 12. Prompt binding is a protocol requirement ---------------------
    freeze = _get(protocol, "pre_hrm_freeze", "protocol")
    binding = _get(freeze, "prompt_binding_rule", "pre_hrm_freeze")
    _require("prompt_hash" in binding,
             "pre_hrm_freeze.prompt_binding_rule must reference prompt_hash")
    _require("ORDERED" in binding or "ordered" in binding,
             "pre_hrm_freeze.prompt_binding_rule must require the prompt to be "
             "composed over the ORDERED packet")
    # The code must be able to produce that binding.
    from .contracts import PacketResult
    for field in ("prompt_hash", "order_hash", "membership_hash",
                  "candidate_pool_hash"):
        _require(field in PacketResult.__dataclass_fields__,
                 f"PacketResult must carry {field} to satisfy the prompt "
                 f"binding and packet-hashing rules")
    _require("prompt binding violated (packet.prompt_hash != generated prompt hash)"
             in aborts,
             "fail_closed_runner.abort_conditions must include the prompt "
             "binding violation")
    checks["prompt_binding_required"] = True

    # --- 13. Diagnostic arms are isolated from the primary ladder ---------
    diagnostic = _get(protocol, "diagnostic_arms", "protocol")
    declared = tuple(_get(diagnostic, "arms", "diagnostic_arms"))
    _require(declared == DIAGNOSTIC_ARMS,
             f"diagnostic_arms.arms must be {list(DIAGNOSTIC_ARMS)}, got "
             f"{list(declared)}")
    isolation = _get(diagnostic, "isolation_rule", "diagnostic_arms")
    for phrase in ("primary", "promotion threshold"):
        _require(phrase in isolation,
                 f"diagnostic_arms.isolation_rule must state the {phrase} "
                 f"exclusion")
    for arm_id in DIAGNOSTIC_ARMS:
        _require(ARMS[arm_id].classification == "DIAGNOSTIC",
                 f"{arm_id} must be classified DIAGNOSTIC in the arm registry, "
                 f"got {ARMS[arm_id].classification!r}")
        _require(arm_id not in PRIMARY_ORDER,
                 f"{arm_id} must not appear in PRIMARY_ORDER")
    _require(tuple(PRIMARY_ORDER) == PRIMARY_ARMS,
             f"PRIMARY_ORDER must be exactly {list(PRIMARY_ARMS)}, got "
             f"{list(PRIMARY_ORDER)}")
    checks["diagnostic_arms_isolated"] = list(DIAGNOSTIC_ARMS)

    # --- 14. Certification must recompute, not trust ----------------------
    certification = _get(protocol, "certification", "protocol")
    recomputed = set(_get(certification, "recompute_from_raw_receipts",
                          "certification"))
    missing_metrics = REQUIRED_RECOMPUTED_METRICS - recomputed
    _require(not missing_metrics,
             f"certification.recompute_from_raw_receipts must include "
             f"{sorted(missing_metrics)}")
    _require("conjunction" in _get(certification, "valid_run_rule", "certification"),
             "certification.valid_run_rule must define VALID_RUN as a "
             "conjunction of derived gates")
    checks["recomputed_metrics"] = sorted(recomputed)

    return checks


def _validate_single_ordering_definition(protocol: Mapping[str, Any],
                                        ordering: Mapping[str, Any]) -> None:
    """Exactly one packet-order definition, and the code must implement it.

    v2 said both "selector score before role priority" (in
    ``determinism_requirements.tie_break_policy``) and "role tier before score"
    (in ``packet_ordering_policy.within_role_tier``). Those cannot both be the
    packet order. v2_1 keeps one definition and scopes the selector's own
    chain-level order separately.
    """
    import inspect

    from .packet_ordering import order_packet
    from .packet_stage import _DETERMINISTIC_ORDER_ARMS

    # 1. The canonical key exists and is exactly the three declared terms.
    key = _get(ordering, "canonical_sort_key", "packet_ordering_policy")
    _require(len(key) == len(CANONICAL_SORT_KEY_TERMS),
             f"canonical_sort_key must have exactly "
             f"{len(CANONICAL_SORT_KEY_TERMS)} rules, got {len(key)}: {key}")
    for position, ((token, label), rule) in enumerate(
            zip(CANONICAL_SORT_KEY_TERMS, key), start=1):
        _require(token in rule.lower(),
                 f"canonical_sort_key rule {position} must be {label}, got "
                 f"{rule!r}")

    # 2. No competing definition may survive anywhere in the protocol.
    det = _get(protocol, "determinism_requirements", "protocol")
    _require("tie_break_policy" not in det,
             "determinism_requirements.tie_break_policy still exists and "
             "contradicts packet_ordering_policy.canonical_sort_key. Exactly "
             "one packet-order definition is permitted.")
    _require("within_role_tier" not in ordering,
             "packet_ordering_policy.within_role_tier duplicates the canonical "
             "sort key; remove it so there is one definition.")
    selector_order = _get(det, "selector_tie_break_policy",
                          "determinism_requirements")
    scope = _get(selector_order, "scope", "selector_tie_break_policy")
    _require("not the packet order" in scope.lower(),
             "selector_tie_break_policy.scope must state that it is not the "
             "packet order")
    _require(bool(_get(det, "packet_ordering_authority",
                       "determinism_requirements")),
             "determinism_requirements.packet_ordering_authority must name the "
             "sole packet-order definition")

    # 3. Selector score is excluded, and the code cannot accept one.
    excluded = _get(ordering, "selector_score_excluded", "packet_ordering_policy")
    _require(_get(excluded, "excluded", "selector_score_excluded") is True,
             "packet_ordering_policy.selector_score_excluded.excluded must be "
             "true: S2c scores chains, not records")
    _require(bool(_get(excluded, "reason", "selector_score_excluded")),
             "selector_score_excluded.reason must be non-empty")
    params = inspect.signature(order_packet).parameters
    _require("selector_scores" not in params,
             "order_packet still accepts selector_scores, but the protocol "
             "excludes selector score from packet ordering. Remove the "
             "parameter rather than passing an empty mapping.")
    _require("retrieval_scores" in params,
             "order_packet must accept retrieval_scores to implement "
             "canonical_sort_key rule 2")

    # 4. The arms the policy applies to must match the code exactly.
    applies_to = set(_get(ordering, "applies_to_arms", "packet_ordering_policy"))
    _require(applies_to == set(_DETERMINISTIC_ORDER_ARMS),
             f"packet_ordering_policy.applies_to_arms {sorted(applies_to)} does "
             f"not match the code's deterministic-order arms "
             f"{sorted(_DETERMINISTIC_ORDER_ARMS)}")


def _validate_metric_implementation(metric_defs: Mapping[str, Any]) -> None:
    """Verify the code's metric functions match the protocol's declarations.

    Checks the quality truth table entry by entry, and probes gap capture so a
    silently swapped numerator (the C4_5-for-C4_4 error) cannot pass.
    """
    from .metrics import compute_quality, oracle_gap_capture, selector_gap_capture

    states = _get(metric_defs["quality"], "states", "metric_definitions.quality")
    expected = {
        "complete_and_correct": compute_quality(correct=True, evidence_complete=True),
        "complete_and_incorrect": compute_quality(correct=False, evidence_complete=True),
        "incomplete_and_correct": compute_quality(correct=True, evidence_complete=False),
        "incomplete_and_incorrect": compute_quality(correct=False, evidence_complete=False),
    }
    for state, declared in states.items():
        _require(state in expected,
                 f"protocol declares unknown quality state {state!r}")
        _require(abs(float(declared) - expected[state]) < 1e-12,
                 f"quality state {state}: protocol declares {declared}, "
                 f"compute_quality returns {expected[state]}")

    # OGC must respond to C4_4 and ignore C4_5.
    base = {"C4_0": 0.0, "C4_3": 0.0, "C4_4": 0.5, "C4_5": 0.9, "C4_6": 1.0}
    ogc = oracle_gap_capture(base)
    _require(ogc is not None and abs(ogc - 0.5) < 1e-12,
             f"OGC must be (C4_4-C4_0)/(C4_6-C4_0)=0.5 on the probe, got {ogc}")
    moved_c5 = dict(base, C4_5=0.1)
    _require(oracle_gap_capture(moved_c5) == ogc,
             "OGC must not depend on C4_5")
    moved_c4 = dict(base, C4_4=0.25)
    _require(oracle_gap_capture(moved_c4) != ogc,
             "OGC must depend on C4_4")

    # SGC must be the selector gap against the C4_5 ceiling.
    sgc = selector_gap_capture(base)
    _require(sgc is not None and abs(sgc - (0.5 - 0.0) / (0.9 - 0.0)) < 1e-12,
             f"SGC must be (C4_4-C4_3)/(C4_5-C4_3), got {sgc}")


def load_and_validate_protocol(path: str | Path) -> tuple[dict, str, dict]:
    """Load, hash and validate a protocol file.

    Returns ``(protocol, sha256, checks)``. Raises :class:`ProtocolViolation`
    if the file is absent, unparseable, or semantically inconsistent with the
    code. A missing file is a violation, never a "NOT_FOUND" string that flows
    into a certification artifact.
    """
    import hashlib

    p = Path(path)
    if not p.is_file():
        raise ProtocolViolation(
            f"Protocol file not found: {p}. Certification cannot proceed "
            f"without the active protocol.")
    raw = p.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolViolation(f"Protocol file {p} is not valid JSON: {exc}") from exc

    checks = validate_c4_protocol(protocol)
    return protocol, sha256, checks
