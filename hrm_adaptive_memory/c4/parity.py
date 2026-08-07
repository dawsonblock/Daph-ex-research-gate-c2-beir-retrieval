"""C4 arm-parity validator — mechanically enforce composition correctness.

Required assertions:
  C4-2 vs C4-3: query identical, candidate IDs identical, scores identical;
                only identity state may differ.
  C4-3 vs C4-4: query identical, candidate IDs identical, identity state
                identical; only selected packet may differ.
  C4-4 vs C4-5: same real candidate pool, same query, same identity state;
                only real vs oracle selection differs.

For C4-0→C4-1 and C4-1→C4-2, the expected changes are documented because
those stages intentionally modify query/retrieval behavior.
"""
from __future__ import annotations

from typing import Sequence, Mapping
from dataclasses import asdict

from .contracts import PreHRMResult


class ParityError(Exception):
    """Raised when arm parity is violated."""


def _assert_equal(actual, expected, label: str, context: str = ""):
    if actual != expected:
        raise ParityError(
            f"Parity violation [{label}] {context}: "
            f"expected {expected!r}, got {actual!r}")


def check_parity_pair(
    results_a: list[PreHRMResult],
    results_b: list[PreHRMResult],
    arm_a_id: str,
    arm_b_id: str,
    expected_diff_field: str,
) -> list[str]:
    """Check parity between two arms across all tasks.

    Returns a list of violation messages (empty if all pass).
    """
    violations: list[str] = []
    by_task_a = {r.task_id: r for r in results_a}
    by_task_b = {r.task_id: r for r in results_b}

    common_tasks = set(by_task_a) & set(by_task_b)
    if len(common_tasks) != len(by_task_a):
        violations.append(
            f"{arm_a_id} vs {arm_b_id}: task count mismatch "
            f"({len(by_task_a)} vs {len(by_task_b)}, common={len(common_tasks)})")

    for tid in sorted(common_tasks):
        a = by_task_a[tid]
        b = by_task_b[tid]
        ctx = f"task={tid}"

        if expected_diff_field == "identity_policy":
            # C4-2 vs C4-3: query, candidates, scores identical; identity differs
            _check_query(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_candidates(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_scores(a, b, arm_a_id, arm_b_id, ctx, violations)
            # Identity SHOULD differ (C4-3 has resolution, C4-2 doesn't)
            # Selection may differ if identity changes the fallback

        elif expected_diff_field == "selector_policy":
            # C4-3 vs C4-4: query, candidates, identity identical; selection differs
            _check_query(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_candidates(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_identity(a, b, arm_a_id, arm_b_id, ctx, violations)
            # Selection SHOULD differ (S2c vs S0)

        elif expected_diff_field == "selector_policy_ceiling":
            # C4-4 vs C4-5: same pool, same query, same identity; selection differs
            _check_query(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_candidates(a, b, arm_a_id, arm_b_id, ctx, violations)
            _check_identity(a, b, arm_a_id, arm_b_id, ctx, violations)
            # Selection SHOULD differ (oracle vs real)

    return violations


def _check_query(a: PreHRMResult, b: PreHRMResult,
                 arm_a: str, arm_b: str, ctx: str, violations: list[str]):
    if a.query.rendered_query != b.query.rendered_query:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: query differs "
            f"{a.query.rendered_query!r} vs {b.query.rendered_query!r}")
    if a.query.query_hash != b.query.query_hash:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: query_hash differs")


def _check_candidates(a: PreHRMResult, b: PreHRMResult,
                      arm_a: str, arm_b: str, ctx: str, violations: list[str]):
    if a.retrieval.candidate_ids != b.retrieval.candidate_ids:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: candidate_ids differ")


def _check_scores(a: PreHRMResult, b: PreHRMResult,
                  arm_a: str, arm_b: str, ctx: str, violations: list[str]):
    if a.retrieval.bm25_ranked != b.retrieval.bm25_ranked:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: bm25_ranked differs")
    if a.retrieval.bge_ranked != b.retrieval.bge_ranked:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: bge_ranked differs")
    if a.retrieval.fusion_ranked != b.retrieval.fusion_ranked:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: fusion_ranked differs")


def _check_identity(a: PreHRMResult, b: PreHRMResult,
                    arm_a: str, arm_b: str, ctx: str, violations: list[str]):
    if a.identity.status != b.identity.status:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: identity status differs "
            f"{a.identity.status} vs {b.identity.status}")
    if a.identity.canonical != b.identity.canonical:
        violations.append(
            f"{arm_a} vs {arm_b} [{ctx}]: identity canonical differs")


def validate_all_parity(
    all_results: Mapping[str, list[PreHRMResult]],
) -> tuple[bool, list[str]]:
    """Validate all parity pairs. Returns (all_pass, violations)."""
    all_violations: list[str] = []

    # C4-2 vs C4-3: only identity differs
    if "C4_2" in all_results and "C4_3" in all_results:
        v = check_parity_pair(
            all_results["C4_2"], all_results["C4_3"],
            "C4_2", "C4_3", "identity_policy")
        all_violations.extend(v)

    # C4-3 vs C4-4: only selector differs
    if "C4_3" in all_results and "C4_4" in all_results:
        v = check_parity_pair(
            all_results["C4_3"], all_results["C4_4"],
            "C4_3", "C4_4", "selector_policy")
        all_violations.extend(v)

    # C4-4 vs C4-5: only real vs oracle selection
    if "C4_4" in all_results and "C4_5" in all_results:
        v = check_parity_pair(
            all_results["C4_4"], all_results["C4_5"],
            "C4_4", "C4_5", "selector_policy_ceiling")
        all_violations.extend(v)

    return (len(all_violations) == 0, all_violations)


def validate_no_leakage(all_results: Mapping[str, list[PreHRMResult]]) -> tuple[bool, list[str]]:
    """Validate that no runtime payload contains oracle keys."""
    from .receipts import assert_runtime_clean
    violations: list[str] = []
    for arm_id, results in all_results.items():
        for r in results:
            try:
                # Check the pre-HRM result for oracle keys
                payload = {
                    "query": {"rendered_query": r.query.rendered_query},
                    "candidate_ids": list(r.retrieval.candidate_ids),
                    "identity": {
                        "status": r.identity.status,
                        "canonical": r.identity.canonical,
                    },
                    "selected_ids": list(r.selection.selected_ids),
                }
                assert_runtime_clean(payload)
            except AssertionError as e:
                violations.append(f"{arm_id} {r.task_id}: {e}")
    return (len(violations) == 0, violations)


def validate_selected_in_pool(all_results: Mapping[str, list[PreHRMResult]]) -> tuple[bool, list[str]]:
    """Validate that selected IDs exist in the candidate pool for C4-0..C4-5."""
    violations: list[str] = []
    for arm_id, results in all_results.items():
        if arm_id == "C4_6":
            continue  # C4-6 is allowed to inject required evidence directly
        for r in results:
            pool = set(r.retrieval.candidate_ids)
            for sid in r.selection.selected_ids:
                if sid not in pool:
                    violations.append(
                        f"{arm_id} {r.task_id}: selected ID {sid!r} not in candidate pool")
    return (len(violations) == 0, violations)


def validate_packet_budgets(all_results: Mapping[str, list[PreHRMResult]]) -> tuple[bool, list[str]]:
    """Validate that packet sizes are within budget."""
    from .contracts import C4_PRIMARY_PACKET_BUDGET
    violations: list[str] = []
    for arm_id, results in all_results.items():
        for r in results:
            if len(r.packet.packet_ids) > r.packet.packet_budget:
                violations.append(
                    f"{arm_id} {r.task_id}: packet has {len(r.packet.packet_ids)} "
                    f"items, budget is {r.packet.packet_budget}")
    return (len(violations) == 0, violations)
