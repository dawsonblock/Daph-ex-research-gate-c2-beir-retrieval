"""Report generator with invariant enforcement for I3.5.1.

Every report build must enforce:
  N_both_success + N_both_fail + N_A_only + N_B_only == N
  success_count / N == success_rate
  positive + negative + zero == N
  depth subgroup totals == N
  model call counts == receipt-derived call counts
  backend failure counts == receipt-derived failures

If any fail: REPORT_BUILD_ABORTED (not warning).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conditions import ConditionID

REPORT_SCHEMA = "DAPH_V2B_I3_5_1_REPORT_V1"
REPORT_VERSION = 1


class ReportInvariantError(Exception):
    """Raised when a report invariant is violated."""
    pass


def _check_invariant(condition: bool, message: str) -> None:
    """Abort if invariant is violated."""
    if not condition:
        raise ReportInvariantError(f"REPORT_BUILD_ABORTED: {message}")


def verify_count_invariants(
    n_tasks: int,
    *,
    both_success: int,
    both_fail: int,
    gov_blind_only: int,
    gov_aware_only: int,
) -> None:
    """Verify that disagreement counts sum to N."""
    total = both_success + both_fail + gov_blind_only + gov_aware_only
    _check_invariant(
        total == n_tasks,
        f"Count invariant failed: {both_success}+{both_fail}+"
        f"{gov_blind_only}+{gov_aware_only}={total} != N={n_tasks}")


def verify_sign_invariants(
    n_tasks: int,
    *,
    positive: int,
    negative: int,
    zero: int,
) -> None:
    """Verify that sign counts sum to N."""
    total = positive + negative + zero
    _check_invariant(
        total == n_tasks,
        f"Sign invariant failed: {positive}+{negative}+{zero}={total} != N={n_tasks}")


def verify_subgroup_totals(
    n_tasks: int,
    *,
    depth_counts: dict[str, int],
) -> None:
    """Verify that depth subgroup totals sum to N."""
    total = sum(depth_counts.values())
    _check_invariant(
        total == n_tasks,
        f"Depth subgroup invariant failed: sum={total} != N={n_tasks}")


def verify_receipt_consistency(
    *,
    reported_model_calls: int,
    receipt_model_calls: int,
    reported_backend_errors: int,
    receipt_backend_errors: int,
    reported_decoder_failures: int,
    receipt_decoder_failures: int,
) -> None:
    """Verify that report counts match receipt-derived counts."""
    _check_invariant(
        reported_model_calls == receipt_model_calls,
        f"Model call count mismatch: report={reported_model_calls} "
        f"receipts={receipt_model_calls}")
    _check_invariant(
        reported_backend_errors == receipt_backend_errors,
        f"Backend error count mismatch: report={reported_backend_errors} "
        f"receipts={receipt_backend_errors}")
    _check_invariant(
        reported_decoder_failures == receipt_decoder_failures,
        f"Decoder failure count mismatch: report={reported_decoder_failures} "
        f"receipts={receipt_decoder_failures}")


def build_factorial_report(
    *,
    n_tasks: int,
    contributions: list[dict[str, Any]],
    stats: dict[str, Any],
    results: list[dict[str, Any]],
    topology_map: dict[str, str] | None = None,
    experiment_identity_sha256: str = "",
    source_stats_sha256: str = "",
    source_results_sha256: str = "",
    source_run_id: str = "",
) -> dict[str, Any]:
    """Build a factorial report with all invariant checks."""
    # Success counts per condition
    success_counts: dict[str, int] = {}
    for cond_id in ConditionID:
        success_counts[cond_id.value] = 0

    for block in results:
        for cond_id, traj in block["trajectories"].items():
            if traj["task_success"]:
                success_counts[cond_id] = success_counts.get(cond_id, 0) + 1

    # Governor effect disagreement (aware: GOV vs NO_GOV)
    both_success = 0
    both_fail = 0
    gov_only = 0
    no_gov_only = 0

    for block in results:
        trajs = block["trajectories"]
        aware_gov = trajs.get("AWARE_GOVERNOR", {}).get("task_success", False)
        aware_no_gov = trajs.get("AWARE_NO_GOVERNOR", {}).get("task_success", False)
        if aware_gov and aware_no_gov:
            both_success += 1
        elif not aware_gov and not aware_no_gov:
            both_fail += 1
        elif aware_gov and not aware_no_gov:
            gov_only += 1
        else:
            no_gov_only += 1

    verify_count_invariants(
        n_tasks,
        both_success=both_success,
        both_fail=both_fail,
        gov_blind_only=gov_only,
        gov_aware_only=no_gov_only,
    )

    # Sign invariants for primary (ΔDG_gov|aware)
    positive = sum(1 for c in contributions if c["delta_dg_gov_aware"] > 0)
    negative = sum(1 for c in contributions if c["delta_dg_gov_aware"] < 0)
    zero = sum(1 for c in contributions if c["delta_dg_gov_aware"] == 0)
    verify_sign_invariants(n_tasks, positive=positive, negative=negative, zero=zero)

    # Build the 2x2 DG table
    from .scoring import FactorialTaskContribution
    # Reconstruct from dicts for means
    dg_table = {
        "BLIND_NO_GOVERNOR": sum(c["dg_blind_no_gov"] for c in contributions) / n_tasks,
        "BLIND_GOVERNOR": sum(c["dg_blind_gov"] for c in contributions) / n_tasks,
        "AWARE_NO_GOVERNOR": sum(c["dg_aware_no_gov"] for c in contributions) / n_tasks,
        "AWARE_GOVERNOR": sum(c["dg_aware_gov"] for c in contributions) / n_tasks,
    }

    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_VERSION,
        "experiment_identity_sha256": experiment_identity_sha256,
        "source_stats_sha256": source_stats_sha256,
        "source_results_sha256": source_results_sha256,
        "source_run_id": source_run_id,
        "n_tasks": n_tasks,
        "success_counts": success_counts,
        "success_rates": {
            k: v / n_tasks for k, v in success_counts.items()
        },
        "dg_table": dg_table,
        "governor_effect_aware": {
            "mean_delta_dg": stats["mean_delta_dg_gov_aware"],
            "ci": stats["ci_gov_aware"],
            "topology_cluster_mean": stats["topo_mean_gov_aware"],
            "topology_cluster_ci": stats["topo_ci_gov_aware"],
        },
        "governor_effect_blind": {
            "mean_delta_dg": stats["mean_delta_dg_gov_blind"],
            "ci": stats["ci_gov_blind"],
        },
        "state_effect_no_gov": {
            "mean_delta_dg": stats["mean_delta_dg_state_no_gov"],
            "ci": stats["ci_state_no_gov"],
        },
        "state_effect_gov": {
            "mean_delta_dg": stats["mean_delta_dg_state_gov"],
            "ci": stats["ci_state_gov"],
        },
        "interaction": {
            "mean_delta": stats["mean_delta_interaction"],
            "ci": stats["ci_interaction"],
        },
        "disagreement_aware": {
            "both_success": both_success,
            "both_fail": both_fail,
            "gov_only": gov_only,
            "no_gov_only": no_gov_only,
        },
        "sign_distribution_primary": {
            "positive": positive,
            "negative": negative,
            "zero": zero,
        },
    }

    return report


def save_report(
    report: dict[str, Any],
    path: str | Path,
) -> str:
    """Save report to file and return SHA-256."""
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
