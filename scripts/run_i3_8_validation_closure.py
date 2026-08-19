#!/usr/bin/env python3
"""I3.8 validation closure: statistics, subgroup analysis, identity freeze.

Computes:
  1. Overall paired bootstrap CI + McNemar
  2. Per-category subgroup analysis
  3. Frozen identity with widened hash boundary
  4. Validation closure summary

Usage:
    PYTHONPATH=. python scripts/run_i3_8_validation_closure.py
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    load_evidence_benchmark, initial_evidence_runtime, EvidenceExecutor,
    build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceState

VALIDATION_MANIFEST = (
    ROOT / "experiments/v2b_i3_8/manifests/i3_8_evidence_validation_v1.json"
)
VALIDATION_RESULTS = (
    ROOT / "experiments/v2b_i3_8/development/i3_8_validation/repair_check_v1.jsonl"
)
OUTPUT_DIR = ROOT / "experiments/v2b_i3_8/development/i3_8_validation"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def paired_bootstrap_ci(
    deltas: list[float],
    n_iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_iterations)]
    hi = boot_means[int(0.975 * n_iterations)]
    return lo, hi


def mcnemar_test(a_success: list[bool], m_success: list[bool]) -> dict:
    b = sum(1 for a, m in zip(a_success, m_success) if a and not m)
    c = sum(1 for a, m in zip(a_success, m_success) if not a and m)
    n_discordant = b + c

    from math import comb
    def binomial_pmf(k, n, p):
        if n == 0:
            return 1.0
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    if n_discordant == 0:
        p_value = 1.0
    else:
        larger = max(b, c)
        tail = sum(binomial_pmf(k, n_discordant, 0.5)
                   for k in range(larger, n_discordant + 1))
        p_value = 2 * tail
        if p_value > 1.0:
            p_value = 1.0

    return {
        "b_a_only_success": b,
        "c_m_only_success": c,
        "n_discordant": n_discordant,
        "exact_p_value": round(p_value, 8),
        "all_favor_m": b == 0 and c > 0,
    }


def compute_overall_stats(results: list[dict]) -> dict:
    n = len(results)
    deltas_u = [r["m_gain"] for r in results]
    a_success = [r["a_success"] for r in results]
    m_success = [r["m_success"] for r in results]

    mean_delta_u = sum(deltas_u) / n
    variance = sum((d - mean_delta_u) ** 2 for d in deltas_u) / (n - 1)
    std_err = math.sqrt(variance) / math.sqrt(n)

    ci_lo, ci_hi = paired_bootstrap_ci(deltas_u, n_iterations=10000, seed=42)

    z = 1.96
    normal_lo = mean_delta_u - z * std_err
    normal_hi = mean_delta_u + z * std_err

    mcnemar = mcnemar_test(a_success, m_success)

    a_s = sum(a_success)
    m_s = sum(m_success)

    a_redundant = sum(r["fork_a"]["redundant_action_count"] for r in results)
    m_redundant = sum(r["fork_m"]["redundant_action_count"] for r in results)
    a_steps = sum(r["fork_a"]["steps"] for r in results)
    m_steps = sum(r["fork_m"]["steps"] for r in results)

    return {
        "estimand_u": "U_M - U_A",
        "pairing_unit": "task_id",
        "n_pairs": n,
        "mean_delta_u": round(mean_delta_u, 4),
        "std_err_delta_u": round(std_err, 4),
        "bootstrap_iterations": 10000,
        "bootstrap_seed": 42,
        "ci_method": "paired_task_bootstrap",
        "ci_method_description": (
            "Two-sided 95% percentile bootstrap CI. The lower endpoint "
            "is the 2.5th percentile of the bootstrap distribution."
        ),
        "ci95_bootstrap": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci95_normal": [round(normal_lo, 4), round(normal_hi, 4)],
        "lower_endpoint_two_sided_95ci_bootstrap": round(ci_lo, 4),
        "lower_endpoint_two_sided_95ci_positive": ci_lo > 0,
        "success": {
            "a_success": a_s,
            "m_success": m_s,
            "delta_success": m_s - a_s,
            "delta_success_percentage_points": round((m_s - a_s) / n * 100, 2),
            "a_rate": round(a_s / n, 4),
            "m_rate": round(m_s / n, 4),
        },
        "mcnemar": mcnemar,
        "redundant_actions": {
            "a_total": a_redundant,
            "m_total": m_redundant,
            "a_rate": round(a_redundant / max(a_steps, 1), 4),
            "m_rate": round(m_redundant / max(m_steps, 1), 4),
        },
        "steps": {
            "a_total": a_steps,
            "m_total": m_steps,
            "a_mean": round(a_steps / n, 4),
            "m_mean": round(m_steps / n, 4),
            "delta_mean": round(m_steps / n - a_steps / n, 4),
        },
    }


def compute_subgroup_analysis(results: list[dict]) -> dict:
    """Per-category breakdown."""
    categories = sorted(set(r["category"] for r in results))
    subgroups = {}

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        a_s = sum(1 for r in cat_results if r["a_success"])
        m_s = sum(1 for r in cat_results if r["m_success"])
        rescues = sum(1 for r in cat_results if not r["a_success"] and r["m_success"])
        breaks = sum(1 for r in cat_results if r["a_success"] and not r["m_success"])
        u_a = sum(r["u_a"] for r in cat_results) / n
        u_m = sum(r["u_m"] for r in cat_results) / n
        a_steps = sum(r["fork_a"]["steps"] for r in cat_results)
        m_steps = sum(r["fork_m"]["steps"] for r in cat_results)
        a_redundant = sum(r["fork_a"]["redundant_action_count"] for r in cat_results)
        m_redundant = sum(r["fork_m"]["redundant_action_count"] for r in cat_results)

        # Subgroup bootstrap CI
        deltas = [r["m_gain"] for r in cat_results]
        if n >= 10:
            ci_lo, ci_hi = paired_bootstrap_ci(deltas, n_iterations=10000, seed=42)
        else:
            ci_lo, ci_hi = float('-inf'), float('inf')

        subgroups[cat] = {
            "n_tasks": n,
            "success": {
                "a": f"{a_s}/{n} ({a_s/n*100:.1f}%)",
                "m": f"{m_s}/{n} ({m_s/n*100:.1f}%)",
                "delta_pp": round((m_s - a_s) / n * 100, 2),
            },
            "rescues": rescues,
            "breaks": breaks,
            "mean_utility": {
                "a": round(u_a, 4),
                "m": round(u_m, 4),
                "delta": round(u_m - u_a, 4),
            },
            "ci95_bootstrap": [round(ci_lo, 4), round(ci_hi, 4)],
            "mean_steps": {
                "a": round(a_steps / n, 2),
                "m": round(m_steps / n, 2),
            },
            "redundant_rate": {
                "a": round(a_redundant / max(a_steps, 1), 4),
                "m": round(m_redundant / max(m_steps, 1), 4),
            },
            "catastrophic_regression": (m_s / n) < (a_s / n) - 0.10,
        }

    return subgroups


def freeze_validation_identity() -> dict:
    """Widen hash boundary for validation identity."""
    import re

    m_source = open(ROOT / "scripts" / "run_i3_7e_compact_governor.py").read()
    prompt_match = re.search(
        r'MINIMAL_DECISION_STATE_SYSTEM_PROMPT = """(.*?)"""',
        m_source, re.DOTALL)

    components = {
        "m_builder_script": {
            "path": "scripts/run_i3_7e_compact_governor.py",
            "sha256": sha256_str(m_source),
        },
        "m_system_prompt": {
            "sha256": sha256_str(prompt_match.group(1)) if prompt_match else None,
        },
        "evidence_snapshot_serializer": {
            "path": "hrm_adaptive_memory/executive/evidence_benchmark/serializer.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/serializer.py"),
        },
        "model_decoder": {
            "path": "hrm_adaptive_memory/executive/model_decoder.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/model_decoder.py"),
        },
        "evidence_executor": {
            "path": "hrm_adaptive_memory/executive/evidence_benchmark/executor.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/executor.py"),
        },
        "evidence_schema": {
            "path": "hrm_adaptive_memory/executive/evidence_benchmark/schema.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/evidence_benchmark/schema.py"),
        },
        "utility_config": {
            "path": "configs/v2b_i3_1_utility_v1.json",
            "sha256": sha256_file(ROOT / "configs/v2b_i3_1_utility_v1.json"),
        },
        "action_vocabulary_core": {
            "path": "hrm_adaptive_memory/cognitive_control/core.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/cognitive_control/core.py"),
        },
        "cognitive_state_enums": {
            "path": "hrm_adaptive_memory/cognitive_control/state.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/cognitive_control/state.py"),
        },
        "resource_budget": {
            "path": "hrm_adaptive_memory/executive/resources.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/resources.py"),
        },
        "model_backend": {
            "path": "hrm_adaptive_memory/executive/model_backend.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/model_backend.py"),
        },
        "pinned_model_controller": {
            "path": "hrm_adaptive_memory/executive/pinned_model_controller.py",
            "sha256": sha256_file(ROOT / "hrm_adaptive_memory/executive/pinned_model_controller.py"),
        },
        "validation_manifest": {
            "path": "experiments/v2b_i3_8/manifests/i3_8_evidence_validation_v1.json",
            "sha256": sha256_file(VALIDATION_MANIFEST),
        },
        "validation_results": {
            "path": "experiments/v2b_i3_8/development/i3_8_validation/repair_check_v1.jsonl",
            "sha256": sha256_file(VALIDATION_RESULTS),
        },
        "validation_protocol": {
            "path": "experiments/v2b_i3_8/development/validation_protocol_v1.json",
            "sha256": sha256_file(ROOT / "experiments/v2b_i3_8/development/validation_protocol_v1.json"),
        },
        "model_configuration": {
            "model": "deepseek-chat",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": "json_object",
        },
    }

    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT
    ).decode().strip()
    components["source_commit"] = {"git_sha": commit_hash}

    return components


def main():
    print("I3.8 VALIDATION CLOSURE")
    print("=" * 78)

    with open(VALIDATION_RESULTS) as f:
        results = [json.loads(line) for line in f]
    print(f"Loaded {len(results)} paired results")

    # Overall statistics
    print("\n1. Overall paired statistics...")
    stats = compute_overall_stats(results)
    stats["frozen_validation_claims"] = {
        "C1_E_delta_U_gt_0": stats["mean_delta_u"] > 0,
        "C1_lower_endpoint_95ci_positive": stats["lower_endpoint_two_sided_95ci_positive"],
        "C2_success_M_gt_success_A": stats["success"]["m_success"] > stats["success"]["a_success"],
        "C3_rescues_gt_breaks": stats["mcnemar"]["c_m_only_success"] > stats["mcnemar"]["b_a_only_success"],
        "C4_redundant_M_lt_redundant_A": stats["redundant_actions"]["m_rate"] < stats["redundant_actions"]["a_rate"],
        "C5_steps_M_lt_steps_A": stats["steps"]["m_mean"] < stats["steps"]["a_mean"],
    }
    stats["primary_confirmatory_criterion"] = {
        "criterion": "lower endpoint of two-sided 95% paired bootstrap CI(U_M - U_A) > 0",
        "satisfied": stats["lower_endpoint_two_sided_95ci_positive"],
    }
    stats["safety_criterion"] = {
        "criterion": "Breaks_M <= Rescues_M",
        "breaks": stats["mcnemar"]["b_a_only_success"],
        "rescues": stats["mcnemar"]["c_m_only_success"],
        "satisfied": stats["mcnemar"]["b_a_only_success"] <= stats["mcnemar"]["c_m_only_success"],
        "strictly_zero_breaks": stats["mcnemar"]["b_a_only_success"] == 0,
    }

    stats_path = OUTPUT_DIR / "paired_statistics_v1.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"   Saved: {stats_path}")
    print(f"   Delta U: {stats['mean_delta_u']:+.4f}")
    print(f"   Bootstrap 95% CI: [{stats['ci95_bootstrap'][0]:+.4f}, {stats['ci95_bootstrap'][1]:+.4f}]")
    print(f"   Lower endpoint > 0: {stats['lower_endpoint_two_sided_95ci_positive']}")
    print(f"   McNemar: b={stats['mcnemar']['b_a_only_success']}, c={stats['mcnemar']['c_m_only_success']}, p={stats['mcnemar']['exact_p_value']}")

    # Subgroup analysis
    print("\n2. Subgroup analysis by category...")
    subgroups = compute_subgroup_analysis(results)
    subgroup_path = OUTPUT_DIR / "subgroup_analysis_v1.json"
    subgroup_path.write_text(json.dumps(subgroups, indent=2, sort_keys=True) + "\n")
    print(f"   Saved: {subgroup_path}")
    for cat, sg in subgroups.items():
        print(f"\n   {cat} (n={sg['n_tasks']}):")
        print(f"     Success: A={sg['success']['a']}  M={sg['success']['m']}  delta={sg['success']['delta_pp']}pp")
        print(f"     Rescues: {sg['rescues']}  Breaks: {sg['breaks']}")
        print(f"     Mean U: A={sg['mean_utility']['a']:+.2f}  M={sg['mean_utility']['m']:+.2f}  delta={sg['mean_utility']['delta']:+.2f}")
        print(f"     CI95: [{sg['ci95_bootstrap'][0]:+.2f}, {sg['ci95_bootstrap'][1]:+.2f}]")
        print(f"     Catastrophic regression: {sg['catastrophic_regression']}")

    any_catastrophic = any(sg["catastrophic_regression"] for sg in subgroups.values())
    print(f"\n   Any catastrophic subgroup regression: {any_catastrophic}")

    # Freeze identity
    print("\n3. Freezing validation identity...")
    identity = freeze_validation_identity()
    closure = {
        "schema": "DAPH_V2B_I3_8_VALIDATION_CLOSURE_V1",
        "frozen_artifacts": identity,
        "overall_statistics": {
            "n_pairs": stats["n_pairs"],
            "mean_delta_u": stats["mean_delta_u"],
            "ci95_bootstrap": stats["ci95_bootstrap"],
            "lower_endpoint_two_sided_95ci_positive": stats["lower_endpoint_two_sided_95ci_positive"],
            "rescues": stats["mcnemar"]["c_m_only_success"],
            "breaks": stats["mcnemar"]["b_a_only_success"],
            "mcnemar_p": stats["mcnemar"]["exact_p_value"],
            "all_claims_pass": all(stats["frozen_validation_claims"].values()),
        },
        "subgroup_summary": {
            cat: {
                "n": sg["n_tasks"],
                "rescues": sg["rescues"],
                "breaks": sg["breaks"],
                "delta_success_pp": sg["success"]["delta_pp"],
                "delta_u": sg["mean_utility"]["delta"],
                "catastrophic_regression": sg["catastrophic_regression"],
            }
            for cat, sg in subgroups.items()
        },
        "generalization_criterion": {
            "no catastrophic subgroup regression": not any_catastrophic,
            "expected_primary_gain_source": "evidence_conflict",
            "primary_gain_source": max(
                subgroups.items(),
                key=lambda x: x[1]["rescues"]
            )[0],
        },
        "m_definition": {
            "name": "Minimal Decision State Governor (MDSG)",
            "type": "deterministic controller-visible state compressor",
            "function": "M_t = f(O_t)",
            "does_not_override_actions": True,
            "mechanism": (
                "Detects when the current evidence state is sufficient for "
                "termination and signals READY_TO_ANSWER. The model retains "
                "full action authority."
            ),
        },
        "architectural_finding": (
            "Useful governor = decision-state compressor / stopping controller, "
            "not action recommender."
        ),
    }
    closure_path = OUTPUT_DIR / "validation_closure_v1.json"
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    print(f"   Saved: {closure_path}")

    # Final summary
    print(f"\n{'=' * 78}")
    print("I3.8 VALIDATION SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Tasks: {stats['n_pairs']}")
    print(f"\n  PRIMARY CRITERION (C1):")
    print(f"    lower endpoint of two-sided 95% paired bootstrap CI(Delta U) > 0")
    print(f"    Delta U = {stats['mean_delta_u']:+.4f}")
    print(f"    Bootstrap 95% CI = [{stats['ci95_bootstrap'][0]:+.4f}, {stats['ci95_bootstrap'][1]:+.4f}]")
    print(f"    PASSED: {stats['lower_endpoint_two_sided_95ci_positive']}")
    print(f"\n  SAFETY CRITERION:")
    print(f"    Breaks = {stats['safety_criterion']['breaks']}")
    print(f"    Rescues = {stats['safety_criterion']['rescues']}")
    print(f"    PASSED: {stats['safety_criterion']['satisfied']}")
    print(f"    Zero breaks: {stats['safety_criterion']['strictly_zero_breaks']}")
    print(f"\n  McNemar exact test: p = {stats['mcnemar']['exact_p_value']}")
    print(f"    All discordant pairs favor M: {stats['mcnemar']['all_favor_m']}")
    print(f"\n  FROZEN VALIDATION CLAIMS:")
    for claim, passed in stats["frozen_validation_claims"].items():
        print(f"    {claim}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  SUBGROUP ANALYSIS:")
    for cat, sg in subgroups.items():
        status = "OK" if not sg["catastrophic_regression"] else "CATASTROPHIC"
        print(f"    {cat}: n={sg['n_tasks']}, rescues={sg['rescues']}, breaks={sg['breaks']}, delta_U={sg['mean_utility']['delta']:+.2f} [{status}]")
    print(f"\n  GENERALIZATION:")
    print(f"    No catastrophic subgroup regression: {not any_catastrophic}")
    print(f"    Primary gain source: {closure['generalization_criterion']['primary_gain_source']}")
    print(f"\n  ALL CLAIMS PASS: {all(stats['frozen_validation_claims'].values())}")


if __name__ == "__main__":
    main()
