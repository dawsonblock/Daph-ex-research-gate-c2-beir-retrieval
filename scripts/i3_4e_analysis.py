#!/usr/bin/env python3
"""
I3.4e analysis suite.

Implements the 7 analyses specified for the frozen-permutation experiment:
  1. Permutation distribution summary (mean, median, SD, IQR, min/max, P2/B1 percentile)
  2. P(PS_k > P2) and P(PS_k > B0) across 16 frozen mappings
  3. Two-level bootstrap: paired task-level CIs + permutation-ensemble uncertainty
  4. one_live stratum analysis (success, utility, action rates, rescues/breaks vs P0)
  5. Structural characterization of each permutation (rank per action per phase) +
     regression of performance against mapping properties
  6. P2 vs B0 in the context of the permutation distribution
  7. Freeze the Phase B report

Cautious scientific language throughout. One fixed mapping outperforming B1
does not prove arbitrary rankings help.
"""
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def paired_bootstrap_ci(diffs, n_bootstrap=10000, confidence=0.95, seed=42):
    if not diffs:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0, "excludes_zero": None}
    rng = random.Random(seed)
    n = len(diffs)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = (1 - confidence) / 2
    lower = boot_means[int(alpha * n_bootstrap)]
    upper = boot_means[int((1 - alpha) * n_bootstrap)]
    mean = sum(diffs) / n
    return {
        "mean": round(mean, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "n": n,
        "excludes_zero": (lower > 0) or (upper < 0),
    }


def arm_utilities(results):
    """Return {arm: {task_id: utility}} and {arm: [utility, ...]}."""
    task_arm = defaultdict(dict)
    arm_list = defaultdict(list)
    for r in results:
        arm = r["arm"]
        tid = r["task_id"]
        u = float(r.get("realized_utility", 0.0))
        task_arm[tid][arm] = u
        arm_list[arm].append(u)
    return task_arm, arm_list


# ============================================================
# Analysis 1: Permutation distribution summary
# ============================================================
def analysis_1_distribution(phase_b_results, phase_a_results, output):
    print("\n" + "=" * 60)
    print("Analysis 1: Permutation distribution summary")
    print("=" * 60)

    _, arm_utils = arm_utilities(phase_b_results)
    ps_arms = sorted([a for a in arm_utils if a.startswith("PS") and len(a) > 2])
    ps_means = [statistics.mean(arm_utils[a]) for a in ps_arms]
    ps_medians = [statistics.median(arm_utils[a]) for a in ps_arms]

    # P2 and B0 from Phase A (same dataset, different task subset)
    _, pa_utils = arm_utilities(phase_a_results)
    p2_mean = statistics.mean(pa_utils.get("P2", [0]))
    b0_mean = statistics.mean(pa_utils.get("B0", [0]))

    # Percentile of P2/B0 within the permutation distribution
    p2_percentile = sum(1 for m in ps_means if m < p2_mean) / len(ps_means) * 100
    b0_percentile = sum(1 for m in ps_means if m < b0_mean) / len(ps_means) * 100

    result = {
        "n_permutations": len(ps_arms),
        "mean_of_means": round(statistics.mean(ps_means), 4),
        "median_of_means": round(statistics.median(ps_means), 4),
        "sd_of_means": round(statistics.stdev(ps_means), 4) if len(ps_means) > 1 else None,
        "iqr_lower": round(statistics.quantiles(ps_means, n=4)[0], 4) if len(ps_means) >= 4 else None,
        "iqr_upper": round(statistics.quantiles(ps_means, n=4)[2], 4) if len(ps_means) >= 4 else None,
        "min_mean": round(min(ps_means), 4),
        "max_mean": round(max(ps_means), 4),
        "p2_mean_phaseA": round(p2_mean, 4),
        "b0_mean_phaseA": round(b0_mean, 4),
        "p2_percentile_in_PS_distribution": round(p2_percentile, 2),
        "b0_percentile_in_PS_distribution": round(b0_percentile, 2),
        "per_arm": {
            a: {
                "mean_utility": round(statistics.mean(arm_utils[a]), 4),
                "median_utility": round(statistics.median(arm_utils[a]), 4),
                "success_rate": round(sum(1 for r in phase_b_results if r["arm"] == a and r["success"]) / len(arm_utils[a]), 4),
                "n": len(arm_utils[a]),
            }
            for a in ps_arms
        },
    }

    print(f"  N permutations: {result['n_permutations']}")
    print(f"  Mean of means: {result['mean_of_means']}")
    print(f"  Median of means: {result['median_of_means']}")
    print(f"  SD of means: {result['sd_of_means']}")
    print(f"  IQR: [{result['iqr_lower']}, {result['iqr_upper']}]")
    print(f"  Min/Max: {result['min_mean']} / {result['max_mean']}")
    print(f"  P2 (Phase A) mean: {result['p2_mean_phaseA']} → percentile {result['p2_percentile_in_PS_distribution']}%")
    print(f"  B0 (Phase A) mean: {result['b0_mean_phaseA']} → percentile {result['b0_percentile_in_PS_distribution']}%")

    save_json(output / "01_permutation_distribution.json", result)
    return result


# ============================================================
# Analysis 2: P(PS_k > P2) and P(PS_k > B0)
# ============================================================
def analysis_2_beat_probability(phase_b_results, phase_a_results, output):
    print("\n" + "=" * 60)
    print("Analysis 2: P(PS_k > P2) and P(PS_k > B0)")
    print("=" * 60)

    task_arm_b, arm_utils_b = arm_utilities(phase_b_results)
    task_arm_a, arm_utils_a = arm_utilities(phase_a_results)

    ps_arms = sorted([a for a in arm_utils_b if a.startswith("PS") and len(a) > 2])

    # For P2 and B0, we need Phase A results on the SAME tasks as Phase B
    # Phase B uses a 30-task subset of the 80-task Phase A dataset
    phase_b_task_ids = set(r["task_id"] for r in phase_b_results)
    p2_on_b_tasks = [task_arm_a[tid]["P2"] for tid in phase_b_task_ids if "P2" in task_arm_a.get(tid, {})]
    b0_on_b_tasks = [task_arm_a[tid]["B0"] for tid in phase_b_task_ids if "B0" in task_arm_a.get(tid, {})]

    p2_mean = statistics.mean(p2_on_b_tasks) if p2_on_b_tasks else None
    b0_mean = statistics.mean(b0_on_b_tasks) if b0_on_b_tasks else None

    # Count how many PS arms beat P2/B0
    ps_beat_p2 = 0
    ps_beat_b0 = 0
    for arm in ps_arms:
        ps_mean = statistics.mean(arm_utils_b[arm])
        if p2_mean is not None and ps_mean > p2_mean:
            ps_beat_p2 += 1
        if b0_mean is not None and ps_mean > b0_mean:
            ps_beat_b0 += 1

    p_beat_p2 = ps_beat_p2 / len(ps_arms)
    p_beat_b0 = ps_beat_b0 / len(ps_arms)

    # Also compute paired: for each PS arm, how many tasks does it beat P2/B0?
    paired_beat = {}
    for arm in ps_arms:
        beat_p2_tasks = 0
        beat_b0_tasks = 0
        total_tasks = 0
        for tid in phase_b_task_ids:
            if arm in task_arm_b.get(tid, {}) and "P2" in task_arm_a.get(tid, {}):
                total_tasks += 1
                if task_arm_b[tid][arm] > task_arm_a[tid]["P2"]:
                    beat_p2_tasks += 1
            if arm in task_arm_b.get(tid, {}) and "B0" in task_arm_a.get(tid, {}):
                if task_arm_b[tid][arm] > task_arm_a[tid]["B0"]:
                    beat_b0_tasks += 1
        paired_beat[arm] = {
            "beat_p2_tasks": beat_p2_tasks,
            "beat_b0_tasks": beat_b0_tasks,
            "total_comparable_tasks": total_tasks,
        }

    result = {
        "p2_mean_on_b_tasks": round(p2_mean, 4) if p2_mean else None,
        "b0_mean_on_b_tasks": round(b0_mean, 4) if b0_mean else None,
        "n_ps_beating_p2": ps_beat_p2,
        "n_ps_beating_b0": ps_beat_b0,
        "p_ps_beat_p2": round(p_beat_p2, 4),
        "p_ps_beat_b0": round(p_beat_b0, 4),
        "paired_per_arm": paired_beat,
    }

    print(f"  P2 mean on Phase B tasks: {result['p2_mean_on_b_tasks']}")
    print(f"  B0 mean on Phase B tasks: {result['b0_mean_on_b_tasks']}")
    print(f"  PS arms beating P2: {ps_beat_p2}/{len(ps_arms)} = {p_beat_p2:.1%}")
    print(f"  PS arms beating B0: {ps_beat_b0}/{len(ps_arms)} = {p_beat_b0:.1%}")

    save_json(output / "02_beat_probability.json", result)
    return result


# ============================================================
# Analysis 3: Two-level bootstrap
# ============================================================
def analysis_3_two_level_bootstrap(phase_b_results, phase_a_results, output):
    print("\n" + "=" * 60)
    print("Analysis 3: Two-level bootstrap")
    print("=" * 60)

    task_arm_b, arm_utils_b = arm_utilities(phase_b_results)
    ps_arms = sorted([a for a in arm_utils_b if a.startswith("PS") and len(a) > 2])
    phase_b_task_ids = sorted(set(r["task_id"] for r in phase_b_results))

    # Level 1: Paired task-level CIs within each permutation
    _, arm_utils_a = arm_utilities(phase_a_results)
    p2_utils = {tid: task_arm_b.get(tid, {}).get("P2") for tid in phase_b_task_ids}
    # Actually P2 is in Phase A, not B. Get P2 on Phase B tasks from Phase A
    task_arm_a, _ = arm_utilities(phase_a_results)

    per_arm_cis = {}
    for arm in ps_arms:
        diffs_vs_p2 = []
        diffs_vs_b0 = []
        for tid in phase_b_task_ids:
            ps_u = task_arm_b.get(tid, {}).get(arm)
            p2_u = task_arm_a.get(tid, {}).get("P2")
            b0_u = task_arm_a.get(tid, {}).get("B0")
            if ps_u is not None and p2_u is not None:
                diffs_vs_p2.append(ps_u - p2_u)
            if ps_u is not None and b0_u is not None:
                diffs_vs_b0.append(ps_u - b0_u)
        per_arm_cis[arm] = {
            "vs_p2": paired_bootstrap_ci(diffs_vs_p2, seed=42 + int(arm[2:])),
            "vs_b0": paired_bootstrap_ci(diffs_vs_b0, seed=100 + int(arm[2:])),
        }

    # Level 2: Uncertainty over the permutation ensemble
    # Bootstrap over the 16 PS arms: sample 16 arms with replacement, compute ensemble mean
    rng = random.Random(999)
    ps_arm_means = [statistics.mean(arm_utils_b[a]) for a in ps_arms]
    n_boot = 10000
    ensemble_boot = []
    for _ in range(n_boot):
        sample = [ps_arm_means[rng.randrange(len(ps_arm_means))] for _ in range(len(ps_arm_means))]
        ensemble_boot.append(statistics.mean(sample))
    ensemble_boot.sort()
    ensemble_ci = {
        "mean": round(statistics.mean(ps_arm_means), 4),
        "ci_lower": round(ensemble_boot[int(0.025 * n_boot)], 4),
        "ci_upper": round(ensemble_boot[int(0.975 * n_boot)], 4),
    }

    result = {
        "level1_paired_cis": per_arm_cis,
        "level2_ensemble_ci": ensemble_ci,
        "note": "Level 1: paired task-level bootstrap within each PS arm. Level 2: bootstrap over the 16 PS arm means. Do not infer general random-mapping performance from the single best PS arm.",
    }

    print(f"  Ensemble mean: {ensemble_ci['mean']} CI=[{ensemble_ci['ci_lower']}, {ensemble_ci['ci_upper']}]")
    for arm in ps_arms:
        ci_p2 = per_arm_cis[arm]["vs_p2"]
        ci_b0 = per_arm_cis[arm]["vs_b0"]
        print(f"  {arm} vs P2: mean={ci_p2['mean']:+.2f} CI=[{ci_p2['ci_lower']:+.2f}, {ci_p2['ci_upper']:+.2f}] {'EXCL 0' if ci_p2['excludes_zero'] else 'incl 0'}")
        print(f"  {arm} vs B0: mean={ci_b0['mean']:+.2f} CI=[{ci_b0['ci_lower']:+.2f}, {ci_b0['ci_upper']:+.2f}] {'EXCL 0' if ci_b0['excludes_zero'] else 'incl 0'}")

    save_json(output / "03_two_level_bootstrap.json", result)
    return result


# ============================================================
# Analysis 4: one_live stratum analysis
# ============================================================
def analysis_4_one_live_stratum(phase_b_results, phase_a_results, output):
    print("\n" + "=" * 60)
    print("Analysis 4: one_live stratum analysis")
    print("=" * 60)

    # Identify one_live tasks
    one_live_tasks = set(r["task_id"] for r in phase_b_results if "one_live" in r["task_id"])
    if not one_live_tasks:
        # Try from Phase A
        one_live_tasks = set(r["task_id"] for r in phase_a_results if "one_live" in r["task_id"])
    print(f"  one_live tasks: {len(one_live_tasks)}")

    if not one_live_tasks:
        print("  No one_live tasks found — skipping")
        save_json(output / "04_one_live_stratum.json", {"error": "no one_live tasks"})
        return None

    # Filter to one_live tasks
    b_ol = [r for r in phase_b_results if r["task_id"] in one_live_tasks]
    a_ol = [r for r in phase_a_results if r["task_id"] in one_live_tasks]

    _, arm_utils_b = arm_utilities(b_ol)
    task_arm_b, _ = arm_utilities(b_ol)
    task_arm_a, _ = arm_utilities(a_ol)

    ps_arms = sorted([a for a in arm_utils_b if a.startswith("PS") and len(a) > 2])
    one_live_task_ids = sorted(one_live_tasks)

    per_arm = {}
    for arm in ps_arms + ["P0", "P2", "B0"]:
        rs = [r for r in b_ol if r["arm"] == arm] if arm in arm_utils_b else [r for r in a_ol if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        mean_util = sum(r["realized_utility"] for r in rs) / n
        success_rate = sum(1 for r in rs if r["success"]) / n

        # Action rates from receipts or results
        defer_rate = sum(1 for r in rs if r.get("terminal_action") == "DEFER") / n
        answer_rate = sum(1 for r in rs if r.get("terminal_action") == "ANSWER") / n

        # Step-level action rates (from mechanism receipts)
        # We'll compute from results if available
        action_counts = defaultdict(int)
        total_steps = 0
        for r in rs:
            steps = r.get("n_steps", 0)
            total_steps += steps
            # Terminal action already counted
        # For detailed action rates we'd need receipts; use terminal for now
        per_arm[arm] = {
            "n": n,
            "mean_utility": round(mean_util, 4),
            "success_rate": round(success_rate, 4),
            "defer_rate": round(defer_rate, 4),
            "answer_rate": round(answer_rate, 4),
        }

    # Rescues/breaks vs P0 for each PS arm
    p0_ol = {r["task_id"]: r for r in a_ol if r["arm"] == "P0"}
    rescues_breaks = {}
    for arm in ps_arms:
        rescue = 0
        break_ = 0
        both_success = 0
        both_fail = 0
        total = 0
        for tid in one_live_task_ids:
            ps_r = next((r for r in b_ol if r["arm"] == arm and r["task_id"] == tid), None)
            p0_r = p0_ol.get(tid)
            if ps_r and p0_r:
                total += 1
                if ps_r["success"] and not p0_r["success"]:
                    rescue += 1
                elif not ps_r["success"] and p0_r["success"]:
                    break_ += 1
                elif ps_r["success"] and p0_r["success"]:
                    both_success += 1
                else:
                    both_fail += 1
        rescues_breaks[arm] = {
            "rescue": rescue,
            "break": break_,
            "both_success": both_success,
            "both_fail": both_fail,
            "total": total,
            "net_rescue": rescue - break_,
        }

    result = {
        "n_one_live_tasks": len(one_live_tasks),
        "per_arm": per_arm,
        "rescues_breaks_vs_p0": rescues_breaks,
    }

    print(f"  Per-arm summary (one_live):")
    for arm in sorted(per_arm.keys()):
        a = per_arm[arm]
        print(f"    {arm}: n={a['n']} util={a['mean_utility']:+.2f} success={a['success_rate']:.2f} DEFER={a['defer_rate']:.2f} ANSWER={a['answer_rate']:.2f}")
    print(f"  Rescues/breaks vs P0:")
    for arm in ps_arms:
        rb = rescues_breaks[arm]
        print(f"    {arm}: rescue={rb['rescue']} break={rb['break']} net={rb['net_rescue']:+d}")

    save_json(output / "04_one_live_stratum.json", result)
    return result


# ============================================================
# Analysis 5: Structural characterization + regression
# ============================================================
def analysis_5_structural_regression(phase_b_results, output):
    print("\n" + "=" * 60)
    print("Analysis 5: Structural characterization + regression")
    print("=" * 60)

    # Load all 16 PS mappings
    ps_dir = REPO_ROOT / "experiments/i3_4/value/ps_ensemble"
    ps_arms = sorted([a for a in set(r["arm"] for r in phase_b_results) if a.startswith("PS") and len(a) > 2])

    _, arm_utils = arm_utilities(phase_b_results)

    # For each PS arm, characterize the mapping structure
    mapping_props = {}
    for arm in ps_arms:
        idx = arm[2:]
        mapping_path = ps_dir / f"ps{idx}_mapping.json"
        if not mapping_path.exists():
            print(f"  WARNING: {mapping_path} not found")
            continue
        mapping_data = load_json(mapping_path)
        mapping = mapping_data["mapping"]

        # For each phase, record which action gets rank 1, 2, etc.
        # Rank = descending order of value
        phases = sorted(mapping.keys())
        action_ranks = {}  # {action: [rank_in_phase1, rank_in_phase2, ...]}
        for phase in phases:
            values = mapping[phase]
            sorted_actions = sorted(values.keys(), key=lambda a: values[a], reverse=True)
            for rank, action in enumerate(sorted_actions, 1):
                if action not in action_ranks:
                    action_ranks[action] = []
                action_ranks[action].append(rank)

        # Average rank per action across phases
        avg_ranks = {a: round(statistics.mean(ranks), 4) for a, ranks in action_ranks.items()}

        # Mean utility for this arm
        mean_util = statistics.mean(arm_utils[arm])

        mapping_props[arm] = {
            "mean_utility": round(mean_util, 4),
            "action_ranks_per_phase": {p: {a: i + 1 for i, a in enumerate(sorted(mapping[p].keys(), key=lambda x: mapping[p][x], reverse=True))} for p in phases},
            "avg_rank_per_action": avg_ranks,
            "mapping_sha256": mapping_data.get("sha256", ""),
        }

    # Simple regression: mean_util ~ avg_rank(DEFER) + avg_rank(RETRIEVE) + avg_rank(ANSWER) + avg_rank(VERIFY)
    # Use numpy least squares
    actions_to_test = ["DEFER", "RETRIEVE", "ANSWER", "VERIFY", "SEARCH_MORE"]
    X = []
    y = []
    for arm in ps_arms:
        if arm not in mapping_props:
            continue
        props = mapping_props[arm]
        row = []
        for action in actions_to_test:
            row.append(props["avg_rank_per_action"].get(action, 3.0))  # default mid-rank
        X.append(row)
        y.append(props["mean_utility"])

    X = np.array(X)
    y = np.array(y)

    # Fit OLS
    if len(X) >= len(actions_to_test) + 1:
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        y_pred = X_with_intercept @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        regression_result = {
            "actions": actions_to_test,
            "coefficients": {
                "intercept": round(float(coeffs[0]), 4),
                **{a: round(float(coeffs[i + 1]), 4) for i, a in enumerate(actions_to_test)},
            },
            "r_squared": round(float(r_squared), 4),
            "n": len(y),
        }
        print(f"  Regression R² = {r_squared:.4f}")
        print(f"  Coefficients: intercept={coeffs[0]:.2f}", end="")
        for i, a in enumerate(actions_to_test):
            print(f" {a}={coeffs[i+1]:.2f}", end="")
        print()
    else:
        regression_result = {"error": "insufficient data for regression"}

    # Also compute simple correlations
    correlations = {}
    for i, action in enumerate(actions_to_test):
        if len(X) > 2:
            corr = float(np.corrcoef(X[:, i], y)[0, 1])
            correlations[action] = round(corr, 4)

    result = {
        "mapping_properties": mapping_props,
        "regression": regression_result,
        "correlations_rank_vs_utility": correlations,
    }

    print(f"  Correlations (avg rank vs utility):")
    for a, c in correlations.items():
        print(f"    {a}: r={c:+.4f}")

    save_json(output / "05_structural_regression.json", result)
    return result


# ============================================================
# Analysis 6: P2 vs B0 in context of permutation distribution
# ============================================================
def analysis_6_p2_vs_b0_context(phase_a_results, phase_b_results, output):
    print("\n" + "=" * 60)
    print("Analysis 6: P2 vs B0 in permutation context")
    print("=" * 60)

    task_arm_a, arm_utils_a = arm_utilities(phase_a_results)
    _, arm_utils_b = arm_utilities(phase_b_results)

    # P2 vs B0 paired on Phase A
    p2_b0_diffs = []
    for tid in task_arm_a:
        if "P2" in task_arm_a[tid] and "B0" in task_arm_a[tid]:
            p2_b0_diffs.append(task_arm_a[tid]["P2"] - task_arm_a[tid]["B0"])
    p2_b0_ci = paired_bootstrap_ci(p2_b0_diffs)

    # Where does P2-B0 sit relative to PS-P2 spread?
    ps_arms = sorted([a for a in arm_utils_b if a.startswith("PS") and len(a) > 2])
    ps_p2_diffs = []
    task_arm_b, _ = arm_utilities(phase_b_results)
    phase_b_task_ids = sorted(set(r["task_id"] for r in phase_b_results))
    for arm in ps_arms:
        for tid in phase_b_task_ids:
            ps_u = task_arm_b.get(tid, {}).get(arm)
            p2_u = task_arm_a.get(tid, {}).get("P2")
            if ps_u is not None and p2_u is not None:
                ps_p2_diffs.append((arm, ps_u - p2_u))

    # Per-arm PS-P2 mean
    ps_p2_per_arm = defaultdict(list)
    for arm, diff in ps_p2_diffs:
        ps_p2_per_arm[arm].append(diff)
    ps_p2_means = {arm: statistics.mean(diffs) for arm, diffs in ps_p2_per_arm.items()}

    # P2-B0 on Phase B tasks only
    p2_b0_diffs_b = []
    for tid in phase_b_task_ids:
        if "P2" in task_arm_a.get(tid, {}) and "B0" in task_arm_a.get(tid, {}):
            p2_b0_diffs_b.append(task_arm_a[tid]["P2"] - task_arm_a[tid]["B0"])
    p2_b0_ci_b = paired_bootstrap_ci(p2_b0_diffs_b)

    result = {
        "p2_vs_b0_phaseA_all": p2_b0_ci,
        "p2_vs_b0_phaseB_tasks": p2_b0_ci_b,
        "ps_minus_p2_per_arm": {a: round(m, 4) for a, m in ps_p2_means.items()},
        "interpretation": "If P2-B0 CI includes 0 and multiple PS arms match or exceed P2, the parsimonious champion is B0 unless P2 shows a distinct advantage on specific hard strata.",
    }

    print(f"  P2-B0 (Phase A, all 80 tasks): mean={p2_b0_ci['mean']:+.4f} CI=[{p2_b0_ci['ci_lower']:+.4f}, {p2_b0_ci['ci_upper']:+.4f}] {'EXCL 0' if p2_b0_ci['excludes_zero'] else 'incl 0'}")
    print(f"  P2-B0 (Phase B tasks only):    mean={p2_b0_ci_b['mean']:+.4f} CI=[{p2_b0_ci_b['ci_lower']:+.4f}, {p2_b0_ci_b['ci_upper']:+.4f}] {'EXCL 0' if p2_b0_ci_b['excludes_zero'] else 'incl 0'}")
    print(f"  PS-P2 per arm:")
    for a, m in sorted(ps_p2_means.items()):
        print(f"    {a}: {m:+.4f}")

    save_json(output / "06_p2_vs_b0_context.json", result)
    return result


# ============================================================
# Analysis 7: Frozen report
# ============================================================
def analysis_7_frozen_report(phase_a_results, phase_b_results, all_results, output):
    print("\n" + "=" * 60)
    print("Analysis 7: Frozen Phase B report")
    print("=" * 60)

    _, arm_utils_all = arm_utilities(all_results)

    report = {
        "experiment": "i3_4e",
        "phases": {
            "A": {"n_trajectories": 560, "n_tasks": 80, "arms": ["P0", "P2", "B0", "CONST", "DEFER", "PV", "PR"]},
            "B": {"n_trajectories": 480, "n_tasks": 30, "arms": [f"PS{i:02d}" for i in range(1, 17)]},
        },
        "phase_a_summary": {
            arm: {
                "mean_utility": round(statistics.mean(arm_utils_all[arm]), 4),
                "success_rate": round(sum(1 for r in phase_a_results if r["arm"] == arm and r["success"]) / len(arm_utils_all[arm]), 4),
                "n": len(arm_utils_all[arm]),
            }
            for arm in ["P0", "P2", "B0", "CONST", "DEFER", "PV", "PR"] if arm in arm_utils_all
        },
        "phase_b_summary": {
            arm: {
                "mean_utility": round(statistics.mean(arm_utils_all[arm]), 4),
                "success_rate": round(sum(1 for r in phase_b_results if r["arm"] == arm and r["success"]) / len(arm_utils_all[arm]), 4),
                "n": len(arm_utils_all[arm]),
            }
            for arm in sorted(arm_utils_all.keys()) if arm.startswith("PS") and len(arm) > 2
        },
        "cautionary_note": (
            "One particular fixed alternative action-value mapping substantially outperforms B1 "
            "on this held-out task distribution. This does not prove that arbitrary rankings help. "
            "The permutation distribution reveals which mappings succeed and why, but the sample "
            "of 16 frozen permutations is not a random sample from the space of all possible mappings."
        ),
    }

    save_json(output / "07_frozen_report.json", report)
    print(f"  Report saved to {output / '07_frozen_report.json'}")
    return report


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a", default="experiments/i3_4/runs/i3_4e_phaseA/results.jsonl")
    parser.add_argument("--phase-b", default="experiments/i3_4/runs/i3_4e_phaseB/results.jsonl")
    parser.add_argument("--output", default="experiments/i3_4/runs/i3_4e_analysis")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    phase_a = load_jsonl(REPO_ROOT / args.phase_a)
    phase_b = load_jsonl(REPO_ROOT / args.phase_b)

    print(f"Phase A: {len(phase_a)} trajectories")
    print(f"Phase B: {len(phase_b)} trajectories")

    # Run all 7 analyses
    analysis_1_distribution(phase_b, phase_a, output)
    analysis_2_beat_probability(phase_b, phase_a, output)
    analysis_3_two_level_bootstrap(phase_b, phase_a, output)
    analysis_4_one_live_stratum(phase_b, phase_a, output)
    analysis_5_structural_regression(phase_b, output)
    analysis_6_p2_vs_b0_context(phase_a, phase_b, output)
    analysis_7_frozen_report(phase_a, phase_b, phase_a + phase_b, output)

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"Results in: {output}")


if __name__ == "__main__":
    main()
