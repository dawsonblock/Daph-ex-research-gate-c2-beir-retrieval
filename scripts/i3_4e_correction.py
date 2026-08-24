#!/usr/bin/env python3
"""
I3.4E-ANALYSIS-CORRECTION-001

Corrects the cross-task percentile inconsistency in the frozen Phase B report.

The original report (01_permutation_distribution.json) compared Phase A P2/B0
means (all 80 tasks) against the Phase B PS distribution (30 tasks). Those are
different task samples. This correction recomputes every permutation-relative
statistic on the same 30 Phase B tasks.

Old (incorrect) values:
  P2 percentile: 50.0%  (using Phase A 80-task mean of 42.43)
  B0 percentile: 18.75% (using Phase A 80-task mean of 37.57)

Corrected values (same 30 Phase B tasks):
  P2 on Phase B tasks: 47.74  → 75.0th percentile (12/16 PS below)
  B0 on Phase B tasks: 41.19  → 43.8th percentile (7/16 PS below)

This correction makes P2/B1 look somewhat better, not worse.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def main():
    phase_a = load_jsonl(REPO_ROOT / "experiments/i3_4/runs/i3_4e_phaseA/results.jsonl")
    phase_b = load_jsonl(REPO_ROOT / "experiments/i3_4/runs/i3_4e_phaseB/results.jsonl")

    b_task_ids = set(r["task_id"] for r in phase_b)

    # P2, B0, P0 on Phase B tasks (from Phase A results)
    task_arm_a = defaultdict(dict)
    for r in phase_a:
        task_arm_a[r["task_id"]][r["arm"]] = float(r["realized_utility"])

    p2_on_b = [task_arm_a[tid]["P2"] for tid in b_task_ids if "P2" in task_arm_a.get(tid, {})]
    b0_on_b = [task_arm_a[tid]["B0"] for tid in b_task_ids if "B0" in task_arm_a.get(tid, {})]
    p0_on_b = [task_arm_a[tid]["P0"] for tid in b_task_ids if "P0" in task_arm_a.get(tid, {})]

    p2_mean = statistics.mean(p2_on_b)
    b0_mean = statistics.mean(b0_on_b)
    p0_mean = statistics.mean(p0_on_b)

    # PS means on Phase B
    task_arm_b = defaultdict(dict)
    for r in phase_b:
        task_arm_b[r["task_id"]][r["arm"]] = float(r["realized_utility"])

    ps_arms = sorted([a for a in set(r["arm"] for r in phase_b) if a.startswith("PS") and len(a) > 2])
    ps_means = {}
    for arm in ps_arms:
        utils = [task_arm_b[tid][arm] for tid in b_task_ids if arm in task_arm_b.get(tid, {})]
        ps_means[arm] = statistics.mean(utils)

    ps_mean_list = list(ps_means.values())

    # Corrected percentiles
    p2_below = sum(1 for m in ps_mean_list if m < p2_mean)
    b0_below = sum(1 for m in ps_mean_list if m < b0_mean)
    p2_pct = p2_below / len(ps_mean_list) * 100
    b0_pct = b0_below / len(ps_mean_list) * 100

    # PS - P2 and PS - B0
    ps_minus_p2 = {a: round(m - p2_mean, 4) for a, m in ps_means.items()}
    ps_minus_b0 = {a: round(m - b0_mean, 4) for a, m in ps_means.items()}

    beat_p2 = sum(1 for m in ps_mean_list if m > p2_mean)
    beat_b0 = sum(1 for m in ps_mean_list if m > b0_mean)

    # Distribution stats on same 30 tasks
    dist_result = {
        "correction_id": "I3.4E-ANALYSIS-CORRECTION-001",
        "n_phase_b_tasks": len(b_task_ids),
        "p2_mean_on_phase_b_tasks": round(p2_mean, 4),
        "b0_mean_on_phase_b_tasks": round(b0_mean, 4),
        "p0_mean_on_phase_b_tasks": round(p0_mean, 4),
        "ps_distribution_on_phase_b_tasks": {
            "n_permutations": len(ps_arms),
            "mean_of_means": round(statistics.mean(ps_mean_list), 4),
            "median_of_means": round(statistics.median(ps_mean_list), 4),
            "sd_of_means": round(statistics.stdev(ps_mean_list), 4),
            "iqr_lower": round(statistics.quantiles(ps_mean_list, n=4)[0], 4),
            "iqr_upper": round(statistics.quantiles(ps_mean_list, n=4)[2], 4),
            "min_mean": round(min(ps_mean_list), 4),
            "max_mean": round(max(ps_mean_list), 4),
            "per_arm": {a: round(m, 4) for a, m in ps_means.items()},
        },
    }

    percentile_result = {
        "correction_id": "I3.4E-ANALYSIS-CORRECTION-001",
        "p2_percentile_corrected": round(p2_pct, 2),
        "b0_percentile_corrected": round(b0_pct, 2),
        "n_ps_below_p2": p2_below,
        "n_ps_below_b0": b0_below,
        "n_ps_beating_p2": beat_p2,
        "n_ps_beating_b0": beat_b0,
        "median_ps_minus_p2": round(statistics.median(ps_minus_p2.values()), 4),
        "mean_ps_minus_p2": round(statistics.mean(ps_minus_p2.values()), 4),
        "median_ps_minus_b0": round(statistics.median(ps_minus_b0.values()), 4),
        "mean_ps_minus_b0": round(statistics.mean(ps_minus_b0.values()), 4),
        "ps_minus_p2_per_arm": ps_minus_p2,
        "ps_minus_b0_per_arm": ps_minus_b0,
        "old_incorrect_values": {
            "p2_percentile_old": 50.0,
            "b0_percentile_old": 18.75,
            "p2_mean_used_old": 42.4305,
            "b0_mean_used_old": 37.5726,
            "explanation": "Old report used Phase A 80-task means compared against Phase B 30-task PS distribution.",
        },
        "corrected_interpretation": (
            "P2 performs above most frozen permutations (75th percentile on matched tasks). "
            "B0 performs around the middle (43.8th percentile). "
            "Some fixed mappings still outperform P2 (4/16). "
            "P2 does not significantly beat B0 (CI includes 0). "
            "B1 is useful but not sufficiently state-specific."
        ),
    }

    output_dir = REPO_ROOT / "experiments/i3_4/runs/i3_4e_analysis/correction"
    save_json(output_dir / "phase_b_same_task_distribution.json", dist_result)
    save_json(output_dir / "phase_b_corrected_percentiles.json", percentile_result)

    print("Correction complete.")
    print(f"  P2 percentile: {p2_pct:.1f}% (was 50.0%)")
    print(f"  B0 percentile: {b0_pct:.1f}% (was 18.75%)")
    print(f"  PS > P2: {beat_p2}/16")
    print(f"  PS > B0: {beat_b0}/16")
    print(f"  median(PS-P2): {statistics.median(ps_minus_p2.values()):+.4f}")
    print(f"  median(PS-B0): {statistics.median(ps_minus_b0.values()):+.4f}")


if __name__ == "__main__":
    main()
