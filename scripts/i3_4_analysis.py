#!/usr/bin/env python3
"""DAPH I3.4d — Phase-specific analysis with paired bootstrap CIs.

Analyzes the P0/P1/P2/PS executive experiment:
  - Utility contrasts: ΔU_P1, ΔU_P2, ΔU_PS, ΔU_P2-P1, ΔU_P2-PS (paired bootstrap CIs)
  - Phase-specific analysis: per-phase action distribution, utility, success
  - Terminal-action distribution
  - Success rates
  - Phase transition analysis
  - PS causal control: P2 > PS shows correct values matter, not just structure

Usage:
    PYTHONPATH=scripts:. python3 scripts/i3_4_analysis.py \
        --results /path/to/results.jsonl \
        --receipts /path/to/mechanism_receipts.jsonl \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/analysis/
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph.phase.ontology import Phase, ALL_PHASES


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def paired_bootstrap_ci(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
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


def step1_integrity(results: list[dict], dataset: list[dict], output: Path) -> dict:
    print("Step 1: Integrity")
    n_traj = len(results)
    n_tasks = len(dataset)
    task_ids_dataset = set(t["task_id"] for t in dataset)
    task_ids_results = set(r["task_id"] for r in results)
    arms = set(r["arm"] for r in results)
    decoder_errors = sum(r.get("decoder_errors", 0) for r in results)
    n_arms = len(arms)

    result = {
        "n_trajectories": n_traj,
        "n_tasks": n_tasks,
        "n_arms": n_arms,
        "task_ids_match": task_ids_dataset == task_ids_results,
        "arms": sorted(arms),
        "decoder_errors": decoder_errors,
        "trajectory_count_matches": n_traj == n_tasks * n_arms,
    }
    save_json(output / "01_integrity.json", result)
    print(f"  Trajectories: {n_traj}, Tasks: {n_tasks}, Arms: {n_arms}, Match: {result['task_ids_match']}")
    return result


def step2_utility_contrasts(results: list[dict], output: Path) -> dict:
    print("Step 2: Utility contrasts (paired bootstrap CIs)")
    task_arm_utility: dict[str, dict[str, float]] = defaultdict(dict)
    arm_utilities: dict[str, list[float]] = defaultdict(list)

    for r in results:
        arm = r["arm"]
        task_id = r["task_id"]
        utility = float(r.get("realized_utility", 0.0))
        task_arm_utility[task_id][arm] = utility
        arm_utilities[arm].append(utility)

    mean_utility = {arm: sum(us) / len(us) for arm, us in arm_utilities.items() if us}

    def paired_diffs(treated, control="P0"):
        return [
            arms[treated] - arms[control]
            for arms in task_arm_utility.values()
            if treated in arms and control in arms
        ]

    contrasts = {}

    # Standard contrasts (always present)
    p1_diffs = paired_diffs("P1")
    p2_diffs = paired_diffs("P2")
    p2_p1_diffs = paired_diffs("P2", "P1")

    contrasts["delta_P1"] = paired_bootstrap_ci(p1_diffs)
    contrasts["delta_P2"] = paired_bootstrap_ci(p2_diffs)
    contrasts["delta_P2_minus_P1"] = paired_bootstrap_ci(p2_p1_diffs)

    # PS contrasts (only if PS arm exists)
    if "PS" in arm_utilities:
        ps_diffs = paired_diffs("PS")
        p2_ps_diffs = paired_diffs("P2", "PS")
        ps_p1_diffs = paired_diffs("PS", "P1")
        contrasts["delta_PS"] = paired_bootstrap_ci(ps_diffs)
        contrasts["delta_P2_minus_PS"] = paired_bootstrap_ci(p2_ps_diffs)
        contrasts["delta_PS_minus_P1"] = paired_bootstrap_ci(ps_p1_diffs)

    result = {
        "mean_utility": {k: round(v, 4) for k, v in mean_utility.items()},
        "contrasts": contrasts,
        "n_per_arm": {arm: len(us) for arm, us in arm_utilities.items()},
    }
    save_json(output / "02_utility_contrasts.json", result)

    print(f"  Mean utility: {result['mean_utility']}")
    for name, ci in result["contrasts"].items():
        excl = "EXCLUDES 0" if ci["excludes_zero"] else "includes 0"
        print(f"  {name}: mean={ci['mean']:+.4f} CI=[{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] {excl}")
    return result


def step3_success_rates(results: list[dict], output: Path) -> dict:
    print("Step 3: Success rates")
    arm_success: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        arm = r["arm"]
        arm_success[arm]["total"] += 1
        if r.get("success"):
            arm_success[arm]["success"] += 1

    result = {
        arm: {
            "success": d["success"],
            "total": d["total"],
            "rate": d["success"] / d["total"] if d["total"] else 0.0,
        }
        for arm, d in arm_success.items()
    }
    save_json(output / "03_success_rates.json", result)
    for arm, d in sorted(result.items()):
        print(f"  {arm}: {d['success']}/{d['total']} = {d['rate']:.4f}")
    return result


def step4_terminal_actions(results: list[dict], output: Path) -> dict:
    print("Step 4: Terminal-action distribution")
    arm_terminals: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        arm = r["arm"]
        terminal = r.get("terminal_action", "NONE")
        arm_terminals[arm][terminal] += 1

    result = {arm: dict(counter) for arm, counter in arm_terminals.items()}
    save_json(output / "04_terminal_actions.json", result)
    for arm, dist in sorted(result.items()):
        print(f"  {arm}: {dist}")
    return result


def step5_phase_specific(receipts: list[dict], results: list[dict], output: Path) -> dict:
    print("Step 5: Phase-specific analysis")
    # Group receipts by (arm, phase)
    arm_phase_actions: dict[tuple[str, str], Counter] = defaultdict(Counter)
    arm_phase_count: dict[tuple[str, str], int] = defaultdict(int)

    for r in receipts:
        arm = r.get("arm", "")
        phase = r.get("phase_before", "UNKNOWN")
        action = r.get("selected_action", "")
        arm_phase_actions[(arm, phase)][action] += 1
        arm_phase_count[(arm, phase)] += 1

    result = {}
    for arm in sorted(set(r.get("arm", "") for r in receipts)):
        result[arm] = {}
        for phase in sorted(ALL_PHASES, key=lambda p: p.value):
            pval = phase.value
            key = (arm, pval)
            n = arm_phase_count.get(key, 0)
            actions = dict(arm_phase_actions.get(key, Counter()))
            result[arm][pval] = {
                "n": n,
                "action_distribution": {
                    a: round(c / n, 4) for a, c in sorted(actions.items())
                } if n else {},
            }

    save_json(output / "05_phase_specific.json", result)

    # Print summary
    for arm in sorted(result):
        print(f"  {arm}:")
        for phase in sorted(result[arm]):
            d = result[arm][phase]
            if d["n"] > 0:
                dist = ", ".join(f"{a}={p:.2f}" for a, p in sorted(d["action_distribution"].items(), key=lambda x: -x[1]))
                print(f"    {phase:30s} n={d['n']:4d}  {dist}")
    return result


def step6_phase_transitions(receipts: list[dict], output: Path) -> dict:
    print("Step 6: Phase transitions")
    # Group by trajectory and compute transitions
    traj_receipts: dict[str, list[dict]] = defaultdict(list)
    for r in receipts:
        traj_receipts[r.get("trajectory_key", "")].append(r)
    for key in traj_receipts:
        traj_receipts[key].sort(key=lambda r: r.get("step", 0))

    transition_counts: dict[str, Counter] = defaultdict(Counter)
    for key, recs in traj_receipts.items():
        for i in range(len(recs) - 1):
            before = recs[i].get("phase_before", "UNKNOWN")
            after = recs[i+1].get("phase_before", "UNKNOWN")
            transition_counts[before][after] += 1

    result = {}
    for phase_before in sorted(transition_counts):
        total = sum(transition_counts[phase_before].values())
        result[phase_before] = {
            phase_after: round(count / total, 4)
            for phase_after, count in sorted(transition_counts[phase_before].items())
        }
    save_json(output / "06_phase_transitions.json", result)
    for phase_before in sorted(result):
        parts = [f"{p}={v:.2f}" for p, v in sorted(result[phase_before].items())]
        print(f"  {phase_before:30s} → {', '.join(parts)}")
    return result


def step7_ablation_summary(contrasts: dict, success: dict, output: Path) -> dict:
    print("Step 7: Disposition summary")

    p2_ci = contrasts.get("delta_P2", {})
    p1_ci = contrasts.get("delta_P1", {})
    p2_p1_ci = contrasts.get("delta_P2_minus_P1", {})
    p2_ps_ci = contrasts.get("delta_P2_minus_PS", {})
    ps_ci = contrasts.get("delta_PS", {})

    p2_positive = p2_ci.get("excludes_zero", False) and p2_ci.get("mean", 0) > 0
    p1_neutral = not p1_ci.get("excludes_zero", False)
    p2_better_than_p1 = p2_p1_ci.get("excludes_zero", False) and p2_p1_ci.get("mean", 0) > 0

    # PS causal control checks
    has_ps = "delta_P2_minus_PS" in contrasts
    p2_better_than_ps = (
        has_ps and p2_ps_ci.get("excludes_zero", False) and p2_ps_ci.get("mean", 0) > 0
    )
    ps_not_better_than_p0 = (
        has_ps and not (ps_ci.get("excludes_zero", False) and ps_ci.get("mean", 0) > 0)
    )

    # Success non-inferiority check
    p0_success = success.get("P0", {}).get("rate", 0)
    p2_success = success.get("P2", {}).get("rate", 0)
    success_non_inferior = p2_success >= p0_success - 0.05  # 5% margin

    # Promotion requires P2 positive, non-inferior success, and (if PS exists) P2 > PS
    p2_promotion = p2_positive and success_non_inferior
    if has_ps:
        p2_promotion = p2_promotion and p2_better_than_ps

    result = {
        "P2_positive": p2_positive,
        "P1_neutral": p1_neutral,
        "P2_better_than_P1": p2_better_than_p1,
        "has_PS_control": has_ps,
        "P2_better_than_PS": p2_better_than_ps,
        "PS_not_better_than_P0": ps_not_better_than_p0,
        "success_non_inferior": success_non_inferior,
        "P2_promotion": p2_promotion,
        "contrasts": contrasts,
        "success_rates": success,
    }

    print(f"  P2 positive (CI excludes 0): {p2_positive}")
    print(f"  P1 neutral: {p1_neutral}")
    print(f"  P2 > P1: {p2_better_than_p1}")
    if has_ps:
        print(f"  P2 > PS (correct values matter): {p2_better_than_ps}")
        print(f"  PS not > P0 (structure alone doesn't help): {ps_not_better_than_p0}")
    print(f"  Success non-inferior: {success_non_inferior}")
    print(f"  P2 promotion: {result['P2_promotion']}")

    save_json(output / "07_disposition.json", result)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="I3.4c Analysis")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = load_jsonl(args.results)
    receipts = load_jsonl(args.receipts)
    dataset = load_jsonl(args.dataset)

    print(f"Results: {len(results)}, Receipts: {len(receipts)}, Dataset: {len(dataset)}")
    print()

    s1 = step1_integrity(results, dataset, args.output)
    print()
    s2 = step2_utility_contrasts(results, args.output)
    print()
    s3 = step3_success_rates(results, args.output)
    print()
    s4 = step4_terminal_actions(results, args.output)
    print()
    s5 = step5_phase_specific(receipts, results, args.output)
    print()
    s6 = step6_phase_transitions(receipts, args.output)
    print()
    s7 = step7_ablation_summary(s2["contrasts"], s3, args.output)

    # Summary
    summary = {
        "integrity_ok": s1["task_ids_match"] and s1["trajectory_count_matches"],
        "n_trajectories": s1["n_trajectories"],
        "utility_contrasts": s2["contrasts"],
        "success_rates": {arm: d["rate"] for arm, d in s3.items()},
        "P2_promotion": s7["P2_promotion"],
    }
    save_json(args.output / "summary.json", summary)

    print(f"\n=== Analysis Summary ===")
    print(f"  Integrity OK: {summary['integrity_ok']}")
    print(f"  Trajectories: {summary['n_trajectories']}")
    for name, ci in summary["utility_contrasts"].items():
        excl = "EXCLUDES 0" if ci.get("excludes_zero") else "includes 0"
        print(f"  {name}: mean={ci.get('mean', 0):+.4f} CI=[{ci.get('ci_lower', 0):+.4f}, {ci.get('ci_upper', 0):+.4f}] {excl}")
    print(f"  P2 promotion: {summary['P2_promotion']}")


if __name__ == "__main__":
    main()
