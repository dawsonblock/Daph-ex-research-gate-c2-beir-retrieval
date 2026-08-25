#!/usr/bin/env python3
"""I3.5-PQ Phase 21: Interface-ablation analysis.

Computes all preregistered endpoints for the interface-ablation experiment.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_trajectories(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * 0.5 ** n
    return 2 * p


def bootstrap_ci_paired(differences: list[float], n_bootstrap: int = 10000,
                         confidence: float = 0.95) -> tuple[float, float]:
    if not differences:
        return (0.0, 0.0)
    n = len(differences)
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(differences, size=n, replace=True)
        means.append(np.mean(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return (float(lower), float(upper))


def get_subtype(tid: str) -> str:
    parts = tid.split("_")
    return "_".join(parts[2:4])


def main():
    ablation_dir = REPO_ROOT / "experiments/i3_5/interface_ablation"

    print("Loading data...")
    trajectories = load_trajectories(ablation_dir / "trajectories_v1.jsonl")
    print(f"  {len(trajectories)} trajectories")

    arms = ["C0", "I0", "I1", "I2", "I3", "I4"]
    arm_names = {
        "C0": "no guidance (baseline)",
        "I0": "current normalized values",
        "I1": "raw centered advantages",
        "I2": "epsilon near-optimal set",
        "I3": "confidence-aware recommendation",
        "I4": "clear-choice only (gap > tau)",
    }

    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in trajectories))
    subtypes = sorted(set(get_subtype(tid) for tid in task_ids))

    # ================================================================
    # 1. Overall results per arm
    # ================================================================
    print("\n" + "=" * 80)
    print("1. OVERALL RESULTS PER ARM")
    print("=" * 80)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        us = [r["realized_utility"] for r in recs]
        sr = sum(1 for r in recs if r["success"]) / n
        retr = [r["retrieve_count"] for r in recs]
        max_rep = [r["max_action_repeat"] for r in recs]
        max_consec = [r["max_consecutive_repeat"] for r in recs]
        pd = sum(1 for r in recs if r["premature_defer"])
        pa = sum(1 for r in recs if r["premature_answer"])
        steps = [r["steps"] for r in recs]
        print(f"  {arm:4s} ({arm_names[arm]:40s}):")
        print(f"       n={n} mean_U={sum(us)/n:.2f} success={sr:.4f} "
              f"mean_retr={sum(retr)/n:.2f} mean_max_repeat={sum(max_rep)/n:.2f} "
              f"mean_max_consec={sum(max_consec)/n:.2f}")
        print(f"       mean_steps={sum(steps)/n:.2f} pD={pd} pA={pa}")

    # ================================================================
    # 2. Per-subtype results (the key diagnostic)
    # ================================================================
    print("\n" + "=" * 80)
    print("2. PER-SUBTYPE RESULTS")
    print("=" * 80)

    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        print(f"\n  {subtype} ({len(subtype_tasks)} tasks):")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in subtype_tasks if (tid, arm) in by_task_arm]
            if recs:
                us = [r["realized_utility"] for r in recs]
                sr = sum(1 for r in recs if r["success"]) / len(recs)
                retr = [r["retrieve_count"] for r in recs]
                max_rep = [r["max_action_repeat"] for r in recs]
                print(f"    {arm:4s}: n={len(recs):2d} mean_U={sum(us)/len(us):.2f} "
                      f"success={sr:.2f} mean_retr={sum(retr)/len(retr):.2f} "
                      f"max_repeat={sum(max_rep)/len(max_rep):.2f}")

    # ================================================================
    # 3. Primary target: E[#RETRIEVE] on ol_retrieve
    # ================================================================
    print("\n" + "=" * 80)
    print("3. PRIMARY TARGET: E[#RETRIEVE] on ol_retrieve")
    print("=" * 80)

    ol_ret_tasks = [tid for tid in task_ids if get_subtype(tid) == "ol_retrieve"]
    print(f"\n  Target: E[#RETRIEVE] should fall from ~3 (I0) toward ~1 (C0) without reducing success")
    print()
    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in ol_ret_tasks if (tid, arm) in by_task_arm]
        if recs:
            retr = [r["retrieve_count"] for r in recs]
            us = [r["realized_utility"] for r in recs]
            sr = sum(1 for r in recs if r["success"]) / len(recs)
            stops = sum(1 for r in recs if r["terminal_result"] == "STOP")
            print(f"  {arm:4s}: E[#RETRIEVE]={sum(retr)/len(retr):.2f} "
                  f"mean_U={sum(us)/len(us):.2f} success={sr:.2f} STOPs={stops}")

    # ================================================================
    # 4. Delta-U contrasts (paired 95% CI)
    # ================================================================
    print("\n" + "=" * 80)
    print("4. DELTA-U CONTRASTS (paired 95% CI)")
    print("=" * 80)

    contrasts = [
        ("I2", "I0"), ("I3", "I0"), ("I4", "I0"),  # vs broken interface
        ("I2", "C0"), ("I3", "C0"), ("I4", "C0"),  # vs no guidance
        ("I1", "I0"),  # raw advantages vs normalized
    ]
    for a, b in contrasts:
        diffs = []
        for tid in task_ids:
            ra = by_task_arm.get((tid, a))
            rb = by_task_arm.get((tid, b))
            if ra and rb:
                diffs.append(ra["realized_utility"] - rb["realized_utility"])
        mean_diff = sum(diffs) / len(diffs) if diffs else 0
        ci_lo, ci_hi = bootstrap_ci_paired(diffs)
        excludes_zero = ci_lo > 0 or ci_hi < 0
        print(f"  ΔU({a:>3s} - {b:>3s}) = {mean_diff:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"{'EXCLUDES 0' if excludes_zero else 'includes 0'}")

    # ================================================================
    # 5. Repeated-action analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("5. REPEATED-ACTION ANALYSIS")
    print("=" * 80)

    for subtype in ["ol_retrieve", "ol_verify", "ol_search", "tl_retrieve", "tl_search"]:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        if not subtype_tasks:
            continue
        print(f"\n  {subtype}:")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in subtype_tasks if (tid, arm) in by_task_arm]
            if recs:
                max_consec = [r["max_consecutive_repeat"] for r in recs]
                retr = [r["retrieve_count"] for r in recs]
                consec_retr = []
                for r in recs:
                    # Count max consecutive RETRIEVEs
                    max_c = 0
                    cur = 0
                    for a in r["actions_taken"]:
                        if a == "RETRIEVE":
                            cur += 1
                            max_c = max(max_c, cur)
                        else:
                            cur = 0
                    consec_retr.append(max_c)
                print(f"    {arm:4s}: mean_max_consec_repeat={sum(max_consec)/len(max_consec):.2f} "
                      f"mean_consec_RETRIEVE={sum(consec_retr)/len(consec_retr):.2f} "
                      f"mean_total_RETRIEVE={sum(retr)/len(retr):.2f}")

    # ================================================================
    # 6. Resource exhaustion and STOP analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("6. RESOURCE EXHAUSTION AND STOP ANALYSIS")
    print("=" * 80)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        stops = sum(1 for r in recs if r["terminal_result"] == "STOP")
        step_limits = sum(1 for r in recs if r["terminal_result"] == "STEP_LIMIT")
        resource_exh = sum(1 for r in recs if r.get("resource_exhausted", False))
        print(f"  {arm:4s}: STOPs={stops} STEP_LIMITs={step_limits} resource_exhausted={resource_exh}")

    # ================================================================
    # 7. Action sequence patterns for ol_retrieve
    # ================================================================
    print("\n" + "=" * 80)
    print("7. ACTION SEQUENCE PATTERNS FOR ol_retrieve")
    print("=" * 80)

    ol_ret_tasks = [tid for tid in task_ids if get_subtype(tid) == "ol_retrieve"]
    for arm in arms:
        sequences = []
        for tid in ol_ret_tasks:
            r = by_task_arm.get((tid, arm))
            if r and r["actions_taken"]:
                sequences.append(tuple(r["actions_taken"]))
        seq_counts = Counter(sequences)
        print(f"\n  {arm} ({arm_names[arm]}):")
        for seq, count in seq_counts.most_common(3):
            print(f"    {list(seq)} (n={count})")

    # ================================================================
    # 8. Summary: which interfaces fix the over-retrieval?
    # ================================================================
    print("\n" + "=" * 80)
    print("8. SUMMARY: WHICH INTERFACES FIX THE OVER-RETRIEVAL?")
    print("=" * 80)

    print(f"\n  {'Arm':4s} {'Description':40s} {'ol_retr_U':>10s} {'ol_retr_retr':>12s} "
          f"{'ol_ver_U':>10s} {'overall_U':>10s} {'success':>8s}")
    for arm in arms:
        recs_all = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        recs_ret = [by_task_arm[(tid, arm)] for tid in ol_ret_tasks if (tid, arm) in by_task_arm]
        recs_ver = [by_task_arm[(tid, arm)] for tid in task_ids
                    if get_subtype(tid) == "ol_verify" and (tid, arm) in by_task_arm]

        overall_u = sum(r["realized_utility"] for r in recs_all) / len(recs_all)
        overall_sr = sum(1 for r in recs_all if r["success"]) / len(recs_all)
        ret_u = sum(r["realized_utility"] for r in recs_ret) / len(recs_ret) if recs_ret else 0
        ret_retr = sum(r["retrieve_count"] for r in recs_ret) / len(recs_ret) if recs_ret else 0
        ver_u = sum(r["realized_utility"] for r in recs_ver) / len(recs_ver) if recs_ver else 0

        print(f"  {arm:4s} {arm_names[arm]:40s} {ret_u:10.2f} {ret_retr:12.2f} "
              f"{ver_u:10.2f} {overall_u:10.2f} {overall_sr:8.4f}")

    # ================================================================
    # 9. Interface selection recommendation
    # ================================================================
    print("\n" + "=" * 80)
    print("9. INTERFACE SELECTION RECOMMENDATION")
    print("=" * 80)

    # Compute key metrics for selection
    c0_recs = [by_task_arm[(tid, "C0")] for tid in task_ids if (tid, arm) in by_task_arm]
    c0_u = sum(r["realized_utility"] for r in c0_recs) / len(c0_recs)
    c0_sr = sum(1 for r in c0_recs if r["success"]) / len(c0_recs)

    print(f"\n  Baseline (C0): mean_U={c0_u:.2f}, success={c0_sr:.4f}")
    print()

    best_arm = None
    best_u = -float("inf")
    for arm in ["I2", "I3", "I4"]:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        u = sum(r["realized_utility"] for r in recs) / len(recs)
        sr = sum(1 for r in recs if r["success"]) / len(recs)
        # Check if it matches C0 on ol_retrieve
        ret_recs = [by_task_arm[(tid, arm)] for tid in ol_ret_tasks if (tid, arm) in by_task_arm]
        ret_retr = sum(r["retrieve_count"] for r in ret_recs) / len(ret_recs) if ret_recs else 0
        print(f"  {arm}: mean_U={u:.2f}, success={sr:.4f}, ol_retrieve_E[retr]={ret_retr:.2f}")
        if u > best_u:
            best_u = u
            best_arm = arm

    print(f"\n  Recommended interface: {best_arm} ({arm_names[best_arm]})")

    # Save results
    results = {
        "n_trajectories": len(trajectories),
        "arms": arms,
        "arm_descriptions": arm_names,
        "best_arm": best_arm,
    }
    output_path = ablation_dir / "analysis_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
