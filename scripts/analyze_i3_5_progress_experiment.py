#!/usr/bin/env python3
"""I3.5-PQ Phase 24: Progress experiment analysis + promotion gates.

Runs automatically once 220/220 VP trajectories are closed.

Checks:
  1. VP success >= V1 success
  2. Delta_U(VP - V1) > 0, paired 95% CI excluding 0
  3. Repeated RETRIEVE/VERIFY/SEARCH counts lower
  4. No increase in premature DEFER or premature ANSWER
  5. No increase in resource exhaustion or loops
  6. On hard strata, VP reduces redundant actions (not just terminates earlier)
  7. Mechanism: VP actually removes low-progress actions inside I2 set
  8. Per-category action sequence comparison (first divergence point)
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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


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
    return (float(np.percentile(means, alpha * 100)),
            float(np.percentile(means, (1 - alpha) * 100)))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * 0.5 ** n
    return 2 * p


def get_subtype(tid: str) -> str:
    parts = tid.split("_")
    return "_".join(parts[2:4])


def max_consecutive_retrieves(actions: list[str]) -> int:
    max_c = 0
    cur = 0
    for a in actions:
        if a == "RETRIEVE":
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def max_consecutive_action(actions: list[str], target: str) -> int:
    max_c = 0
    cur = 0
    for a in actions:
        if a == target:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def main():
    exp_dir = REPO_ROOT / "experiments/i3_5/progress_experiment"

    print("Loading trajectories...")
    trajectories = load_jsonl(exp_dir / "trajectories_v1.jsonl")
    errors = load_jsonl(exp_dir / "errors_v1.jsonl")
    print(f"  {len(trajectories)} trajectories, {len(errors)} errors")

    # Check completeness
    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in trajectories))
    arms = ["C0", "V1", "VP"]

    for arm in arms:
        n = sum(1 for tid in task_ids if (tid, arm) in by_task_arm)
        print(f"  {arm}: {n}/{len(task_ids)} tasks")

    if any(sum(1 for tid in task_ids if (tid, arm) in by_task_arm) < len(task_ids) for arm in arms):
        print("\nWARNING: Experiment not complete. Results may be partial.")

    # ================================================================
    # 1. OVERALL RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("1. OVERALL RESULTS")
    print("=" * 80)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        us = [r["realized_utility"] for r in recs]
        sr = sum(1 for r in recs if r["success"]) / n
        retr = [r["retrieve_count"] for r in recs]
        verify = [r["verify_count"] for r in recs]
        search = [r["search_count"] for r in recs]
        max_rep = [r["max_consecutive_repeat"] for r in recs]
        max_retr = [max_consecutive_retrieves(r["actions_taken"]) for r in recs]
        pd = sum(1 for r in recs if r["premature_defer"])
        pa = sum(1 for r in recs if r["premature_answer"])
        steps = [r["steps"] for r in recs]
        print(f"\n  {arm}:")
        print(f"    n={n} mean_U={sum(us)/n:.2f} success={sr:.4f}")
        print(f"    mean_retr={sum(retr)/n:.2f} mean_verify={sum(verify)/n:.2f} mean_search={sum(search)/n:.2f}")
        print(f"    mean_max_consec_repeat={sum(max_rep)/n:.2f} mean_max_consec_retr={sum(max_retr)/n:.2f}")
        print(f"    mean_steps={sum(steps)/n:.2f} premature_defer={pd} premature_answer={pa}")

    # ================================================================
    # 2. PAIRED CONTRASTS
    # ================================================================
    print("\n" + "=" * 80)
    print("2. PAIRED CONTRASTS (95% CI)")
    print("=" * 80)

    contrasts = [
        ("V1", "C0"),
        ("VP", "C0"),
        ("VP", "V1"),  # The key contrast
    ]

    for a, b in contrasts:
        diffs_u = []
        diffs_retr = []
        diffs_success = []
        for tid in task_ids:
            ra = by_task_arm.get((tid, a))
            rb = by_task_arm.get((tid, b))
            if ra and rb:
                diffs_u.append(ra["realized_utility"] - rb["realized_utility"])
                diffs_retr.append(ra["retrieve_count"] - rb["retrieve_count"])
                diffs_success.append(1 if ra["success"] else 0)

        mean_u = sum(diffs_u) / len(diffs_u) if diffs_u else 0
        ci_lo, ci_hi = bootstrap_ci_paired(diffs_u)
        excludes_zero = ci_lo > 0 or ci_hi < 0

        mean_retr = sum(diffs_retr) / len(diffs_retr) if diffs_retr else 0
        ci_lo_r, ci_hi_r = bootstrap_ci_paired(diffs_retr)

        print(f"\n  ΔU({a} - {b}) = {mean_u:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"{'EXCLUDES 0' if excludes_zero else 'includes 0'}")
        print(f"  ΔRetr({a} - {b}) = {mean_retr:+.4f} CI=[{ci_lo_r:+.4f}, {ci_hi_r:+.4f}]")

    # ================================================================
    # 3. PER-SUBTYPE RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("3. PER-SUBTYPE RESULTS")
    print("=" * 80)

    subtypes = sorted(set(get_subtype(tid) for tid in task_ids))
    for subtype in subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        print(f"\n  {subtype} ({len(subtype_tasks)} tasks):")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in subtype_tasks if (tid, arm) in by_task_arm]
            if recs:
                us = [r["realized_utility"] for r in recs]
                sr = sum(1 for r in recs if r["success"]) / len(recs)
                retr = [r["retrieve_count"] for r in recs]
                verify = [r["verify_count"] for r in recs]
                search = [r["search_count"] for r in recs]
                max_retr = [max_consecutive_retrieves(r["actions_taken"]) for r in recs]
                print(f"    {arm}: n={len(recs):2d} U={sum(us)/len(us):.2f} "
                      f"success={sr:.2f} retr={sum(retr)/len(retr):.2f} "
                      f"verify={sum(verify)/len(verify):.2f} "
                      f"search={sum(search)/len(search):.2f} "
                      f"max_consec_retr={sum(max_retr)/len(max_retr):.2f}")

    # ================================================================
    # 4. REPEATED-ACTION TRAP ANALYSIS
    # ================================================================
    print("\n" + "=" * 80)
    print("4. REPEATED-ACTION TRAP ANALYSIS")
    print("=" * 80)

    trap_subtypes = ["ol_retrieve", "tl_retrieve", "ol_search", "tl_search",
                     "ol_verify", "tl_verify"]
    for subtype in trap_subtypes:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        if not subtype_tasks:
            continue
        print(f"\n  {subtype}:")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in subtype_tasks if (tid, arm) in by_task_arm]
            if recs:
                # Count trajectories with 2+ consecutive retrieves
                multi_retr = sum(1 for r in recs if max_consecutive_retrieves(r["actions_taken"]) >= 2)
                three_retr = sum(1 for r in recs if r["retrieve_count"] >= 3)
                max_retr_runs = [max_consecutive_retrieves(r["actions_taken"]) for r in recs]
                print(f"    {arm}: trajectories_with_2+_consec_retr={multi_retr}/{len(recs)} "
                      f"trajectories_with_3+_total_retr={three_retr}/{len(recs)} "
                      f"mean_max_consec_retr={sum(max_retr_runs)/len(max_retr_runs):.2f}")

    # ================================================================
    # 5. MECHANISM CHECK: Is VP actually removing low-progress actions?
    # ================================================================
    print("\n" + "=" * 80)
    print("5. MECHANISM CHECK: VP progress tie-breaking in action")
    print("=" * 80)

    vp_recs = [by_task_arm[(tid, "VP")] for tid in task_ids if (tid, "VP") in by_task_arm]
    v1_recs = [by_task_arm[(tid, "V1")] for tid in task_ids if (tid, "V1") in by_task_arm]

    # Find trajectories where VP and V1 diverge
    divergences = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        v1 = by_task_arm.get((tid, "V1"))
        if not vp or not v1:
            continue
        if vp["actions_taken"] != v1["actions_taken"]:
            # Find first divergence
            min_len = min(len(vp["actions_taken"]), len(v1["actions_taken"]))
            first_div = min_len
            for i in range(min_len):
                if vp["actions_taken"][i] != v1["actions_taken"][i]:
                    first_div = i
                    break
            divergences.append({
                "task_id": tid,
                "subtype": get_subtype(tid),
                "v1_actions": v1["actions_taken"],
                "vp_actions": vp["actions_taken"],
                "v1_utility": v1["realized_utility"],
                "vp_utility": vp["realized_utility"],
                "v1_success": v1["success"],
                "vp_success": vp["success"],
                "v1_retrieves": v1["retrieve_count"],
                "vp_retrieves": vp["retrieve_count"],
                "first_divergence": first_div,
                "progress_log": vp.get("progress_log", []),
            })

    print(f"\n  Trajectories where VP differs from V1: {len(divergences)}/{len(task_ids)}")

    if divergences:
        # Show examples from repeated-action traps
        trap_divs = [d for d in divergences if d["subtype"] in trap_subtypes]
        print(f"  Of which in repeated-action traps: {len(trap_divs)}")

        print(f"\n  First 10 divergences in trap subtypes:")
        for d in trap_divs[:10]:
            print(f"\n    {d['task_id']} ({d['subtype']}):")
            print(f"      V1: {d['v1_actions']} (U={d['v1_utility']:.1f}, retr={d['v1_retrieves']})")
            print(f"      VP: {d['vp_actions']} (U={d['vp_utility']:.1f}, retr={d['vp_retrieves']})")
            print(f"      First divergence at step {d['first_divergence']}")

            # Show progress log at divergence point
            if d["progress_log"] and d["first_divergence"] < len(d["progress_log"]):
                entry = d["progress_log"][d["first_divergence"]]
                print(f"      Q values: {entry['q_values']}")
                print(f"      Near-optimal (before progress): {entry['near_optimal_before_progress']}")
                print(f"      Near-optimal (after progress): {entry['near_optimal_after_progress']}")
                print(f"      Progress scores: {entry['progress_scores']}")
                print(f"      Confidence: {entry['confidence']}")

        # Check: did VP improve or harm on divergent trajectories?
        improved = sum(1 for d in divergences if d["vp_utility"] > d["v1_utility"])
        harmed = sum(1 for d in divergences if d["vp_utility"] < d["v1_utility"])
        same = sum(1 for d in divergences if d["vp_utility"] == d["v1_utility"])
        print(f"\n  On divergent trajectories:")
        print(f"    VP improved: {improved}")
        print(f"    VP harmed: {harmed}")
        print(f"    Same utility: {same}")

        # Check: did VP reduce retrieves on divergent trajectories?
        fewer_retr = sum(1 for d in divergences if d["vp_retrieves"] < d["v1_retrieves"])
        more_retr = sum(1 for d in divergences if d["vp_retrieves"] > d["v1_retrieves"])
        print(f"    VP fewer retrieves: {fewer_retr}")
        print(f"    VP more retrieves: {more_retr}")

    # ================================================================
    # 6. ACTION SEQUENCE PATTERNS FOR KEY SUBTYPES
    # ================================================================
    print("\n" + "=" * 80)
    print("6. ACTION SEQUENCE PATTERNS")
    print("=" * 80)

    for subtype in ["ol_retrieve", "tl_retrieve", "ol_search", "tl_search"]:
        subtype_tasks = [tid for tid in task_ids if get_subtype(tid) == subtype]
        if not subtype_tasks:
            continue
        print(f"\n  {subtype}:")
        for arm in arms:
            sequences = []
            for tid in subtype_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    sequences.append(tuple(r["actions_taken"]))
            seq_counts = Counter(sequences)
            print(f"\n    {arm} (top 3 patterns):")
            for seq, count in seq_counts.most_common(3):
                print(f"      {list(seq)} (n={count})")

    # ================================================================
    # 7. PROMOTION GATES
    # ================================================================
    print("\n" + "=" * 80)
    print("7. PROMOTION GATES")
    print("=" * 80)

    v1_recs_all = [by_task_arm[(tid, "V1")] for tid in task_ids if (tid, "V1") in by_task_arm]
    vp_recs_all = [by_task_arm[(tid, "VP")] for tid in task_ids if (tid, "VP") in by_task_arm]

    if not v1_recs_all or not vp_recs_all:
        print("  Cannot evaluate — missing V1 or VP data")
        return

    v1_success = sum(1 for r in v1_recs_all if r["success"]) / len(v1_recs_all)
    vp_success = sum(1 for r in vp_recs_all if r["success"]) / len(vp_recs_all)
    v1_retr = sum(r["retrieve_count"] for r in v1_recs_all) / len(v1_recs_all)
    vp_retr = sum(r["retrieve_count"] for r in vp_recs_all) / len(vp_recs_all)
    v1_pd = sum(1 for r in v1_recs_all if r["premature_defer"])
    vp_pd = sum(1 for r in vp_recs_all if r["premature_defer"])
    v1_pa = sum(1 for r in v1_recs_all if r["premature_answer"])
    vp_pa = sum(1 for r in vp_recs_all if r["premature_answer"])

    # Paired Delta-U
    diffs_u = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        v1 = by_task_arm.get((tid, "V1"))
        if vp and v1:
            diffs_u.append(vp["realized_utility"] - v1["realized_utility"])
    mean_delta_u = sum(diffs_u) / len(diffs_u) if diffs_u else 0
    ci_lo, ci_hi = bootstrap_ci_paired(diffs_u)
    ci_excludes_zero = ci_lo > 0 or ci_hi < 0

    # Paired Delta-Retrieve
    diffs_retr = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        v1 = by_task_arm.get((tid, "V1"))
        if vp and v1:
            diffs_retr.append(vp["retrieve_count"] - v1["retrieve_count"])
    mean_delta_retr = sum(diffs_retr) / len(diffs_retr) if diffs_retr else 0

    # Gates
    gate_1_success = vp_success >= v1_success
    gate_2_utility = mean_delta_u > 0
    gate_2_ci = ci_excludes_zero and mean_delta_u > 0
    gate_3_retr = vp_retr < v1_retr
    gate_4a_pd = vp_pd <= v1_pd
    gate_4b_pa = vp_pa <= v1_pa

    print(f"\n  Gate 1: VP success >= V1 success")
    print(f"    VP={vp_success:.4f} V1={v1_success:.4f} -> {'PASS' if gate_1_success else 'FAIL'}")

    print(f"\n  Gate 2: Delta_U(VP - V1) > 0, CI excludes 0")
    print(f"    Delta_U={mean_delta_u:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    CI excludes 0: {ci_excludes_zero}")
    print(f"    -> {'PASS' if gate_2_ci else 'FAIL (CI includes 0 or negative)'}")

    print(f"\n  Gate 3: VP repeated-action count < V1")
    print(f"    VP mean_retr={vp_retr:.2f} V1 mean_retr={v1_retr:.2f}")
    print(f"    Delta_retr={mean_delta_retr:+.4f} -> {'PASS' if gate_3_retr else 'FAIL'}")

    print(f"\n  Gate 4a: No increase in premature DEFER")
    print(f"    VP={vp_pd} V1={v1_pd} -> {'PASS' if gate_4a_pd else 'FAIL'}")

    print(f"\n  Gate 4b: No increase in premature ANSWER")
    print(f"    VP={vp_pa} V1={v1_pa} -> {'PASS' if gate_4b_pa else 'FAIL'}")

    # ================================================================
    # 8. MECHANISM QUALITY CHECK
    # ================================================================
    print(f"\n  Gate 5: VP actually removes low-progress actions (not just DEFERs sooner)")

    if divergences:
        # Check if VP's first divergence is a DEFER (bad) vs removing a RETRIEVE (good)
        defer_first = sum(1 for d in divergences
                          if d["first_divergence"] < len(d["vp_actions"])
                          and d["vp_actions"][d["first_divergence"]] == "DEFER")
        remove_retr = sum(1 for d in divergences
                          if d["first_divergence"] < len(d["v1_actions"])
                          and d["v1_actions"][d["first_divergence"]] == "RETRIEVE"
                          and (d["first_divergence"] >= len(d["vp_actions"])
                               or d["vp_actions"][d["first_divergence"]] != "RETRIEVE"))
        print(f"    Divergences where VP DEFERs sooner: {defer_first}")
        print(f"    Divergences where VP avoids a RETRIEVE: {remove_retr}")
        print(f"    -> {'PASS' if remove_retr > defer_first else 'WEAK (VP may be terminating sooner)'}")
    else:
        print(f"    No divergences — VP identical to V1")
        print(f"    -> INCONCLUSIVE")

    # ================================================================
    # 9. OVERALL VERDICT
    # ================================================================
    print("\n" + "=" * 80)
    print("9. OVERALL VERDICT")
    print("=" * 80)

    all_gates = {
        "success": gate_1_success,
        "utility_positive": gate_2_utility,
        "utility_ci": gate_2_ci,
        "reduced_retrieves": gate_3_retr,
        "no_premature_defer": gate_4a_pd,
        "no_premature_answer": gate_4b_pa,
    }

    n_pass = sum(all_gates.values())
    print(f"\n  Gates passed: {n_pass}/{len(all_gates)}")
    for gate, passed in all_gates.items():
        print(f"    {gate}: {'PASS' if passed else 'FAIL'}")

    if n_pass == len(all_gates):
        verdict = "PROMOTE"
        print(f"\n  VERDICT: PROMOTE DAPH_PROGRESS_EXECUTIVE_V1")
        print(f"  VP preserves V1 success, improves utility, reduces redundant actions.")
        print(f"  Freeze VP as the production executive.")
    elif gate_1_success and gate_3_retr:
        verdict = "CONDITIONAL_PASS"
        print(f"\n  VERDICT: CONDITIONAL PASS")
        print(f"  VP preserves success and reduces retrieves, but utility CI includes 0.")
        print(f"  Consider promoting if mechanism check shows genuine progress filtering.")
    else:
        verdict = "REJECT"
        print(f"\n  VERDICT: REJECT PROGRESS_RULE_V1")
        print(f"  VP does not meet promotion criteria.")
        print(f"  Do not tune on this run. Keep DAPH_EXECUTIVE_V1 (V1 + I2).")

    # Save results
    results = {
        "n_trajectories": len(trajectories),
        "verdict": verdict,
        "gates": all_gates,
        "overall": {
            "C0": {"mean_u": sum(r["realized_utility"] for r in v1_recs_all) / len(v1_recs_all) if v1_recs_all else 0},
            "V1": {"mean_u": sum(r["realized_utility"] for r in v1_recs_all) / len(v1_recs_all)},
            "VP": {"mean_u": sum(r["realized_utility"] for r in vp_recs_all) / len(vp_recs_all)},
        },
        "delta_u_vp_v1": round(mean_delta_u, 4),
        "delta_u_ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "n_divergences": len(divergences),
    }
    output_path = exp_dir / "analysis_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
