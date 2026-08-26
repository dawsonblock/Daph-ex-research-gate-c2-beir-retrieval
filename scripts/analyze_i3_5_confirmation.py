#!/usr/bin/env python3
"""I3.5-PQ Phase 25: Confirmation analysis + promotion gates.

Runs automatically once 720/720 trajectories are closed.

Pre-registered primary comparisons:
  VP - C0: Does the progress executive beat the unguided model?
  VP - B0: Does the progress executive beat the strongest simple heuristic?

Mechanism comparison:
  VP - V1: Does Progress add value beyond Q+I2?

7 promotion gates:
  1. Delta_U(VP - C0) > 0, CI excludes 0
  2. Delta_U(VP - B0) > 0, CI excludes 0
  3. VP success >= C0 and VP success >= B0
  4. VP rescues more than it breaks vs B0 (McNemar preferred)
  5. VP premature DEFER <= C0 and <= B0
  6. VP premature ANSWER <= C0 and <= B0
  7. VP mean steps <= C0 and <= B0

Also performs:
  - Per-stratum analysis (15 subtypes)
  - Rescue reconstruction with mechanism traces
  - Resource exhaustion verification
  - Action sequence pattern comparison
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


def main():
    exp_dir = REPO_ROOT / "experiments/i3_5/confirmation"

    print("Loading trajectories...")
    trajectories = load_jsonl(exp_dir / "trajectories_v1.jsonl")
    errors = load_jsonl(exp_dir / "errors_v1.jsonl")
    print(f"  {len(trajectories)} trajectories, {len(errors)} errors")

    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    task_ids = sorted(set(r["task_id"] for r in trajectories))
    arms = ["C0", "B0", "V1", "VP"]

    for arm in arms:
        n = sum(1 for tid in task_ids if (tid, arm) in by_task_arm)
        print(f"  {arm}: {n}/{len(task_ids)} tasks")

    if any(sum(1 for tid in task_ids if (tid, arm) in by_task_arm) < len(task_ids) for arm in arms):
        print("\nWARNING: Experiment not complete. Results may be partial.")

    # ================================================================
    # 0. RESOURCE EXHAUSTION VERIFICATION
    # ================================================================
    print("\n" + "=" * 80)
    print("0. RESOURCE EXHAUSTION VERIFICATION")
    print("=" * 80)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        rex = sum(1 for r in recs if r.get("resource_exhaustion", False))
        rex_by_cat = defaultdict(int)
        for r in recs:
            if r.get("resource_exhaustion", False):
                rex_by_cat[r["category"]] += 1
        print(f"\n  {arm}: {rex}/{len(recs)} resource exhaustion cases")
        for cat, n in sorted(rex_by_cat.items()):
            print(f"    {cat}: {n}")

    # Check if exhaustion is concentrated in specific budget profiles
    print("\n  Exhaustion by budget profile:")
    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        rex_by_budget = defaultdict(int)
        total_by_budget = defaultdict(int)
        for r in recs:
            total_by_budget[r.get("budget_profile", "unknown")] += 1
            if r.get("resource_exhaustion", False):
                rex_by_budget[r.get("budget_profile", "unknown")] += 1
        print(f"\n    {arm}:")
        for bp in sorted(total_by_budget.keys()):
            print(f"      {bp}: {rex_by_budget[bp]}/{total_by_budget[bp]} exhausted")

    # ================================================================
    # 1. OVERALL RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("1. OVERALL RESULTS")
    print("=" * 80)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        if n == 0:
            continue
        us = [r["realized_utility"] for r in recs]
        sr = sum(1 for r in recs if r["success"]) / n
        retr = [r["retrieve_count"] for r in recs]
        verify = [r["verify_count"] for r in recs]
        search = [r["search_count"] for r in recs]
        steps = [r["steps"] for r in recs]
        max_retr = [max_consecutive_retrieves(r["actions_taken"]) for r in recs]
        pd = sum(1 for r in recs if r["premature_defer"])
        pa = sum(1 for r in recs if r["premature_answer"])
        rex = sum(1 for r in recs if r.get("resource_exhaustion", False))
        print(f"\n  {arm}:")
        print(f"    n={n} mean_U={sum(us)/n:.2f} success={sr:.4f}")
        print(f"    mean_retr={sum(retr)/n:.2f} mean_verify={sum(verify)/n:.2f} mean_search={sum(search)/n:.2f}")
        print(f"    mean_steps={sum(steps)/n:.2f} mean_max_consec_retr={sum(max_retr)/n:.2f}")
        print(f"    premature_defer={pd} premature_answer={pa} resource_exhaustion={rex}")

    # ================================================================
    # 2. PAIRED CONTRASTS
    # ================================================================
    print("\n" + "=" * 80)
    print("2. PAIRED CONTRASTS (95% CI)")
    print("=" * 80)

    contrasts = [
        ("VP", "C0"),
        ("VP", "B0"),
        ("VP", "V1"),
        ("V1", "C0"),
        ("V1", "B0"),
        ("B0", "C0"),
    ]

    contrast_results = {}
    for a, b in contrasts:
        diffs_u = []
        diffs_retr = []
        diffs_steps = []
        for tid in task_ids:
            ra = by_task_arm.get((tid, a))
            rb = by_task_arm.get((tid, b))
            if ra and rb:
                diffs_u.append(ra["realized_utility"] - rb["realized_utility"])
                diffs_retr.append(ra["retrieve_count"] - rb["retrieve_count"])
                diffs_steps.append(ra["steps"] - rb["steps"])

        mean_u = sum(diffs_u) / len(diffs_u) if diffs_u else 0
        ci_lo, ci_hi = bootstrap_ci_paired(diffs_u)
        excludes_zero = ci_lo > 0 or ci_hi < 0

        mean_retr = sum(diffs_retr) / len(diffs_retr) if diffs_retr else 0
        mean_steps = sum(diffs_steps) / len(diffs_steps) if diffs_steps else 0

        contrast_results[f"{a}-{b}"] = {
            "mean_delta_u": mean_u,
            "ci": [ci_lo, ci_hi],
            "excludes_zero": excludes_zero,
            "mean_delta_retr": mean_retr,
            "mean_delta_steps": mean_steps,
        }

        print(f"\n  ΔU({a} - {b}) = {mean_u:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"{'EXCLUDES 0' if excludes_zero else 'includes 0'}")
        print(f"  ΔRetr({a} - {b}) = {mean_retr:+.4f}")
        print(f"  ΔSteps({a} - {b}) = {mean_steps:+.4f}")

    # ================================================================
    # 3. SUCCESS RESCUES/BREAKS (McNemar)
    # ================================================================
    print("\n" + "=" * 80)
    print("3. SUCCESS RESCUES/BREAKS (McNemar exact)")
    print("=" * 80)

    for a, b in [("VP", "C0"), ("VP", "B0"), ("VP", "V1"), ("V1", "C0"), ("V1", "B0")]:
        rescues = 0  # a succeeds, b fails
        breaks = 0   # a fails, b succeeds
        for tid in task_ids:
            ra = by_task_arm.get((tid, a))
            rb = by_task_arm.get((tid, b))
            if ra and rb:
                if ra["success"] and not rb["success"]:
                    rescues += 1
                elif not ra["success"] and rb["success"]:
                    breaks += 1
        p = mcnemar_exact(rescues, breaks)
        print(f"  {a} vs {b}: rescues={rescues} breaks={breaks} p={p:.4f} "
              f"{'(significant)' if p < 0.05 else ''}")

    # ================================================================
    # 4. PER-STRATUM RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("4. PER-STRATUM RESULTS")
    print("=" * 80)

    categories = sorted(set(r["category"] for r in trajectories))
    for cat in categories:
        cat_tasks = [tid for tid in task_ids if by_task_arm.get((tid, "C0"), {}).get("category") == cat]
        print(f"\n  {cat} ({len(cat_tasks)} tasks):")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in cat_tasks if (tid, arm) in by_task_arm]
            if recs:
                us = [r["realized_utility"] for r in recs]
                sr = sum(1 for r in recs if r["success"]) / len(recs)
                retr = [r["retrieve_count"] for r in recs]
                steps = [r["steps"] for r in recs]
                rex = sum(1 for r in recs if r.get("resource_exhaustion", False))
                print(f"    {arm}: n={len(recs):2d} U={sum(us)/len(us):.2f} "
                      f"success={sr:.2f} retr={sum(retr)/len(retr):.2f} "
                      f"steps={sum(steps)/len(steps):.1f} exhaust={rex}")

    # ================================================================
    # 5. RESCUE RECONSTRUCTION WITH MECHANISM TRACES
    # ================================================================
    print("\n" + "=" * 80)
    print("5. RESCUE RECONSTRUCTION (VP rescues C0/V1 failures)")
    print("=" * 80)

    # VP rescues vs C0
    vp_rescues_c0 = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        c0 = by_task_arm.get((tid, "C0"))
        if vp and c0 and vp["success"] and not c0["success"]:
            vp_rescues_c0.append(tid)

    print(f"\n  VP rescues C0 failures: {len(vp_rescues_c0)}")
    for tid in vp_rescues_c0[:15]:
        vp = by_task_arm[(tid, "VP")]
        c0 = by_task_arm[(tid, "C0")]
        v1 = by_task_arm.get((tid, "V1"))
        print(f"\n    {tid} ({vp['category']}):")
        print(f"      C0: {c0['actions_taken']} (U={c0['realized_utility']:.1f}, "
              f"result={c0['terminal_result']}, exhaust={c0.get('resource_exhaustion', False)})")
        if v1:
            print(f"      V1: {v1['actions_taken']} (U={v1['realized_utility']:.1f}, "
                  f"success={v1['success']}, exhaust={v1.get('resource_exhaustion', False)})")
        print(f"      VP: {vp['actions_taken']} (U={vp['realized_utility']:.1f}, "
              f"success={vp['success']})")

        # Show progress log at first divergence from V1
        if v1 and vp.get("progress_log"):
            min_len = min(len(vp["actions_taken"]), len(v1["actions_taken"]))
            first_div = min_len
            for i in range(min_len):
                if vp["actions_taken"][i] != v1["actions_taken"][i]:
                    first_div = i
                    break
            if first_div < len(vp["progress_log"]):
                entry = vp["progress_log"][first_div]
                print(f"      First divergence at step {first_div}:")
                print(f"        Q: {entry['q_values']}")
                print(f"        Near-optimal (before progress): {entry['near_optimal_before_progress']}")
                print(f"        Near-optimal (after progress): {entry['near_optimal_after_progress']}")
                print(f"        Progress scores: {entry['progress_scores']}")
                print(f"        Confidence: {entry['confidence']}")

    # VP rescues vs V1
    vp_rescues_v1 = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        v1 = by_task_arm.get((tid, "V1"))
        if vp and v1 and vp["success"] and not v1["success"]:
            vp_rescues_v1.append(tid)

    print(f"\n\n  VP rescues V1 failures: {len(vp_rescues_v1)}")
    for tid in vp_rescues_v1[:15]:
        vp = by_task_arm[(tid, "VP")]
        v1 = by_task_arm[(tid, "V1")]
        c0 = by_task_arm.get((tid, "C0"))
        print(f"\n    {tid} ({vp['category']}):")
        if c0:
            print(f"      C0: {c0['actions_taken']} (U={c0['realized_utility']:.1f}, "
                  f"success={c0['success']})")
        print(f"      V1: {v1['actions_taken']} (U={v1['realized_utility']:.1f}, "
              f"result={v1['terminal_result']}, exhaust={v1.get('resource_exhaustion', False)})")
        print(f"      VP: {vp['actions_taken']} (U={vp['realized_utility']:.1f}, "
              f"success={vp['success']})")

        if vp.get("progress_log"):
            min_len = min(len(vp["actions_taken"]), len(v1["actions_taken"]))
            first_div = min_len
            for i in range(min_len):
                if vp["actions_taken"][i] != v1["actions_taken"][i]:
                    first_div = i
                    break
            if first_div < len(vp["progress_log"]):
                entry = vp["progress_log"][first_div]
                print(f"      First divergence at step {first_div}:")
                print(f"        Q: {entry['q_values']}")
                print(f"        Near-optimal (before progress): {entry['near_optimal_before_progress']}")
                print(f"        Near-optimal (after progress): {entry['near_optimal_after_progress']}")
                print(f"        Progress scores: {entry['progress_scores']}")

    # VP breaks (VP fails where others succeed)
    vp_breaks_c0 = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        c0 = by_task_arm.get((tid, "C0"))
        if vp and c0 and not vp["success"] and c0["success"]:
            vp_breaks_c0.append(tid)

    vp_breaks_v1 = []
    for tid in task_ids:
        vp = by_task_arm.get((tid, "VP"))
        v1 = by_task_arm.get((tid, "V1"))
        if vp and v1 and not vp["success"] and v1["success"]:
            vp_breaks_v1.append(tid)

    print(f"\n\n  VP breaks C0 successes: {len(vp_breaks_c0)}")
    for tid in vp_breaks_c0[:5]:
        vp = by_task_arm[(tid, "VP")]
        c0 = by_task_arm[(tid, "C0")]
        print(f"    {tid}: C0={c0['actions_taken']} VP={vp['actions_taken']} "
              f"(VP result={vp['terminal_result']})")

    print(f"\n  VP breaks V1 successes: {len(vp_breaks_v1)}")
    for tid in vp_breaks_v1[:5]:
        vp = by_task_arm[(tid, "VP")]
        v1 = by_task_arm[(tid, "V1")]
        print(f"    {tid}: V1={v1['actions_taken']} VP={vp['actions_taken']} "
              f"(VP result={vp['terminal_result']})")

    # ================================================================
    # 6. ACTION SEQUENCE PATTERNS
    # ================================================================
    print("\n" + "=" * 80)
    print("6. ACTION SEQUENCE PATTERNS (top 3 per stratum)")
    print("=" * 80)

    for cat in categories:
        cat_tasks = [tid for tid in task_ids if by_task_arm.get((tid, "C0"), {}).get("category") == cat]
        if not cat_tasks:
            continue
        print(f"\n  {cat}:")
        for arm in arms:
            sequences = []
            for tid in cat_tasks:
                r = by_task_arm.get((tid, arm))
                if r and r["actions_taken"]:
                    sequences.append(tuple(r["actions_taken"]))
            if sequences:
                seq_counts = Counter(sequences)
                print(f"    {arm} (top 3):")
                for seq, count in seq_counts.most_common(3):
                    print(f"      {list(seq)} (n={count})")

    # ================================================================
    # 7. PROMOTION GATES
    # ================================================================
    print("\n" + "=" * 80)
    print("7. PROMOTION GATES (pre-registered)")
    print("=" * 80)

    vp_recs = [by_task_arm[(tid, "VP")] for tid in task_ids if (tid, "VP") in by_task_arm]
    c0_recs = [by_task_arm[(tid, "C0")] for tid in task_ids if (tid, "C0") in by_task_arm]
    b0_recs = [by_task_arm[(tid, "B0")] for tid in task_ids if (tid, "B0") in by_task_arm]
    v1_recs = [by_task_arm[(tid, "V1")] for tid in task_ids if (tid, "V1") in by_task_arm]

    if not all([vp_recs, c0_recs, b0_recs, v1_recs]):
        print("  Cannot evaluate — missing data")
        return

    vp_success = sum(1 for r in vp_recs if r["success"]) / len(vp_recs)
    c0_success = sum(1 for r in c0_recs if r["success"]) / len(c0_recs)
    b0_success = sum(1 for r in b0_recs if r["success"]) / len(b0_recs)
    v1_success = sum(1 for r in v1_recs if r["success"]) / len(v1_recs)

    vp_pd = sum(1 for r in vp_recs if r["premature_defer"])
    c0_pd = sum(1 for r in c0_recs if r["premature_defer"])
    b0_pd = sum(1 for r in b0_recs if r["premature_defer"])

    vp_pa = sum(1 for r in vp_recs if r["premature_answer"])
    c0_pa = sum(1 for r in c0_recs if r["premature_answer"])
    b0_pa = sum(1 for r in b0_recs if r["premature_answer"])

    vp_steps = sum(r["steps"] for r in vp_recs) / len(vp_recs)
    c0_steps = sum(r["steps"] for r in c0_recs) / len(c0_recs)
    b0_steps = sum(r["steps"] for r in b0_recs) / len(b0_recs)

    # Gate 1: Delta_U(VP - C0) > 0, CI excludes 0
    vp_c0 = contrast_results.get("VP-C0", {})
    gate1 = vp_c0.get("excludes_zero", False) and vp_c0.get("mean_delta_u", 0) > 0
    print(f"\n  Gate 1: ΔU(VP - C0) > 0, CI excludes 0")
    print(f"    ΔU={vp_c0.get('mean_delta_u', 0):+.4f} CI={vp_c0.get('ci', [0,0])}")
    print(f"    -> {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: Delta_U(VP - B0) > 0, CI excludes 0
    vp_b0 = contrast_results.get("VP-B0", {})
    gate2 = vp_b0.get("excludes_zero", False) and vp_b0.get("mean_delta_u", 0) > 0
    print(f"\n  Gate 2: ΔU(VP - B0) > 0, CI excludes 0")
    print(f"    ΔU={vp_b0.get('mean_delta_u', 0):+.4f} CI={vp_b0.get('ci', [0,0])}")
    print(f"    -> {'PASS' if gate2 else 'FAIL'}")

    # Gate 3: VP success >= C0 and VP success >= B0
    gate3 = vp_success >= c0_success and vp_success >= b0_success
    print(f"\n  Gate 3: VP success >= C0 and >= B0")
    print(f"    VP={vp_success:.4f} C0={c0_success:.4f} B0={b0_success:.4f}")
    print(f"    -> {'PASS' if gate3 else 'FAIL'}")

    # Gate 4: VP rescues more than it breaks vs B0 (McNemar preferred)
    vp_b0_rescues = sum(1 for tid in task_ids
                        if by_task_arm.get((tid, "VP"), {}).get("success")
                        and not by_task_arm.get((tid, "B0"), {}).get("success"))
    vp_b0_breaks = sum(1 for tid in task_ids
                       if not by_task_arm.get((tid, "VP"), {}).get("success")
                       and by_task_arm.get((tid, "B0"), {}).get("success"))
    mc_p = mcnemar_exact(vp_b0_rescues, vp_b0_breaks)
    gate4 = vp_b0_rescues > vp_b0_breaks and mc_p < 0.05
    gate4_weak = vp_b0_rescues > vp_b0_breaks
    print(f"\n  Gate 4: VP rescues > breaks vs B0 (McNemar p < 0.05 preferred)")
    print(f"    rescues={vp_b0_rescues} breaks={vp_b0_breaks} p={mc_p:.4f}")
    print(f"    -> {'PASS' if gate4 else 'WEAK PASS' if gate4_weak else 'FAIL'}")

    # Gate 5: VP premature DEFER <= C0 and <= B0
    gate5 = vp_pd <= c0_pd and vp_pd <= b0_pd
    print(f"\n  Gate 5: VP premature DEFER <= C0 and <= B0")
    print(f"    VP={vp_pd} C0={c0_pd} B0={b0_pd}")
    print(f"    -> {'PASS' if gate5 else 'FAIL'}")

    # Gate 6: VP premature ANSWER <= C0 and <= B0
    gate6 = vp_pa <= c0_pa and vp_pa <= b0_pa
    print(f"\n  Gate 6: VP premature ANSWER <= C0 and <= B0")
    print(f"    VP={vp_pa} C0={c0_pa} B0={b0_pa}")
    print(f"    -> {'PASS' if gate6 else 'FAIL'}")

    # Gate 7: VP mean steps <= C0 and <= B0
    gate7 = vp_steps <= c0_steps and vp_steps <= b0_steps
    print(f"\n  Gate 7: VP mean steps <= C0 and <= B0")
    print(f"    VP={vp_steps:.2f} C0={c0_steps:.2f} B0={b0_steps:.2f}")
    print(f"    -> {'PASS' if gate7 else 'FAIL'}")

    # ================================================================
    # 8. OVERALL VERDICT
    # ================================================================
    print("\n" + "=" * 80)
    print("8. OVERALL VERDICT")
    print("=" * 80)

    all_gates = {
        "gate_1_utility_vs_c0": gate1,
        "gate_2_utility_vs_b0": gate2,
        "gate_3_success_noninferiority": gate3,
        "gate_4_rescue_advantage": gate4,
        "gate_5_no_pathological_defer": gate5,
        "gate_6_no_premature_answer": gate6,
        "gate_7_resource_efficiency": gate7,
    }

    n_pass = sum(all_gates.values())
    print(f"\n  Gates passed: {n_pass}/{len(all_gates)}")
    for gate, passed in all_gates.items():
        print(f"    {gate}: {'PASS' if passed else 'FAIL'}")

    if n_pass == len(all_gates):
        verdict = "CONFIRMED"
        print(f"\n  VERDICT: CONFIRMED — DAPH_PROGRESS_EXECUTIVE_V1 passes confirmation")
        print(f"  VP beats both C0 and B0 on a hostile unseen benchmark.")
        print(f"  The Progress layer changes terminal outcomes, not just action costs.")
        print(f"  Proceed to cross-model replication.")
    elif gate1 and gate2 and gate3:
        verdict = "CONDITIONAL_CONFIRMATION"
        print(f"\n  VERDICT: CONDITIONAL CONFIRMATION")
        print(f"  VP beats C0 and B0 in utility with non-inferior success.")
        print(f"  Some secondary gates failed. Review mechanism before full promotion.")
    else:
        verdict = "FALSIFIED"
        print(f"\n  VERDICT: FALSIFIED")
        print(f"  VP does not meet the pre-registered promotion bar.")
        print(f"  Keep DAPH_EXECUTIVE_V1 (V1 + I2) as the executive.")
        print(f"  Do not tune on this benchmark. Investigate the failure off-benchmark.")

    # Save results
    results = {
        "n_trajectories": len(trajectories),
        "verdict": verdict,
        "gates": all_gates,
        "n_gates_passed": n_pass,
        "n_gates_total": len(all_gates),
        "overall": {
            "C0": {"mean_u": sum(r["realized_utility"] for r in c0_recs)/len(c0_recs),
                   "success": c0_success},
            "B0": {"mean_u": sum(r["realized_utility"] for r in b0_recs)/len(b0_recs),
                   "success": b0_success},
            "V1": {"mean_u": sum(r["realized_utility"] for r in v1_recs)/len(v1_recs),
                   "success": v1_success},
            "VP": {"mean_u": sum(r["realized_utility"] for r in vp_recs)/len(vp_recs),
                   "success": vp_success},
        },
        "contrasts": contrast_results,
        "rescues": {
            "vp_rescues_c0": len(vp_rescues_c0),
            "vp_breaks_c0": len(vp_breaks_c0),
            "vp_rescues_v1": len(vp_rescues_v1),
            "vp_breaks_v1": len(vp_breaks_v1),
            "vp_rescues_b0": vp_b0_rescues,
            "vp_breaks_b0": vp_b0_breaks,
            "mcnemar_vp_vs_b0_p": mc_p,
        },
    }
    output_path = exp_dir / "analysis_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
