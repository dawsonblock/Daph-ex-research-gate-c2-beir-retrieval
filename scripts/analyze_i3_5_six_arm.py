#!/usr/bin/env python3
"""I3.5-PQ Phase 19: Six-arm experiment analysis.

Computes all preregistered endpoints:
  - Success rate per arm
  - Paired rescues/breaks with McNemar exact tests
  - Premature DEFER/ANSWER rates
  - Loop rate
  - Resource exhaustion
  - Mean steps/tokens
  - Mean causal regret of chosen action
  - Near-optimal selected-action rate at epsilon=3
  - Stratified by gap bucket (clear >10, moderate 3-10, near_tie <=3)
  - Delta U contrasts with paired 95% CIs

Primary contrasts:
  - QCAUSAL > B0
  - QCAUSAL > QOBS
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
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


def load_model_calls(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_pinned_causal(path: Path) -> dict[str, dict[str, float]]:
    """Load pinned-policy causal data to get Q values for regret computation."""
    by_cp_action = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Use checkpoint_id + category to identify the state
            # Actually, we need to match by task_id + state
            # The causal data has checkpoint_id which is derived from the task
            # For regret, we need Q(s,a) for each action at each state
            key = r["checkpoint_id"]
            by_cp_action[key][r["forced_action"]] = r["pinned_policy_utility"]
    return dict(by_cp_action)


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar exact test (binomial). b = discordant pairs where arm1 succeeds, arm2 fails.
    c = discordant pairs where arm1 fails, arm2 succeeds."""
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided exact binomial test
    k = min(b, c)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * 0.5 ** n
    return 2 * p  # two-sided


def bootstrap_ci_paired(differences: list[float], n_bootstrap: int = 10000,
                         confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI for mean of paired differences."""
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


def main():
    six_arm_dir = REPO_ROOT / "experiments/i3_5/six_arm"
    pinned_dir = REPO_ROOT / "experiments/i3_5/pinned_policy"

    # Load data
    print("Loading data...")
    trajectories = load_trajectories(six_arm_dir / "trajectories_v1.jsonl")
    model_calls = load_model_calls(six_arm_dir / "model_calls_v1.jsonl")
    print(f"  {len(trajectories)} trajectories")
    print(f"  {len(model_calls)} model calls")

    # Load pinned-policy causal data for regret computation
    # We need to map task_id -> {action: Q_value}
    # The causal data has checkpoint_id, which corresponds to the initial state of a task
    causal_records = []
    with open(pinned_dir / "pinned_causal_actions_v1.jsonl") as f:
        for line in f:
            causal_records.append(json.loads(line))

    # Build task_id -> {action: Q_value} mapping
    # The causal data uses checkpoint_id which is derived from the task
    # We need to match by the state features at step 0
    # For now, use the category to match
    by_category_action = defaultdict(list)
    for r in causal_records:
        by_category_action[(r["category"], r["forced_action"])].append(r["pinned_policy_utility"])

    # Build category -> {action: mean_Q} mapping
    category_q = {}
    for (cat, action), values in by_category_action.items():
        if cat not in category_q:
            category_q[cat] = {}
        category_q[cat][action] = sum(values) / len(values)

    # Build task_id -> category mapping from trajectories
    task_to_category = {}
    for r in trajectories:
        tid = r["task_id"]
        # Extract category from task_id: i3_5_ol_retrieve_0000 -> ol_retrieve
        parts = tid.split("_")
        cat = "_".join(parts[2:4])
        task_to_category[tid] = cat

    # Load gap data for stratification
    # We need to compute gaps per checkpoint
    by_cp = defaultdict(dict)
    for r in causal_records:
        by_cp[r["checkpoint_id"]][r["forced_action"]] = r["pinned_policy_utility"]

    # Compute gaps and map to categories
    category_gaps = defaultdict(list)
    for cp_id, action_qs in by_cp.items():
        qs = sorted(action_qs.values(), reverse=True)
        if len(qs) >= 2:
            gap = qs[0] - qs[1]
        else:
            gap = 0
        # Find the category for this checkpoint
        # We need to match checkpoint_id to category
        # For now, use the first record's category
        for r in causal_records:
            if r["checkpoint_id"] == cp_id:
                category_gaps[r["category"]].append(gap)
                break

    # Determine gap bucket per task category
    # Use the mean gap for each category
    cat_to_bucket = {}
    for cat, gaps in category_gaps.items():
        mean_gap = sum(gaps) / len(gaps) if gaps else 0
        if mean_gap > 10:
            cat_to_bucket[cat] = "clear_choice"
        elif mean_gap > 3:
            cat_to_bucket[cat] = "moderate_choice"
        else:
            cat_to_bucket[cat] = "near_tie"

    print(f"  Category -> bucket: {cat_to_bucket}")

    # ================================================================
    # Organize data by (task_id, arm)
    # ================================================================
    by_task_arm = {}
    for r in trajectories:
        by_task_arm[(r["task_id"], r["arm"])] = r

    arms = ["P0", "B0", "B1", "PS05", "QOBS", "QCAUSAL"]
    task_ids = sorted(set(r["task_id"] for r in trajectories))

    # ================================================================
    # 1. Success rate per arm
    # ================================================================
    print("\n" + "=" * 70)
    print("1. SUCCESS RATE PER ARM")
    print("=" * 70)

    success_by_arm = {}
    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        successes = sum(1 for r in recs if r["success"])
        rate = successes / n if n else 0
        success_by_arm[arm] = {"n": n, "successes": successes, "rate": rate}
        print(f"  {arm:10s}: {successes}/{n} ({rate:.4f})")

    # ================================================================
    # 2. Paired rescues/breaks with McNemar exact tests
    # ================================================================
    print("\n" + "=" * 70)
    print("2. PAIRED RESCUES/BREAKS (McNemar exact test)")
    print("=" * 70)

    # For each pair (QCAUSAL vs X), count:
    # b = QCAUSAL succeeds, X fails (rescues)
    # c = QCAUSAL fails, X succeeds (breaks)
    for x_arm in ["P0", "B0", "B1", "PS05", "QOBS"]:
        b = 0  # QCAUSAL rescues X
        c = 0  # QCAUSAL breaks X
        for tid in task_ids:
            q = by_task_arm.get((tid, "QCAUSAL"))
            x = by_task_arm.get((tid, x_arm))
            if q and x:
                if q["success"] and not x["success"]:
                    b += 1
                elif not q["success"] and x["success"]:
                    c += 1
        p = mcnemar_exact(b, c)
        print(f"  QCAUSAL vs {x_arm:10s}: rescues={b} breaks={c} p={p:.4f}")

    # Also QOBS vs B0
    b = 0
    c = 0
    for tid in task_ids:
        q = by_task_arm.get((tid, "QOBS"))
        x = by_task_arm.get((tid, "B0"))
        if q and x:
            if q["success"] and not x["success"]:
                b += 1
            elif not q["success"] and x["success"]:
                c += 1
    p = mcnemar_exact(b, c)
    print(f"  QOBS vs B0        : rescues={b} breaks={c} p={p:.4f}")

    # ================================================================
    # 3. Premature DEFER/ANSWER rates
    # ================================================================
    print("\n" + "=" * 70)
    print("3. PREMATURE DEFER/ANSWER RATES")
    print("=" * 70)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        pd = sum(1 for r in recs if r["premature_defer"])
        pa = sum(1 for r in recs if r["premature_answer"])
        print(f"  {arm:10s}: premature_defer={pd}/{n} ({pd/n:.4f}) premature_answer={pa}/{n} ({pa/n:.4f})")

    # ================================================================
    # 4. Loop rate, resource exhaustion, mean steps
    # ================================================================
    print("\n" + "=" * 70)
    print("4. LOOP RATE, RESOURCE EXHAUSTION, MEAN STEPS")
    print("=" * 70)

    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        n = len(recs)
        loops = sum(1 for r in recs if r["terminal_result"] == "loop_detected")
        resource_exh = sum(1 for r in recs if "resource" in r.get("terminal_result", "").lower()
                          or r["terminal_result"] == "STEP_LIMIT")
        steps = [r["steps"] for r in recs]
        calls = [r["model_calls"] for r in recs]
        print(f"  {arm:10s}: loops={loops}/{n} ({loops/n:.4f}) "
              f"step_limit={resource_exh}/{n} ({resource_exh/n:.4f}) "
              f"mean_steps={sum(steps)/n:.2f} mean_calls={sum(calls)/n:.2f}")

    # ================================================================
    # 5. Mean tokens
    # ================================================================
    print("\n" + "=" * 70)
    print("5. MEAN TOKENS")
    print("=" * 70)

    tokens_by_arm = defaultdict(list)
    for c in model_calls:
        tokens_by_arm[c["arm"]].append(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0))
    for arm in arms:
        toks = tokens_by_arm.get(arm, [])
        if toks:
            print(f"  {arm:10s}: mean_tokens={sum(toks)/len(toks):.1f} n_calls={len(toks)}")

    # ================================================================
    # 6. Mean realized utility per arm
    # ================================================================
    print("\n" + "=" * 70)
    print("6. MEAN REALIZED UTILITY")
    print("=" * 70)

    utility_by_arm = {}
    for arm in arms:
        recs = [by_task_arm[(tid, arm)] for tid in task_ids if (tid, arm) in by_task_arm]
        us = [r["realized_utility"] for r in recs]
        mean_u = sum(us) / len(us) if us else 0
        utility_by_arm[arm] = us
        print(f"  {arm:10s}: mean_U={mean_u:.4f} std={np.std(us):.4f}")

    # ================================================================
    # 7. Delta U contrasts with paired 95% CIs
    # ================================================================
    print("\n" + "=" * 70)
    print("7. DELTA U CONTRASTS (paired 95% CI)")
    print("=" * 70)

    contrasts = [
        ("QCAUSAL", "P0"),
        ("QCAUSAL", "B0"),
        ("QCAUSAL", "B1"),
        ("QCAUSAL", "PS05"),
        ("QCAUSAL", "QOBS"),
        ("QOBS", "B0"),
    ]

    contrast_results = {}
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
        contrast_results[f"{a}-{b}"] = {
            "mean_delta": round(mean_diff, 4),
            "ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "excludes_zero": excludes_zero,
            "n": len(diffs),
        }
        print(f"  ΔU({a:>8s} - {b:>8s}) = {mean_diff:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"{'EXCLUDES 0' if excludes_zero else 'includes 0'}")

    # ================================================================
    # 8. Stratified by gap bucket
    # ================================================================
    print("\n" + "=" * 70)
    print("8. STRATIFIED BY GAP BUCKET")
    print("=" * 70)

    # Map tasks to buckets
    task_to_bucket = {}
    for tid in task_ids:
        cat = task_to_category.get(tid, "unknown")
        task_to_bucket[tid] = cat_to_bucket.get(cat, "near_tie")

    for bucket in ["clear_choice", "moderate_choice", "near_tie"]:
        bucket_tasks = [tid for tid in task_ids if task_to_bucket[tid] == bucket]
        if not bucket_tasks:
            continue
        print(f"\n  {bucket} ({len(bucket_tasks)} tasks):")
        for arm in arms:
            recs = [by_task_arm[(tid, arm)] for tid in bucket_tasks if (tid, arm) in by_task_arm]
            if recs:
                us = [r["realized_utility"] for r in recs]
                sr = sum(1 for r in recs if r["success"]) / len(recs)
                print(f"    {arm:10s}: n={len(recs):3d} mean_U={sum(us)/len(us):.2f} success={sr:.3f}")

        # Delta U for QCAUSAL vs B0 in this bucket
        diffs = []
        for tid in bucket_tasks:
            ra = by_task_arm.get((tid, "QCAUSAL"))
            rb = by_task_arm.get((tid, "B0"))
            if ra and rb:
                diffs.append(ra["realized_utility"] - rb["realized_utility"])
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            ci_lo, ci_hi = bootstrap_ci_paired(diffs)
            print(f"    ΔU(QCAUSAL-B0) = {mean_diff:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # ================================================================
    # 9. Near-optimal selected-action rate at epsilon=3
    # ================================================================
    print("\n" + "=" * 70)
    print("9. NEAR-OPTIMAL SELECTED-ACTION RATE (epsilon=3)")
    print("=" * 70)
    print("  (Measures: P(a_LLM in OptimalSet_epsilon=3 | arm))")
    print("  Using first action selected by the LLM in each trajectory")

    # For each task, get the Q values for each action from the causal data
    # Then check if the LLM's first action is in the optimal set
    for arm in arms:
        correct = 0
        total = 0
        for tid in task_ids:
            r = by_task_arm.get((tid, arm))
            if not r or not r["actions_taken"]:
                continue
            cat = task_to_category.get(tid, "unknown")
            q_values = category_q.get(cat, {})
            if not q_values:
                continue
            q_star = max(q_values.values())
            optimal_set = {a for a, q in q_values.items() if q >= q_star - 3}
            first_action = r["actions_taken"][0]
            if first_action in optimal_set:
                correct += 1
            total += 1
        rate = correct / total if total else 0
        print(f"  {arm:10s}: {correct}/{total} ({rate:.4f})")

    # ================================================================
    # 10. Mean causal regret of chosen action
    # ================================================================
    print("\n" + "=" * 70)
    print("10. MEAN CAUSAL REGRET OF CHOSEN ACTION")
    print("=" * 70)
    print("  (Using first action selected by the LLM)")

    for arm in arms:
        regrets = []
        for tid in task_ids:
            r = by_task_arm.get((tid, arm))
            if not r or not r["actions_taken"]:
                continue
            cat = task_to_category.get(tid, "unknown")
            q_values = category_q.get(cat, {})
            if not q_values:
                continue
            q_star = max(q_values.values())
            first_action = r["actions_taken"][0]
            q_chosen = q_values.get(first_action, q_star)
            regrets.append(q_star - q_chosen)
        mean_regret = sum(regrets) / len(regrets) if regrets else 0
        print(f"  {arm:10s}: mean_regret={mean_regret:.4f} n={len(regrets)}")

    # ================================================================
    # 11. Summary and promotion gate
    # ================================================================
    print("\n" + "=" * 70)
    print("11. PROMOTION GATE")
    print("=" * 70)

    # Gate 1: ΔU(QCAUSAL-B0) > 0 with CI excluding zero
    qcausal_b0 = contrast_results["QCAUSAL-B0"]
    gate1 = qcausal_b0["mean_delta"] > 0 and qcausal_b0["excludes_zero"]
    print(f"  Gate 1: ΔU(QCAUSAL-B0) > 0 with CI excluding zero: "
          f"{'PASS' if gate1 else 'FAIL'} "
          f"(delta={qcausal_b0['mean_delta']:+.4f}, CI={qcausal_b0['ci']})")

    # Gate 2: Improved or non-inferior success rate
    qcausal_sr = success_by_arm["QCAUSAL"]["rate"]
    b0_sr = success_by_arm["B0"]["rate"]
    gate2 = qcausal_sr >= b0_sr
    print(f"  Gate 2: Success rate >= B0: {'PASS' if gate2 else 'FAIL'} "
          f"(QCAUSAL={qcausal_sr:.4f}, B0={b0_sr:.4f})")

    # Gate 3: No increase in premature DEFER
    qcausal_pd = sum(1 for tid in task_ids if by_task_arm.get((tid, "QCAUSAL"), {}).get("premature_defer", False))
    b0_pd = sum(1 for tid in task_ids if by_task_arm.get((tid, "B0"), {}).get("premature_defer", False))
    gate3 = qcausal_pd <= b0_pd
    print(f"  Gate 3: No increase in premature DEFER: {'PASS' if gate3 else 'FAIL'} "
          f"(QCAUSAL={qcausal_pd}, B0={b0_pd})")

    # Gate 4: No increase in premature ANSWER
    qcausal_pa = sum(1 for tid in task_ids if by_task_arm.get((tid, "QCAUSAL"), {}).get("premature_answer", False))
    b0_pa = sum(1 for tid in task_ids if by_task_arm.get((tid, "B0"), {}).get("premature_answer", False))
    gate4 = qcausal_pa <= b0_pa
    print(f"  Gate 4: No increase in premature ANSWER: {'PASS' if gate4 else 'FAIL'} "
          f"(QCAUSAL={qcausal_pa}, B0={b0_pa})")

    # Gate 5: QCAUSAL > QOBS with CI excluding zero
    qcausal_qobs = contrast_results["QCAUSAL-QOBS"]
    gate5 = qcausal_qobs["mean_delta"] > 0 and qcausal_qobs["excludes_zero"]
    print(f"  Gate 5: ΔU(QCAUSAL-QOBS) > 0 with CI excluding zero: "
          f"{'PASS' if gate5 else 'FAIL'} "
          f"(delta={qcausal_qobs['mean_delta']:+.4f}, CI={qcausal_qobs['ci']})")

    overall = gate1 and gate2 and gate3 and gate4
    print(f"\n  OVERALL PROMOTION: {'PASS' if overall else 'FAIL'}")
    print(f"  (Gate 5 is a secondary criterion, not required for promotion)")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "success_by_arm": success_by_arm,
        "contrast_results": contrast_results,
        "gates": {
            "gate1_qcausal_gt_b0_ci": gate1,
            "gate2_success_noninferior": gate2,
            "gate3_no_increase_premature_defer": gate3,
            "gate4_no_increase_premature_answer": gate4,
            "gate5_qcausal_gt_qobs_ci": gate5,
            "overall": overall,
        },
    }
    output_path = six_arm_dir / "analysis_v1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
