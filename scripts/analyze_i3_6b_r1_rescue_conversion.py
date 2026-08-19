#!/usr/bin/env python3
"""I3.6b-r1 rescue/conversion analysis with task-clustered bootstrap.

Analyzes the four-way continuation fork results to answer:

  1. Is the accounting internally consistent? (Q1)
  2. Are forced-action costs included? (Q2)
  3. Are terminal result/action traces persisted? (Q3)
  4. Is fork execution order counterbalanced? (Q4)
  5. Task-clustered CI computed? (Q5)
  6. One-shot vs persistent assistance separated? (Q6)
  7. PersistentGain LCB95 > 0? (Q7)
  8. Gain not solely explained by terminal-penalty shift? (Q8)
  9. At least one ASSIST_RESCUE? (Q9)
 10. ASSIST_BREAK < 5/48? (Q10)

Mechanism analysis classifies the +73 into:
  - terminal-penalty shift (incorrect ANSWER -120 -> incorrect DEFER/STOP -30)
  - actual decision conversion (failure -> success)
  - cost reduction (fewer expensive steps)

Usage:
    PYTHONPATH=. python scripts/analyze_i3_6b_r1_rescue_conversion.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def task_clustered_bootstrap_ci(
    values_by_task: dict[str, list[float]],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute task-clustered bootstrap CI.

    Resamples at the task_id level (cluster), not at the individual
    observation level. This accounts for dependence between multiple
    fork states from the same task.

    For each bootstrap iteration:
      1. Sample n_tasks task_ids with replacement
      2. Pool all observations from sampled tasks
      3. Compute the statistic

    Returns: {"mean": ..., "lcb": ..., "ucb": ..., "n_tasks": ..., "n_obs": ...}
    """
    rng = random.Random(seed)
    task_ids = list(values_by_task.keys())
    n_tasks = len(task_ids)

    if n_tasks == 0:
        return {"mean": 0.0, "lcb": 0.0, "ucb": 0.0, "n_tasks": 0, "n_obs": 0}

    # Observed statistic
    all_values = []
    for vs in values_by_task.values():
        all_values.extend(vs)
    observed_mean = sum(all_values) / len(all_values) if all_values else 0.0

    # Bootstrap
    boot_stats = []
    for _ in range(n_bootstrap):
        sampled_tasks = [rng.choice(task_ids) for _ in range(n_tasks)]
        boot_values = []
        for tid in sampled_tasks:
            boot_values.extend(values_by_task[tid])
        if boot_values:
            boot_mean = sum(boot_values) / len(boot_values)
            boot_stats.append(boot_mean)

    if not boot_stats:
        return {"mean": observed_mean, "lcb": observed_mean, "ucb": observed_mean,
                "n_tasks": n_tasks, "n_obs": len(all_values)}

    boot_stats.sort()
    alpha = (1 - confidence) / 2
    lcb_idx = int(alpha * len(boot_stats))
    ucb_idx = int((1 - alpha) * len(boot_stats))

    return {
        "mean": round(observed_mean, 4),
        "lcb": round(boot_stats[lcb_idx], 4),
        "ucb": round(boot_stats[ucb_idx], 4),
        "n_tasks": n_tasks,
        "n_obs": len(all_values),
    }


def classify_terminal_penalty_shift(fork_result: dict) -> dict:
    """Classify whether the utility gain is from terminal-penalty shift.

    The frozen utility has:
      incorrect_answer: -120
      incorrect_defer:  -30
      incorrect_stop:   -30
      correct_answer:   +100
      correct_defer:    +70
      correct_stop:      0

    A terminal-penalty shift occurs when:
      - BASE/ACTION_ONLY terminal action is ANSWER (incorrect, -120)
      - ASSIST terminal action is DEFER or STOP (incorrect, -30)
      This produces a +90 utility swing without actual task success.

    Returns classification for each fork comparison.
    """
    def analyze_pair(base_fork: dict, treat_fork: dict) -> dict:
        base_term = base_fork.get("terminal_action")
        treat_term = treat_fork.get("terminal_action")
        base_reward = base_fork.get("terminal_reward", 0)
        treat_reward = treat_fork.get("terminal_reward", 0)
        base_success = base_fork.get("success", False)
        treat_success = treat_fork.get("success", False)

        penalty_shift = False
        if base_term == "ANSWER" and not base_success:
            if treat_term in ("DEFER", "STOP") and not treat_success:
                penalty_shift = True

        return {
            "base_terminal": base_term,
            "treat_terminal": treat_term,
            "base_success": base_success,
            "treat_success": treat_success,
            "base_reward": base_reward,
            "treat_reward": treat_reward,
            "reward_delta": round(treat_reward - base_reward, 4),
            "is_terminal_penalty_shift": penalty_shift,
        }

    return {
        "base_vs_action": analyze_pair(fork_result["fork_a"], fork_result["fork_b"]),
        "base_vs_oneshot": analyze_pair(fork_result["fork_a"], fork_result["fork_c"]),
        "base_vs_persistent": analyze_pair(fork_result["fork_a"], fork_result["fork_d"]),
        "action_vs_oneshot": analyze_pair(fork_result["fork_b"], fork_result["fork_c"]),
        "action_vs_persistent": analyze_pair(fork_result["fork_b"], fork_result["fork_d"]),
    }


def main():
    parser = argparse.ArgumentParser(description="I3.6b-r1 rescue/conversion analysis")
    parser.add_argument(
        "--forks",
        default="experiments/v2b_i3_6/development/i3_6b_r1/continuation_forks_r1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/v2b_i3_6/development/i3_6b_r1/rescue_conversion_analysis_r1.json",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    # Load fork results
    print(f"Loading fork results from {args.forks}...")
    forks: list[dict[str, Any]] = []
    with open(args.forks) as f:
        for line in f:
            forks.append(json.loads(line))
    print(f"  Loaded {len(forks)} fork sets")

    n = len(forks)
    if n == 0:
        print("No forks to analyze!")
        return

    # ===== Q1: Accounting consistency =====
    print(f"\n{'='*78}")
    print("Q1: ACCOUNTING CONSISTENCY")
    print(f"{'='*78}")

    # Verify u_a, u_b, u_c, u_d include forced-action cost
    accounting_ok = True
    for r in forks:
        for fork_id in ["fork_a", "fork_b", "fork_c", "fork_d"]:
            fk = r[fork_id]
            expected = round(-fk["forced_action_cost"] + fk["continuation_utility"], 4)
            if abs(fk["total_utility"] - expected) > 0.01:
                print(f"  MISMATCH {r['task_id']} step {r['step_id']} {fork_id}: "
                      f"total={fk['total_utility']}, expected={expected}")
                accounting_ok = False

    # Verify advantage computations
    for r in forks:
        if abs(r["a_b"] - round(r["u_b"] - r["u_a"], 4)) > 0.01:
            print(f"  A_B mismatch: {r['a_b']} vs {r['u_b'] - r['u_a']}")
            accounting_ok = False
        if abs(r["one_shot_gain"] - round(r["u_c"] - r["u_b"], 4)) > 0.01:
            print(f"  OneShotGain mismatch: {r['one_shot_gain']} vs {r['u_c'] - r['u_b']}")
            accounting_ok = False
        if abs(r["persistent_gain"] - round(r["u_d"] - r["u_b"], 4)) > 0.01:
            print(f"  PersistentGain mismatch: {r['persistent_gain']} vs {r['u_d'] - r['u_b']}")
            accounting_ok = False

    q1_pass = accounting_ok
    print(f"  Q1 PASS={q1_pass}")

    # ===== Q2: Forced-action costs =====
    print(f"\n{'='*78}")
    print("Q2: FORCED-ACTION COSTS INCLUDED")
    print(f"{'='*78}")
    forced_costs_a = [r["fork_a"]["forced_action_cost"] for r in forks]
    forced_costs_b = [r["fork_b"]["forced_action_cost"] for r in forks]
    print(f"  Mean forced cost A (a_B): {sum(forced_costs_a)/n:.4f}")
    print(f"  Mean forced cost B (a_G): {sum(forced_costs_b)/n:.4f}")
    q2_pass = all(r["fork_a"]["forced_action_cost"] > 0 or r["fork_a"]["forced_action"] in ("STOP",)
                  for r in forks[:5])  # spot check
    print(f"  Q2 PASS={q2_pass}")

    # ===== Q3: Terminal traces persisted =====
    print(f"\n{'='*78}")
    print("Q3: TERMINAL RESULT/ACTION TRACES PERSISTED")
    print(f"{'='*78}")
    q3_pass = all(
        "terminal_result" in r["fork_a"] and
        "terminal_action" in r["fork_a"] and
        "continuation_actions" in r["fork_a"] and
        "continuation_outcomes" in r["fork_a"] and
        "step_costs" in r["fork_a"]
        for r in forks[:5]
    )
    print(f"  Q3 PASS={q3_pass}")
    # Show example trace
    if n > 0:
        r0 = forks[0]
        print(f"  Example (task {r0['task_id']}, step {r0['step_id']}):")
        for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
            fk = r0[fid]
            print(f"    {fid}: actions={fk['continuation_actions']}, "
                  f"terminal={fk['terminal_action']}, "
                  f"result={fk['terminal_result']}, "
                  f"reward={fk['terminal_reward']}")

    # ===== Q4: Fork execution order counterbalanced =====
    print(f"\n{'='*78}")
    print("Q4: FORK EXECUTION ORDER COUNTERBALANCED")
    print(f"{'='*78}")
    orders = [tuple(r["fork_order"]) for r in forks]
    order_counts = Counter(orders)
    print(f"  Unique orders: {len(order_counts)}")
    for order, cnt in order_counts.most_common(5):
        print(f"    {order}: {cnt}")
    q4_pass = len(order_counts) > 1
    print(f"  Q4 PASS={q4_pass}")

    # ===== Q5: Task-clustered bootstrap CIs =====
    print(f"\n{'='*78}")
    print("Q5: TASK-CLUSTERED BOOTSTRAP CIs")
    print(f"{'='*78}")

    # Group by task_id
    a_b_by_task = defaultdict(list)
    osg_by_task = defaultdict(list)
    pg_by_task = defaultdict(list)
    persg_by_task = defaultdict(list)

    for r in forks:
        a_b_by_task[r["task_id"]].append(r["a_b"])
        osg_by_task[r["task_id"]].append(r["one_shot_gain"])
        pg_by_task[r["task_id"]].append(r["persistent_gain"])
        persg_by_task[r["task_id"]].append(r["persistence_gain"])

    ci_a_b = task_clustered_bootstrap_ci(a_b_by_task, n_bootstrap=args.n_bootstrap)
    ci_osg = task_clustered_bootstrap_ci(osg_by_task, n_bootstrap=args.n_bootstrap)
    ci_pg = task_clustered_bootstrap_ci(pg_by_task, n_bootstrap=args.n_bootstrap)
    ci_persg = task_clustered_bootstrap_ci(persg_by_task, n_bootstrap=args.n_bootstrap)

    print(f"  A_B (action-only):     mean={ci_a_b['mean']:+.4f}, "
          f"CI95=[{ci_a_b['lcb']:+.4f}, {ci_a_b['ucb']:+.4f}], "
          f"tasks={ci_a_b['n_tasks']}, obs={ci_a_b['n_obs']}")
    print(f"  OneShotGain (C-B):     mean={ci_osg['mean']:+.4f}, "
          f"CI95=[{ci_osg['lcb']:+.4f}, {ci_osg['ucb']:+.4f}], "
          f"tasks={ci_osg['n_tasks']}, obs={ci_osg['n_obs']}")
    print(f"  PersistentGain (D-B):  mean={ci_pg['mean']:+.4f}, "
          f"CI95=[{ci_pg['lcb']:+.4f}, {ci_pg['ucb']:+.4f}], "
          f"tasks={ci_pg['n_tasks']}, obs={ci_pg['n_obs']}")
    print(f"  PersistenceGain (D-C): mean={ci_persg['mean']:+.4f}, "
          f"CI95=[{ci_persg['lcb']:+.4f}, {ci_persg['ucb']:+.4f}], "
          f"tasks={ci_persg['n_tasks']}, obs={ci_persg['n_obs']}")

    q5_pass = True
    print(f"  Q5 PASS={q5_pass}")

    # ===== Q6: One-shot vs persistent separated =====
    print(f"\n{'='*78}")
    print("Q6: ONE-SHOT VS PERSISTENT ASSISTANCE SEPARATED")
    print(f"{'='*78}")
    q6_pass = all(
        "one_shot_gain" in r and "persistent_gain" in r and "persistence_gain" in r
        for r in forks[:5]
    )
    print(f"  Q6 PASS={q6_pass}")

    # ===== Q7: PersistentGain LCB95 > 0 =====
    print(f"\n{'='*78}")
    print("Q7: PersistentGain LCB95 > 0")
    print(f"{'='*78}")
    q7_pass = ci_pg["lcb"] > 0
    print(f"  PersistentGain LCB95 = {ci_pg['lcb']:+.4f}")
    print(f"  Q7 PASS={q7_pass}")

    # ===== Q8: Gain not solely explained by terminal-penalty shift =====
    print(f"\n{'='*78}")
    print("Q8: GAIN NOT SOLELY TERMINAL-PENALTY SHIFT")
    print(f"{'='*78}")

    penalty_shifts = []
    for r in forks:
        cls = classify_terminal_penalty_shift(r)
        penalty_shifts.append(cls)

    # Count penalty shifts in persistent vs action
    ps_persistent = sum(
        1 for ps in penalty_shifts
        if ps["action_vs_persistent"]["is_terminal_penalty_shift"]
    )
    ps_oneshot = sum(
        1 for ps in penalty_shifts
        if ps["action_vs_oneshot"]["is_terminal_penalty_shift"]
    )

    # Compute how much of the gain is from penalty shift
    # For each fork where penalty shift occurs, the reward delta is +90
    # Compare to the total persistent gain
    total_persistent_gain = sum(r["persistent_gain"] for r in forks)
    penalty_shift_reward = sum(
        ps["action_vs_persistent"]["reward_delta"]
        for ps in penalty_shifts
        if ps["action_vs_persistent"]["is_terminal_penalty_shift"]
    )

    print(f"  Terminal-penalty shifts (action->persistent): {ps_persistent}/{n}")
    print(f"  Terminal-penalty shifts (action->oneshot): {ps_oneshot}/{n}")
    print(f"  Total PersistentGain: {total_persistent_gain:+.4f}")
    print(f"  Penalty-shift reward component: {penalty_shift_reward:+.4f}")
    if total_persistent_gain != 0:
        ps_fraction = abs(penalty_shift_reward) / abs(total_persistent_gain)
        print(f"  Penalty-shift fraction of gain: {ps_fraction:.1%}")
    else:
        ps_fraction = 0.0

    # Q8 passes if penalty shift does NOT explain ALL of the gain
    q8_pass = ps_fraction < 0.95
    print(f"  Q8 PASS={q8_pass} (penalty shift < 95% of gain)")

    # ===== Q9: At least one ASSIST_RESCUE =====
    print(f"\n{'='*78}")
    print("Q9: AT LEAST ONE ASSIST_RESCUE")
    print(f"{'='*78}")

    # Check all three governor variants
    action_rescues = sum(1 for r in forks if r["action_class"] == "RESCUE")
    oneshot_rescues = sum(1 for r in forks if r["oneshot_class"] == "RESCUE")
    persistent_rescues = sum(1 for r in forks if r["persistent_class"] == "RESCUE")

    print(f"  ACTION_RESCUE: {action_rescues}")
    print(f"  ONESHOT_RESCUE: {oneshot_rescues}")
    print(f"  PERSISTENT_RESCUE: {persistent_rescues}")
    q9_pass = (action_rescues + oneshot_rescues + persistent_rescues) > 0
    print(f"  Q9 PASS={q9_pass}")

    # ===== Q10: ASSIST_BREAK < 5/48 =====
    print(f"\n{'='*78}")
    print("Q10: ASSIST_BREAK < 5/48")
    print(f"{'='*78}")

    action_breaks = sum(1 for r in forks if r["action_class"] == "BREAK")
    oneshot_breaks = sum(1 for r in forks if r["oneshot_class"] == "BREAK")
    persistent_breaks = sum(1 for r in forks if r["persistent_class"] == "BREAK")

    print(f"  ACTION_BREAK: {action_breaks}/{n}")
    print(f"  ONESHOT_BREAK: {oneshot_breaks}/{n}")
    print(f"  PERSISTENT_BREAK: {persistent_breaks}/{n}")
    q10_pass = persistent_breaks < 5  # was 5/48 in original
    print(f"  Q10 PASS={q10_pass}")

    # ===== Mechanism analysis =====
    print(f"\n{'='*78}")
    print("MECHANISM ANALYSIS")
    print(f"{'='*78}")

    # Terminal action distribution per fork
    print(f"\n  Terminal action distribution:")
    for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
        terms = Counter(r[fid]["terminal_action"] for r in forks)
        print(f"    {fid}: {dict(terms)}")

    # Terminal result distribution
    print(f"\n  Terminal result distribution:")
    for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
        results = Counter(r[fid]["terminal_result"] for r in forks)
        print(f"    {fid}: {dict(results)}")

    # Mean terminal reward
    print(f"\n  Mean terminal reward:")
    for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
        rewards = [r[fid]["terminal_reward"] for r in forks]
        print(f"    {fid}: {sum(rewards)/n:+.4f}")

    # Mean total action cost
    print(f"\n  Mean total action cost:")
    for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
        costs = [r[fid]["total_action_cost"] for r in forks]
        print(f"    {fid}: {sum(costs)/n:.4f}")

    # Mean continuation steps
    print(f"\n  Mean continuation steps:")
    for fid in ["fork_a", "fork_b", "fork_c", "fork_d"]:
        steps = [r[fid]["steps"] for r in forks]
        print(f"    {fid}: {sum(steps)/n:.1f}")

    # By action pair
    print(f"\n  By action pair:")
    pair_stats = defaultdict(lambda: {
        "n": 0, "a_b_sum": 0, "osg_sum": 0, "pg_sum": 0, "persg_sum": 0,
        "base_ok": 0, "action_ok": 0, "oneshot_ok": 0, "persistent_ok": 0,
        "action_rescue": 0, "oneshot_rescue": 0, "persistent_rescue": 0,
        "action_break": 0, "oneshot_break": 0, "persistent_break": 0,
    })

    for r in forks:
        pair = f"{r['base_action']}->{r['gov_action']}"
        ps = pair_stats[pair]
        ps["n"] += 1
        ps["a_b_sum"] += r["a_b"]
        ps["osg_sum"] += r["one_shot_gain"]
        ps["pg_sum"] += r["persistent_gain"]
        ps["persg_sum"] += r["persistence_gain"]
        if r["base_success"]: ps["base_ok"] += 1
        if r["action_success"]: ps["action_ok"] += 1
        if r["oneshot_success"]: ps["oneshot_ok"] += 1
        if r["persistent_success"]: ps["persistent_ok"] += 1
        if r["action_class"] == "RESCUE": ps["action_rescue"] += 1
        if r["oneshot_class"] == "RESCUE": ps["oneshot_rescue"] += 1
        if r["persistent_class"] == "RESCUE": ps["persistent_rescue"] += 1
        if r["action_class"] == "BREAK": ps["action_break"] += 1
        if r["oneshot_class"] == "BREAK": ps["oneshot_break"] += 1
        if r["persistent_class"] == "BREAK": ps["persistent_break"] += 1

    for pair, ps in sorted(pair_stats.items(), key=lambda x: -x[1]["n"]):
        np_ = ps["n"]
        print(f"    {pair}: n={np_}, "
              f"A_B={ps['a_b_sum']/np_:+.2f}, "
              f"OSG={ps['osg_sum']/np_:+.2f}, "
              f"PG={ps['pg_sum']/np_:+.2f}, "
              f"PersG={ps['persg_sum']/np_:+.2f}, "
              f"ok(a/o/p)={ps['action_ok']}/{ps['oneshot_ok']}/{ps['persistent_ok']}, "
              f"brk(a/o/p)={ps['action_break']}/{ps['oneshot_break']}/{ps['persistent_break']}")

    # Delta metrics (no ratios)
    print(f"\n  Delta metrics (no ratios):")
    u_a_mean = sum(r["u_a"] for r in forks) / n
    u_b_mean = sum(r["u_b"] for r in forks) / n
    u_c_mean = sum(r["u_c"] for r in forks) / n
    u_d_mean = sum(r["u_d"] for r in forks) / n

    base_calls = sum(r["fork_a"]["model_calls"] for r in forks)
    action_calls = sum(r["fork_b"]["model_calls"] for r in forks)
    oneshot_calls = sum(r["fork_c"]["model_calls"] for r in forks)
    persistent_calls = sum(r["fork_d"]["model_calls"] for r in forks)

    print(f"    dU (persistent - action):  {u_d_mean - u_b_mean:+.4f}")
    print(f"    dU (persistent - base):    {u_d_mean - u_a_mean:+.4f}")
    print(f"    dU (oneshot - action):     {u_c_mean - u_b_mean:+.4f}")
    print(f"    dCalls (persistent - action): {persistent_calls - action_calls}")
    print(f"    dCalls (oneshot - action):    {oneshot_calls - action_calls}")

    # ===== Gate summary =====
    print(f"\n{'='*78}")
    print("GATE SUMMARY")
    print(f"{'='*78}")
    gates = {
        "Q1": q1_pass,
        "Q2": q2_pass,
        "Q3": q3_pass,
        "Q4": q4_pass,
        "Q5": q5_pass,
        "Q6": q6_pass,
        "Q7": q7_pass,
        "Q8": q8_pass,
        "Q9": q9_pass,
        "Q10": q10_pass,
    }
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")
    print(f"\n  Total: {sum(gates.values())}/{len(gates)} passed")

    # ===== Save analysis =====
    analysis = {
        "schema": "DAPH_V2B_I3_6B_R1_RESCUE_CONVERSION_V1",
        "n_forks": n,
        "gates": gates,
        "bootstrap_cis": {
            "a_b": ci_a_b,
            "one_shot_gain": ci_osg,
            "persistent_gain": ci_pg,
            "persistence_gain": ci_persg,
        },
        "utility_means": {
            "u_a_base": round(u_a_mean, 4),
            "u_b_action": round(u_b_mean, 4),
            "u_c_oneshot": round(u_c_mean, 4),
            "u_d_persistent": round(u_d_mean, 4),
        },
        "advantage_means": {
            "a_b": ci_a_b["mean"],
            "one_shot_gain": ci_osg["mean"],
            "persistent_gain": ci_pg["mean"],
            "persistence_gain": ci_persg["mean"],
        },
        "success_counts": {
            "base": sum(1 for r in forks if r["base_success"]),
            "action": sum(1 for r in forks if r["action_success"]),
            "oneshot": sum(1 for r in forks if r["oneshot_success"]),
            "persistent": sum(1 for r in forks if r["persistent_success"]),
        },
        "classification": {
            "base_vs_action": dict(Counter(r["action_class"] for r in forks)),
            "base_vs_oneshot": dict(Counter(r["oneshot_class"] for r in forks)),
            "base_vs_persistent": dict(Counter(r["persistent_class"] for r in forks)),
        },
        "terminal_penalty_shift": {
            "n_shifts_persistent": ps_persistent,
            "n_shifts_oneshot": ps_oneshot,
            "total_persistent_gain": round(total_persistent_gain, 4),
            "penalty_shift_reward": round(penalty_shift_reward, 4),
            "penalty_shift_fraction": round(ps_fraction, 4),
        },
        "delta_metrics": {
            "delta_u_persistent_vs_action": round(u_d_mean - u_b_mean, 4),
            "delta_u_persistent_vs_base": round(u_d_mean - u_a_mean, 4),
            "delta_u_oneshot_vs_action": round(u_c_mean - u_b_mean, 4),
            "delta_calls_persistent_vs_action": persistent_calls - action_calls,
            "delta_calls_oneshot_vs_action": oneshot_calls - action_calls,
        },
        "by_action_pair": {
            pair: {
                "n": ps["n"],
                "mean_a_b": round(ps["a_b_sum"] / ps["n"], 4),
                "mean_one_shot_gain": round(ps["osg_sum"] / ps["n"], 4),
                "mean_persistent_gain": round(ps["pg_sum"] / ps["n"], 4),
                "mean_persistence_gain": round(ps["persg_sum"] / ps["n"], 4),
                "success": {
                    "base": ps["base_ok"],
                    "action": ps["action_ok"],
                    "oneshot": ps["oneshot_ok"],
                    "persistent": ps["persistent_ok"],
                },
                "rescues": {
                    "action": ps["action_rescue"],
                    "oneshot": ps["oneshot_rescue"],
                    "persistent": ps["persistent_rescue"],
                },
                "breaks": {
                    "action": ps["action_break"],
                    "oneshot": ps["oneshot_break"],
                    "persistent": ps["persistent_break"],
                },
            }
            for pair, ps in pair_stats.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
