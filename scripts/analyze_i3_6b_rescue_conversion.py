#!/usr/bin/env python3
"""I3.6b rescue/conversion analysis.

Analyzes the continuation fork results to understand:
  1. Why base successes are broken by governor interventions
  2. The information conversion pipeline
  3. Action-pair-specific execution gain patterns
  4. Cost-adjusted utility comparison

Usage:
    PYTHONPATH=. python scripts/analyze_i3_6b_rescue_conversion.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="I3.6b rescue/conversion analysis")
    parser.add_argument(
        "--forks",
        default="experiments/v2b_i3_6/development/i3_6b/continuation_forks_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/v2b_i3_6/development/i3_6b/rescue_conversion_analysis_v1.json",
    )
    args = parser.parse_args()

    # Load fork results
    print(f"Loading fork results from {args.forks}...")
    forks: list[dict[str, Any]] = []
    with open(args.forks) as f:
        for line in f:
            forks.append(json.loads(line))
    print(f"  Loaded {len(forks)} forks")

    n = len(forks)
    if n == 0:
        print("No forks to analyze!")
        return

    # 1. Analyze broken successes
    broken_by_assist = [r for r in forks if r["rescue_class"] == "ASSIST_BREAK"]
    broken_by_action = [r for r in forks if r["action_class"] == "ACTION_BREAK"]
    base_successes = [r for r in forks if r["base_success"]]

    print(f"\n{'='*78}")
    print("RESCUE/CONVERSION ANALYSIS")
    print(f"{'='*78}")

    print(f"\n--- Base success analysis ({len(base_successes)} successes) ---")
    for r in base_successes:
        print(f"  {r['task_id']} step {r['step_id']}: "
              f"{r['base_action']} -> {r['gov_action']}, "
              f"U_base={r['u_base']:+.2f}, U_action={r['u_action_only']:+.2f}, "
              f"U_assist={r['u_exec_assist']:+.2f}, "
              f"action_{'OK' if r['action_success'] else 'BREAK'}, "
              f"assist_{'OK' if r['assist_success'] else 'BREAK'}")

    # 2. Action-pair-specific analysis
    print(f"\n--- By action pair ---")
    pair_stats = defaultdict(lambda: {
        "n": 0, "a_b_sum": 0, "a_e_sum": 0, "eg_sum": 0,
        "base_success": 0, "action_success": 0, "assist_success": 0,
        "rescue": 0, "break": 0,
    })

    for r in forks:
        pair = f"{r['base_action']}->{r['gov_action']}"
        ps = pair_stats[pair]
        ps["n"] += 1
        ps["a_b_sum"] += r["a_b"]
        ps["a_e_sum"] += r["a_e"]
        ps["eg_sum"] += r["execution_gain"]
        if r["base_success"]:
            ps["base_success"] += 1
        if r["action_success"]:
            ps["action_success"] += 1
        if r["assist_success"]:
            ps["assist_success"] += 1
        if r["rescue_class"] == "ASSIST_RESCUE":
            ps["rescue"] += 1
        if r["rescue_class"] == "ASSIST_BREAK":
            ps["break"] += 1

    for pair, ps in sorted(pair_stats.items(), key=lambda x: -x[1]["n"]):
        n_pair = ps["n"]
        print(f"  {pair}: n={n_pair}, "
              f"mean_A_B={ps['a_b_sum']/n_pair:+.2f}, "
              f"mean_A_E={ps['a_e_sum']/n_pair:+.2f}, "
              f"mean_EG={ps['eg_sum']/n_pair:+.2f}, "
              f"base_ok={ps['base_success']}, "
              f"assist_ok={ps['assist_success']}, "
              f"rescue={ps['rescue']}, break={ps['break']}")

    # 3. Utility distribution
    u_base = [r["u_base"] for r in forks]
    u_action = [r["u_action_only"] for r in forks]
    u_assist = [r["u_exec_assist"] for r in forks]

    a_b_values = [r["a_b"] for r in forks]
    a_e_values = [r["a_e"] for r in forks]
    eg_values = [r["execution_gain"] for r in forks]

    # Count positive/negative
    a_b_pos = sum(1 for x in a_b_values if x > 0)
    a_b_neg = sum(1 for x in a_b_values if x < 0)
    a_e_pos = sum(1 for x in a_e_values if x > 0)
    a_e_neg = sum(1 for x in a_e_values if x < 0)
    eg_pos = sum(1 for x in eg_values if x > 0)
    eg_neg = sum(1 for x in eg_values if x < 0)

    print(f"\n--- Advantage distribution ---")
    print(f"  A_B (action-only):  positive={a_b_pos}, negative={a_b_neg}, "
          f"mean={sum(a_b_values)/n:+.4f}")
    print(f"  A_E (exec-assist):  positive={a_e_pos}, negative={a_e_neg}, "
          f"mean={sum(a_e_values)/n:+.4f}")
    print(f"  ExecutionGain:      positive={eg_pos}, negative={eg_neg}, "
          f"mean={sum(eg_values)/n:+.4f}")

    # 4. Cost-adjusted analysis
    base_calls = sum(r["base_model_calls"] for r in forks)
    action_calls = sum(r["action_model_calls"] for r in forks)
    assist_calls = sum(r["assist_model_calls"] for r in forks)

    # Mean per-fork model calls
    base_calls_mean = base_calls / n
    action_calls_mean = action_calls / n
    assist_calls_mean = assist_calls / n

    # Utility per model call
    u_per_call_base = sum(u_base) / max(base_calls, 1)
    u_per_call_action = sum(u_action) / max(action_calls, 1)
    u_per_call_assist = sum(u_assist) / max(assist_calls, 1)

    print(f"\n--- Cost-adjusted analysis ---")
    print(f"  Mean model calls per fork:")
    print(f"    BASE:        {base_calls_mean:.1f}")
    print(f"    ACTION_ONLY: {action_calls_mean:.1f}")
    print(f"    EXEC_ASSIST: {assist_calls_mean:.1f}")
    print(f"  Utility per model call:")
    print(f"    BASE:        {u_per_call_base:+.4f}")
    print(f"    ACTION_ONLY: {u_per_call_action:+.4f}")
    print(f"    EXEC_ASSIST: {u_per_call_assist:+.4f}")

    # 5. Conversion pipeline analysis
    # For each fork, classify the conversion stage
    conversion_stages = {
        "base_success_action_fail": 0,
        "base_success_assist_fail": 0,
        "base_fail_action_success": 0,
        "base_fail_assist_success": 0,
        "both_success": 0,
        "both_fail": 0,
    }

    for r in forks:
        b_ok = r["base_success"]
        a_ok = r["action_success"]
        e_ok = r["assist_success"]

        if b_ok and a_ok:
            conversion_stages["both_success"] += 1
        elif b_ok and not a_ok:
            conversion_stages["base_success_action_fail"] += 1
        elif not b_ok and a_ok:
            conversion_stages["base_fail_action_success"] += 1
        else:
            conversion_stages["both_fail"] += 1

    print(f"\n--- Conversion pipeline (BASE vs ACTION_ONLY) ---")
    for stage, count in conversion_stages.items():
        print(f"  {stage}: {count}")

    # 6. Step distribution of broken successes
    print(f"\n--- Broken success step distribution ---")
    break_steps = Counter(r["step_id"] for r in broken_by_assist)
    for step, count in sorted(break_steps.items()):
        print(f"  step {step}: {count} breaks")

    # 7. Summary
    analysis = {
        "schema": "DAPH_V2B_I3_6B_RESCUE_CONVERSION_V1",
        "n_forks": n,
        "base_successes": len(base_successes),
        "broken_by_action": len(broken_by_action),
        "broken_by_assist": len(broken_by_assist),
        "rescues_by_action": sum(1 for r in forks if r["action_class"] == "ACTION_RESCUE"),
        "rescues_by_assist": sum(1 for r in forks if r["rescue_class"] == "ASSIST_RESCUE"),
        "advantage_distribution": {
            "a_b": {"positive": a_b_pos, "negative": a_b_neg, "mean": sum(a_b_values) / n},
            "a_e": {"positive": a_e_pos, "negative": a_e_neg, "mean": sum(a_e_values) / n},
            "execution_gain": {"positive": eg_pos, "negative": eg_neg, "mean": sum(eg_values) / n},
        },
        "by_action_pair": {
            pair: {
                "n": ps["n"],
                "mean_a_b": ps["a_b_sum"] / ps["n"],
                "mean_a_e": ps["a_e_sum"] / ps["n"],
                "mean_execution_gain": ps["eg_sum"] / ps["n"],
                "base_success": ps["base_success"],
                "action_success": ps["action_success"],
                "assist_success": ps["assist_success"],
                "rescue": ps["rescue"],
                "break": ps["break"],
            }
            for pair, ps in pair_stats.items()
        },
        "cost_adjusted": {
            "mean_model_calls": {
                "base": base_calls_mean,
                "action_only": action_calls_mean,
                "exec_assist": assist_calls_mean,
            },
            "utility_per_call": {
                "base": u_per_call_base,
                "action_only": u_per_call_action,
                "exec_assist": u_per_call_assist,
            },
        },
        "conversion_stages": conversion_stages,
        "broken_success_details": [
            {
                "task_id": r["task_id"],
                "step_id": r["step_id"],
                "base_action": r["base_action"],
                "gov_action": r["gov_action"],
                "u_base": r["u_base"],
                "u_action": r["u_action_only"],
                "u_assist": r["u_exec_assist"],
            }
            for r in broken_by_assist
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
