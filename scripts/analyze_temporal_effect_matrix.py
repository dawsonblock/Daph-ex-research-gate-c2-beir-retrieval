#!/usr/bin/env python3
"""I3.5-PQ Phase 22: Temporal-effect matrix + V2 decision.

Combines pilot depth data + targeted depth data, then for every
category/action pair computes:
  - Q(d=0), Q(d=1), Q(d=2)
  - Delta_01 = Q(d=1) - Q(d=0)
  - Delta_12 = Q(d=2) - Q(d=1)
  - Pattern: FLAT / MONOTONIC_DECREASE / MONOTONIC_INCREASE / NON_MONOTONIC
  - Bootstrap 95% CI on Delta_02

Then applies the frozen V2 decision rule:
  Build V2 only if ALL of:
    1. >=30% of tested category/action pairs show |Delta_Q| > 5
    2. The effect is reproducible across matched states
    3. V1 fails to reflect those effects
    4. That V1 error causes live-policy harm that I2 does not already prevent

Condition 4 matters most.
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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000,
                  confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    return (float(np.percentile(means, alpha * 100)),
            float(np.percentile(means, (1 - alpha) * 100)))


def classify_pattern(d0: float, d1: float, d2: float, threshold: float = 5.0) -> str:
    """Classify the temporal pattern."""
    delta_01 = d1 - d0
    delta_12 = d2 - d1
    delta_02 = d2 - d0

    if abs(delta_02) < threshold:
        return "FLAT"

    if delta_01 < -threshold and delta_12 < -threshold:
        return "MONOTONIC_DECREASE"
    if delta_01 > threshold and delta_12 > threshold:
        return "MONOTONIC_INCREASE"

    return "NON_MONOTONIC"


def main():
    pilot_path = REPO_ROOT / "experiments/i3_5/pinned_policy_depth/depth_causal_actions_v1.jsonl"
    targeted_path = REPO_ROOT / "experiments/i3_5/pinned_policy_targeted_depth/targeted_depth_actions_v1.jsonl"

    print("Loading data...")
    pilot = load_jsonl(pilot_path)
    targeted = load_jsonl(targeted_path)
    print(f"  Pilot: {len(pilot)} records")
    print(f"  Targeted: {len(targeted)} records")

    all_records = pilot + targeted
    print(f"  Combined: {len(all_records)} records")

    # Group by (category, action, depth)
    by_cat_action_depth = defaultdict(lambda: defaultdict(list))
    # Also group by (task_id, action, depth) for matched-state analysis
    by_task_action_depth = defaultdict(lambda: defaultdict(list))

    for r in all_records:
        cat = r["category"]
        action = r["forced_action"]
        depth = r["depth"]
        util = r["pinned_policy_utility"]
        by_cat_action_depth[(cat, action)][depth].append(util)
        by_task_action_depth[(r["task_id"], action)][depth].append(util)

    # ================================================================
    # Temporal-effect matrix
    # ================================================================
    print("\n" + "=" * 90)
    print("TEMPORAL-EFFECT MATRIX")
    print("=" * 90)
    print()
    print(f"  {'Category':>15s} {'Action':>12s} {'n_d0':>5s} {'Q(d0)':>8s} "
          f"{'n_d1':>5s} {'Q(d1)':>8s} {'n_d2':>5s} {'Q(d2)':>8s} "
          f"{'Delta_02':>10s} {'CI_95%':>20s} {'Pattern':>20s}")
    print("-" * 130)

    results = []
    patterns = defaultdict(int)
    meaningful_effects = 0
    total_pairs = 0

    for (cat, action), by_depth in sorted(by_cat_action_depth.items()):
        if action not in {"RETRIEVE", "VERIFY", "SEARCH_MORE"}:
            continue

        d0s = by_depth.get(0, [])
        d1s = by_depth.get(1, [])
        d2s = by_depth.get(2, [])

        if not d0s or not d1s or not d2s:
            continue

        q0 = sum(d0s) / len(d0s)
        q1 = sum(d1s) / len(d1s)
        q2 = sum(d2s) / len(d2s)

        delta_02 = q2 - q0

        # Bootstrap CI on per-task deltas
        # Find matched tasks (tasks with data at all 3 depths)
        matched_deltas = []
        for tid in sorted(set(r["task_id"] for r in all_records
                              if r["category"] == cat and r["forced_action"] == action)):
            t_d0 = by_task_action_depth.get((tid, action), {}).get(0, [])
            t_d2 = by_task_action_depth.get((tid, action), {}).get(2, [])
            if t_d0 and t_d2:
                matched_deltas.append(t_d2[0] - t_d0[0])

        ci_lo, ci_hi = bootstrap_ci(matched_deltas) if matched_deltas else (0, 0)

        pattern = classify_pattern(q0, q1, q2, threshold=5.0)
        patterns[pattern] += 1
        total_pairs += 1

        if abs(delta_02) > 5:
            meaningful_effects += 1

        print(f"  {cat:>15s} {action:>12s} {len(d0s):5d} {q0:8.2f} "
              f"{len(d1s):5d} {q1:8.2f} {len(d2s):5d} {q2:8.2f} "
              f"{delta_02:+10.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}] {pattern:>20s}")

        results.append({
            "category": cat,
            "action": action,
            "n_d0": len(d0s), "n_d1": len(d1s), "n_d2": len(d2s),
            "q_d0": round(q0, 4), "q_d1": round(q1, 4), "q_d2": round(q2, 4),
            "delta_01": round(q1 - q0, 4),
            "delta_12": round(q2 - q1, 4),
            "delta_02": round(delta_02, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "pattern": pattern,
            "n_matched": len(matched_deltas),
        })

    print()
    print(f"  Pattern distribution:")
    for p, n in sorted(patterns.items()):
        print(f"    {p}: {n}")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Meaningful effects (|Delta_02| > 5): {meaningful_effects} / {total_pairs} "
          f"({meaningful_effects/total_pairs*100:.1f}%)" if total_pairs else "")

    # ================================================================
    # V2 Decision Rule
    # ================================================================
    print("\n" + "=" * 90)
    print("V2 DECISION RULE")
    print("=" * 90)
    print()
    print("  Build V2 only if ALL of:")
    print("    1. >=30% of tested category/action pairs show |Delta_Q| > 5")
    print("    2. The effect is reproducible across matched states")
    print("    3. V1 fails to reflect those effects")
    print("    4. That V1 error causes live-policy harm that I2 does not already prevent")
    print()
    print("  Condition 4 matters most.")
    print()

    pct_meaningful = meaningful_effects / total_pairs * 100 if total_pairs else 0
    condition_1 = pct_meaningful >= 30
    print(f"  Condition 1: {pct_meaningful:.1f}% of pairs show |Delta_Q| > 5 "
          f"(threshold: 30%) -> {'PASS' if condition_1 else 'FAIL'}")

    # Condition 2: reproducible — check if matched CIs exclude 0
    reproducible = 0
    for r in results:
        if abs(r["delta_02"]) > 5:
            if r["ci_95"][0] > 0 or r["ci_95"][1] < 0:
                reproducible += 1
    condition_2 = reproducible >= meaningful_effects * 0.5 if meaningful_effects > 0 else False
    print(f"  Condition 2: {reproducible} / {meaningful_effects} meaningful effects "
          f"have CIs excluding 0 -> {'PASS' if condition_2 else 'FAIL'}")

    # Condition 3: V1 fails to reflect those effects
    # We check this by loading V1 and predicting Q at each depth
    # For now, flag it as needing verification
    print(f"  Condition 3: V1 prediction check (requires model loading) -> PENDING")

    # Condition 4: live-policy harm that I2 doesn't prevent
    # From Phase 21 results: I2 has 100% success, 88/220 trajectories with 3 RETRIEVEs
    # but those all succeed. The 21 utility-point loss is from step costs, not failures.
    print(f"  Condition 4: I2 already prevents live-policy harm (100% success)")
    print(f"    The 88/220 3-RETRIEVE trajectories succeed but lose ~21 utility points")
    print(f"    from step costs. This is waste, not failure.")
    print(f"    -> V2 would need to eliminate this waste to justify added complexity")

    # ================================================================
    # Overall verdict
    # ================================================================
    print()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print()

    if not condition_1:
        print(f"  V2 NOT JUSTIFIED.")
        print(f"  Only {pct_meaningful:.1f}% of pairs show meaningful temporal effects.")
        print(f"  The threshold is 30%.")
        print()
        print(f"  The primary target (ol_retrieve RETRIEVE) is FLAT at 93.53.")
        print(f"  The I2 interface already prevents over-retrieval.")
        print(f"  Adding V2 complexity would not solve a remaining failure.")
        print()
        print(f"  RECOMMENDATION: Freeze DAPH_EXECUTIVE_V1 (QCAUSAL_V1 + I2)")
        print(f"  and proceed to the confirmation benchmark.")
    elif not condition_2:
        print(f"  V2 NOT JUSTIFIED.")
        print(f"  Temporal effects exist but are not reproducible across matched states.")
        print()
        print(f"  RECOMMENDATION: Freeze DAPH_EXECUTIVE_V1 and proceed to confirmation.")
    else:
        print(f"  V2 MAY BE JUSTIFIED — conditions 1 and 2 pass.")
        print(f"  Must verify conditions 3 and 4 before building V2.")
        print()
        print(f"  Next steps:")
        print(f"    1. Load V1 and check if it predicts flat Q across depths")
        print(f"    2. Check if I2's 3-RETRIEVE waste can be eliminated by V2")
        print(f"    3. If both pass, train V2 on D_V1 + D_depth")

    # Save results
    output = {
        "temporal_matrix": results,
        "pattern_distribution": dict(patterns),
        "n_meaningful_effects": meaningful_effects,
        "n_total_pairs": total_pairs,
        "pct_meaningful": round(pct_meaningful, 2),
        "condition_1_pass": condition_1,
        "condition_2_pass": condition_2,
        "verdict": "V2_NOT_JUSTIFIED" if not condition_1 or not condition_2 else "V2_MAY_BE_JUSTIFIED",
    }
    output_path = REPO_ROOT / "experiments/i3_5/executive_v1/temporal_effect_matrix_v1.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
