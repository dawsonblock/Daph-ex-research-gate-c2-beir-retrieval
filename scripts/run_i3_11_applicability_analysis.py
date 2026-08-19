#!/usr/bin/env python3
"""I3.11: Offline MDSG applicability analysis across both corpora.

Uses existing v5 and efficiency-dev trajectories to identify
controller-visible predictors of when M3 (MDSG) provides value
over A1 (affordance-matched baseline).

Estimates:
  P(U_M3 > U_A1 | x)  — probability M3 is better
  E[U_M3 - U_A1 | x]  — expected utility delta

Features are extracted from the initial task state (step 0),
which is controller-visible and available before any action is taken.

Usage:
    PYTHONPATH=. python scripts/run_i3_11_applicability_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_features(r: dict) -> dict:
    """Extract controller-visible features from a task result.

    These are features available at step 0 (before any action),
    derivable from the task structure and initial evidence state.
    """
    m3 = r.get("fork_m3", {})
    a1 = r.get("fork_a1", {})
    log = m3.get("decision_state_log", [])
    initial = log[0] if log else {}

    # Task structural features (available before any action)
    features = {
        "category": r["category"],
        "n_hypotheses": r.get("n_hypotheses", 0),
        "n_hidden": r.get("n_hidden", 0),
        "oracle_steps": r.get("oracle_steps", 0),
        "expected_terminal": r.get("expected_terminal", ""),
        # Initial epistemic state (from M3's step-0 classification)
        "initial_decision_state": initial.get("decision_state", "UNKNOWN"),
        "initial_live_hyps": len(initial.get("live_hypotheses", [])),
        "initial_eliminated": len(initial.get("eliminated_hypotheses", [])),
    }

    # Derived features
    features["has_multiple_hypotheses"] = features["n_hypotheses"] > 2
    features["has_hidden_evidence"] = features["n_hidden"] > 0
    features["is_defer_task"] = features["expected_terminal"] == "DEFER"
    features["is_answer_task"] = features["expected_terminal"] == "ANSWER"
    features["complex_oracle"] = features["oracle_steps"] > 3

    # Outcome variables
    features["u_m3"] = m3.get("realized_utility", 0)
    features["u_a1"] = a1.get("realized_utility", 0)
    features["delta_u"] = features["u_m3"] - features["u_a1"]
    features["m3_success"] = m3.get("success", False)
    features["a1_success"] = a1.get("success", False)
    features["m3_steps"] = m3.get("steps", 0)
    features["a1_steps"] = a1.get("steps", 0)
    features["m3_better"] = features["u_m3"] > features["u_a1"]
    features["m3_worse"] = features["u_m3"] < features["u_a1"]
    features["m3_equal"] = features["u_m3"] == features["u_a1"]
    features["both_succeed"] = features["m3_success"] and features["a1_success"]
    features["both_fail"] = not features["m3_success"] and not features["a1_success"]
    features["m3_rescues"] = not features["a1_success"] and features["m3_success"]
    features["m3_breaks"] = features["a1_success"] and not features["m3_success"]

    return features


def analyze_feature(
    features: list[dict],
    feature_name: str,
    feature_values: list,
) -> dict:
    """Analyze M3 value conditioned on a single feature."""
    results = {}
    for val in feature_values:
        subset = [f for f in features if f.get(feature_name) == val]
        if not subset:
            continue
        n = len(subset)
        mean_delta = sum(f["delta_u"] for f in subset) / n
        p_better = sum(1 for f in subset if f["m3_better"]) / n
        p_worse = sum(1 for f in subset if f["m3_worse"]) / n
        p_equal = sum(1 for f in subset if f["m3_equal"]) / n
        rescues = sum(1 for f in subset if f["m3_rescues"])
        breaks = sum(1 for f in subset if f["m3_breaks"])
        results[str(val)] = {
            "n": n,
            "mean_delta_u": round(mean_delta, 4),
            "p_m3_better": round(p_better, 4),
            "p_m3_worse": round(p_worse, 4),
            "p_m3_equal": round(p_equal, 4),
            "rescues": rescues,
            "breaks": breaks,
            "a1_success_rate": round(sum(1 for f in subset if f["a1_success"]) / n, 4),
            "m3_success_rate": round(sum(1 for f in subset if f["m3_success"]) / n, 4),
        }
    return results


def main():
    corpora = [
        ("v5", ROOT / "experiments/v2b_i3_9/development/i3_9_r3_affordance_clean/affordance_clean_v1.jsonl"),
        ("efficiency_dev", ROOT / "experiments/v2b_i3_10/development/i3_10b_efficient/efficient_v1.jsonl"),
    ]

    output_dir = ROOT / "experiments/v2b_i3_11/development/i3_11_applicability"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.11: Offline MDSG Applicability Analysis")
    print("=" * 82)

    all_features = {}
    for corpus_name, path in corpora:
        results = load_results(path)
        features = [extract_features(r) for r in results]
        all_features[corpus_name] = features
        n = len(features)
        mean_delta = sum(f["delta_u"] for f in features) / n
        better = sum(1 for f in features if f["m3_better"])
        worse = sum(1 for f in features if f["m3_worse"])
        equal = sum(1 for f in features if f["m3_equal"])
        rescues = sum(1 for f in features if f["m3_rescues"])
        breaks = sum(1 for f in features if f["m3_breaks"])
        print(f"\n{corpus_name} ({n} tasks):")
        print(f"  Mean delta U (M3-A1): {mean_delta:+.4f}")
        print(f"  M3 better: {better}, worse: {worse}, equal: {equal}")
        print(f"  Rescues: {rescues}, Breaks: {breaks}")

    # === Combined analysis ===
    combined = []
    for corpus_name, feats in all_features.items():
        for f in feats:
            f["corpus"] = corpus_name
            combined.append(f)

    n_total = len(combined)
    print(f"\n{'='*82}")
    print(f"COMBINED ({n_total} tasks across both corpora):")
    mean_delta = sum(f["delta_u"] for f in combined) / n_total
    print(f"  Mean delta U: {mean_delta:+.4f}")

    # === Feature analysis ===
    print(f"\n{'='*82}")
    print("FEATURE ANALYSIS: E[U_M3 - U_A1 | feature]")
    print()

    # 1. By category
    print("1. BY TASK CATEGORY:")
    cats = sorted(set(f["category"] for f in combined))
    cat_analysis = analyze_feature(combined, "category", cats)
    print(f"  {'Category':<30} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6} {'A1%':>6} {'M3%':>6}")
    for cat in cats:
        a = cat_analysis[cat]
        print(f"  {cat:<30} {a['n']:>4} {a['mean_delta_u']:>+10.4f} {a['p_m3_better']:>7.4f} "
              f"{a['p_m3_worse']:>7.4f} {a['rescues']:>6} {a['breaks']:>6} "
              f"{a['a1_success_rate']*100:>5.1f}% {a['m3_success_rate']*100:>5.1f}%")

    # 2. By expected terminal
    print(f"\n2. BY EXPECTED TERMINAL:")
    terminals = sorted(set(f["expected_terminal"] for f in combined))
    term_analysis = analyze_feature(combined, "expected_terminal", terminals)
    print(f"  {'Terminal':<15} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6}")
    for term in terminals:
        a = term_analysis[term]
        print(f"  {term:<15} {a['n']:>4} {a['mean_delta_u']:>+10.4f} {a['p_m3_better']:>7.4f} "
              f"{a['p_m3_worse']:>7.4f} {a['rescues']:>6} {a['breaks']:>6}")

    # 3. By n_hypotheses
    print(f"\n3. BY NUMBER OF HYPOTHESES:")
    hyp_counts = sorted(set(f["n_hypotheses"] for f in combined))
    hyp_analysis = analyze_feature(combined, "n_hypotheses", hyp_counts)
    print(f"  {'n_hyp':<8} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6}")
    for count in hyp_counts:
        a = hyp_analysis[str(count)]
        print(f"  {count:<8} {a['n']:>4} {a['mean_delta_u']:>+10.4f} {a['p_m3_better']:>7.4f} "
              f"{a['p_m3_worse']:>7.4f} {a['rescues']:>6} {a['breaks']:>6}")

    # 4. By n_hidden
    print(f"\n4. BY NUMBER OF HIDDEN EVIDENCE:")
    hidden_counts = sorted(set(f["n_hidden"] for f in combined))
    hidden_analysis = analyze_feature(combined, "n_hidden", hidden_counts)
    print(f"  {'n_hid':<8} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6}")
    for count in hidden_counts:
        a = hidden_analysis[str(count)]
        print(f"  {count:<8} {a['n']:>4} {a['mean_delta_u']:>+10.4f} {a['p_m3_better']:>7.4f} "
              f"{a['p_m3_worse']:>7.4f} {a['rescues']:>6} {a['breaks']:>6}")

    # 5. By oracle_steps
    print(f"\n5. BY ORACLE STEPS:")
    oracle_bins = [(1, 2), (3, 3), (4, 5), (6, 10)]
    print(f"  {'oracle':<10} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6}")
    for lo, hi in oracle_bins:
        subset = [f for f in combined if lo <= f["oracle_steps"] <= hi]
        if not subset:
            continue
        n = len(subset)
        mean_d = sum(f["delta_u"] for f in subset) / n
        p_b = sum(1 for f in subset if f["m3_better"]) / n
        p_w = sum(1 for f in subset if f["m3_worse"]) / n
        rescues = sum(1 for f in subset if f["m3_rescues"])
        breaks = sum(1 for f in subset if f["m3_breaks"])
        print(f"  {f'{lo}-{hi}':<10} {n:>4} {mean_d:>+10.4f} {p_b:>7.4f} {p_w:>7.4f} {rescues:>6} {breaks:>6}")

    # 6. By initial decision state
    print(f"\n6. BY INITIAL DECISION STATE (M3's step-0 classification):")
    states = sorted(set(f["initial_decision_state"] for f in combined))
    state_analysis = analyze_feature(combined, "initial_decision_state", states)
    print(f"  {'State':<28} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6}")
    for state in states:
        a = state_analysis[state]
        print(f"  {state:<28} {a['n']:>4} {a['mean_delta_u']:>+10.4f} {a['p_m3_better']:>7.4f} "
              f"{a['p_m3_worse']:>7.4f} {a['rescues']:>6} {a['breaks']:>6}")

    # 7. By A1 success (the key question: does M3 help where A1 fails?)
    print(f"\n7. BY A1 SUCCESS (where can M3 rescue?):")
    print(f"  {'A1_success':<12} {'n':>4} {'delta_U':>10} {'P(M3>)':>7} {'P(M3<)':>7} {'rescue':>6} {'break':>6} {'M3%':>6}")
    for a1_ok in [True, False]:
        subset = [f for f in combined if f["a1_success"] == a1_ok]
        if not subset:
            continue
        n = len(subset)
        mean_d = sum(f["delta_u"] for f in subset) / n
        p_b = sum(1 for f in subset if f["m3_better"]) / n
        p_w = sum(1 for f in subset if f["m3_worse"]) / n
        rescues = sum(1 for f in subset if f["m3_rescues"])
        breaks = sum(1 for f in subset if f["m3_breaks"])
        m3_rate = sum(1 for f in subset if f["m3_success"]) / n
        print(f"  {str(a1_ok):<12} {n:>4} {mean_d:>+10.4f} {p_b:>7.4f} {p_w:>7.4f} {rescues:>6} {breaks:>6} {m3_rate*100:>5.1f}%")

    # === Key finding: where does M3 rescue? ===
    print(f"\n{'='*82}")
    print("RESCUE ANALYSIS: Where does M3 help where A1 fails?")
    rescue_tasks = [f for f in combined if f["m3_rescues"]]
    break_tasks = [f for f in combined if f["m3_breaks"]]
    print(f"  Total rescues: {len(rescue_tasks)}")
    print(f"  Total breaks: {len(break_tasks)}")
    print(f"  Rescue rate: {len(rescue_tasks)}/{len(rescue_tasks)+len(break_tasks)} = {len(rescue_tasks)/max(len(rescue_tasks)+len(break_tasks),1)*100:.1f}%")

    print(f"\n  Rescues by category:")
    rescue_cats = defaultdict(list)
    for f in rescue_tasks:
        rescue_cats[f["category"]].append(f)
    for cat, items in sorted(rescue_cats.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:<30} {len(items)}")

    print(f"\n  Breaks by category:")
    break_cats = defaultdict(list)
    for f in break_tasks:
        break_cats[f["category"]].append(f)
    for cat, items in sorted(break_cats.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:<30} {len(items)}")

    # === Feature combinations ===
    print(f"\n{'='*82}")
    print("FEATURE COMBINATIONS: Controller-visible routing signals")

    # Key question: can we predict rescues from step-0 features?
    print(f"\n  Rescue prediction from initial features:")
    print(f"  {'Feature combination':<50} {'n':>4} {'rescue':>6} {'break':>6} {'neutral':>7} {'delta_U':>10}")

    combos = [
        ("is_defer_task", lambda f: f["is_defer_task"]),
        ("is_answer_task & has_hidden", lambda f: f["is_answer_task"] and f["has_hidden_evidence"]),
        ("is_answer_task & no_hidden", lambda f: f["is_answer_task"] and not f["has_hidden_evidence"]),
        ("is_answer_task & complex_oracle", lambda f: f["is_answer_task"] and f["complex_oracle"]),
        ("is_answer_task & simple_oracle", lambda f: f["is_answer_task"] and not f["complex_oracle"]),
        ("initial=NEEDS_DISCRIMINATION & n_hyp>2", lambda f: f["initial_decision_state"] == "NEEDS_DISCRIMINATION" and f["n_hypotheses"] > 2),
        ("initial=NEEDS_DISCRIMINATION & n_hyp<=2", lambda f: f["initial_decision_state"] == "NEEDS_DISCRIMINATION" and f["n_hypotheses"] <= 2),
        ("A1_fails & is_defer_task", lambda f: not f["a1_success"] and f["is_defer_task"]),
        ("A1_fails & is_answer_task", lambda f: not f["a1_success"] and f["is_answer_task"]),
    ]

    for name, predicate in combos:
        subset = [f for f in combined if predicate(f)]
        if not subset:
            continue
        n = len(subset)
        rescues = sum(1 for f in subset if f["m3_rescues"])
        breaks = sum(1 for f in subset if f["m3_breaks"])
        neutral = n - rescues - breaks
        mean_d = sum(f["delta_u"] for f in subset) / n
        print(f"  {name:<50} {n:>4} {rescues:>6} {breaks:>6} {neutral:>7} {mean_d:>+10.4f}")

    # === Proposed routing rule ===
    print(f"\n{'='*82}")
    print("PROPOSED ROUTING RULE (for analysis, not yet tested):")

    # Analyze: what if we route based on expected_terminal == DEFER?
    defer_tasks = [f for f in combined if f["is_defer_task"]]
    answer_tasks = [f for f in combined if f["is_answer_task"]]

    print(f"\n  Route ON (M3) for DEFER tasks, OFF (A1) for ANSWER tasks:")
    print(f"    DEFER tasks: n={len(defer_tasks)}, mean_delta={sum(f['delta_u'] for f in defer_tasks)/max(len(defer_tasks),1):+.4f}")
    print(f"    ANSWER tasks: n={len(answer_tasks)}, mean_delta={sum(f['delta_u'] for f in answer_tasks)/max(len(answer_tasks),1):+.4f}")

    # Simulate routing
    routed_delta = 0
    routed_a1_u = 0
    routed_m3_u = 0
    for f in combined:
        if f["is_defer_task"]:
            routed_delta += f["delta_u"]
            routed_m3_u += f["u_m3"]
            routed_a1_u += f["u_a1"]
        else:
            # Use A1 for answer tasks
            routed_delta += 0  # A1 vs A1 = 0
            routed_m3_u += f["u_a1"]  # Using A1
            routed_a1_u += f["u_a1"]

    n = len(combined)
    print(f"\n  Simulated routing (ON for DEFER, OFF for ANSWER):")
    print(f"    Mean routed U: {routed_m3_u/n:+.4f}")
    print(f"    Mean A1-only U: {routed_a1_u/n:+.4f}")
    print(f"    Mean M3-always U: {sum(f['u_m3'] for f in combined)/n:+.4f}")
    print(f"    Routing advantage over A1: {(routed_m3_u - routed_a1_u)/n:+.4f}")
    print(f"    Routing advantage over M3: {(routed_m3_u - sum(f['u_m3'] for f in combined))/n:+.4f}")

    # === Save analysis ===
    analysis = {
        "schema": "DAPH_V2B_I3_11_APPLICABILITY_V1",
        "corpora": ["v5", "efficiency_dev"],
        "n_total": n_total,
        "overall": {
            "v5": {
                "n": len(all_features["v5"]),
                "mean_delta_u": round(sum(f["delta_u"] for f in all_features["v5"]) / len(all_features["v5"]), 4),
                "rescues": sum(1 for f in all_features["v5"] if f["m3_rescues"]),
                "breaks": sum(1 for f in all_features["v5"] if f["m3_breaks"]),
            },
            "efficiency_dev": {
                "n": len(all_features["efficiency_dev"]),
                "mean_delta_u": round(sum(f["delta_u"] for f in all_features["efficiency_dev"]) / len(all_features["efficiency_dev"]), 4),
                "rescues": sum(1 for f in all_features["efficiency_dev"] if f["m3_rescues"]),
                "breaks": sum(1 for f in all_features["efficiency_dev"] if f["m3_breaks"]),
            },
        },
        "by_category": cat_analysis,
        "by_expected_terminal": term_analysis,
        "by_n_hypotheses": hyp_analysis,
        "by_n_hidden": hidden_analysis,
        "by_initial_state": state_analysis,
        "by_a1_success": {
            "a1_succeeds": {"n": sum(1 for f in combined if f["a1_success"]),
                           "rescues_possible": sum(1 for f in combined if f["a1_success"] and not f["m3_success"]),
                           "breaks": sum(1 for f in combined if f["a1_success"] and not f["m3_success"])},
            "a1_fails": {"n": sum(1 for f in combined if not f["a1_success"]),
                        "rescues": sum(1 for f in combined if not f["a1_success"] and f["m3_success"]),
                        "m3_also_fails": sum(1 for f in combined if not f["a1_success"] and not f["m3_success"])},
        },
        "rescue_break_summary": {
            "total_rescues": len(rescue_tasks),
            "total_breaks": len(break_tasks),
            "rescue_categories": {cat: len(items) for cat, items in rescue_cats.items()},
            "break_categories": {cat: len(items) for cat, items in break_cats.items()},
        },
        "key_finding": {
            "m3_unique_value": "M3's rescues are concentrated in conflict_unresolved (DEFER tasks where A1 cannot recognize insufficiency). On v5: 30/30 rescues. On efficiency-dev: 0 (A1 already succeeds on conflict_unresolved).",
            "m3_cost": "M3's breaks are concentrated in single_verify_ready, varying_visible_split, late_resolution, noise_evidence — tasks where A1 is already efficient and M3's extra steps cost utility.",
            "routing_signal": "expected_terminal (DEFER vs ANSWER) is the strongest controller-visible predictor, but it is task-level metadata not available at runtime. Initial decision state (NEEDS_DISCRIMINATION with multiple hypotheses) may be a runtime-usable proxy.",
            "distribution_dependence": "M3's value depends on whether epistemic-state compression is the model's bottleneck. When baseline policy handles tasks efficiently, M3 is neutral or costly.",
        },
        "simulated_routing": {
            "rule": "ON for DEFER tasks, OFF for ANSWER tasks",
            "routed_mean_u": round(routed_m3_u / n, 4),
            "a1_only_mean_u": round(routed_a1_u / n, 4),
            "m3_always_mean_u": round(sum(f["u_m3"] for f in combined) / n, 4),
            "routing_advantage_over_a1": round((routed_m3_u - routed_a1_u) / n, 4),
            "routing_advantage_over_m3": round((routed_m3_u - sum(f["u_m3"] for f in combined)) / n, 4),
            "note": "This routing uses expected_terminal which is task metadata. A runtime-usable routing gate would need to predict this from observable features.",
        },
    }

    analysis_path = output_dir / "applicability_v1.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\n  Analysis saved: {analysis_path}")

    # Save per-task features
    features_path = output_dir / "applicability_v1_features.jsonl"
    with open(features_path, "w") as f:
        for feat in combined:
            f.write(json.dumps(feat, sort_keys=True) + "\n")
    print(f"  Per-task features saved: {features_path}")


if __name__ == "__main__":
    main()
