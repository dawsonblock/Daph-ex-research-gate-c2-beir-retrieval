#!/usr/bin/env python3
"""R14-C: Complete the preregistered λ-scan and deployment-view analysis.

CORRECTED VERSION (Addendum 1):
- Uses 3-run mean latency L̄(s,a) = (1/3) Σ_r L(s,a,r), not seed-42 only
- Threshold analysis: correct direction for entropy/uncertainty, equivalence-class splits
- Reports full threshold accuracy/latency frontier, not single "best"

Uses the frozen 810 cells from r14_c_executions.jsonl.
No new inference. No protocol modification.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

R14_DIR = PROJECT_ROOT / "experiments/daph_x/r14"
EXEC_PATH = R14_DIR / "r14_c_executions.jsonl"

SEEDS = [42, 123, 2024]
OPERATORS = ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]


def load_results():
    with open(EXEC_PATH) as f:
        return [json.loads(line) for line in f]


def build_replicated(results):
    """Build per-(checkpoint, operator) records averaged across 3 seeds.

    Q̄(s,a) = correctness (identical across seeds under temp=0)
    L̄(s,a) = (1/3) Σ_r L(s,a,r)  -- 3-run mean wall time
    """
    by_ca = defaultdict(lambda: defaultdict(list))
    for r in results:
        op = r.get("operator_id_canonical", r["operator_id"])
        by_ca[r["checkpoint_id"]][op].append(r)

    replicated = {}
    for cp_id, ops_map in by_ca.items():
        replicated[cp_id] = {}
        for op_id, records in ops_map.items():
            assert len(records) == 3, f"Expected 3 seeds for {cp_id}/{op_id}, got {len(records)}"
            q = 1.0 if records[0].get("correct") else 0.0  # identical across seeds
            walls = [r.get("wall_ms_observed", 0) for r in records]
            l_mean = sum(walls) / len(walls) / 1000.0  # seconds
            l_median = sorted(walls)[len(walls) // 2] / 1000.0
            replicated[cp_id][op_id] = {
                "q": q,
                "l_mean": l_mean,
                "l_median": l_median,
                "walls_ms": walls,
            }
    assert len(replicated) == 90, f"Expected 90 checkpoints, got {len(replicated)}"
    return replicated


def lambda_scan(replicated):
    """For each λ, compute J_oracle(λ) and J_best-fixed(λ) using 3-run mean latency.

    J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a)
    J_oracle(λ) = (1/N) Σ_s max_a J_λ(s,a)
    J_best-fixed(λ) = max_a (1/N) Σ_s J_λ(s,a)
    routing_headroom(λ) = J_oracle(λ) - J_best-fixed(λ)
    """
    per_cp = []
    for cp_id, ops in replicated.items():
        row = {}
        for op_id in OPERATORS:
            r = ops[op_id]
            row[op_id] = {"q": r["q"], "l": r["l_mean"]}
        per_cp.append(row)

    N = len(per_cp)
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.054, 0.057, 0.06, 0.08, 0.1, 0.2, 0.5, 1.0]

    results = []
    for lam in lambdas:
        j_oracle = sum(max(row[op]["q"] - lam * row[op]["l"] for op in OPERATORS) for row in per_cp) / N
        j_fixed = {op: sum(row[op]["q"] - lam * row[op]["l"] for row in per_cp) / N for op in OPERATORS}
        best_op = max(j_fixed, key=j_fixed.get)
        j_best_fixed = j_fixed[best_op]
        results.append({
            "lambda": lam,
            "j_oracle": j_oracle,
            "j_best_fixed": j_best_fixed,
            "best_fixed_op": best_op,
            "routing_headroom": j_oracle - j_best_fixed,
            "j_stop": j_fixed["STOP"],
            "j_re2": j_fixed["OPT_RE2"],
            "j_cot": j_fixed["OPT_COT_REFLECT"],
        })
    return results


def deployment_view(replicated):
    """Deployment view using 3-run mean latency."""
    # Fixed policies
    fixed = {}
    for op in OPERATORS:
        correct = sum(1 for cp_id, ops in replicated.items() if ops[op]["q"] > 0.5)
        mean_lat = sum(ops[op]["l_mean"] for ops in replicated.values()) / 90
        fixed[op] = {"accuracy": correct / 90, "mean_latency_s": mean_lat, "n_correct": correct}

    # STOP→COT oracle
    oracle_sc = {"correct": 0, "latency_s": 0.0}
    for cp_id, ops in replicated.items():
        if ops["STOP"]["q"] > 0.5:
            oracle_sc["correct"] += 1
        else:
            if ops["OPT_COT_REFLECT"]["q"] > 0.5:
                oracle_sc["correct"] += 1
            oracle_sc["latency_s"] += ops["OPT_COT_REFLECT"]["l_mean"]
    oracle_sc["accuracy"] = oracle_sc["correct"] / 90
    oracle_sc["mean_latency_s"] = oracle_sc["latency_s"] / 90

    # STOP→RE2 oracle
    oracle_sr = {"correct": 0, "latency_s": 0.0}
    for cp_id, ops in replicated.items():
        if ops["STOP"]["q"] > 0.5:
            oracle_sr["correct"] += 1
        else:
            if ops["OPT_RE2"]["q"] > 0.5:
                oracle_sr["correct"] += 1
            oracle_sr["latency_s"] += ops["OPT_RE2"]["l_mean"]
    oracle_sr["accuracy"] = oracle_sr["correct"] / 90
    oracle_sr["mean_latency_s"] = oracle_sr["latency_s"] / 90

    # 3-way oracle (cheapest correct action: STOP → RE2 → COT)
    oracle_3way = {"correct": 0, "latency_s": 0.0}
    for cp_id, ops in replicated.items():
        if ops["STOP"]["q"] > 0.5:
            oracle_3way["correct"] += 1
        elif ops["OPT_RE2"]["q"] > 0.5:
            oracle_3way["correct"] += 1
            oracle_3way["latency_s"] += ops["OPT_RE2"]["l_mean"]
        elif ops["OPT_COT_REFLECT"]["q"] > 0.5:
            oracle_3way["correct"] += 1
            oracle_3way["latency_s"] += ops["OPT_COT_REFLECT"]["l_mean"]
    oracle_3way["accuracy"] = oracle_3way["correct"] / 90
    oracle_3way["mean_latency_s"] = oracle_3way["latency_s"] / 90

    return {
        "fixed": fixed,
        "oracle_stop_to_cot": oracle_sc,
        "oracle_stop_to_re2": oracle_sr,
        "oracle_3way": oracle_3way,
    }


def threshold_frontier(replicated, checkpoints_path):
    """Full threshold accuracy/latency frontier using equivalence-class splits.

    For each feature, sweep over unique values as thresholds.
    Direction is feature-specific:
    - p_top1, margin, agreement_rate: escalate if BELOW threshold (low confidence)
    - entropy, uncertainty_current: escalate if ABOVE threshold (high uncertainty)
    - n_unique_answers: escalate if ABOVE threshold (more disagreement)

    Reports the full frontier, not a single "best".
    """
    with open(checkpoints_path) as f:
        checkpoints = {cp["checkpoint_id"]: cp for cp in (json.loads(l) for l in f)}

    features = {}
    for cp_id, cp in checkpoints.items():
        of = cp["runtime_state"].get("observable_features", {})
        features[cp_id] = of

    # Feature direction: True = escalate if value > threshold, False = escalate if value < threshold
    feature_config = {
        "p_top1": {"escalate_above": False, "label": "escalate if p_top1 < t"},
        "agreement_rate": {"escalate_above": False, "label": "escalate if agreement_rate < t"},
        "margin": {"escalate_above": False, "label": "escalate if margin < t"},
        "entropy": {"escalate_above": True, "label": "escalate if entropy > t"},
        "uncertainty_current": {"escalate_above": True, "label": "escalate if uncertainty > t"},
        "n_unique_answers": {"escalate_above": True, "label": "escalate if n_unique > t"},
    }

    cot_mean_lat = sum(ops["OPT_COT_REFLECT"]["l_mean"] for ops in replicated.values()) / 90

    all_frontiers = {}
    for feat_name, config in feature_config.items():
        escalate_above = config["escalate_above"]

        # Get unique values, sorted (equivalence classes)
        vals = sorted(set(features[cp_id].get(feat_name) for cp_id in replicated
                          if features[cp_id].get(feat_name) is not None))
        if not vals:
            continue

        frontier = []
        # Option 1: never escalate (always STOP)
        correct = sum(1 for cp_id, ops in replicated.items() if ops["STOP"]["q"] > 0.5)
        frontier.append({
            "threshold": None, "n_escalated": 0,
            "accuracy": correct / 90, "mean_latency_s": 0.0, "n_correct": correct,
            "policy": "never escalate (always STOP)",
        })

        # For each unique value as threshold, escalate the appropriate side
        for i, thresh in enumerate(vals):
            correct = 0
            latency_s = 0.0
            n_esc = 0
            for cp_id, ops in replicated.items():
                v = features[cp_id].get(feat_name)
                if v is None:
                    v = float("inf") if escalate_above else float("-inf")
                should_escalate = (v > thresh) if escalate_above else (v < thresh)
                if should_escalate:
                    n_esc += 1
                    if ops["OPT_COT_REFLECT"]["q"] > 0.5:
                        correct += 1
                    latency_s += ops["OPT_COT_REFLECT"]["l_mean"]
                else:
                    if ops["STOP"]["q"] > 0.5:
                        correct += 1
            frontier.append({
                "threshold": thresh, "n_escalated": n_esc,
                "accuracy": correct / 90, "mean_latency_s": latency_s / 90,
                "n_correct": correct,
                "policy": f"{config['label']}",
            })

        # Option last: always escalate (always COT)
        correct = sum(1 for cp_id, ops in replicated.items() if ops["OPT_COT_REFLECT"]["q"] > 0.5)
        frontier.append({
            "threshold": None, "n_escalated": 90,
            "accuracy": correct / 90, "mean_latency_s": cot_mean_lat,
            "n_correct": correct,
            "policy": "always escalate (always COT)",
        })

        # Pareto filter: keep only points on the accuracy/latency frontier
        pareto = []
        for p in frontier:
            dominated = False
            for q in frontier:
                if q is p:
                    continue
                if q["accuracy"] >= p["accuracy"] and q["mean_latency_s"] <= p["mean_latency_s"] and \
                   (q["accuracy"] > p["accuracy"] or q["mean_latency_s"] < p["mean_latency_s"]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        all_frontiers[feat_name] = {
            "label": config["label"],
            "n_thresholds": len(vals),
            "pareto_frontier": pareto,
            "all_points": frontier,
        }

    return all_frontiers


def main():
    results = load_results()
    replicated = build_replicated(results)

    # Verify 3-run mean latencies
    print("=" * 80)
    print("R14-C CORRECTED ANALYSIS: 3-RUN MEAN LATENCY")
    print("=" * 80)
    print()
    print("L̄(s,a) = (1/3) Σ_r L(s,a,r)  -- averaged across seeds 42, 123, 2024")
    print()

    for op in OPERATORS:
        lats = [ops[op]["l_mean"] for ops in replicated.values()]
        walls_all = []
        for ops in replicated.values():
            walls_all.extend(ops[op]["walls_ms"])
        print(f"  {op}:")
        print(f"    mean L̄ = {sum(lats)/90:.3f}s")
        print(f"    median L̄ = {sorted(lats)[45]:.3f}s")
        print(f"    p90 L̄ = {sorted(lats)[81]:.3f}s")
        print(f"    p95 L̄ = {sorted(lats)[85]:.3f}s")
        print(f"    max L̄ = {max(lats):.3f}s")
        print()

    print("=" * 80)
    print("R14-C CORRECTED ANALYSIS: λ-SCAN (3-run mean latency)")
    print("=" * 80)
    print()
    print("J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a)")
    print()

    lam_results = lambda_scan(replicated)
    print(f"{'λ':>8} {'J_oracle':>10} {'J_best_fix':>10} {'best_op':>15} {'headroom':>10} {'J_STOP':>8} {'J_RE2':>8} {'J_COT':>8}")
    print("-" * 90)
    for r in lam_results:
        print(f"{r['lambda']:>8.4f} {r['j_oracle']:>10.4f} {r['j_best_fixed']:>10.4f} {r['best_fixed_op']:>15} {r['routing_headroom']:>+10.4f} {r['j_stop']:>8.4f} {r['j_re2']:>8.4f} {r['j_cot']:>8.4f}")

    # Find RE2 best-fixed interval
    re2_best = [r for r in lam_results if r["best_fixed_op"] == "OPT_RE2"]
    if re2_best:
        print(f"\n  RE2 is best fixed at λ = {[r['lambda'] for r in re2_best]}")

    print()
    print("=" * 80)
    print("R14-C CORRECTED ANALYSIS: DEPLOYMENT VIEW (3-run mean latency)")
    print("=" * 80)
    print()

    dep = deployment_view(replicated)

    print("Fixed policies:")
    print(f"  {'Policy':<25} {'Accuracy':>8} {'Mean lat (s)':>13} {'N correct':>10}")
    print(f"  {'-'*60}")
    for op in OPERATORS:
        f = dep["fixed"][op]
        print(f"  {op:<25} {f['accuracy']:>8.3f} {f['mean_latency_s']:>13.3f} {f['n_correct']:>10}")

    print()
    print("Oracle policies:")
    print(f"  {'Policy':<25} {'Accuracy':>8} {'Mean lat (s)':>13} {'N correct':>10}")
    print(f"  {'-'*60}")
    for name, key in [("STOP→COT oracle", "oracle_stop_to_cot"),
                       ("STOP→RE2 oracle", "oracle_stop_to_re2"),
                       ("3-way oracle", "oracle_3way")]:
        o = dep[key]
        print(f"  {name:<25} {o['accuracy']:>8.3f} {o['mean_latency_s']:>13.3f} {o['correct']:>10}")

    # Latency savings
    cot_lat = dep["fixed"]["OPT_COT_REFLECT"]["mean_latency_s"]
    cot_acc = dep["fixed"]["OPT_COT_REFLECT"]["accuracy"]
    sc_lat = dep["oracle_stop_to_cot"]["mean_latency_s"]
    sc_acc = dep["oracle_stop_to_cot"]["accuracy"]
    ow_lat = dep["oracle_3way"]["mean_latency_s"]
    ow_acc = dep["oracle_3way"]["accuracy"]

    print()
    print("Latency savings vs always-COT:")
    print(f"  STOP→COT oracle: {(1-sc_lat/cot_lat)*100:.1f}% saving, {sc_acc-cot_acc:+.1f}pp accuracy")
    print(f"  3-way oracle:    {(1-ow_lat/cot_lat)*100:.1f}% saving, {ow_acc-cot_acc:+.1f}pp accuracy")

    print()
    print("=" * 80)
    print("R14-C CORRECTED ANALYSIS: THRESHOLD FRONTIER (DEV EVIDENCE ONLY)")
    print("=" * 80)
    print()
    print("WARNING: Tuned on same 90 checkpoints used for evaluation.")
    print("Confirmation performance must be evaluated on held-out tasks (R15-A).")
    print()

    checkpoints_path = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
    frontiers = threshold_frontier(replicated, checkpoints_path)

    for feat_name, data in frontiers.items():
        print(f"  {feat_name}: {data['label']}")
        print(f"    Pareto frontier ({len(data['pareto_frontier'])} points):")
        print(f"    {'threshold':>10} {'n_esc':>6} {'accuracy':>8} {'lat(s)':>8} {'n_correct':>10}")
        for p in data["pareto_frontier"]:
            t = f"{p['threshold']:.4f}" if p['threshold'] is not None else "  N/A"
            print(f"    {t:>10} {p['n_escalated']:>6} {p['accuracy']:>8.3f} {p['mean_latency_s']:>8.3f} {p['n_correct']:>10}")
        print()

    # Save full analysis
    output = {
        "experiment": "R14-C",
        "analysis": "corrected_lambda_scan_and_deployment",
        "corrections": [
            "3-run mean latency L̄(s,a) = (1/3) Σ_r L(s,a,r) instead of seed-42 only",
            "threshold direction corrected for entropy/uncertainty (escalate if ABOVE)",
            "threshold splits use equivalence classes of identical values",
            "full Pareto frontier reported instead of single best",
            "λ grid refined around RE2 best-fixed interval (~0.054-0.057)",
        ],
        "n_checkpoints": 90,
        "lambda_scan": lam_results,
        "deployment": dep,
        "threshold_frontiers_dev_only": {
            feat: {
                "label": data["label"],
                "pareto_frontier": data["pareto_frontier"],
            }
            for feat, data in frontiers.items()
        },
    }
    out_path = R14_DIR / "r14_c_lambda_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Full analysis saved to {out_path}")


if __name__ == "__main__":
    main()
