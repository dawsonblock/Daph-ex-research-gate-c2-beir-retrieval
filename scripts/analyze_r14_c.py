#!/usr/bin/env python3
"""R14-C: Complete the preregistered λ-scan and deployment-view analysis.

Uses the frozen 810 cells from r14_c_executions.jsonl.
No new inference. No protocol modification.

Produces:
1. λ-scan: J_oracle(λ) vs J_best-fixed(λ) and routing headroom
2. Deployment view: min latency at ≥85%/88%/90% accuracy
3. Simple threshold→COT baselines (DEVELOPMENT EVIDENCE ONLY — same 90 checkpoints)
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


def load_results():
    with open(EXEC_PATH) as f:
        return [json.loads(line) for line in f]


def build_by_sc(results):
    """Group by (seed, checkpoint_id) -> {operator: record}."""
    by_sc = defaultdict(dict)
    for r in results:
        op = r.get("operator_id_canonical", r["operator_id"])
        by_sc[(r["seed"], r["checkpoint_id"])][op] = r
    return by_sc


def get_unique_90(by_sc):
    """Extract unique 90 checkpoints (seed 42, since all seeds identical under temp=0)."""
    unique = {}
    for (seed, cp_id), ops in by_sc.items():
        if seed == 42:
            unique[cp_id] = ops
    assert len(unique) == 90, f"Expected 90, got {len(unique)}"
    return unique


def lambda_scan(unique):
    """For each λ, compute J_oracle(λ) and J_best-fixed(λ).

    J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a)
    where Q̄ = correctness (0/1), L̄ = wall_ms in seconds.

    J_oracle(λ) = (1/N) Σ_s max_a J_λ(s,a)
    J_best-fixed(λ) = max_a (1/N) Σ_s J_λ(s,a)
    routing_headroom(λ) = J_oracle(λ) - J_best-fixed(λ)
    """
    operators = ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]
    # Build per-checkpoint Q and L arrays
    per_cp = []
    for cp_id, ops in unique.items():
        row = {}
        for op_id in operators:
            r = ops.get(op_id, {})
            q = 1.0 if r.get("correct") else 0.0
            l = r.get("wall_ms_observed", 0) / 1000.0  # seconds
            row[op_id] = {"q": q, "l": l}
        per_cp.append(row)

    N = len(per_cp)

    # λ range: 0 to 0.5 (units: accuracy points per second of latency)
    # At λ=0: pure accuracy
    # At λ=0.01: 1 second of latency is worth 1pp of accuracy
    # At λ=0.1: 1 second of latency is worth 10pp of accuracy
    # At λ=1.0: 1 second of latency is worth 100pp of accuracy (very latency-sensitive)
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []
    for lam in lambdas:
        # J_oracle: per-checkpoint best
        j_oracle = 0.0
        for row in per_cp:
            best_j = max(row[op]["q"] - lam * row[op]["l"] for op in operators)
            j_oracle += best_j
        j_oracle /= N

        # J_best-fixed: best single action averaged
        j_fixed = {}
        for op in operators:
            j = sum(row[op]["q"] - lam * row[op]["l"] for row in per_cp) / N
            j_fixed[op] = j
        best_op = max(j_fixed, key=j_fixed.get)
        j_best_fixed = j_fixed[best_op]

        headroom = j_oracle - j_best_fixed
        results.append({
            "lambda": lam,
            "j_oracle": j_oracle,
            "j_best_fixed": j_best_fixed,
            "best_fixed_op": best_op,
            "routing_headroom": headroom,
            "j_stop": j_fixed["STOP"],
            "j_re2": j_fixed["OPT_RE2"],
            "j_cot": j_fixed["OPT_COT_REFLECT"],
        })

    return results


def deployment_view(unique):
    """For each target accuracy, find minimum-latency policy.

    Policies:
    - Always STOP
    - Always RE2
    - Always COT
    - STOP→COT oracle (knows when STOP is wrong)
    - STOP→RE2 oracle (knows when STOP is wrong)
    - 3-way oracle
    - Simple threshold on observable features (DEV EVIDENCE ONLY)
    """
    operators = ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]

    # Fixed policies
    fixed = {}
    for op in operators:
        correct = sum(1 for cp_id, ops in unique.items() if ops.get(op, {}).get("correct"))
        walls = [ops.get(op, {}).get("wall_ms_observed", 0) for cp_id, ops in unique.items()]
        fixed[op] = {
            "accuracy": correct / 90,
            "mean_latency_s": sum(walls) / 90 / 1000,
            "n_correct": correct,
        }

    # STOP→COT oracle
    oracle_sc = {"correct": 0, "latency_ms": 0}
    for cp_id, ops in unique.items():
        stop_c = ops.get("STOP", {}).get("correct", False)
        cot_c = ops.get("OPT_COT_REFLECT", {}).get("correct", False)
        cot_wall = ops.get("OPT_COT_REFLECT", {}).get("wall_ms_observed", 0)
        if stop_c:
            oracle_sc["correct"] += 1
            oracle_sc["latency_ms"] += 0
        else:
            if cot_c:
                oracle_sc["correct"] += 1
            oracle_sc["latency_ms"] += cot_wall
    oracle_sc["accuracy"] = oracle_sc["correct"] / 90
    oracle_sc["mean_latency_s"] = oracle_sc["latency_ms"] / 90 / 1000

    # STOP→RE2 oracle
    oracle_sr = {"correct": 0, "latency_ms": 0}
    for cp_id, ops in unique.items():
        stop_c = ops.get("STOP", {}).get("correct", False)
        re2_c = ops.get("OPT_RE2", {}).get("correct", False)
        re2_wall = ops.get("OPT_RE2", {}).get("wall_ms_observed", 0)
        if stop_c:
            oracle_sr["correct"] += 1
        else:
            if re2_c:
                oracle_sr["correct"] += 1
            oracle_sr["latency_ms"] += re2_wall
    oracle_sr["accuracy"] = oracle_sr["correct"] / 90
    oracle_sr["mean_latency_s"] = oracle_sr["latency_ms"] / 90 / 1000

    # 3-way oracle
    oracle_3way = {"correct": 0, "latency_ms": 0}
    # For 3-way oracle, we need to pick the cheapest action that is correct
    # If STOP correct: use STOP (0ms)
    # Elif RE2 correct: use RE2
    # Elif COT correct: use COT
    # Else: use STOP (cheapest wrong answer)
    for cp_id, ops in unique.items():
        stop_c = ops.get("STOP", {}).get("correct", False)
        re2_c = ops.get("OPT_RE2", {}).get("correct", False)
        cot_c = ops.get("OPT_COT_REFLECT", {}).get("correct", False)
        re2_wall = ops.get("OPT_RE2", {}).get("wall_ms_observed", 0)
        cot_wall = ops.get("OPT_COT_REFLECT", {}).get("wall_ms_observed", 0)

        if stop_c:
            oracle_3way["correct"] += 1
            # latency = 0
        elif re2_c:
            oracle_3way["correct"] += 1
            oracle_3way["latency_ms"] += re2_wall
        elif cot_c:
            oracle_3way["correct"] += 1
            oracle_3way["latency_ms"] += cot_wall
        else:
            # all wrong, use cheapest
            pass
    oracle_3way["accuracy"] = oracle_3way["correct"] / 90
    oracle_3way["mean_latency_s"] = oracle_3way["latency_ms"] / 90 / 1000

    # Threshold baselines (DEV EVIDENCE ONLY)
    # Load checkpoint features
    checkpoints_path = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
    with open(checkpoints_path) as f:
        checkpoints = {cp["checkpoint_id"]: cp for cp in (json.loads(l) for l in f)}

    # Extract features per checkpoint
    features = {}
    for cp_id, cp in checkpoints.items():
        rs = cp["runtime_state"]
        of = rs.get("observable_features", {})
        features[cp_id] = {
            "p_top1": of.get("p_top1"),
            "p_top2": of.get("p_top2"),
            "margin": of.get("margin"),
            "entropy": of.get("entropy"),
            "n_unique": of.get("n_unique"),
            "agreement_rate": of.get("agreement_rate"),
            "uncertainty_current": of.get("uncertainty_current"),
            "uncertainty_ema": of.get("uncertainty_ema"),
            "k": rs.get("k"),
        }

    # For each feature, find the best threshold that separates STOP-correct from STOP-wrong
    # Then evaluate: if feature indicates STOP sufficient, use STOP; else use COT
    threshold_features = ["p_top1", "margin", "agreement_rate", "entropy", "uncertainty_current"]
    threshold_results = {}

    for feat_name in threshold_features:
        vals = [(features[cp_id][feat_name], cp_id) for cp_id in unique
                if features[cp_id][feat_name] is not None]
        if not vals:
            continue
        vals.sort()

        best = None
        for i in range(len(vals) + 1):
            # Split: first i are "escalate" (low confidence), rest are "STOP"
            # For p_top1, margin, agreement_rate: HIGH = confident = STOP
            # For entropy, uncertainty: LOW = confident = STOP
            escalate_cps = set(cp_id for _, cp_id in vals[:i])
            correct = 0
            latency_ms = 0
            for cp_id, ops in unique.items():
                if cp_id in escalate_cps:
                    # Use COT
                    if ops.get("OPT_COT_REFLECT", {}).get("correct"):
                        correct += 1
                    latency_ms += ops.get("OPT_COT_REFLECT", {}).get("wall_ms_observed", 0)
                else:
                    # Use STOP
                    if ops.get("STOP", {}).get("correct"):
                        correct += 1
            acc = correct / 90
            lat = latency_ms / 90 / 1000
            if best is None or acc > best["accuracy"] or (acc == best["accuracy"] and lat < best["mean_latency_s"]):
                threshold_val = vals[i][0] if i < len(vals) else float("inf")
                best = {
                    "feature": feat_name,
                    "threshold": threshold_val,
                    "direction": "escalate_if_below",
                    "accuracy": acc,
                    "mean_latency_s": lat,
                    "n_escalated": len(escalate_cps),
                    "n_correct": correct,
                }
        threshold_results[feat_name] = best

    return {
        "fixed": fixed,
        "oracle_stop_to_cot": oracle_sc,
        "oracle_stop_to_re2": oracle_sr,
        "oracle_3way": oracle_3way,
        "threshold_baselines_dev_only": threshold_results,
    }


def main():
    results = load_results()
    by_sc = build_by_sc(results)
    unique = get_unique_90(by_sc)

    print("=" * 80)
    print("R14-C PREREGISTERED ANALYSIS: λ-SCAN")
    print("=" * 80)
    print()
    print("J_λ(s,a) = Q̄(s,a) - λ_L · L̄(s,a)")
    print("J_oracle(λ) = (1/N) Σ_s max_a J_λ(s,a)")
    print("J_best-fixed(λ) = max_a (1/N) Σ_s J_λ(s,a)")
    print("routing_headroom = J_oracle - J_best-fixed")
    print()
    print("Units: Q̄ = correctness (0/1), L̄ = wall seconds, λ = accuracy points per second")
    print()

    lam_results = lambda_scan(unique)
    print(f"{'λ':>8} {'J_oracle':>10} {'J_best_fix':>10} {'best_op':>15} {'headroom':>10} {'J_STOP':>8} {'J_RE2':>8} {'J_COT':>8}")
    print("-" * 90)
    for r in lam_results:
        print(f"{r['lambda']:>8.4f} {r['j_oracle']:>10.4f} {r['j_best_fixed']:>10.4f} {r['best_fixed_op']:>15} {r['routing_headroom']:>+10.4f} {r['j_stop']:>8.4f} {r['j_re2']:>8.4f} {r['j_cot']:>8.4f}")

    print()
    print("Interpretation:")
    print("  λ=0:    pure accuracy. Oracle = 0.933, best fixed = COT at 0.900. Headroom = +3.3pp")
    print("  λ=0.01: 1s latency costs 1pp accuracy. COT still best fixed if 7.6s < 41pp gap to STOP.")
    print("  λ=0.1:  1s latency costs 10pp. COT pays 76pp in latency for 41pp in accuracy → STOP wins.")
    print("  The crossover where best-fixed switches from COT to STOP is the key deployment parameter.")

    print()
    print("=" * 80)
    print("R14-C PREREGISTERED ANALYSIS: DEPLOYMENT VIEW")
    print("=" * 80)
    print()

    dep = deployment_view(unique)

    print("Fixed policies:")
    print(f"  {'Policy':<25} {'Accuracy':>8} {'Mean lat (s)':>13} {'N correct':>10}")
    print(f"  {'-'*60}")
    for op in ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]:
        f = dep["fixed"][op]
        print(f"  {op:<25} {f['accuracy']:>8.3f} {f['mean_latency_s']:>13.2f} {f['n_correct']:>10}")

    print()
    print("Oracle policies:")
    print(f"  {'Policy':<25} {'Accuracy':>8} {'Mean lat (s)':>13} {'N correct':>10}")
    print(f"  {'-'*60}")
    for name, key in [("STOP→COT oracle", "oracle_stop_to_cot"),
                       ("STOP→RE2 oracle", "oracle_stop_to_re2"),
                       ("3-way oracle", "oracle_3way")]:
        o = dep[key]
        print(f"  {name:<25} {o['accuracy']:>8.3f} {o['mean_latency_s']:>13.2f} {o['correct']:>10}")

    print()
    print("Threshold baselines (DEV EVIDENCE ONLY — tuned on same 90 checkpoints):")
    print(f"  {'Feature':<25} {'Threshold':>10} {'Accuracy':>8} {'Mean lat (s)':>13} {'N esc':>6}")
    print(f"  {'-'*65}")
    for feat_name, t in sorted(dep["threshold_baselines_dev_only"].items(),
                                key=lambda x: -x[1]["accuracy"]):
        print(f"  {feat_name:<25} {t['threshold']:>10.4f} {t['accuracy']:>8.3f} {t['mean_latency_s']:>13.2f} {t['n_escalated']:>6}")

    print()
    print("Deployment targets (min latency at ≥ target accuracy):")
    print(f"  {'Target':>8} {'Best policy':<30} {'Accuracy':>8} {'Mean lat (s)':>13}")
    print(f"  {'-'*62}")

    all_policies = []
    for op in ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]:
        f = dep["fixed"][op]
        all_policies.append((f"Always {op}", f["accuracy"], f["mean_latency_s"]))
    for name, key in [("STOP→COT oracle", "oracle_stop_to_cot"),
                       ("STOP→RE2 oracle", "oracle_stop_to_re2"),
                       ("3-way oracle", "oracle_3way")]:
        o = dep[key]
        all_policies.append((name, o["accuracy"], o["mean_latency_s"]))
    for feat_name, t in dep["threshold_baselines_dev_only"].items():
        all_policies.append((f"Threshold({feat_name}) [DEV]", t["accuracy"], t["mean_latency_s"]))

    for target in [0.85, 0.88, 0.90]:
        feasible = [(name, acc, lat) for name, acc, lat in all_policies if acc >= target]
        if feasible:
            best = min(feasible, key=lambda x: x[2])
            print(f"  {target:>7.1%} {best[0]:<30} {best[1]:>8.3f} {best[2]:>13.2f}")
        else:
            print(f"  {target:>7.1%} {'INFEASIBLE':<30} {'N/A':>8} {'N/A':>13}")

    # Save full analysis as JSON
    output = {
        "experiment": "R14-C",
        "analysis": "preregistered_lambda_scan_and_deployment",
        "n_checkpoints": 90,
        "note": "Uses unique 90 checkpoints (seed 42). All seeds identical under temp=0.0.",
        "lambda_scan": lam_results,
        "deployment": {
            "fixed": dep["fixed"],
            "oracle_stop_to_cot": dep["oracle_stop_to_cot"],
            "oracle_stop_to_re2": dep["oracle_stop_to_re2"],
            "oracle_3way": dep["oracle_3way"],
            "threshold_baselines_dev_only": dep["threshold_baselines_dev_only"],
            "deployment_targets": {
                str(target): {
                    "best_policy": min(
                        [(name, acc, lat) for name, acc, lat in all_policies if acc >= target],
                        key=lambda x: x[2]
                    ) if any(acc >= target for _, acc, _ in all_policies) else None
                }
                for target in [0.85, 0.88, 0.90]
            }
        }
    }
    out_path = R14_DIR / "r14_c_lambda_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
