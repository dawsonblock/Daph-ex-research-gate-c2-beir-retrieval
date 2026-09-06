#!/usr/bin/env python3
"""R15-A: Apply frozen champion to confirmation set and report results.

Loads:
- Frozen champion (r15_a_champion.json)
- Confirmation executions (r15_a_confirmation_executions.jsonl)
- Confirmation manifest (r15_a_confirmation_manifest.jsonl)

Applies the champion's decision rule to each confirmation checkpoint.
Reports accuracy, latency, oracle headroom recovery, and pass/fail.
"""
from __future__ import annotations

import json
import sys
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

R15_DIR = PROJECT_ROOT / "experiments/daph_x/r15"

NUMERIC_FEATURES = [
    "k", "p_top1", "p_top2", "margin", "entropy",
    "n_unique_answers", "agreement_rate",
    "uncertainty_current", "uncertainty_delta",
    "margin_delta", "answer_changed", "stable_prefix_count",
]


def load_champion():
    with open(R15_DIR / "r15_a_champion.json") as f:
        return json.load(f)


def load_executions():
    with open(R15_DIR / "r15_a_confirmation_executions.jsonl") as f:
        return [json.loads(l) for l in f]


def load_manifest():
    with open(R15_DIR / "r15_a_confirmation_manifest.jsonl") as f:
        return {e["checkpoint_id"]: e for e in (json.loads(l) for l in f)}


def build_feature_vector(features, difficulty, category, champion):
    """Build feature vector matching champion's preprocessing."""
    difficulties = champion["difficulties"]
    categories = champion["categories"]
    row = [features.get(f, 0.0) for f in NUMERIC_FEATURES]
    for d in difficulties:
        row.append(1.0 if difficulty == d else 0.0)
    for c in categories:
        row.append(1.0 if category == c else 0.0)
    return np.array(row)


def apply_champion(champion, X):
    """Apply frozen champion to feature matrix."""
    if champion["model"] == "C_ridge_deltaJ":
        scaler_mean = np.array(champion["scaler_mean"])
        scaler_scale = np.array(champion["scaler_scale"])
        X_scaled = (X - scaler_mean) / scaler_scale
        coef = np.array(champion["coefficients"])
        intercept = champion["intercept"]
        pred = X_scaled @ coef + intercept
        return pred > 0  # escalate if ΔJ > 0
    elif champion["model"] == "B_logistic":
        scaler_mean = np.array(champion["scaler_mean"])
        scaler_scale = np.array(champion["scaler_scale"])
        X_scaled = (X - scaler_mean) / scaler_scale
        coef = np.array(champion["coefficients"])
        intercept = champion["intercept"]
        logit = X_scaled @ coef + intercept
        p = 1 / (1 + np.exp(-logit))
        return p > champion["threshold"]
    elif champion["model"] == "A_threshold":
        # Not the champion, but handle for completeness
        feat_idx = NUMERIC_FEATURES.index(champion["feature"])
        vals = X[:, feat_idx]
        if champion["direction"] == "below":
            return vals < champion["threshold"]
        else:
            return vals > champion["threshold"]
    else:
        raise ValueError(f"Unknown model: {champion['model']}")


def main():
    champion = load_champion()
    print(f"Champion: {champion['model']}")
    print(f"  Hash: {champion['champion_hash']}")
    print()

    executions = load_executions()
    manifest = load_manifest()

    # Group by checkpoint
    by_cp = defaultdict(dict)
    for r in executions:
        op = r.get("operator_id_canonical", r["operator_id"])
        by_cp[r["checkpoint_id"]][op] = r

    # Build feature matrix and outcome vectors
    cp_ids = sorted(by_cp.keys())
    X_list = []
    y_stop = []
    y_cot = []
    lat_cot = []
    task_ids = []

    for cp_id in cp_ids:
        ops = by_cp[cp_id]
        entry = manifest[cp_id]
        of = entry["runtime_state"].get("observable_features", {})
        difficulty = entry["runtime_state"].get("difficulty", "unknown")
        category = entry["runtime_state"].get("category", "unknown")

        x = build_feature_vector(of, difficulty, category, champion)
        X_list.append(x)

        stop_correct = 1 if ops.get("STOP", {}).get("correct") else 0
        cot_correct = 1 if ops.get("OPT_COT_REFLECT", {}).get("correct") else 0
        cot_wall = ops.get("OPT_COT_REFLECT", {}).get("wall_ms_observed", 0) / 1000.0

        y_stop.append(stop_correct)
        y_cot.append(cot_correct)
        lat_cot.append(cot_wall)
        task_ids.append(entry["task_id"])

    X = np.array(X_list)
    y_stop = np.array(y_stop)
    y_cot = np.array(y_cot)
    lat_cot = np.array(lat_cot)

    N = len(cp_ids)
    print(f"Confirmation set: {N} checkpoints, {len(set(task_ids))} unique tasks")
    print()

    # Apply champion
    escalate_mask = apply_champion(champion, X)

    # Router performance
    router_q = np.where(escalate_mask, y_cot, y_stop)
    router_l = np.where(escalate_mask, lat_cot, 0.0)
    router_acc = router_q.mean()
    router_lat = router_l.mean()
    n_escalated = escalate_mask.sum()

    # Baselines
    stop_acc = y_stop.mean()
    stop_lat = 0.0
    cot_acc = y_cot.mean()
    cot_lat = lat_cot.mean()

    # Oracle (STOP→COT: use COT only when STOP is wrong)
    oracle_q = np.where(y_stop > 0.5, y_stop, y_cot)
    oracle_l = np.where(y_stop > 0.5, 0.0, lat_cot)
    oracle_acc = oracle_q.mean()
    oracle_lat = oracle_l.mean()

    # 3-way oracle (not applicable here since we only have STOP and COT)
    # But compute the 2-way oracle
    print("=" * 70)
    print("R15-A CONFIRMATION RESULTS")
    print("=" * 70)
    print()
    print(f"{'Policy':<25} {'Accuracy':>8} {'Mean lat (s)':>13} {'N correct':>10}")
    print(f"{'-'*60}")
    print(f"{'Always STOP':<25} {stop_acc:>8.4f} {stop_lat:>13.3f} {int(stop_acc*N):>10}")
    print(f"{'Always COT':<25} {cot_acc:>8.4f} {cot_lat:>13.3f} {int(cot_acc*N):>10}")
    print(f"{'STOP→COT oracle':<25} {oracle_acc:>8.4f} {oracle_lat:>13.3f} {int(oracle_acc*N):>10}")
    print(f"{'Champion router':<25} {router_acc:>8.4f} {router_lat:>13.3f} {int(router_acc*N):>10}")
    print()

    # Confusion matrix
    stop_kept = ~escalate_mask
    cot_esc = escalate_mask
    sk_correct = (stop_kept & (y_stop > 0.5)).sum()
    sk_wrong = (stop_kept & (y_stop < 0.5)).sum()
    ce_correct = (cot_esc & (y_cot > 0.5)).sum()
    ce_wrong = (cot_esc & (y_cot < 0.5)).sum()
    print(f"Confusion matrix:")
    print(f"  STOP-kept correct:   {sk_correct}")
    print(f"  STOP-kept wrong:     {sk_wrong}")
    print(f"  COT-escalated correct: {ce_correct}")
    print(f"  COT-escalated wrong:   {ce_wrong}")
    print(f"  Total escalated: {n_escalated}/{N} ({n_escalated/N:.1%})")
    print()

    # Non-inferiority gate
    acc_diff = router_acc - cot_acc
    print(f"Accuracy non-inferiority gate:")
    print(f"  A_router = {router_acc:.4f}")
    print(f"  A_COT    = {cot_acc:.4f}")
    print(f"  A_router - A_COT = {acc_diff:+.4f}")
    print(f"  Gate: A_router >= A_COT - 0.005 = {cot_acc - 0.005:.4f}")
    print(f"  {'PASS' if router_acc >= cot_acc - 0.005 else 'FAIL'}: {router_acc:.4f} {'>=' if router_acc >= cot_acc - 0.005 else '<'} {cot_acc - 0.005:.4f}")
    print()

    # Latency saving
    router_saving = 1 - router_lat / cot_lat
    oracle_saving = 1 - oracle_lat / cot_lat
    r_l = router_saving / oracle_saving if oracle_saving > 0 else 0
    print(f"Latency analysis:")
    print(f"  Router mean latency:   {router_lat:.3f}s")
    print(f"  Always-COT latency:    {cot_lat:.3f}s")
    print(f"  Oracle latency:        {oracle_lat:.3f}s")
    print(f"  Router saving:         {router_saving*100:.1f}%")
    print(f"  Oracle saving:         {oracle_saving*100:.1f}%")
    print(f"  Oracle-headroom recovery R_L: {r_l*100:.1f}%")
    print()

    # Tier
    print(f"Tier assessment:")
    tier = "NONE"
    if router_acc >= cot_acc - 0.005:
        if r_l >= 0.90:
            tier = "GOLD"
        elif r_l >= 0.75:
            tier = "SILVER"
        elif r_l >= 0.50:
            tier = "BRONZE"
    print(f"  Tier: {tier}")
    print(f"  Bronze (>=50% recovery): {'PASS' if r_l >= 0.50 and router_acc >= cot_acc - 0.005 else 'FAIL'}")
    print(f"  Silver (>=75% recovery): {'PASS' if r_l >= 0.75 and router_acc >= cot_acc - 0.005 else 'FAIL'}")
    print(f"  Gold   (>=90% recovery): {'PASS' if r_l >= 0.90 and router_acc >= cot_acc - 0.005 else 'FAIL'}")
    print()

    # Task-clustered bootstrap CI for A_router - A_COT
    unique_tasks = sorted(set(task_ids))
    task_to_idx = defaultdict(list)
    for i, t in enumerate(task_ids):
        task_to_idx[t].append(i)

    random.seed(0)
    n_boot = 2000
    boot_diffs = []
    for _ in range(n_boot):
        sampled_tasks = [random.choice(unique_tasks) for _ in range(len(unique_tasks))]
        boot_router = []
        boot_cot = []
        for t in sampled_tasks:
            idxs = task_to_idx[t]
            for idx in idxs:
                boot_router.append(router_q[idx])
                boot_cot.append(y_cot[idx])
        boot_diffs.append(np.mean(boot_router) - np.mean(boot_cot))
    boot_diffs.sort()
    ci_lo = boot_diffs[int(0.025 * n_boot)]
    ci_hi = boot_diffs[int(0.975 * n_boot)]
    print(f"Task-clustered bootstrap CI for A_router - A_COT:")
    print(f"  Mean diff: {acc_diff:+.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()

    # Latency distribution
    print(f"Latency distribution (router):")
    router_lats = router_l[router_l > 0]
    if len(router_lats) > 0:
        sl = sorted(router_lats)
        print(f"  mean={router_lats.mean():.3f}s, median={sl[len(sl)//2]:.3f}s, p90={sl[int(len(sl)*0.9)]:.3f}s, p95={sl[int(len(sl)*0.95)]:.3f}s")
    print()

    # Final verdict
    print("=" * 70)
    print("R15-A FINAL VERDICT")
    print("=" * 70)
    if tier != "NONE":
        print(f"  R15-A {tier}: DAPH-X has demonstrated real engineering leverage.")
        print(f"  A frozen linear router retains {router_acc:.1%} accuracy (COT: {cot_acc:.1%})")
        print(f"  while recovering {r_l:.1%} of the available latency saving.")
    else:
        if router_acc < cot_acc - 0.005:
            print(f"  R15-A FAIL: Router accuracy {router_acc:.4f} is below non-inferiority gate {cot_acc - 0.005:.4f}")
            print(f"  The observable features are not predictive enough to justify a controller.")
            print(f"  Defaulting to COT_REFLECT all the time is the better system.")
        else:
            print(f"  R15-A FAIL: Router meets accuracy gate but recovers only {r_l:.1%} of latency saving (need >=50%).")
            print(f"  The router escalates too often ({n_escalated}/{N}) to save meaningful latency.")

    # Save results
    results = {
        "experiment": "R15-A",
        "champion_hash": champion["champion_hash"],
        "n_checkpoints": N,
        "n_unique_tasks": len(set(task_ids)),
        "n_escalated": int(n_escalated),
        "always_stop": {"accuracy": float(stop_acc), "mean_latency_s": float(stop_lat)},
        "always_cot": {"accuracy": float(cot_acc), "mean_latency_s": float(cot_lat)},
        "oracle_stop_to_cot": {"accuracy": float(oracle_acc), "mean_latency_s": float(oracle_lat)},
        "router": {
            "accuracy": float(router_acc),
            "mean_latency_s": float(router_lat),
            "n_correct": int(router_acc * N),
        },
        "confusion": {
            "stop_kept_correct": int(sk_correct),
            "stop_kept_wrong": int(sk_wrong),
            "cot_escalated_correct": int(ce_correct),
            "cot_escalated_wrong": int(ce_wrong),
        },
        "non_inferiority_gate": {
            "A_router": float(router_acc),
            "A_COT": float(cot_acc),
            "margin": 0.005,
            "threshold": float(cot_acc - 0.005),
            "passed": bool(router_acc >= cot_acc - 0.005),
        },
        "latency": {
            "router_saving": float(router_saving),
            "oracle_saving": float(oracle_saving),
            "oracle_headroom_recovery": float(r_l),
        },
        "tier": tier,
        "bootstrap_ci_acc_diff": {
            "mean": float(acc_diff),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        },
    }
    out_path = R15_DIR / "r15_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
