#!/usr/bin/env python3
"""DAPH-X R1: Controlled evaluation with matched probe budgets.

Four systems compared on the same tasks with the same candidates:
  1. Base: model's first candidate (temp=0), no probes
  2. Probe baseline: run probes on all candidates, pick best probe pass rate
  3. Simple reranker: Q_MB + probe pass rate (no learned authority)
  4. DAPH-X: full authority stack (Q_res + pairwise + risk + conformal)

All systems except Base get the same probe budget (first K tests).
Frozen splits: dev (60%), cal (15%), confirmation (25%).

Usage:
    python scripts/run_r1_evaluation.py \\
        --corpus experiments/daph_x/r1/r1_corpus.jsonl \\
        --n_probe_tests 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import get_task, CodingTask
from daph_x.coding.code_executor import execute_solution
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

R1_DIR = REPO_ROOT / "experiments/daph_x/r1"


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def split_tasks(tasks: list[dict], train_frac=0.6, cal_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n = len(tasks)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    return (
        [tasks[i] for i in indices[:n_train]],
        [tasks[i] for i in indices[n_train:n_train + n_cal]],
        [tasks[i] for i in indices[n_train + n_cal:]],
    )


def get_feature_keys(records: list[dict]) -> list[str]:
    all_keys = set()
    for r in records:
        all_keys.update(r["features"].keys())
    return sorted(all_keys)


def build_feature_vector(features: dict, probe: dict, feature_keys: list[str]) -> np.ndarray:
    base = [float(features.get(k, 0.0)) for k in feature_keys]
    probe_feats = [
        probe.get("probe_pass_rate", 0.0),
        probe.get("probe_n_passed", 0),
        probe.get("probe_n_total", 0),
        probe.get("probe_has_error", 0.0),
    ]
    return np.array(base + probe_feats)


def flatten_candidates(tasks: list[dict]) -> list[dict]:
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


def compute_eval_utility(record: dict, n_probe: int) -> float:
    """Utility based on tests AFTER probe tests."""
    task = get_task(record["task_id"])
    if task is None:
        return record["full"]["utility"]
    total = len(task.tests)
    eval_tests = total - n_probe
    if eval_tests <= 0:
        return record["full"]["utility"]
    total_passed = record["full"]["tests_passed"]
    probe_passed = min(total_passed, n_probe)
    eval_passed = total_passed - probe_passed
    return (eval_passed / eval_tests) * 100.0


# ─── System 1: Base ───
def system_base(task: dict) -> dict:
    """Pick first candidate (temp=0). No probes."""
    base = task["candidates"][0]
    return {
        "pick_id": base["candidate_id"],
        "utility": base["full"]["utility"],
        "tests_passed": base["full"]["tests_passed"],
        "tests_total": base["full"]["tests_total"],
        "probe_cost": 0,
    }


# ─── System 2: Probe baseline ───
def system_probe_baseline(task: dict, n_probe: int) -> dict:
    """Run probes on all candidates, pick best probe pass rate."""
    best = max(task["candidates"], key=lambda c: (
        c["probe"]["probe_pass_rate"],
        -c["probe"]["probe_has_error"],
    ))
    return {
        "pick_id": best["candidate_id"],
        "utility": best["full"]["utility"],
        "tests_passed": best["full"]["tests_passed"],
        "tests_total": best["full"]["tests_total"],
        "probe_cost": n_probe * len(task["candidates"]),
    }


# ─── System 3: Simple reranker (Q_MB + probe) ───
def system_simple_reranker(task: dict, n_probe: int) -> dict:
    """Score = Q_MB + 10 * probe_pass_rate. Pick highest."""
    best = max(task["candidates"], key=lambda c: (
        c["q_mb"] + 10.0 * c["probe"]["probe_pass_rate"]
    ))
    return {
        "pick_id": best["candidate_id"],
        "utility": best["full"]["utility"],
        "tests_passed": best["full"]["tests_passed"],
        "tests_total": best["full"]["tests_total"],
        "probe_cost": n_probe * len(task["candidates"]),
    }


# ─── System 4: DAPH-X (full authority) ───
def system_daphx(task: dict, q_res_model, pairwise_model, risk_model,
                 conformal: dict, feature_keys: list[str], n_probe: int,
                 rho: float = 0.05, tau_delta: float = -50.0) -> dict:
    """Full authority stack with learned models."""
    cands = task["candidates"]
    base = cands[0]
    base_util = compute_eval_utility(base, n_probe)

    # Compute Q_X for each candidate
    for c in cands:
        feats = build_feature_vector(c["features"], c["probe"], feature_keys)
        c["q_x"] = float(q_res_model.predict([feats])[0])

    # DAPH-X pick: highest Q_X
    daphx = max(cands, key=lambda c: c["q_x"])
    disagreement = daphx["candidate_id"] != base["candidate_id"]

    if not disagreement:
        return {
            "pick_id": base["candidate_id"],
            "utility": base["full"]["utility"],
            "tests_passed": base["full"]["tests_passed"],
            "tests_total": base["full"]["tests_total"],
            "probe_cost": n_probe * len(cands),
            "would_force": False,
            "disagreement": False,
        }

    # Authority gate
    delta_q = daphx["q_x"] - base["q_x"]
    lcb = delta_q - conformal["q_90"]

    feats_d = build_feature_vector(daphx["features"], daphx["probe"], feature_keys)
    feats_b = build_feature_vector(base["features"], base["probe"], feature_keys)

    risk_prob = 0.0
    if risk_model is not None:
        risk_prob = float(risk_model.predict_proba([feats_d])[0, 1])

    pw_pred = 0.0
    if pairwise_model is not None:
        pair_feats = np.concatenate([
            feats_d - feats_b, feats_d, feats_b,
            [daphx["q_mb"] - base["q_mb"]],
            [daphx["q_mb"], base["q_mb"]],
            [daphx["probe"]["probe_pass_rate"] - base["probe"]["probe_pass_rate"]],
            [daphx["probe"]["probe_n_passed"] - base["probe"]["probe_n_passed"]],
        ])
        pw_pred = float(pairwise_model.predict([pair_feats])[0])

    would_force = lcb > tau_delta and risk_prob < rho and pw_pred > 0

    if would_force:
        pick = daphx
    else:
        pick = base

    return {
        "pick_id": pick["candidate_id"],
        "utility": pick["full"]["utility"],
        "tests_passed": pick["full"]["tests_passed"],
        "tests_total": pick["full"]["tests_total"],
        "probe_cost": n_probe * len(cands),
        "would_force": would_force,
        "disagreement": True,
        "delta_q": delta_q,
        "lcb": lcb,
        "risk_prob": risk_prob,
        "pairwise_pred": pw_pred,
    }


def train_models(train_tasks, cal_tasks, feature_keys, n_probe):
    """Train all authority models."""
    train_records = flatten_candidates(train_tasks)

    # Q_res
    X, y = [], []
    for r in train_records:
        feats = build_feature_vector(r["features"], r["probe"], feature_keys)
        X.append(feats)
        y.append(compute_eval_utility(r, n_probe))
    X, y = np.array(X), np.array(y)
    q_res_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    q_res_model.fit(X, y)

    # Pairwise
    X_pairs, y_pairs = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j: continue
                a, b = cands[i], cands[j]
                fa = build_feature_vector(a["features"], a["probe"], feature_keys)
                fb = build_feature_vector(b["features"], b["probe"], feature_keys)
                pair_feats = np.concatenate([
                    fa - fb, fa, fb,
                    [a["q_mb"] - b["q_mb"]],
                    [a["q_mb"], b["q_mb"]],
                    [a["probe"]["probe_pass_rate"] - b["probe"]["probe_pass_rate"]],
                    [a["probe"]["probe_n_passed"] - b["probe"]["probe_n_passed"]],
                ])
                delta_u = compute_eval_utility(a, n_probe) - compute_eval_utility(b, n_probe)
                X_pairs.append(pair_feats)
                y_pairs.append(delta_u)
    X_pairs, y_pairs = np.array(X_pairs), np.array(y_pairs)
    pairwise_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    pairwise_model.fit(X_pairs, y_pairs)

    # Risk
    X_r, y_r = [], []
    by_task = defaultdict(list)
    for r in train_records:
        by_task[r["task_id"]].append(r)
    for task_id, cands in by_task.items():
        base = cands[0]
        base_util = compute_eval_utility(base, n_probe)
        for c in cands:
            feats = build_feature_vector(c["features"], c["probe"], feature_keys)
            X_r.append(feats)
            y_r.append(1 if compute_eval_utility(c, n_probe) < base_util - 0.5 else 0)
    X_r, y_r = np.array(X_r), np.array(y_r)
    risk_model = None
    if len(set(y_r)) >= 2:
        risk_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42)
        risk_model.fit(X_r, y_r)

    # Conformal
    scores = []
    for task in cal_tasks:
        for c in task["candidates"]:
            feats = build_feature_vector(c["features"], c["probe"], feature_keys)
            q_x = q_res_model.predict([feats])[0]
            actual = compute_eval_utility(c, n_probe)
            scores.append(abs(q_x - actual))
    scores = np.array(sorted(scores))
    n = len(scores)
    q_90 = scores[min(int(np.ceil(0.90 * (n + 1))) - 1, n - 1)]
    conformal = {"q_90": float(q_90), "n_cal": n}

    return q_res_model, pairwise_model, risk_model, conformal


def evaluate_all_systems(eval_tasks, q_res_model, pairwise_model, risk_model,
                         conformal, feature_keys, n_probe):
    """Evaluate all 4 systems on the same eval tasks."""
    results = {
        "base": [],
        "probe_baseline": [],
        "simple_reranker": [],
        "daphx": [],
    }

    for task in eval_tasks:
        results["base"].append(system_base(task))
        results["probe_baseline"].append(system_probe_baseline(task, n_probe))
        results["simple_reranker"].append(system_simple_reranker(task, n_probe))
        results["daphx"].append(system_daphx(
            task, q_res_model, pairwise_model, risk_model,
            conformal, feature_keys, n_probe))

    return results


def compute_summary(results: dict, eval_tasks: list, n_probe: int) -> dict:
    """Compute summary statistics for all systems."""
    summary = {}

    # Find rescue opportunities
    rescue_opp = 0
    for task in eval_tasks:
        base = task["candidates"][0]
        best = max(task["candidates"], key=lambda c: c["full"]["utility"])
        if best["full"]["utility"] > base["full"]["utility"] + 0.5:
            rescue_opp += 1

    for system_name, system_results in results.items():
        n = len(system_results)
        utilities = [r["utility"] for r in system_results]
        tests_passed = [r["tests_passed"] for r in system_results]
        tests_total = [r["tests_total"] for r in system_results]
        probe_costs = [r["probe_cost"] for r in system_results]

        # Success: all tests passed
        successes = sum(1 for r in system_results if r["tests_passed"] == r["tests_total"])

        # For DAPH-X: count interventions
        if system_name == "daphx":
            n_disagree = sum(1 for r in system_results if r.get("disagreement", False))
            n_force = sum(1 for r in system_results if r.get("would_force", False))

            # Count rescues and breaks
            rescues = 0
            breaks = 0
            for i, r in enumerate(system_results):
                if r.get("would_force", False):
                    base_util = eval_tasks[i]["candidates"][0]["full"]["utility"]
                    pick_util = r["utility"]
                    if pick_util > base_util + 0.5:
                        rescues += 1
                    elif pick_util < base_util - 0.5:
                        breaks += 1

            # Rescue recall: of the rescue_opp tasks, how many did DAPH-X rescue?
            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                base = task["candidates"][0]
                best = max(task["candidates"], key=lambda c: c["full"]["utility"])
                if best["full"]["utility"] > base["full"]["utility"] + 0.5:
                    # This is a rescue opportunity
                    if results["daphx"][i]["pick_id"] != base["candidate_id"]:
                        if results["daphx"][i]["utility"] > base["full"]["utility"] + 0.5:
                            daphx_rescues += 1

            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "std_utility": np.std(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "mean_tests_passed": np.mean(tests_passed),
                "mean_probe_cost": np.mean(probe_costs),
                "total_probe_cost": sum(probe_costs),
                "n_disagreements": n_disagree,
                "n_force": n_force,
                "rescues": rescues,
                "breaks": breaks,
                "break_rate": breaks / max(n_force, 1),
                "rescue_recall": daphx_rescues / max(rescue_opp, 1),
                "n_rescue_opportunities": rescue_opp,
                "break_rate_upper_95": (3.0 / n_force) if (n_force > 0 and breaks == 0) else None,
            }
        else:
            # For non-DAPH-X systems, count how many times they beat base
            base_utilities = [eval_tasks[i]["candidates"][0]["full"]["utility"]
                              for i in range(len(eval_tasks))]
            improvements = sum(1 for u, bu in zip(utilities, base_utilities) if u > bu + 0.5)
            regressions = sum(1 for u, bu in zip(utilities, base_utilities) if u < bu - 0.5)

            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "std_utility": np.std(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "mean_tests_passed": np.mean(tests_passed),
                "mean_probe_cost": np.mean(probe_costs),
                "total_probe_cost": sum(probe_costs),
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
            }

    return summary


def print_results_table(summary: dict):
    """Print the main results table."""
    print(f"\n{'='*90}")
    print(f"  DAPH-X R1: Real Coding Authority Qualification")
    print(f"{'='*90}")
    print()
    print(f"{'Metric':<25} {'Base':>12} {'Probe':>12} {'Reranker':>12} {'DAPH-X':>12}")
    print(f"{'':25} {'':>12} {'Baseline':>12} {'(Q_MB+probe)':>12} {'(authority)':>12}")
    print("-" * 90)

    metrics = [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".2%"),
        ("Mean tests passed", "mean_tests_passed", ".1f"),
        ("Mean probe cost", "mean_probe_cost", ".1f"),
        ("Improvements vs base", "improvements_vs_base", "d"),
        ("Regressions vs base", "regressions_vs_base", "d"),
    ]

    for label, key, fmt in metrics:
        vals = []
        for sys_name in ["base", "probe_baseline", "simple_reranker", "daphx"]:
            if sys_name in summary and key in summary[sys_name]:
                v = summary[sys_name][key]
                if fmt == "d":
                    vals.append(f"{int(v):>12}")
                elif fmt == ".2%":
                    vals.append(f"{v:>12.1%}")
                else:
                    vals.append(f"{v:>12{fmt}}")
            else:
                vals.append(f"{'N/A':>12}")
        print(f"{label:<25} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

    # DAPH-X specific
    if "daphx" in summary:
        d = summary["daphx"]
        print()
        print(f"  DAPH-X Authority Details:")
        print(f"    Rescue opportunities: {d['n_rescue_opportunities']}")
        print(f"    Disagreements:        {d['n_disagreements']}")
        print(f"    Would FORCE:          {d['n_force']}")
        print(f"    Rescues:              {d['rescues']}")
        print(f"    Breaks:               {d['breaks']}")
        print(f"    Break rate:           {d['break_rate']:.4f}")
        if d.get("break_rate_upper_95"):
            print(f"    Break rate 95% upper: {d['break_rate_upper_95']:.4f}")
        print(f"    Rescue recall:        {d['rescue_recall']:.4f}")

    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R1_DIR / "r1_corpus.jsonl"))
    parser.add_argument("--n_probe_tests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--tau_delta", type=float, default=-50.0)
    args = parser.parse_args()

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    # Frozen split
    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)} + 4 probe = {len(feature_keys)+4}")

    # Train models on dev, calibrate on cal
    print("\nTraining authority models...")
    q_res_model, pairwise_model, risk_model, conformal = train_models(
        train_tasks, cal_tasks, feature_keys, args.n_probe_tests)
    print(f"  Conformal q_90 = {conformal['q_90']:.2f}")

    # Evaluate all 4 systems on confirmation
    print(f"\nEvaluating 4 systems on {len(eval_tasks)} confirmation tasks...")
    results = evaluate_all_systems(
        eval_tasks, q_res_model, pairwise_model, risk_model,
        conformal, feature_keys, args.n_probe_tests)

    # Summary
    summary = compute_summary(results, eval_tasks, args.n_probe_tests)
    print_results_table(summary)

    # Save
    output = {
        "config": {
            "n_probe_tests": args.n_probe_tests,
            "rho": args.rho,
            "tau_delta": args.tau_delta,
            "seed": args.seed,
            "n_dev": len(train_tasks),
            "n_cal": len(cal_tasks),
            "n_confirmation": len(eval_tasks),
        },
        "summary": summary,
        "conformal": conformal,
        "per_task_results": {
            "base": results["base"],
            "probe_baseline": results["probe_baseline"],
            "simple_reranker": results["simple_reranker"],
            "daphx": results["daphx"],
        },
    }
    output_path = R1_DIR / "r1_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
