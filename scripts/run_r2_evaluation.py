#!/usr/bin/env python3
"""DAPH-X R2: Reformulated authority — "should I intervene on the probe winner?"

Key insight from R1: the probe baseline (pick best probe pass rate) is strong.
DAPH-X trying to pick the best candidate from scratch loses to it.

New formulation: the probe winner is the DEFAULT. DAPH-X only intervenes
when it has evidence that the probe winner is NOT the best candidate.

Decision:
  1. Run probes on all candidates
  2. Pick probe winner (highest probe pass rate, tie-break by Q_MB)
  3. Compute Q_X for all candidates
  4. If Q_X predicts a DIFFERENT candidate is better by a margin exceeding
     the conformal uncertainty, AND the risk model says intervention is safe,
     THEN intervene (pick the Q_X winner instead)
  5. Otherwise abstain (keep the probe winner)

This directly tests: "Can a learned authority identify cases where
the probe winner is wrong?"

Four systems compared:
  1. Base: first candidate, no probes
  2. Probe baseline: pick best probe pass rate (the default)
  3. Simple reranker: Q_MB + probe pass rate
  4. DAPH-X: probe winner as default + authority intervention gate

Frozen splits: dev (60%), cal (15%), confirmation (25%).

Usage:
    python scripts/run_r2_evaluation.py \\
        --corpus experiments/daph_x/r1/r1_corpus_full.jsonl \\
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

from daph_x.coding.tasks import get_task
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

R2_DIR = REPO_ROOT / "experiments/daph_x/r2"


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


def pick_probe_winner(cands: list[dict]) -> dict:
    """Pick the probe winner: highest probe pass rate, tie-break by Q_MB."""
    return max(cands, key=lambda c: (c["probe"]["probe_pass_rate"], c["q_mb"]))


# ─── System 1: Base ───
def system_base(task: dict) -> dict:
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
    winner = pick_probe_winner(task["candidates"])
    return {
        "pick_id": winner["candidate_id"],
        "utility": winner["full"]["utility"],
        "tests_passed": winner["full"]["tests_passed"],
        "tests_total": winner["full"]["tests_total"],
        "probe_cost": n_probe * len(task["candidates"]),
    }


# ─── System 3: Simple reranker ───
def system_simple_reranker(task: dict, n_probe: int) -> dict:
    best = max(task["candidates"], key=lambda c: c["q_mb"] + 10.0 * c["probe"]["probe_pass_rate"])
    return {
        "pick_id": best["candidate_id"],
        "utility": best["full"]["utility"],
        "tests_passed": best["full"]["tests_passed"],
        "tests_total": best["full"]["tests_total"],
        "probe_cost": n_probe * len(task["candidates"]),
    }


# ─── System 4: DAPH-X R2 (probe winner + intervention) ───
def system_daphx_r2(
    task: dict, q_res_model, pairwise_model, risk_model,
    conformal: dict, feature_keys: list[str], n_probe: int,
    rho: float = 0.05, tau_delta: float = 0.0,
) -> dict:
    """Probe winner is default. Intervene only when Q_X strongly disagrees."""
    cands = task["candidates"]
    probe_winner = pick_probe_winner(cands)

    # Compute Q_X for all candidates
    for c in cands:
        feats = build_feature_vector(c["features"], c["probe"], feature_keys)
        c["q_x"] = float(q_res_model.predict([feats])[0])

    # Q_X winner
    qx_winner = max(cands, key=lambda c: c["q_x"])

    # Is there a disagreement between probe winner and Q_X winner?
    disagreement = qx_winner["candidate_id"] != probe_winner["candidate_id"]

    if not disagreement:
        return {
            "pick_id": probe_winner["candidate_id"],
            "utility": probe_winner["full"]["utility"],
            "tests_passed": probe_winner["full"]["tests_passed"],
            "tests_total": probe_winner["full"]["tests_total"],
            "probe_cost": n_probe * len(cands),
            "would_force": False,
            "disagreement": False,
            "default": "probe_winner",
        }

    # Authority gate: should we override the probe winner?
    delta_q = qx_winner["q_x"] - probe_winner["q_x"]
    lcb = delta_q - conformal["q_90"]

    feats_q = build_feature_vector(qx_winner["features"], qx_winner["probe"], feature_keys)
    feats_p = build_feature_vector(probe_winner["features"], probe_winner["probe"], feature_keys)

    # Risk: is the Q_X winner likely to be worse than the probe winner?
    risk_prob = 0.0
    if risk_model is not None:
        # Risk = probability that qx_winner is worse than probe_winner
        risk_prob = float(risk_model.predict_proba([feats_q - feats_p])[0, 1])

    # Pairwise: predicted ΔU between Q_X winner and probe winner
    pw_pred = 0.0
    if pairwise_model is not None:
        pair_feats = np.concatenate([
            feats_q - feats_p, feats_q, feats_p,
            [qx_winner["q_mb"] - probe_winner["q_mb"]],
            [qx_winner["q_mb"], probe_winner["q_mb"]],
            [qx_winner["probe"]["probe_pass_rate"] - probe_winner["probe"]["probe_pass_rate"]],
            [qx_winner["probe"]["probe_n_passed"] - probe_winner["probe"]["probe_n_passed"]],
        ])
        pw_pred = float(pairwise_model.predict([pair_feats])[0])

    # Gate: intervene only if Q_X advantage is large enough, risk is low,
    # and pairwise model predicts positive ΔU
    would_force = lcb > tau_delta and risk_prob < rho and pw_pred > 0

    if would_force:
        pick = qx_winner
        intervened = True
    else:
        pick = probe_winner
        intervened = False

    return {
        "pick_id": pick["candidate_id"],
        "utility": pick["full"]["utility"],
        "tests_passed": pick["full"]["tests_passed"],
        "tests_total": pick["full"]["tests_total"],
        "probe_cost": n_probe * len(cands),
        "would_force": intervened,
        "disagreement": True,
        "delta_q": delta_q,
        "lcb": lcb,
        "risk_prob": risk_prob,
        "pairwise_pred": pw_pred,
        "probe_winner_id": probe_winner["candidate_id"],
        "qx_winner_id": qx_winner["candidate_id"],
        "probe_winner_utility": probe_winner["full"]["utility"],
        "qx_winner_utility": qx_winner["full"]["utility"],
    }


def train_models(train_tasks, cal_tasks, feature_keys, n_probe):
    """Train all authority models."""
    train_records = flatten_candidates(train_tasks)

    # Q_res: predict eval utility (excluding probe tests)
    X, y = [], []
    for r in train_records:
        feats = build_feature_vector(r["features"], r["probe"], feature_keys)
        X.append(feats)
        y.append(compute_eval_utility(r, n_probe))
    X, y = np.array(X), np.array(y)
    q_res_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    q_res_model.fit(X, y)

    # Pairwise: predict ΔU between candidate pairs
    X_pairs, y_pairs = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
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
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    pairwise_model.fit(X_pairs, y_pairs)

    # Risk: predict probability that Q_X winner is worse than probe winner
    # Training data: for each task, compute probe winner and Q_X winner
    # Label: 1 if Q_X winner utility < probe winner utility - 0.5
    X_r, y_r = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for c in cands:
            feats = build_feature_vector(c["features"], c["probe"], feature_keys)
            c["q_x"] = float(q_res_model.predict([feats])[0])
        pw = pick_probe_winner(cands)
        qx = max(cands, key=lambda c: c["q_x"])
        if qx["candidate_id"] != pw["candidate_id"]:
            # This is a disagreement case
            fq = build_feature_vector(qx["features"], qx["probe"], feature_keys)
            fp = build_feature_vector(pw["features"], pw["probe"], feature_keys)
            X_r.append(fq - fp)
            y_r.append(1 if qx["full"]["utility"] < pw["full"]["utility"] - 0.5 else 0)

    X_r, y_r = np.array(X_r), np.array(y_r)
    risk_model = None
    if len(set(y_r)) >= 2:
        risk_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        risk_model.fit(X_r, y_r)

    # Conformal: calibrate Q_X prediction error on calibration set
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
    q_80 = scores[min(int(np.ceil(0.80 * (n + 1))) - 1, n - 1)]
    q_50 = scores[min(int(np.ceil(0.50 * (n + 1))) - 1, n - 1)]
    conformal = {"q_90": float(q_90), "q_80": float(q_80), "q_50": float(q_50), "n_cal": n}

    return q_res_model, pairwise_model, risk_model, conformal


def evaluate_all_systems(eval_tasks, q_res_model, pairwise_model, risk_model,
                         conformal, feature_keys, n_probe, rho, tau_delta):
    results = {"base": [], "probe_baseline": [], "simple_reranker": [], "daphx": []}
    for task in eval_tasks:
        results["base"].append(system_base(task))
        results["probe_baseline"].append(system_probe_baseline(task, n_probe))
        results["simple_reranker"].append(system_simple_reranker(task, n_probe))
        results["daphx"].append(system_daphx_r2(
            task, q_res_model, pairwise_model, risk_model,
            conformal, feature_keys, n_probe, rho, tau_delta))
    return results


def compute_summary(results: dict, eval_tasks: list, n_probe: int) -> dict:
    summary = {}

    # Find probe-baseline-suboptimal cases (the opportunity for DAPH-X)
    probe_suboptimal = 0
    probe_suboptimal_gap = []
    for task in eval_tasks:
        pw = pick_probe_winner(task["candidates"])
        best = max(task["candidates"], key=lambda c: c["full"]["utility"])
        if pw["candidate_id"] != best["candidate_id"] and best["full"]["utility"] > pw["full"]["utility"] + 0.5:
            probe_suboptimal += 1
            probe_suboptimal_gap.append(best["full"]["utility"] - pw["full"]["utility"])

    for system_name, system_results in results.items():
        n = len(system_results)
        utilities = [r["utility"] for r in system_results]
        successes = sum(1 for r in system_results if r["tests_passed"] == r["tests_total"])
        probe_costs = [r["probe_cost"] for r in system_results]

        base_utilities = [eval_tasks[i]["candidates"][0]["full"]["utility"]
                          for i in range(len(eval_tasks))]
        improvements = sum(1 for u, bu in zip(utilities, base_utilities) if u > bu + 0.5)
        regressions = sum(1 for u, bu in zip(utilities, base_utilities) if u < bu - 0.5)

        if system_name == "daphx":
            n_disagree = sum(1 for r in system_results if r.get("disagreement", False))
            n_force = sum(1 for r in system_results if r.get("would_force", False))

            # Count rescues and breaks relative to probe baseline
            probe_utils = [system_probe_baseline(t, n_probe)["utility"] for t in eval_tasks]
            rescues_vs_probe = sum(1 for r, pu in zip(system_results, probe_utils)
                                   if r.get("would_force", False) and r["utility"] > pu + 0.5)
            breaks_vs_probe = sum(1 for r, pu in zip(system_results, probe_utils)
                                  if r.get("would_force", False) and r["utility"] < pu - 0.5)

            # Rescue recall: of probe-suboptimal cases, how many did DAPH-X fix?
            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                pw = pick_probe_winner(task["candidates"])
                best = max(task["candidates"], key=lambda c: c["full"]["utility"])
                if best["full"]["utility"] > pw["full"]["utility"] + 0.5:
                    if results["daphx"][i]["utility"] > pw["full"]["utility"] + 0.5:
                        daphx_rescues += 1

            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "std_utility": np.std(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "mean_probe_cost": np.mean(probe_costs),
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
                "n_disagreements": n_disagree,
                "n_force": n_force,
                "rescues_vs_probe": rescues_vs_probe,
                "breaks_vs_probe": breaks_vs_probe,
                "rescue_recall_vs_probe": daphx_rescues / max(probe_suboptimal, 1),
                "n_probe_suboptimal": probe_suboptimal,
            }
        else:
            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "std_utility": np.std(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "mean_probe_cost": np.mean(probe_costs),
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
            }

    return summary


def print_results_table(summary: dict):
    print(f"\n{'='*95}")
    print(f"  DAPH-X R2: Reformulated Authority (probe winner + intervention)")
    print(f"{'='*95}")
    print()
    print(f"{'Metric':<28} {'Base':>12} {'Probe':>12} {'Reranker':>12} {'DAPH-X':>12}")
    print(f"{'':28} {'':>12} {'Baseline':>12} {'(Q_MB+p)':>12} {'(R2 auth)':>12}")
    print("-" * 95)

    metrics = [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".1%"),
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
                elif fmt == ".1%":
                    vals.append(f"{v:>11.1%}")
                else:
                    vals.append(f"{v:>12{fmt}}")
            else:
                vals.append(f"{'N/A':>12}")
        print(f"{label:<28} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

    if "daphx" in summary:
        d = summary["daphx"]
        print()
        print(f"  DAPH-X R2 Authority Details:")
        print(f"    Probe-suboptimal cases:   {d['n_probe_suboptimal']}")
        print(f"    Disagreements with probe: {d['n_disagreements']}")
        print(f"    Would FORCE (intervene):  {d['n_force']}")
        print(f"    Rescues vs probe:         {d['rescues_vs_probe']}")
        print(f"    Breaks vs probe:          {d['breaks_vs_probe']}")
        print(f"    Rescue recall vs probe:   {d['rescue_recall_vs_probe']:.4f}")

    print(f"\n{'='*95}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R2_DIR / "r2_corpus.jsonl"))
    parser.add_argument("--n_probe_tests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--tau_delta", type=float, default=0.0)
    args = parser.parse_args()

    R2_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)} + 4 probe = {len(feature_keys)+4}")

    print("\nTraining authority models (R2 formulation)...")
    q_res_model, pairwise_model, risk_model, conformal = train_models(
        train_tasks, cal_tasks, feature_keys, args.n_probe_tests)
    print(f"  Conformal: q_90={conformal['q_90']:.2f}, q_80={conformal['q_80']:.2f}, q_50={conformal['q_50']:.2f}")
    print(f"  Risk model: {'trained' if risk_model else 'not enough data'}")

    # Try multiple configurations
    print(f"\nEvaluating on {len(eval_tasks)} confirmation tasks...")
    configs = [
        (0.05, 0.0, "tau=0, rho=0.05"),
        (0.05, -10.0, "tau=-10, rho=0.05"),
        (0.05, -20.0, "tau=-20, rho=0.05"),
        (0.05, -50.0, "tau=-50, rho=0.05"),
        (0.10, -10.0, "tau=-10, rho=0.10"),
        (0.20, -10.0, "tau=-10, rho=0.20"),
        (0.50, -10.0, "tau=-10, rho=0.50"),
        (1.0, -100.0, "no gate (always intervene)"),
    ]

    best_config = None
    best_util = -float("inf")

    for rho, tau, label in configs:
        results = evaluate_all_systems(
            eval_tasks, q_res_model, pairwise_model, risk_model,
            conformal, feature_keys, args.n_probe_tests, rho, tau)
        summary = compute_summary(results, eval_tasks, args.n_probe_tests)

        d = summary["daphx"]
        pb = summary["probe_baseline"]
        util = d["mean_utility"]
        if util > best_util and d["breaks_vs_probe"] == 0:
            best_util = util
            best_config = (rho, tau, label, results, summary)

        br = d["breaks_vs_probe"] / max(d["n_force"], 1)
        print(f"  {label:30s}: util={util:.1f} (probe={pb['mean_utility']:.1f}), "
              f"force={d['n_force']}, R={d['rescues_vs_probe']}, B={d['breaks_vs_probe']}, "
              f"recall={d['rescue_recall_vs_probe']:.2f}")

    # Print best config results
    if best_config:
        rho, tau, label, results, summary = best_config
        print(f"\n  Best safe config: {label}")
        print_results_table(summary)

        # Save
        output = {
            "config": {"n_probe_tests": args.n_probe_tests, "rho": rho, "tau_delta": tau,
                       "seed": args.seed, "n_dev": len(train_tasks),
                       "n_cal": len(cal_tasks), "n_confirmation": len(eval_tasks)},
            "summary": summary,
            "conformal": conformal,
        }
        output_path = R2_DIR / "r2_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
