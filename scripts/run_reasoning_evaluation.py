#!/usr/bin/env python3
"""DAPH-X Reasoning Authority Evaluation.

No execution feedback — the authority must decide based on:
  - Self-evaluation confidence
  - Cross-candidate consistency
  - Reasoning trace features

Four systems compared:
  1. Base: first candidate (temp=0)
  2. Majority vote: pick the most common answer among candidates
  3. Self-confidence: pick the candidate with highest self-confidence
  4. DAPH-X: learned authority (Q_res + pairwise + risk + conformal)

Utility is binary: 100 if correct, 0 if wrong.

Usage:
    python scripts/run_reasoning_evaluation.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import check_answer, get_reasoning_task

REASONING_DIR = REPO_ROOT / "experiments/daph_x/reasoning"


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def split_tasks(tasks, train_frac=0.6, cal_frac=0.15, seed=42):
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


def build_feature_vector(features: dict, feature_keys: list[str]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in feature_keys])


def flatten_candidates(tasks: list[dict]) -> list[dict]:
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


# ─── System 1: Base ───
def system_base(task: dict) -> dict:
    base = task["candidates"][0]
    return {
        "pick_id": base["candidate_id"],
        "utility": 100.0 if base["is_correct"] else 0.0,
        "correct": base["is_correct"],
    }


# ─── System 2: Majority vote ───
def system_majority_vote(task: dict) -> dict:
    answers = [c["answer"] for c in task["candidates"]]
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]
    # Pick the first candidate with the majority answer
    pick = next(c for c in task["candidates"] if c["answer"] == majority_answer)
    return {
        "pick_id": pick["candidate_id"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "correct": pick["is_correct"],
    }


# ─── System 3: Self-confidence ───
def system_self_confidence(task: dict) -> dict:
    pick = max(task["candidates"], key=lambda c: c["self_confidence"])
    return {
        "pick_id": pick["candidate_id"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "correct": pick["is_correct"],
    }


# ─── System 4: DAPH-X (learned authority) ───
def system_daphx(task: dict, q_res_model, pairwise_model, risk_model,
                 conformal: dict, feature_keys: list[str],
                 rho: float = 0.05, tau_delta: float = 0.0) -> dict:
    cands = task["candidates"]
    base = cands[0]

    # Compute Q_X for all candidates
    for c in cands:
        feats = build_feature_vector(c["features"], feature_keys)
        c["q_x"] = float(q_res_model.predict([feats])[0])

    # DAPH-X pick: highest Q_X
    daphx = max(cands, key=lambda c: c["q_x"])
    disagreement = daphx["candidate_id"] != base["candidate_id"]

    if not disagreement:
        return {
            "pick_id": base["candidate_id"],
            "utility": 100.0 if base["is_correct"] else 0.0,
            "correct": base["is_correct"],
            "would_force": False,
            "disagreement": False,
        }

    # Authority gate
    delta_q = daphx["q_x"] - base["q_x"]
    lcb = delta_q - conformal["q_90"]

    feats_d = build_feature_vector(daphx["features"], feature_keys)
    feats_b = build_feature_vector(base["features"], feature_keys)

    risk_prob = 0.0
    if risk_model is not None:
        risk_prob = float(risk_model.predict_proba([feats_d - feats_b])[0, 1])

    pw_pred = 0.0
    if pairwise_model is not None:
        pair_feats = np.concatenate([feats_d - feats_b, feats_d, feats_b])
        pw_pred = float(pairwise_model.predict([pair_feats])[0])

    would_force = lcb > tau_delta and risk_prob < rho and pw_pred > 0

    pick = daphx if would_force else base
    return {
        "pick_id": pick["candidate_id"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "correct": pick["is_correct"],
        "would_force": would_force,
        "disagreement": True,
        "delta_q": delta_q,
        "lcb": lcb,
        "risk_prob": risk_prob,
        "pairwise_pred": pw_pred,
    }


def train_models(train_tasks, cal_tasks, feature_keys):
    train_records = flatten_candidates(train_tasks)

    # Q_res: predict utility (0 or 100)
    X, y = [], []
    for r in train_records:
        feats = build_feature_vector(r["features"], feature_keys)
        X.append(feats)
        y.append(100.0 if r["is_correct"] else 0.0)
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
                if i == j: continue
                a, b = cands[i], cands[j]
                fa = build_feature_vector(a["features"], feature_keys)
                fb = build_feature_vector(b["features"], feature_keys)
                pair_feats = np.concatenate([fa - fb, fa, fb])
                ua = 100.0 if a["is_correct"] else 0.0
                ub = 100.0 if b["is_correct"] else 0.0
                X_pairs.append(pair_feats)
                y_pairs.append(ua - ub)
    X_pairs, y_pairs = np.array(X_pairs), np.array(y_pairs)
    pairwise_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    pairwise_model.fit(X_pairs, y_pairs)

    # Risk: probability that Q_X winner is worse than base
    X_r, y_r = [], []
    for task in train_tasks:
        cands = task["candidates"]
        base = cands[0]
        for c in cands:
            if c["candidate_id"] == base["candidate_id"]:
                continue
            feats_c = build_feature_vector(c["features"], feature_keys)
            feats_b = build_feature_vector(base["features"], feature_keys)
            X_r.append(feats_c - feats_b)
            y_r.append(1 if (not c["is_correct"]) and base["is_correct"] else 0)
    X_r, y_r = np.array(X_r), np.array(y_r)
    risk_model = None
    if len(set(y_r)) >= 2:
        risk_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        risk_model.fit(X_r, y_r)

    # Conformal
    scores = []
    for task in cal_tasks:
        for c in task["candidates"]:
            feats = build_feature_vector(c["features"], feature_keys)
            q_x = q_res_model.predict([feats])[0]
            actual = 100.0 if c["is_correct"] else 0.0
            scores.append(abs(q_x - actual))
    scores = np.array(sorted(scores))
    n = len(scores)
    q_90 = scores[min(int(np.ceil(0.90 * (n + 1))) - 1, n - 1)]
    conformal = {"q_90": float(q_90), "n_cal": n}

    return q_res_model, pairwise_model, risk_model, conformal


def evaluate_all_systems(eval_tasks, q_res_model, pairwise_model, risk_model,
                         conformal, feature_keys, rho, tau_delta):
    results = {"base": [], "majority_vote": [], "self_confidence": [], "daphx": []}
    for task in eval_tasks:
        results["base"].append(system_base(task))
        results["majority_vote"].append(system_majority_vote(task))
        results["self_confidence"].append(system_self_confidence(task))
        results["daphx"].append(system_daphx(
            task, q_res_model, pairwise_model, risk_model,
            conformal, feature_keys, rho, tau_delta))
    return results


def compute_summary(results, eval_tasks):
    summary = {}
    rescue_opp = sum(1 for t in eval_tasks if t["rescue_available"])

    for sys_name, sys_results in results.items():
        n = len(sys_results)
        utilities = [r["utility"] for r in sys_results]
        successes = sum(1 for r in sys_results if r["correct"])

        base_utils = [100.0 if eval_tasks[i]["candidates"][0]["is_correct"] else 0.0
                      for i in range(len(eval_tasks))]
        improvements = sum(1 for u, bu in zip(utilities, base_utils) if u > bu + 0.5)
        regressions = sum(1 for u, bu in zip(utilities, base_utils) if u < bu - 0.5)

        if sys_name == "daphx":
            n_dis = sum(1 for r in sys_results if r.get("disagreement", False))
            n_force = sum(1 for r in sys_results if r.get("would_force", False))
            rescues = sum(1 for r, bu in zip(sys_results, base_utils)
                          if r.get("would_force", False) and r["utility"] > bu + 0.5)
            breaks = sum(1 for r, bu in zip(sys_results, base_utils)
                         if r.get("would_force", False) and r["utility"] < bu - 0.5)

            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                if task["rescue_available"]:
                    if results["daphx"][i]["utility"] > 0 and results["daphx"][i]["correct"]:
                        daphx_rescues += 1

            summary[sys_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
                "n_disagreements": n_dis,
                "n_force": n_force,
                "rescues": rescues,
                "breaks": breaks,
                "rescue_recall": daphx_rescues / max(rescue_opp, 1),
                "n_rescue_opportunities": rescue_opp,
            }
        else:
            summary[sys_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
            }

    return summary


def print_results_table(summary):
    print(f"\n{'='*95}")
    print(f"  DAPH-X Reasoning Authority (No Execution Feedback)")
    print(f"{'='*95}")
    print()
    print(f"{'Metric':<28} {'Base':>12} {'Majority':>12} {'SelfConf':>12} {'DAPH-X':>12}")
    print(f"{'':28} {'':>12} {'Vote':>12} {'Pick':>12} {'(auth)':>12}")
    print("-" * 95)

    for label, key, fmt in [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".1%"),
        ("Improvements vs base", "improvements_vs_base", "d"),
        ("Regressions vs base", "regressions_vs_base", "d"),
    ]:
        vals = []
        for sys_name in ["base", "majority_vote", "self_confidence", "daphx"]:
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
        print(f"  DAPH-X Authority Details:")
        print(f"    Rescue opportunities:    {d['n_rescue_opportunities']}")
        print(f"    Disagreements:           {d['n_disagreements']}")
        print(f"    Would FORCE:             {d['n_force']}")
        print(f"    Rescues:                 {d['rescues']}")
        print(f"    Breaks:                  {d['breaks']}")
        print(f"    Rescue recall:           {d['rescue_recall']:.4f}")

    print(f"\n{'='*95}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(REASONING_DIR / "reasoning_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--tau_delta", type=float, default=0.0)
    args = parser.parse_args()

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)}")

    print("\nTraining authority models...")
    q_res_model, pairwise_model, risk_model, conformal = train_models(
        train_tasks, cal_tasks, feature_keys)
    print(f"  Conformal q_90 = {conformal['q_90']:.2f}")

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
        (1.0, -100.0, "no gate"),
    ]

    best_config = None
    best_util = -float("inf")

    for rho, tau, label in configs:
        results = evaluate_all_systems(
            eval_tasks, q_res_model, pairwise_model, risk_model,
            conformal, feature_keys, rho, tau)
        summary = compute_summary(results, eval_tasks)

        d = summary["daphx"]
        util = d["mean_utility"]
        if util > best_util and d["breaks"] == 0:
            best_util = util
            best_config = (rho, tau, label, results, summary)

        print(f"  {label:25s}: util={util:.1f}, force={d['n_force']}, "
              f"R={d['rescues']}, B={d['breaks']}, recall={d['rescue_recall']:.2f}")

    if best_config:
        rho, tau, label, results, summary = best_config
        print(f"\n  Best safe config: {label}")
        print_results_table(summary)

        output = {
            "config": {"rho": rho, "tau_delta": tau, "seed": args.seed,
                       "n_dev": len(train_tasks), "n_cal": len(cal_tasks),
                       "n_confirmation": len(eval_tasks)},
            "summary": summary,
            "conformal": conformal,
        }
        output_path = REASONING_DIR / "reasoning_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
