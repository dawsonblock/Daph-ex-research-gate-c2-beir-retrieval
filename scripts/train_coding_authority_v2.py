#!/usr/bin/env python3
"""Train coding authority with execution-feedback features.

Key insight: static code features (AST metrics) cannot predict test outcomes
well enough. This script adds "probe test" features — running each candidate
on the first K tests and using those results as features.

This simulates a real-world scenario where the authority can run a few quick
checks before deciding which candidate to commit to.

Probe tests: first 2 tests of each task (used as features)
Evaluation: remaining tests (used for utility labels)

Usage:
    python scripts/train_coding_authority_v2.py \\
        --corpus experiments/daph_x/coding/coding_corpus_combined.jsonl \\
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

from daph_x.coding.tasks import get_all_tasks, get_task
from daph_x.coding.code_executor import execute_solution
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

CODING_DIR = REPO_ROOT / "experiments/daph_x/coding"


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def get_feature_keys(records: list[dict]) -> list[str]:
    all_keys = set()
    for r in records:
        all_keys.update(r["features"].keys())
    return sorted(all_keys)


def build_feature_vector(features: dict, probe_results: dict, feature_keys: list[str]) -> np.ndarray:
    """Build feature vector including probe test results."""
    base = [float(features.get(k, 0.0)) for k in feature_keys]
    probe = [
        probe_results.get("probe_pass_rate", 0.0),
        probe_results.get("probe_n_passed", 0),
        probe_results.get("probe_n_total", 0),
        probe_results.get("probe_has_error", 0.0),
        probe_results.get("probe_timeout", 0.0),
    ]
    return np.array(base + probe)


def run_probe_tests(task_id: str, solution_code: str, n_probe: int = 2) -> dict:
    """Run candidate on first n_probe tests and return results as features."""
    task = get_task(task_id)
    if task is None:
        return {"probe_pass_rate": 0.0, "probe_n_passed": 0, "probe_n_total": 0,
                "probe_has_error": 1.0, "probe_timeout": 0.0}

    # Create a modified task with only the first n_probe tests
    from daph_x.coding.tasks import CodingTask
    probe_task = CodingTask(
        task_id=task.task_id,
        description=task.description,
        function_name=task.function_name,
        signature=task.signature,
        docstring=task.docstring,
        difficulty=task.difficulty,
        tests=task.tests[:n_probe],
        imports=task.imports,
        common_errors=task.common_errors,
    )

    result = execute_solution(probe_task, solution_code, timeout_seconds=5.0)
    return {
        "probe_pass_rate": result.pass_rate,
        "probe_n_passed": result.tests_passed,
        "probe_n_total": result.tests_total,
        "probe_has_error": 1.0 if result.error else 0.0,
        "probe_timeout": 1.0 if result.error and "Timeout" in str(result.error) else 0.0,
    }


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


def flatten_candidates(tasks: list[dict]) -> list[dict]:
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


def compute_reduced_utility(record: dict, n_probe: int) -> float:
    """Compute utility based on tests AFTER the probe tests."""
    task = get_task(record["task_id"])
    if task is None:
        return record["utility"]

    total_tests = len(task.tests)
    eval_tests = total_tests - n_probe
    if eval_tests <= 0:
        return record["utility"]

    # The record has tests_passed and tests_total for ALL tests
    # We need to figure out how many of the non-probe tests passed
    # Since we don't have per-test results in the corpus, we approximate:
    # if probe passed all, then remaining passed = tests_passed - n_probe
    # This is an approximation — the actual per-test results would be better
    total_passed = record["tests_passed"]
    # Assume probe tests are the first n_probe tests
    # If the candidate passed all tests, it passed all probe tests
    # If it passed some, we need to estimate
    probe_passed = min(total_passed, n_probe)
    eval_passed = total_passed - probe_passed
    eval_pass_rate = eval_passed / eval_tests
    return eval_pass_rate * 100.0


def train_q_res_v2(train_records, feature_keys, n_probe):
    """Train Q_res with probe features."""
    X = []
    y = []
    for r in train_records:
        probe = run_probe_tests(r["task_id"], r["solution_code"], n_probe)
        feats = build_feature_vector(r["features"], probe, feature_keys)
        X.append(feats)
        # Use reduced utility (excluding probe tests)
        y.append(compute_reduced_utility(r, n_probe))

    X = np.array(X)
    y = np.array(y)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X, y)
    return model


def train_pairwise_v2(train_tasks, feature_keys, n_probe):
    """Train pairwise model with probe features."""
    X_pairs = []
    y_pairs = []

    for task in train_tasks:
        cands = task["candidates"]
        # Precompute probe results for all candidates
        probes = {}
        for c in cands:
            probes[c["candidate_id"]] = run_probe_tests(c["task_id"], c["solution_code"], n_probe)

        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                a, b = cands[i], cands[j]
                fa = build_feature_vector(a["features"], probes[a["candidate_id"]], feature_keys)
                fb = build_feature_vector(b["features"], probes[b["candidate_id"]], feature_keys)

                pair_feats = np.concatenate([
                    fa - fb, fa, fb,
                    [a["q_mb"] - b["q_mb"]],
                    [a["q_mb"], b["q_mb"]],
                    # Probe difference features
                    [probes[a["candidate_id"]]["probe_pass_rate"] - probes[b["candidate_id"]]["probe_pass_rate"]],
                    [probes[a["candidate_id"]]["probe_n_passed"] - probes[b["candidate_id"]]["probe_n_passed"]],
                ])
                delta_u = compute_reduced_utility(a, n_probe) - compute_reduced_utility(b, n_probe)
                X_pairs.append(pair_feats)
                y_pairs.append(delta_u)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X_pairs, y_pairs)
    return model, len(X_pairs)


def train_risk_v2(train_records, feature_keys, n_probe):
    """Train risk model with probe features."""
    X = []
    y = []
    by_task = defaultdict(list)
    for r in train_records:
        by_task[r["task_id"]].append(r)

    for task_id, cands in by_task.items():
        base = cands[0]
        base_util = compute_reduced_utility(base, n_probe)
        for c in cands:
            probe = run_probe_tests(c["task_id"], c["solution_code"], n_probe)
            feats = build_feature_vector(c["features"], probe, feature_keys)
            X.append(feats)
            c_util = compute_reduced_utility(c, n_probe)
            y.append(1 if c_util < base_util - 0.5 else 0)

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2:
        return None, len(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X, y)
    return model, len(X)


def conformal_calibrate_v2(cal_tasks, q_res_model, feature_keys, n_probe):
    scores = []
    for task in cal_tasks:
        for c in task["candidates"]:
            probe = run_probe_tests(c["task_id"], c["solution_code"], n_probe)
            feats = build_feature_vector(c["features"], probe, feature_keys)
            q_x = q_res_model.predict([feats])[0]
            actual_u = compute_reduced_utility(c, n_probe)
            scores.append(abs(q_x - actual_u))

    scores = np.array(sorted(scores))
    n = len(scores)
    q_90 = scores[min(int(np.ceil(0.90 * (n + 1))) - 1, n - 1)]
    q_50 = scores[min(int(np.ceil(0.50 * (n + 1))) - 1, n - 1)]
    return {"q_90": float(q_90), "q_50": float(q_50), "n_cal": n}


def evaluate_v2(eval_tasks, q_res_model, pairwise_model, risk_model,
                conformal, feature_keys, n_probe, rho=0.05, tau_delta=-50.0):
    results = []
    for task in eval_tasks:
        cands = task["candidates"]
        base = cands[0]
        base_util = compute_reduced_utility(base, n_probe)

        # Compute Q_X with probe features
        probes = {}
        for c in cands:
            probes[c["candidate_id"]] = run_probe_tests(c["task_id"], c["solution_code"], n_probe)
            feats = build_feature_vector(c["features"], probes[c["candidate_id"]], feature_keys)
            c["q_x_v2"] = q_res_model.predict([feats])[0]

        daphx = max(cands, key=lambda c: c["q_x_v2"])
        disagreement = daphx["candidate_id"] != base["candidate_id"]

        if not disagreement:
            results.append({"task_id": task["task_id"], "disagreement": False,
                           "would_force": False, "delta_u": 0.0, "outcome": "AGREE"})
            continue

        delta_q = daphx["q_x_v2"] - base["q_x_v2"]
        lcb = delta_q - conformal["q_90"]

        risk_prob = 0.0
        if risk_model is not None:
            feats_d = build_feature_vector(daphx["features"], probes[daphx["candidate_id"]], feature_keys)
            risk_prob = float(risk_model.predict_proba([feats_d])[0, 1])

        pw_pred = 0.0
        if pairwise_model is not None:
            fa = build_feature_vector(daphx["features"], probes[daphx["candidate_id"]], feature_keys)
            fb = build_feature_vector(base["features"], probes[base["candidate_id"]], feature_keys)
            pair_feats = np.concatenate([
                fa - fb, fa, fb,
                [daphx["q_mb"] - base["q_mb"]],
                [daphx["q_mb"], base["q_mb"]],
                [probes[daphx["candidate_id"]]["probe_pass_rate"] - probes[base["candidate_id"]]["probe_pass_rate"]],
                [probes[daphx["candidate_id"]]["probe_n_passed"] - probes[base["candidate_id"]]["probe_n_passed"]],
            ])
            pw_pred = float(pairwise_model.predict([pair_feats])[0])

        would_force = lcb > tau_delta and risk_prob < rho and pw_pred > 0
        daphx_util = compute_reduced_utility(daphx, n_probe)
        delta_u = daphx_util - base_util

        if would_force:
            outcome = "RESCUE" if delta_u > 0.5 else ("BREAK" if delta_u < -0.5 else "TIE_FORCE")
        else:
            outcome = "ABSTAIN"

        results.append({
            "task_id": task["task_id"], "disagreement": True, "would_force": would_force,
            "delta_u": delta_u, "delta_q_hat": delta_q, "lcb": lcb,
            "risk_prob": risk_prob, "pairwise_pred": pw_pred, "outcome": outcome,
            "daphx_id": daphx["candidate_id"], "base_id": base["candidate_id"],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CODING_DIR / "coding_corpus_combined.jsonl"))
    parser.add_argument("--n_probe_tests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} train, {len(cal_tasks)} cal, {len(eval_tasks)} eval")

    train_records = flatten_candidates(train_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)} + 5 probe = {len(feature_keys)+5}")

    print(f"\nRunning probe tests (n_probe={args.n_probe_tests})...")

    print("\nTraining Q_res v2 (with probe features)...")
    q_res_model = train_q_res_v2(train_records, feature_keys, args.n_probe_tests)

    # Evaluate Q_res
    for split_name, records in [("train", train_records), ("eval", flatten_candidates(eval_tasks))]:
        X = []
        y = []
        for r in records:
            probe = run_probe_tests(r["task_id"], r["solution_code"], args.n_probe_tests)
            feats = build_feature_vector(r["features"], probe, feature_keys)
            X.append(feats)
            y.append(compute_reduced_utility(r, args.n_probe_tests))
        X = np.array(X)
        y = np.array(y)
        q_x = q_res_model.predict(X)
        mae = np.mean(np.abs(q_x - y))
        print(f"  {split_name}: MAE(Q_X_v2) = {mae:.2f}")

    print("\nTraining pairwise v2...")
    pairwise_model, n_pairs = train_pairwise_v2(train_tasks, feature_keys, args.n_probe_tests)
    print(f"  Trained on {n_pairs} pairs")

    print("\nTraining risk v2...")
    risk_model, n_risk = train_risk_v2(train_records, feature_keys, args.n_probe_tests)

    print("\nCalibrating conformal v2...")
    conformal = conformal_calibrate_v2(cal_tasks, q_res_model, feature_keys, args.n_probe_tests)
    print(f"  q_90 = {conformal['q_90']:.2f}, q_50 = {conformal['q_50']:.2f}")

    print(f"\nEvaluating authority v2...")
    results = evaluate_v2(eval_tasks, q_res_model, pairwise_model, risk_model,
                          conformal, feature_keys, args.n_probe_tests,
                          rho=0.05, tau_delta=-50.0)

    n_disagree = sum(1 for r in results if r["disagreement"])
    n_force = sum(1 for r in results if r["would_force"])
    n_rescue = sum(1 for r in results if r["outcome"] == "RESCUE")
    n_break = sum(1 for r in results if r["outcome"] == "BREAK")
    n_tie = sum(1 for r in results if r["outcome"] == "TIE_FORCE")
    n_abstain = sum(1 for r in results if r["outcome"] == "ABSTAIN")

    print(f"\n{'='*60}")
    print(f"  AUTHORITY V2 RESULTS (with probe features)")
    print(f"{'='*60}")
    print(f"  Eval tasks: {len(eval_tasks)}")
    print(f"  Disagreements: {n_disagree}")
    print(f"  Would FORCE: {n_force}")
    print(f"  Rescues: {n_rescue}")
    print(f"  Breaks: {n_break}")
    print(f"  Ties (forced): {n_tie}")
    print(f"  Abstained: {n_abstain}")

    if n_force > 0:
        print(f"  Force precision: {n_rescue / n_force:.4f}")
        print(f"  Break rate: {n_break / n_force:.4f}")
        if n_break == 0:
            print(f"  Break rate 95% upper: {3.0 / n_force:.4f}")

    # Q_MB-only baseline
    print(f"\n  Q_MB-only baseline:")
    qmb_dis = qmb_res = qmb_brk = qmb_tie = 0
    for task in eval_tasks:
        cands = task["candidates"]
        base = cands[0]
        qmb_pick = max(cands, key=lambda c: c["q_mb"])
        if qmb_pick["candidate_id"] != base["candidate_id"]:
            qmb_dis += 1
            du = compute_reduced_utility(qmb_pick, args.n_probe_tests) - compute_reduced_utility(base, args.n_probe_tests)
            if du > 0.5: qmb_res += 1
            elif du < -0.5: qmb_brk += 1
            else: qmb_tie += 1
    print(f"    Disagree={qmb_dis}, R={qmb_res}, B={qmb_brk}, T={qmb_tie}")

    # No-gate baseline
    print(f"\n  No-gate baseline (always force on Q_X_v2 disagreement):")
    nogate_res = nogate_brk = nogate_tie = 0
    for task in eval_tasks:
        cands = task["candidates"]
        base = cands[0]
        probes = {}
        for c in cands:
            probes[c["candidate_id"]] = run_probe_tests(c["task_id"], c["solution_code"], args.n_probe_tests)
            feats = build_feature_vector(c["features"], probes[c["candidate_id"]], feature_keys)
            c["q_x_v2"] = q_res_model.predict([feats])[0]
        daphx = max(cands, key=lambda c: c["q_x_v2"])
        if daphx["candidate_id"] != base["candidate_id"]:
            du = compute_reduced_utility(daphx, args.n_probe_tests) - compute_reduced_utility(base, args.n_probe_tests)
            if du > 0.5: nogate_res += 1
            elif du < -0.5: nogate_brk += 1
            else: nogate_tie += 1
    print(f"    R={nogate_res}, B={nogate_brk}, T={nogate_tie}")

    # Save
    import joblib
    model_path = CODING_DIR / "coding_authority_models_v2.pkl"
    joblib.dump({
        "q_res_model": q_res_model,
        "pairwise_model": pairwise_model,
        "risk_model": risk_model,
        "conformal": conformal,
        "feature_keys": feature_keys,
        "n_probe": args.n_probe_tests,
    }, model_path)
    print(f"\n  Models saved to {model_path}")

    results_path = CODING_DIR / "coding_authority_results_v2.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {"n_probe": args.n_probe_tests, "rho": 0.05, "tau_delta": -50.0},
            "results": results,
            "summary": {
                "n_disagree": n_disagree, "n_force": n_force,
                "n_rescue": n_rescue, "n_break": n_break, "n_tie": n_tie,
                "qmb_baseline": {"n_disagree": qmb_dis, "n_rescue": qmb_res, "n_break": qmb_brk},
            },
            "conformal": conformal,
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
