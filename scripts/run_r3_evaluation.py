#!/usr/bin/env python3
"""DAPH-X R3: Focused on breaking probe ties.

Key insight from R2: DAPH-X matches but doesn't beat the probe baseline.
The Q_X model can't distinguish candidates that pass the same probes.

R3 approach: Train a specialized "tie-breaker" model that only operates
when multiple candidates have the same probe pass rate. The model is
trained ONLY on probe-tie pairs, with the target being which candidate
does better on the remaining (hidden) tests.

This is a much more focused learning problem:
  - Input: feature difference between two candidates with same probe rate
  - Target: which one passes more hidden tests
  - Output: probability that candidate A is better than B

The authority decision becomes:
  1. Run probes on all candidates
  2. Pick probe winner (highest probe pass rate)
  3. If there's a tie at the top probe rate, use the tie-breaker model
  4. If the tie-breaker is confident enough, pick its winner
  5. Otherwise, fall back to Q_MB tie-break

Usage:
    python scripts/run_r3_evaluation.py \\
        --corpus experiments/daph_x/r1/r1_corpus_all.jsonl \\
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
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import get_task
from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb

R3_DIR = REPO_ROOT / "experiments/daph_x/r3"


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
    return max(cands, key=lambda c: (c["probe"]["probe_pass_rate"], c["q_mb"]))


# ─── Systems 1-3 (same as R2) ───
def system_base(task: dict) -> dict:
    base = task["candidates"][0]
    return {"pick_id": base["candidate_id"], "utility": base["full"]["utility"],
            "tests_passed": base["full"]["tests_passed"], "tests_total": base["full"]["tests_total"],
            "probe_cost": 0}


def system_probe_baseline(task: dict, n_probe: int) -> dict:
    winner = pick_probe_winner(task["candidates"])
    return {"pick_id": winner["candidate_id"], "utility": winner["full"]["utility"],
            "tests_passed": winner["full"]["tests_passed"], "tests_total": winner["full"]["tests_total"],
            "probe_cost": n_probe * len(task["candidates"])}


def system_simple_reranker(task: dict, n_probe: int) -> dict:
    best = max(task["candidates"], key=lambda c: c["q_mb"] + 10.0 * c["probe"]["probe_pass_rate"])
    return {"pick_id": best["candidate_id"], "utility": best["full"]["utility"],
            "tests_passed": best["full"]["tests_passed"], "tests_total": best["full"]["tests_total"],
            "probe_cost": n_probe * len(task["candidates"])}


# ─── System 4: DAPH-X R3 (probe winner + tie-breaker) ───
def system_daphx_r3(
    task: dict, tiebreaker_model, conformal: dict,
    feature_keys: list[str], n_probe: int,
    confidence_threshold: float = 0.65,
) -> dict:
    """Probe winner as default. Use tie-breaker model only on probe ties."""
    cands = task["candidates"]
    probe_winner = pick_probe_winner(cands)

    # Group candidates by probe pass rate
    by_probe = defaultdict(list)
    for c in cands:
        by_probe[c["probe"]["probe_pass_rate"]].append(c)

    top_rate = max(by_probe.keys())
    top_cands = by_probe[top_rate]

    # If only one candidate at top probe rate, no tie to break
    if len(top_cands) == 1:
        return {
            "pick_id": probe_winner["candidate_id"],
            "utility": probe_winner["full"]["utility"],
            "tests_passed": probe_winner["full"]["tests_passed"],
            "tests_total": probe_winner["full"]["tests_total"],
            "probe_cost": n_probe * len(cands),
            "would_force": False,
            "had_tie": False,
            "n_tied": 1,
        }

    # There's a tie at the top probe rate — use tie-breaker model
    # Compare all pairs among top candidates
    best_pick = top_cands[0]
    best_score = -float("inf")

    for c in top_cands:
        # Score = probability that this candidate is better than the others
        scores = []
        for other in top_cands:
            if c["candidate_id"] == other["candidate_id"]:
                continue
            fa = build_feature_vector(c["features"], c["probe"], feature_keys)
            fb = build_feature_vector(other["features"], other["probe"], feature_keys)
            pair_feats = np.concatenate([fa - fb, fa, fb])
            # Probability that c is better than other
            prob_better = tiebreaker_model.predict_proba([pair_feats])[0, 1]
            scores.append(prob_better)
        avg_score = np.mean(scores) if scores else 0.5
        if avg_score > best_score:
            best_score = avg_score
            best_pick = c

    # Only intervene if confident enough
    would_force = best_score > confidence_threshold and best_pick["candidate_id"] != probe_winner["candidate_id"]

    if would_force:
        pick = best_pick
    else:
        pick = probe_winner  # Fall back to probe winner (Q_MB tie-break)

    return {
        "pick_id": pick["candidate_id"],
        "utility": pick["full"]["utility"],
        "tests_passed": pick["full"]["tests_passed"],
        "tests_total": pick["full"]["tests_total"],
        "probe_cost": n_probe * len(cands),
        "would_force": would_force,
        "had_tie": True,
        "n_tied": len(top_cands),
        "tiebreaker_confidence": best_score,
        "probe_winner_id": probe_winner["candidate_id"],
        "tiebreaker_pick_id": best_pick["candidate_id"],
        "probe_winner_utility": probe_winner["full"]["utility"],
        "tiebreaker_pick_utility": best_pick["full"]["utility"],
    }


def train_tiebreaker(train_tasks, feature_keys, n_probe):
    """Train a tie-breaker model on probe-tie pairs only.

    For each task, find candidates with the same probe pass rate.
    For each pair, create a training example:
      features = [fa - fb, fa, fb]
      label = 1 if a has higher eval utility than b, else 0
    """
    X, y = [], []

    for task in train_tasks:
        cands = task["candidates"]
        # Group by probe pass rate
        by_probe = defaultdict(list)
        for c in cands:
            by_probe[c["probe"]["probe_pass_rate"]].append(c)

        # For each group with ties, create pairwise examples
        for rate, group in by_probe.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j:
                        continue
                    a, b = group[i], group[j]
                    fa = build_feature_vector(a["features"], a["probe"], feature_keys)
                    fb = build_feature_vector(b["features"], b["probe"], feature_keys)
                    pair_feats = np.concatenate([fa - fb, fa, fb])
                    ua = compute_eval_utility(a, n_probe)
                    ub = compute_eval_utility(b, n_probe)
                    X.append(pair_feats)
                    y.append(1 if ua > ub + 0.5 else (0 if ub > ua + 0.5 else 0.5))

    X = np.array(X)
    y = np.array(y)

    # Filter out ties (label = 0.5) for classification
    mask = y != 0.5
    X_cls = X[mask]
    y_cls = y[mask].astype(int)

    if len(set(y_cls)) < 2 or len(X_cls) < 10:
        # Not enough data — fall back to regression
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        model.fit(X, y)
        return model, "regression", len(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X_cls, y_cls)
    return model, "classification", len(X_cls)


def calibrate_tiebreaker(cal_tasks, tiebreaker_model, feature_keys, n_probe):
    """Calibrate the tie-breaker: measure how often it's right on calibration data."""
    correct = 0
    total = 0
    intervened = 0
    intervened_correct = 0
    intervened_harmful = 0

    for task in cal_tasks:
        cands = task["candidates"]
        by_probe = defaultdict(list)
        for c in cands:
            by_probe[c["probe"]["probe_pass_rate"]].append(c)

        top_rate = max(by_probe.keys())
        top_cands = by_probe[top_rate]

        if len(top_cands) < 2:
            continue

        # Use the model to pick among tied candidates
        best_pick = top_cands[0]
        best_score = -float("inf")
        for c in top_cands:
            scores = []
            for other in top_cands:
                if c["candidate_id"] == other["candidate_id"]:
                    continue
                fa = build_feature_vector(c["features"], c["probe"], feature_keys)
                fb = build_feature_vector(other["features"], other["probe"], feature_keys)
                pair_feats = np.concatenate([fa - fb, fa, fb])
                if hasattr(tiebreaker_model, 'predict_proba'):
                    prob = tiebreaker_model.predict_proba([pair_feats])[0, 1]
                else:
                    prob = tiebreaker_model.predict([pair_feats])[0]
                scores.append(prob)
            avg = np.mean(scores) if scores else 0.5
            if avg > best_score:
                best_score = avg
                best_pick = c

        probe_winner = pick_probe_winner(cands)
        actual_best = max(top_cands, key=lambda c: c["full"]["utility"])

        total += 1
        if best_pick["candidate_id"] == actual_best["candidate_id"]:
            correct += 1

        if best_pick["candidate_id"] != probe_winner["candidate_id"]:
            intervened += 1
            if best_pick["full"]["utility"] > probe_winner["full"]["utility"] + 0.5:
                intervened_correct += 1
            elif best_pick["full"]["utility"] < probe_winner["full"]["utility"] - 0.5:
                intervened_harmful += 1

    return {
        "n_cal_ties": total,
        "accuracy": correct / max(total, 1),
        "n_intervened": intervened,
        "n_correct_interventions": intervened_correct,
        "n_harmful_interventions": intervened_harmful,
    }


def evaluate_all_systems(eval_tasks, tiebreaker_model, conformal, feature_keys, n_probe,
                         confidence_threshold):
    results = {"base": [], "probe_baseline": [], "simple_reranker": [], "daphx": []}
    for task in eval_tasks:
        results["base"].append(system_base(task))
        results["probe_baseline"].append(system_probe_baseline(task, n_probe))
        results["simple_reranker"].append(system_simple_reranker(task, n_probe))
        results["daphx"].append(system_daphx_r3(
            task, tiebreaker_model, conformal, feature_keys, n_probe,
            confidence_threshold))
    return results


def compute_summary(results: dict, eval_tasks: list, n_probe: int) -> dict:
    summary = {}

    # Find probe-tie cases (the opportunity for DAPH-X)
    probe_tie_tasks = 0
    probe_tie_with_gap = 0
    for task in eval_tasks:
        cands = task["candidates"]
        by_probe = defaultdict(list)
        for c in cands:
            by_probe[c["probe"]["probe_pass_rate"]].append(c)
        top_rate = max(by_probe.keys())
        top_cands = by_probe[top_rate]
        if len(top_cands) > 1:
            probe_tie_tasks += 1
            utils = [c["full"]["utility"] for c in top_cands]
            if max(utils) - min(utils) > 5:
                probe_tie_with_gap += 1

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
            n_tie = sum(1 for r in system_results if r.get("had_tie", False))
            n_force = sum(1 for r in system_results if r.get("would_force", False))

            # Rescues and breaks relative to probe baseline
            probe_utils = [system_probe_baseline(t, n_probe)["utility"] for t in eval_tasks]
            rescues = sum(1 for r, pu in zip(system_results, probe_utils)
                          if r.get("would_force", False) and r["utility"] > pu + 0.5)
            breaks = sum(1 for r, pu in zip(system_results, probe_utils)
                         if r.get("would_force", False) and r["utility"] < pu - 0.5)

            # Rescue recall: of probe-tie-with-gap cases, how many did DAPH-X fix?
            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                cands = task["candidates"]
                by_probe = defaultdict(list)
                for c in cands:
                    by_probe[c["probe"]["probe_pass_rate"]].append(c)
                top_rate = max(by_probe.keys())
                top_cands = by_probe[top_rate]
                if len(top_cands) > 1:
                    utils = [c["full"]["utility"] for c in top_cands]
                    if max(utils) - min(utils) > 5:
                        pw = pick_probe_winner(cands)
                        if results["daphx"][i]["utility"] > pw["full"]["utility"] + 0.5:
                            daphx_rescues += 1

            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
                "n_probe_ties": n_tie,
                "n_probe_ties_with_gap": probe_tie_with_gap,
                "n_force": n_force,
                "rescues_vs_probe": rescues,
                "breaks_vs_probe": breaks,
                "rescue_recall_vs_probe": daphx_rescues / max(probe_tie_with_gap, 1),
            }
        else:
            summary[system_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
            }

    return summary


def print_results_table(summary: dict):
    print(f"\n{'='*95}")
    print(f"  DAPH-X R3: Probe-Tie Breaker")
    print(f"{'='*95}")
    print()
    print(f"{'Metric':<28} {'Base':>12} {'Probe':>12} {'Reranker':>12} {'DAPH-X':>12}")
    print(f"{'':28} {'':>12} {'Baseline':>12} {'(Q_MB+p)':>12} {'(R3 tie)':>12}")
    print("-" * 95)

    for label, key, fmt in [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".1%"),
        ("Improvements vs base", "improvements_vs_base", "d"),
        ("Regressions vs base", "regressions_vs_base", "d"),
    ]:
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
        print(f"  DAPH-X R3 Tie-Breaker Details:")
        print(f"    Probe-tie tasks:           {d['n_probe_ties']}")
        print(f"    Probe-tie with gap > 5:    {d['n_probe_ties_with_gap']}")
        print(f"    Would FORCE (intervene):   {d['n_force']}")
        print(f"    Rescues vs probe:          {d['rescues_vs_probe']}")
        print(f"    Breaks vs probe:           {d['breaks_vs_probe']}")
        print(f"    Rescue recall vs probe:    {d['rescue_recall_vs_probe']:.4f}")

    print(f"\n{'='*95}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R3_DIR / "r3_corpus.jsonl"))
    parser.add_argument("--n_probe_tests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R3_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)} + 4 probe = {len(feature_keys)+4}")

    print("\nTraining tie-breaker model (on probe-tie pairs only)...")
    tiebreaker_model, model_type, n_train = train_tiebreaker(
        train_tasks, feature_keys, args.n_probe_tests)
    print(f"  Model type: {model_type}, trained on {n_train} pairs")

    print("\nCalibrating tie-breaker...")
    cal_results = calibrate_tiebreaker(cal_tasks, tiebreaker_model, feature_keys, args.n_probe_tests)
    print(f"  Calibration ties: {cal_results['n_cal_ties']}")
    print(f"  Accuracy: {cal_results['accuracy']:.2f}")
    print(f"  Intervened: {cal_results['n_intervened']}")
    print(f"  Correct interventions: {cal_results['n_correct_interventions']}")
    print(f"  Harmful interventions: {cal_results['n_harmful_interventions']}")

    # Try different confidence thresholds
    print(f"\nEvaluating on {len(eval_tasks)} confirmation tasks...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]

    best_config = None
    best_util = -float("inf")

    for thresh in thresholds:
        results = evaluate_all_systems(
            eval_tasks, tiebreaker_model, cal_results, feature_keys,
            args.n_probe_tests, thresh)
        summary = compute_summary(results, eval_tasks, args.n_probe_tests)

        d = summary["daphx"]
        pb = summary["probe_baseline"]
        util = d["mean_utility"]
        if util > best_util and d["breaks_vs_probe"] == 0:
            best_util = util
            best_config = (thresh, results, summary)

        print(f"  thresh={thresh:.2f}: util={util:.1f} (probe={pb['mean_utility']:.1f}), "
              f"force={d['n_force']}, R={d['rescues_vs_probe']}, B={d['breaks_vs_probe']}, "
              f"ties={d['n_probe_ties']}, tie_gap={d['n_probe_ties_with_gap']}")

    if best_config:
        thresh, results, summary = best_config
        print(f"\n  Best safe config: confidence_threshold={thresh}")
        print_results_table(summary)

        output = {
            "config": {"n_probe_tests": args.n_probe_tests, "confidence_threshold": thresh,
                       "seed": args.seed, "n_dev": len(train_tasks),
                       "n_cal": len(cal_tasks), "n_confirmation": len(eval_tasks)},
            "summary": summary,
            "calibration": cal_results,
        }
        output_path = R3_DIR / "r3_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
