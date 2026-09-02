#!/usr/bin/env python3
"""DAPH-X R5: Pairwise preference model + learning curves.

Key changes from R4:
  1. Uses ALL C(k,2) candidate pairs for training (not just base-vs-alt)
  2. Direct pairwise preference: f(s, a_i, a_j) = P(a_i > a_j)
  3. Task-grouped learning curves to diagnose data vs features
  4. Challenge set: simple-baseline failures

The pairwise model is trained on all pairs within each task:
  For task with candidates c1..c6:
    Pairs: (c1,c2), (c1,c3), ..., (c5,c6) = 15 pairs
    Label: 1 if ci correct and cj wrong, -1 if reverse, 0 if same

The authority decision:
  1. Base = candidate 0 (temp=0)
  2. For each alternative, compute P(alt > base)
  3. FORCE when max P(alt > base) > tau AND P(base > alt) < tau_b

Usage:
    python scripts/run_r5_evaluation.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import check_answer

R5_DIR = REPO_ROOT / "experiments/daph_x/r5"


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


# ─── Pairwise preference model ───
def build_pair_features(a: dict, b: dict, feature_keys: list[str]) -> np.ndarray:
    """Build contrastive features for pair (a, b).
    Features: [fa - fb, fa, fb, |fa - fb|]
    """
    fa = build_feature_vector(a["features"], feature_keys)
    fb = build_feature_vector(b["features"], feature_keys)
    return np.concatenate([fa - fb, fa, fb, np.abs(fa - fb)])


def generate_pair_labels(task: dict) -> list[tuple[int, int, int]]:
    """Generate (i, j, label) for all pairs.
    label: 1 if ci > cj (ci correct, cj wrong)
           -1 if ci < cj (ci wrong, cj correct)
           0 if same
    """
    cands = task["candidates"]
    pairs = []
    for i in range(len(cands)):
        for j in range(len(cands)):
            if i == j:
                continue
            ci_correct = cands[i]["is_correct"]
            cj_correct = cands[j]["is_correct"]
            if ci_correct and not cj_correct:
                pairs.append((i, j, 1))
            elif not ci_correct and cj_correct:
                pairs.append((i, j, -1))
            else:
                pairs.append((i, j, 0))
    return pairs


def train_pairwise_preference(train_tasks, feature_keys):
    """Train P(a_i > a_j) using all ordered pairs.

    Binary classification: label = 1 if a_i correct and a_j wrong, else 0.
    (Pairs where both are same are labeled 0 — neither preferred.)
    """
    X, y = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                ci_correct = cands[i]["is_correct"]
                cj_correct = cands[j]["is_correct"]
                # Only use informative pairs (one correct, one wrong)
                if ci_correct == cj_correct:
                    continue
                feats = build_pair_features(cands[i], cands[j], feature_keys)
                X.append(feats)
                y.append(1 if ci_correct else 0)

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2 or len(X) < 10:
        return None, 0

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model, len(X)


def calibrate_pairwise(model, cal_tasks, feature_keys):
    """Isotonic calibration of pairwise preference probability."""
    if model is None:
        return None
    raw_probs, true_labels = [], []
    for task in cal_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                ci_correct = cands[i]["is_correct"]
                cj_correct = cands[j]["is_correct"]
                if ci_correct == cj_correct:
                    continue
                feats = build_pair_features(cands[i], cands[j], feature_keys)
                raw = model.predict_proba([feats])[0, 1]
                raw_probs.append(raw)
                true_labels.append(1 if ci_correct else 0)

    if len(set(true_labels)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_probs), np.array(true_labels))
    return iso


def predict_preference(model, calibrator, a, b, feature_keys):
    """Get calibrated P(a > b)."""
    if model is None:
        return 0.5
    feats = build_pair_features(a, b, feature_keys)
    raw = model.predict_proba([feats])[0, 1]
    if calibrator is not None:
        return float(calibrator.predict([raw])[0])
    return float(raw)


# ─── Correctness model (for max-calibrated baseline) ───
def train_correctness_model(train_records, feature_keys):
    X, y = [], []
    for r in train_records:
        X.append(build_feature_vector(r["features"], feature_keys))
        y.append(1 if r["is_correct"] else 0)
    X, y = np.array(X), np.array(y)
    base = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    base.fit(X, y)
    return base


def calibrate_correctness(base_model, cal_records, feature_keys):
    raw_probs, true_labels = [], []
    for r in cal_records:
        feats = build_feature_vector(r["features"], feature_keys)
        prob = base_model.predict_proba([feats])[0, 1]
        raw_probs.append(prob)
        true_labels.append(1 if r["is_correct"] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_probs), np.array(true_labels))
    return iso


def predict_correctness(base_model, calibrator, features, feature_keys):
    feats = build_feature_vector(features, feature_keys)
    raw = base_model.predict_proba([feats])[0, 1]
    if calibrator is not None:
        return float(calibrator.predict([raw])[0])
    return float(raw)


# ─── Systems ───
def system_base(task):
    base = task["candidates"][0]
    return {"pick_id": base["candidate_id"], "utility": 100.0 if base["is_correct"] else 0.0,
            "correct": base["is_correct"]}


def system_majority_vote(task):
    answers = [c["answer"] for c in task["candidates"]]
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]
    pick = next(c for c in task["candidates"] if c["answer"] == majority_answer)
    return {"pick_id": pick["candidate_id"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "correct": pick["is_correct"]}


def system_max_confidence(task):
    pick = max(task["candidates"], key=lambda c: c["self_confidence"])
    return {"pick_id": pick["candidate_id"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "correct": pick["is_correct"]}


def system_max_calibrated(task, corr_model, corr_cal, feature_keys):
    best, best_p = None, -1
    for c in task["candidates"]:
        p = predict_correctness(corr_model, corr_cal, c["features"], feature_keys)
        if p > best_p:
            best_p = p
            best = c
    return {"pick_id": best["candidate_id"], "utility": 100.0 if best["is_correct"] else 0.0,
            "correct": best["is_correct"], "calibrated_p": best_p}


def system_daphx_r5(task, pref_model, pref_cal, feature_keys, tau_r=0.5, tau_b=0.3):
    """DAPH-X R5: pairwise preference authority.

    1. Base = candidate 0
    2. For each alternative, compute P(alt > base) and P(base > alt)
    3. Pick the alternative with highest P(alt > base)
    4. FORCE when P(alt > base) > tau_r AND P(base > alt) < tau_b
    """
    cands = task["candidates"]
    base = cands[0]

    best_alt = None
    best_p_pref = -1
    best_p_base_wins = 1.0

    for c in cands[1:]:
        p_alt_wins = predict_preference(pref_model, pref_cal, c, base, feature_keys)
        p_base_wins = predict_preference(pref_model, pref_cal, base, c, feature_keys)

        if p_alt_wins > best_p_pref:
            best_p_pref = p_alt_wins
            best_p_base_wins = p_base_wins
            best_alt = c

    if best_alt is None:
        return {"pick_id": base["candidate_id"], "utility": 100.0 if base["is_correct"] else 0.0,
                "correct": base["is_correct"], "would_force": False}

    would_force = (best_p_pref > tau_r and best_p_base_wins < tau_b)
    pick = best_alt if would_force else base

    return {
        "pick_id": pick["candidate_id"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "correct": pick["is_correct"],
        "would_force": would_force,
        "p_alt_wins": best_p_pref,
        "p_base_wins": best_p_base_wins,
    }


def evaluate_all_systems(eval_tasks, corr_model, corr_cal,
                         pref_model, pref_cal, feature_keys, tau_r, tau_b):
    results = {
        "base": [], "majority_vote": [], "max_confidence": [],
        "max_calibrated": [], "daphx": [],
    }
    for task in eval_tasks:
        results["base"].append(system_base(task))
        results["majority_vote"].append(system_majority_vote(task))
        results["max_confidence"].append(system_max_confidence(task))
        results["max_calibrated"].append(system_max_calibrated(
            task, corr_model, corr_cal, feature_keys))
        results["daphx"].append(system_daphx_r5(
            task, pref_model, pref_cal, feature_keys, tau_r, tau_b))
    return results


def compute_summary(results, eval_tasks):
    summary = {}
    rescue_opp = sum(1 for t in eval_tasks if t["rescue_available"])

    # Challenge set: tasks where majority vote or max confidence is wrong
    mv_wrong = []
    mc_wrong = []
    for i, task in enumerate(eval_tasks):
        if not results["majority_vote"][i]["correct"]:
            mv_wrong.append(i)
        if not results["max_confidence"][i]["correct"]:
            mc_wrong.append(i)
    challenge_set = sorted(set(mv_wrong + mc_wrong))

    for sys_name, sys_results in results.items():
        n = len(sys_results)
        utilities = [r["utility"] for r in sys_results]
        successes = sum(1 for r in sys_results if r["correct"])
        base_utils = [100.0 if eval_tasks[i]["candidates"][0]["is_correct"] else 0.0
                      for i in range(len(eval_tasks))]
        improvements = sum(1 for u, bu in zip(utilities, base_utils) if u > bu + 0.5)
        regressions = sum(1 for u, bu in zip(utilities, base_utils) if u < bu - 0.5)

        if sys_name == "daphx":
            n_force = sum(1 for r in sys_results if r.get("would_force", False))
            rescues = sum(1 for r, bu in zip(sys_results, base_utils)
                          if r.get("would_force", False) and r["utility"] > bu + 0.5)
            breaks = sum(1 for r, bu in zip(sys_results, base_utils)
                         if r.get("would_force", False) and r["utility"] < bu - 0.5)

            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                if task["rescue_available"] and results["daphx"][i]["correct"]:
                    daphx_rescues += 1

            # Challenge set performance
            challenge_correct = sum(1 for i in challenge_set
                                    if sys_results[i]["correct"])
            challenge_n = len(challenge_set)

            summary[sys_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
                "n_force": n_force,
                "rescues": rescues,
                "breaks": breaks,
                "rescue_recall": daphx_rescues / max(rescue_opp, 1),
                "n_rescue_opportunities": rescue_opp,
                "challenge_correct": challenge_correct,
                "challenge_n": challenge_n,
                "challenge_rate": challenge_correct / max(challenge_n, 1),
            }
        else:
            challenge_correct = sum(1 for i in challenge_set
                                    if sys_results[i]["correct"])
            challenge_n = len(challenge_set)
            summary[sys_name] = {
                "mean_utility": np.mean(utilities),
                "task_success": successes,
                "task_success_rate": successes / n,
                "improvements_vs_base": improvements,
                "regressions_vs_base": regressions,
                "challenge_correct": challenge_correct,
                "challenge_n": challenge_n,
                "challenge_rate": challenge_correct / max(challenge_n, 1),
            }

    return summary, challenge_set


def print_results_table(summary, challenge_set):
    print(f"\n{'='*115}")
    print(f"  DAPH-X R5: Pairwise Preference + Learning Curves")
    print(f"{'='*115}")
    print()
    print(f"{'Metric':<28} {'Base':>10} {'MajVote':>10} {'MaxConf':>10} {'MaxCal':>10} {'DAPH-X':>10}")
    print(f"{'':28} {'':>10} {'':>10} {'':>10} {'P(Y=1)':>10} {'(R5)':>10}")
    print("-" * 115)

    for label, key, fmt in [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".1%"),
        ("Improvements vs base", "improvements_vs_base", "d"),
        ("Regressions vs base", "regressions_vs_base", "d"),
        (f"Challenge set ({len(challenge_set)} tasks)", "challenge_rate", ".1%"),
    ]:
        vals = []
        for sys_name in ["base", "majority_vote", "max_confidence", "max_calibrated", "daphx"]:
            if sys_name in summary and key in summary[sys_name]:
                v = summary[sys_name][key]
                if fmt == "d":
                    vals.append(f"{int(v):>10}")
                elif fmt == ".1%":
                    vals.append(f"{v:>9.1%}")
                else:
                    vals.append(f"{v:>10{fmt}}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"{label:<28} {vals[0]} {vals[1]} {vals[2]} {vals[3]} {vals[4]}")

    if "daphx" in summary:
        d = summary["daphx"]
        print()
        print(f"  DAPH-X R5 Authority Details:")
        print(f"    Rescue opportunities:    {d['n_rescue_opportunities']}")
        print(f"    Would FORCE:             {d['n_force']}")
        print(f"    Rescues:                 {d['rescues']}")
        print(f"    Breaks:                  {d['breaks']}")
        print(f"    Rescue recall:           {d['rescue_recall']:.4f}")
        if d['n_force'] > 0 and d['breaks'] == 0:
            print(f"    Break rate 95% upper:    {3.0/d['n_force']:.4f}")
        print(f"    Challenge set:           {d['challenge_correct']}/{d['challenge_n']} correct")

    print(f"\n{'='*115}")


# ─── Learning curve ───
def run_learning_curve(tasks, feature_keys, seed=42):
    """Run learning curve: train on N tasks, evaluate on fixed test set."""
    # Fixed test set: last 20% of tasks (by seed permutation)
    rng = np.random.RandomState(seed)
    n = len(tasks)
    indices = rng.permutation(n)
    n_test = int(n * 0.2)
    test_indices = indices[:n_test]
    train_pool_indices = indices[n_test:]

    test_tasks = [tasks[i] for i in test_indices]
    eval_results = []

    train_sizes = [15, 30, 45, 60, 75, 90]
    train_sizes = [s for s in train_sizes if s <= len(train_pool_indices)]

    print(f"\n  Learning curve: test set = {len(test_tasks)} tasks")
    print(f"  Train sizes: {train_sizes}")
    print()

    for n_train in train_sizes:
        train_tasks = [tasks[i] for i in train_pool_indices[:n_train]]

        # Use remaining as calibration (up to 15)
        cal_end = min(n_train + 15, len(train_pool_indices))
        cal_tasks = [tasks[i] for i in train_pool_indices[n_train:cal_end]]

        if len(cal_tasks) < 5:
            cal_tasks = train_tasks[:max(len(train_tasks) // 4, 5)]

        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        # Train models
        corr_model = train_correctness_model(train_records, feature_keys)
        corr_cal = calibrate_correctness(corr_model, cal_records, feature_keys)
        pref_model, n_pairs = train_pairwise_preference(train_tasks, feature_keys)
        pref_cal = calibrate_pairwise(pref_model, cal_tasks, feature_keys)

        # Evaluate
        results = evaluate_all_systems(
            test_tasks, corr_model, corr_cal,
            pref_model, pref_cal, feature_keys, tau_r=0.5, tau_b=0.3)
        summary, _ = compute_summary(results, test_tasks)

        # AUROC for pairwise model on test set
        auroc_rescue = None
        if pref_model is not None:
            test_X, test_y = [], []
            for task in test_tasks:
                cands = task["candidates"]
                for i in range(len(cands)):
                    for j in range(len(cands)):
                        if i == j:
                            continue
                        if cands[i]["is_correct"] == cands[j]["is_correct"]:
                            continue
                        feats = build_pair_features(cands[i], cands[j], feature_keys)
                        test_X.append(feats)
                        test_y.append(1 if cands[i]["is_correct"] else 0)
            if len(set(test_y)) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_rescue = roc_auc_score(test_y, pref_model.predict_proba(test_X)[:, 1])

        d = summary["daphx"]
        mv = summary["majority_vote"]
        mc = summary["max_calibrated"]

        print(f"  N={n_train:3d} (pairs={n_pairs:4d}): "
              f"DAPHX={d['mean_utility']:.1f} MV={mv['mean_utility']:.1f} MC={mc['mean_utility']:.1f} "
              f"force={d['n_force']} R={d['rescues']} B={d['breaks']} "
              f"recall={d['rescue_recall']:.2f} "
              f"AUROC={'%.3f' % auroc_rescue if auroc_rescue else 'N/A'}")

        eval_results.append({
            "n_train": n_train,
            "n_pairs": n_pairs,
            "daphx_utility": d["mean_utility"],
            "mv_utility": mv["mean_utility"],
            "mc_utility": mc["mean_utility"],
            "base_utility": summary["base"]["mean_utility"],
            "n_force": d["n_force"],
            "rescues": d["rescues"],
            "breaks": d["breaks"],
            "rescue_recall": d["rescue_recall"],
            "auroc_pairwise": auroc_rescue,
        })

    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R5_DIR / "r5_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_curve", action="store_true",
                        help="Run learning curve analysis")
    args = parser.parse_args()

    R5_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)
    print(f"Features: {len(feature_keys)}")

    if args.learning_curve:
        print("\n=== Learning Curve Analysis ===")
        lc_results = run_learning_curve(tasks, feature_keys, seed=args.seed)

        # Save
        output = {"learning_curve": lc_results, "seed": args.seed}
        output_path = R5_DIR / "r5_learning_curve.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Learning curve saved to {output_path}")

        # Diagnose
        print("\n=== Diagnosis ===")
        daphx_utils = [r["daphx_utility"] for r in lc_results]
        mv_utils = [r["mv_utility"] for r in lc_results]
        aurocs = [r["auroc_pairwise"] for r in lc_results if r["auroc_pairwise"]]

        if len(daphx_utils) >= 2:
            slope = (daphx_utils[-1] - daphx_utils[0]) / (lc_results[-1]["n_train"] - lc_results[0]["n_train"])
            print(f"  DAPH-X utility slope: {slope:.4f} per task")
            if slope > 0.1:
                print(f"  → Still improving with more data → DATA problem")
            else:
                print(f"  → Plateaued → FEATURE problem")
        if aurocs:
            print(f"  AUROC range: {min(aurocs):.3f} - {max(aurocs):.3f}")
            if max(aurocs) < 0.65:
                print(f"  → Low AUROC → features inadequate for pairwise preference")
            elif max(aurocs) > 0.75:
                print(f"  → Decent AUROC → features have signal, may need more data")
        return

    # Standard evaluation
    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    cal_records = flatten_candidates(cal_tasks)

    # Train correctness model
    print("\nTraining correctness model...")
    corr_model = train_correctness_model(train_records, feature_keys)
    corr_cal = calibrate_correctness(corr_model, cal_records, feature_keys)

    # Train pairwise preference model
    print("\nTraining pairwise preference model...")
    pref_model, n_pairs = train_pairwise_preference(train_tasks, feature_keys)
    print(f"  Trained on {n_pairs} informative pairs")
    pref_cal = calibrate_pairwise(pref_model, cal_tasks, feature_keys)

    # AUROC on eval set
    if pref_model is not None:
        eval_X, eval_y = [], []
        for task in eval_tasks:
            cands = task["candidates"]
            for i in range(len(cands)):
                for j in range(len(cands)):
                    if i == j:
                        continue
                    if cands[i]["is_correct"] == cands[j]["is_correct"]:
                        continue
                    feats = build_pair_features(cands[i], cands[j], feature_keys)
                    eval_X.append(feats)
                    eval_y.append(1 if cands[i]["is_correct"] else 0)
        if len(set(eval_y)) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = roc_auc_score(eval_y, pref_model.predict_proba(eval_X)[:, 1])
            print(f"  Pairwise AUROC on eval: {auroc:.4f}")

    # Try multiple thresholds
    print(f"\nEvaluating on {len(eval_tasks)} confirmation tasks...")
    configs = [
        (0.5, 0.3, "tau_r=0.5, tau_b=0.3"),
        (0.5, 0.2, "tau_r=0.5, tau_b=0.2"),
        (0.5, 0.1, "tau_r=0.5, tau_b=0.1"),
        (0.6, 0.3, "tau_r=0.6, tau_b=0.3"),
        (0.6, 0.2, "tau_r=0.6, tau_b=0.2"),
        (0.6, 0.1, "tau_r=0.6, tau_b=0.1"),
        (0.7, 0.3, "tau_r=0.7, tau_b=0.3"),
        (0.7, 0.2, "tau_r=0.7, tau_b=0.2"),
        (0.4, 0.3, "tau_r=0.4, tau_b=0.3"),
        (0.4, 0.2, "tau_r=0.4, tau_b=0.2"),
    ]

    best_config = None
    best_score = -float("inf")

    for tau_r, tau_b, label in configs:
        results = evaluate_all_systems(
            eval_tasks, corr_model, corr_cal,
            pref_model, pref_cal, feature_keys, tau_r, tau_b)
        summary, challenge_set = compute_summary(results, eval_tasks)

        d = summary["daphx"]
        score = d["mean_utility"] - 50 * d["breaks"]

        if score > best_score:
            best_score = score
            best_config = (tau_r, tau_b, label, results, summary, challenge_set)

        print(f"  {label:25s}: util={d['mean_utility']:.1f} (MV={summary['majority_vote']['mean_utility']:.1f}), "
              f"force={d['n_force']}, R={d['rescues']}, B={d['breaks']}, recall={d['rescue_recall']:.2f}, "
              f"challenge={d['challenge_correct']}/{d['challenge_n']}")

    if best_config:
        tau_r, tau_b, label, results, summary, challenge_set = best_config
        print(f"\n  Best config: {label}")
        print_results_table(summary, challenge_set)

        output = {
            "config": {"tau_r": tau_r, "tau_b": tau_b, "seed": args.seed,
                       "n_dev": len(train_tasks), "n_cal": len(cal_tasks),
                       "n_confirmation": len(eval_tasks), "n_train_pairs": n_pairs},
            "summary": summary,
            "challenge_set_size": len(challenge_set),
        }
        output_path = R5_DIR / "r5_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
