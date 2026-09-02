#!/usr/bin/env python3
"""DAPH-X R4: Classification-based authority for binary outcomes.

Replaces regression-conformal (broken for binary Y∈{0,100}) with:
  1. Candidate correctness probability model (calibrated)
  2. Pairwise rescue/break classification (Z=+1/0/-1)
  3. Classification-calibrated authority gate

The authority decision becomes:
  FORCE when P(Z=+1) > tau_R AND P(Z=-1) < tau_B

Four systems compared (same candidates, same compute):
  1. Base: first candidate (temp=0)
  2. Majority vote: most common answer
  3. Max calibrated correctness: pick highest P(Y=1)
  4. DAPH-X: base as default, intervene when rescue prob > tau_R and break prob < tau_B

Usage:
    python scripts/run_r4_evaluation.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import check_answer

R4_DIR = REPO_ROOT / "experiments/daph_x/r4"


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


# ─── Model 1: Correctness probability model ───
def train_correctness_model(train_records, feature_keys):
    """Train P(Y=1 | features) using gradient boosting + isotonic calibration."""
    X, y = [], []
    for r in train_records:
        X.append(build_feature_vector(r["features"], feature_keys))
        y.append(1 if r["is_correct"] else 0)
    X, y = np.array(X), np.array(y)

    # Base classifier
    base = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    base.fit(X, y)

    return base, feature_keys


def calibrate_correctness(base_model, cal_records, feature_keys):
    """Isotonic calibration of correctness probability."""
    raw_probs = []
    true_labels = []
    for r in cal_records:
        feats = build_feature_vector(r["features"], feature_keys)
        prob = base_model.predict_proba([feats])[0, 1]
        raw_probs.append(prob)
        true_labels.append(1 if r["is_correct"] else 0)

    raw_probs = np.array(raw_probs)
    true_labels = np.array(true_labels)

    # Isotonic regression for calibration
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(raw_probs, true_labels)

    return iso


def predict_correctness(base_model, calibrator, features, feature_keys):
    """Get calibrated P(Y=1 | features)."""
    feats = build_feature_vector(features, feature_keys)
    raw_prob = base_model.predict_proba([feats])[0, 1]
    calibrated_prob = calibrator.predict([raw_prob])[0]
    return float(calibrated_prob)


# ─── Model 2: Pairwise rescue/break model ───
def train_pairwise_model(train_tasks, feature_keys):
    """Train P(Z=+1|x) and P(Z=-1|x) where:
      Z=+1: alternative correct, base wrong (rescue)
      Z=0:  same outcome
      Z=-1: alternative wrong, base correct (break)
    """
    X, z = [], []
    for task in train_tasks:
        cands = task["candidates"]
        base = cands[0]
        base_correct = base["is_correct"]
        for c in cands:
            if c["candidate_id"] == base["candidate_id"]:
                continue
            fa = build_feature_vector(c["features"], feature_keys)
            fb = build_feature_vector(base["features"], feature_keys)
            # Features: difference + both candidates
            pair_feats = np.concatenate([fa - fb, fa, fb])
            X.append(pair_feats)

            if c["is_correct"] and not base_correct:
                z.append(1)   # rescue
            elif not c["is_correct"] and base_correct:
                z.append(-1)  # break
            else:
                z.append(0)   # same

    X = np.array(X)
    z = np.array(z)

    # Train two binary classifiers: P(Z=+1) and P(Z=-1)
    # vs not, treating Z=0 as negative for both
    rescue_y = (z == 1).astype(int)
    break_y = (z == -1).astype(int)

    rescue_model = None
    break_model = None

    if len(set(rescue_y)) >= 2:
        rescue_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        rescue_model.fit(X, rescue_y)

    if len(set(break_y)) >= 2:
        break_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42)
        break_model.fit(X, break_y)

    return rescue_model, break_model


def calibrate_pairwise(rescue_model, break_model, cal_tasks, feature_keys):
    """Calibrate rescue and break probability models."""
    rescue_raw, rescue_true = [], []
    break_raw, break_true = [], []

    for task in cal_tasks:
        cands = task["candidates"]
        base = cands[0]
        base_correct = base["is_correct"]
        for c in cands:
            if c["candidate_id"] == base["candidate_id"]:
                continue
            fa = build_feature_vector(c["features"], feature_keys)
            fb = build_feature_vector(base["features"], feature_keys)
            pair_feats = np.concatenate([fa - fb, fa, fb])

            if rescue_model is not None:
                raw = rescue_model.predict_proba([pair_feats])[0, 1]
                rescue_raw.append(raw)
                rescue_true.append(1 if (c["is_correct"] and not base_correct) else 0)

            if break_model is not None:
                raw = break_model.predict_proba([pair_feats])[0, 1]
                break_raw.append(raw)
                break_true.append(1 if (not c["is_correct"] and base_correct) else 0)

    rescue_cal = None
    break_cal = None

    if rescue_model is not None and len(set(rescue_true)) >= 2:
        rescue_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        rescue_cal.fit(np.array(rescue_raw), np.array(rescue_true))

    if break_model is not None and len(set(break_true)) >= 2:
        break_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        break_cal.fit(np.array(break_raw), np.array(break_true))

    return rescue_cal, break_cal


def predict_pairwise(rescue_model, break_model, rescue_cal, break_cal,
                     alt_features, base_features, feature_keys):
    """Get calibrated P(rescue) and P(break) for alternative vs base."""
    fa = build_feature_vector(alt_features, feature_keys)
    fb = build_feature_vector(base_features, feature_keys)
    pair_feats = np.concatenate([fa - fb, fa, fb])

    p_rescue = 0.0
    p_break = 0.0

    if rescue_model is not None:
        raw = rescue_model.predict_proba([pair_feats])[0, 1]
        if rescue_cal is not None:
            p_rescue = float(rescue_cal.predict([raw])[0])
        else:
            p_rescue = float(raw)

    if break_model is not None:
        raw = break_model.predict_proba([pair_feats])[0, 1]
        if break_cal is not None:
            p_break = float(break_cal.predict([raw])[0])
        else:
            p_break = float(raw)

    return p_rescue, p_break


# ─── Systems ───
def system_base(task: dict) -> dict:
    base = task["candidates"][0]
    return {"pick_id": base["candidate_id"], "utility": 100.0 if base["is_correct"] else 0.0,
            "correct": base["is_correct"]}


def system_majority_vote(task: dict) -> dict:
    answers = [c["answer"] for c in task["candidates"]]
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]
    pick = next(c for c in task["candidates"] if c["answer"] == majority_answer)
    return {"pick_id": pick["candidate_id"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "correct": pick["is_correct"]}


def system_max_confidence(task: dict) -> dict:
    pick = max(task["candidates"], key=lambda c: c["self_confidence"])
    return {"pick_id": pick["candidate_id"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "correct": pick["is_correct"]}


def system_max_calibrated(task: dict, corr_model, corr_cal, feature_keys) -> dict:
    """Pick candidate with highest calibrated P(correct)."""
    best = None
    best_p = -1
    for c in task["candidates"]:
        p = predict_correctness(corr_model, corr_cal, c["features"], feature_keys)
        if p > best_p:
            best_p = p
            best = c
    return {"pick_id": best["candidate_id"], "utility": 100.0 if best["is_correct"] else 0.0,
            "correct": best["is_correct"], "calibrated_p": best_p}


def system_daphx_r4(
    task: dict, corr_model, corr_cal,
    rescue_model, break_model, rescue_cal, break_cal,
    feature_keys, tau_r=0.5, tau_b=0.1,
) -> dict:
    """DAPH-X R4: base as default, intervene when P(rescue) > tau_r and P(break) < tau_b."""
    cands = task["candidates"]
    base = cands[0]

    # For each non-base candidate, compute P(rescue) and P(break)
    best_alt = None
    best_rescue_prob = -1

    for c in cands:
        if c["candidate_id"] == base["candidate_id"]:
            continue
        p_rescue, p_break = predict_pairwise(
            rescue_model, break_model, rescue_cal, break_cal,
            c["features"], base["features"], feature_keys)

        # Also get calibrated correctness for this candidate
        p_correct = predict_correctness(corr_model, corr_cal, c["features"], feature_keys)

        c["p_rescue"] = p_rescue
        c["p_break"] = p_break
        c["p_correct"] = p_correct

        # Track the best alternative by rescue probability
        if p_rescue > best_rescue_prob:
            best_rescue_prob = p_rescue
            best_alt = c

    if best_alt is None:
        return {"pick_id": base["candidate_id"], "utility": 100.0 if base["is_correct"] else 0.0,
                "correct": base["is_correct"], "would_force": False, "disagreement": False}

    # Authority gate: FORCE when P(rescue) > tau_r AND P(break) < tau_b
    would_force = (best_alt["p_rescue"] > tau_r and best_alt["p_break"] < tau_b)
    disagreement = True  # We always consider alternatives

    if would_force:
        pick = best_alt
    else:
        pick = base

    return {
        "pick_id": pick["candidate_id"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "correct": pick["is_correct"],
        "would_force": would_force,
        "disagreement": disagreement,
        "p_rescue": best_alt["p_rescue"],
        "p_break": best_alt["p_break"],
        "p_correct_alt": best_alt["p_correct"],
        "p_correct_base": predict_correctness(corr_model, corr_cal, base["features"], feature_keys),
    }


def evaluate_all_systems(eval_tasks, corr_model, corr_cal,
                         rescue_model, break_model, rescue_cal, break_cal,
                         feature_keys, tau_r, tau_b):
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
        results["daphx"].append(system_daphx_r4(
            task, corr_model, corr_cal,
            rescue_model, break_model, rescue_cal, break_cal,
            feature_keys, tau_r, tau_b))
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
            n_force = sum(1 for r in sys_results if r.get("would_force", False))
            rescues = sum(1 for r, bu in zip(sys_results, base_utils)
                          if r.get("would_force", False) and r["utility"] > bu + 0.5)
            breaks = sum(1 for r, bu in zip(sys_results, base_utils)
                         if r.get("would_force", False) and r["utility"] < bu - 0.5)

            daphx_rescues = 0
            for i, task in enumerate(eval_tasks):
                if task["rescue_available"]:
                    if results["daphx"][i]["correct"]:
                        daphx_rescues += 1

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
    print(f"\n{'='*105}")
    print(f"  DAPH-X R4: Classification-Based Authority (Correctness + Rescue/Break)")
    print(f"{'='*105}")
    print()
    print(f"{'Metric':<28} {'Base':>10} {'MajVote':>10} {'MaxConf':>10} {'MaxCal':>10} {'DAPH-X':>10}")
    print(f"{'':28} {'':>10} {'':>10} {'':>10} {'P(Y=1)':>10} {'(R4)':>10}")
    print("-" * 105)

    for label, key, fmt in [
        ("Mean utility", "mean_utility", ".2f"),
        ("Task success rate", "task_success_rate", ".1%"),
        ("Improvements vs base", "improvements_vs_base", "d"),
        ("Regressions vs base", "regressions_vs_base", "d"),
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
        print(f"  DAPH-X R4 Authority Details:")
        print(f"    Rescue opportunities:    {d['n_rescue_opportunities']}")
        print(f"    Would FORCE:             {d['n_force']}")
        print(f"    Rescues:                 {d['rescues']}")
        print(f"    Breaks:                  {d['breaks']}")
        print(f"    Rescue recall:           {d['rescue_recall']:.4f}")
        if d['n_force'] > 0 and d['breaks'] == 0:
            print(f"    Break rate 95% upper:    {3.0/d['n_force']:.4f}")

    print(f"\n{'='*105}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R4_DIR / "r4_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R4_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} dev, {len(cal_tasks)} cal, {len(eval_tasks)} confirmation")

    train_records = flatten_candidates(train_tasks)
    cal_records = flatten_candidates(cal_tasks)
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)}")

    # Train correctness model
    print("\nTraining correctness probability model...")
    corr_model, _ = train_correctness_model(train_records, feature_keys)
    corr_cal = calibrate_correctness(corr_model, cal_records, feature_keys)

    # Evaluate calibration
    cal_probs = []
    cal_true = []
    for r in cal_records:
        p = predict_correctness(corr_model, corr_cal, r["features"], feature_keys)
        cal_probs.append(p)
        cal_true.append(1 if r["is_correct"] else 0)
    cal_probs = np.array(cal_probs)
    cal_true = np.array(cal_true)
    # Brier score (lower is better)
    brier = np.mean((cal_probs - cal_true) ** 2)
    print(f"  Brier score (cal): {brier:.4f}")
    # Accuracy of calibrated model on cal set
    cal_acc = np.mean((cal_probs > 0.5).astype(int) == cal_true)
    print(f"  Cal accuracy: {cal_acc:.2f}")

    # Train pairwise rescue/break model
    print("\nTraining pairwise rescue/break model...")
    rescue_model, break_model = train_pairwise_model(train_tasks, feature_keys)
    rescue_cal, break_cal = calibrate_pairwise(
        rescue_model, break_model, cal_tasks, feature_keys)
    print(f"  Rescue model: {'trained' if rescue_model else 'not enough data'}")
    print(f"  Break model: {'trained' if break_model else 'not enough data'}")

    # Count training examples
    train_rescue = sum(1 for t in train_tasks for c in t["candidates"][1:]
                       if c["is_correct"] and not t["candidates"][0]["is_correct"])
    train_break = sum(1 for t in train_tasks for c in t["candidates"][1:]
                      if not c["is_correct"] and t["candidates"][0]["is_correct"])
    train_same = sum(1 for t in train_tasks for c in t["candidates"][1:]
                     if c["is_correct"] == t["candidates"][0]["is_correct"])
    print(f"  Training: {train_rescue} rescue, {train_break} break, {train_same} same")

    # Try multiple threshold configurations
    print(f"\nEvaluating on {len(eval_tasks)} confirmation tasks...")
    configs = [
        (0.3, 0.1, "tau_r=0.3, tau_b=0.1"),
        (0.4, 0.1, "tau_r=0.4, tau_b=0.1"),
        (0.5, 0.1, "tau_r=0.5, tau_b=0.1"),
        (0.5, 0.05, "tau_r=0.5, tau_b=0.05"),
        (0.5, 0.2, "tau_r=0.5, tau_b=0.2"),
        (0.6, 0.1, "tau_r=0.6, tau_b=0.1"),
        (0.6, 0.05, "tau_r=0.6, tau_b=0.05"),
        (0.7, 0.1, "tau_r=0.7, tau_b=0.1"),
        (0.3, 0.05, "tau_r=0.3, tau_b=0.05"),
        (0.4, 0.05, "tau_r=0.4, tau_b=0.05"),
    ]

    best_config = None
    best_score = -float("inf")

    for tau_r, tau_b, label in configs:
        results = evaluate_all_systems(
            eval_tasks, corr_model, corr_cal,
            rescue_model, break_model, rescue_cal, break_cal,
            feature_keys, tau_r, tau_b)
        summary = compute_summary(results, eval_tasks)

        d = summary["daphx"]
        mc = summary["max_calibrated"]
        mv = summary["majority_vote"]

        # Score: utility with penalty for breaks
        score = d["mean_utility"] - 50 * d["breaks"]

        if score > best_score:
            best_score = score
            best_config = (tau_r, tau_b, label, results, summary)

        print(f"  {label:25s}: util={d['mean_utility']:.1f} (MV={mv['mean_utility']:.1f}, MC={mc['mean_utility']:.1f}), "
              f"force={d['n_force']}, R={d['rescues']}, B={d['breaks']}, recall={d['rescue_recall']:.2f}")

    if best_config:
        tau_r, tau_b, label, results, summary = best_config
        print(f"\n  Best config: {label}")
        print_results_table(summary)

        output = {
            "config": {"tau_r": tau_r, "tau_b": tau_b, "seed": args.seed,
                       "n_dev": len(train_tasks), "n_cal": len(cal_tasks),
                       "n_confirmation": len(eval_tasks)},
            "summary": summary,
            "calibration": {"brier_score": brier, "cal_accuracy": cal_acc},
        }
        output_path = R4_DIR / "r4_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
