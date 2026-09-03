#!/usr/bin/env python3
"""DAPH-X R11: Adaptive Candidate Acquisition.

The oracle@K study showed:
  Oracle@6  = 74%  (on 50-task subset)
  Oracle@12 = 78%  (on 50-task subset)
  MaxCal@6  = 60%
  MaxCal@12 = 64%

More candidates help. The question: can DAPH-X spend compute selectively
rather than uniformly?

Architecture:
  For each task, DAPH-X starts with 6 candidates and chooses an action:
    ANSWER(a_MaxCal)   — commit to the MaxCal pick, no extra compute
    GENERATE_1         — generate 1 more candidate, then re-evaluate
    GENERATE_N         — generate N more candidates, then re-evaluate
    VERIFY             — run verification on current best, then decide
    ABSTAIN            — commit to base (raw model pick)

The executive policy is learned: when should it spend more compute?

Comparison systems (all with same average compute budget):
  MaxCal@6     — 6 candidates, always (budget = 6)
  MaxCal@12    — 12 candidates, always (budget = 12)
  DAPH-X@avg8  — adaptive: ~8 avg candidates (6 + ~2 extra on hard tasks)
  DAPH-X@avg12 — adaptive: ~12 avg candidates

The scientific question:
  Can an executive spend extra compute selectively rather than uniformly?
  Can DAPH-X@avg8 beat MaxCal@8? Can DAPH-X@avg12 beat MaxCal@12?

Usage:
    python scripts/run_r11_evaluation.py \\
        --corpus experiments/daph_x/oracle/oracle_corpus_extended.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

R11_DIR = REPO_ROOT / "experiments/daph_x/r11"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, build_feature_vector,
    flatten_candidates, split_tasks, load_corpus,
)
from run_r9_evaluation import (
    add_multiround_verification_features, add_pairwise_features,
    add_answer_semantic_features, train_correctness_r9, calibrate_r9,
    predict_correctness_r9,
)


def compute_maxcal_picks(tasks, corr_model, corr_cal, fk):
    for task in tasks:
        for c in task["candidates"]:
            c["p_correct"] = predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk)
        task["maxcal_pick"] = max(task["candidates"], key=lambda c: c["p_correct"])
        task["maxcal_p"] = task["maxcal_pick"]["p_correct"]
    return tasks


def compute_state_features(task):
    """Features describing the current state — used to decide whether to
    spend more compute.

    These features capture:
    - MaxCal uncertainty (how confident is the best candidate?)
    - Agreement (do candidates agree?)
    - Verification (does verification support the best candidate?)
    - Diversity (how many unique answers?)
    """
    cands = task["candidates"]
    maxcal = task["maxcal_pick"]
    p_maxcal = task["maxcal_p"]

    # Uncertainty
    p_values = sorted([c["p_correct"] for c in cands], reverse=True)
    margin = p_values[0] - p_values[1] if len(p_values) > 1 else 1.0

    # Agreement
    answers = [c["answer"] for c in cands]
    n_unique = len(set(answers))
    max_agreement = max(Counter(answers).values())
    agreement_rate = max_agreement / len(cands)

    # Verification
    v_maxcal = maxcal.get("verification_score", 0.5)
    v_consistency = maxcal.get("verification_consistent", 0)

    # Confidence
    c_maxcal = maxcal["self_confidence"] / 100.0

    # Answer diversity
    n_cands = len(cands)

    # P(correct) distribution
    p_mean = np.mean([c["p_correct"] for c in cands])
    p_std = np.std([c["p_correct"] for c in cands])

    # Rescue opportunity indicator (does any candidate look good?)
    p_any_high = max(c["p_correct"] for c in cands)
    p_second = p_values[1] if len(p_values) > 1 else 0.0

    return {
        "n_cands": n_cands,
        "p_maxcal": p_maxcal,
        "p_margin": margin,
        "p_mean": p_mean,
        "p_std": p_std,
        "p_any_high": p_any_high,
        "p_second": p_second,
        "agreement_rate": agreement_rate,
        "n_unique": n_unique,
        "v_maxcal": v_maxcal,
        "v_consistency": v_consistency,
        "c_maxcal": c_maxcal,
    }


STATE_FEATURE_KEYS = [
    "n_cands", "p_maxcal", "p_margin", "p_mean", "p_std",
    "p_any_high", "p_second", "agreement_rate", "n_unique",
    "v_maxcal", "v_consistency", "c_maxcal",
]


def state_feature_vector(feats):
    return np.array([feats.get(k, 0.0) for k in STATE_FEATURE_KEYS])


# ─── Training: learn when extra compute helps ───

def train_compute_policy(train_tasks, corr_model, corr_cal, fk):
    """Learn a policy for when to generate more candidates.

    For each training task, simulate:
    1. Start with 6 candidates
    2. Compute state features
    3. Check if generating more candidates would have helped
       (i.e., did any of candidates 7-12 turn out correct AND MaxCal@6 was wrong?)

    Train a classifier: P(generate_more_helps | state_features)
    """
    examples = []
    for task in train_tasks:
        cands = task["candidates"]
        if len(cands) <= 6:
            continue

        # Simulate having only 6 candidates
        task_6 = dict(task)
        task_6["candidates"] = cands[:6]
        compute_maxcal_picks([task_6], corr_model, corr_cal, fk)

        state = compute_state_features(task_6)
        maxcal_6_correct = task_6["maxcal_pick"]["is_correct"]

        # Did extra candidates help?
        # Help = MaxCal@6 wrong AND some candidate in 7-12 is correct
        extra_cands = cands[6:]
        any_extra_correct = any(c["is_correct"] for c in extra_cands)
        generate_helped = (not maxcal_6_correct) and any_extra_correct

        # Also check: did extra candidates change MaxCal pick?
        task_12 = dict(task)
        task_12["candidates"] = cands[:12]
        compute_maxcal_picks([task_12], corr_model, corr_cal, fk)
        maxcal_12_correct = task_12["maxcal_pick"]["is_correct"]
        maxcal_changed = task_6["maxcal_pick"]["candidate_id"] != task_12["maxcal_pick"]["candidate_id"]

        examples.append({
            "state": state,
            "generate_helped": generate_helped,
            "maxcal_6_correct": maxcal_6_correct,
            "maxcal_12_correct": maxcal_12_correct,
            "maxcal_changed": maxcal_changed,
            "improvement": (1 if maxcal_12_correct and not maxcal_6_correct else 0)
                          - (1 if maxcal_6_correct and not maxcal_12_correct else 0),
        })

    # Train: P(generating more candidates will improve the answer)
    X, y = [], []
    for ex in examples:
        X.append(state_feature_vector(ex["state"]))
        y.append(1 if ex["improvement"] > 0 else 0)

    X, y = np.array(X), np.array(y)
    if y.sum() < 3:
        print(f"    WARNING: only {y.sum()} positive examples for compute policy")
        return None

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"    Compute policy: {n_pos} positive, {n_neg} negative examples")

    return model


def predict_compute_need(model, state):
    if model is None:
        return 0.0
    x = state_feature_vector(state).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


# ─── R11 Systems ───

def sys_maxcal_at_k(task, k):
    """MaxCal with first k candidates."""
    cands = task["candidates"][:k]
    pick = max(cands, key=lambda c: c.get("p_correct", c["self_confidence"] / 100.0))
    return {"correct": pick["is_correct"],
            "utility": 100.0 if pick["is_correct"] else 0.0,
            "n_candidates": k}


def sys_oracle_at_k(task, k):
    """Oracle with first k candidates (upper bound)."""
    cands = task["candidates"][:k]
    any_correct = any(c["is_correct"] for c in cands)
    return {"correct": any_correct,
            "utility": 100.0 if any_correct else 0.0,
            "n_candidates": k}


def sys_daphx_r11(task, compute_model, corr_model, corr_cal, fk,
                  p_threshold=0.3, max_extra=6):
    """DAPH-X R11: Adaptive Candidate Acquisition.

    1. Start with 6 candidates, compute MaxCal
    2. Compute state features
    3. If P(generate_helps) > threshold, use candidates 7-12
    4. Otherwise, commit to MaxCal@6
    """
    cands = task["candidates"]
    n_available = len(cands)

    # Phase 1: 6 candidates
    task_6 = dict(task)
    task_6["candidates"] = cands[:6]
    compute_maxcal_picks([task_6], corr_model, corr_cal, fk)
    state = compute_state_features(task_6)

    # Decide whether to generate more
    p_need = predict_compute_need(compute_model, state)

    if p_need > p_threshold and n_available > 6:
        # Use extra candidates (up to max_extra more)
        n_use = min(n_available, 6 + max_extra)
        task_n = dict(task)
        task_n["candidates"] = cands[:n_use]
        compute_maxcal_picks([task_n], corr_model, corr_cal, fk)
        pick = task_n["maxcal_pick"]
        return {"correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "n_candidates": n_use,
                "generated_extra": True}
    else:
        # Commit to MaxCal@6
        pick = task_6["maxcal_pick"]
        return {"correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "n_candidates": 6,
                "generated_extra": False}


def sys_daphx_r11_oracle(task, corr_model, corr_cal, fk, max_extra=6):
    """Oracle DAPH-X: knows which tasks need extra candidates.

    This is an upper bound on the adaptive policy — it generates extra
    candidates ONLY when MaxCal@6 is wrong AND extra candidates would help.
    """
    cands = task["candidates"]
    n_available = len(cands)

    task_6 = dict(task)
    task_6["candidates"] = cands[:6]
    compute_maxcal_picks([task_6], corr_model, corr_cal, fk)
    maxcal_6_correct = task_6["maxcal_pick"]["is_correct"]

    if maxcal_6_correct:
        # No need for extra candidates
        return {"correct": True, "utility": 100.0, "n_candidates": 6,
                "generated_extra": False}
    elif n_available > 6:
        # MaxCal@6 is wrong — try extra candidates
        n_use = min(n_available, 6 + max_extra)
        task_n = dict(task)
        task_n["candidates"] = cands[:n_use]
        compute_maxcal_picks([task_n], corr_model, corr_cal, fk)
        pick = task_n["maxcal_pick"]
        return {"correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "n_candidates": n_use,
                "generated_extra": True}
    else:
        return {"correct": False, "utility": 0.0, "n_candidates": 6,
                "generated_extra": False}


def evaluate_all(eval_tasks, compute_model, corr_model, corr_cal, fk):
    results = {name: [] for name in [
        "maxcal_6", "maxcal_8", "maxcal_12",
        "oracle_6", "oracle_8", "oracle_12",
        "daphx_r11_p20", "daphx_r11_p30", "daphx_r11_p50",
        "daphx_oracle",
    ]}

    for task in eval_tasks:
        results["maxcal_6"].append(sys_maxcal_at_k(task, 6))
        results["maxcal_8"].append(sys_maxcal_at_k(task, 8))
        results["maxcal_12"].append(sys_maxcal_at_k(task, 12))
        results["oracle_6"].append(sys_oracle_at_k(task, 6))
        results["oracle_8"].append(sys_oracle_at_k(task, 8))
        results["oracle_12"].append(sys_oracle_at_k(task, 12))

        for thresh, name in [(0.2, "daphx_r11_p20"), (0.3, "daphx_r11_p30"), (0.5, "daphx_r11_p50")]:
            results[name].append(sys_daphx_r11(
                task, compute_model, corr_model, corr_cal, fk, p_threshold=thresh))

        results["daphx_oracle"].append(sys_daphx_r11_oracle(
            task, corr_model, corr_cal, fk))

    return results


def summarize(results):
    summary = {}
    for name, res_list in results.items():
        utils = [r["utility"] for r in res_list]
        n_cands = [r["n_candidates"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        n_extra = sum(1 for r in res_list if r.get("generated_extra", False))

        summary[name] = {
            "mean_utility": float(np.mean(utils)),
            "success_rate": successes / len(res_list),
            "avg_candidates": float(np.mean(n_cands)),
            "n_generated_extra": n_extra,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R11_DIR / "r11_corpus.jsonl"))
    args = parser.parse_args()

    R11_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")
    n_with_extra = sum(1 for t in tasks if len(t["candidates"]) > 6)
    print(f"  Tasks with >6 candidates: {n_with_extra}")

    if n_with_extra == 0:
        print("ERROR: Need extended corpus with >6 candidates per task.")
        print("Run: python scripts/run_oracle_at_k.py --generate_extra --max_k 12")
        return

    # Feature extraction
    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    print("Computing enriched features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")

        # Train correctness model
        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, cal_records, feature_keys)

        # Compute MaxCal picks
        train_tasks = compute_maxcal_picks(train_tasks, corr_model, corr_cal, feature_keys)
        eval_tasks = compute_maxcal_picks(eval_tasks, corr_model, corr_cal, feature_keys)

        # Train compute policy
        print("  Training compute policy...")
        compute_model = train_compute_policy(train_tasks, corr_model, corr_cal, feature_keys)

        # Evaluate
        results = evaluate_all(eval_tasks, compute_model, corr_model, corr_cal, feature_keys)
        summary = summarize(results)

        print(f"{'System':<20} {'Util':>7} {'Acc':>7} {'AvgK':>7} {'Extra':>6}")
        print("-" * 55)
        for name in ["maxcal_6", "maxcal_8", "maxcal_12",
                     "oracle_6", "oracle_8", "oracle_12",
                     "daphx_r11_p20", "daphx_r11_p30", "daphx_r11_p50",
                     "daphx_oracle"]:
            s = summary[name]
            print(f"{name:<20} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['avg_candidates']:>7.1f} {s['n_generated_extra']:>6}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  R11 AGGREGATE: Adaptive Candidate Acquisition")
    print(f"{'='*100}")
    print(f"{'System':<20} {'Mean':>7} {'Std':>7} {'AvgK':>7} {'Extra':>6}")
    print("-" * 55)

    names = ["maxcal_6", "maxcal_8", "maxcal_12",
             "oracle_6", "oracle_8", "oracle_12",
             "daphx_r11_p20", "daphx_r11_p30", "daphx_r11_p50",
             "daphx_oracle"]
    agg = {}
    for name in names:
        utils = [all_results[s][name]["mean_utility"] for s in [42, 123, 7, 99, 2024]]
        avg_k = [all_results[s][name]["avg_candidates"] for s in [42, 123, 7, 99, 2024]]
        extra = [all_results[s][name]["n_generated_extra"] for s in [42, 123, 7, 99, 2024]]
        agg[name] = {"mean": float(np.mean(utils)), "std": float(np.std(utils)),
                     "avg_k": float(np.mean(avg_k)), "avg_extra": float(np.mean(extra))}
        print(f"{name:<20} {np.mean(utils):>7.1f} {np.std(utils):>7.1f} "
              f"{np.mean(avg_k):>7.1f} {np.mean(extra):>6.1f}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    print(f"  MaxCal@6:  {agg['maxcal_6']['mean']:.1f} (budget=6)")
    print(f"  MaxCal@8:  {agg['maxcal_8']['mean']:.1f} (budget=8)")
    print(f"  MaxCal@12: {agg['maxcal_12']['mean']:.1f} (budget=12)")
    print(f"  Oracle@6:  {agg['oracle_6']['mean']:.1f} (upper bound, K=6)")
    print(f"  Oracle@12: {agg['oracle_12']['mean']:.1f} (upper bound, K=12)")
    print()

    for name, label in [("daphx_r11_p20", "DAPH-X p=0.2"),
                        ("daphx_r11_p30", "DAPH-X p=0.3"),
                        ("daphx_r11_p50", "DAPH-X p=0.5"),
                        ("daphx_oracle", "DAPH-X Oracle")]:
        avg_k = agg[name]["avg_k"]
        # Find the MaxCal with similar budget
        if avg_k <= 7:
            baseline = agg["maxcal_6"]
            baseline_name = "MaxCal@6"
        elif avg_k <= 10:
            baseline = agg["maxcal_8"]
            baseline_name = "MaxCal@8"
        else:
            baseline = agg["maxcal_12"]
            baseline_name = "MaxCal@12"

        diff = agg[name]["mean"] - baseline["mean"]
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {label}: {agg[name]['mean']:.1f} (avg K={avg_k:.1f}) vs "
              f"{baseline_name}: {baseline['mean']:.1f} → {diff:+.1f} {verdict}")

    # Efficiency analysis
    print(f"\n  EFFICIENCY ANALYSIS:")
    print(f"  If DAPH-X@avg8 matches MaxCal@12, it saves 33% compute")
    print(f"  If DAPH-X@avg6.5 matches MaxCal@8, it saves 19% compute")

    output = {"aggregate": agg, "per_seed": all_results}
    output_path = R11_DIR / "r11_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
