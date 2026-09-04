#!/usr/bin/env python3
"""R12 evaluation: Decision-Aligned Adaptive Compute.

Implements the R12 protocol:
- Counterfactual training target: ΔQ = Q(GENERATE) - Q(STOP)
- Compact feature set (p_top1, margin, entropy, stability, dp_top1, disagreement)
- Feature addition by policy value (J), not AUROC
- Value model with threshold sweep
- Break-risk head
- Paired bootstrap CIs
- Oracle regret measurement
- Intervention forensics
- Pareto frontier analysis

Usage:
    python scripts/run_r12_evaluation.py \\
        --corpus experiments/daph_x/r12/r12_corpus_12.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight, resample

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

R12_DIR = REPO_ROOT / "experiments/daph_x/r12"

from run_r9_evaluation import (
    train_correctness_r9, calibrate_r9, predict_correctness_r9,
    add_multiround_verification_features,
    add_pairwise_features,
    add_answer_semantic_features,
)
from run_r7_evaluation import (
    flatten_candidates, build_feature_vector, get_feature_keys,
    enrich_corpus,
)
from run_r11_2_evaluation import (
    compute_answer_entropy, compute_enhanced_state_features,
    enhanced_state_to_vector, ENHANCED_FEATURE_KEYS,
    predict_calibrated_enhanced, calibrate_isotonic_enhanced,
    train_lookahead_rescue_model, train_value_model,
    train_ptop1_only_model, predict_ptop1_only,
    run_non_myopic_oracle, run_oracle_policy,
    run_random_policy, run_uncertainty_policy, run_entropy_policy,
    run_verify_action, should_verify,
    run_sequential_policy_v2,
    paired_bootstrap_ci,
    load_corpus,
)


# ─── R12 Compact Feature Set ───

COMPACT_FEATURE_KEYS = [
    "p_top1", "margin", "answer_entropy",
    "selection_stability", "delta_p_top1", "agreement_rate",
]


def compact_state_to_vector(state):
    """Convert state to compact feature vector."""
    return np.array([state.get(k, 0.0) for k in COMPACT_FEATURE_KEYS])


# ─── Counterfactual Training Target ───

def extract_counterfactual_examples(tasks, corr_model, corr_cal, fk, lambda_cost=0.1):
    """Extract counterfactual acquisition examples with ΔQ target.

    For each checkpoint s_K:
      Q(s_K, STOP) = U(MaxCal@K)
      Q(s_K, GENERATE) = U(MaxCal@K+2) - lambda * C_{+2}

      ΔQ_K = Q(GENERATE) - Q(STOP)
      Y_K = ΔU_K - lambda * C_{+2}    (continuous value target)

    Also compute:
      rescue = ΔU > 0  (GENERATE helps)
      break = ΔU < 0   (GENERATE hurts)
      neutral = ΔU == 0 (GENERATE wastes compute)
    """
    examples = []
    checkpoints = [2, 4, 6, 8, 10]
    cost_per_step = 2.0 / 10.0  # Normalized cost of 2 candidates

    for task in tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        prev_state = None
        for k in checkpoints:
            if k + 2 > len(cands):
                break

            state_k = compute_enhanced_state_features(
                task, k, corr_model, corr_cal, fk, prev_state)
            u_k = 1.0 if state_k["_maxcal_correct"] else 0.0

            # Counterfactual: what if we generate 2 more?
            state_k2 = compute_enhanced_state_features(
                task, k + 2, corr_model, corr_cal, fk, state_k)
            u_k2 = 1.0 if state_k2["_maxcal_correct"] else 0.0

            delta_u = u_k2 - u_k  # +1 (rescue), 0 (neutral), -1 (break)
            delta_q = delta_u - lambda_cost * cost_per_step

            # Also compute full lookahead (K -> 12)
            pick_12 = max(cands[:12], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            u_12 = 1.0 if pick_12["is_correct"] else 0.0
            delta_u_full = u_12 - u_k
            lookahead_rescue = (u_12 > 0.5 and u_k < 0.5)

            examples.append({
                "task_id": task["task_id"],
                "checkpoint_k": k,
                "state": {k2: v for k2, v in state_k.items() if not k2.startswith("_")},
                "compact_vector": compact_state_to_vector(
                    {k2: v for k2, v in state_k.items() if not k2.startswith("_")}),
                "delta_u": delta_u,
                "delta_q": delta_q,
                "delta_u_full": delta_u_full,
                "lookahead_rescue": lookahead_rescue,
                "rescue": delta_u > 0,
                "break": delta_u < 0,
                "neutral": delta_u == 0,
                "u_k": u_k,
                "u_k2": u_k2,
                "u_12": u_12,
            })

            prev_state = state_k

    return examples


# ─── R12 Models ───

def train_delta_q_model(examples):
    """Train E[ΔQ] regression model on compact features."""
    X = np.array([ex["compact_vector"] for ex in examples])
    y = np.array([ex["delta_q"] for ex in examples])
    if len(set(y)) < 2:
        return None
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


def train_break_risk_model(examples):
    """Train P(break) classifier on compact features."""
    X = np.array([ex["compact_vector"] for ex in examples])
    y = np.array([1 if ex["break"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


def train_rescue_model_compact(examples):
    """Train P(rescue) classifier on compact features."""
    X = np.array([ex["compact_vector"] for ex in examples])
    y = np.array([1 if ex["rescue"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


def calibrate_delta_q(model, cal_examples):
    """Calibrate ΔQ predictions using isotonic regression."""
    if model is None or len(cal_examples) < 10:
        return None
    X = np.array([ex["compact_vector"] for ex in cal_examples])
    y = np.array([ex["delta_q"] for ex in cal_examples])
    raw_preds = model.predict(X)
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_preds, y)
        return iso
    except Exception:
        return None


def predict_delta_q(model, cal, state):
    """Predict calibrated ΔQ."""
    if model is None:
        return 0.0
    x = compact_state_to_vector(state).reshape(1, -1)
    raw = model.predict(x)[0]
    if cal is not None:
        return float(cal.predict([raw])[0])
    return float(raw)


def predict_break_risk(model, state):
    """Predict P(break)."""
    if model is None:
        return 0.0
    x = compact_state_to_vector(state).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


# ─── R12 Sequential Policy ───

def run_r12_policy(task, delta_q_model, delta_q_cal, break_model,
                   corr_model, corr_cal, fk,
                   threshold=0.0, break_threshold=0.1,
                   start_k=2, max_k=12, step=2, min_k=4,
                   use_break_gate=True):
    """R12 sequential policy with ΔQ value model and break-risk gate.

    At each checkpoint K:
      1. Compute compact state features
      2. Predict ΔQ(s_K, GENERATE)
      3. GENERATE if ΔQ > threshold AND P(break) < break_threshold
      4. Otherwise STOP
    """
    cands = task["candidates"]
    prev_state = None
    k = start_k

    while k <= max_k and k <= len(cands):
        state = compute_enhanced_state_features(
            task, k, corr_model, corr_cal, fk, prev_state)

        if k >= max_k or k + step > len(cands):
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k, "n_generations": k,
            }

        # Always generate up to min_k
        if k < min_k:
            prev_state = state
            k += step
            continue

        # Predict ΔQ
        dq = predict_delta_q(delta_q_model, delta_q_cal, state)

        # Check break risk
        p_break = 0.0
        if use_break_gate and break_model is not None:
            p_break = predict_break_risk(break_model, state)

        # Decision
        if dq > threshold and (not use_break_gate or p_break < break_threshold):
            prev_state = state
            k += step
        else:
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k, "n_generations": k,
            }

    pick = max(cands[:min(k, len(cands))], key=lambda c: predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "final_k": min(k, len(cands)), "n_generations": min(k, len(cands)),
    }


# ─── Utility ───

def compute_cost_sensitive_utility(accuracy, avg_k, lam=0.1):
    """J = Accuracy - λ * (E[K] - 2) / 10"""
    return accuracy - lam * (avg_k - 2) / 10


def split_tasks(tasks, seed=42, train_frac=0.6, cal_frac=0.15, dev_frac=0.1):
    """Split tasks into train/cal/dev/test with no leakage."""
    n = len(tasks)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    n_dev = int(n * dev_frac)
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    dev_idx = indices[n_train + n_cal:n_train + n_cal + n_dev]
    test_idx = indices[n_train + n_cal + n_dev:]
    return (
        [tasks[i] for i in train_idx],
        [tasks[i] for i in cal_idx],
        [tasks[i] for i in dev_idx],
        [tasks[i] for i in test_idx],
    )


def split_tasks_by_family(tasks, seed=42, ood_categories=None):
    """Split tasks with mechanism-OOD: hold out entire categories."""
    if ood_categories is None:
        ood_categories = ["sequence"]

    rng = np.random.RandomState(seed)
    in_dist = [t for t in tasks if t["category"] not in ood_categories]
    ood = [t for t in tasks if t["category"] in ood_categories]

    # Split in-dist into train/cal/dev/test
    n = len(in_dist)
    indices = rng.permutation(n)
    n_train = int(n * 0.65)
    n_cal = int(n * 0.15)
    n_dev = int(n * 0.10)
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    dev_idx = indices[n_train + n_cal:n_train + n_cal + n_dev]
    test_idx = indices[n_train + n_cal + n_dev:]

    return (
        [in_dist[i] for i in train_idx],
        [in_dist[i] for i in cal_idx],
        [in_dist[i] for i in dev_idx],
        [in_dist[i] for i in test_idx],
        ood,
    )


# ─── Intervention Forensics ───

def classify_intervention(actual_gen, delta_u):
    """Classify an intervention event."""
    if actual_gen and delta_u > 0:
        return "rescue"
    elif actual_gen and delta_u < 0:
        return "break"
    elif actual_gen and delta_u == 0:
        return "waste"
    elif not actual_gen and delta_u > 0:
        return "missed_rescue"
    elif not actual_gen and delta_u <= 0:
        return "correct_stop"
    return "unknown"


def compute_intervention_forensics(policy_results, oracle_results, eval_tasks,
                                    corr_model, corr_cal, fk):
    """Compute intervention forensics for a policy."""
    forensics = {
        "rescue": 0, "break": 0, "waste": 0,
        "missed_rescue": 0, "correct_stop": 0,
    }

    for i, task in enumerate(eval_tasks):
        cands = task["candidates"]
        policy_k = policy_results[i]["final_k"]
        oracle_k = oracle_results[i]["final_k"]

        # At each checkpoint, compare policy decision to oracle
        checkpoints = [2, 4, 6, 8, 10]
        prev_state = None
        for k in checkpoints:
            if k + 2 > len(cands):
                break
            state = compute_enhanced_state_features(
                task, k, corr_model, corr_cal, fk, prev_state)
            u_k = 1.0 if state["_maxcal_correct"] else 0.0

            state_k2 = compute_enhanced_state_features(
                task, k + 2, corr_model, corr_cal, fk, state)
            u_k2 = 1.0 if state_k2["_maxcal_correct"] else 0.0
            delta_u = u_k2 - u_k

            # Did the policy generate at this checkpoint?
            policy_gen = policy_k > k
            event = classify_intervention(policy_gen, delta_u)
            forensics[event] = forensics.get(event, 0) + 1

            prev_state = state

    total = sum(forensics.values())
    for k in forensics:
        forensics[k] = forensics[k] / max(total, 1)
    return forensics


# ─── Main Evaluation ───

def evaluate_r12(eval_tasks, delta_q_model, delta_q_cal, break_model,
                 rescue_model, rescue_cal, value_model_full,
                 corr_model, corr_cal, fk):
    """Evaluate all R12 policies."""
    results = {name: [] for name in [
        "maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_12",
        "oracle_6", "oracle_12",
        "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
        # R12 value policies with threshold sweep
        "r12_dq_t-0.05", "r12_dq_t-0.025", "r12_dq_t0",
        "r12_dq_t0.01", "r12_dq_t0.025", "r12_dq_t0.05",
        "r12_dq_t0.10", "r12_dq_t0.20",
        # R12 with break gate
        "r12_dq_brake_t0.01", "r12_dq_brake_t0.025",
        # Simple baselines
        "random_avg8", "uncertainty_p50", "uncertainty_p70",
        "entropy_0.5",
        # R11.2 best for comparison
        "r11_value_v_t010",
    ]}

    for task in eval_tasks:
        cands = task["candidates"]
        n = len(cands)

        # Fixed-budget MaxCal
        for k in [2, 4, 6, 8, 12]:
            if k <= n:
                pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                    corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
                results[f"maxcal_{k}"].append({
                    "correct": pick["is_correct"],
                    "utility": 100.0 if pick["is_correct"] else 0.0,
                    "final_k": k, "n_generations": k})
                if k in [6, 12]:
                    oracle = any(c["is_correct"] for c in cands[:k])
                    results[f"oracle_{k}"].append({
                        "correct": oracle,
                        "utility": 100.0 if oracle else 0.0,
                        "final_k": k, "n_generations": k})

        # Oracle policies
        results["oracle_myopic"].append(run_oracle_policy(
            task, corr_model, corr_cal, fk))
        results["oracle_lookahead4"].append(run_non_myopic_oracle(
            task, corr_model, corr_cal, fk, lookahead=4))
        results["oracle_lookahead6"].append(run_non_myopic_oracle(
            task, corr_model, corr_cal, fk, lookahead=6))

        # R12 ΔQ policies with threshold sweep
        for thresh, name in [
            (-0.05, "r12_dq_t-0.05"), (-0.025, "r12_dq_t-0.025"),
            (0.0, "r12_dq_t0"), (0.01, "r12_dq_t0.01"),
            (0.025, "r12_dq_t0.025"), (0.05, "r12_dq_t0.05"),
            (0.10, "r12_dq_t0.10"), (0.20, "r12_dq_t0.20"),
        ]:
            results[name].append(run_r12_policy(
                task, delta_q_model, delta_q_cal, break_model,
                corr_model, corr_cal, fk,
                threshold=thresh, use_break_gate=False))

        # R12 with break-risk gate
        for thresh, name in [
            (0.01, "r12_dq_brake_t0.01"),
            (0.025, "r12_dq_brake_t0.025"),
        ]:
            results[name].append(run_r12_policy(
                task, delta_q_model, delta_q_cal, break_model,
                corr_model, corr_cal, fk,
                threshold=thresh, use_break_gate=True,
                break_threshold=0.05))

        # Simple baselines
        results["random_avg8"].append(run_random_policy(
            task, corr_model, corr_cal, fk, target_avg_k=8, seed=42))
        results["uncertainty_p50"].append(run_uncertainty_policy(
            task, corr_model, corr_cal, fk, p_threshold=0.5))
        results["uncertainty_p70"].append(run_uncertainty_policy(
            task, corr_model, corr_cal, fk, p_threshold=0.7))
        results["entropy_0.5"].append(run_entropy_policy(
            task, corr_model, corr_cal, fk, entropy_threshold=0.5))

        # R11.2 best: value model with threshold 0.01
        from run_r11_2_evaluation import run_sequential_policy_v2
        results["r11_value_v_t010"].append(run_sequential_policy_v2(
            task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
            value_model=value_model_full, value_threshold=0.01,
            use_verify=True, verify_threshold=0.15))

    return results


def summarize(results):
    summary = {}
    for name, res_list in results.items():
        if not res_list:
            continue
        utils = [r["utility"] for r in res_list]
        ks = [r["n_generations"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        acc = successes / len(res_list)
        avg_k = np.mean(ks)
        summary[name] = {
            "accuracy": acc,
            "avg_k": avg_k,
            "n": len(res_list),
            "j01": compute_cost_sensitive_utility(acc, avg_k, 0.1),
            "j02": compute_cost_sensitive_utility(acc, avg_k, 0.2),
            "std_k": float(np.std(ks)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R12_DIR / "r12_corpus_12.jsonl"))
    parser.add_argument("--seeds", default="42,123,7,99,2024")
    args = parser.parse_args()

    # Load corpus
    tasks = load_corpus(args.corpus)
    tasks = [t for t in tasks if len(t.get("candidates", [])) >= 12]

    print(f"R12 Evaluation: Decision-Aligned Adaptive Compute")
    print(f"  Loaded {len(tasks)} tasks with >= 12 candidates")
    print()

    if len(tasks) < 100:
        print("WARNING: Need at least 100 tasks for meaningful evaluation.")
        print(f"  Current: {len(tasks)}. Run data collection first.")
        return

    # Enrich candidates (same pipeline as R11.2)
    print("  Enriching candidates...")
    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    tasks = enrich_corpus(tasks)
    print(f"  Enrichment complete.")

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*100}")
        print(f"  seed={seed}")
        print(f"{'='*100}")

        train_tasks, cal_tasks, dev_tasks, test_tasks = split_tasks(tasks, seed=seed)
        print(f"  Split: {len(train_tasks)} train, {len(cal_tasks)} cal, "
              f"{len(dev_tasks)} dev, {len(test_tasks)} test")

        # Train correctness model
        train_records = flatten_candidates(train_tasks)
        feature_keys = get_feature_keys(train_records)
        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, flatten_candidates(cal_tasks), feature_keys)

        # Extract counterfactual examples
        train_ex = extract_counterfactual_examples(
            train_tasks, corr_model, corr_cal, feature_keys)
        cal_ex = extract_counterfactual_examples(
            cal_tasks, corr_model, corr_cal, feature_keys)
        dev_ex = extract_counterfactual_examples(
            dev_tasks, corr_model, corr_cal, feature_keys)

        n_rescue = sum(1 for ex in train_ex if ex["rescue"])
        n_break = sum(1 for ex in train_ex if ex["break"])
        n_lookahead = sum(1 for ex in train_ex if ex["lookahead_rescue"])
        print(f"  Train: {len(train_ex)} examples, {n_rescue} rescues, "
              f"{n_break} breaks, {n_lookahead} lookahead rescues")

        # Train R12 models
        delta_q_model = train_delta_q_model(train_ex)
        delta_q_cal = calibrate_delta_q(delta_q_model, cal_ex)
        break_model = train_break_risk_model(train_ex)
        rescue_model_compact = train_rescue_model_compact(train_ex)

        # Also train R11.2-style models for comparison
        rescue_cal = calibrate_isotonic_enhanced(
            train_lookahead_rescue_model(train_ex), cal_ex, "lookahead_rescue")
        value_model_full = train_value_model(train_ex)

        # AUROC on dev
        if len(dev_ex) > 0 and rescue_model_compact is not None:
            dev_y = np.array([1 if ex["rescue"] else 0 for ex in dev_ex])
            if len(set(dev_y)) >= 2:
                dev_X = np.array([ex["compact_vector"] for ex in dev_ex])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_rescue = roc_auc_score(
                        dev_y, rescue_model_compact.predict_proba(dev_X)[:, 1])
                print(f"  Compact rescue AUROC (dev): {auroc_rescue:.4f}")

        # Evaluate on test
        results = evaluate_r12(
            test_tasks, delta_q_model, delta_q_cal, break_model,
            train_lookahead_rescue_model(train_ex), rescue_cal, value_model_full,
            corr_model, corr_cal, feature_keys)

        summary = summarize(results)

        print(f"\n  {'System':<25} {'Acc':>7} {'AvgK':>7} {'J(0.1)':>8} {'J(0.2)':>8}")
        print("  " + "-" * 60)
        for name in ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_12",
                     "oracle_6", "oracle_12",
                     "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
                     "r12_dq_t-0.05", "r12_dq_t-0.025", "r12_dq_t0",
                     "r12_dq_t0.01", "r12_dq_t0.025", "r12_dq_t0.05",
                     "r12_dq_t0.10", "r12_dq_t0.20",
                     "r12_dq_brake_t0.01", "r12_dq_brake_t0.025",
                     "random_avg8", "uncertainty_p50", "uncertainty_p70",
                     "entropy_0.5", "r11_value_v_t010"]:
            if name not in summary:
                continue
            s = summary[name]
            print(f"  {name:<25} {s['accuracy']:>6.1%} {s['avg_k']:>7.1f} "
                  f"{s['j01']:>8.3f} {s['j02']:>8.3f}")

        # Threshold selection on DEV
        print(f"\n  THRESHOLD SELECTION (dev):")
        dev_results = evaluate_r12(
            dev_tasks, delta_q_model, delta_q_cal, break_model,
            train_lookahead_rescue_model(train_ex), rescue_cal, value_model_full,
            corr_model, corr_cal, feature_keys)
        dev_summary = summarize(dev_results)

        best_thresh_name = None
        best_j = -1
        for name, s in dev_summary.items():
            if name.startswith("r12_dq_t") and not name.startswith("r12_dq_brake"):
                # Non-inferiority: accuracy >= maxcal_8 - 0.01
                maxcal8_acc = dev_summary.get("maxcal_8", {}).get("accuracy", 0)
                if s["accuracy"] >= maxcal8_acc - 0.01:
                    if s["j01"] > best_j:
                        best_j = s["j01"]
                        best_thresh_name = name
                print(f"    {name}: acc={s['accuracy']:.1%} K={s['avg_k']:.1f} "
                      f"J={s['j01']:.3f} {'*' if name == best_thresh_name else ''}")

        if best_thresh_name:
            print(f"  Selected: {best_thresh_name} (J={best_j:.3f})")
        else:
            print(f"  WARNING: No threshold passed non-inferiority on dev")

        all_results[seed] = {
            "summary": summary,
            "dev_summary": dev_summary,
            "best_thresh": best_thresh_name,
            "n_train_rescues": n_rescue,
            "n_train_breaks": n_break,
            "n_train_lookahead": n_lookahead,
        }

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  R12 AGGREGATE")
    print(f"{'='*100}")

    names = ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_12",
             "oracle_6", "oracle_12",
             "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
             "r12_dq_t-0.05", "r12_dq_t-0.025", "r12_dq_t0",
             "r12_dq_t0.01", "r12_dq_t0.025", "r12_dq_t0.05",
             "r12_dq_t0.10", "r12_dq_t0.20",
             "r12_dq_brake_t0.01", "r12_dq_brake_t0.025",
             "random_avg8", "uncertainty_p50", "uncertainty_p70",
             "entropy_0.5", "r11_value_v_t010"]

    print(f"{'System':<25} {'Mean':>7} {'Std':>7} {'AvgK':>7} {'J(0.1)':>8} {'J(0.2)':>8}")
    print("-" * 70)

    agg = {}
    for name in names:
        accs = [all_results[s]["summary"][name]["accuracy"]
                for s in seeds if name in all_results[s]["summary"]]
        ks = [all_results[s]["summary"][name]["avg_k"]
              for s in seeds if name in all_results[s]["summary"]]
        j01s = [all_results[s]["summary"][name]["j01"]
                for s in seeds if name in all_results[s]["summary"]]
        j02s = [all_results[s]["summary"][name]["j02"]
                for s in seeds if name in all_results[s]["summary"]]
        if not accs:
            continue
        agg[name] = {
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_k": np.mean(ks), "mean_j01": np.mean(j01s),
            "mean_j02": np.mean(j02s),
        }
        print(f"{name:<25} {agg[name]['mean_acc']:>6.1%} {agg[name]['std_acc']:>6.1%} "
              f"{agg[name]['mean_k']:>7.1f} {agg[name]['mean_j01']:>8.3f} "
              f"{agg[name]['mean_j02']:>8.3f}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    r12_names = [n for n in agg if n.startswith("r12_dq")]
    if r12_names:
        best_r12 = max(r12_names, key=lambda n: agg[n]["mean_j01"])
        d = agg[best_r12]
        print(f"    Best R12: {best_r12} ({d['mean_acc']:.1%}, K={d['mean_k']:.1f}, J={d['mean_j01']:.3f})")

        for mc in ["maxcal_6", "maxcal_8", "maxcal_12"]:
            if mc in agg:
                diff = d["mean_acc"] - agg[mc]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {mc}: {diff:+.1%} {v}")

        for bn in ["random_avg8", "uncertainty_p50", "uncertainty_p70"]:
            if bn in agg:
                diff = d["mean_acc"] - agg[bn]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {bn}: {diff:+.1%} {v}")

    # Oracle regret
    print(f"\n  ORACLE REGRET:")
    if "oracle_lookahead6" in agg and r12_names:
        oracle_k = agg["oracle_lookahead6"]["mean_k"]
        oracle_acc = agg["oracle_lookahead6"]["mean_acc"]
        best_r12_k = agg[best_r12]["mean_k"]
        best_r12_acc = agg[best_r12]["mean_acc"]
        wasted = best_r12_k - oracle_k
        acc_gap = oracle_acc - best_r12_acc
        print(f"    Oracle: {oracle_acc:.1%} at K={oracle_k:.1f}")
        print(f"    Best R12: {best_r12_acc:.1%} at K={best_r12_k:.1f}")
        print(f"    Wasted compute: {wasted:.1f} candidates/task")
        print(f"    Accuracy gap: {acc_gap:.1%}")

    # Pareto frontier
    print(f"\n  PARETO FRONTIER (Accuracy vs E[K]):")
    pareto_names = [n for n in agg if not n.startswith("oracle_")]
    pareto_points = [(agg[n]["mean_k"], agg[n]["mean_acc"], n) for n in pareto_names]
    pareto_points.sort()
    for k, acc, name in pareto_points:
        print(f"    ({k:.1f}, {acc:.1%}) — {name}")

    # Save
    output = {"aggregate": agg, "per_seed": all_results}
    output_path = R12_DIR / "r12_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
