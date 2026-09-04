#!/usr/bin/env python3
"""Diagnose why R11.1 learned policy underperforms despite good rescue AUROC.

Questions:
1. At what threshold does the rescue model maximize accuracy?
2. Is the problem calibration, threshold, or feature limitation?
3. What is the optimal policy if we could set the threshold perfectly?
4. How much of the oracle headroom can the rescue model theoretically capture?
5. Are the rescues predictable from uncertainty alone, or does the model add signal?
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, flatten_candidates, split_tasks, load_corpus,
)
from run_r9_evaluation import (
    add_multiround_verification_features, add_pairwise_features,
    add_answer_semantic_features, train_correctness_r9, calibrate_r9,
    predict_correctness_r9,
)
from run_r11_1_evaluation import (
    extract_acquisition_examples, state_to_vector, STATE_FEATURE_KEYS,
    compute_state_features, run_sequential_policy, run_uncertainty_policy,
    run_random_policy, predict_calibrated, calibrate_isotonic_safe,
    train_rescue_model, run_oracle_policy,
)


def diagnose_threshold_sensitivity(eval_ex, rescue_model, rescue_cal):
    """Find the optimal threshold for the rescue model."""
    # Get raw predictions
    raw_probs = []
    true_labels = []
    for ex in eval_ex:
        p = predict_calibrated(rescue_model, rescue_cal, ex["state"])
        raw_probs.append(p)
        true_labels.append(1 if ex["is_rescue"] else 0)

    raw_probs = np.array(raw_probs)
    true_labels = np.array(true_labels)

    print("  Threshold sensitivity analysis:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N_gen':>8} {'N_rescue':>10}")

    best_f1 = 0
    best_thresh = 0
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        predicted = raw_probs > thresh
        tp = ((predicted) & (true_labels == 1)).sum()
        fp = ((predicted) & (true_labels == 0)).sum()
        fn = ((~predicted) & (true_labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_gen = predicted.sum()
        print(f"  {thresh:>10.3f} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f} {n_gen:>8d} {tp:>10d}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n  Best F1 threshold: {best_thresh:.3f} (F1={best_f1:.3f})")
    return best_thresh, raw_probs, true_labels


def simulate_optimal_threshold_policy(eval_tasks, rescue_model, rescue_cal,
                                       corr_model, corr_cal, fk, threshold):
    """Simulate the sequential policy with a specific threshold."""
    results = []
    for task in eval_tasks:
        cands = task["candidates"]
        k = 2
        prev_state = None
        while k < 12 and k + 2 <= len(cands):
            state = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)
            p_r = predict_calibrated(rescue_model, rescue_cal, state)
            if p_r > threshold:
                prev_state = state
                k += 2
            else:
                break
        pick = max(cands[:k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        results.append({
            "correct": pick["is_correct"],
            "utility": 100.0 if pick["is_correct"] else 0.0,
            "final_k": k,
        })
    acc = np.mean([r["correct"] for r in results])
    avg_k = np.mean([r["final_k"] for r in results])
    return acc, avg_k


def compare_feature_importances(rescue_model, eval_ex):
    """Check which features drive the rescue model."""
    importances = rescue_model.feature_importances_
    pairs = list(zip(STATE_FEATURE_KEYS, importances))
    pairs.sort(key=lambda x: -x[1])
    print("\n  Feature importances (rescue model):")
    for name, imp in pairs[:10]:
        print(f"    {name:<30} {imp:.4f}")
    return pairs


def uncertainty_vs_learned_signal(eval_ex, rescue_model, rescue_cal):
    """Check if the rescue model adds signal beyond uncertainty alone."""
    # Get rescue model predictions
    rescue_probs = []
    p_top1_values = []
    true_labels = []
    for ex in eval_ex:
        p = predict_calibrated(rescue_model, rescue_cal, ex["state"])
        rescue_probs.append(p)
        p_top1_values.append(ex["state"]["p_top1"])
        true_labels.append(1 if ex["is_rescue"] else 0)

    rescue_probs = np.array(rescue_probs)
    p_top1_values = np.array(p_top1_values)
    true_labels = np.array(true_labels)

    # AUROC of rescue model
    if len(set(true_labels)) >= 2:
        auroc_rescue = roc_auc_score(true_labels, rescue_probs)
        auroc_uncertainty = roc_auc_score(true_labels, -p_top1_values)  # low p_top1 → high rescue prob
        print(f"\n  AUROC comparison:")
        print(f"    Rescue model:      {auroc_rescue:.4f}")
        print(f"    Uncertainty (1-p): {auroc_uncertainty:.4f}")

        # Conditional analysis: among uncertain states (p_top1 < 0.5), can the model distinguish?
        uncertain_mask = p_top1_values < 0.5
        if uncertain_mask.sum() > 10 and true_labels[uncertain_mask].sum() > 0:
            auroc_conditional = roc_auc_score(
                true_labels[uncertain_mask],
                rescue_probs[uncertain_mask])
            print(f"    Rescue | uncertain: {auroc_conditional:.4f} (n={uncertain_mask.sum()}, pos={true_labels[uncertain_mask].sum()})")

        # Average precision
        ap_rescue = average_precision_score(true_labels, rescue_probs)
        ap_uncertainty = average_precision_score(true_labels, -p_top1_values)
        print(f"\n  Average precision:")
        print(f"    Rescue model:      {ap_rescue:.4f}")
        print(f"    Uncertainty (1-p): {ap_uncertainty:.4f}")


def analyze_oracle_headroom_decomposition(eval_tasks, corr_model, corr_cal, fk):
    """Decompose the oracle headroom: how much is theoretically capturable?"""
    # For each task, find the minimum K at which MaxCal gets the right answer
    # and the minimum K at which Oracle gets the right answer
    maxcal_first_correct = []
    oracle_first_correct = []
    never_correct = 0

    for task in eval_tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        # Find first K where MaxCal is correct
        mc_correct_k = None
        for k in [2, 4, 6, 8, 10, 12]:
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            if pick["is_correct"]:
                mc_correct_k = k
                break

        # Find first K where Oracle is correct
        oracle_correct_k = None
        for k in [2, 4, 6, 8, 10, 12]:
            if any(c["is_correct"] for c in cands[:k]):
                oracle_correct_k = k
                break

        if oracle_correct_k is None:
            never_correct += 1
        else:
            oracle_first_correct.append(oracle_correct_k)

        if mc_correct_k is None and oracle_correct_k is not None:
            maxcal_first_correct.append(None)  # MaxCal never gets it
        elif mc_correct_k is not None:
            maxcal_first_correct.append(mc_correct_k)

    print(f"\n  Oracle headroom decomposition:")
    print(f"    Never correct (even at K=12): {never_correct}/{len(eval_tasks)}")

    # How many tasks does MaxCal get right at each K?
    print(f"\n    MaxCal first-correct K distribution:")
    for k in [2, 4, 6, 8, 10, 12, None]:
        if k is None:
            n = sum(1 for x in maxcal_first_correct if x is None)
            print(f"      Never: {n}")
        else:
            n = sum(1 for x in maxcal_first_correct if x == k)
            print(f"      K={k}: {n}")

    # How many tasks does Oracle get right at each K?
    print(f"\n    Oracle first-correct K distribution:")
    for k in [2, 4, 6, 8, 10, 12]:
        n = sum(1 for x in oracle_first_correct if x == k)
        print(f"      K={k}: {n}")

    # The gap: tasks where Oracle gets it right but MaxCal doesn't at the same K
    rescue_at_k = {}
    for k in [2, 4, 6, 8, 10, 12]:
        mc_right = sum(1 for task in eval_tasks if len(task["candidates"]) >= k
                       for _ in [1]
                       if max(task["candidates"][:k],
                              key=lambda c: predict_correctness_r9(
                                  corr_model, corr_cal,
                                  c.get("enriched_features", {}), c, fk))["is_correct"])
        oracle_right = sum(1 for task in eval_tasks if len(task["candidates"]) >= k
                           for _ in [1]
                           if any(c["is_correct"] for c in task["candidates"][:k]))
        rescue_at_k[k] = oracle_right - mc_right

    print(f"\n    Rescue opportunities at each K (Oracle - MaxCal):")
    for k in [2, 4, 6, 8, 10, 12]:
        print(f"      K={k}: {rescue_at_k[k]} tasks ({rescue_at_k[k]/len(eval_tasks):.1%})")


def main():
    corpus_path = REPO_ROOT / "experiments/daph_x/r11/r11_corpus_12.jsonl"
    tasks = load_corpus(str(corpus_path))
    print(f"Loaded {len(tasks)} tasks")

    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    fk = get_feature_keys(all_records)

    for seed in [42, 99]:
        print(f"\n{'='*80}")
        print(f"  DIAGNOSIS: seed={seed}")
        print(f"{'='*80}")

        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        corr_model = train_correctness_r9(train_records, fk)
        corr_cal = calibrate_r9(corr_model, cal_records, fk)

        train_ex = extract_acquisition_examples(train_tasks, corr_model, corr_cal, fk)
        cal_ex = extract_acquisition_examples(cal_tasks, corr_model, corr_cal, fk)
        eval_ex = extract_acquisition_examples(eval_tasks, corr_model, corr_cal, fk)

        n_rescue = sum(1 for ex in train_ex if ex["is_rescue"])
        n_eval_rescue = sum(1 for ex in eval_ex if ex["is_rescue"])
        print(f"  Train: {n_rescue} rescues / {len(train_ex)} examples ({n_rescue/len(train_ex):.1%})")
        print(f"  Eval:  {n_eval_rescue} rescues / {len(eval_ex)} examples ({n_eval_rescue/len(eval_ex):.1%})")

        # Train rescue model
        rescue_model = train_rescue_model(train_ex)
        rescue_cal = calibrate_isotonic_safe(rescue_model, cal_ex, "is_rescue")

        # AUROC
        eval_y = np.array([1 if ex["is_rescue"] else 0 for ex in eval_ex])
        eval_X = np.array([state_to_vector(ex["state"]) for ex in eval_ex])
        if len(set(eval_y)) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = roc_auc_score(eval_y, rescue_model.predict_proba(eval_X)[:, 1])
            print(f"  Rescue AUROC: {auroc:.4f}")

        # 1. Threshold sensitivity
        print(f"\n  --- Threshold sensitivity ---")
        best_thresh, raw_probs, true_labels = diagnose_threshold_sensitivity(
            eval_ex, rescue_model, rescue_cal)

        # 2. Simulate policy at different thresholds
        print(f"\n  --- Policy simulation at different thresholds ---")
        print(f"  {'Threshold':>10} {'Accuracy':>10} {'AvgK':>8} {'J(0.1)':>8}")
        for thresh in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            acc, avg_k = simulate_optimal_threshold_policy(
                eval_tasks, rescue_model, rescue_cal, corr_model, corr_cal, fk, thresh)
            j = acc - 0.1 * (avg_k - 2) / 10
            print(f"  {thresh:>10.3f} {acc:>10.1%} {avg_k:>8.1f} {j:>8.3f}")

        # 3. Feature importances
        compare_feature_importances(rescue_model, eval_ex)

        # 4. Uncertainty vs learned signal
        uncertainty_vs_learned_signal(eval_ex, rescue_model, rescue_cal)

        # 5. Oracle headroom decomposition
        analyze_oracle_headroom_decomposition(eval_tasks, corr_model, corr_cal, fk)

        # 6. Upper bound: what if we had a perfect rescue classifier?
        print(f"\n  --- Oracle policy upper bound ---")
        oracle_results = []
        for task in eval_tasks:
            res = run_oracle_policy(task, corr_model, corr_cal, fk)
            oracle_results.append(res)
        oracle_acc = np.mean([r["correct"] for r in oracle_results])
        oracle_k = np.mean([r["final_k"] for r in oracle_results])
        print(f"  Oracle adaptive: {oracle_acc:.1%} at K={oracle_k:.1f}")

        # 7. What if we just always generate to K=8?
        print(f"\n  --- Fixed budget comparison ---")
        for k in [2, 4, 6, 8, 10, 12]:
            results = []
            for task in eval_tasks:
                cands = task["candidates"]
                if len(cands) < k:
                    continue
                pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                    corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
                results.append(pick["is_correct"])
            if results:
                print(f"  MaxCal@{k}: {np.mean(results):.1%}")

    print(f"\n{'='*80}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    print("""
    Key questions answered:
    1. Is the rescue model's AUROC real? → Yes (0.74-0.90)
    2. Does it add signal beyond uncertainty? → Check AUROC comparison
    3. Is the threshold optimal? → Check threshold sweep
    4. How much oracle headroom is capturable? → Check decomposition
    5. What features matter most? → Check importances
    """)


if __name__ == "__main__":
    main()
