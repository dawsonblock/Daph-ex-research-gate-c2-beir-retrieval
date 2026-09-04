#!/usr/bin/env python3
"""DAPH-X R11.2: Non-myopic adaptive compute with expanded action space.

Fixes based on R11.1 diagnosis:

Fix 1: Non-myopic training target
  Instead of P(rescue at K+2), train on:
    Y_lookahead(s_K) = 1[MaxCal@12 correct AND MaxCal@K wrong]
  This captures the FULL downstream benefit of generating more candidates,
  not just the next-step benefit.

Fix 2: Generator diversity and convergence features
  New features:
    - answer_diversity: normalized unique answer count
    - convergence_rate: how quickly MaxCal pick is stabilizing
    - pick_change_rate: fraction of checkpoints where MaxCal pick changed
    - generator_temperature_variance: diversity of generation temperatures
    - answer_agreement_rate: fraction of candidates agreeing with MaxCal pick
    - cluster_concentration: Herfindahl index of answer clusters
    - margin_trend: is margin increasing (converging) or decreasing?

Fix 3: Non-myopic oracle
  Instead of checking only K+2, check if MaxCal@K+4 or MaxCal@K+6 improves.
  This captures multi-step benefits.

Fix 4: VERIFY action
  Use existing verification scores to re-rank candidates.
  If verification changes the pick, that's a "verification rescue."

Usage:
    python scripts/run_r11_2_evaluation.py \\
        --corpus experiments/daph_x/r11/r11_corpus_12.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

R11_DIR = REPO_ROOT / "experiments/daph_x/r11"

from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, build_feature_vector,
    flatten_candidates, split_tasks, load_corpus,
)
from run_r9_evaluation import (
    add_multiround_verification_features, add_pairwise_features,
    add_answer_semantic_features, train_correctness_r9, calibrate_r9,
    predict_correctness_r9,
)
from run_r11_1_evaluation import (
    compute_answer_entropy, predict_calibrated, calibrate_isotonic_safe,
    run_random_policy, run_uncertainty_policy, run_entropy_policy,
    compute_cost_sensitive_utility, run_oracle_policy,
)


# ─── Fix 2: Enhanced state features ───

def compute_enhanced_state_features(task, k, corr_model, corr_cal, fk, prev_state=None):
    """Compute state features with generator diversity and convergence."""
    cands_k = task["candidates"][:k]

    # Calibrated P(correct) for all candidates
    p_values = []
    for c in cands_k:
        p = predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk)
        p_values.append(p)

    p_values_sorted = sorted(p_values, reverse=True)
    p_top1 = p_values_sorted[0]
    p_top2 = p_values_sorted[1] if len(p_values_sorted) > 1 else 0.0
    margin = p_top1 - p_top2

    # MaxCal pick
    maxcal_idx = np.argmax(p_values)
    maxcal_pick = cands_k[maxcal_idx]

    # Answer entropy
    entropy = compute_answer_entropy(cands_k)

    # Majority fraction
    answers = [c["answer"] for c in cands_k]
    counts = Counter(answers)
    majority_frac = max(counts.values()) / k
    n_unique = len(counts)

    # Confidence statistics
    confidences = [c["self_confidence"] / 100.0 for c in cands_k]
    conf_var = float(np.var(confidences))

    # Verification statistics
    verifications = [c.get("verification_score", 0.5) for c in cands_k]
    ver_var = float(np.var(verifications))
    v_maxcal = maxcal_pick.get("verification_score", 0.5)
    v_consistency = maxcal_pick.get("verification_consistent", 0)

    c_maxcal = maxcal_pick["self_confidence"] / 100.0
    cv_disagreement = abs(c_maxcal - v_maxcal)

    # Selection stability
    stability = 0.0
    if prev_state is not None:
        stability = 1.0 if prev_state.get("_maxcal_answer", "") == maxcal_pick["answer"] else 0.0

    # Dynamic features
    delta_p_top1 = 0.0
    delta_entropy = 0.0
    if prev_state is not None:
        delta_p_top1 = p_top1 - prev_state.get("p_top1", p_top1)
        delta_entropy = entropy - prev_state.get("answer_entropy", entropy)

    # ─── NEW: Generator diversity features ───
    # Answer diversity (normalized)
    answer_diversity = n_unique / k

    # Cluster concentration (Herfindahl index)
    cluster_concentration = sum((count / k) ** 2 for count in counts.values())

    # Agreement rate: fraction of candidates agreeing with MaxCal pick
    agreement_rate = counts.get(maxcal_pick["answer"], 0) / k

    # Temperature variance (generator diversity)
    temps = [c.get("temperature", 0.7) for c in cands_k]
    temp_variance = float(np.var(temps)) if len(temps) > 1 else 0.0

    # ─── NEW: Convergence features ───
    # Pick change rate: how many times has the MaxCal pick changed?
    pick_changes = 0
    total_checkpoints = 0
    if prev_state is not None:
        total_checkpoints = prev_state.get("_total_checkpoints", 0) + 1
        pick_changes = prev_state.get("_pick_changes", 0)
        if prev_state.get("_maxcal_answer", "") != maxcal_pick["answer"]:
            pick_changes += 1
    else:
        total_checkpoints = 1

    pick_change_rate = pick_changes / max(total_checkpoints, 1)

    # Margin trend: is margin increasing (converging) or decreasing?
    margin_trend = 0.0
    if prev_state is not None:
        prev_margin = prev_state.get("margin", margin)
        margin_trend = margin - prev_margin

    # Top cluster size trend
    top_cluster_trend = 0.0
    if prev_state is not None:
        top_cluster_trend = max(counts.values()) - prev_state.get("_top_cluster_size", max(counts.values()))

    features = {
        # Original features
        "k": float(k),
        "p_top1": p_top1,
        "p_top2": p_top2,
        "margin": margin,
        "answer_entropy": entropy,
        "majority_fraction": majority_frac,
        "n_unique_answers": float(n_unique),
        "semantic_cluster_count": float(n_unique),
        "top_cluster_size": float(max(counts.values())),
        "confidence_variance": conf_var,
        "verification_variance": ver_var,
        "v_maxcal": v_maxcal,
        "v_consistency": float(v_consistency),
        "c_maxcal": c_maxcal,
        "cv_disagreement": cv_disagreement,
        "selection_stability": stability,
        "delta_p_top1": delta_p_top1,
        "delta_entropy": delta_entropy,
        # NEW: Generator diversity
        "answer_diversity": answer_diversity,
        "cluster_concentration": cluster_concentration,
        "agreement_rate": agreement_rate,
        "temp_variance": temp_variance,
        # NEW: Convergence
        "pick_change_rate": pick_change_rate,
        "margin_trend": margin_trend,
        "top_cluster_trend": top_cluster_trend,
        "total_checkpoints": float(total_checkpoints),
    }

    # Store for dynamic features at next checkpoint
    features["_maxcal_answer"] = maxcal_pick["answer"]
    features["_maxcal_correct"] = maxcal_pick["is_correct"]
    features["_maxcal_p"] = p_top1
    features["_top_cluster_size"] = max(counts.values())
    features["_total_checkpoints"] = total_checkpoints
    features["_pick_changes"] = pick_changes

    return features


ENHANCED_FEATURE_KEYS = [
    # Original
    "k", "p_top1", "p_top2", "margin", "answer_entropy",
    "majority_fraction", "n_unique_answers", "semantic_cluster_count",
    "top_cluster_size", "confidence_variance", "verification_variance",
    "v_maxcal", "v_consistency", "c_maxcal", "cv_disagreement",
    "selection_stability", "delta_p_top1", "delta_entropy",
    # NEW
    "answer_diversity", "cluster_concentration", "agreement_rate",
    "temp_variance", "pick_change_rate", "margin_trend",
    "top_cluster_trend", "total_checkpoints",
]


def enhanced_state_to_vector(feats):
    return np.array([feats.get(k, 0.0) for k in ENHANCED_FEATURE_KEYS])


# ─── Fix 1: Non-myopic training target ───

def extract_non_myopic_examples(tasks, corr_model, corr_cal, fk):
    """Extract non-myopic acquisition examples.

    Training target: Y_lookahead(s_K) = 1[MaxCal@12 correct AND MaxCal@K wrong]

    This captures the FULL downstream benefit of generating more candidates,
    not just the next-step benefit.
    """
    examples = []
    checkpoints = [2, 4, 6, 8, 10]

    for task in tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        # Compute MaxCal@12 correctness
        pick_12 = max(cands[:12], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        u_12 = 1.0 if pick_12["is_correct"] else 0.0

        prev_state = None
        for k in checkpoints:
            if k > len(cands):
                break

            state_k = compute_enhanced_state_features(task, k, corr_model, corr_cal, fk, prev_state)
            u_k = 1.0 if state_k["_maxcal_correct"] else 0.0

            # Non-myopic target: will MaxCal@12 be correct when MaxCal@K is wrong?
            lookahead_rescue = (u_12 > 0.5 and u_k < 0.5)

            # Also compute next-step rescue for comparison
            if k + 2 <= len(cands):
                state_k2 = compute_enhanced_state_features(task, k + 2, corr_model, corr_cal, fk, state_k)
                u_k2 = 1.0 if state_k2["_maxcal_correct"] else 0.0
                next_step_rescue = (u_k2 > 0.5 and u_k < 0.5)
                next_step_break = (u_k2 < 0.5 and u_k > 0.5)
            else:
                next_step_rescue = False
                next_step_break = False

            # Value: how much improvement from K to 12
            delta_u_full = u_12 - u_k  # +1, 0, or -1

            examples.append({
                "task_id": task["task_id"],
                "checkpoint_k": k,
                "state": {k: v for k, v in state_k.items() if not k.startswith("_")},
                "lookahead_rescue": lookahead_rescue,
                "next_step_rescue": next_step_rescue,
                "next_step_break": next_step_break,
                "delta_u_full": delta_u_full,
                "u_k": u_k,
                "u_12": u_12,
            })

            prev_state = state_k

    return examples


# ─── Fix 3: Non-myopic oracle ───

def run_non_myopic_oracle(task, corr_model, corr_cal, fk,
                          start_k=2, max_k=12, step=2, lookahead=4):
    """Oracle that looks ahead multiple steps.

    At checkpoint K, checks if MaxCal@K+lookahead > MaxCal@K.
    If yes, generate. If no, stop.
    """
    cands = task["candidates"]
    k = start_k

    while k < max_k and k + step <= len(cands):
        # Current MaxCal@K
        pick_k = max(cands[:k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        u_k = 1.0 if pick_k["is_correct"] else 0.0

        # Look ahead: check MaxCal@K+lookahead (or max_k, whichever is smaller)
        look_k = min(k + lookahead, max_k, len(cands))
        pick_look = max(cands[:look_k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        u_look = 1.0 if pick_look["is_correct"] else 0.0

        if u_look > u_k:
            k += step
        else:
            break

    pick = max(cands[:k], key=lambda c: predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "final_k": k,
        "n_generations": k,
    }


# ─── Fix 4: VERIFY action ───

def run_verify_action(task, k, corr_model, corr_cal, fk, verify_weight=0.4):
    """VERIFY: use verification scores to re-rank candidates.

    Instead of MaxCal (calibrated P(correct)), use verification-weighted score.
    Returns the verified pick and whether it differs from MaxCal.
    """
    cands_k = task["candidates"][:k]

    # MaxCal pick
    p_values = [predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk) for c in cands_k]
    maxcal_idx = np.argmax(p_values)
    maxcal_pick = cands_k[maxcal_idx]

    # Verification-weighted pick: combine P(correct) with verification score
    verify_scores = []
    for i, c in enumerate(cands_k):
        v = c.get("verification_score", 0.5)
        p = p_values[i]
        combined = (1 - verify_weight) * p + verify_weight * v
        verify_scores.append(combined)

    verify_idx = np.argmax(verify_scores)
    verify_pick = cands_k[verify_idx]

    return {
        "maxcal_pick": maxcal_pick,
        "verify_pick": verify_pick,
        "changed": maxcal_pick["answer"] != verify_pick["answer"],
        "correct": verify_pick["is_correct"],
        "maxcal_correct": maxcal_pick["is_correct"],
    }


def should_verify(state, cv_disagreement_threshold=0.15):
    """Decide whether VERIFY is worthwhile.

    Only verify when confidence and verification disagree — that's when
    verification might change the pick. If they agree, verify is wasted compute.
    """
    return state.get("cv_disagreement", 0.0) > cv_disagreement_threshold


def run_sequential_policy_v2(task, rescue_model, rescue_cal,
                              corr_model, corr_cal, fk,
                              threshold=0.01, start_k=2, max_k=12, step=2,
                              min_k=4, use_verify=True,
                              verify_threshold=0.15, value_model=None,
                              value_threshold=0.02, ptop1_model_override=None):
    """Sequential policy with non-myopic rescue model and selective VERIFY.

    At each checkpoint K:
      1. Compute enhanced state features
      2. Predict P(lookahead rescue) or use value model
      3. If prediction > threshold: GENERATE(+2)
      4. Else: STOP, selectively VERIFY if confidence-verification disagree
    """
    cands = task["candidates"]
    prev_state = None
    k = start_k

    while k <= max_k and k <= len(cands):
        state = compute_enhanced_state_features(task, k, corr_model, corr_cal, fk, prev_state)

        if k >= max_k or k + step > len(cands):
            # At max: selectively verify
            if use_verify and should_verify(state, verify_threshold):
                v = run_verify_action(task, k, corr_model, corr_cal, fk)
                return {
                    "correct": v["correct"],
                    "utility": 100.0 if v["correct"] else 0.0,
                    "final_k": k,
                    "n_generations": k,
                    "used_verify": True,
                    "verify_changed": v["changed"],
                }
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k,
                "n_generations": k,
                "used_verify": False,
            }

        # Always generate up to min_k
        if k < min_k:
            prev_state = state
            k += step
            continue

        # Predict: use value model if available, else rescue model, else ptop1 ablation
        if value_model is not None:
            x = enhanced_state_to_vector(state).reshape(1, -1)
            score = value_model.predict(x)[0]
            should_gen = score > value_threshold
        elif ptop1_model_override is not None:
            p_r = predict_ptop1_only(ptop1_model_override, state)
            should_gen = p_r > threshold
        else:
            p_r = 0.0
            if rescue_model is not None:
                p_r = predict_calibrated_enhanced(rescue_model, rescue_cal, state)
            should_gen = p_r > threshold

        if should_gen:
            prev_state = state
            k += step
        else:
            # STOP: selectively verify
            if use_verify and should_verify(state, verify_threshold):
                v = run_verify_action(task, k, corr_model, corr_cal, fk)
                return {
                    "correct": v["correct"],
                    "utility": 100.0 if v["correct"] else 0.0,
                    "final_k": k,
                    "n_generations": k,
                    "used_verify": True,
                    "verify_changed": v["changed"],
                }
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k,
                "n_generations": k,
                "used_verify": False,
            }

    pick = max(cands[:min(k, len(cands))], key=lambda c: predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "final_k": min(k, len(cands)),
        "n_generations": min(k, len(cands)),
        "used_verify": False,
    }


# ─── Training ───

def calibrate_isotonic_enhanced(model, examples, label_key):
    """Calibrate using enhanced state vector."""
    raw_p, true_l = [], []
    for ex in examples:
        x = enhanced_state_to_vector(ex["state"]).reshape(1, -1)
        raw_p.append(model.predict_proba(x)[0, 1])
        true_l.append(1 if ex[label_key] else 0)
    n_pos = sum(true_l)
    if n_pos < 3:
        return None
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def predict_calibrated_enhanced(model, cal, state):
    """Predict using enhanced state vector."""
    x = enhanced_state_to_vector(state).reshape(1, -1)
    raw = model.predict_proba(x)[0, 1]
    if cal is None:
        return float(raw)
    return float(cal.predict([raw])[0])


def train_lookahead_rescue_model(examples):
    """Train P(lookahead rescue) = P(MaxCal@12 correct AND MaxCal@K wrong)."""
    X = np.array([enhanced_state_to_vector(ex["state"]) for ex in examples])
    y = np.array([1 if ex["lookahead_rescue"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


def train_value_model(examples):
    """Train E[delta_u_full] = E[U(MaxCal@12) - U(MaxCal@K)]."""
    X = np.array([enhanced_state_to_vector(ex["state"]) for ex in examples])
    y = np.array([ex["delta_u_full"] for ex in examples])
    if len(set(y)) < 2:
        return None
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


def train_ptop1_only_model(examples):
    """Ablation: rescue model using only p_top1 (uncertainty alone)."""
    X = np.array([[ex["state"]["p_top1"]] for ex in examples])
    y = np.array([1 if ex["lookahead_rescue"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


def predict_ptop1_only(model, state):
    """Predict with p_top1-only model."""
    x = np.array([[state["p_top1"]]])
    return float(model.predict_proba(x)[0, 1])


def paired_bootstrap_ci(daphx_utils, baseline_utils, n_bootstrap=2000, ci=95):
    """Paired bootstrap confidence interval for utility difference.

    Returns (mean_diff, ci_low, ci_high).
    """
    from sklearn.utils import resample
    diffs = np.array(daphx_utils) - np.array(baseline_utils)
    n = len(diffs)
    boot_means = []
    for _ in range(n_bootstrap):
        boot_idx = resample(range(n), n_samples=n)
        boot_means.append(np.mean(diffs[boot_idx]))
    alpha = (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_means, [alpha, 100 - alpha])
    return float(np.mean(diffs)), float(ci_low), float(ci_high)


# ─── Evaluation ───

def evaluate_all_v2(eval_tasks, rescue_model, rescue_cal, value_model,
                     ptop1_model, corr_model, corr_cal, fk):
    results = {name: [] for name in [
        "maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
        "oracle_6", "oracle_12",
        "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
        "daphx_t001", "daphx_t005", "daphx_t010", "daphx_t020",
        "daphx_t001_verify", "daphx_t005_verify", "daphx_t010_verify",
        "daphx_value", "daphx_value_v",
        "daphx_value_v_t005", "daphx_value_v_t010", "daphx_value_v_t020",
        "daphx_t001_mink6", "daphx_t005_mink6",
        "daphx_ptop1_t001", "daphx_ptop1_t005", "daphx_ptop1_t010",
        "random_avg8", "random_avg10",
        "uncertainty_p50", "uncertainty_p70",
        "entropy_1.0", "entropy_0.5",
        "verify_only_6", "verify_only_8", "verify_only_12",
        "verify_selective_6", "verify_selective_8",
    ]}

    for task in eval_tasks:
        cands = task["candidates"]
        n = len(cands)

        # Fixed-budget MaxCal
        for k in [2, 4, 6, 8, 10, 12]:
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

        # DAPH-X with different thresholds (no verify)
        for thresh, name in [(0.01, "daphx_t001"), (0.05, "daphx_t005"),
                              (0.10, "daphx_t010"), (0.20, "daphx_t020")]:
            results[name].append(run_sequential_policy_v2(
                task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
                threshold=thresh, use_verify=False))

        # DAPH-X with selective VERIFY
        for thresh, name in [(0.01, "daphx_t001_verify"), (0.05, "daphx_t005_verify"),
                              (0.10, "daphx_t010_verify")]:
            results[name].append(run_sequential_policy_v2(
                task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
                threshold=thresh, use_verify=True, verify_threshold=0.15))

        # DAPH-X with min_k=6
        for thresh, name in [(0.01, "daphx_t001_mink6"), (0.05, "daphx_t005_mink6")]:
            results[name].append(run_sequential_policy_v2(
                task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
                threshold=thresh, min_k=6, use_verify=False))

        # DAPH-X value-based (no verify)
        results["daphx_value"].append(run_sequential_policy_v2(
            task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
            value_model=value_model, value_threshold=0.02, use_verify=False))

        # DAPH-X value-based + selective verify
        results["daphx_value_v"].append(run_sequential_policy_v2(
            task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
            value_model=value_model, value_threshold=0.02, use_verify=True))

        # Value model with different thresholds + verify
        for vt, name in [(0.005, "daphx_value_v_t005"), (0.01, "daphx_value_v_t010"),
                         (0.02, "daphx_value_v_t020")]:
            results[name].append(run_sequential_policy_v2(
                task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
                value_model=value_model, value_threshold=vt, use_verify=True))

        # Ablation: p_top1-only model
        if ptop1_model is not None:
            for thresh, name in [(0.01, "daphx_ptop1_t001"), (0.05, "daphx_ptop1_t005"),
                                 (0.10, "daphx_ptop1_t010")]:
                results[name].append(run_sequential_policy_v2(
                    task, rescue_model, rescue_cal, corr_model, corr_cal, fk,
                    threshold=thresh, use_verify=False,
                    ptop1_model_override=ptop1_model))

        # Random and heuristic baselines
        results["random_avg8"].append(run_random_policy(
            task, corr_model, corr_cal, fk, target_avg_k=8, seed=42))
        results["random_avg10"].append(run_random_policy(
            task, corr_model, corr_cal, fk, target_avg_k=10, seed=42))
        results["uncertainty_p50"].append(run_uncertainty_policy(
            task, corr_model, corr_cal, fk, p_threshold=0.5))
        results["uncertainty_p70"].append(run_uncertainty_policy(
            task, corr_model, corr_cal, fk, p_threshold=0.7))
        results["entropy_1.0"].append(run_entropy_policy(
            task, corr_model, corr_cal, fk, entropy_threshold=1.0))
        results["entropy_0.5"].append(run_entropy_policy(
            task, corr_model, corr_cal, fk, entropy_threshold=0.5))

        # Verify-only baselines (always verify)
        for k in [6, 8, 12]:
            if k <= n:
                v = run_verify_action(task, k, corr_model, corr_cal, fk)
                results[f"verify_only_{k}"].append({
                    "correct": v["correct"],
                    "utility": 100.0 if v["correct"] else 0.0,
                    "final_k": k, "n_generations": k})

        # Selective verify: only when cv_disagreement > 0.15
        for k in [6, 8]:
            if k <= n:
                state = compute_enhanced_state_features(task, k, corr_model, corr_cal, fk)
                if should_verify(state, 0.15):
                    v = run_verify_action(task, k, corr_model, corr_cal, fk)
                    results[f"verify_selective_{k}"].append({
                        "correct": v["correct"],
                        "utility": 100.0 if v["correct"] else 0.0,
                        "final_k": k, "n_generations": k})
                else:
                    pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
                    results[f"verify_selective_{k}"].append({
                        "correct": pick["is_correct"],
                        "utility": 100.0 if pick["is_correct"] else 0.0,
                        "final_k": k, "n_generations": k})

    return results


def summarize(results):
    summary = {}
    for name, res_list in results.items():
        if not res_list:
            continue
        utils = [r["utility"] for r in res_list]
        ks = [r["n_generations"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        n_verify = sum(1 for r in res_list if r.get("used_verify", False))
        n_changed = sum(1 for r in res_list if r.get("verify_changed", False))
        summary[name] = {
            "mean_utility": float(np.mean(utils)),
            "success_rate": successes / len(res_list),
            "avg_k": float(np.mean(ks)),
            "n": len(res_list),
            "n_verify_used": n_verify,
            "n_verify_changed": n_changed,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R11_DIR / "r11_corpus_12.jsonl"))
    args = parser.parse_args()

    R11_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    n_with_12 = sum(1 for t in tasks if len(t["candidates"]) >= 12)
    print(f"Loaded {len(tasks)} tasks, {n_with_12} with 12 candidates")

    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} train, {len(eval_tasks)} eval) ===")

        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, cal_records, feature_keys)

        # Extract non-myopic examples
        train_ex = extract_non_myopic_examples(train_tasks, corr_model, corr_cal, feature_keys)
        cal_ex = extract_non_myopic_examples(cal_tasks, corr_model, corr_cal, feature_keys)
        eval_ex = extract_non_myopic_examples(eval_tasks, corr_model, corr_cal, feature_keys)

        n_lookahead = sum(1 for ex in train_ex if ex["lookahead_rescue"])
        n_next = sum(1 for ex in train_ex if ex["next_step_rescue"])
        n_eval_lookahead = sum(1 for ex in eval_ex if ex["lookahead_rescue"])
        print(f"  Train: {n_lookahead} lookahead rescues ({n_lookahead/len(train_ex):.1%}), "
              f"{n_next} next-step rescues ({n_next/len(train_ex):.1%})")
        print(f"  Eval:  {n_eval_lookahead} lookahead rescues ({n_eval_lookahead/len(eval_ex):.1%})")

        # Train lookahead rescue model
        rescue_model = train_lookahead_rescue_model(train_ex)
        rescue_cal = calibrate_isotonic_enhanced(rescue_model, cal_ex, "lookahead_rescue")
        if rescue_cal is None:
            print(f"  Rescue: skipping calibration (too few positives in cal set)")

        # Train value model
        value_model = train_value_model(train_ex)

        # Train ptop1-only ablation model
        ptop1_model = train_ptop1_only_model(train_ex)

        # Ablation AUROC
        if ptop1_model and n_eval_lookahead > 0:
            eval_y = np.array([1 if ex["lookahead_rescue"] else 0 for ex in eval_ex])
            if len(set(eval_y)) >= 2:
                ptop1_probs = np.array([predict_ptop1_only(ptop1_model, ex["state"]) for ex in eval_ex])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_ptop1 = roc_auc_score(eval_y, ptop1_probs)
                print(f"  p_top1-only AUROC:      {auroc_ptop1:.4f}")

        # AUROC
        if rescue_model and n_eval_lookahead > 0:
            eval_y = np.array([1 if ex["lookahead_rescue"] else 0 for ex in eval_ex])
            if len(set(eval_y)) >= 2:
                eval_X = np.array([enhanced_state_to_vector(ex["state"]) for ex in eval_ex])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc = roc_auc_score(eval_y, rescue_model.predict_proba(eval_X)[:, 1])
                print(f"  Lookahead rescue AUROC: {auroc:.4f}")

                # Compare with uncertainty alone
                p_top1_values = np.array([ex["state"]["p_top1"] for ex in eval_ex])
                if len(set(eval_y)) >= 2:
                    auroc_unc = roc_auc_score(eval_y, -p_top1_values)
                    print(f"  Uncertainty AUROC:       {auroc_unc:.4f}")

        # Feature importances
        if rescue_model:
            importances = rescue_model.feature_importances_
            pairs = list(zip(ENHANCED_FEATURE_KEYS, importances))
            pairs.sort(key=lambda x: -x[1])
            print(f"  Top 5 features: {', '.join(f'{n}={v:.3f}' for n, v in pairs[:5])}")

        # Evaluate
        results = evaluate_all_v2(eval_tasks, rescue_model, rescue_cal, value_model,
                                   ptop1_model, corr_model, corr_cal, feature_keys)
        summary = summarize(results)

        print(f"\n  {'System':<25} {'Acc':>7} {'AvgK':>7} {'J(0.1)':>8} {'Verify':>7}")
        print("  " + "-" * 60)
        for name in ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
                     "oracle_6", "oracle_12",
                     "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
                     "daphx_t001", "daphx_t005", "daphx_t010", "daphx_t020",
                     "daphx_t001_verify", "daphx_t005_verify", "daphx_t010_verify",
                     "daphx_value", "daphx_value_v",
                     "daphx_value_v_t005", "daphx_value_v_t010", "daphx_value_v_t020",
                     "daphx_t001_mink6", "daphx_t005_mink6",
                     "daphx_ptop1_t001", "daphx_ptop1_t005", "daphx_ptop1_t010",
                     "random_avg8", "random_avg10",
                     "uncertainty_p50", "uncertainty_p70",
                     "entropy_1.0", "entropy_0.5",
                     "verify_only_6", "verify_only_8", "verify_only_12",
                     "verify_selective_6", "verify_selective_8"]:
            if name not in summary:
                continue
            s = summary[name]
            j = compute_cost_sensitive_utility(s["success_rate"], s["avg_k"], lam=0.1)
            v_info = f"{s['n_verify_used']}/{s['n']}" if s.get("n_verify_used", 0) > 0 else ""
            print(f"  {name:<25} {s['success_rate']:>6.1%} {s['avg_k']:>7.1f} {j:>8.3f} {v_info:>7}")

        all_results[seed] = {"summary": summary,
                             "n_lookahead_train": n_lookahead,
                             "n_lookahead_eval": n_eval_lookahead}

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  R11.2 AGGREGATE: Non-myopic Adaptive Compute + VERIFY")
    print(f"{'='*100}")

    names = ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
             "oracle_6", "oracle_12",
             "oracle_myopic", "oracle_lookahead4", "oracle_lookahead6",
             "daphx_t001", "daphx_t005", "daphx_t010", "daphx_t020",
             "daphx_t001_verify", "daphx_t005_verify", "daphx_t010_verify",
             "daphx_value", "daphx_value_v",
             "daphx_value_v_t005", "daphx_value_v_t010", "daphx_value_v_t020",
             "daphx_t001_mink6", "daphx_t005_mink6",
             "daphx_ptop1_t001", "daphx_ptop1_t005", "daphx_ptop1_t010",
             "random_avg8", "random_avg10",
             "uncertainty_p50", "uncertainty_p70",
             "entropy_1.0", "entropy_0.5",
             "verify_only_6", "verify_only_8", "verify_only_12",
             "verify_selective_6", "verify_selective_8"]

    print(f"{'System':<25} {'Mean':>7} {'Std':>7} {'AvgK':>7} {'J(0.1)':>8} {'J(0.2)':>8}")
    print("-" * 70)

    agg = {}
    for name in names:
        accs = [all_results[s]["summary"][name]["success_rate"]
                for s in [42, 123, 7, 99, 2024] if name in all_results[s]["summary"]]
        ks = [all_results[s]["summary"][name]["avg_k"]
              for s in [42, 123, 7, 99, 2024] if name in all_results[s]["summary"]]
        if not accs:
            continue
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_k = float(np.mean(ks))
        j01 = compute_cost_sensitive_utility(mean_acc, mean_k, lam=0.1)
        j02 = compute_cost_sensitive_utility(mean_acc, mean_k, lam=0.2)
        agg[name] = {"mean_acc": mean_acc, "std_acc": std_acc, "mean_k": mean_k,
                     "j01": j01, "j02": j02}
        print(f"{name:<25} {mean_acc:>6.1%} {std_acc:>7.1%} {mean_k:>7.1f} {j01:>8.3f} {j02:>8.3f}")

    # Pareto frontier
    print(f"\n  PARETO FRONTIER (Accuracy vs E[K]):")
    pareto_names = [n for n in names if n in agg and not n.startswith("oracle_")]
    pareto_points = [(agg[n]["mean_k"], agg[n]["mean_acc"], n) for n in pareto_names]
    pareto_points.sort()
    for k, acc, name in pareto_points:
        print(f"    ({k:.1f}, {acc:.1%}) — {name}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")
    daphx_names = [n for n in agg if n.startswith("daphx_")]
    if daphx_names:
        best_daphx = max(daphx_names, key=lambda n: agg[n]["j01"])
        d = agg[best_daphx]
        print(f"    Best DAPH-X: {best_daphx} ({d['mean_acc']:.1%}, K={d['mean_k']:.1f}, J={d['j01']:.3f})")

        for mk in [6, 8, 12]:
            mc = f"maxcal_{mk}"
            if mc in agg and abs(agg[mc]["mean_k"] - d["mean_k"]) < 4:
                diff = d["mean_acc"] - agg[mc]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {mc} ({agg[mc]['mean_acc']:.1%}, K={agg[mc]['mean_k']:.1f}): {diff:+.1%} {v}")

        for rn in ["random_avg8", "random_avg10"]:
            if rn in agg:
                diff = d["mean_acc"] - agg[rn]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {rn}: {diff:+.1%} {v}")

        for un in ["uncertainty_p50", "uncertainty_p70"]:
            if un in agg:
                diff = d["mean_acc"] - agg[un]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {un}: {diff:+.1%} {v}")

    # Oracle comparison
    print(f"\n  ORACLE COMPARISON:")
    for on in ["oracle_myopic", "oracle_lookahead4", "oracle_lookahead6"]:
        if on in agg:
            print(f"    {on}: {agg[on]['mean_acc']:.1%} at K={agg[on]['mean_k']:.1f}")

    # Verify analysis
    print(f"\n  VERIFY ANALYSIS:")
    for vn in ["verify_only_6", "verify_only_8", "verify_only_12",
               "verify_selective_6", "verify_selective_8"]:
        if vn in agg:
            mc_name = vn.replace("verify_only", "maxcal").replace("verify_selective", "maxcal")
            if mc_name in agg:
                diff = agg[vn]["mean_acc"] - agg[mc_name]["mean_acc"]
                v = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    {vn} vs {mc_name}: {diff:+.1%} {v}")

    # Ablation: full features vs p_top1 only
    print(f"\n  ABLATION: Full features vs p_top1 only:")
    for thresh in ["t001", "t005", "t010"]:
        full_name = f"daphx_{thresh}"
        ptop1_name = f"daphx_ptop1_{thresh}"
        if full_name in agg and ptop1_name in agg:
            diff = agg[full_name]["mean_acc"] - agg[ptop1_name]["mean_acc"]
            v = "ADDS" if diff > 0.01 else ("SAME" if abs(diff) < 0.01 else "HURTS")
            print(f"    {thresh}: full={agg[full_name]['mean_acc']:.1%} vs "
                  f"ptop1={agg[ptop1_name]['mean_acc']:.1%}: {diff:+.1%} {v}")

    # Value model threshold sweep
    print(f"\n  VALUE MODEL THRESHOLD SWEEP:")
    for vn in ["daphx_value_v_t005", "daphx_value_v_t010", "daphx_value_v_t020", "daphx_value_v"]:
        if vn in agg:
            print(f"    {vn}: {agg[vn]['mean_acc']:.1%} at K={agg[vn]['mean_k']:.1f}, J={agg[vn]['j01']:.3f}")

    # Paired bootstrap CI for best DAPH-X vs key baselines
    print(f"\n  PAIRED BOOTSTRAP CI (seed 42, per-task):")
    if 42 in all_results:
        train_tasks_s42, cal_tasks_s42, eval_tasks_s42 = split_tasks(tasks, seed=42)
        corr_model_s42 = train_correctness_r9(flatten_candidates(train_tasks_s42), feature_keys)
        corr_cal_s42 = calibrate_r9(corr_model_s42, flatten_candidates(cal_tasks_s42), feature_keys)
        train_ex_s42 = extract_non_myopic_examples(train_tasks_s42, corr_model_s42, corr_cal_s42, feature_keys)
        rescue_model_s42 = train_lookahead_rescue_model(train_ex_s42)
        rescue_cal_s42 = calibrate_isotonic_enhanced(rescue_model_s42,
            extract_non_myopic_examples(cal_tasks_s42, corr_model_s42, corr_cal_s42, feature_keys),
            "lookahead_rescue")
        value_model_s42 = train_value_model(train_ex_s42)

        # Get per-task utilities for bootstrap
        best_daphx_name = max(daphx_names, key=lambda n: agg[n]["j01"]) if daphx_names else None
        if best_daphx_name:
            daphx_utils = []
            baseline_utils = {}
            for task in eval_tasks_s42:
                daphx_res = run_sequential_policy_v2(
                    task, rescue_model_s42, rescue_cal_s42, corr_model_s42, corr_cal_s42, feature_keys,
                    threshold=0.01, use_verify=True, verify_threshold=0.15)
                daphx_utils.append(daphx_res["utility"])

                # Baselines
                for bname, bfn in [
                    ("maxcal_6", lambda t: max(t["candidates"][:6], key=lambda c: predict_correctness_r9(
                        corr_model_s42, corr_cal_s42, c.get("enriched_features", {}), c, feature_keys))),
                    ("uncertainty_p50", None),
                    ("uncertainty_p70", None),
                ]:
                    if bfn is not None:
                        pick = bfn(task)
                        baseline_utils.setdefault(bname, []).append(100.0 if pick["is_correct"] else 0.0)
                    elif bname == "uncertainty_p50":
                        r = run_uncertainty_policy(task, corr_model_s42, corr_cal_s42, feature_keys, p_threshold=0.5)
                        baseline_utils.setdefault(bname, []).append(r["utility"])
                    elif bname == "uncertainty_p70":
                        r = run_uncertainty_policy(task, corr_model_s42, corr_cal_s42, feature_keys, p_threshold=0.7)
                        baseline_utils.setdefault(bname, []).append(r["utility"])

            for bname, butils in baseline_utils.items():
                if len(daphx_utils) == len(butils):
                    mean_diff, ci_low, ci_high = paired_bootstrap_ci(daphx_utils, butils)
                    sig = "SIGNIFICANT" if (ci_low > 0 or ci_high < 0) else "n.s."
                    print(f"    {best_daphx_name} vs {bname}: "
                          f"mean diff = {mean_diff:+.1f} "
                          f"95% CI = [{ci_low:+.1f}, {ci_high:+.1f}] {sig}")

    # Intervention analysis
    print(f"\n  INTERVENTION ANALYSIS:")
    for s in [42, 123, 7, 99, 2024]:
        if s not in all_results:
            continue
        print(f"    seed {s}: {all_results[s]['n_lookahead_train']} lookahead rescues in train, "
              f"{all_results[s]['n_lookahead_eval']} in eval")

    output = {"aggregate": agg, "per_seed": all_results}
    output_path = R11_DIR / "r11_2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
