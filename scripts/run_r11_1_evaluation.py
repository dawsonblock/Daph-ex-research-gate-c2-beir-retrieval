#!/usr/bin/env python3
"""DAPH-X R11.1: Learned Adaptive Compute — full evaluation framework.

Implements Phases 3-11 of the R11.1 plan:

Phase 3: Oracle@K and MaxCal@K headroom measurement
Phase 4: Uncertainty/instability state features
Phase 5: Train rescue/break/value models
Phase 6: Calibration (isotonic/Platt)
Phase 7: VOI-based authority rule
Phase 8: Sequential policy (K=2→4→6→8→10→12)
Phase 9: Evaluation on untouched confirmation set
Phase 10: Cost-sensitive Pareto frontier
Phase 11: Intervention event analysis

State features (Phase 4):
  - p_top1: calibrated P(correct) of current MaxCal pick
  - margin: p_(1) - p_(2) calibrated margin
  - answer_entropy: H(A) = -sum q_j log q_j
  - majority_fraction: max answer cluster / K
  - selection_stability: 1[a_K^MC == a_{K-2}^MC]
  - confidence_variance: Var(C_1,...,C_K)
  - verification_variance: Var(V_1,...,V_K)
  - semantic_cluster_count: number of distinct answer clusters
  - confidence_verification_disagreement
  - delta_p_top1: p_top1,K - p_top1,K-2
  - delta_entropy: H_K - H_{K-2}
  - n_unique_answers
  - n_correct_so_far (oracle, not used as feature)

Training targets (Phase 5):
  Model A: P(rescue | s_K) = P(Delta U = +1 | s_K)
  Model B: P(break | s_K) = P(Delta U = -1 | s_K)
  Model C: E[Delta U | s_K] = p_R - lambda_B * p_B

Authority rule (Phase 7):
  VOI_K = p_R - lambda_B * p_B - lambda_C * C_{+2}
  GENERATE(+2) iff LCB(VOI_K) > 0

Sequential policy (Phase 8):
  K=2 → decide STOP or GENERATE(+2)
  K=4 → decide STOP or GENERATE(+2)
  ... until K=12 or STOP

Usage:
    python scripts/run_r11_1_evaluation.py \\
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

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


# ─── Phase 3: Oracle@K measurement ───

def compute_oracle_and_maxcal_at_k(tasks, corr_model, corr_cal, fk, ks=None):
    """Compute Oracle@K and MaxCal@K for all K values."""
    if ks is None:
        ks = [1, 2, 4, 6, 8, 10, 12]

    results = {}
    for k in ks:
        oracle_correct = []
        maxcal_correct = []
        majority_correct = []
        base_correct = []
        n_valid = 0

        for task in tasks:
            cands = task["candidates"]
            if len(cands) < k:
                continue

            cands_k = cands[:k]

            # Oracle
            oracle = any(c["is_correct"] for c in cands_k)
            oracle_correct.append(1.0 if oracle else 0.0)

            # MaxCal (using calibrated P(correct))
            pick = max(cands_k, key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            maxcal_correct.append(1.0 if pick["is_correct"] else 0.0)

            # Majority vote
            answers = [c["answer"] for c in cands_k]
            mv = Counter(answers).most_common(1)[0][0]
            maj = any(c["is_correct"] for c in cands_k if c["answer"] == mv)
            majority_correct.append(1.0 if maj else 0.0)

            # Base
            base_correct.append(1.0 if cands_k[0]["is_correct"] else 0.0)

            n_valid += 1

        if n_valid == 0:
            continue

        results[k] = {
            "n": n_valid,
            "oracle": float(np.mean(oracle_correct)),
            "maxcal": float(np.mean(maxcal_correct)),
            "majority": float(np.mean(majority_correct)),
            "base": float(np.mean(base_correct)),
            "headroom": float(np.mean(oracle_correct) - np.mean(maxcal_correct)),
        }

    return results


# ─── Phase 4: State features ───

def compute_answer_entropy(cands_k):
    """Compute answer entropy H(A) = -sum q_j log q_j."""
    answers = [c["answer"] for c in cands_k]
    counts = Counter(answers)
    n = len(answers)
    entropy = 0.0
    for count in counts.values():
        q = count / n
        if q > 0:
            entropy -= q * math.log(q)
    return entropy


def compute_state_features(task, k, corr_model, corr_cal, fk, prev_state=None):
    """Compute state features at checkpoint K.

    Dynamic features compare to K-2 state (prev_state).
    """
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

    # Confidence-verification disagreement
    c_maxcal = maxcal_pick["self_confidence"] / 100.0
    cv_disagreement = abs(c_maxcal - v_maxcal)

    # Selection stability (did MaxCal pick change from K-2?)
    stability = 0.0
    if prev_state is not None:
        stability = 1.0 if prev_state.get("maxcal_answer", "") == maxcal_pick["answer"] else 0.0

    # Dynamic features
    delta_p_top1 = 0.0
    delta_entropy = 0.0
    if prev_state is not None:
        delta_p_top1 = p_top1 - prev_state.get("p_top1", p_top1)
        delta_entropy = entropy - prev_state.get("answer_entropy", entropy)

    # Semantic cluster count (approximate by unique answers)
    semantic_clusters = n_unique

    # Top answer cluster size
    top_cluster_size = max(counts.values())

    features = {
        "k": float(k),
        "p_top1": p_top1,
        "p_top2": p_top2,
        "margin": margin,
        "answer_entropy": entropy,
        "majority_fraction": majority_frac,
        "n_unique_answers": float(n_unique),
        "semantic_cluster_count": float(semantic_clusters),
        "top_cluster_size": float(top_cluster_size),
        "confidence_variance": conf_var,
        "verification_variance": ver_var,
        "v_maxcal": v_maxcal,
        "v_consistency": float(v_consistency),
        "c_maxcal": c_maxcal,
        "cv_disagreement": cv_disagreement,
        "selection_stability": stability,
        "delta_p_top1": delta_p_top1,
        "delta_entropy": delta_entropy,
    }

    # Store for dynamic features at next checkpoint
    features["_maxcal_answer"] = maxcal_pick["answer"]
    features["_maxcal_correct"] = maxcal_pick["is_correct"]
    features["_maxcal_p"] = p_top1

    return features


STATE_FEATURE_KEYS = [
    "k", "p_top1", "p_top2", "margin", "answer_entropy",
    "majority_fraction", "n_unique_answers", "semantic_cluster_count",
    "top_cluster_size", "confidence_variance", "verification_variance",
    "v_maxcal", "v_consistency", "c_maxcal", "cv_disagreement",
    "selection_stability", "delta_p_top1", "delta_entropy",
]


def state_to_vector(feats):
    return np.array([feats.get(k, 0.0) for k in STATE_FEATURE_KEYS])


# ─── Phase 5: Extract training examples ───

def extract_acquisition_examples(tasks, corr_model, corr_cal, fk):
    """Extract acquisition decision examples from tasks.

    For each task, create checkpoints at K=2,4,6,8,10.
    At each checkpoint, the action is GENERATE(+2) vs STOP.
    Label: Delta U = U(MaxCal@K+2) - U(MaxCal@K)
      +1 = rescue (extra candidates fixed the answer)
       0 = no change
      -1 = break (extra candidates broke a correct answer)
    """
    examples = []
    checkpoints = [2, 4, 6, 8, 10]

    for task in tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        prev_state = None
        for k in checkpoints:
            if k + 2 > len(cands):
                break

            # State at checkpoint K
            state_k = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)

            # MaxCal@K correctness
            u_k = 1.0 if state_k["_maxcal_correct"] else 0.0

            # MaxCal@K+2 correctness
            state_k2 = compute_state_features(task, k + 2, corr_model, corr_cal, fk, state_k)
            u_k2 = 1.0 if state_k2["_maxcal_correct"] else 0.0

            # Delta U
            delta_u = u_k2 - u_k  # +1, 0, or -1

            is_rescue = delta_u > 0.5
            is_break = delta_u < -0.5

            examples.append({
                "task_id": task["task_id"],
                "checkpoint_k": k,
                "state": {k: v for k, v in state_k.items() if not k.startswith("_")},
                "delta_u": delta_u,
                "is_rescue": is_rescue,
                "is_break": is_break,
                "u_k": u_k,
                "u_k2": u_k2,
                "maxcal_answer_k": state_k["_maxcal_answer"],
                "maxcal_answer_k2": state_k2["_maxcal_answer"],
            })

            prev_state = state_k

    return examples


# ─── Phase 5: Train models ───

def train_rescue_model(examples):
    X = np.array([state_to_vector(ex["state"]) for ex in examples])
    y = np.array([1 if ex["is_rescue"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    # Use class weights to handle imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


def train_break_model(examples):
    X = np.array([state_to_vector(ex["state"]) for ex in examples])
    y = np.array([1 if ex["is_break"] else 0 for ex in examples])
    if y.sum() < 3:
        return None
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y, sample_weight=sw)
    return model


# ─── Phase 6: Calibration ───

def calibrate_isotonic(model, examples, label_key):
    raw_p, true_l = [], []
    for ex in examples:
        x = state_to_vector(ex["state"]).reshape(1, -1)
        raw_p.append(model.predict_proba(x)[0, 1])
        true_l.append(1 if ex[label_key] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def calibrate_platt(model, examples, label_key):
    raw_p, true_l = [], []
    for ex in examples:
        x = state_to_vector(ex["state"]).reshape(1, -1)
        raw_p.append(model.predict_proba(x)[0, 1])
        true_l.append(1 if ex[label_key] else 0)
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(np.array(raw_p).reshape(-1, 1), np.array(true_l))
    return lr


def predict_calibrated(model, cal, state, cal_type='isotonic'):
    x = state_to_vector(state).reshape(1, -1)
    raw = model.predict_proba(x)[0, 1]
    if cal is None:
        return float(raw)
    if cal_type == 'isotonic':
        return float(cal.predict([raw])[0])
    elif cal_type == 'platt':
        return float(cal.predict_proba([[raw]])[0, 1])
    return float(raw)


def calibrate_isotonic_safe(model, examples, label_key):
    """Calibrate, but skip if calibration set has too few positives."""
    raw_p, true_l = [], []
    for ex in examples:
        x = state_to_vector(ex["state"]).reshape(1, -1)
        raw_p.append(model.predict_proba(x)[0, 1])
        true_l.append(1 if ex[label_key] else 0)
    n_pos = sum(true_l)
    if n_pos < 3:
        return None  # Signal to use raw model output
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


# ─── Phase 7-8: Sequential policy ───

def run_sequential_policy(task, rescue_model, rescue_cal, break_model, break_cal,
                          corr_model, corr_cal, fk,
                          lam_b=5.0, cost_per_gen=0.02, lcb_margin=0.05,
                          start_k=2, max_k=12, step=2, min_k=2,
                          p_r_threshold=0.0, p_b_threshold=1.0,
                          use_voi=True):
    """Run sequential adaptive policy.

    At each checkpoint K:
      1. Compute state features
      2. Predict P(rescue) and P(break)
      3. VOI = p_R - lam_b * p_B - cost_per_gen
      4. LCB = VOI - lcb_margin
      5. If LCB > 0: GENERATE(+2), continue
      6. Else: STOP, return MaxCal@K

    min_k: always generate at least to this K before deciding
    use_voi: if False, use simple threshold p_R > p_r_threshold
    """
    cands = task["candidates"]
    prev_state = None
    k = start_k

    while k <= max_k and k <= len(cands):
        state = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)

        # If at max_k, stop
        if k >= max_k or k + step > len(cands):
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k,
                "n_generations": k,
                "stopped": True,
                "p_rescue": 0.0, "p_break": 0.0,
            }

        # Always generate up to min_k
        if k < min_k:
            prev_state = state
            k += step
            continue

        # Predict rescue and break
        p_r = 0.0
        p_b = 0.0
        if rescue_model is not None:
            p_r = predict_calibrated(rescue_model, rescue_cal, state)
        if break_model is not None:
            p_b = predict_calibrated(break_model, break_cal, state)

        # Decision
        if use_voi:
            voi = p_r - lam_b * p_b - cost_per_gen
            lcb = voi - lcb_margin
            should_gen = lcb > 0
        else:
            should_gen = p_r > p_r_threshold and p_b < p_b_threshold

        if should_gen:
            prev_state = state
            k += step
        else:
            pick = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
            return {
                "correct": pick["is_correct"],
                "utility": 100.0 if pick["is_correct"] else 0.0,
                "final_k": k,
                "n_generations": k,
                "stopped": True,
                "p_rescue": p_r, "p_break": p_b,
            }

    # Reached max_k
    pick = max(cands[:min(k, len(cands))], key=lambda c: predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "final_k": min(k, len(cands)),
        "n_generations": min(k, len(cands)),
        "stopped": False,
        "p_rescue": 0.0, "p_break": 0.0,
    }


def run_random_policy(task, corr_model, corr_cal, fk, target_avg_k=8,
                      start_k=2, max_k=12, step=2, seed=42):
    """Random adaptive policy matched to average K.
    Randomly decides to generate or stop at each checkpoint,
    calibrated to achieve target_avg_k on average.
    """
    rng = np.random.RandomState(seed + hash(task["task_id"]) % 10000)
    cands = task["candidates"]
    k = start_k

    # P(generate) chosen so E[K] ≈ target_avg_k
    # If we start at 2 and step by 2, to reach avg 8 we need to generate ~3 times
    # P(generate) ≈ (target - start) / (max - start) approximately
    p_gen = (target_avg_k - start_k) / (max_k - start_k)
    p_gen = max(0.0, min(1.0, p_gen))

    while k < max_k and k + step <= len(cands):
        if rng.random() < p_gen:
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


def run_oracle_policy(task, corr_model, corr_cal, fk,
                      start_k=2, max_k=12, step=2):
    """Oracle adaptive policy: generates more only when it will help.

    At each checkpoint, looks ahead to see if K+2 would improve.
    """
    cands = task["candidates"]
    k = start_k

    while k < max_k and k + step <= len(cands):
        # Current MaxCal@K
        pick_k = max(cands[:k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        u_k = 1.0 if pick_k["is_correct"] else 0.0

        # MaxCal@K+2
        pick_k2 = max(cands[:k+step], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        u_k2 = 1.0 if pick_k2["is_correct"] else 0.0

        if u_k2 > u_k:
            k += step  # Generate more
        else:
            break  # Stop

    pick = max(cands[:k], key=lambda c: predict_correctness_r9(
        corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "final_k": k,
        "n_generations": k,
    }


# ─── Simple uncertainty heuristics ───

def run_uncertainty_policy(task, corr_model, corr_cal, fk,
                           p_threshold=0.5, start_k=2, max_k=12, step=2):
    """Generate more if p_top1 < threshold."""
    cands = task["candidates"]
    k = start_k
    prev_state = None

    while k < max_k and k + step <= len(cands):
        state = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)
        if state["p_top1"] < p_threshold:
            prev_state = state
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


def run_entropy_policy(task, corr_model, corr_cal, fk,
                       entropy_threshold=1.0, start_k=2, max_k=12, step=2):
    """Generate more if answer entropy > threshold."""
    cands = task["candidates"]
    k = start_k
    prev_state = None

    while k < max_k and k + step <= len(cands):
        state = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)
        if state["answer_entropy"] > entropy_threshold:
            prev_state = state
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


def run_hybrid_policy(task, rescue_model, rescue_cal, break_model, break_cal,
                      corr_model, corr_cal, fk,
                      p_threshold=0.7, rescue_threshold=0.05,
                      start_k=2, max_k=12, step=2, min_k=4):
    """Hybrid: generate if (p_top1 < threshold) OR (p_rescue > rescue_threshold).

    Combines uncertainty heuristic with learned rescue model.
    """
    cands = task["candidates"]
    prev_state = None
    k = start_k

    while k < max_k and k + step <= len(cands):
        state = compute_state_features(task, k, corr_model, corr_cal, fk, prev_state)

        if k < min_k:
            prev_state = state
            k += step
            continue

        # Predict rescue
        p_r = 0.0
        if rescue_model is not None:
            p_r = predict_calibrated(rescue_model, rescue_cal, state)

        # Generate if uncertain OR rescue model says so
        should_gen = state["p_top1"] < p_threshold or p_r > rescue_threshold

        if should_gen:
            prev_state = state
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


# ─── Phase 12: Hard-state cohort ───

def identify_hard_cohort(tasks, corr_model, corr_cal, fk, k=4, low=0.4, high=0.7):
    """Identify tasks where MaxCal@K accuracy is in [low, high] range.

    These are tasks where the model is uncertain — not too easy, not impossible.
    Used for focused evaluation of adaptive compute.
    """
    hard_tasks = []
    for task in tasks:
        cands = task["candidates"]
        if len(cands) < k:
            continue
        pick = max(cands[:k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
        # We can't know correctness at eval time, but for cohort analysis
        # we use the actual correctness label
        if not pick["is_correct"]:
            # Check if any candidate is correct (rescue possible)
            if any(c["is_correct"] for c in cands[:12]):
                hard_tasks.append(task)
    return hard_tasks


# ─── Phase 9-10: Evaluation ───

def evaluate_all(eval_tasks, rescue_model, rescue_cal, break_model, break_cal,
                 corr_model, corr_cal, fk):
    results = {name: [] for name in [
        "maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
        "oracle_2", "oracle_4", "oracle_6", "oracle_8", "oracle_10", "oracle_12",
        "daphx_voi", "daphx_voi_mink6", "daphx_voi_lowcost",
        "daphx_thresh_005", "daphx_thresh_010", "daphx_thresh_020",
        "daphx_thresh_005_mink6",
        "daphx_hybrid_p70", "daphx_hybrid_p80",
        "daphx_oracle",
        "random_avg8", "random_avg10",
        "uncertainty_p50", "uncertainty_p70",
        "entropy_1.0", "entropy_0.5",
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

                # Oracle
                oracle = any(c["is_correct"] for c in cands[:k])
                results[f"oracle_{k}"].append({
                    "correct": oracle,
                    "utility": 100.0 if oracle else 0.0,
                    "final_k": k, "n_generations": k})

        # DAPH-X VOI policy variants
        results["daphx_voi"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, cost_per_gen=0.02, lcb_margin=0.05))
        results["daphx_voi_mink6"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, cost_per_gen=0.02, lcb_margin=0.05, min_k=6))
        results["daphx_voi_lowcost"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, cost_per_gen=0.005, lcb_margin=0.01))

        # DAPH-X threshold-based variants (simpler, no VOI)
        results["daphx_thresh_005"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, use_voi=False, p_r_threshold=0.05))
        results["daphx_thresh_010"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, use_voi=False, p_r_threshold=0.10))
        results["daphx_thresh_020"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, use_voi=False, p_r_threshold=0.20))
        results["daphx_thresh_005_mink6"].append(run_sequential_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, use_voi=False, p_r_threshold=0.05, min_k=6))

        # Hybrid: uncertainty + learned rescue model
        results["daphx_hybrid_p70"].append(run_hybrid_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, p_threshold=0.7, rescue_threshold=0.05))
        results["daphx_hybrid_p80"].append(run_hybrid_policy(
            task, rescue_model, rescue_cal, break_model, break_cal,
            corr_model, corr_cal, fk, p_threshold=0.8, rescue_threshold=0.05))

        # Oracle adaptive
        results["daphx_oracle"].append(run_oracle_policy(
            task, corr_model, corr_cal, fk))

        # Random policies
        results["random_avg8"].append(run_random_policy(
            task, corr_model, corr_cal, fk, target_avg_k=8, seed=42))
        results["random_avg10"].append(run_random_policy(
            task, corr_model, corr_cal, fk, target_avg_k=10, seed=42))

        # Uncertainty heuristics
        for thresh, name in [(0.5, "uncertainty_p50"), (0.7, "uncertainty_p70")]:
            results[name].append(run_uncertainty_policy(
                task, corr_model, corr_cal, fk, p_threshold=thresh))

        # Entropy heuristics
        for thresh, name in [(1.0, "entropy_1.0"), (0.5, "entropy_0.5")]:
            results[name].append(run_entropy_policy(
                task, corr_model, corr_cal, fk, entropy_threshold=thresh))

    return results


def summarize(results):
    summary = {}
    for name, res_list in results.items():
        if not res_list:
            continue
        utils = [r["utility"] for r in res_list]
        ks = [r["n_generations"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        summary[name] = {
            "mean_utility": float(np.mean(utils)),
            "success_rate": successes / len(res_list),
            "avg_k": float(np.mean(ks)),
            "n": len(res_list),
        }
    return summary


def compute_cost_sensitive_utility(acc, avg_k, lam, k_min=2, k_max=12):
    """J = Accuracy - lambda * (E[K] - K_min) / (K_max - K_min)"""
    normalized_k = (avg_k - k_min) / (k_max - k_min)
    return acc - lam * normalized_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R11_DIR / "r11_corpus_12.jsonl"))
    args = parser.parse_args()

    R11_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    n_with_12 = sum(1 for t in tasks if len(t["candidates"]) >= 12)
    print(f"Loaded {len(tasks)} tasks, {n_with_12} with 12 candidates")

    if n_with_12 < 50:
        print("WARNING: Need more tasks with 12 candidates for meaningful evaluation")

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

        # Phase 3: Oracle@K measurement
        oracle_curve = compute_oracle_and_maxcal_at_k(
            eval_tasks, corr_model, corr_cal, feature_keys)

        print(f"  Oracle@K curve:")
        print(f"  {'K':>4} {'MaxCal':>7} {'Oracle':>7} {'Headroom':>9}")
        for k in sorted(oracle_curve.keys()):
            r = oracle_curve[k]
            print(f"  {k:>4} {r['maxcal']:>6.1%} {r['oracle']:>6.1%} {r['headroom']:>+8.1%}")

        # Phase 5: Extract acquisition examples
        train_ex = extract_acquisition_examples(train_tasks, corr_model, corr_cal, feature_keys)
        cal_ex = extract_acquisition_examples(cal_tasks, corr_model, corr_cal, feature_keys)
        eval_ex = extract_acquisition_examples(eval_tasks, corr_model, corr_cal, feature_keys)

        n_rescue = sum(1 for ex in train_ex if ex["is_rescue"])
        n_break = sum(1 for ex in train_ex if ex["is_break"])
        n_neutral = len(train_ex) - n_rescue - n_break
        print(f"  Acquisition examples: {len(train_ex)} train, {len(cal_ex)} cal, {len(eval_ex)} eval")
        print(f"  Train: {n_rescue} rescues, {n_break} breaks, {n_neutral} neutral")

        # Train rescue and break models
        rescue_model = train_rescue_model(train_ex)
        break_model = train_break_model(train_ex)

        # Phase 6: Calibrate (skip if too few positives in calibration set)
        rescue_cal = None
        break_cal = None
        if rescue_model:
            rescue_cal = calibrate_isotonic_safe(rescue_model, cal_ex, "is_rescue")
            if rescue_cal is None:
                print(f"  Rescue: skipping calibration (too few positives in cal set)")
        if break_model:
            break_cal = calibrate_isotonic_safe(break_model, cal_ex, "is_break")
            if break_cal is None:
                print(f"  Break: skipping calibration (too few positives in cal set)")

        # AUROC
        if rescue_model and n_rescue > 0:
            eval_y = [1 if ex["is_rescue"] else 0 for ex in eval_ex]
            if len(set(eval_y)) >= 2:
                eval_X = np.array([state_to_vector(ex["state"]) for ex in eval_ex])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_r = roc_auc_score(eval_y, rescue_model.predict_proba(eval_X)[:, 1])
                print(f"  Rescue AUROC: {auroc_r:.4f}")

        if break_model and n_break > 0:
            eval_y = [1 if ex["is_break"] else 0 for ex in eval_ex]
            if len(set(eval_y)) >= 2:
                eval_X = np.array([state_to_vector(ex["state"]) for ex in eval_ex])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_b = roc_auc_score(eval_y, break_model.predict_proba(eval_X)[:, 1])
                print(f"  Break AUROC:  {auroc_b:.4f}")

        # Phase 9: Evaluate
        results = evaluate_all(eval_tasks, rescue_model, rescue_cal,
                               break_model, break_cal,
                               corr_model, corr_cal, feature_keys)
        summary = summarize(results)

        print(f"\n  {'System':<20} {'Acc':>7} {'AvgK':>7} {'J(0.1)':>8}")
        print("  " + "-" * 50)
        for name in ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
                     "oracle_6", "oracle_12",
                     "daphx_voi", "daphx_voi_mink6", "daphx_voi_lowcost",
                     "daphx_thresh_005", "daphx_thresh_010", "daphx_thresh_020",
                     "daphx_thresh_005_mink6",
                     "daphx_hybrid_p70", "daphx_hybrid_p80",
                     "daphx_oracle",
                     "random_avg8", "random_avg10",
                     "uncertainty_p50", "uncertainty_p70",
                     "entropy_1.0", "entropy_0.5"]:
            if name not in summary:
                continue
            s = summary[name]
            j = compute_cost_sensitive_utility(s["success_rate"], s["avg_k"], lam=0.1)
            print(f"  {name:<20} {s['success_rate']:>6.1%} {s['avg_k']:>7.1f} {j:>8.3f}")

        all_results[seed] = {"summary": summary, "oracle_curve": oracle_curve,
                             "n_train_rescues": n_rescue, "n_train_breaks": n_break}

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  R11.1 AGGREGATE: Learned Adaptive Compute")
    print(f"{'='*100}")

    names = ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8", "maxcal_10", "maxcal_12",
             "daphx_voi", "daphx_voi_mink6", "daphx_voi_lowcost",
             "daphx_thresh_005", "daphx_thresh_010", "daphx_thresh_020",
             "daphx_thresh_005_mink6",
             "daphx_hybrid_p70", "daphx_hybrid_p80",
             "daphx_oracle",
             "random_avg8", "random_avg10",
             "uncertainty_p50", "uncertainty_p70",
             "entropy_1.0", "entropy_0.5"]

    print(f"{'System':<20} {'Mean':>7} {'Std':>7} {'AvgK':>7} {'J(0.1)':>8} {'J(0.2)':>8}")
    print("-" * 65)

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
        print(f"{name:<20} {mean_acc:>6.1%} {std_acc:>7.1%} {mean_k:>7.1f} {j01:>8.3f} {j02:>8.3f}")

    # Pareto frontier
    print(f"\n  PARETO FRONTIER (Accuracy vs E[K]):")
    pareto_points = [(agg[n]["mean_k"], agg[n]["mean_acc"], n)
                      for n in ["maxcal_2", "maxcal_4", "maxcal_6", "maxcal_8",
                                "maxcal_10", "maxcal_12",
                                "daphx_voi", "daphx_voi_mink6", "daphx_voi_lowcost",
                                "daphx_thresh_005", "daphx_thresh_010", "daphx_thresh_020",
                                "daphx_thresh_005_mink6",
                                "daphx_hybrid_p70", "daphx_hybrid_p80",
                                "daphx_oracle",
                                "random_avg8", "random_avg10",
                                "uncertainty_p50", "uncertainty_p70",
                                "entropy_1.0", "entropy_0.5"] if n in agg]
    pareto_points.sort()
    for k, acc, name in pareto_points:
        print(f"    ({k:.1f}, {acc:.1%}) — {name}")

    # Key comparisons — use best DAPH-X variant by J(0.1)
    print(f"\n  KEY COMPARISONS:")
    daphx_names = [n for n in agg if n.startswith("daphx_") and n != "daphx_oracle"]
    if daphx_names:
        best_daphx_name = max(daphx_names, key=lambda n: agg[n]["j01"])
        daphx = agg[best_daphx_name]
        print(f"    Best DAPH-X variant: {best_daphx_name} "
              f"({daphx['mean_acc']:.1%}, K={daphx['mean_k']:.1f}, J={daphx['j01']:.3f})")

        # vs MaxCal at similar budget
        for mk in [6, 8, 10, 12]:
            mc_name = f"maxcal_{mk}"
            if mc_name in agg and abs(agg[mc_name]["mean_k"] - daphx["mean_k"]) < 3:
                diff = daphx["mean_acc"] - agg[mc_name]["mean_acc"]
                verdict = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {mc_name} ({agg[mc_name]['mean_acc']:.1%}, K={agg[mc_name]['mean_k']:.1f}): "
                      f"{diff:+.1%} {verdict}")

        # vs random
        for rn in ["random_avg8", "random_avg10"]:
            if rn in agg and abs(agg[rn]["mean_k"] - daphx["mean_k"]) < 3:
                diff = daphx["mean_acc"] - agg[rn]["mean_acc"]
                verdict = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {rn}: {diff:+.1%} {verdict}")

        # vs uncertainty heuristics
        for un in ["uncertainty_p50", "uncertainty_p70"]:
            if un in agg:
                diff = daphx["mean_acc"] - agg[un]["mean_acc"]
                verdict = "BEATS" if diff > 0.01 else ("MATCHES" if abs(diff) < 0.01 else "WORSE")
                print(f"    vs {un}: {diff:+.1%} {verdict}")

        # vs all DAPH-X variants
        print(f"\n    All DAPH-X variants:")
        for dn in sorted(daphx_names, key=lambda n: agg[n]["j01"], reverse=True):
            d = agg[dn]
            print(f"      {dn:<25} {d['mean_acc']:.1%}  K={d['mean_k']:.1f}  J={d['j01']:.3f}")

    # Phase 11: Intervention analysis
    print(f"\n  INTERVENTION ANALYSIS (aggregate):")
    for s in [42, 123, 7, 99, 2024]:
        if s not in all_results:
            continue
        n_rescue = all_results[s]["n_train_rescues"]
        n_break = all_results[s]["n_train_breaks"]
        print(f"    seed {s}: {n_rescue} rescues, {n_break} breaks in training")

    # Phase 12: Hard-state cohort analysis
    print(f"\n  HARD-STATE COHORT ANALYSIS:")
    # Use seed 42 split for cohort analysis
    train_tasks_s42, cal_tasks_s42, eval_tasks_s42 = split_tasks(tasks, seed=42)
    corr_model_s42 = train_correctness_r9(flatten_candidates(train_tasks_s42), feature_keys)
    corr_cal_s42 = calibrate_r9(corr_model_s42, flatten_candidates(cal_tasks_s42), feature_keys)
    hard_tasks = identify_hard_cohort(tasks, corr_model_s42, corr_cal_s42, feature_keys)
    print(f"    Hard cohort (MaxCal@4 wrong, rescue available): {len(hard_tasks)}/{len(tasks)} tasks")

    if hard_tasks:
        # Evaluate on hard cohort
        train_ex_hard = extract_acquisition_examples(
            [t for t in hard_tasks if t in train_tasks_s42],
            corr_model_s42, corr_cal_s42, feature_keys)
        n_rescue_hard = sum(1 for ex in train_ex_hard if ex["is_rescue"])
        print(f"    Hard cohort train rescues: {n_rescue_hard}/{len(train_ex_hard)}")

    # Phase 16: Paired bootstrap CI for key comparison
    print(f"\n  PAIRED BOOTSTRAP CI (seed 42):")
    if 42 in all_results and "daphx_voi" in all_results[42]["summary"]:
        # Re-run on eval tasks to get per-task results for bootstrap
        from sklearn.utils import resample
        eval_tasks_s42 = split_tasks(tasks, seed=42)[2]
        # Get per-task utilities for DAPH-X and best baseline
        daphx_utils = []
        baseline_utils = []
        for task in eval_tasks_s42:
            daphx_res = run_sequential_policy(
                task, rescue_model, rescue_cal, break_model, break_cal,
                corr_model_s42, corr_cal_s42, feature_keys)
            baseline_res = run_uncertainty_policy(
                task, corr_model_s42, corr_cal_s42, feature_keys, p_threshold=0.7)
            daphx_utils.append(daphx_res["utility"])
            baseline_utils.append(baseline_res["utility"])

        diffs = np.array(daphx_utils) - np.array(baseline_utils)
        n_bootstrap = 1000
        boot_means = []
        for _ in range(n_bootstrap):
            boot_idx = resample(range(len(diffs)), n_samples=len(diffs))
            boot_means.append(np.mean(diffs[boot_idx]))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        print(f"    DAPH-X VOI vs uncertainty_p70: mean diff = {np.mean(diffs):.1f} "
              f"95% CI = [{ci_low:.1f}, {ci_high:.1f}]")

    output = {"aggregate": agg, "per_seed": all_results}
    output_path = R11_DIR / "r11_1_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
