#!/usr/bin/env python3
"""DAPH-X R10: Selective Policy Improvement over MaxCal.

The key shift from R9:
  R9: base = raw model (candidates[0]), override = MaxCal pick
      → Gate A was just a noisy filter on when to trust MaxCal
      → ALL interventions were "follow MaxCal over raw base"
      → Zero incremental value over MaxCal

  R10: base = MaxCal pick, override = a different candidate
       → Authority must find cases where MaxCal is WRONG
       → Learn P(R) and P(B) only on disagreement states
       → Use asymmetric utility gate: FORCE if LCB(P(R) - λ*P(B) - C) > 0

Mathematical formulation:
  π_B = MaxCal (frozen as base policy)
  a_X = candidate that authority might prefer over MaxCal

  For each task:
    1. MaxCal picks a_B = argmax P(correct_i)
    2. Authority evaluates: should we override to some a_X ≠ a_B?
    3. If yes: E[ΔU] = P(R) - λ*P(B) - C
    4. FORCE only if LCB(E[ΔU]) > 0

  P(R) = P(a_X correct AND a_B wrong | features)
  P(B) = P(a_X wrong AND a_B correct | features)

Training:
  - Only use disagreement states (where MaxCal ≠ best possible)
  - Learn P(R) and P(B) directly from data
  - Use conformal lower bound on E[ΔU]

Systems compared:
  1. Base (raw model)
  2. Majority vote
  3. Max confidence
  4. MaxCal (the new base policy)
  5. R10 Gate: MaxCal + selective override with asymmetric utility
  6. R10 Gate (various λ values)

Usage:
    python scripts/run_r10_evaluation.py \\
        --corpus experiments/daph_x/cross_verification/cv_corpus_v2.jsonl
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
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

R10_DIR = REPO_ROOT / "experiments/daph_x/r10"

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
    """Compute MaxCal pick and P(correct) for all candidates in all tasks."""
    for task in tasks:
        for c in task["candidates"]:
            c["p_correct"] = predict_correctness_r9(
                corr_model, corr_cal, c["enriched_features"], c, fk)
        task["maxcal_pick"] = max(task["candidates"], key=lambda c: c["p_correct"])
        task["maxcal_p"] = task["maxcal_pick"]["p_correct"]
    return tasks


def build_disagreement_features(task, alt_candidate):
    """Build features for the decision: should we override MaxCal to alt_candidate?

    Features describe the RELATIONSHIP between MaxCal pick and the alternative,
    not just the alternative's standalone quality.
    """
    maxcal = task["maxcal_pick"]
    alt = alt_candidate

    # P(correct) gap
    p_gap = alt["p_correct"] - maxcal["p_correct"]

    # Verification signals
    v_maxcal = maxcal.get("verification_score", 0.5)
    v_alt = alt.get("verification_score", 0.5)
    v_gap = v_alt - v_maxcal
    v_maxcal_low = 1.0 if v_maxcal < 0.5 else 0.0
    v_alt_high = 1.0 if v_alt > 0.6 else 0.0
    v_disagreement = 1.0 if abs(v_gap) > 0.2 else 0.0

    # Verification consistency
    vc_maxcal = maxcal.get("verification_consistent", 0)
    vc_alt = alt.get("verification_consistent", 0)
    vc_gap = vc_alt - vc_maxcal

    # Confidence signals
    c_maxcal = maxcal["self_confidence"] / 100.0
    c_alt = alt["self_confidence"] / 100.0
    c_gap = c_alt - c_maxcal

    # Pairwise signals
    pw_maxcal = maxcal.get("pairwise_winrate", 0.5)
    pw_alt = alt.get("pairwise_winrate", 0.5)
    pw_gap = pw_alt - pw_maxcal

    # Answer-level features
    ans_maxcal = maxcal["answer"]
    ans_alt = alt["answer"]
    answers_same = 1.0 if ans_maxcal == ans_alt else 0.0
    ans_len_diff = len(ans_alt) - len(ans_maxcal)

    # Majority vote
    answers = [c["answer"] for c in task["candidates"]]
    majority = Counter(answers).most_common(1)[0][0]
    majority_agrees_maxcal = 1.0 if majority == ans_maxcal else 0.0
    majority_agrees_alt = 1.0 if majority == ans_alt else 0.0
    majority_disagrees_maxcal = 1.0 if majority != ans_maxcal else 0.0

    # How many candidates agree with MaxCal vs alt
    n_agree_maxcal = sum(1 for a in answers if a == ans_maxcal)
    n_agree_alt = sum(1 for a in answers if a == ans_alt)
    n_unique = len(set(answers))
    agreement_gap = n_agree_alt - n_agree_maxcal

    # MaxCal uncertainty
    maxcal_p = task["maxcal_p"]
    maxcal_uncertainty = 1.0 - maxcal_p

    # Alt P(correct) absolute
    alt_p = alt["p_correct"]

    # Answer similarity (are they format variants?)
    ans_sim = alt.get("answer_avg_similarity", 0.0)

    # Numeric equivalence check
    try:
        num_maxcal = float(ans_maxcal.replace(",", ""))
        num_alt = float(ans_alt.replace(",", ""))
        numeric_equal = 1.0 if abs(num_maxcal - num_alt) < 1e-9 else 0.0
    except (ValueError, AttributeError):
        numeric_equal = 0.0 if ans_maxcal != ans_alt else 1.0

    features = {
        "p_gap": p_gap,
        "p_maxcal": maxcal_p,
        "p_alt": alt_p,
        "maxcal_uncertainty": maxcal_uncertainty,
        "v_maxcal": v_maxcal,
        "v_alt": v_alt,
        "v_gap": v_gap,
        "v_maxcal_low": v_maxcal_low,
        "v_alt_high": v_alt_high,
        "v_disagreement": v_disagreement,
        "vc_maxcal": vc_maxcal,
        "vc_alt": vc_alt,
        "vc_gap": vc_gap,
        "c_maxcal": c_maxcal,
        "c_alt": c_alt,
        "c_gap": c_gap,
        "pw_maxcal": pw_maxcal,
        "pw_alt": pw_alt,
        "pw_gap": pw_gap,
        "answers_same": answers_same,
        "ans_len_diff": ans_len_diff,
        "majority_agrees_maxcal": majority_agrees_maxcal,
        "majority_agrees_alt": majority_agrees_alt,
        "majority_disagrees_maxcal": majority_disagrees_maxcal,
        "n_agree_maxcal": n_agree_maxcal,
        "n_agree_alt": n_agree_alt,
        "n_unique": n_unique,
        "agreement_gap": agreement_gap,
        "ans_sim": ans_sim,
        "numeric_equal": numeric_equal,
    }

    return features


DISAGREEMENT_FEATURE_KEYS = [
    "p_gap", "p_maxcal", "p_alt", "maxcal_uncertainty",
    "v_maxcal", "v_alt", "v_gap", "v_maxcal_low", "v_alt_high", "v_disagreement",
    "vc_maxcal", "vc_alt", "vc_gap",
    "c_maxcal", "c_alt", "c_gap",
    "pw_maxcal", "pw_alt", "pw_gap",
    "answers_same", "ans_len_diff",
    "majority_agrees_maxcal", "majority_agrees_alt", "majority_disagrees_maxcal",
    "n_agree_maxcal", "n_agree_alt", "n_unique", "agreement_gap",
    "ans_sim", "numeric_equal",
]


def extract_disagreement_examples(tasks):
    """Extract all (MaxCal, alternative) pairs for disagreement training.

    For each task, MaxCal picks one candidate. Every OTHER candidate is a
    potential override target. We label each pair:
      rescue: MaxCal wrong, alt correct
      break:  MaxCal correct, alt wrong
      neutral: both same outcome
    """
    examples = []
    for task in tasks:
        maxcal = task["maxcal_pick"]
        for c in task["candidates"]:
            if c["candidate_id"] == maxcal["candidate_id"]:
                continue
            feats = build_disagreement_features(task, c)
            is_rescue = (not maxcal["is_correct"]) and c["is_correct"]
            is_break = maxcal["is_correct"] and (not c["is_correct"])
            examples.append({
                "features": feats,
                "is_rescue": is_rescue,
                "is_break": is_break,
                "maxcal_correct": maxcal["is_correct"],
                "alt_correct": c["is_correct"],
                "task_id": task["task_id"],
                "maxcal_answer": maxcal["answer"],
                "alt_answer": c["answer"],
            })
    return examples


def build_feature_vector_from_dict(feats, keys):
    return np.array([feats.get(k, 0.0) for k in keys])


def train_rescue_model(train_examples, keys):
    """Train P(rescue | features) — MaxCal wrong AND alt correct."""
    X, y = [], []
    for ex in train_examples:
        X.append(build_feature_vector_from_dict(ex["features"], keys))
        y.append(1 if ex["is_rescue"] else 0)
    X, y = np.array(X), np.array(y)
    if y.sum() < 3:
        return None
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


def train_break_model(train_examples, keys):
    """Train P(break | features) — MaxCal correct AND alt wrong."""
    X, y = [], []
    for ex in train_examples:
        X.append(build_feature_vector_from_dict(ex["features"], keys))
        y.append(1 if ex["is_break"] else 0)
    X, y = np.array(X), np.array(y)
    if y.sum() < 3:
        return None
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model


def calibrate_model(model, cal_examples, keys, label_key):
    """Isotonic calibration of model probabilities."""
    raw_p, true_l = [], []
    for ex in cal_examples:
        x = build_feature_vector_from_dict(ex["features"], keys)
        raw_p.append(model.predict_proba([x])[0, 1])
        true_l.append(1 if ex[label_key] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def predict_rescue(model, cal, feats, keys):
    if model is None:
        return 0.0
    x = build_feature_vector_from_dict(feats, keys)
    raw = model.predict_proba([x])[0, 1]
    return float(cal.predict([raw])[0]) if cal else float(raw)


def predict_break(model, cal, feats, keys):
    if model is None:
        return 0.0
    x = build_feature_vector_from_dict(feats, keys)
    raw = model.predict_proba([x])[0, 1]
    return float(cal.predict([raw])[0]) if cal else float(raw)


# ─── Heuristic rescue gates (don't need rescue training examples) ───

def r10_verification_gate(task, break_model, break_cal, keys,
                          v_maxcal_low=0.4, v_alt_high=0.6, p_break_max=0.1):
    """Override MaxCal when verification strongly disagrees, IF break model says safe.

    Trigger: MaxCal verification is low (< v_maxcal_low) AND
             an alternative has high verification (> v_alt_high)
    Safety:  P(break) < p_break_max from the trained break model
    """
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]
    v_maxcal = maxcal.get("verification_score", 0.5)

    # If MaxCal verification is high, trust it
    if v_maxcal >= v_maxcal_low:
        return {"correct": maxcal["is_correct"],
                "utility": 100.0 if maxcal["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal["candidate_id"],
                "p_rescue": 0.0, "p_break": 0.0}

    # MaxCal verification is low — look for high-verification alternative
    best_alt = None
    best_v = v_alt_high
    for c in cands:
        if c["candidate_id"] == maxcal["candidate_id"]:
            continue
        v = c.get("verification_score", 0.5)
        if v > best_v:
            feats = build_disagreement_features(task, c)
            p_b = predict_break(break_model, break_cal, feats, keys)
            if p_b < p_break_max:
                best_v = v
                best_alt = c

    if best_alt:
        feats = build_disagreement_features(task, best_alt)
        p_b = predict_break(break_model, break_cal, feats, keys)
        return {"correct": best_alt["is_correct"],
                "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"],
                "p_rescue": best_v, "p_break": p_b}

    return {"correct": maxcal["is_correct"],
            "utility": 100.0 if maxcal["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal["candidate_id"],
            "p_rescue": 0.0, "p_break": 0.0}


def r10_majority_gate(task, break_model, break_cal, keys, p_break_max=0.1):
    """Override MaxCal when majority vote disagrees, IF break model says safe.

    Trigger: Majority vote picks a different candidate than MaxCal
    Safety:  P(break) < p_break_max
    """
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]
    answers = [c["answer"] for c in cands]
    majority_answer = Counter(answers).most_common(1)[0][0]

    if majority_answer == maxcal["answer"]:
        return {"correct": maxcal["is_correct"],
                "utility": 100.0 if maxcal["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal["candidate_id"],
                "p_rescue": 0.0, "p_break": 0.0}

    # Majority disagrees with MaxCal — find the majority candidate
    majority_cand = next(c for c in cands if c["answer"] == majority_answer)
    feats = build_disagreement_features(task, majority_cand)
    p_b = predict_break(break_model, break_cal, feats, keys)

    if p_b < p_break_max:
        return {"correct": majority_cand["is_correct"],
                "utility": 100.0 if majority_cand["is_correct"] else 0.0,
                "would_force": True, "pick_id": majority_cand["candidate_id"],
                "p_rescue": 0.0, "p_break": p_b}

    return {"correct": maxcal["is_correct"],
            "utility": 100.0 if maxcal["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal["candidate_id"],
            "p_rescue": 0.0, "p_break": 0.0}


def r10_combined_gate(task, break_model, break_cal, keys,
                      v_maxcal_low=0.4, v_alt_high=0.6, p_break_max=0.15):
    """Combined gate: override when EITHER verification OR majority disagrees,
    IF break model says safe.

    This uses two rescue triggers but one safety filter (break model).
    """
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]
    v_maxcal = maxcal.get("verification_score", 0.5)
    answers = [c["answer"] for c in cands]
    majority_answer = Counter(answers).most_common(1)[0][0]

    # If MaxCal verification is high AND majority agrees, trust it
    if v_maxcal >= v_maxcal_low and majority_answer == maxcal["answer"]:
        return {"correct": maxcal["is_correct"],
                "utility": 100.0 if maxcal["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal["candidate_id"],
                "p_rescue": 0.0, "p_break": 0.0}

    # At least one signal disagrees — find best alternative
    candidates = []
    for c in cands:
        if c["candidate_id"] == maxcal["candidate_id"]:
            continue
        v = c.get("verification_score", 0.5)
        is_majority = (c["answer"] == majority_answer)
        is_high_v = (v >= v_alt_high)

        if is_majority or is_high_v:
            feats = build_disagreement_features(task, c)
            p_b = predict_break(break_model, break_cal, feats, keys)
            if p_b < p_break_max:
                # Score: prioritize majority agreement, then verification
                score = (0.5 if is_majority else 0.0) + (0.3 * v) - (0.5 * p_b)
                candidates.append((score, c, p_b))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        best_alt = candidates[0][1]
        p_b = candidates[0][2]
        return {"correct": best_alt["is_correct"],
                "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"],
                "p_rescue": 0.0, "p_break": p_b}

    return {"correct": maxcal["is_correct"],
            "utility": 100.0 if maxcal["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal["candidate_id"],
            "p_rescue": 0.0, "p_break": 0.0}


def r10_lowconfidence_gate(task, break_model, break_cal, keys,
                           p_maxcal_low=0.5, p_break_max=0.1):
    """Override MaxCal when its P(correct) is low, IF break model says safe.

    Trigger: MaxCal P(correct) < p_maxcal_low (uncertain)
    Action:  Pick highest P(correct) alternative with low break probability
    """
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]
    p_maxcal = task["maxcal_p"]

    if p_maxcal >= p_maxcal_low:
        return {"correct": maxcal["is_correct"],
                "utility": 100.0 if maxcal["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal["candidate_id"],
                "p_rescue": 0.0, "p_break": 0.0}

    # MaxCal is uncertain — look for better alternative
    best_alt = None
    best_p = -1
    for c in cands:
        if c["candidate_id"] == maxcal["candidate_id"]:
            continue
        feats = build_disagreement_features(task, c)
        p_b = predict_break(break_model, break_cal, feats, keys)
        if p_b < p_break_max and c["p_correct"] > best_p:
            best_p = c["p_correct"]
            best_alt = c

    if best_alt:
        feats = build_disagreement_features(task, best_alt)
        p_b = predict_break(break_model, break_cal, feats, keys)
        return {"correct": best_alt["is_correct"],
                "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"],
                "p_rescue": 0.0, "p_break": p_b}

    return {"correct": maxcal["is_correct"],
            "utility": 100.0 if maxcal["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal["candidate_id"],
            "p_rescue": 0.0, "p_break": 0.0}


# ─── R10 Authority Gate ───
def r10_gate(task, rescue_model, rescue_cal, break_model, break_cal,
             keys, lam=5.0, cost=0.0, lcb_quantile=0.10):
    """R10 Selective Policy Improvement gate.

    1. MaxCal is the base policy
    2. For each alternative candidate, compute P(R) and P(B)
    3. E[ΔU] = P(R) - λ*P(B) - cost
    4. FORCE to the best alternative if LCB(E[ΔU]) > 0

    The lower confidence bound uses the calibration set to estimate
    prediction uncertainty. We use a simple conservative approach:
    subtract a margin proportional to the uncertainty of the predictions.
    """
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]

    best_alt = None
    best_e_du = -float('inf')
    best_p_r, best_p_b = 0.0, 0.0

    for c in cands:
        if c["candidate_id"] == maxcal["candidate_id"]:
            continue

        feats = build_disagreement_features(task, c)
        p_r = predict_rescue(rescue_model, rescue_cal, feats, keys)
        p_b = predict_break(break_model, break_cal, feats, keys)

        # Expected delta utility
        e_du = p_r - lam * p_b - cost

        # Conservative lower bound: penalize uncertainty
        # The more uncertain we are about rescue, the more conservative
        uncertainty = p_r * (1 - p_r) + p_b * (1 - p_b)
        lcb = e_du - uncertainty * 2.0  # conservative margin

        if lcb > best_e_du:
            best_e_du = lcb
            best_alt = c
            best_p_r = p_r
            best_p_b = p_b

    # FORCE only if LCB > 0
    if best_alt is not None and best_e_du > 0:
        pick = best_alt
        would_force = True
        p_r = best_p_r
        p_b = best_p_b
    else:
        pick = maxcal
        would_force = False
        p_r = 0.0
        p_b = 0.0

    return {
        "correct": pick["is_correct"],
        "utility": 100.0 if pick["is_correct"] else 0.0,
        "would_force": would_force,
        "pick_id": pick["candidate_id"],
        "p_rescue": p_r,
        "p_break": p_b,
    }


def r10_gate_threshold(task, rescue_model, rescue_cal, break_model, break_cal,
                       keys, tau_r=0.5, tau_b=0.1):
    """Alternative gate: FORCE if P(R) > tau_r AND P(B) < tau_b."""
    maxcal = task["maxcal_pick"]
    cands = task["candidates"]

    best_alt = None
    best_p_r = 0.0
    best_p_b = 1.0

    for c in cands:
        if c["candidate_id"] == maxcal["candidate_id"]:
            continue

        feats = build_disagreement_features(task, c)
        p_r = predict_rescue(rescue_model, rescue_cal, feats, keys)
        p_b = predict_break(break_model, break_cal, feats, keys)

        if p_r > best_p_r:
            best_p_r = p_r
            best_p_b = p_b
            best_alt = c

    if best_alt and best_p_r > tau_r and best_p_b < tau_b:
        return {"correct": best_alt["is_correct"],
                "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"],
                "p_rescue": best_p_r, "p_break": best_p_b}
    return {"correct": maxcal["is_correct"],
            "utility": 100.0 if maxcal["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal["candidate_id"],
            "p_rescue": 0.0, "p_break": 0.0}


# ─── Simple baselines ───
def sys_base(task):
    c = task["candidates"][0]
    return {"correct": c["is_correct"], "utility": 100.0 if c["is_correct"] else 0.0}

def sys_majority_vote(task):
    answers = [c["answer"] for c in task["candidates"]]
    mv = Counter(answers).most_common(1)[0][0]
    pick = next(c for c in task["candidates"] if c["answer"] == mv)
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0}

def sys_max_confidence(task):
    pick = max(task["candidates"], key=lambda c: c["self_confidence"])
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0}

def sys_maxcal(task):
    pick = task["maxcal_pick"]
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0}


def evaluate_all(eval_tasks, rescue_model, rescue_cal, break_model, break_cal, keys):
    results = {name: [] for name in [
        "base", "majority_vote", "max_confidence", "maxcal",
        "r10_lam3", "r10_lam5", "r10_lam10",
        "r10_thresh_r50_b10", "r10_thresh_r30_b05",
        "r10_verify_gate", "r10_majority_gate", "r10_combined_gate",
        "r10_lowconf_gate",
    ]}

    for task in eval_tasks:
        results["base"].append(sys_base(task))
        results["majority_vote"].append(sys_majority_vote(task))
        results["max_confidence"].append(sys_max_confidence(task))
        results["maxcal"].append(sys_maxcal(task))

        # R10 with different lambda values (utility-based, needs rescue model)
        for lam, name in [(3, "r10_lam3"), (5, "r10_lam5"), (10, "r10_lam10")]:
            results[name].append(r10_gate(
                task, rescue_model, rescue_cal, break_model, break_cal,
                keys, lam=lam, cost=0.0))

        # R10 with threshold gates
        results["r10_thresh_r50_b10"].append(r10_gate_threshold(
            task, rescue_model, rescue_cal, break_model, break_cal,
            keys, tau_r=0.5, tau_b=0.1))
        results["r10_thresh_r30_b05"].append(r10_gate_threshold(
            task, rescue_model, rescue_cal, break_model, break_cal,
            keys, tau_r=0.3, tau_b=0.05))

        # Heuristic rescue gates (don't need rescue training examples)
        results["r10_verify_gate"].append(r10_verification_gate(
            task, break_model, break_cal, keys))
        results["r10_majority_gate"].append(r10_majority_gate(
            task, break_model, break_cal, keys))
        results["r10_combined_gate"].append(r10_combined_gate(
            task, break_model, break_cal, keys))
        results["r10_lowconf_gate"].append(r10_lowconfidence_gate(
            task, break_model, break_cal, keys))

    return results


def summarize(results, eval_tasks):
    summary = {}
    maxcal_utils = [100.0 if eval_tasks[i]["maxcal_pick"]["is_correct"] else 0.0
                    for i in range(len(eval_tasks))]

    for name, res_list in results.items():
        utils = [r["utility"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        n_force = sum(1 for r in res_list if r.get("would_force", False))
        # Rescues and breaks are now relative to MaxCal (the base policy)
        rescues = sum(1 for r, mu in zip(res_list, maxcal_utils)
                      if r.get("would_force", False) and r["utility"] > mu + 0.5)
        breaks = sum(1 for r, mu in zip(res_list, maxcal_utils)
                     if r.get("would_force", False) and r["utility"] < mu - 0.5)

        summary[name] = {
            "mean_utility": float(np.mean(utils)),
            "success_rate": successes / len(res_list),
            "n_force": n_force,
            "rescues": rescues,
            "breaks": breaks,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R10_DIR / "r10_corpus.jsonl"))
    args = parser.parse_args()

    R10_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    # Feature extraction
    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    print("Computing enriched features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)
    print(f"Correctness features: {len(feature_keys)} + 10 extra = {len(feature_keys)+10}")
    print(f"Disagreement features: {len(DISAGREEMENT_FEATURE_KEYS)}")

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")

        # Train correctness model (same as R9)
        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, cal_records, feature_keys)

        # Compute MaxCal picks for all tasks
        train_tasks = compute_maxcal_picks(train_tasks, corr_model, corr_cal, feature_keys)
        cal_tasks = compute_maxcal_picks(cal_tasks, corr_model, corr_cal, feature_keys)
        eval_tasks = compute_maxcal_picks(eval_tasks, corr_model, corr_cal, feature_keys)

        # Extract disagreement examples
        train_disagree = extract_disagreement_examples(train_tasks)
        cal_disagree = extract_disagreement_examples(cal_tasks)
        eval_disagree = extract_disagreement_examples(eval_tasks)

        n_rescue_train = sum(1 for ex in train_disagree if ex["is_rescue"])
        n_break_train = sum(1 for ex in train_disagree if ex["is_break"])
        n_rescue_cal = sum(1 for ex in cal_disagree if ex["is_rescue"])
        n_break_cal = sum(1 for ex in cal_disagree if ex["is_break"])
        print(f"  Disagreement examples: {len(train_disagree)} train, {len(cal_disagree)} cal")
        print(f"  Train: {n_rescue_train} rescues, {n_break_train} breaks")
        print(f"  Cal:   {n_rescue_cal} rescues, {n_break_cal} breaks")

        # Train rescue and break models
        rescue_model = train_rescue_model(train_disagree, DISAGREEMENT_FEATURE_KEYS)
        break_model = train_break_model(train_disagree, DISAGREEMENT_FEATURE_KEYS)

        rescue_cal = None
        break_cal = None
        if rescue_model and n_rescue_cal >= 3:
            rescue_cal = calibrate_model(rescue_model, cal_disagree,
                                         DISAGREEMENT_FEATURE_KEYS, "is_rescue")
        if break_model and n_break_cal >= 3:
            break_cal = calibrate_model(break_model, cal_disagree,
                                        DISAGREEMENT_FEATURE_KEYS, "is_break")

        # AUROC for rescue/break models
        if rescue_model and n_rescue_cal > 0:
            eval_rescue_y = [1 if ex["is_rescue"] else 0 for ex in eval_disagree]
            if len(set(eval_rescue_y)) >= 2:
                eval_X = [build_feature_vector_from_dict(ex["features"], DISAGREEMENT_FEATURE_KEYS)
                          for ex in eval_disagree]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_r = roc_auc_score(eval_rescue_y, rescue_model.predict_proba(eval_X)[:, 1])
                print(f"  Rescue AUROC: {auroc_r:.4f}")

        if break_model and n_break_cal > 0:
            eval_break_y = [1 if ex["is_break"] else 0 for ex in eval_disagree]
            if len(set(eval_break_y)) >= 2:
                eval_X = [build_feature_vector_from_dict(ex["features"], DISAGREEMENT_FEATURE_KEYS)
                          for ex in eval_disagree]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auroc_b = roc_auc_score(eval_break_y, break_model.predict_proba(eval_X)[:, 1])
                print(f"  Break AUROC:  {auroc_b:.4f}")

        # Evaluate
        results = evaluate_all(eval_tasks, rescue_model, rescue_cal,
                               break_model, break_cal, DISAGREEMENT_FEATURE_KEYS)
        summary = summarize(results, eval_tasks)

        print(f"{'System':<25} {'Util':>7} {'Acc':>7} {'Force':>6} {'Resc':>5} {'Brk':>5}")
        print("-" * 65)
        for name in ["base", "majority_vote", "max_confidence", "maxcal",
                     "r10_lam3", "r10_lam5", "r10_lam10",
                     "r10_thresh_r50_b10", "r10_thresh_r30_b05",
                     "r10_verify_gate", "r10_majority_gate",
                     "r10_combined_gate", "r10_lowconf_gate"]:
            s = summary[name]
            print(f"{name:<25} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['n_force']:>6} {s['rescues']:>5} {s['breaks']:>5}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  R10 AGGREGATE: Selective Policy Improvement over MaxCal")
    print(f"{'='*100}")
    print(f"{'System':<25} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'TotF':>6} {'TotR':>6} {'TotB':>6}")
    print("-" * 100)

    names = ["base", "majority_vote", "max_confidence", "maxcal",
             "r10_lam3", "r10_lam5", "r10_lam10",
             "r10_thresh_r50_b10", "r10_thresh_r30_b05",
             "r10_verify_gate", "r10_majority_gate",
             "r10_combined_gate", "r10_lowconf_gate"]
    agg = {}
    for name in names:
        utils = [all_results[s][name]["mean_utility"] for s in [42, 123, 7, 99, 2024]]
        tot_f = sum(all_results[s][name]["n_force"] for s in [42, 123, 7, 99, 2024])
        tot_r = sum(all_results[s][name]["rescues"] for s in [42, 123, 7, 99, 2024])
        tot_b = sum(all_results[s][name]["breaks"] for s in [42, 123, 7, 99, 2024])
        agg[name] = {"mean": float(np.mean(utils)), "std": float(np.std(utils)),
                     "min": float(min(utils)), "max": float(max(utils)),
                     "tot_f": tot_f, "tot_r": tot_r, "tot_b": tot_b}
        print(f"{name:<25} {np.mean(utils):>7.1f} {np.std(utils):>7.1f} "
              f"{min(utils):>7.1f} {max(utils):>7.1f} "
              f"{tot_f:>6} {tot_r:>6} {tot_b:>6}")

    # Key comparison: R10 vs MaxCal
    maxcal_mean = agg["maxcal"]["mean"]
    print(f"\n  MaxCal (base policy): {maxcal_mean:.1f}")
    for name in ["r10_lam3", "r10_lam5", "r10_lam10",
                 "r10_thresh_r50_b10", "r10_thresh_r30_b05",
                 "r10_verify_gate", "r10_majority_gate",
                 "r10_combined_gate", "r10_lowconf_gate"]:
        diff = agg[name]["mean"] - maxcal_mean
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {name}: {agg[name]['mean']:.1f} ({diff:+.1f}) {verdict} "
              f"[F={agg[name]['tot_f']}, R={agg[name]['tot_r']}, B={agg[name]['tot_b']}]")

    # Safety analysis
    print(f"\n  Safety analysis (rule of three for zero breaks):")
    for name in ["r10_lam3", "r10_lam5", "r10_lam10",
                 "r10_thresh_r50_b10", "r10_thresh_r30_b05",
                 "r10_verify_gate", "r10_majority_gate",
                 "r10_combined_gate", "r10_lowconf_gate"]:
        n_int = agg[name]["tot_f"]
        n_break = agg[name]["tot_b"]
        if n_break == 0 and n_int > 0:
            upper = 3.0 / n_int
            print(f"    {name}: 0 breaks in {n_int} interventions → "
                  f"95% upper bound ≈ {upper:.1%}")
        elif n_int > 0:
            print(f"    {name}: {n_break} breaks in {n_int} interventions → "
                  f"break rate = {n_break/n_int:.1%}")

    output = {"aggregate": agg, "per_seed": all_results,
              "n_disagreement_features": len(DISAGREEMENT_FEATURE_KEYS)}
    output_path = R10_DIR / "r10_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
