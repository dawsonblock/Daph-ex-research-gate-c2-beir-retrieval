#!/usr/bin/env python3
"""DAPH-X R7: Enriched contrastive features + targeted MaxCal-error model.

Adds the user's prescribed contrastive features to the existing corpus
WITHOUT re-collection. All features computed from existing reasoning
traces and answers.

New feature groups:
  1. TF-IDF semantic similarity between reasoning traces
  2. Answer clustering (frequency, cluster size, isolation)
  3. Numerical consistency (do intermediate numbers agree?)
  4. Contradiction detection (does one answer appear in another's reasoning?)
  5. Reasoning structure (step count, pattern matching)
  6. Cross-candidate margin features

Then trains a TARGETED model: instead of generic P(correct), train
specifically on cases where MaxCal picks wrong. This is the model
that could beat MaxCal.

Usage:
    python scripts/run_r7_evaluation.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import check_answer

R7_DIR = REPO_ROOT / "experiments/daph_x/r7"


def load_corpus(path):
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


def flatten_candidates(tasks):
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


# ─── Enriched feature extraction ───

def extract_all_numbers(text: str) -> list[float]:
    """Extract all numbers from text, including fractions."""
    # Match integers, decimals, and fractions
    numbers = []
    for match in re.finditer(r"[-+]?\d+\.?\d*(?:/\d+\.?\d*)?", text):
        try:
            s = match.group()
            if "/" in s:
                parts = s.split("/")
                numbers.append(float(parts[0]) / float(parts[1]))
            else:
                numbers.append(float(s))
        except (ValueError, ZeroDivisionError):
            pass
    return numbers


def extract_reasoning_steps(response: str) -> list[str]:
    """Extract individual reasoning steps from response."""
    # Split by numbered steps, bullet points, or double newlines
    steps = re.split(r"\n\s*(?:\d+[\.\)]|•|\-|\*\*)\s*", response)
    steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]
    return steps


def compute_tfidf_similarity(responses: list[str]) -> np.ndarray:
    """Compute TF-IDF cosine similarity matrix between responses."""
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(responses)
        sim_matrix = cosine_similarity(tfidf_matrix)
        return sim_matrix
    except ValueError:
        n = len(responses)
        return np.ones((n, n))


def compute_answer_clusters(candidates: list[dict]) -> dict:
    """Cluster candidates by answer similarity."""
    answers = [c["answer"].strip().lower() for c in candidates]
    counter = Counter(answers)

    clusters = {}
    for i, c in enumerate(candidates):
        ans = answers[i]
        cluster_size = counter[ans]
        clusters[i] = {
            "answer": ans,
            "cluster_size": cluster_size,
            "is_majority": ans == counter.most_common(1)[0][0],
            "isolation": 1.0 / cluster_size,  # 1 if unique, 1/n if all same
            "frequency": cluster_size / len(candidates),
        }
    return clusters


def compute_numerical_consistency(candidates: list[dict]) -> dict:
    """Check if intermediate numbers in reasoning are consistent across candidates."""
    all_numbers = {}
    for i, c in enumerate(candidates):
        nums = extract_all_numbers(c["response"])
        all_numbers[i] = nums

    # For each candidate, how many of its numbers appear in other candidates?
    consistency = {}
    for i in range(len(candidates)):
        my_nums = set(all_numbers[i])
        if not my_nums:
            consistency[i] = {"num_agreement": 0.0, "num_unique": 0, "num_shared": 0}
            continue

        shared_count = 0
        for j in range(len(candidates)):
            if i == j:
                continue
            other_nums = set(all_numbers[j])
            shared_count += len(my_nums & other_nums)

        avg_shared = shared_count / max(len(candidates) - 1, 1)
        consistency[i] = {
            "num_agreement": avg_shared / max(len(my_nums), 1),
            "num_unique": len(my_nums),
            "num_shared": avg_shared,
        }
    return consistency


def compute_contradiction_features(candidates: list[dict]) -> dict:
    """Detect if one candidate's answer appears in another's reasoning as a different step."""
    features = {}
    for i, c in enumerate(candidates):
        my_answer = c["answer"].strip().lower()
        appears_in_others = 0
        others_appear_in_mine = 0

        for j, other in enumerate(candidates):
            if i == j:
                continue
            other_answer = other["answer"].strip().lower()
            # Does my answer appear in other's reasoning?
            if my_answer and my_answer != other_answer:
                if my_answer in other["response"].lower():
                    appears_in_others += 1
            # Does other's answer appear in my reasoning?
            if other_answer and my_answer != other_answer:
                if other_answer in c["response"].lower():
                    others_appear_in_mine += 1

        n_others = max(len(candidates) - 1, 1)
        features[i] = {
            "answer_in_others_reasoning": appears_in_others / n_others,
            "others_answer_in_my_reasoning": others_appear_in_mine / n_others,
            "contradiction_score": (appears_in_others + others_appear_in_mine) / (2 * n_others),
        }
    return features


def compute_enriched_features(task: dict) -> list[dict]:
    """Compute all enriched features for a task's candidates."""
    candidates = task["candidates"]
    n = len(candidates)
    responses = [c["response"] for c in candidates]

    # TF-IDF similarity
    sim_matrix = compute_tfidf_similarity(responses)

    # Answer clusters
    clusters = compute_answer_clusters(candidates)

    # Numerical consistency
    num_consistency = compute_numerical_consistency(candidates)

    # Contradiction features
    contradictions = compute_contradiction_features(candidates)

    # Reasoning steps
    all_steps = [extract_reasoning_steps(c["response"]) for c in candidates]

    enriched = []
    for i, c in enumerate(candidates):
        # Original features
        feats = dict(c.get("features", {}))

        # TF-IDF similarity features
        sims_to_others = [sim_matrix[i, j] for j in range(n) if j != i]
        feats["tfidf_mean_sim"] = float(np.mean(sims_to_others)) if sims_to_others else 0.0
        feats["tfidf_max_sim"] = float(np.max(sims_to_others)) if sims_to_others else 0.0
        feats["tfidf_min_sim"] = float(np.min(sims_to_others)) if sims_to_others else 0.0
        feats["tfidf_std_sim"] = float(np.std(sims_to_others)) if sims_to_others else 0.0

        # Answer cluster features
        cl = clusters[i]
        feats["answer_cluster_size"] = cl["cluster_size"]
        feats["answer_is_majority"] = cl["is_majority"]
        feats["answer_isolation"] = cl["isolation"]
        feats["answer_frequency"] = cl["frequency"]

        # Numerical consistency
        nc = num_consistency[i]
        feats["num_agreement"] = nc["num_agreement"]
        feats["num_unique_count"] = nc["num_unique"]
        feats["num_shared"] = nc["num_shared"]

        # Contradiction features
        ct = contradictions[i]
        feats["answer_in_others"] = ct["answer_in_others_reasoning"]
        feats["others_in_mine"] = ct["others_answer_in_my_reasoning"]
        feats["contradiction_score"] = ct["contradiction_score"]

        # Reasoning structure
        steps = all_steps[i]
        feats["n_reasoning_steps"] = len(steps)
        feats["avg_step_length"] = float(np.mean([len(s) for s in steps])) if steps else 0.0
        feats["total_step_length"] = sum(len(s) for s in steps)

        # Cross-candidate margin features
        my_conf = c["self_confidence"]
        other_confs = [candidates[j]["self_confidence"] for j in range(n) if j != i]
        feats["conf_margin_vs_max"] = my_conf - max(other_confs) if other_confs else 0.0
        feats["conf_margin_vs_mean"] = my_conf - float(np.mean(other_confs)) if other_confs else 0.0
        feats["conf_rank"] = sum(1 for oc in other_confs if oc > my_conf)

        # Answer entropy (how uncertain is the answer distribution?)
        answers = [cand["answer"].strip().lower() for cand in candidates]
        counter = Counter(answers)
        probs = [count / n for count in counter.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        feats["answer_entropy"] = entropy
        feats["n_unique_answers"] = len(counter)

        # TF-IDF similarity to correct-looking candidates
        # (candidates with high self-confidence)
        high_conf_idx = [j for j in range(n) if j != i and candidates[j]["self_confidence"] > 60]
        if high_conf_idx:
            feats["tfidf_to_high_conf"] = float(np.mean([sim_matrix[i, j] for j in high_conf_idx]))
        else:
            feats["tfidf_to_high_conf"] = 0.0

        # TF-IDF similarity to majority answer group
        majority_answer = counter.most_common(1)[0][0]
        majority_idx = [j for j in range(n) if answers[j] == majority_answer and j != i]
        if majority_idx:
            feats["tfidf_to_majority"] = float(np.mean([sim_matrix[i, j] for j in majority_idx]))
        else:
            feats["tfidf_to_majority"] = 0.0

        enriched.append(feats)

    return enriched


def enrich_corpus(tasks: list[dict]) -> list[dict]:
    """Add enriched features to all tasks in corpus."""
    enriched_tasks = []
    for task in tasks:
        enriched_feats = compute_enriched_features(task)
        new_candidates = []
        for i, c in enumerate(task["candidates"]):
            new_c = dict(c)
            new_c["enriched_features"] = enriched_feats[i]
            new_candidates.append(new_c)
        new_task = dict(task)
        new_task["candidates"] = new_candidates
        enriched_tasks.append(new_task)
    return enriched_tasks


def get_feature_keys(records, feat_key="enriched_features"):
    all_keys = set()
    for r in records:
        all_keys.update(r[feat_key].keys())
    return sorted(all_keys)


def build_feature_vector(features, feature_keys):
    return np.array([float(features.get(k, 0.0)) for k in feature_keys])


def build_pair_features(a, b, feature_keys):
    fa = build_feature_vector(a["enriched_features"], feature_keys)
    fb = build_feature_vector(b["enriched_features"], feature_keys)
    return np.concatenate([fa - fb, fa, fb, np.abs(fa - fb)])


# ─── Models ───

def train_correctness_model(train_records, feature_keys):
    X, y = [], []
    for r in train_records:
        X.append(build_feature_vector(r["enriched_features"], feature_keys))
        y.append(1 if r["is_correct"] else 0)
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model


def calibrate_model(model, cal_records, feature_keys):
    raw_p, true_l = [], []
    for r in cal_records:
        f = build_feature_vector(r["enriched_features"], feature_keys)
        raw_p.append(model.predict_proba([f])[0, 1])
        true_l.append(1 if r["is_correct"] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def predict_correctness(model, cal, features, fk):
    f = build_feature_vector(features, fk)
    raw = model.predict_proba([f])[0, 1]
    return float(cal.predict([raw])[0])


def train_pairwise_model(train_tasks, feature_keys):
    X, y = [], []
    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j: continue
                if cands[i]["is_correct"] == cands[j]["is_correct"]: continue
                X.append(build_pair_features(cands[i], cands[j], feature_keys))
                y.append(1 if cands[i]["is_correct"] else 0)
    if len(set(y)) < 2 or len(X) < 10:
        return None, 0
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model, len(X)


def train_maxcal_error_model(train_tasks, feature_keys, corr_model, corr_cal):
    """Train a model specifically on cases where MaxCal picks wrong.

    Target: P(MaxCal picks wrong AND this candidate is correct)
    This is the targeted model that could beat MaxCal.
    """
    X, y = [], []
    for task in train_tasks:
        cands = task["candidates"]
        # Find MaxCal pick
        maxcal_pick = max(cands, key=lambda c: predict_correctness(corr_model, corr_cal, c["enriched_features"], feature_keys))
        maxcal_correct = maxcal_pick["is_correct"]

        for c in cands:
            feats = build_feature_vector(c["enriched_features"], feature_keys)
            # Label: 1 if this candidate is correct AND MaxCal is wrong
            # (i.e., this candidate could rescue a MaxCal error)
            if not maxcal_correct and c["is_correct"]:
                y.append(1)
            else:
                y.append(0)
            X.append(feats)

    X, y = np.array(X), np.array(y)
    if len(set(y)) < 2 or sum(y) < 5:
        return None, 0
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(X, y)
    return model, sum(y)


# ─── Systems ───

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

def sys_max_calibrated(task, corr_model, corr_cal, fk):
    best, best_p = None, -1
    for c in task["candidates"]:
        p = predict_correctness(corr_model, corr_cal, c["enriched_features"], fk)
        if p > best_p:
            best_p = p
            best = c
    return {"correct": best["is_correct"], "utility": 100.0 if best["is_correct"] else 0.0,
            "pick_id": best["candidate_id"]}


def sys_gate_a(task, corr_model, corr_cal, fk, margin=0.10):
    """MaxCal + margin abstention."""
    base = task["candidates"][0]
    p_base = predict_correctness(corr_model, corr_cal, base["enriched_features"], fk)

    best, best_p = base, p_base
    for c in task["candidates"][1:]:
        p = predict_correctness(corr_model, corr_cal, c["enriched_features"], fk)
        if p > best_p:
            best_p = p
            best = c

    would_force = (best["candidate_id"] != base["candidate_id"]) and (best_p - p_base > margin)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force, "pick_id": pick["candidate_id"]}


def sys_targeted(task, corr_model, corr_cal, error_model, fk, margin=0.10, error_thresh=0.3):
    """Targeted model: MaxCal pick as default, override when error model flags.

    1. Find MaxCal pick
    2. If error model says MaxCal likely wrong (P(error) > error_thresh)
    3. Pick the candidate with highest P(correct) among non-MaxCal candidates
    """
    cands = task["candidates"]
    # MaxCal pick
    maxcal_pick = max(cands, key=lambda c: predict_correctness(corr_model, corr_cal, c["enriched_features"], fk))
    p_maxcal = predict_correctness(corr_model, corr_cal, maxcal_pick["enriched_features"], fk)

    if error_model is None:
        return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # Check if MaxCal likely wrong
    maxcal_feats = build_feature_vector(maxcal_pick["enriched_features"], fk)
    p_error = error_model.predict_proba([maxcal_feats])[0, 1]

    if p_error < error_thresh:
        # MaxCal likely correct, keep it
        return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # MaxCal likely wrong — find best alternative
    best_alt = None
    best_p = -1
    for c in cands:
        if c["candidate_id"] == maxcal_pick["candidate_id"]:
            continue
        p = predict_correctness(corr_model, corr_cal, c["enriched_features"], fk)
        if p > best_p:
            best_p = p
            best_alt = c

    # Only override if alternative is clearly better
    if best_alt and best_p > p_maxcal + margin:
        return {"correct": best_alt["is_correct"], "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"]}
    else:
        return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}


def sys_targeted_plus_pairwise(task, corr_model, corr_cal, error_model, pref_model, pref_cal, fk,
                                margin=0.10, error_thresh=0.3, pairwise_thresh=0.6):
    """Targeted model + pairwise confirmation.

    Like targeted, but also require pairwise model to confirm the override.
    """
    cands = task["candidates"]
    maxcal_pick = max(cands, key=lambda c: predict_correctness(corr_model, corr_cal, c["enriched_features"], fk))
    p_maxcal = predict_correctness(corr_model, corr_cal, maxcal_pick["enriched_features"], fk)

    if error_model is None or pref_model is None:
        return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    maxcal_feats = build_feature_vector(maxcal_pick["enriched_features"], fk)
    p_error = error_model.predict_proba([maxcal_feats])[0, 1]

    if p_error < error_thresh:
        return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # Find best alternative
    best_alt = None
    best_p = -1
    for c in cands:
        if c["candidate_id"] == maxcal_pick["candidate_id"]:
            continue
        p = predict_correctness(corr_model, corr_cal, c["enriched_features"], fk)
        if p > best_p:
            best_p = p
            best_alt = c

    if best_alt and best_p > p_maxcal + margin:
        # Confirm with pairwise
        if pref_cal is not None:
            pf = build_pair_features(best_alt, maxcal_pick, fk)
            raw = pref_model.predict_proba([pf])[0, 1]
            p_pairwise = float(pref_cal.predict([raw])[0])
        else:
            p_pairwise = pref_model.predict_proba([build_pair_features(best_alt, maxcal_pick, fk)])[0, 1]

        if p_pairwise > pairwise_thresh:
            return {"correct": best_alt["is_correct"], "utility": 100.0 if best_alt["is_correct"] else 0.0,
                    "would_force": True, "pick_id": best_alt["candidate_id"]}

    return {"correct": maxcal_pick["is_correct"], "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal_pick["candidate_id"]}


def evaluate_all(eval_tasks, models):
    fk = models["feature_keys"]
    results = {name: [] for name in [
        "base", "majority_vote", "max_confidence", "max_calibrated",
        "gate_a", "targeted", "targeted+pw"]}

    for task in eval_tasks:
        results["base"].append(sys_base(task))
        results["majority_vote"].append(sys_majority_vote(task))
        results["max_confidence"].append(sys_max_confidence(task))
        results["max_calibrated"].append(sys_max_calibrated(
            task, models["corr_model"], models["corr_cal"], fk))
        results["gate_a"].append(sys_gate_a(
            task, models["corr_model"], models["corr_cal"], fk))
        results["targeted"].append(sys_targeted(
            task, models["corr_model"], models["corr_cal"],
            models["error_model"], fk))
        results["targeted+pw"].append(sys_targeted_plus_pairwise(
            task, models["corr_model"], models["corr_cal"],
            models["error_model"], models["pref_model"], models["pref_cal"], fk))

    return results


def summarize(results, eval_tasks):
    summary = {}
    base_utils = [100.0 if eval_tasks[i]["candidates"][0]["is_correct"] else 0.0
                  for i in range(len(eval_tasks))]

    for name, res_list in results.items():
        utils = [r["utility"] for r in res_list]
        successes = sum(1 for r in res_list if r["correct"])
        improvements = sum(1 for u, bu in zip(utils, base_utils) if u > bu + 0.5)
        regressions = sum(1 for u, bu in zip(utils, base_utils) if u < bu - 0.5)
        n_force = sum(1 for r in res_list if r.get("would_force", False))
        rescues = sum(1 for r, bu in zip(res_list, base_utils)
                      if r.get("would_force", False) and r["utility"] > bu + 0.5)
        breaks = sum(1 for r, bu in zip(res_list, base_utils)
                     if r.get("would_force", False) and r["utility"] < bu - 0.5)

        summary[name] = {
            "mean_utility": np.mean(utils),
            "success_rate": successes / len(res_list),
            "improvements": improvements,
            "regressions": regressions,
            "n_force": n_force,
            "rescues": rescues,
            "breaks": breaks,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R7_DIR / "r7_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R7_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    # Enrich with new features
    print("Computing enriched contrastive features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)
    print(f"Features: {len(feature_keys)} (enriched)")

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")

        # Train correctness model
        corr_model = train_correctness_model(train_records, feature_keys)
        corr_cal = calibrate_model(corr_model, cal_records, feature_keys)

        # Train pairwise model
        pref_model, n_pairs = train_pairwise_model(train_tasks, feature_keys)
        pref_cal = None
        if pref_model is not None:
            raw_p, true_l = [], []
            for task in cal_tasks:
                cands = task["candidates"]
                for i in range(len(cands)):
                    for j in range(len(cands)):
                        if i == j: continue
                        if cands[i]["is_correct"] == cands[j]["is_correct"]: continue
                        pf = build_pair_features(cands[i], cands[j], feature_keys)
                        raw_p.append(pref_model.predict_proba([pf])[0, 1])
                        true_l.append(1 if cands[i]["is_correct"] else 0)
            if len(set(true_l)) >= 2:
                pref_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                pref_cal.fit(np.array(raw_p), np.array(true_l))

        # Train targeted MaxCal-error model
        error_model, n_errors = train_maxcal_error_model(
            train_tasks, feature_keys, corr_model, corr_cal)
        print(f"  Pairwise pairs: {n_pairs}, MaxCal-error examples: {n_errors}")

        # AUROC for correctness model on eval
        eval_X, eval_y = [], []
        for task in eval_tasks:
            for c in task["candidates"]:
                eval_X.append(build_feature_vector(c["enriched_features"], feature_keys))
                eval_y.append(1 if c["is_correct"] else 0)
        if len(set(eval_y)) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = roc_auc_score(eval_y, corr_model.predict_proba(eval_X)[:, 1])
            print(f"  Correctness AUROC: {auroc:.4f}")

        models = {
            "corr_model": corr_model, "corr_cal": corr_cal,
            "pref_model": pref_model, "pref_cal": pref_cal,
            "error_model": error_model,
            "feature_keys": feature_keys,
        }

        results = evaluate_all(eval_tasks, models)
        summary = summarize(results, eval_tasks)

        print(f"{'System':<20} {'Util':>7} {'Acc':>7} {'Imp':>5} {'Reg':>5} {'Force':>6} {'Resc':>5} {'Brk':>5}")
        print("-" * 70)
        for name in ["base", "majority_vote", "max_confidence", "max_calibrated",
                     "gate_a", "targeted", "targeted+pw"]:
            s = summary[name]
            print(f"{name:<20} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['improvements']:>5} {s['regressions']:>5} "
                  f"{s['n_force']:>6} {s['rescues']:>5} {s['breaks']:>5}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*95}")
    print(f"  AGGREGATE ACROSS 5 SEEDS (enriched features)")
    print(f"{'='*95}")
    print(f"{'System':<20} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'TotF':>6} {'TotR':>6} {'TotB':>6}")
    print("-" * 95)

    names = ["base", "majority_vote", "max_confidence", "max_calibrated",
             "gate_a", "targeted", "targeted+pw"]
    agg = {}
    for name in names:
        utils = [all_results[s][name]["mean_utility"] for s in [42, 123, 7, 99, 2024]]
        tot_f = sum(all_results[s][name]["n_force"] for s in [42, 123, 7, 99, 2024])
        tot_r = sum(all_results[s][name]["rescues"] for s in [42, 123, 7, 99, 2024])
        tot_b = sum(all_results[s][name]["breaks"] for s in [42, 123, 7, 99, 2024])
        agg[name] = {"mean": np.mean(utils), "std": np.std(utils),
                     "min": min(utils), "max": max(utils),
                     "tot_f": tot_f, "tot_r": tot_r, "tot_b": tot_b}
        print(f"{name:<20} {np.mean(utils):>7.1f} {np.std(utils):>7.1f} "
              f"{min(utils):>7.1f} {max(utils):>7.1f} "
              f"{tot_f:>6} {tot_r:>6} {tot_b:>6}")

    best_simple = max(agg["majority_vote"]["mean"], agg["max_confidence"]["mean"],
                      agg["max_calibrated"]["mean"])
    print(f"\n  Best simple baseline: {best_simple:.1f}")
    for name in ["gate_a", "targeted", "targeted+pw"]:
        diff = agg[name]["mean"] - best_simple
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {name}: {agg[name]['mean']:.1f} ({diff:+.1f}) {verdict} "
              f"[F={agg[name]['tot_f']}, R={agg[name]['tot_r']}, B={agg[name]['tot_b']}]")

    output = {"aggregate": agg, "per_seed": all_results,
              "n_features": len(feature_keys)}
    output_path = R7_DIR / "r7_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
