#!/usr/bin/env python3
"""DAPH-X R9: Multi-round verification + pairwise comparisons + answer embeddings.

New signals beyond R8:
  1. Multi-round verification (3 rounds, averaged → more robust)
  2. Pairwise comparisons (model judges A vs B for all C(6,2)=15 pairs)
  3. Answer-level semantic features (TF-IDF + length/structure)
  4. Pairwise-derived ranking score (Copeland/Bradley-Terry)

Systems compared:
  1. Base
  2. Majority vote
  3. Max confidence
  4. Max calibrated P(correct) [enriched + verification]
  5. Max multi-round verification
  6. Max pairwise score (Copeland)
  7. MaxCal + verification (R8 best simple)
  8. Gate A + multi-round verification
  9. DAPH-X R9: multi-signal authority (Cal + Verify + Pairwise)

Usage:
    python scripts/run_r9_evaluation.py \\
        --corpus experiments/daph_x/cross_verification/cv_corpus_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

R9_DIR = REPO_ROOT / "experiments/daph_x/r9"

# Import from R7/R8
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, build_feature_vector,
    flatten_candidates, split_tasks, load_corpus,
)
from run_r8_evaluation import (
    add_verification_features,
    train_correctness_with_verification,
    calibrate_correctness,
    predict_correctness_v,
)


def add_multiround_verification_features(tasks: list[dict]) -> list[dict]:
    """Extract multi-round verification features from v2 corpus."""
    for task in tasks:
        for cand in task["candidates"]:
            v = cand.get("verification", {})
            cand["verification_score"] = v.get("verification_score", 0.5)
            cand["verification_std"] = v.get("verification_std", 0.0)
            cand["verification_consistent"] = v.get("verification_consistent", 0)
            # Round-level scores
            rounds = v.get("verification_rounds", [])
            cand["verification_round_scores"] = [r.get("score", 0.5) for r in rounds]
    return tasks


def compute_pairwise_scores(task: dict) -> dict[int, float]:
    """Compute Copeland score for each candidate from pairwise comparisons.

    Copeland score = wins - losses.
    A candidate "wins" a pairwise comparison if the model prefers it.
    """
    cands = task["candidates"]
    n = len(cands)
    scores = {i: 0.0 for i in range(n)}
    conf_weight = {}

    pairs = task.get("pairwise_comparisons", [])
    for p in pairs:
        i, j = p["i"], p["j"]
        pref = p["preference"]  # 1=i wins, -1=j wins, 0=tie
        conf = p.get("confidence", 0.5)

        if pref == 1:
            scores[i] += conf
            scores[j] -= conf
        elif pref == -1:
            scores[j] += conf
            scores[i] -= conf
        # tie: no change

    return scores


def compute_bradley_terry(pairwise_data: list[dict], n_candidates: int,
                          n_iter: int = 50) -> np.ndarray:
    """Simple Bradley-Terry estimation via iterative algorithm.

    Returns log-strength parameters for each candidate (relative).
    """
    # Aggregate wins per candidate across all tasks
    # For BT we need global strengths, but since candidates differ per task,
    # we compute per-task BT scores instead.
    # This function is kept for completeness but we use Copeland per-task.
    return np.zeros(n_candidates)


def add_pairwise_features(tasks: list[dict]) -> list[dict]:
    """Add pairwise-derived features to each candidate."""
    for task in tasks:
        copeland = compute_pairwise_scores(task)
        n = len(task["candidates"])
        max_score = max(abs(v) for v in copeland.values()) if copeland else 1
        max_score = max(max_score, 1.0)

        for i, cand in enumerate(task["candidates"]):
            cand["pairwise_copeland"] = copeland.get(i, 0.0)
            cand["pairwise_normalized"] = copeland.get(i, 0.0) / max_score
            # Win rate
            pairs = task.get("pairwise_comparisons", [])
            wins = sum(1 for p in pairs if p["i"] == i and p["preference"] == 1)
            wins += sum(1 for p in pairs if p["j"] == i and p["preference"] == -1)
            total = sum(1 for p in pairs if p["i"] == i or p["j"] == i)
            cand["pairwise_winrate"] = wins / total if total > 0 else 0.5

    return tasks


def add_answer_semantic_features(tasks: list[dict]) -> list[dict]:
    """Add answer-level semantic features (TF-IDF similarity, length, structure)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    for task in tasks:
        answers = [c["answer"] for c in task["candidates"]]
        n = len(answers)

        # TF-IDF similarity matrix
        try:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            tfidf = vec.fit_transform(answers)
            sim_matrix = cosine_similarity(tfidf)
        except ValueError:
            sim_matrix = np.ones((n, n))

        for i, cand in enumerate(task["candidates"]):
            ans = cand["answer"]
            # Answer length features
            cand["answer_length"] = len(ans)
            cand["answer_word_count"] = len(ans.split())
            cand["answer_has_digit"] = 1.0 if any(ch.isdigit() for ch in ans) else 0.0
            cand["answer_has_decimal"] = 1.0 if "." in ans else 0.0
            cand["answer_has_negative"] = 1.0 if "-" in ans else 0.0
            cand["answer_has_fraction"] = 1.0 if "/" in ans else 0.0
            cand["answer_is_numeric"] = 1.0 if ans.replace("-", "").replace(".", "").replace("/", "").isdigit() else 0.0

            # Similarity to other answers
            sims = [sim_matrix[i, j] for j in range(n) if j != i]
            cand["answer_avg_similarity"] = float(np.mean(sims)) if sims else 0.0
            cand["answer_max_similarity"] = float(max(sims)) if sims else 0.0
            cand["answer_min_similarity"] = float(min(sims)) if sims else 0.0

            # Agreement count (how many others are identical)
            identical = sum(1 for j in range(n) if j != i and answers[j] == ans)
            cand["answer_identical_count"] = identical

    return tasks


def train_correctness_r9(train_records, feature_keys):
    """Train P(correct) with enriched + verification + pairwise features."""
    X, y = [], []
    for r in train_records:
        feats = build_feature_vector(r["enriched_features"], feature_keys)
        extra = np.array([
            r.get("verification_score", 0.5),
            r.get("verification_std", 0.0),
            r.get("verification_consistent", 0),
            r.get("pairwise_copeland", 0.0),
            r.get("pairwise_normalized", 0.0),
            r.get("pairwise_winrate", 0.5),
            r.get("answer_length", 0),
            r.get("answer_word_count", 0),
            r.get("answer_avg_similarity", 0.0),
            r.get("answer_identical_count", 0),
        ])
        X.append(np.concatenate([feats, extra]))
        y.append(1 if r["is_correct"] else 0)
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model


def calibrate_r9(model, cal_records, feature_keys):
    raw_p, true_l = [], []
    for r in cal_records:
        feats = build_feature_vector(r["enriched_features"], feature_keys)
        extra = np.array([
            r.get("verification_score", 0.5),
            r.get("verification_std", 0.0),
            r.get("verification_consistent", 0),
            r.get("pairwise_copeland", 0.0),
            r.get("pairwise_normalized", 0.0),
            r.get("pairwise_winrate", 0.5),
            r.get("answer_length", 0),
            r.get("answer_word_count", 0),
            r.get("answer_avg_similarity", 0.0),
            r.get("answer_identical_count", 0),
        ])
        x = np.concatenate([feats, extra])
        raw_p.append(model.predict_proba([x])[0, 1])
        true_l.append(1 if r["is_correct"] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def predict_correctness_r9(model, cal, features, extra_dict, fk):
    feats = build_feature_vector(features, fk)
    extra = np.array([
        extra_dict.get("verification_score", 0.5),
        extra_dict.get("verification_std", 0.0),
        extra_dict.get("verification_consistent", 0),
        extra_dict.get("pairwise_copeland", 0.0),
        extra_dict.get("pairwise_normalized", 0.0),
        extra_dict.get("pairwise_winrate", 0.5),
        extra_dict.get("answer_length", 0),
        extra_dict.get("answer_word_count", 0),
        extra_dict.get("answer_avg_similarity", 0.0),
        extra_dict.get("answer_identical_count", 0),
    ])
    x = np.concatenate([feats, extra])
    raw = model.predict_proba([x])[0, 1]
    return float(cal.predict([raw])[0])


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

def sys_max_verification(task):
    pick = max(task["candidates"], key=lambda c: c.get("verification_score", 0.5))
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": True, "pick_id": pick["candidate_id"]}

def sys_max_pairwise(task):
    """Pick candidate with highest Copeland pairwise score."""
    pick = max(task["candidates"], key=lambda c: c.get("pairwise_copeland", 0.0))
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": True, "pick_id": pick["candidate_id"]}

def sys_max_calibrated_r9(task, corr_model, corr_cal, fk):
    best, best_p = None, -1
    for c in task["candidates"]:
        p = predict_correctness_r9(corr_model, corr_cal, c["enriched_features"], c, fk)
        if p > best_p:
            best_p = p
            best = c
    return {"correct": best["is_correct"], "utility": 100.0 if best["is_correct"] else 0.0,
            "pick_id": best["candidate_id"]}

def sys_gate_a_r9(task, corr_model, corr_cal, fk, margin=0.10):
    """Gate A with R9 features (enriched + verification + pairwise)."""
    base = task["candidates"][0]
    p_base = predict_correctness_r9(corr_model, corr_cal, base["enriched_features"], base, fk)

    best, best_p = base, p_base
    for c in task["candidates"][1:]:
        p = predict_correctness_r9(corr_model, corr_cal, c["enriched_features"], c, fk)
        if p > best_p:
            best_p = p
            best = c

    would_force = (best["candidate_id"] != base["candidate_id"]) and (best_p - p_base > margin)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force, "pick_id": pick["candidate_id"]}

def sys_daphx_r9(task, corr_model, corr_cal, fk, margin=0.10, v_thresh=0.6, v_low=0.3):
    """DAPH-X R9: Multi-signal authority (Cal + Verify + Pairwise).

    Decision logic:
    1. Compute P(correct) for all candidates (with all features)
    2. Find MaxCal pick
    3. If MaxCal verification is high AND pairwise score is high: KEEP (trust)
    4. If MaxCal verification is low OR pairwise score is low: consider override
    5. Override to candidate with best combined score
    6. Only override if P(correct) margin is sufficient
    """
    cands = task["candidates"]
    for c in cands:
        c["p_correct_r9"] = predict_correctness_r9(
            corr_model, corr_cal, c["enriched_features"], c, fk)

    maxcal_pick = max(cands, key=lambda c: c["p_correct_r9"])
    v_maxcal = maxcal_pick.get("verification_score", 0.5)
    pw_maxcal = maxcal_pick.get("pairwise_normalized", 0.0)
    p_maxcal = maxcal_pick["p_correct_r9"]

    # Trust MaxCal if both verification AND pairwise agree
    if v_maxcal >= v_thresh and pw_maxcal >= 0.0:
        return {"correct": maxcal_pick["is_correct"],
                "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # MaxCal signal is weak — look for better
    # Combined score: P(correct) * verification * (1 + pairwise)
    def combined_score(c):
        v = c.get("verification_score", 0.5)
        pw = c.get("pairwise_normalized", 0.0)
        p = c["p_correct_r9"]
        return p * (0.5 + 0.3 * v + 0.2 * (pw + 1) / 2)

    best_combined = max(cands, key=combined_score)

    if best_combined["candidate_id"] != maxcal_pick["candidate_id"]:
        if best_combined["p_correct_r9"] > p_maxcal + margin:
            return {"correct": best_combined["is_correct"],
                    "utility": 100.0 if best_combined["is_correct"] else 0.0,
                    "would_force": True, "pick_id": best_combined["candidate_id"]}

    # No good alternative — keep MaxCal
    return {"correct": maxcal_pick["is_correct"],
            "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

def sys_pairwise_consensus(task, corr_model, corr_cal, fk):
    """Use pairwise comparisons to build consensus, then verify with P(correct).

    1. Find candidate with highest pairwise win rate
    2. If that candidate also has high P(correct), pick it
    3. Otherwise fall back to MaxCal
    """
    cands = task["candidates"]
    for c in cands:
        c["p_correct_r9"] = predict_correctness_r9(
            corr_model, corr_cal, c["enriched_features"], c, fk)

    pw_pick = max(cands, key=lambda c: c.get("pairwise_winrate", 0.5))
    maxcal_pick = max(cands, key=lambda c: c["p_correct_r9"])

    # If pairwise winner has competitive P(correct), use it
    if pw_pick["p_correct_r9"] > maxcal_pick["p_correct_r9"] - 0.05:
        would_force = pw_pick["candidate_id"] != task["candidates"][0]["candidate_id"]
        return {"correct": pw_pick["is_correct"],
                "utility": 100.0 if pw_pick["is_correct"] else 0.0,
                "would_force": would_force, "pick_id": pw_pick["candidate_id"]}

    would_force = maxcal_pick["candidate_id"] != task["candidates"][0]["candidate_id"]
    return {"correct": maxcal_pick["is_correct"],
            "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
            "would_force": would_force, "pick_id": maxcal_pick["candidate_id"]}


def evaluate_all(eval_tasks, corr_model, corr_cal, fk):
    results = {name: [] for name in [
        "base", "majority_vote", "max_confidence", "max_verification",
        "max_pairwise", "max_calibrated_r9", "gate_a_r9",
        "pairwise_consensus", "daphx_r9"]}

    for task in eval_tasks:
        results["base"].append(sys_base(task))
        results["majority_vote"].append(sys_majority_vote(task))
        results["max_confidence"].append(sys_max_confidence(task))
        results["max_verification"].append(sys_max_verification(task))
        results["max_pairwise"].append(sys_max_pairwise(task))
        results["max_calibrated_r9"].append(sys_max_calibrated_r9(task, corr_model, corr_cal, fk))
        results["gate_a_r9"].append(sys_gate_a_r9(task, corr_model, corr_cal, fk))
        results["pairwise_consensus"].append(sys_pairwise_consensus(task, corr_model, corr_cal, fk))
        results["daphx_r9"].append(sys_daphx_r9(task, corr_model, corr_cal, fk))

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
            "mean_utility": float(np.mean(utils)),
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
    parser.add_argument("--corpus", default=str(R9_DIR / "r9_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R9_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    # Add multi-round verification features
    tasks = add_multiround_verification_features(tasks)
    # Add pairwise features
    tasks = add_pairwise_features(tasks)
    # Add answer semantic features
    tasks = add_answer_semantic_features(tasks)
    # Enrich with contrastive features (from R7)
    print("Computing enriched features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)
    n_extra = 10  # verification + pairwise + answer features
    print(f"Features: {len(feature_keys)} enriched + {n_extra} extra = {len(feature_keys)+n_extra}")

    # Check signal quality
    v_correct = [r.get("verification_score", 0.5) for r in all_records if r["is_correct"]]
    v_wrong = [r.get("verification_score", 0.5) for r in all_records if not r["is_correct"]]
    print(f"Multi-round verification: correct={np.mean(v_correct):.3f}, wrong={np.mean(v_wrong):.3f}")

    pw_correct = [r.get("pairwise_winrate", 0.5) for r in all_records if r["is_correct"]]
    pw_wrong = [r.get("pairwise_winrate", 0.5) for r in all_records if not r["is_correct"]]
    print(f"Pairwise win rate: correct={np.mean(pw_correct):.3f}, wrong={np.mean(pw_wrong):.3f}")

    c_correct = [r["self_confidence"] for r in all_records if r["is_correct"]]
    c_wrong = [r["self_confidence"] for r in all_records if not r["is_correct"]]
    print(f"Self-confidence: correct={np.mean(c_correct):.1f}, wrong={np.mean(c_wrong):.1f}")

    # Correlations
    v_scores = [r.get("verification_score", 0.5) for r in all_records]
    c_scores = [r["self_confidence"] / 100.0 for r in all_records]
    pw_scores = [r.get("pairwise_winrate", 0.5) for r in all_records]
    corr_vc = np.corrcoef(v_scores, c_scores)[0, 1]
    corr_vp = np.corrcoef(v_scores, pw_scores)[0, 1]
    corr_cp = np.corrcoef(c_scores, pw_scores)[0, 1]
    print(f"Correlation(verification, confidence): {corr_vc:.3f}")
    print(f"Correlation(verification, pairwise):   {corr_vp:.3f}")
    print(f"Correlation(confidence, pairwise):     {corr_cp:.3f}")

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")

        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, cal_records, feature_keys)

        # AUROC
        eval_X, eval_y = [], []
        for task in eval_tasks:
            for c in task["candidates"]:
                feats = build_feature_vector(c["enriched_features"], feature_keys)
                extra = np.array([
                    c.get("verification_score", 0.5),
                    c.get("verification_std", 0.0),
                    c.get("verification_consistent", 0),
                    c.get("pairwise_copeland", 0.0),
                    c.get("pairwise_normalized", 0.0),
                    c.get("pairwise_winrate", 0.5),
                    c.get("answer_length", 0),
                    c.get("answer_word_count", 0),
                    c.get("answer_avg_similarity", 0.0),
                    c.get("answer_identical_count", 0),
                ])
                eval_X.append(np.concatenate([feats, extra]))
                eval_y.append(1 if c["is_correct"] else 0)
        if len(set(eval_y)) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = roc_auc_score(eval_y, corr_model.predict_proba(eval_X)[:, 1])
            print(f"  Correctness AUROC (R9 all features): {auroc:.4f}")

        results = evaluate_all(eval_tasks, corr_model, corr_cal, feature_keys)
        summary = summarize(results, eval_tasks)

        print(f"{'System':<25} {'Util':>7} {'Acc':>7} {'Imp':>5} {'Reg':>5} {'Force':>6} {'Resc':>5} {'Brk':>5}")
        print("-" * 80)
        for name in ["base", "majority_vote", "max_confidence", "max_verification",
                     "max_pairwise", "max_calibrated_r9", "gate_a_r9",
                     "pairwise_consensus", "daphx_r9"]:
            s = summary[name]
            print(f"{name:<25} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['improvements']:>5} {s['regressions']:>5} "
                  f"{s['n_force']:>6} {s['rescues']:>5} {s['breaks']:>5}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*110}")
    print(f"  AGGREGATE ACROSS 5 SEEDS (R9: enriched + multi-round verify + pairwise + answer)")
    print(f"{'='*110}")
    print(f"{'System':<25} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'TotF':>6} {'TotR':>6} {'TotB':>6}")
    print("-" * 110)

    names = ["base", "majority_vote", "max_confidence", "max_verification",
             "max_pairwise", "max_calibrated_r9", "gate_a_r9",
             "pairwise_consensus", "daphx_r9"]
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

    best_simple = max(agg["majority_vote"]["mean"], agg["max_confidence"]["mean"],
                      agg["max_calibrated_r9"]["mean"], agg["max_verification"]["mean"],
                      agg["max_pairwise"]["mean"])
    print(f"\n  Best simple baseline: {best_simple:.1f}")
    for name in ["gate_a_r9", "pairwise_consensus", "daphx_r9"]:
        diff = agg[name]["mean"] - best_simple
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {name}: {agg[name]['mean']:.1f} ({diff:+.1f}) {verdict} "
              f"[F={agg[name]['tot_f']}, R={agg[name]['tot_r']}, B={agg[name]['tot_b']}]")

    output = {"aggregate": agg, "per_seed": all_results,
              "verification_correlation": corr_vc,
              "pairwise_verification_correlation": corr_vp,
              "confidence_pairwise_correlation": corr_cp,
              "n_features": len(feature_keys) + n_extra}
    output_path = R9_DIR / "r9_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
