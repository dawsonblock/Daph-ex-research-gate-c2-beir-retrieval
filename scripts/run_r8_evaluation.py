#!/usr/bin/env python3
"""DAPH-X R8: Cross-verification + enriched features.

Uses model-based cross-verification as a genuinely new signal:
  - Self-confidence: "How confident are you in YOUR answer?" (generation-time)
  - Verification: "Is THIS answer correct?" (evaluation-time, different prompt)

The verification signal is independent of self-confidence because:
  1. It evaluates a specific answer, not the model's own output
  2. It uses a different prompt structure (verification vs generation)
  3. It can catch confidently-wrong answers

Systems compared:
  1. Base
  2. Majority vote
  3. Max confidence
  4. Max calibrated P(correct) [enriched features]
  5. Max verification score
  6. MaxCal + verification (combine both signals)
  7. Gate A + verification (MaxCal+margin, but also check verification)
  8. DAPH-X: targeted override using verification disagreement

Usage:
    python scripts/run_r8_evaluation.py \\
        --corpus experiments/daph_x/cross_verification/cv_corpus.jsonl
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

R8_DIR = REPO_ROOT / "experiments/daph_x/r8"

# Import enrichment from R7
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, build_feature_vector,
    flatten_candidates, split_tasks, load_corpus,
    compute_enriched_features,
)


def add_verification_features(tasks: list[dict]) -> list[dict]:
    """Add verification score as a feature to each candidate."""
    for task in tasks:
        for cand in task["candidates"]:
            v = cand.get("verification", {})
            cand["verification_score"] = v.get("verification_score", 0.5)
    return tasks


def train_correctness_with_verification(train_records, feature_keys):
    """Train P(correct) using enriched features + verification score."""
    X, y = [], []
    for r in train_records:
        feats = build_feature_vector(r["enriched_features"], feature_keys)
        # Append verification score
        v_score = np.array([r.get("verification_score", 0.5)])
        X.append(np.concatenate([feats, v_score]))
        y.append(1 if r["is_correct"] else 0)
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model


def calibrate_correctness(model, cal_records, feature_keys):
    raw_p, true_l = [], []
    for r in cal_records:
        feats = build_feature_vector(r["enriched_features"], feature_keys)
        v_score = np.array([r.get("verification_score", 0.5)])
        x = np.concatenate([feats, v_score])
        raw_p.append(model.predict_proba([x])[0, 1])
        true_l.append(1 if r["is_correct"] else 0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(np.array(raw_p), np.array(true_l))
    return iso


def predict_correctness_v(model, cal, features, verification_score, fk):
    feats = build_feature_vector(features, fk)
    x = np.concatenate([feats, np.array([verification_score])])
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
    """Pick candidate with highest verification score."""
    pick = max(task["candidates"], key=lambda c: c.get("verification_score", 0.5))
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": True, "pick_id": pick["candidate_id"]}

def sys_max_calibrated_v(task, corr_model, corr_cal, fk):
    """Max calibrated P(correct) with verification."""
    best, best_p = None, -1
    for c in task["candidates"]:
        v = c.get("verification_score", 0.5)
        p = predict_correctness_v(corr_model, corr_cal, c["enriched_features"], v, fk)
        if p > best_p:
            best_p = p
            best = c
    return {"correct": best["is_correct"], "utility": 100.0 if best["is_correct"] else 0.0,
            "pick_id": best["candidate_id"]}

def sys_gate_a_v(task, corr_model, corr_cal, fk, margin=0.10):
    """Gate A with verification-enhanced P(correct)."""
    base = task["candidates"][0]
    v_base = base.get("verification_score", 0.5)
    p_base = predict_correctness_v(corr_model, corr_cal, base["enriched_features"], v_base, fk)

    best, best_p = base, p_base
    for c in task["candidates"][1:]:
        v = c.get("verification_score", 0.5)
        p = predict_correctness_v(corr_model, corr_cal, c["enriched_features"], v, fk)
        if p > best_p:
            best_p = p
            best = c

    would_force = (best["candidate_id"] != base["candidate_id"]) and (best_p - p_base > margin)
    pick = best if would_force else base
    return {"correct": pick["is_correct"], "utility": 100.0 if pick["is_correct"] else 0.0,
            "would_force": would_force, "pick_id": pick["candidate_id"]}

def sys_verification_override(task, corr_model, corr_cal, fk, margin=0.10, v_thresh=0.7):
    """MaxCal pick as default, override when verification strongly disagrees.

    1. Find MaxCal pick
    2. If MaxCal's verification score is low (< v_thresh)
    3. Find candidate with highest verification among high-P(correct) candidates
    4. Override if that candidate's verification is high AND P(correct) is competitive
    """
    cands = task["candidates"]
    # MaxCal pick
    maxcal_pick = max(cands, key=lambda c: predict_correctness_v(
        corr_model, corr_cal, c["enriched_features"],
        c.get("verification_score", 0.5), fk))
    p_maxcal = predict_correctness_v(
        corr_model, corr_cal, maxcal_pick["enriched_features"],
        maxcal_pick.get("verification_score", 0.5), fk)
    v_maxcal = maxcal_pick.get("verification_score", 0.5)

    # If MaxCal's verification is high, trust it
    if v_maxcal >= v_thresh:
        return {"correct": maxcal_pick["is_correct"],
                "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # MaxCal's verification is low — look for a better candidate
    best_alt = None
    best_score = -1
    for c in cands:
        if c["candidate_id"] == maxcal_pick["candidate_id"]:
            continue
        v = c.get("verification_score", 0.5)
        p = predict_correctness_v(corr_model, corr_cal, c["enriched_features"], v, fk)
        # Combined score: P(correct) weighted by verification
        combined = p * v + p * 0.5  # verification amplifies P(correct)
        if combined > best_score:
            best_score = combined
            best_alt = c

    if best_alt and best_score > p_maxcal * v_maxcal + p_maxcal * 0.5 + margin:
        return {"correct": best_alt["is_correct"],
                "utility": 100.0 if best_alt["is_correct"] else 0.0,
                "would_force": True, "pick_id": best_alt["candidate_id"]}
    else:
        return {"correct": maxcal_pick["is_correct"],
                "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

def sys_daphx_r8(task, corr_model, corr_cal, fk, margin=0.10, v_thresh=0.6, v_low=0.3):
    """DAPH-X R8: Multi-signal authority.

    Decision logic:
    1. Compute P(correct) for all candidates (with verification)
    2. Find MaxCal pick
    3. If MaxCal verification is high (>= v_thresh): KEEP (trust)
    4. If MaxCal verification is low (< v_low): consider override
    5. Override to highest P(correct) candidate with high verification
    6. Only override if P(correct) margin is sufficient
    """
    cands = task["candidates"]
    # Compute P(correct) for all
    for c in cands:
        v = c.get("verification_score", 0.5)
        c["p_correct_v"] = predict_correctness_v(
            corr_model, corr_cal, c["enriched_features"], v, fk)

    maxcal_pick = max(cands, key=lambda c: c["p_correct_v"])
    v_maxcal = maxcal_pick.get("verification_score", 0.5)
    p_maxcal = maxcal_pick["p_correct_v"]

    # Trust MaxCal if verification is high
    if v_maxcal >= v_thresh:
        return {"correct": maxcal_pick["is_correct"],
                "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
                "would_force": False, "pick_id": maxcal_pick["candidate_id"]}

    # MaxCal verification is low — look for better
    # Find candidates with high verification
    high_v_cands = [c for c in cands if c.get("verification_score", 0.5) >= v_thresh
                    and c["candidate_id"] != maxcal_pick["candidate_id"]]

    if high_v_cands:
        best_alt = max(high_v_cands, key=lambda c: c["p_correct_v"])
        if best_alt["p_correct_v"] > p_maxcal + margin:
            return {"correct": best_alt["is_correct"],
                    "utility": 100.0 if best_alt["is_correct"] else 0.0,
                    "would_force": True, "pick_id": best_alt["candidate_id"]}

    # No good alternative — keep MaxCal
    return {"correct": maxcal_pick["is_correct"],
            "utility": 100.0 if maxcal_pick["is_correct"] else 0.0,
            "would_force": False, "pick_id": maxcal_pick["candidate_id"]}


def evaluate_all(eval_tasks, corr_model, corr_cal, fk):
    results = {name: [] for name in [
        "base", "majority_vote", "max_confidence", "max_verification",
        "max_calibrated_v", "gate_a_v", "verification_override", "daphx_r8"]}

    for task in eval_tasks:
        results["base"].append(sys_base(task))
        results["majority_vote"].append(sys_majority_vote(task))
        results["max_confidence"].append(sys_max_confidence(task))
        results["max_verification"].append(sys_max_verification(task))
        results["max_calibrated_v"].append(sys_max_calibrated_v(task, corr_model, corr_cal, fk))
        results["gate_a_v"].append(sys_gate_a_v(task, corr_model, corr_cal, fk))
        results["verification_override"].append(sys_verification_override(
            task, corr_model, corr_cal, fk))
        results["daphx_r8"].append(sys_daphx_r8(task, corr_model, corr_cal, fk))

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
    parser.add_argument("--corpus", default=str(R8_DIR / "r8_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    R8_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks")

    # Add verification features
    tasks = add_verification_features(tasks)

    # Enrich with contrastive features
    print("Computing enriched features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)
    print(f"Features: {len(feature_keys)} + 1 verification = {len(feature_keys)+1}")

    # Check verification signal quality
    v_correct = [r.get("verification_score", 0.5) for r in all_records if r["is_correct"]]
    v_wrong = [r.get("verification_score", 0.5) for r in all_records if not r["is_correct"]]
    print(f"Verification score: correct={np.mean(v_correct):.3f}, wrong={np.mean(v_wrong):.3f}")

    # Self-confidence for comparison
    c_correct = [r["self_confidence"] for r in all_records if r["is_correct"]]
    c_wrong = [r["self_confidence"] for r in all_records if not r["is_correct"]]
    print(f"Self-confidence: correct={np.mean(c_correct):.1f}, wrong={np.mean(c_wrong):.1f}")

    # Correlation between verification and self-confidence
    v_scores = [r.get("verification_score", 0.5) for r in all_records]
    c_scores = [r["self_confidence"] / 100.0 for r in all_records]
    corr = np.corrcoef(v_scores, c_scores)[0, 1]
    print(f"Correlation(verification, self_confidence): {corr:.3f}")
    print(f"  → {'Independent signal!' if abs(corr) < 0.5 else 'Correlated with self-confidence'}")

    all_results = {}

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        print(f"\n=== seed={seed} ({len(train_tasks)} dev, {len(eval_tasks)} eval) ===")

        corr_model = train_correctness_with_verification(train_records, feature_keys)
        corr_cal = calibrate_correctness(corr_model, cal_records, feature_keys)

        # AUROC
        eval_X, eval_y = [], []
        for task in eval_tasks:
            for c in task["candidates"]:
                feats = build_feature_vector(c["enriched_features"], feature_keys)
                v = np.array([c.get("verification_score", 0.5)])
                eval_X.append(np.concatenate([feats, v]))
                eval_y.append(1 if c["is_correct"] else 0)
        if len(set(eval_y)) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = roc_auc_score(eval_y, corr_model.predict_proba(eval_X)[:, 1])
            print(f"  Correctness AUROC (with verification): {auroc:.4f}")

        results = evaluate_all(eval_tasks, corr_model, corr_cal, feature_keys)
        summary = summarize(results, eval_tasks)

        print(f"{'System':<25} {'Util':>7} {'Acc':>7} {'Imp':>5} {'Reg':>5} {'Force':>6} {'Resc':>5} {'Brk':>5}")
        print("-" * 75)
        for name in ["base", "majority_vote", "max_confidence", "max_verification",
                     "max_calibrated_v", "gate_a_v", "verification_override", "daphx_r8"]:
            s = summary[name]
            print(f"{name:<25} {s['mean_utility']:>7.1f} {s['success_rate']:>6.1%} "
                  f"{s['improvements']:>5} {s['regressions']:>5} "
                  f"{s['n_force']:>6} {s['rescues']:>5} {s['breaks']:>5}")

        all_results[seed] = summary

    # Aggregate
    print(f"\n{'='*100}")
    print(f"  AGGREGATE ACROSS 5 SEEDS (enriched + verification)")
    print(f"{'='*100}")
    print(f"{'System':<25} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7} {'TotF':>6} {'TotR':>6} {'TotB':>6}")
    print("-" * 100)

    names = ["base", "majority_vote", "max_confidence", "max_verification",
             "max_calibrated_v", "gate_a_v", "verification_override", "daphx_r8"]
    agg = {}
    for name in names:
        utils = [all_results[s][name]["mean_utility"] for s in [42, 123, 7, 99, 2024]]
        tot_f = sum(all_results[s][name]["n_force"] for s in [42, 123, 7, 99, 2024])
        tot_r = sum(all_results[s][name]["rescues"] for s in [42, 123, 7, 99, 2024])
        tot_b = sum(all_results[s][name]["breaks"] for s in [42, 123, 7, 99, 2024])
        agg[name] = {"mean": np.mean(utils), "std": np.std(utils),
                     "min": min(utils), "max": max(utils),
                     "tot_f": tot_f, "tot_r": tot_r, "tot_b": tot_b}
        print(f"{name:<25} {np.mean(utils):>7.1f} {np.std(utils):>7.1f} "
              f"{min(utils):>7.1f} {max(utils):>7.1f} "
              f"{tot_f:>6} {tot_r:>6} {tot_b:>6}")

    best_simple = max(agg["majority_vote"]["mean"], agg["max_confidence"]["mean"],
                      agg["max_calibrated_v"]["mean"], agg["max_verification"]["mean"])
    print(f"\n  Best simple baseline: {best_simple:.1f}")
    for name in ["gate_a_v", "verification_override", "daphx_r8"]:
        diff = agg[name]["mean"] - best_simple
        verdict = "BEATS" if diff > 0.5 else ("MATCHES" if abs(diff) < 0.5 else "WORSE")
        print(f"  {name}: {agg[name]['mean']:.1f} ({diff:+.1f}) {verdict} "
              f"[F={agg[name]['tot_f']}, R={agg[name]['tot_r']}, B={agg[name]['tot_b']}]")

    output = {"aggregate": agg, "per_seed": all_results,
              "verification_correlation": corr,
              "n_features": len(feature_keys) + 1}
    output_path = R8_DIR / "r8_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
