#!/usr/bin/env python3
"""Analyze the 6 breaks from R9 Gate A to understand failure mechanisms.

For each break, we extract:
  - task_id, category, difficulty
  - base candidate (MaxCal pick) correctness, confidence, verification
  - override candidate (DAPH-X pick) correctness, confidence, verification
  - P(correct) margin that triggered the override
  - pairwise scores
  - answer similarity
  - whether verification agreed or disagreed with the override
  - whether majority vote agreed with base or override

This identifies which mechanisms caused the authority to believe override was safe.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_r7_evaluation import (
    enrich_corpus, get_feature_keys, build_feature_vector,
    flatten_candidates, split_tasks, load_corpus,
)
from run_r9_evaluation import (
    add_multiround_verification_features, add_pairwise_features,
    add_answer_semantic_features, predict_correctness_r9,
    train_correctness_r9, calibrate_r9,
)


def analyze_breaks(corpus_path: str):
    tasks = load_corpus(corpus_path)
    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    print("Computing enriched features...")
    tasks = enrich_corpus(tasks)

    all_records = flatten_candidates(tasks)
    feature_keys = get_feature_keys(all_records)

    all_breaks = []
    all_rescues = []

    for seed in [42, 123, 7, 99, 2024]:
        train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=seed)
        train_records = flatten_candidates(train_tasks)
        cal_records = flatten_candidates(cal_tasks)

        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, cal_records, feature_keys)

        for task in eval_tasks:
            cands = task["candidates"]
            for c in cands:
                c["p_correct_r9"] = predict_correctness_r9(
                    corr_model, corr_cal, c["enriched_features"], c, feature_keys)

            # R9 Gate A: base = candidates[0] (raw model), NOT MaxCal
            base_pick = cands[0]
            p_base = base_pick["p_correct_r9"]

            # Also compute MaxCal pick for comparison
            maxcal_pick = max(cands, key=lambda c: c["p_correct_r9"])
            p_maxcal = maxcal_pick["p_correct_r9"]

            # Gate A: find best P(correct) among ALL candidates
            best, best_p = base_pick, p_base
            for c in cands:
                if c["p_correct_r9"] > best_p:
                    best_p = c["p_correct_r9"]
                    best = c

            margin = 0.10
            would_force = (best["candidate_id"] != base_pick["candidate_id"]) and (best_p - p_base > margin)

            if not would_force:
                continue

            # This is an intervention (relative to raw base)
            is_rescue = (not base_pick["is_correct"]) and best["is_correct"]
            is_break = base_pick["is_correct"] and (not best["is_correct"])

            entry = {
                "seed": seed,
                "task_id": task["task_id"],
                "category": task.get("category", "?"),
                "difficulty": task.get("difficulty", "?"),
                # Raw base (what Gate A overrides)
                "base_answer": base_pick["answer"],
                "base_correct": base_pick["is_correct"],
                "base_confidence": base_pick["self_confidence"],
                "base_verification": base_pick.get("verification_score", 0.5),
                "base_p_correct": p_base,
                "base_pairwise_winrate": base_pick.get("pairwise_winrate", 0.5),
                # MaxCal pick (the new R10 base policy)
                "maxcal_answer": maxcal_pick["answer"],
                "maxcal_correct": maxcal_pick["is_correct"],
                "maxcal_confidence": maxcal_pick["self_confidence"],
                "maxcal_verification": maxcal_pick.get("verification_score", 0.5),
                "maxcal_p_correct": p_maxcal,
                "maxcal_pairwise_winrate": maxcal_pick.get("pairwise_winrate", 0.5),
                "maxcal_is_base": maxcal_pick["candidate_id"] == base_pick["candidate_id"],
                # Override (what Gate A chose)
                "override_answer": best["answer"],
                "override_correct": best["is_correct"],
                "override_confidence": best["self_confidence"],
                "override_verification": best.get("verification_score", 0.5),
                "override_p_correct": best_p,
                "override_pairwise_winrate": best.get("pairwise_winrate", 0.5),
                "p_margin_over_base": best_p - p_base,
                "p_margin_over_maxcal": best_p - p_maxcal,
                "answers_same": base_pick["answer"] == best["answer"],
                # Majority vote
                "majority_answer": Counter([c["answer"] for c in cands]).most_common(1)[0][0],
                "majority_agrees_with_base": Counter([c["answer"] for c in cands]).most_common(1)[0][0] == base_pick["answer"],
                "majority_agrees_with_override": Counter([c["answer"] for c in cands]).most_common(1)[0][0] == best["answer"],
                "majority_agrees_with_maxcal": Counter([c["answer"] for c in cands]).most_common(1)[0][0] == maxcal_pick["answer"],
                # Verification agreement
                "verification_favors_override_over_base": base_pick.get("verification_score", 0.5) < best.get("verification_score", 0.5),
                "verification_favors_override_over_maxcal": maxcal_pick.get("verification_score", 0.5) < best.get("verification_score", 0.5),
                # Answer similarity
                "answer_similarity": best.get("answer_avg_similarity", 0.0),
                # Would this break also happen over MaxCal?
                "break_over_maxcal": maxcal_pick["is_correct"] and (not best["is_correct"]) and (maxcal_pick["candidate_id"] != best["candidate_id"]),
                "rescue_over_maxcal": (not maxcal_pick["is_correct"]) and best["is_correct"] and (maxcal_pick["candidate_id"] != best["candidate_id"]),
            }

            if is_break:
                all_breaks.append(entry)
            elif is_rescue:
                all_rescues.append(entry)

    print(f"\n{'='*100}")
    print(f"BREAK ANALYSIS: {len(all_breaks)} breaks across 5 seeds")
    print(f"{'='*100}")

    for i, b in enumerate(all_breaks):
        print(f"\n--- Break {i+1}/{len(all_breaks)} ---")
        print(f"  Task: {b['task_id']} ({b['category']}, {b['difficulty']})")
        print(f"  Seed: {b['seed']}")
        print(f"  Raw base:    answer='{b['base_answer']}' correct={b['base_correct']}")
        print(f"    confidence={b['base_confidence']:.1f} verification={b['base_verification']:.3f}")
        print(f"    P(correct)={b['base_p_correct']:.3f} pairwise_wr={b['base_pairwise_winrate']:.3f}")
        print(f"  MaxCal pick: answer='{b['maxcal_answer']}' correct={b['maxcal_correct']}")
        print(f"    confidence={b['maxcal_confidence']:.1f} verification={b['maxcal_verification']:.3f}")
        print(f"    P(correct)={b['maxcal_p_correct']:.3f} pairwise_wr={b['maxcal_pairwise_winrate']:.3f}")
        print(f"    MaxCal==base? {b['maxcal_is_base']}")
        print(f"  Override:    answer='{b['override_answer']}' correct={b['override_correct']}")
        print(f"    confidence={b['override_confidence']:.1f} verification={b['override_verification']:.3f}")
        print(f"    P(correct)={b['override_p_correct']:.3f} pairwise_wr={b['override_pairwise_winrate']:.3f}")
        print(f"  P(correct) margin over base:   {b['p_margin_over_base']:.3f}")
        print(f"  P(correct) margin over MaxCal: {b['p_margin_over_maxcal']:.3f}")
        print(f"  Majority vote: '{b['majority_answer']}'")
        print(f"    agrees with base: {b['majority_agrees_with_base']}")
        print(f"    agrees with MaxCal: {b['majority_agrees_with_maxcal']}")
        print(f"    agrees with override: {b['majority_agrees_with_override']}")
        print(f"  Verification favors override over base:   {b['verification_favors_override_over_base']}")
        print(f"  Verification favors override over MaxCal: {b['verification_favors_override_over_maxcal']}")
        print(f"  Answer similarity: {b['answer_similarity']:.3f}")
        print(f"  Would also break over MaxCal? {b['break_over_maxcal']}")
        print(f"  Would rescue over MaxCal?     {b['rescue_over_maxcal']}")

    # Mechanism summary
    print(f"\n{'='*100}")
    print(f"BREAK MECHANISM SUMMARY")
    print(f"{'='*100}")

    mechanisms = Counter()
    for b in all_breaks:
        if b["p_margin_over_base"] > 0.20:
            mechanisms["high_p_margin_over_base (>0.20)"] += 1
        if b["p_margin_over_maxcal"] > 0.10:
            mechanisms["high_p_margin_over_maxcal (>0.10)"] += 1
        if b["override_confidence"] > b["base_confidence"]:
            mechanisms["override higher confidence than base"] += 1
        if b["verification_favors_override_over_base"]:
            mechanisms["verification favored override over base"] += 1
        if b["verification_favors_override_over_maxcal"]:
            mechanisms["verification favored override over maxcal"] += 1
        if b["majority_agrees_with_override"]:
            mechanisms["majority favored override"] += 1
        if b["override_pairwise_winrate"] > b["base_pairwise_winrate"]:
            mechanisms["pairwise favored override"] += 1
        if b["base_verification"] < 0.5:
            mechanisms["base verification low"] += 1
        if b["base_confidence"] < 50:
            mechanisms["base confidence low"] += 1
        if b["maxcal_is_base"]:
            mechanisms["maxcal == base (same pick)"] += 1
        if b["break_over_maxcal"]:
            mechanisms["would also break over MaxCal"] += 1

    print(f"\nBreak mechanism frequencies:")
    for mech, count in mechanisms.most_common():
        print(f"  {mech}: {count}/{len(all_breaks)}")

    # Compare to rescues
    print(f"\n{'='*100}")
    print(f"RESCUE vs BREAK COMPARISON ({len(all_rescues)} rescues, {len(all_breaks)} breaks)")
    print(f"{'='*100}")

    if all_rescues and all_breaks:
        print(f"\n{'Feature':<35} {'Rescues (mean)':>15} {'Breaks (mean)':>15}")
        print("-" * 70)
        for key in ["p_margin_over_base", "p_margin_over_maxcal",
                     "base_confidence", "override_confidence", "maxcal_confidence",
                     "base_verification", "override_verification", "maxcal_verification",
                     "base_p_correct", "override_p_correct", "maxcal_p_correct",
                     "base_pairwise_winrate", "override_pairwise_winrate", "maxcal_pairwise_winrate",
                     "answer_similarity"]:
            r_vals = [r[key] for r in all_rescues]
            b_vals = [b[key] for b in all_breaks]
            print(f"  {key:<33} {np.mean(r_vals):>15.3f} {np.mean(b_vals):>15.3f}")

        # Majority agreement
        r_maj = sum(1 for r in all_rescues if r["majority_agrees_with_override"]) / len(all_rescues)
        b_maj = sum(1 for b in all_breaks if b["majority_agrees_with_override"]) / len(all_breaks)
        print(f"  {'majority favors override':<33} {r_maj:>15.1%} {b_maj:>15.1%}")

        # Verification agreement
        r_ver = sum(1 for r in all_rescues if r.get("verification_favors_override_over_base", False)) / len(all_rescues)
        b_ver = sum(1 for b in all_breaks if b.get("verification_favors_override_over_base", False)) / len(all_breaks)
        print(f"  {'verification favors override':<33} {r_ver:>15.1%} {b_ver:>15.1%}")

        # Critical: how many breaks/rescues are just "MaxCal over raw base"?
        r_maxcal_is_override = sum(1 for r in all_rescues if r["maxcal_answer"] == r["override_answer"]) / len(all_rescues)
        b_maxcal_is_override = sum(1 for b in all_breaks if b["maxcal_answer"] == b["override_answer"]) / len(all_breaks)
        print(f"\n  CRITICAL: Override == MaxCal pick?")
        print(f"    Rescues: {r_maxcal_is_override:.1%} (override is just MaxCal)")
        print(f"    Breaks:  {b_maxcal_is_override:.1%} (override is just MaxCal)")

        r_break_over_maxcal = sum(1 for r in all_rescues if r.get("break_over_maxcal", False))
        b_break_over_maxcal = sum(1 for b in all_breaks if b.get("break_over_maxcal", False))
        r_rescue_over_maxcal = sum(1 for r in all_rescues if r.get("rescue_over_maxcal", False))
        b_rescue_over_maxcal = sum(1 for b in all_breaks if b.get("rescue_over_maxcal", False))
        print(f"\n  If base were MaxCal instead of raw model:")
        print(f"    Rescues that would still be rescues over MaxCal: {r_rescue_over_maxcal}/{len(all_rescues)}")
        print(f"    Breaks that would still be breaks over MaxCal:  {b_break_over_maxcal}/{len(all_breaks)}")
        print(f"    → Switching to MaxCal base eliminates {len(all_breaks) - b_break_over_maxcal}/{len(all_breaks)} breaks")
        print(f"    → But also eliminates {len(all_rescues) - r_rescue_over_maxcal}/{len(all_rescues)} rescues")

    # Save detailed analysis
    output_path = REPO_ROOT / "experiments/daph_x/r9/r9_break_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"breaks": all_breaks, "rescues": all_rescues,
                   "mechanism_counts": dict(mechanisms)}, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO_ROOT / "experiments/daph_x/cross_verification/cv_corpus_v2.jsonl")
    analyze_breaks(corpus)
