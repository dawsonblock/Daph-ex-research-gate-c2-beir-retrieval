#!/usr/bin/env python3
"""R12 Phase 4: Build checkpoint-level counterfactual records.

For each task and each checkpoint K in {2, 4, 6, 8, 10}:
  - Record state s_K (compact + enhanced features)
  - Record Q(s_K, STOP) = U(MaxCal@K)
  - Record Q(s_K, GENERATE) = U(MaxCal@K+2)
  - Record Q(s_K, FULL) = U(MaxCal@12)
  - Record ΔU = Q(GENERATE) - Q(STOP)
  - Record ΔQ = ΔU - λ*C
  - Classify: rescue / break / waste / missed_rescue / correct_stop

Output: experiments/daph_x/r12/r12_counterfactuals.jsonl

Usage:
    python scripts/run_r12_counterfactual.py \\
        --corpus experiments/daph_x/r12/r12_corpus_12.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

R12_DIR = REPO_ROOT / "experiments/daph_x"

from run_r9_evaluation import (
    train_correctness_r9, calibrate_r9, predict_correctness_r9,
    add_multiround_verification_features,
    add_pairwise_features,
    add_answer_semantic_features,
)
from run_r7_evaluation import (
    flatten_candidates, get_feature_keys, enrich_corpus,
)
from run_r11_2_evaluation import (
    compute_enhanced_state_features,
    load_corpus,
)


CHECKPOINTS = [2, 4, 6, 8, 10]
LAMBDA_COST = 0.1
COST_PER_STEP = 2.0 / 10.0  # Normalized cost of generating 2 more candidates


def build_counterfactuals(tasks, corr_model, corr_cal, feature_keys):
    """Build checkpoint-level counterfactual records for all tasks."""
    records = []

    for task in tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        prev_state = None
        for k in CHECKPOINTS:
            if k + 2 > len(cands):
                break

            # State at checkpoint K
            state_k = compute_enhanced_state_features(
                task, k, corr_model, corr_cal, feature_keys, prev_state)

            # Utility of stopping at K
            pick_k = max(cands[:k], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, feature_keys))
            u_stop = 1.0 if pick_k["is_correct"] else 0.0

            # Utility of generating 2 more (K+2)
            state_k2 = compute_enhanced_state_features(
                task, k + 2, corr_model, corr_cal, feature_keys, state_k)
            pick_k2 = max(cands[:k + 2], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, feature_keys))
            u_generate = 1.0 if pick_k2["is_correct"] else 0.0

            # Utility of full generation (K=12)
            pick_12 = max(cands[:12], key=lambda c: predict_correctness_r9(
                corr_model, corr_cal, c.get("enriched_features", {}), c, feature_keys))
            u_full = 1.0 if pick_12["is_correct"] else 0.0

            # Counterfactual quantities
            delta_u = u_generate - u_stop
            delta_q = delta_u - LAMBDA_COST * COST_PER_STEP
            delta_u_full = u_full - u_stop

            # Classify the GENERATE action
            if delta_u > 0:
                action_class = "rescue"
            elif delta_u < 0:
                action_class = "break"
            else:
                action_class = "waste"  # Generated but no benefit

            # Classify the STOP action
            if delta_u > 0:
                stop_class = "missed_rescue"
            else:
                stop_class = "correct_stop"

            # Extract compact features
            compact = {
                "p_top1": state_k.get("p_top1", 0.0),
                "margin": state_k.get("margin", 0.0),
                "answer_entropy": state_k.get("answer_entropy", 0.0),
                "selection_stability": state_k.get("selection_stability", 0.0),
                "delta_p_top1": state_k.get("delta_p_top1", 0.0),
                "agreement_rate": state_k.get("agreement_rate", 0.0),
            }

            # Extract enhanced features (for ablation)
            enhanced = {k2: v for k2, v in state_k.items() if not k2.startswith("_")}

            record = {
                "task_id": task["task_id"],
                "category": task.get("category", "unknown"),
                "difficulty": task.get("difficulty", "unknown"),
                "checkpoint_k": k,
                "n_candidates_total": len(cands),
                # Utilities
                "u_stop": u_stop,
                "u_generate": u_generate,
                "u_full": u_full,
                # Counterfactual
                "delta_u": delta_u,
                "delta_q": delta_q,
                "delta_u_full": delta_u_full,
                "cost_per_step": COST_PER_STEP,
                "lambda": LAMBDA_COST,
                # Classification
                "action_class": action_class,
                "stop_class": stop_class,
                # Features
                "compact_features": compact,
                "enhanced_features": enhanced,
                # MaxCal picks
                "maxcal_k_answer": pick_k["answer"],
                "maxcal_k_correct": pick_k["is_correct"],
                "maxcal_k2_answer": pick_k2["answer"],
                "maxcal_k2_correct": pick_k2["is_correct"],
                "maxcal_12_answer": pick_12["answer"],
                "maxcal_12_correct": pick_12["is_correct"],
            }
            records.append(record)

            prev_state = state_k

    return records


def summarize_counterfactuals(records):
    """Summarize counterfactual statistics."""
    n = len(records)
    if n == 0:
        return {}

    classes = {}
    for r in records:
        classes[r["action_class"]] = classes.get(r["action_class"], 0) + 1

    delta_us = [r["delta_u"] for r in records]
    delta_qs = [r["delta_q"] for r in records]
    delta_u_fulls = [r["delta_u_full"] for r in records]

    # Per-checkpoint statistics
    per_k = {}
    for k in CHECKPOINTS:
        k_records = [r for r in records if r["checkpoint_k"] == k]
        if not k_records:
            continue
        per_k[k] = {
            "n": len(k_records),
            "rescue_rate": sum(1 for r in k_records if r["action_class"] == "rescue") / len(k_records),
            "break_rate": sum(1 for r in k_records if r["action_class"] == "break") / len(k_records),
            "waste_rate": sum(1 for r in k_records if r["action_class"] == "waste") / len(k_records),
            "mean_delta_u": float(np.mean([r["delta_u"] for r in k_records])),
            "mean_delta_q": float(np.mean([r["delta_q"] for r in k_records])),
            "u_stop_rate": float(np.mean([r["u_stop"] for r in k_records])),
            "u_generate_rate": float(np.mean([r["u_generate"] for r in k_records])),
        }

    return {
        "n_total": n,
        "n_rescue": classes.get("rescue", 0),
        "n_break": classes.get("break", 0),
        "n_waste": classes.get("waste", 0),
        "rescue_rate": classes.get("rescue", 0) / n,
        "break_rate": classes.get("break", 0) / n,
        "waste_rate": classes.get("waste", 0) / n,
        "mean_delta_u": float(np.mean(delta_us)),
        "mean_delta_q": float(np.mean(delta_qs)),
        "mean_delta_u_full": float(np.mean(delta_u_fulls)),
        "per_checkpoint": per_k,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(R12_DIR / "r12/r12_corpus_12.jsonl"))
    parser.add_argument("--output", default=str(R12_DIR / "r12/r12_counterfactuals.jsonl"))
    parser.add_argument("--seeds", default="42,123,7,99,2024")
    args = parser.parse_args()

    # Load and enrich corpus
    tasks = load_corpus(args.corpus)
    tasks = [t for t in tasks if len(t.get("candidates", [])) >= 12]
    print(f"Loaded {len(tasks)} tasks with >= 12 candidates")

    if len(tasks) < 50:
        print("WARNING: Need at least 50 tasks. Run data collection first.")
        return

    print("Enriching candidates...")
    tasks = add_multiround_verification_features(tasks)
    tasks = add_pairwise_features(tasks)
    tasks = add_answer_semantic_features(tasks)
    tasks = enrich_corpus(tasks)

    seeds = [int(s) for s in args.seeds.split(",")]
    all_summaries = {}

    for seed in seeds:
        print(f"\n=== seed={seed} ===")

        # Split tasks
        n = len(tasks)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        n_train = int(n * 0.6)
        n_cal = int(n * 0.15)
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:n_train + n_cal]
        test_idx = indices[n_train + n_cal:]

        train_tasks = [tasks[i] for i in train_idx]
        cal_tasks = [tasks[i] for i in cal_idx]
        test_tasks = [tasks[i] for i in test_idx]

        # Train correctness model
        train_records = flatten_candidates(train_tasks)
        feature_keys = get_feature_keys(train_records)
        corr_model = train_correctness_r9(train_records, feature_keys)
        corr_cal = calibrate_r9(corr_model, flatten_candidates(cal_tasks), feature_keys)

        # Build counterfactuals for ALL tasks
        records = build_counterfactuals(tasks, corr_model, corr_cal, feature_keys)
        summary = summarize_counterfactuals(records)

        print(f"  Total records: {summary['n_total']}")
        print(f"  Rescue: {summary['n_rescue']} ({summary['rescue_rate']:.1%})")
        print(f"  Break: {summary['n_break']} ({summary['break_rate']:.1%})")
        print(f"  Waste: {summary['n_waste']} ({summary['waste_rate']:.1%})")
        print(f"  Mean ΔU: {summary['mean_delta_u']:.4f}")
        print(f"  Mean ΔQ: {summary['mean_delta_q']:.4f}")
        print(f"  Mean ΔU_full: {summary['mean_delta_u_full']:.4f}")

        print(f"  Per-checkpoint:")
        for k in CHECKPOINTS:
            if k in summary["per_checkpoint"]:
                pk = summary["per_checkpoint"][k]
                print(f"    K={k:2d}: rescue={pk['rescue_rate']:.1%} "
                      f"break={pk['break_rate']:.1%} "
                      f"waste={pk['waste_rate']:.1%} "
                      f"ΔU={pk['mean_delta_u']:.4f}")

        all_summaries[seed] = summary

        # Save records for first seed only (they're similar across seeds)
        if seed == 42:
            with open(args.output, "w") as f:
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"  Saved {len(records)} records to {args.output}")

    # Aggregate summary
    print(f"\n{'='*60}")
    print(f"  AGGREGATE (mean across seeds)")
    print(f"{'='*60}")

    rescue_rates = [all_summaries[s]["rescue_rate"] for s in seeds]
    break_rates = [all_summaries[s]["break_rate"] for s in seeds]
    waste_rates = [all_summaries[s]["waste_rate"] for s in seeds]
    delta_us = [all_summaries[s]["mean_delta_u"] for s in seeds]
    delta_qs = [all_summaries[s]["mean_delta_q"] for s in seeds]

    print(f"  Rescue rate: {np.mean(rescue_rates):.1%} ± {np.std(rescue_rates):.1%}")
    print(f"  Break rate: {np.mean(break_rates):.1%} ± {np.std(break_rates):.1%}")
    print(f"  Waste rate: {np.mean(waste_rates):.1%} ± {np.std(waste_rates):.1%}")
    print(f"  Mean ΔU: {np.mean(delta_us):.4f} ± {np.std(delta_us):.4f}")
    print(f"  Mean ΔQ: {np.mean(delta_qs):.4f} ± {np.std(delta_qs):.4f}")

    # Save aggregate
    agg_path = Path(args.output).parent / "r12_counterfactual_summary.json"
    with open(agg_path, "w") as f:
        json.dump({
            "per_seed": all_summaries,
            "aggregate": {
                "rescue_rate": float(np.mean(rescue_rates)),
                "break_rate": float(np.mean(break_rates)),
                "waste_rate": float(np.mean(waste_rates)),
                "mean_delta_u": float(np.mean(delta_us)),
                "mean_delta_q": float(np.mean(delta_qs)),
            },
        }, f, indent=2, default=str)
    print(f"  Summary saved to {agg_path}")


if __name__ == "__main__":
    main()
