#!/usr/bin/env python3
"""Train the pairwise advantage gate for V2B-I3.5.3-r1.

Trains a model to predict:
  ΔQ_π(s, a_B, a_G) = Q^{π_B}(s, a_G) - Q^{π_B}(s, a_B)

Input: state features + a_B one-hot + a_G one-hot
Output: scalar ΔQ_π

Training data: expanded fork dataset where each record has:
  - features(s)
  - a_B, a_G
  - ΔQ_π = U(a_G + OFF continuation) - U(a_B + OFF continuation)

Includes a constant-baseline comparator: "always predict ΔQ_π = 0"
This is the trivial policy that always prefers the baseline action.
If the ML model doesn't beat the constant baseline, the ML model
is not adding predictive value.

Usage:
    python scripts/train_pairwise_advantage_gate.py \\
        --fork-dataset experiments/v2b_i3_5_2/development/i353r1/expanded_fork_dataset_v1.jsonl \\
        --output-dir experiments/v2b_i3_5_2/development/i353r1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.selective_governor.features import (
    InterventionFeatures, extract_features, FEATURE_NAMES,
)
from hrm_adaptive_memory.executive.selective_governor.pairwise_advantage_predictor import (
    PairwiseAdvantagePredictor, ACTION_NAMES, PAIRWISE_FEATURE_NAMES,
    pairwise_feature_vector,
)


def load_fork_dataset(path: str) -> list[dict[str, Any]]:
    """Load the expanded fork dataset."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def features_from_dict(d: dict[str, Any]) -> InterventionFeatures:
    """Reconstruct InterventionFeatures from a dict."""
    return InterventionFeatures(
        remaining_steps=d["remaining_steps"],
        prior_action_count=d["prior_action_count"],
        last_action=d["last_action"],
        last_outcome=d["last_outcome"],
        repeated_no_gain=d["repeated_no_gain"],
        has_cognitive_state=d["has_cognitive_state"],
        evidence_count=d["evidence_count"],
        verified_count=d["verified_count"],
        verification_state=d["verification_state"],
        temporal_status=d["temporal_status"],
        conflict_count=d["conflict_count"],
        reasoning_depth=d["reasoning_depth"],
        retrieval_budget_remaining=d["retrieval_budget_remaining"],
        verification_budget_remaining=d["verification_budget_remaining"],
        search_budget_remaining=d["search_budget_remaining"],
        reasoning_budget_remaining=d["reasoning_budget_remaining"],
        chain_started=d["chain_started"],
        chain_completed=d["chain_completed"],
        chain_length=d["chain_length"],
        chain_stage=d["chain_stage"],
    )


def train_and_evaluate(
    records: list[dict[str, Any]],
    output_dir: Path,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train the pairwise advantage model with fold-isolated CV."""

    # Build feature matrix and target
    X = []
    y = []
    task_ids = []
    action_pairs = []

    for rec in records:
        features = features_from_dict(rec["features"])
        vec = pairwise_feature_vector(features, rec["base_action"], rec["gov_action"])
        X.append(vec)
        y.append(rec["delta_q_pi"])
        task_ids.append(rec["task_id"])
        action_pairs.append((rec["base_action"], rec["gov_action"]))

    X = np.array(X)
    y = np.array(y)
    task_ids = np.array(task_ids)

    print(f"\nTraining data: {len(X)} samples, {X.shape[1]} features")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"  Action pairs: {Counter(action_pairs)}")

    unique_tasks = sorted(set(task_ids))
    print(f"  Unique tasks: {len(unique_tasks)}")

    # Fold-isolated cross-validation by task
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_tasks)):
        train_tasks = set(unique_tasks[t] for t in train_idx)
        test_tasks = set(unique_tasks[t] for t in test_idx)

        train_mask = np.array([t in train_tasks for t in task_ids])
        test_mask = np.array([t in test_tasks for t in task_ids])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Sign accuracy: does the model correctly predict the sign of ΔQ_π?
        # (positive = intervention helps, negative = intervention hurts)
        sign_correct = sum(
            1 for p, a in zip(y_pred, y_test)
            if (p > 0 and a > 0) or (p < 0 and a < 0) or (abs(p) <= 1 and abs(a) <= 1)
        )
        sign_accuracy = sign_correct / len(y_test) if len(y_test) > 0 else 0

        # Constant baseline comparator: always predict ΔQ_π = 0
        # This is the "always prefer baseline" policy
        const_pred = np.zeros(len(y_test))
        const_mse = mean_squared_error(y_test, const_pred)
        const_r2 = r2_score(y_test, const_pred)
        const_sign_correct = sum(1 for a in y_test if abs(a) <= 1)
        const_sign_accuracy = const_sign_correct / len(y_test) if len(y_test) > 0 else 0

        # Intervention decision accuracy:
        # If we use threshold τ=5, how many interventions would the model approve
        # and how many of those would actually be positive?
        threshold = 5.0
        model_intervene = sum(1 for p in y_pred if p > threshold)
        model_intervene_positive = sum(
            1 for p, a in zip(y_pred, y_test) if p > threshold and a > 0)
        model_intervene_negative = sum(
            1 for p, a in zip(y_pred, y_test) if p > threshold and a < 0)

        const_intervene = 0  # constant baseline never intervenes
        const_intervene_positive = 0
        const_intervene_negative = 0

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model_mse": round(mse, 4),
            "model_r2": round(r2, 4),
            "model_sign_accuracy": round(sign_accuracy, 4),
            "const_mse": round(const_mse, 4),
            "const_r2": round(const_r2, 4),
            "const_sign_accuracy": round(const_sign_accuracy, 4),
            "model_intervene_approved": model_intervene,
            "model_intervene_positive": model_intervene_positive,
            "model_intervene_negative": model_intervene_negative,
            "const_intervene_approved": const_intervene,
        })
        print(f"  Fold {fold_idx}: n_train={len(X_train)}, n_test={len(X_test)}")
        print(f"    Model:  MSE={mse:.2f}, R²={r2:.4f}, sign_acc={sign_accuracy:.2%}")
        print(f"    Const0: MSE={const_mse:.2f}, R²={const_r2:.4f}, sign_acc={const_sign_accuracy:.2%}")
        print(f"    Model interventions (τ={threshold}): {model_intervene} "
              f"(positive={model_intervene_positive}, negative={model_intervene_negative})")
        print(f"    Const interventions: {const_intervene}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    final_model.fit(X, y)

    # Feature importance
    feature_importance = sorted(
        zip(PAIRWISE_FEATURE_NAMES, final_model.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\nTop 10 features:")
    for name, imp in feature_importance[:10]:
        print(f"  {name}: {imp:.4f}")

    # Save model
    predictor = PairwiseAdvantagePredictor(
        model=final_model,
        delta_threshold=5.0,
        lcb_margin=5.0,
    )
    model_path = output_dir / "pairwise_advantage_gate_v1.pkl"
    predictor.save(model_path)
    print(f"\nSaved model: {model_path}")

    # Compute model SHA-256
    import hashlib
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Summary
    mean_model_mse = sum(f["model_mse"] for f in fold_results) / len(fold_results)
    mean_model_r2 = sum(f["model_r2"] for f in fold_results) / len(fold_results)
    mean_model_sign = sum(f["model_sign_accuracy"] for f in fold_results) / len(fold_results)
    mean_const_mse = sum(f["const_mse"] for f in fold_results) / len(fold_results)
    mean_const_r2 = sum(f["const_r2"] for f in fold_results) / len(fold_results)
    mean_const_sign = sum(f["const_sign_accuracy"] for f in fold_results) / len(fold_results)
    total_model_intervene = sum(f["model_intervene_approved"] for f in fold_results)
    total_model_positive = sum(f["model_intervene_positive"] for f in fold_results)
    total_model_negative = sum(f["model_intervene_negative"] for f in fold_results)

    summary = {
        "schema": "DAPH_V2B_I3_5_3R1_PAIRWISE_TRAINING_V1",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_tasks": len(unique_tasks),
        "target_stats": {
            "min": round(float(y.min()), 4),
            "max": round(float(y.max()), 4),
            "mean": round(float(y.mean()), 4),
            "std": round(float(y.std()), 4),
        },
        "action_pair_distribution": {
            f"{b}->{g}": cnt for (b, g), cnt in Counter(action_pairs).most_common()
        },
        "fold_results": fold_results,
        "model_performance": {
            "mean_mse": round(mean_model_mse, 4),
            "mean_r2": round(mean_model_r2, 4),
            "mean_sign_accuracy": round(mean_model_sign, 4),
            "total_interventions_approved": total_model_intervene,
            "total_interventions_positive": total_model_positive,
            "total_interventions_negative": total_model_negative,
        },
        "constant_baseline": {
            "mean_mse": round(mean_const_mse, 4),
            "mean_r2": round(mean_const_r2, 4),
            "mean_sign_accuracy": round(mean_const_sign, 4),
            "total_interventions_approved": 0,
        },
        "model_beats_baseline": {
            "mse": mean_model_mse < mean_const_mse,
            "r2": mean_model_r2 > mean_const_r2,
            "sign_accuracy": mean_model_sign > mean_const_sign,
        },
        "top_features": [{"name": n, "importance": round(float(i), 6)}
                         for n, i in feature_importance[:15]],
        "model_path": str(model_path),
        "model_sha256": model_sha,
    }

    summary_path = output_dir / "pairwise_advantage_training_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train pairwise advantage gate")
    parser.add_argument(
        "--fork-dataset",
        default="experiments/v2b_i3_5_2/development/i353r1/expanded_fork_dataset_v1.jsonl",
    )
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development/i353r1")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fork dataset from {args.fork_dataset}...")
    records = load_fork_dataset(args.fork_dataset)
    print(f"Loaded {len(records)} fork records")

    print("\n" + "=" * 78)
    print("TRAINING PAIRWISE ADVANTAGE GATE")
    print("=" * 78)
    summary = train_and_evaluate(records, output_dir, n_splits=args.n_splits)

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Model R²: {summary['model_performance']['mean_r2']:.4f}")
    print(f"  Model sign accuracy: {summary['model_performance']['mean_sign_accuracy']:.2%}")
    print(f"  Const0 R²: {summary['constant_baseline']['mean_r2']:.4f}")
    print(f"  Const0 sign accuracy: {summary['constant_baseline']['mean_sign_accuracy']:.2%}")
    print(f"  Model beats baseline (R²): {summary['model_beats_baseline']['r2']}")
    print(f"  Model beats baseline (sign): {summary['model_beats_baseline']['sign_accuracy']}")
    print(f"  Model interventions (CV total): {summary['model_performance']['total_interventions_approved']}")
    print(f"    positive: {summary['model_performance']['total_interventions_positive']}")
    print(f"    negative: {summary['model_performance']['total_interventions_negative']}")
    print(f"  Model SHA-256: {summary['model_sha256'][:16]}...")


if __name__ == "__main__":
    main()
