#!/usr/bin/env python3
"""Train Q_res on M3 structural-heldout data.

Trains on simple structures, evaluates on novel topology families
with zero structural signature overlap.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.train_q_res import extract_features, compute_q_mb


def load_corpus(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def evaluate_groupwise_regret(records, q_total, group_key="counterfactual_group_id"):
    """Compute groupwise regret: max_utility - utility(selected)."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r[group_key]].append((i, r))

    regrets = []
    for gid, group in groups.items():
        if len(group) < 2:
            continue
        oracle_utility = max(r["utility"] for _, r in group)
        pred_idx = np.argmax([q_total[i] for i, _ in group])
        regrets.append(oracle_utility - group[pred_idx][1]["utility"])

    return np.mean(regrets) if regrets else 0.0


def main():
    train_path = REPO_ROOT / "experiments/daph_x/m3_structural/m3_train.jsonl"
    test_path = REPO_ROOT / "experiments/daph_x/m3_structural/m3_test.jsonl"

    train_records = load_corpus(train_path)
    test_records = load_corpus(test_path)

    print(f"Train: {len(train_records)} records, {len(set(r['counterfactual_group_id'] for r in train_records))} groups")
    print(f"Test: {len(test_records)} records, {len(set(r['counterfactual_group_id'] for r in test_records))} groups")

    # Extract features
    train_feats = [extract_features(r, {}) for r in train_records]
    test_feats = [extract_features(r, {}) for r in test_records]
    feature_keys = sorted(train_feats[0].keys())

    X_train = np.array([[f[k] for k in feature_keys] for f in train_feats])
    X_test = np.array([[f[k] for k in feature_keys] for f in test_feats])

    y_train = np.array([r["utility"] for r in train_records])
    y_test = np.array([r["utility"] for r in test_records])

    q_mb_train = np.array([compute_q_mb(r) for r in train_records])
    q_mb_test = np.array([compute_q_mb(r) for r in test_records])

    y_res_train = y_train - q_mb_train

    # Train
    print(f"\nTraining on {X_train.shape[0]} records...")
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    model.fit(X_train, y_res_train)

    # Predict
    q_res_pred_train = model.predict(X_train)
    q_res_pred_test = model.predict(X_test)

    q_hybrid_train = q_mb_train + q_res_pred_train
    q_hybrid_test = q_mb_test + q_res_pred_test

    # MAE
    mae_mb_train = mean_absolute_error(y_train, q_mb_train)
    mae_hybrid_train = mean_absolute_error(y_train, q_hybrid_train)
    mae_mb_test = mean_absolute_error(y_test, q_mb_test)
    mae_hybrid_test = mean_absolute_error(y_test, q_hybrid_test)

    print(f"\n{'='*60}")
    print(f"M3 STRUCTURAL HELDOUT RESULTS")
    print(f"{'='*60}")
    print(f"TRAIN:")
    print(f"  MAE(Q_MB):     {mae_mb_train:.2f}")
    print(f"  MAE(Q_hybrid): {mae_hybrid_train:.2f}")
    print(f"TEST (structural heldout):")
    print(f"  MAE(Q_MB):     {mae_mb_test:.2f}")
    print(f"  MAE(Q_hybrid): {mae_hybrid_test:.2f}")
    print(f"  Improvement:   {mae_mb_test - mae_hybrid_test:.2f} ({100*(mae_mb_test - mae_hybrid_test)/max(mae_mb_test, 1e-9):.1f}%)")

    # Regret
    regret_mb_train = evaluate_groupwise_regret(train_records, q_mb_train)
    regret_hybrid_train = evaluate_groupwise_regret(train_records, q_hybrid_train)
    regret_mb_test = evaluate_groupwise_regret(test_records, q_mb_test)
    regret_hybrid_test = evaluate_groupwise_regret(test_records, q_hybrid_test)

    print(f"\nREGRET:")
    print(f"  TRAIN: MB={regret_mb_train:.2f}, Hybrid={regret_hybrid_train:.2f}")
    print(f"  TEST:  MB={regret_mb_test:.2f}, Hybrid={regret_hybrid_test:.2f}")
    print(f"  Improvement: {regret_mb_test - regret_hybrid_test:.2f}")

    # Top-1 accuracy
    def top1_accuracy(records, q_total):
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r["counterfactual_group_id"]].append((i, r))
        correct = 0
        total = 0
        for gid, group in groups.items():
            if len(group) < 2:
                continue
            oracle_idx = np.argmax([r["utility"] for _, r in group])
            pred_idx = np.argmax([q_total[i] for i, _ in group])
            if oracle_idx == pred_idx:
                correct += 1
            total += 1
        return correct / max(total, 1)

    top1_mb = top1_accuracy(test_records, q_mb_test)
    top1_hybrid = top1_accuracy(test_records, q_hybrid_test)

    print(f"\nTOP-1 ACCURACY (test):")
    print(f"  Q_MB:     {top1_mb:.3f}")
    print(f"  Q_hybrid: {top1_hybrid:.3f}")

    # Feature importance
    print(f"\nFeature importance:")
    importances = model.feature_importances_
    for k, imp in sorted(zip(feature_keys, importances), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {imp:.3f}")

    # Save
    import joblib
    model_path = REPO_ROOT / "experiments/daph_x/m3_structural/q_res_m3.pkl"
    joblib.dump({
        "model": model,
        "feature_keys": feature_keys,
        "mae_mb_test": mae_mb_test,
        "mae_hybrid_test": mae_hybrid_test,
        "regret_mb_test": regret_mb_test,
        "regret_hybrid_test": regret_hybrid_test,
        "top1_mb": top1_mb,
        "top1_hybrid": top1_hybrid,
    }, model_path)
    print(f"\nSaved to {model_path}")


if __name__ == "__main__":
    main()
