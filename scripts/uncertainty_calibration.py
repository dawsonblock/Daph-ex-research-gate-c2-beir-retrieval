#!/usr/bin/env python3
"""M3.7: Calibrated Decision Uncertainty.

Train an ensemble of K residual models with different seeds.
Compute sigma_Q, sigma_Delta, LCB, and calibration tables.

Usage:
    python scripts/uncertainty_calibration.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.train_q_res import extract_features, compute_q_mb


K_ENSEMBLE = 10


def load_corpus(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def train_ensemble(X_train, y_res_train, k=K_ENSEMBLE):
    """Train K ensemble models with bootstrap sampling (bagging).

    Each model is trained on a bootstrap resample of the training data.
    This creates actual diversity in the ensemble.
    """
    models = []
    n = X_train.shape[0]
    for i in range(k):
        # Bootstrap resample
        rng = np.random.RandomState(42 + i)
        indices = rng.choice(n, size=n, replace=True)
        X_boot = X_train[indices]
        y_boot = y_res_train[indices]

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42 + i,
        )
        model.fit(X_boot, y_boot)
        models.append(model)
    return models


def ensemble_predict(models, X):
    """Return mean and std of ensemble predictions."""
    preds = np.array([m.predict(X) for m in models])  # (K, N)
    mu = np.mean(preds, axis=0)
    sigma = np.std(preds, axis=0, ddof=1)
    return mu, sigma


def evaluate_groupwise(records, q_total, group_key="counterfactual_group_id"):
    """Compute groupwise regret and top-1 accuracy."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r[group_key]].append((i, r))

    regrets = []
    top1_correct = 0
    total_groups = 0

    for gid, group in groups.items():
        if len(group) < 2:
            continue
        oracle_utility = max(r["utility"] for _, r in group)
        pred_idx = np.argmax([q_total[i] for i, _ in group])
        regrets.append(oracle_utility - group[pred_idx][1]["utility"])

        oracle_idx = np.argmax([r["utility"] for _, r in group])
        if pred_idx == oracle_idx:
            top1_correct += 1
        total_groups += 1

    return {
        "regret": np.mean(regrets) if regrets else 0.0,
        "top1": top1_correct / max(total_groups, 1),
        "n_groups": total_groups,
    }


def main():
    train_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_train.jsonl")
    test_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_test.jsonl")

    print(f"Train: {len(train_records)} records")
    print(f"Test: {len(test_records)} records")

    # Features
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

    # Train ensemble
    print(f"\nTraining ensemble of {K_ENSEMBLE} models...")
    models = train_ensemble(X_train, y_res_train, k=K_ENSEMBLE)

    # Ensemble predictions
    mu_res_train, sigma_res_train = ensemble_predict(models, X_train)
    mu_res_test, sigma_res_test = ensemble_predict(models, X_test)

    q_total_train = q_mb_train + mu_res_train
    q_total_test = q_mb_test + mu_res_test

    # MAE
    mae_mb = mean_absolute_error(y_test, q_mb_test)
    mae_hybrid = mean_absolute_error(y_test, q_total_test)

    print(f"\n{'='*60}")
    print(f"M3.7 UNCERTAINTY CALIBRATION")
    print(f"{'='*60}")
    print(f"MAE(Q_MB):     {mae_mb:.2f}")
    print(f"MAE(Q_hybrid): {mae_hybrid:.2f}")

    # Groupwise evaluation
    eval_mb = evaluate_groupwise(test_records, q_mb_test)
    eval_hybrid = evaluate_groupwise(test_records, q_total_test)

    print(f"\nRegret(Q_MB):     {eval_mb['regret']:.2f}")
    print(f"Regret(Q_hybrid): {eval_hybrid['regret']:.2f}")
    print(f"Top1(Q_MB):       {eval_mb['top1']:.3f}")
    print(f"Top1(Q_hybrid):   {eval_hybrid['top1']:.3f}")

    # ============================================================
    # Uncertainty calibration
    # ============================================================
    abs_error = np.abs(q_total_test - y_test)

    # Bin by sigma_Q
    sigma_bins = np.percentile(sigma_res_test, [0, 20, 40, 60, 80, 100])
    print(f"\nUncertainty calibration (sigma_Q bins):")
    print(f"{'Bin':>10} {'Mean sigma':>12} {'Mean abs error':>15} {'Mean regret':>12} {'Top-1 error':>12}")

    bin_data = []
    for b in range(5):
        mask = (sigma_res_test >= sigma_bins[b]) & (sigma_res_test < sigma_bins[b + 1] + 1e-9)
        if mask.sum() == 0:
            continue
        bin_sigma = sigma_res_test[mask].mean()
        bin_error = abs_error[mask].mean()

        # Compute regret for this bin
        bin_records = [r for r, m in zip(test_records, mask) if m]
        bin_q = q_total_test[mask]
        bin_eval = evaluate_groupwise(bin_records, bin_q)
        bin_regret = bin_eval["regret"]
        bin_top1_err = 1.0 - bin_eval["top1"]

        print(f"Q{b+1} {'':>5} {bin_sigma:>12.4f} {bin_error:>15.4f} {bin_regret:>12.4f} {bin_top1_err:>12.3f}")
        bin_data.append({
            "bin": f"Q{b+1}",
            "mean_sigma": float(bin_sigma),
            "mean_abs_error": float(bin_error),
            "mean_regret": float(bin_regret),
            "top1_error": float(bin_top1_err),
        })

    # Correlation: sigma vs abs error
    corr_sigma_error = np.corrcoef(sigma_res_test, abs_error)[0, 1]
    print(f"\nCorrelation(sigma_Q, |error|): {corr_sigma_error:.4f}")

    # Check monotonicity
    sigmas = [d["mean_sigma"] for d in bin_data]
    errors = [d["mean_abs_error"] for d in bin_data]
    regrets = [d["mean_regret"] for d in bin_data]
    is_monotonic_error = all(errors[i] <= errors[i+1] for i in range(len(errors)-1))
    is_monotonic_regret = all(regrets[i] <= regrets[i+1] for i in range(len(regrets)-1))
    print(f"Monotonic sigma -> error: {is_monotonic_error}")
    print(f"Monotonic sigma -> regret: {is_monotonic_regret}")

    # ============================================================
    # Pairwise sigma_Delta (for authority decisions)
    # ============================================================
    print(f"\nPairwise sigma_Delta analysis:")
    # For each test group, compute sigma_Delta between best and second-best action
    groups = defaultdict(list)
    for i, r in enumerate(test_records):
        groups[r["counterfactual_group_id"]].append((i, r))

    sigma_deltas = []
    correct_preferences = 0
    total_preferences = 0

    for gid, group in groups.items():
        if len(group) < 2:
            continue
        # Get predictions and uncertainties for this group
        indices = [i for i, _ in group]
        group_mu = q_total_test[indices]
        group_sigma = sigma_res_test[indices]

        # Sort by predicted value
        sorted_idx = np.argsort(-group_mu)  # Descending
        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1]

        # sigma_Delta between best and second-best
        sigma_delta = np.sqrt(group_sigma[best_idx]**2 + group_sigma[second_idx]**2)
        sigma_deltas.append(sigma_delta)

        # Check if the preference is correct
        true_utilities = [r["utility"] for _, r in group]
        true_best_idx = np.argmax(true_utilities)
        if best_idx == true_best_idx:
            correct_preferences += 1
        total_preferences += 1

    if sigma_deltas:
        sigma_deltas = np.array(sigma_deltas)
        print(f"  Mean sigma_Delta: {sigma_deltas.mean():.4f}")
        print(f"  Correct preference rate: {correct_preferences}/{total_preferences} = {correct_preferences/max(total_preferences,1):.3f}")

        # Bin by sigma_Delta
        sd_bins = np.percentile(sigma_deltas, [0, 25, 50, 75, 100])
        print(f"\n  sigma_Delta calibration:")
        for b in range(4):
            mask = (sigma_deltas >= sd_bins[b]) & (sigma_deltas < sd_bins[b + 1] + 1e-9)
            if mask.sum() == 0:
                continue
            # Compute correct preference rate in this bin
            bin_correct = 0
            bin_total = 0
            for gid2, group2 in groups.items():
                if len(group2) < 2:
                    continue
                indices2 = [i for i, _ in group2]
                group_mu2 = q_total_test[indices2]
                group_sigma2 = sigma_res_test[indices2]
                sorted_idx2 = np.argsort(-group_mu2)
                best2 = sorted_idx2[0]
                second2 = sorted_idx2[1]
                sd = np.sqrt(group_sigma2[best2]**2 + group_sigma2[second2]**2)
                if sd_bins[b] <= sd < sd_bins[b + 1] + 1e-9:
                    true_utils = [r["utility"] for _, r in group2]
                    true_best = np.argmax(true_utils)
                    if best2 == true_best:
                        bin_correct += 1
                    bin_total += 1
            if bin_total > 0:
                print(f"    Bin {b+1}: sigma_Delta=[{sd_bins[b]:.4f},{sd_bins[b+1]:.4f}], "
                      f"correct={bin_correct}/{bin_total}={bin_correct/bin_total:.3f}")

    # LCB coverage
    print(f"\nLCB coverage analysis:")
    z_values = [0.5, 1.0, 1.5, 2.0]
    for z in z_values:
        lcb = q_total_test - z * sigma_res_test
        # LCB coverage: P(true_utility >= LCB)
        coverage = np.mean(y_test >= lcb)
        print(f"  z={z}: coverage={coverage:.3f}")

    # Save results
    results = {
        "k_ensemble": K_ENSEMBLE,
        "mae_mb": float(mae_mb),
        "mae_hybrid": float(mae_hybrid),
        "regret_mb": float(eval_mb["regret"]),
        "regret_hybrid": float(eval_hybrid["regret"]),
        "top1_mb": float(eval_mb["top1"]),
        "top1_hybrid": float(eval_hybrid["top1"]),
        "sigma_calibration": bin_data,
        "corr_sigma_error": float(corr_sigma_error),
        "monotonic_sigma_error": bool(is_monotonic_error),
        "monotonic_sigma_regret": bool(is_monotonic_regret),
        "sigma_delta_mean": float(sigma_deltas.mean()) if len(sigma_deltas) > 0 else 0.0,
        "correct_preference_rate": float(correct_preferences / max(total_preferences, 1)),
        "lcb_coverage": {str(z): float(np.mean(y_test >= q_total_test - z * sigma_res_test)) for z in z_values},
    }

    output_path = REPO_ROOT / "experiments/daph_x/m3_structural/uncertainty_calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
