#!/usr/bin/env python3
"""M3.7 fix: Conformal calibration for ΔQ (pairwise value difference).

Instead of using raw ensemble variance as a confidence interval,
use conformal calibration on the ΔQ residuals.

For each causal pair (a_X, a_L):
  ΔQ_pred = Q(a_X) - Q(a_L)
  ΔU_true = U(a_X) - U(a_L)
  residual = ΔQ_pred - ΔU_true

The LCB is:
  LCB_Δ = ΔQ_pred - q_{1-α}(residuals)

where q_{1-α} is the empirical quantile of residuals on a calibration split.

Usage:
    python scripts/conformal_calibration.py
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


def load_corpus(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def train_model(X, y, seed=42):
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed,
    )
    model.fit(X, y)
    return model


def compute_delta_q_pairs(records, q_total, group_key="counterfactual_group_id"):
    """Compute ΔQ pairs for each counterfactual group.

    For each group, find the best action (by Q) and compare against
    the second-best action. Also compare against DEFER (base policy).

    Returns list of (delta_q_pred, delta_u_true, sigma_delta) tuples.
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r[group_key]].append((i, r))

    pairs = []
    for gid, group in groups.items():
        if len(group) < 2:
            continue

        # Get Q estimates for all actions in this group
        indices = [i for i, _ in group]
        group_q = q_total[indices]

        # Sort by Q (descending)
        sorted_idx = np.argsort(-group_q)
        best_idx = sorted_idx[0]

        # Compare best against each other action
        for j in range(1, len(sorted_idx)):
            other_idx = sorted_idx[j]

            delta_q_pred = group_q[best_idx] - group_q[other_idx]

            # True ΔU
            true_best = group[best_idx][1]["utility"]
            true_other = group[other_idx][1]["utility"]
            delta_u_true = true_best - true_other

            # Sigma for this pair (placeholder — would use ensemble)
            sigma_delta = 0.1  # Placeholder

            pairs.append({
                "group_id": gid,
                "best_action": group[best_idx][1]["action_id"],
                "other_action": group[other_idx][1]["action_id"],
                "delta_q_pred": float(delta_q_pred),
                "delta_u_true": float(delta_u_true),
                "residual": float(delta_q_pred - delta_u_true),
                "sigma_delta": sigma_delta,
            })

    return pairs


def conformal_calibrate(residuals, alpha=0.05):
    """Compute conformal calibration quantile.

    Returns the (1-α) quantile of absolute residuals.
    """
    abs_residuals = np.abs(residuals)
    q = np.quantile(abs_residuals, 1 - alpha)
    return q


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

    # Train single model (for ΔQ estimation)
    model = train_model(X_train, y_res_train)
    q_res_pred_test = model.predict(X_test)
    q_total_test = q_mb_test + q_res_pred_test

    # Split test into calibration and evaluation
    np.random.seed(42)
    test_group_ids = list(set(r["counterfactual_group_id"] for r in test_records))
    np.random.shuffle(test_group_ids)
    split = int(0.5 * len(test_group_ids))
    calib_groups = set(test_group_ids[:split])
    eval_groups = set(test_group_ids[split:])

    calib_records = [r for r in test_records if r["counterfactual_group_id"] in calib_groups]
    eval_records = [r for r in test_records if r["counterfactual_group_id"] in eval_groups]
    calib_mask = np.array([r["counterfactual_group_id"] in calib_groups for r in test_records])
    eval_mask = np.array([r["counterfactual_group_id"] in eval_groups for r in test_records])

    q_total_calib = q_total_test[calib_mask]
    q_total_eval = q_total_test[eval_mask]

    print(f"\nCalibration: {len(calib_records)} records, {len(calib_groups)} groups")
    print(f"Evaluation: {len(eval_records)} records, {len(eval_groups)} groups")

    # Compute ΔQ pairs for calibration
    calib_pairs = compute_delta_q_pairs(calib_records, q_total_calib)
    print(f"\nCalibration pairs: {len(calib_pairs)}")

    # Compute residuals
    residuals = np.array([p["residual"] for p in calib_pairs])
    print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

    # Conformal calibration
    alpha_levels = [0.10, 0.05, 0.01]
    quantiles = {}
    for alpha in alpha_levels:
        q = conformal_calibrate(residuals, alpha=alpha)
        quantiles[alpha] = q
        print(f"  q_{1-alpha:.2f} = {q:.4f}")

    # Evaluate on held-out evaluation set
    eval_pairs = compute_delta_q_pairs(eval_records, q_total_eval)
    print(f"\nEvaluation pairs: {len(eval_pairs)}")

    print(f"\n{'='*60}")
    print(f"CONFORMAL CALIBRATION RESULTS")
    print(f"{'='*60}")

    for alpha in alpha_levels:
        q = quantiles[alpha]
        # LCB_Δ = ΔQ_pred - q
        lcb = np.array([p["delta_q_pred"] - q for p in eval_pairs])

        # Coverage: P(ΔU_true >= LCB_Δ)
        coverage = np.mean([p["delta_u_true"] >= lcb[i] for i, p in enumerate(eval_pairs)])

        # Empirical coverage should be >= 1 - alpha
        print(f"  α={alpha}: nominal={1-alpha:.2f}, empirical coverage={coverage:.3f}")

    # Authority decision analysis
    print(f"\nAuthority decision analysis:")
    for alpha in alpha_levels:
        q = quantiles[alpha]
        # FORCE if LCB_Δ > 0
        force_decisions = [p["delta_q_pred"] - q > 0 for p in eval_pairs]
        n_force = sum(force_decisions)
        # Check if FORCE is correct (ΔU_true > 0)
        correct_force = sum(
            1 for i, p in enumerate(eval_pairs)
            if force_decisions[i] and p["delta_u_true"] > 0
        )
        # Harmful FORCE (LCB > 0 but ΔU < 0)
        harmful_force = sum(
            1 for i, p in enumerate(eval_pairs)
            if force_decisions[i] and p["delta_u_true"] < 0
        )
        print(f"  α={alpha}: FORCE={n_force}/{len(eval_pairs)}, "
              f"correct={correct_force}, harmful={harmful_force}, "
              f"precision={correct_force/max(n_force,1):.3f}")

    # Save
    results = {
        "calibration_pairs": len(calib_pairs),
        "evaluation_pairs": len(eval_pairs),
        "quantiles": {str(a): float(q) for a, q in quantiles.items()},
        "coverage": {
            str(a): float(np.mean([p["delta_u_true"] >= p["delta_q_pred"] - quantiles[a]
                                   for p in eval_pairs]))
            for a in alpha_levels
        },
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
    }
    output_path = REPO_ROOT / "experiments/daph_x/m3_structural/conformal_calibration.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
