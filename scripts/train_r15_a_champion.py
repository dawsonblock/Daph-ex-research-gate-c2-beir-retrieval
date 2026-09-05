#!/usr/bin/env python3
"""R15-A: Train models on development set, select champion, freeze it.

Development set: R13 checkpoints (90 cp, 81 unique tasks) with R14-C outcomes.
Models:
  A: Single threshold on best feature
  B: Logistic regression for P(STOP wrong | s)
  C: Ridge regression for ΔJ_COT(s) = Q_COT - Q_STOP - λ·L_COT

Champion selected via task-grouped 5-fold CV, optimizing J_λ at λ=0.01.
Champion refit on all dev data, coefficients/threshold/hash frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

R14_DIR = PROJECT_ROOT / "experiments/daph_x/r14"
R13_CHECKPOINTS = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
R15_DIR = PROJECT_ROOT / "experiments/daph_x/r15"

# Frozen feature set (no uncertainty_ema)
NUMERIC_FEATURES = [
    "k", "p_top1", "p_top2", "margin", "entropy",
    "n_unique_answers", "agreement_rate",
    "uncertainty_current", "uncertainty_delta",
    "margin_delta", "answer_changed", "stable_prefix_count",
]
CATEGORICAL_FEATURES = ["difficulty", "category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LAMBDA_J = 0.01  # λ for J_λ used in champion selection


def load_dev_data():
    """Load development set: R13 checkpoints with R14-C outcomes."""
    # Load R14-C executions (seed 42, since all identical)
    with open(R14_DIR / "r14_c_executions.jsonl") as f:
        results = [json.loads(l) for l in f]

    by_cp = defaultdict(dict)
    for r in results:
        if r["seed"] != 42:
            continue
        op = r.get("operator_id_canonical", r["operator_id"])
        by_cp[r["checkpoint_id"]][op] = r

    # Load R13 checkpoints for features
    with open(R13_CHECKPOINTS) as f:
        checkpoints = {cp["checkpoint_id"]: cp for cp in (json.loads(l) for l in f)}

    # Build per-checkpoint records
    records = []
    for cp_id, cp in checkpoints.items():
        ops = by_cp.get(cp_id, {})
        if "STOP" not in ops or "OPT_COT_REFLECT" not in ops:
            continue
        of = cp["runtime_state"].get("observable_features", {})
        stop_correct = 1 if ops["STOP"].get("correct") else 0
        cot_correct = 1 if ops["OPT_COT_REFLECT"].get("correct") else 0
        # 3-run mean latency
        cot_walls = [r.get("wall_ms_observed", 0) for r in results
                     if r.get("operator_id_canonical", r["operator_id"]) == "OPT_COT_REFLECT"
                     and r["checkpoint_id"] == cp_id]
        cot_lat_s = sum(cot_walls) / len(cot_walls) / 1000.0 if cot_walls else 0.0

        record = {
            "checkpoint_id": cp_id,
            "task_id": cp["runtime_state"]["task_id"],
            "stop_correct": stop_correct,
            "cot_correct": cot_correct,
            "cot_lat_s": cot_lat_s,
            "delta_j": cot_correct - stop_correct - LAMBDA_J * cot_lat_s,
            "features": {f: of.get(f) for f in NUMERIC_FEATURES},
            "difficulty": cp["runtime_state"].get("difficulty", "unknown"),
            "category": cp["runtime_state"].get("category", "unknown"),
        }
        records.append(record)

    assert len(records) == 90, f"Expected 90 dev records, got {len(records)}"
    return records


def build_feature_matrix(records):
    """Build feature matrix with standardization for numeric, one-hot for categorical."""
    # Numeric features
    X_num = np.array([[r["features"].get(f, 0.0) for f in NUMERIC_FEATURES] for r in records])

    # Categorical one-hot
    difficulties = sorted(set(r["difficulty"] for r in records))
    categories = sorted(set(r["category"] for r in records))
    X_cat = []
    for r in records:
        row = []
        for d in difficulties:
            row.append(1.0 if r["difficulty"] == d else 0.0)
        for c in categories:
            row.append(1.0 if r["category"] == c else 0.0)
        X_cat.append(row)
    X_cat = np.array(X_cat)

    X = np.hstack([X_num, X_cat])
    feature_names = NUMERIC_FEATURES + [f"diff_{d}" for d in difficulties] + [f"cat_{c}" for c in categories]
    return X, feature_names, difficulties, categories


def compute_j_lambda(y_stop, y_cot, lat_cot, escalate_mask, lam=LAMBDA_J):
    """Compute J_λ for a routing policy: escalate if escalate_mask, else STOP."""
    # For escalated: Q = cot_correct, L = cot_lat
    # For non-escalated: Q = stop_correct, L = 0
    q = np.where(escalate_mask, y_cot, y_stop)
    l = np.where(escalate_mask, lat_cot, 0.0)
    return np.mean(q - lam * l)


def evaluate_threshold(y_stop, y_cot, lat_cot, feature_vals, lam=LAMBDA_J):
    """Evaluate all thresholds on a single feature. Returns best J_λ and threshold."""
    # Direction: for p_top1, margin, agreement_rate: escalate if BELOW
    # for entropy, uncertainty_current, n_unique: escalate if ABOVE
    best_j = -999
    best_t = None
    best_n_esc = 0
    unique_vals = sorted(set(v for v in feature_vals if v is not None))
    for t in unique_vals:
        # Try both directions
        for direction in ["below", "above"]:
            if direction == "below":
                mask = np.array([v < t if v is not None else False for v in feature_vals])
            else:
                mask = np.array([v > t if v is not None else False for v in feature_vals])
            j = compute_j_lambda(y_stop, y_cot, lat_cot, mask, lam)
            if j > best_j:
                best_j = j
                best_t = (t, direction)
                best_n_esc = mask.sum()
    return best_j, best_t, best_n_esc


def task_grouped_cv(records, n_splits=5):
    """Task-grouped K-fold. Returns list of (train_idx, test_idx)."""
    task_ids = [r["task_id"] for r in records]
    groups = np.array(task_ids)
    unique_tasks = sorted(set(task_ids))
    # Map task to group index
    task_to_group = {t: i % n_splits for i, t in enumerate(unique_tasks)}
    group_labels = np.array([task_to_group[t] for t in task_ids])
    folds = []
    for i in range(n_splits):
        test_idx = np.where(group_labels == i)[0]
        train_idx = np.where(group_labels != i)[0]
        folds.append((train_idx, test_idx))
    return folds


def train_model_a_threshold(records, folds):
    """Model A: single threshold on best feature."""
    y_stop = np.array([r["stop_correct"] for r in records])
    y_cot = np.array([r["cot_correct"] for r in records])
    lat_cot = np.array([r["cot_lat_s"] for r in records])

    feature_candidates = ["p_top1", "agreement_rate", "entropy", "margin", "uncertainty_current"]

    best_cv_j = -999
    best_feat = None
    best_threshold = None

    for feat in feature_candidates:
        feat_vals = np.array([r["features"].get(feat, 0.0) for r in records])
        cv_js = []
        for train_idx, test_idx in folds:
            # Train: find best threshold on train
            train_vals = feat_vals[train_idx]
            train_j, train_t, _ = evaluate_threshold(
                y_stop[train_idx], y_cot[train_idx], lat_cot[train_idx], train_vals
            )
            # Test: apply threshold
            t, direction = train_t
            if direction == "below":
                test_mask = feat_vals[test_idx] < t
            else:
                test_mask = feat_vals[test_idx] > t
            test_j = compute_j_lambda(y_stop[test_idx], y_cot[test_idx], lat_cot[test_idx], test_mask)
            cv_js.append(test_j)
        mean_cv_j = np.mean(cv_js)
        if mean_cv_j > best_cv_j:
            best_cv_j = mean_cv_j
            best_feat = feat

    # Refit on all data
    feat_vals = np.array([r["features"].get(best_feat, 0.0) for r in records])
    _, best_t, best_n_esc = evaluate_threshold(y_stop, y_cot, lat_cot, feat_vals)

    return {
        "model": "A_threshold",
        "feature": best_feat,
        "threshold": best_t[0],
        "direction": best_t[1],
        "cv_j_lambda": best_cv_j,
        "n_escalated_full": best_n_esc,
    }


def train_model_b_logistic(records, folds, X, feature_names):
    """Model B: logistic regression for P(STOP wrong | s)."""
    y_stop = np.array([r["stop_correct"] for r in records])
    y_cot = np.array([r["cot_correct"] for r in records])
    lat_cot = np.array([r["cot_lat_s"] for r in records])
    y_stop_wrong = 1 - y_stop

    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_cv_j = -999
    best_C = None
    best_threshold = None

    for C in C_grid:
        cv_js = []
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
            model.fit(X_train, y_stop_wrong[train_idx])
            p_test = model.predict_proba(X_test)[:, 1]
            # Find best threshold on train
            p_train = model.predict_proba(X_train)[:, 1]
            best_train_j = -999
            best_train_t = 0.5
            for t in np.linspace(0.01, 0.99, 99):
                mask = p_train > t
                j = compute_j_lambda(y_stop[train_idx], y_cot[train_idx], lat_cot[train_idx], mask)
                if j > best_train_j:
                    best_train_j = j
                    best_train_t = t
            # Apply to test
            test_mask = p_test > best_train_t
            test_j = compute_j_lambda(y_stop[test_idx], y_cot[test_idx], lat_cot[test_idx], test_mask)
            cv_js.append(test_j)
        mean_cv_j = np.mean(cv_js)
        if mean_cv_j > best_cv_j:
            best_cv_j = mean_cv_j
            best_C = C

    # Refit on all data with best C
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y_stop_wrong)
    p_all = model.predict_proba(X_scaled)[:, 1]
    # Find best threshold on full data
    best_j = -999
    best_t = 0.5
    for t in np.linspace(0.01, 0.99, 99):
        mask = p_all > t
        j = compute_j_lambda(y_stop, y_cot, lat_cot, mask)
        if j > best_j:
            best_j = j
            best_t = t

    return {
        "model": "B_logistic",
        "C": best_C,
        "threshold": best_t,
        "cv_j_lambda": best_cv_j,
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
        "n_escalated_full": int((p_all > best_t).sum()),
    }


def train_model_c_ridge(records, folds, X, feature_names):
    """Model C: ridge regression for ΔJ_COT(s)."""
    y_stop = np.array([r["stop_correct"] for r in records])
    y_cot = np.array([r["cot_correct"] for r in records])
    lat_cot = np.array([r["cot_lat_s"] for r in records])
    delta_j = y_cot - y_stop - LAMBDA_J * lat_cot

    alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_cv_j = -999
    best_alpha = None

    for alpha in alpha_grid:
        cv_js = []
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = Ridge(alpha=alpha)
            model.fit(X_train, delta_j[train_idx])
            pred_test = model.predict(X_test)
            # Escalate if predicted ΔJ > 0
            test_mask = pred_test > 0
            test_j = compute_j_lambda(y_stop[test_idx], y_cot[test_idx], lat_cot[test_idx], test_mask)
            cv_js.append(test_j)
        mean_cv_j = np.mean(cv_js)
        if mean_cv_j > best_cv_j:
            best_cv_j = mean_cv_j
            best_alpha = alpha

    # Refit on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=best_alpha)
    model.fit(X_scaled, delta_j)
    pred_all = model.predict(X_scaled)
    mask_all = pred_all > 0

    return {
        "model": "C_ridge_deltaJ",
        "alpha": best_alpha,
        "cv_j_lambda": best_cv_j,
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
        "lambda_j": LAMBDA_J,
        "n_escalated_full": int(mask_all.sum()),
    }


def main():
    print("R15-A: Training models on development set (90 cp, 81 tasks)")
    print("=" * 70)
    records = load_dev_data()
    print(f"Loaded {len(records)} dev records")
    print(f"STOP correct: {sum(r['stop_correct'] for r in records)}/90")
    print(f"COT correct: {sum(r['cot_correct'] for r in records)}/90")
    print(f"STOP wrong: {sum(1 - r['stop_correct'] for r in records)}/90")
    print()

    X, feature_names, difficulties, categories = build_feature_matrix(records)
    print(f"Feature matrix: {X.shape}")
    print(f"Features: {feature_names}")
    print()

    folds = task_grouped_cv(records, n_splits=5)
    print(f"Task-grouped CV: {len(folds)} folds")
    for i, (train_idx, test_idx) in enumerate(folds):
        train_tasks = set(records[j]["task_id"] for j in train_idx)
        test_tasks = set(records[j]["task_id"] for j in test_idx)
        print(f"  Fold {i}: train={len(train_idx)} cp ({len(train_tasks)} tasks), test={len(test_idx)} cp ({len(test_tasks)} tasks)")
    print()

    # Train all models
    print("Training Model A (single threshold)...")
    model_a = train_model_a_threshold(records, folds)
    print(f"  Feature: {model_a['feature']}")
    print(f"  Threshold: {model_a['threshold']} {model_a['direction']}")
    print(f"  CV J_λ: {model_a['cv_j_lambda']:.4f}")
    print(f"  N escalated (full): {model_a['n_escalated_full']}")
    print()

    print("Training Model B (logistic P(STOP wrong))...")
    model_b = train_model_b_logistic(records, folds, X, feature_names)
    print(f"  C: {model_b['C']}")
    print(f"  Threshold: {model_b['threshold']:.4f}")
    print(f"  CV J_λ: {model_b['cv_j_lambda']:.4f}")
    print(f"  N escalated (full): {model_b['n_escalated_full']}")
    print()

    print("Training Model C (ridge ΔJ)...")
    model_c = train_model_c_ridge(records, folds, X, feature_names)
    print(f"  Alpha: {model_c['alpha']}")
    print(f"  CV J_λ: {model_c['cv_j_lambda']:.4f}")
    print(f"  N escalated (full): {model_c['n_escalated_full']}")
    print()

    # Select champion
    models = [("A", model_a), ("B", model_b), ("C", model_c)]
    champion_name, champion = max(models, key=lambda x: x[1]["cv_j_lambda"])
    print(f"Champion: Model {champion_name} ({champion['model']})")
    print(f"  CV J_λ = {champion['cv_j_lambda']:.4f}")
    print()

    # Compute full-data performance for champion
    y_stop = np.array([r["stop_correct"] for r in records])
    y_cot = np.array([r["cot_correct"] for r in records])
    lat_cot = np.array([r["cot_lat_s"] for r in records])

    if champion_name == "A":
        feat_vals = np.array([r["features"].get(champion["feature"], 0.0) for r in records])
        if champion["direction"] == "below":
            mask = feat_vals < champion["threshold"]
        else:
            mask = feat_vals > champion["threshold"]
    elif champion_name == "B":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(C=champion["C"], max_iter=1000, solver="lbfgs")
        model.fit(X_scaled, 1 - y_stop)
        p = model.predict_proba(X_scaled)[:, 1]
        mask = p > champion["threshold"]
    else:  # C
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        delta_j = y_cot - y_stop - LAMBDA_J * lat_cot
        model = Ridge(alpha=champion["alpha"])
        model.fit(X_scaled, delta_j)
        pred = model.predict(X_scaled)
        mask = pred > 0

    acc = np.mean(np.where(mask, y_cot, y_stop))
    lat = np.mean(np.where(mask, lat_cot, 0.0))
    n_esc = mask.sum()
    print(f"Champion full-data performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Mean latency: {lat:.3f}s")
    print(f"  N escalated: {n_esc}/90")
    print(f"  Always-COT accuracy: {np.mean(y_cot):.4f}")
    print(f"  Always-COT latency: {np.mean(lat_cot):.3f}s")
    print(f"  J_λ: {compute_j_lambda(y_stop, y_cot, lat_cot, mask):.4f}")
    print()

    # Compute hash of frozen champion
    champion_str = json.dumps(champion, sort_keys=True)
    champion_hash = hashlib.sha256(champion_str.encode()).hexdigest()
    champion["champion_hash"] = champion_hash
    champion["difficulties"] = difficulties
    champion["categories"] = categories

    # Save champion
    champion_path = R15_DIR / "r15_a_champion.json"
    with open(champion_path, "w") as f:
        json.dump(champion, f, indent=2)
    print(f"Champion frozen to {champion_path}")
    print(f"Champion hash: {champion_hash}")


if __name__ == "__main__":
    main()
