#!/usr/bin/env python3
"""M3.8: Calibrated Intervention Risk Model.

Estimates P(harm) — the probability that overriding the base policy
makes things worse.

Usage:
    python scripts/intervention_risk.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

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


def build_intervention_dataset(records):
    """Build intervention risk dataset from counterfactual groups.

    For each group, the executive-preferred action is the one with
    highest Q_hybrid. The base policy (LLM) proposes a different action.

    The intervention is harmful if:
      U(executive_action) < U(base_action)

    We simulate the base policy as always choosing DEFER (the safe default).
    This creates a meaningful intervention risk estimation problem:
    - ANSWER(correct) is better than DEFER → beneficial override
    - ANSWER(wrong) is worse than DEFER → harmful override
    - DEFER(correct) is same as DEFER → no intervention needed
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["counterfactual_group_id"]].append((i, r))

    intervention_records = []

    for gid, group in groups.items():
        if len(group) < 2:
            continue

        # Find the executive-preferred action using MODEL-BASED Q estimate
        # (not actual utility — the executive doesn't know the true utility)
        # group is list of (index_in_records, record) tuples
        q_estimates = [(j, compute_q_mb(group[j][1])) for j in range(len(group))]
        best_pos = max(range(len(group)), key=lambda j: q_estimates[j][1])
        best_record = group[best_pos][1]
        best_utility = best_record["utility"]  # Actual utility (for evaluation)

        # Simulate base policy: always DEFER
        defer_records = [r for _, r in group if r["action_type"] == "DEFER"]
        if not defer_records:
            continue
        base_record = defer_records[0]

        # Intervention advantage
        delta_u = best_utility - base_record["utility"]

        # Harmful if executive action is worse than base
        is_harmful = 1 if delta_u < 0 else 0

        # Features for the intervention risk model
        # ONLY pre-decision features — no post-hoc utility or outcome
        features = extract_features(best_record, {})
        # Add intervention-specific features (pre-decision only)
        features["executive_action"] = best_record["action_type"]
        features["base_action"] = base_record["action_type"]
        features["topology_n_supported"] = best_record.get("topo_n_supported", 0)
        features["topology_n_contradicted"] = best_record.get("topo_n_contradicted", 0)
        features["topology_has_competition"] = 1.0 if best_record.get("topo_has_competition") else 0.0
        features["topology_unverified_exists"] = 1.0 if best_record.get("topo_unverified_exists") else 0.0
        # Predicted delta Q (from model, not actual utility)
        features["pred_delta_q"] = compute_q_mb(best_record) - compute_q_mb(base_record)
        # Model uncertainty proxy (ensemble spread would go here)
        features["sigma_q"] = 0.1  # Placeholder

        intervention_records.append({
            "group_id": gid,
            "features": features,
            "is_harmful": is_harmful,
            "delta_u": delta_u,
        })

    return intervention_records


def main():
    train_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_train.jsonl")
    test_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_test.jsonl")

    # Build intervention dataset
    train_interventions = build_intervention_dataset(train_records)
    test_interventions = build_intervention_dataset(test_records)

    print(f"Train interventions: {len(train_interventions)}")
    print(f"Test interventions: {len(test_interventions)}")

    # Count harmful vs beneficial
    train_harmful = sum(1 for r in train_interventions if r["is_harmful"])
    test_harmful = sum(1 for r in test_interventions if r["is_harmful"])
    print(f"Train harmful: {train_harmful}/{len(train_interventions)} ({100*train_harmful/max(1,len(train_interventions)):.1f}%)")
    print(f"Test harmful: {test_harmful}/{len(test_interventions)} ({100*test_harmful/max(1,len(test_interventions)):.1f}%)")

    if train_harmful == 0 or test_harmful == 0:
        print("\nWARNING: No harmful interventions in one split. "
              "The intervention risk model cannot be trained without negative examples.")
        print("This is expected when the executive is always correct on the synthetic benchmark.")
        print("For real deployment, you need disagreement cases where the executive is wrong.")

        # Still build a model that always predicts "safe"
        print("\nBuilding trivial model (always safe)...")
        results = {
            "train_interventions": len(train_interventions),
            "test_interventions": len(test_interventions),
            "train_harmful": train_harmful,
            "test_harmful": test_harmful,
            "note": "No harmful interventions — executive is always correct on synthetic benchmark",
        }
        output_path = REPO_ROOT / "experiments/daph_x/m3_structural/intervention_risk.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")
        return

    # Build feature matrix — encode strings as numeric
    all_feature_keys = set()
    for r in train_interventions + test_interventions:
        all_feature_keys.update(r["features"].keys())
    feature_keys = sorted(all_feature_keys)

    # Separate numeric and string features
    numeric_keys = []
    string_keys = []
    for k in feature_keys:
        # Check if any value is a string
        is_string = any(isinstance(r["features"].get(k), str) for r in train_interventions[:5])
        if is_string:
            string_keys.append(k)
        else:
            numeric_keys.append(k)

    # One-hot encode string features
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for k in string_keys:
        all_values = list(set(r["features"].get(k, "") for r in train_interventions + test_interventions))
        encoders[k] = LabelEncoder().fit(all_values)

    def encode_features(interventions):
        rows = []
        for r in interventions:
            row = []
            for k in numeric_keys:
                row.append(float(r["features"].get(k, 0.0)))
            for k in string_keys:
                val = r["features"].get(k, "")
                # One-hot: use the encoded value as a feature
                encoded = encoders[k].transform([val])[0]
                # Add as one-hot columns
                for v in range(len(encoders[k].classes_)):
                    row.append(1.0 if encoded == v else 0.0)
            rows.append(row)
        return np.array(rows)

    X_train = encode_features(train_interventions)
    X_test = encode_features(test_interventions)
    y_train = np.array([r["is_harmful"] for r in train_interventions])
    y_test = np.array([r["is_harmful"] for r in test_interventions])

    print(f"Feature matrix: {X_train.shape[1]} features ({len(numeric_keys)} numeric + {sum(len(encoders[k].classes_) for k in string_keys)} one-hot)")

    # Train
    print(f"\nTraining GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict
    p_harm_train = model.predict_proba(X_train)[:, 1]
    p_harm_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    brier_train = brier_score_loss(y_train, p_harm_train)
    brier_test = brier_score_loss(y_test, p_harm_test)

    try:
        auroc = roc_auc_score(y_test, p_harm_test)
    except ValueError:
        auroc = float('nan')

    print(f"\n{'='*60}")
    print(f"M3.8 INTERVENTION RISK MODEL")
    print(f"{'='*60}")
    print(f"Brier score (train): {brier_train:.4f}")
    print(f"Brier score (test):  {brier_test:.4f}")
    print(f"AUROC (test):        {auroc:.4f}")

    # Calibration bins
    print(f"\nCalibration (test):")
    bins = np.percentile(p_harm_test, [0, 25, 50, 75, 100])
    for b in range(4):
        mask = (p_harm_test >= bins[b]) & (p_harm_test < bins[b + 1] + 1e-9)
        if mask.sum() == 0:
            continue
        mean_pred = p_harm_test[mask].mean()
        mean_actual = y_test[mask].mean()
        print(f"  Bin {b+1}: pred={mean_pred:.3f}, actual={mean_actual:.3f}, n={mask.sum()}")

    # Harm FNR: P(predicted safe | actually harmful)
    threshold = 0.5
    predicted_safe = p_harm_test < threshold
    actually_harmful = y_test == 1
    if actually_harmful.sum() > 0:
        fnr_harm = np.mean(predicted_safe[actually_harmful])
    else:
        fnr_harm = 0.0
    print(f"\nHarm FNR (threshold={threshold}): {fnr_harm:.4f}")

    # Save
    results = {
        "train_interventions": len(train_interventions),
        "test_interventions": len(test_interventions),
        "train_harmful": train_harmful,
        "test_harmful": test_harmful,
        "brier_train": float(brier_train),
        "brier_test": float(brier_test),
        "auroc_test": float(auroc) if not np.isnan(auroc) else None,
        "harm_fnr": float(fnr_harm),
        "feature_keys": feature_keys,
    }
    output_path = REPO_ROOT / "experiments/daph_x/m3_structural/intervention_risk.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
