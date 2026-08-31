#!/usr/bin/env python3
"""M3.8 fix: Feature ablation and leakage tests.

Tests whether the intervention risk model has learned template-specific
features rather than abstract intervention risk.

Usage:
    python scripts/intervention_risk_ablation.py
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

from scripts.intervention_risk import load_corpus, build_intervention_dataset


def main():
    train_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_train.jsonl")
    test_records = load_corpus(REPO_ROOT / "experiments/daph_x/m3_structural/m3_test.jsonl")

    train_interventions = build_intervention_dataset(train_records)
    test_interventions = build_intervention_dataset(test_records)

    print(f"Train: {len(train_interventions)} interventions, {sum(1 for r in train_interventions if r['is_harmful'])} harmful")
    print(f"Test: {len(test_interventions)} interventions, {sum(1 for r in test_interventions if r['is_harmful'])} harmful")

    # Build feature matrix
    all_keys = set()
    for r in train_interventions + test_interventions:
        all_keys.update(r["features"].keys())
    feature_keys = sorted(all_keys)

    # Separate numeric and string features
    numeric_keys = []
    string_keys = []
    for k in feature_keys:
        is_string = any(isinstance(r["features"].get(k), str) for r in train_interventions[:5])
        if is_string:
            string_keys.append(k)
        else:
            numeric_keys.append(k)

    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for k in string_keys:
        all_values = list(set(r["features"].get(k, "") for r in train_interventions + test_interventions))
        encoders[k] = LabelEncoder().fit(all_values)

    def encode(interventions, exclude_keys=None):
        if exclude_keys is None:
            exclude_keys = set()
        rows = []
        for r in interventions:
            row = []
            for k in numeric_keys:
                if k in exclude_keys:
                    continue
                row.append(float(r["features"].get(k, 0.0)))
            for k in string_keys:
                if k in exclude_keys:
                    continue
                val = r["features"].get(k, "")
                encoded = encoders[k].transform([val])[0]
                for v in range(len(encoders[k].classes_)):
                    row.append(1.0 if encoded == v else 0.0)
            rows.append(row)
        return np.array(rows)

    y_train = np.array([r["is_harmful"] for r in train_interventions])
    y_test = np.array([r["is_harmful"] for r in test_interventions])

    # Baseline: all features
    X_train_full = encode(train_interventions)
    X_test_full = encode(test_interventions)

    print(f"\n{'='*70}")
    print(f"FEATURE ABLATION TESTS")
    print(f"{'='*70}")
    print(f"Total features: {X_train_full.shape[1]} ({len(numeric_keys)} numeric + {sum(len(encoders[k].classes_) for k in string_keys)} one-hot)")

    # Train baseline
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train_full, y_train)
    p_baseline = model.predict_proba(X_test_full)[:, 1]
    brier_baseline = brier_score_loss(y_test, p_baseline)
    auroc_baseline = roc_auc_score(y_test, p_baseline)

    print(f"\nBaseline (all features):")
    print(f"  Brier: {brier_baseline:.4f}")
    print(f"  AUROC: {auroc_baseline:.4f}")

    # Feature importance
    print(f"\nFeature importance (top 10):")
    importances = model.feature_importances_
    # Reconstruct feature names
    feat_names = list(numeric_keys)
    for k in string_keys:
        for v in encoders[k].classes_:
            feat_names.append(f"{k}={v}")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {imp:.3f}")

    # Ablation tests
    ablations = [
        ("no_topology", {"topology_n_supported", "topology_n_contradicted", "topology_has_competition", "topology_unverified_exists"}),
        ("no_action_type", {"action_ANSWER", "action_DEFER", "action_VERIFY", "action_COMPARE", "action_STOP"}),
        ("no_delta_u", {"delta_u", "executive_utility", "base_utility"}),
        ("no_executive_action", {"executive_action"}),
        ("no_base_action", {"base_action"}),
        ("minimal", set(feature_keys) - {"topology_n_supported", "topology_has_competition", "action_ANSWER", "action_DEFER"}),
    ]

    print(f"\nAblation results:")
    for name, exclude in ablations:
        X_train_abl = encode(train_interventions, exclude_keys=exclude)
        X_test_abl = encode(test_interventions, exclude_keys=exclude)

        if X_train_abl.shape[1] == 0:
            print(f"  {name}: no features left")
            continue

        model_abl = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model_abl.fit(X_train_abl, y_train)
        p_abl = model_abl.predict_proba(X_test_abl)[:, 1]
        brier_abl = brier_score_loss(y_test, p_abl)
        auroc_abl = roc_auc_score(y_test, p_abl)

        print(f"  {name}: Brier={brier_abl:.4f}, AUROC={auroc_abl:.4f} "
              f"(ΔAUROC={auroc_baseline - auroc_abl:.4f})")

    # Leakage test: can we predict harm from template/category alone?
    print(f"\nLeakage test:")
    # Check if template name is predictive
    template_harm = defaultdict(lambda: [0, 0])
    for r in train_interventions + test_interventions:
        # Extract template from group_id (first part of task_id)
        task_id = r["features"].get("task_id", "")
        # Use the features to identify template
        is_adv = r["features"].get("topology_has_competition", 0) == 1.0
        template_harm[is_adv][r["is_harmful"]] += 1

    for template, (safe, harmful) in template_harm.items():
        total = safe + harmful
        if total > 0:
            print(f"  has_competition={template}: {harmful}/{total} harmful ({100*harmful/total:.1f}%)")

    # Save
    results = {
        "baseline_brier": float(brier_baseline),
        "baseline_auroc": float(auroc_baseline),
        "feature_importance": {name: float(imp) for name, imp in zip(feat_names, importances)},
        "ablations": {
            name: {"brier": float(brier_abl), "auroc": float(auroc_abl)}
            for name, _ in ablations
        },
    }
    output_path = REPO_ROOT / "experiments/daph_x/m3_structural/intervention_risk_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
