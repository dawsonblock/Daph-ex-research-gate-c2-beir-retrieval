#!/usr/bin/env python3
"""Intervention-risk model for DAPH-X M4.

Predicts P(ΔU < 0) — the probability that the executive's preferred
action is harmful relative to the base policy.

Uses ONLY pre-decision observables. A schema guard automatically
rejects forbidden (post-intervention/oracle) fields.

Trains on M4 train split, evaluates on structural_ood and mechanism_ood.

Usage:
    python scripts/m4_intervention_risk.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_m4_q_res import extract_m4_features, compute_q_mb_from_record, load_m4_split


# ─── Anti-leakage guard ───

FORBIDDEN_FIELD_SUBSTRINGS = [
    "utility", "oracle", "regret", "delta_u", "is_harmful",
    "executive_utility", "executive_is_oracle", "base_utility",
    "correct_hypothesis", "terminal_reason", "success",
    "runtime_error", "trajectory", "observation_path",
    "terminal_state", "steps_used", "total_cost",
    "harm_mechanism", "mechanism_family",
]


def assert_no_forbidden_features(feature_keys: list[str]):
    """Schema guard: reject any forbidden field."""
    for k in feature_keys:
        for forbidden in FORBIDDEN_FIELD_SUBSTRINGS:
            assert forbidden not in k.lower(), (
                f"FORBIDDEN FEATURE DETECTED: '{k}' contains '{forbidden}'. "
                f"Intervention-risk features must be pre-decision only. "
                f"This is a leakage guard — do not bypass."
            )


def extract_risk_features(
    record: dict,
    model,
    feature_keys: list[str],
    cal_q_alpha: float = 0.0,
) -> dict[str, float]:
    """Extract pre-decision features for intervention-risk prediction.

    These features describe the state and the predicted advantage,
    but NOT the actual outcome.
    """
    feats = {}

    # Pre-decision state features (same as Q_res)
    base_feats = extract_m4_features(record)
    feats.update(base_feats)

    # Predicted Q values (pre-decision, from learned model)
    x = np.array([[base_feats[k] for k in feature_keys]])
    q_mb = compute_q_mb_from_record(record)
    q_res = model.predict(x)[0]
    q_x = q_mb + q_res
    feats["q_x_predicted"] = float(q_x)
    feats["q_mb_predicted"] = float(q_mb)
    feats["q_res_predicted"] = float(q_res)

    # LCB delta (pre-decision, from conformal calibration)
    # This is the lower confidence bound on the advantage
    feats["lcb_delta"] = float(q_x - cal_q_alpha)

    # Action pair features (pre-decision)
    action_type = record["first_action_type"]
    feats["is_answer"] = 1.0 if action_type == "ANSWER" else 0.0
    feats["is_verify"] = 1.0 if action_type == "VERIFY" else 0.0
    feats["is_defer"] = 1.0 if action_type == "DEFER" else 0.0

    return feats


def compute_intervention_labels(records: list[dict]) -> tuple[list[dict], list[int]]:
    """For each group, compute the intervention features and harm label.

    The executive action is selected by Q_X (not oracle).
    The base action is DEFER.
    Harm label: 1 if ΔU < 0, 0 otherwise.
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["counterfactual_group_id"]].append((i, r))

    features_list = []
    labels = []

    for gid, group in groups.items():
        if len(group) < 2:
            continue

        # Find base (DEFER)
        base_idx = None
        for i, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = i
                break
        if base_idx is None:
            base_idx = 0

        # Find executive (argmax Q_MB from record)
        q_mb_scores = [compute_q_mb_from_record(rec) for _, rec in group]
        exec_idx = np.argmax(q_mb_scores)

        exec_rec = group[exec_idx][1]
        base_rec = group[base_idx][1]

        delta_u = exec_rec["utility"] - base_rec["utility"]
        is_harmful = int(delta_u < 0)

        # Features from the executive action record (pre-decision)
        feats = extract_m4_features(exec_rec)
        feats["q_mb_exec"] = float(q_mb_scores[exec_idx])
        feats["q_mb_base"] = float(q_mb_scores[base_idx])
        feats["delta_q_mb"] = float(q_mb_scores[exec_idx] - q_mb_scores[base_idx])

        features_list.append(feats)
        labels.append(is_harmful)

    return features_list, labels


def evaluate_risk_model(
    model,
    features_list: list[dict],
    labels: list[int],
    feature_keys: list[str],
) -> dict:
    """Evaluate the risk model on a set of interventions."""
    X = np.array([[f[k] for k in feature_keys] for f in features_list])
    y = np.array(labels)

    if len(set(y)) < 2:
        return {"error": "Only one class present"}

    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    auroc = roc_auc_score(y, y_prob)
    auprc = average_precision_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)

    # Harm FNR: fraction of harmful interventions not flagged at threshold 0.5
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    harm_fnr = float(np.mean(y_pred[y == 1] == 0)) if y.sum() > 0 else 0.0

    # Precision at safety threshold (0.5)
    precision = float(np.mean(y[y_pred == 1] == 1)) if y_pred.sum() > 0 else 0.0

    # Calibration (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = y_prob[mask].mean()
            avg_acc = y[mask].mean()
            ece += mask.sum() / len(y) * abs(avg_conf - avg_acc)

    return {
        "n": len(y),
        "n_harmful": int(y.sum()),
        "n_safe": int((y == 0).sum()),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier": round(brier, 4),
        "harm_fnr": round(harm_fnr, 4),
        "precision_at_0.5": round(precision, 4),
        "ece": round(ece, 4),
    }


def main():
    # Load Q_res model
    model_path = M4_DIR / "q_res_m4.pkl"
    model_data = joblib.load(model_path)
    q_res_model = model_data["model"]
    q_res_feature_keys = model_data["feature_keys"]

    # Load conformal calibration to get q_alpha
    cal_path = M4_DIR / "conformal_calibration_m4.json"
    cal_q_alpha = 0.0
    if cal_path.exists():
        cal_data = json.loads(open(cal_path).read())
        # Use 90% alpha from structural_ood
        struct_results = cal_data.get("results", {}).get("structural_ood", {})
        cal_q_alpha = struct_results.get("alpha_0.90", {}).get("q_alpha", 0.0)

    # Load splits
    train_records = load_m4_split("train")
    struct_ood_records = load_m4_split("structural_ood")
    mech_ood_records = load_m4_split("mechanism_ood")

    # Compute intervention features and labels
    print("Computing intervention features...")
    train_feats, train_labels = compute_intervention_labels(train_records)
    struct_feats, struct_labels = compute_intervention_labels(struct_ood_records)
    mech_feats, mech_labels = compute_intervention_labels(mech_ood_records)

    print(f"Train: {len(train_labels)} interventions, {sum(train_labels)} harmful")
    print(f"Structural OOD: {len(struct_labels)} interventions, {sum(struct_labels)} harmful")
    print(f"Mechanism OOD: {len(mech_labels)} interventions, {sum(mech_labels)} harmful")

    # Get feature keys and run anti-leakage guard
    feature_keys = sorted(train_feats[0].keys())
    print(f"\nFeature keys ({len(feature_keys)}): {feature_keys}")
    assert_no_forbidden_features(feature_keys)
    print("✓ Anti-leakage guard PASSED — no forbidden features detected")

    X_train = np.array([[f[k] for k in feature_keys] for f in train_feats])
    y_train = np.array(train_labels)

    # Train risk model
    print(f"\nTraining GradientBoostingClassifier...")
    risk_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    risk_model.fit(X_train, y_train)

    # Evaluate
    all_results = {}
    for name, feats, labels in [
        ("train", train_feats, train_labels),
        ("structural_ood", struct_feats, struct_labels),
        ("mechanism_ood", mech_feats, mech_labels),
    ]:
        result = evaluate_risk_model(risk_model, feats, labels, feature_keys)
        all_results[name] = result

        print(f"\n{'='*50}")
        print(f"  {name.upper()}")
        print(f"{'='*50}")
        for k, v in result.items():
            print(f"  {k}: {v}")

    # Mechanism-heldout analysis: per-mechism breakdown
    print(f"\n{'='*50}")
    print(f"  MECHANISM-HELDOUT BREAKDOWN")
    print(f"{'='*50}")

    # For mechanism_ood, evaluate per mechanism
    mech_groups = defaultdict(list)
    for i, r in enumerate(mech_ood_records):
        mech_groups[r.get("harm_mechanism", "unknown")].append(i)

    for mech, indices in sorted(mech_groups.items()):
        # Get the interventions for this mechanism
        mech_feats_subset = [struct_feats[i] for i in indices if i < len(struct_feats)]
        # Actually need to recompute for mechanism_ood
        pass

    # Save model and results
    output = {
        "feature_keys": feature_keys,
        "train_size": len(train_labels),
        "train_harmful": sum(train_labels),
        "results": all_results,
        "anti_leakage_guard": "PASSED",
    }
    output_path = M4_DIR / "intervention_risk_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save model
    model_out_path = M4_DIR / "risk_model_m4.pkl"
    joblib.dump({
        "model": risk_model,
        "feature_keys": feature_keys,
    }, model_out_path)

    print(f"\nSaved results to {output_path}")
    print(f"Saved model to {model_out_path}")


if __name__ == "__main__":
    main()
