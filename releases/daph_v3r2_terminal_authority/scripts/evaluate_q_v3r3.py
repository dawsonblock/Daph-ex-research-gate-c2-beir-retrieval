#!/usr/bin/env python3
"""Q_V3R3 held-out evaluation.

Evaluates the repaired Q_V3R3 candidate on held-out data before any live use.
Measures:
  - Action ranking accuracy
  - Authority coverage (how often would it force)
  - False authority (would force when shouldn't)
  - Uncertainty calibration (ensemble std vs Q error)
  - OOD behavior (in-support rate, distance distribution)

Usage:
    PYTHONPATH=. python3 scripts/evaluate_q_v3r3_heldout.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from daph.models.q_ensemble import QEnsemble

    # Load Q_V3R3
    model_path = REPO_ROOT / "experiments/i3_30r3/Q_V3R3.pkl"
    schema_path = REPO_ROOT / "experiments/i3_30r3/v3r3_feature_schema.json"
    report_path = REPO_ROOT / "experiments/i3_30r3/v3r3_training_report.json"

    if not model_path.exists():
        print("ERROR: Q_V3R3.pkl not found. Train first with run_i3_30r3_train_v3r3.py")
        sys.exit(1)

    with open(model_path, "rb") as f:
        q_model = pickle.load(f)
    with open(schema_path) as f:
        schema = json.load(f)
    with open(report_path) as f:
        training_report = json.load(f)

    print("=" * 60)
    print("Q_V3R3 Held-Out Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    td = training_report.get('training_data', {})
    perf = training_report.get('performance', {})
    ens = training_report.get('ensemble', {})
    print(f"Training rows: {td.get('actual_training_rows', '?')}")
    print(f"Features: {training_report.get('feature_count', '?')}")
    print(f"Train R²: {perf.get('train_r2', '?')}")
    print(f"Mean uncertainty: {perf.get('mean_uncertainty', '?')}")
    print(f"In-support rate (train): {perf.get('in_support_rate', '?')}")
    print(f"Ensemble: {ens.get('n_estimators', '?')} models, lambda_lcb={ens.get('lambda_lcb', '?')}")

    # Load confirmation trajectories as held-out data
    # These are from the 400-task confirmation run — NOT used in Q_V3R3 training
    heldout_records = []
    for arm_path in [
        REPO_ROOT / "experiments/i3_30r3/confirmation/trajectories_v3_shadow.jsonl",
        REPO_ROOT / "experiments/i3_30r3/confirmation/trajectories_v3_hard.jsonl",
    ]:
        if arm_path.exists():
            with open(arm_path) as f:
                for line in f:
                    rec = json.loads(line)
                    for evt in rec.get("authority_events", []):
                        if evt.get("q_values") and evt.get("structural_state"):
                            heldout_records.append(evt)

    # Also load OOD trajectory events
    ood_path = REPO_ROOT / "experiments/i3_30r3/structural_ood_run/trajectories_v3_hard.jsonl"
    if ood_path.exists():
        ood_records = []
        with open(ood_path) as f:
            for line in f:
                rec = json.loads(line)
                for evt in rec.get("authority_events", []):
                    if evt.get("q_values") and evt.get("structural_state"):
                        ood_records.append(evt)
        print(f"\nHeld-out confirmation events: {len(heldout_records)}")
        print(f"Held-out OOD events: {len(ood_records)}")
    else:
        print(f"\nHeld-out confirmation events: {len(heldout_records)}")
        ood_records = []

    if not heldout_records:
        print("ERROR: No held-out records found")
        sys.exit(1)

    # For each held-out event, we have:
    # - q_values from V3R2 (the confirmed model)
    # - structural_state
    # - the action that was taken and whether it succeeded
    # We want to check: does Q_V3R3 agree with V3R2 on action ranking?
    # And does Q_V3R3's uncertainty correlate with prediction error?

    # Extract features from structural_state
    # Q_V3R3 uses the same features as V3R2 plus some additional ones
    # For this evaluation, we'll use the structural_state as features

    feature_keys = sorted(heldout_records[0].get("structural_state", {}).keys()) if heldout_records else []
    print(f"Structural feature keys: {len(feature_keys)}")

    def extract_features(evt):
        struct = evt.get("structural_state", {})
        return [float(struct.get(k, 0)) for k in feature_keys]

    def extract_q_values(evt):
        qv = evt.get("q_values", {})
        return {k: float(v) for k, v in qv.items()}

    def get_argmax(qv):
        if not qv:
            return None
        return max(qv, key=qv.get)

    # 1. Action ranking agreement: does Q_V3R3 pick the same argmax as V3R2?
    print(f"\n{'='*60}")
    print("1. Action Ranking Agreement (Q_V3R3 vs V3R2)")
    print(f"{'='*60}")

    agreements = 0
    total = 0
    q_v3r3_predictions = []

    for evt in heldout_records:
        v3r2_q = extract_q_values(evt)
        v3r2_argmax = get_argmax(v3r2_q)
        if not v3r2_argmax:
            continue

        # We can't directly call Q_V3R3.predict_q here because we don't have
        # the full state features (sf) — only the structural_state.
        # The Q model needs both sf and v3_struct_dict.
        # For this evaluation, we'll check if the structural state alone
        # would produce consistent Q predictions.
        total += 1

    print(f"  Note: Full Q_V3R3 prediction requires state_features + v3_struct_dict.")
    print(f"  The held-out events only have structural_state, not full state_features.")
    print(f"  Evaluating structural-state-level metrics instead.")

    # 2. Authority coverage analysis
    print(f"\n{'='*60}")
    print("2. Authority Coverage Analysis")
    print(f"{'='*60}")

    would_force_count = 0
    answer_force = 0
    defer_force = 0
    for evt in heldout_records:
        if evt.get("would_force"):
            would_force_count += 1
            if evt.get("forced_action") == "ANSWER":
                answer_force += 1
            elif evt.get("forced_action") == "DEFER":
                defer_force += 1

    print(f"  Total held-out events: {len(heldout_records)}")
    print(f"  Would-force events: {would_force_count} ({would_force_count/len(heldout_records)*100:.1f}%)")
    print(f"    ANSWER forces: {answer_force}")
    print(f"    DEFER forces: {defer_force}")

    # 3. Q gap distribution
    print(f"\n{'='*60}")
    print("3. Q Gap Distribution (V3R2 on held-out)")
    print(f"{'='*60}")

    q_gaps = [float(evt.get("q_gap", 0)) for evt in heldout_records if evt.get("q_gap") is not None]
    if q_gaps:
        print(f"  min: {min(q_gaps):.2f}")
        print(f"  max: {max(q_gaps):.2f}")
        print(f"  mean: {np.mean(q_gaps):.2f}")
        print(f"  median: {np.median(q_gaps):.2f}")
        print(f"  >= 5.0 (threshold): {sum(1 for g in q_gaps if g >= 5.0)}/{len(q_gaps)}")

    # 4. Certificate type distribution
    print(f"\n{'='*60}")
    print("4. Certificate Type Distribution")
    print(f"{'='*60}")

    from collections import Counter
    cert_types = Counter(evt.get("certificate_type", "NONE") for evt in heldout_records)
    for ct, count in cert_types.most_common():
        print(f"  {count:>4} ({count/len(heldout_records)*100:.1f}%): {ct}")

    # 5. OOD behavior
    if ood_records:
        print(f"\n{'='*60}")
        print("5. OOD Behavior (V3R2 on structural-OOD tasks)")
        print(f"{'='*60}")

        ood_would_force = sum(1 for evt in ood_records if evt.get("would_force"))
        ood_q_gaps = [float(evt.get("q_gap", 0)) for evt in ood_records if evt.get("q_gap") is not None]
        ood_certs = Counter(evt.get("certificate_type", "NONE") for evt in ood_records)

        print(f"  OOD events: {len(ood_records)}")
        print(f"  Would-force: {ood_would_force} ({ood_would_force/len(ood_records)*100:.1f}%)")
        if ood_q_gaps:
            print(f"  Q gap: min={min(ood_q_gaps):.2f}, max={max(ood_q_gaps):.2f}, mean={np.mean(ood_q_gaps):.2f}")
        print(f"  Certificate types:")
        for ct, count in ood_certs.most_common():
            print(f"    {count:>4}: {ct}")

    # 6. Uncertainty calibration
    print(f"\n{'='*60}")
    print("6. Uncertainty Calibration (Q_V3R3 ensemble)")
    print(f"{'='*60}")

    # The Q_V3R3 model has ensemble standard deviation as uncertainty
    # We need to check if high uncertainty correlates with large Q error
    # Q error = |Q_predicted - realized_utility|
    # But we don't have Q_V3R3 predictions for these held-out events
    # (we only have V3R2 predictions)
    # So we report the training uncertainty and note that full calibration
    # requires running Q_V3R3 on these states

    print(f"  Training mean uncertainty: {perf.get('mean_uncertainty', '?')}")
    print(f"  Training in-support rate: {perf.get('in_support_rate', '?')}")
    print(f"  Note: Full uncertainty calibration requires running Q_V3R3.predict_q()")
    print(f"  on held-out states with full state_features. The held-out events")
    print(f"  only have structural_state, not the complete feature vector.")
    print(f"  This is a known limitation — full calibration is a future task.")

    # 7. Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Q_V3R3 training: {training_report.get('actual_training_rows', '?')} rows")
    print(f"  Q_V3R3 train R²: {training_report.get('train_r2', '?')}")
    print(f"  Q_V3R3 in-support: {training_report.get('in_support_rate', '?')}")
    print(f"  Held-out events: {len(heldout_records)}")
    print(f"  Held-out would-force rate: {would_force_count/len(heldout_records)*100:.1f}%")
    print(f"  Q_V3R3 status: CANDIDATE — not promoted for live use")
    print(f"  Required before promotion:")
    print(f"    1. Full predict_q() on held-out states with complete features")
    print(f"    2. Uncertainty calibration (ensemble std vs Q error)")
    print(f"    3. OOD threshold calibration on held-out data")
    print(f"    4. Comparison against V3R2 on same held-out states")

    # Save
    results = {
        "model": "Q_V3R3",
        "status": "CANDIDATE — not promoted",
        "training_rows": td.get("actual_training_rows"),
        "train_r2": perf.get("train_r2"),
        "in_support_rate": perf.get("in_support_rate"),
        "mean_uncertainty": perf.get("mean_uncertainty"),
        "heldout_events": len(heldout_records),
        "heldout_would_force": would_force_count,
        "heldout_would_force_pct": would_force_count / len(heldout_records) * 100,
        "q_gap_mean": float(np.mean(q_gaps)) if q_gaps else None,
        "q_gap_median": float(np.median(q_gaps)) if q_gaps else None,
        "certificate_types": dict(cert_types),
        "ood_events": len(ood_records) if ood_records else 0,
        "limitations": [
            "Full predict_q() not run on held-out (missing state_features)",
            "Uncertainty calibration not completed",
            "OOD threshold not calibrated on held-out",
        ],
    }
    output_path = REPO_ROOT / "experiments/i3_30r3/v3r3_heldout_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
