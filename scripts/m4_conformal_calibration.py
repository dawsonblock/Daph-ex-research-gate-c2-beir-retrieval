#!/usr/bin/env python3
"""Pairwise conformal ΔQ calibration for DAPH-X M4.

Calibrates the error of pairwise advantage predictions:
  ΔQ_hat = Q_X(s,a_X) - Q_X(s,a_B)
against true rollout:
  ΔU = Q^π(s,a_X) - Q^π(s,a_B)

Uses the calibration split (NOT train, NOT OOD) to compute conformal quantiles.
Reports empirical coverage at 50/80/90/95/99% on structural_ood and mechanism_ood.

The alpha parameter is COVERAGE (not miscoverage):
  alpha = 0.90 means 90% nominal coverage.
  q_alpha = ceil(alpha * (n_cal + 1)) / n_cal quantile of |residuals|.
  This is the standard split-conformal finite-sample correction.

Outputs:
  LCB_Δ = ΔQ_hat - q_alpha  (lower confidence bound on advantage)

Usage:
    python scripts/m4_conformal_calibration.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"

# Import feature extraction from Q_res training
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_m4_q_res import extract_m4_features, compute_q_mb_from_record, load_m4_split


def compute_pairwise_advantages(records: list[dict], model, feature_keys: list[str]) -> list[dict]:
    """For each counterfactual group, compute pairwise advantages.

    For each group with actions {a_1, ..., a_n}:
      - Compute Q_X(s, a_i) = Q_MB(s,a_i) + Q_res(s,a_i) for each action
      - Identify a_X = argmax Q_X (executive action)
      - Identify a_B = DEFER action (base policy)
      - Compute ΔQ_hat = Q_X(s,a_X) - Q_X(s,a_B)
      - Compute ΔU = U(s,a_X) - U(s,a_B) from rollout utilities
      - Residual r = |ΔQ_hat - ΔU|
    """
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["counterfactual_group_id"]].append((i, r))

    pairs = []
    for gid, group in groups.items():
        if len(group) < 2:
            continue

        # Compute Q_X for each action
        q_x_scores = []
        utilities = []
        for idx, rec in group:
            feats = extract_m4_features(rec)
            x = np.array([[feats[k] for k in feature_keys]])
            q_mb = compute_q_mb_from_record(rec)
            q_res = model.predict(x)[0]
            q_x = q_mb + q_res
            q_x_scores.append(q_x)
            utilities.append(rec["utility"])

        # Executive action: argmax Q_X
        exec_idx = np.argmax(q_x_scores)

        # Base action: find DEFER
        base_idx = None
        for i, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = i
                break
        if base_idx is None:
            base_idx = 0  # fallback

        delta_q_hat = q_x_scores[exec_idx] - q_x_scores[base_idx]
        delta_u = utilities[exec_idx] - utilities[base_idx]
        residual = abs(delta_q_hat - delta_u)

        pairs.append({
            "group_id": gid,
            "exec_action": group[exec_idx][1]["first_action"],
            "base_action": group[base_idx][1]["first_action"],
            "delta_q_hat": float(delta_q_hat),
            "delta_u": float(delta_u),
            "residual": float(residual),
            "is_harmful": int(delta_u < 0),
        })

    return pairs


def conformal_calibrate(
    cal_pairs: list[dict],
    eval_pairs: list[dict],
    alpha_levels: list[float] = [0.50, 0.80, 0.90, 0.95, 0.99],
) -> dict:
    """Conformal calibration of pairwise advantage predictions.

    For each alpha level (alpha = nominal coverage, e.g. 0.90 = 90%):
      q_alpha = alpha quantile of calibration |residuals| with finite-sample correction
      LCB_delta = delta_q_hat - q_alpha
      Coverage = fraction of eval pairs where |delta_q_hat - delta_u| <= q_alpha
    """
    cal_residuals = np.array([p["residual"] for p in cal_pairs])
    n_cal = len(cal_residuals)

    results = {}
    for alpha in alpha_levels:
        # Split-conformal quantile: for coverage level alpha,
        # q_alpha = ceil(alpha * (n_cal + 1)) / n_cal quantile of |residuals|.
        # This is the standard finite-sample correction (Vovk et al.).
        # The +1 and ceil ensure the coverage guarantee holds in finite samples.
        q_level = np.ceil(alpha * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        q_alpha = float(np.quantile(cal_residuals, q_level))

        # Evaluate coverage on eval set
        eval_residuals = np.array([p["residual"] for p in eval_pairs])
        coverage = float(np.mean(eval_residuals <= q_alpha))

        # LCB-based FORCE decisions
        # would_force = 1 if LCB_delta > 0 (i.e., delta_q_hat > q_alpha)
        would_force = np.array([p["delta_q_hat"] > q_alpha for p in eval_pairs])
        true_harm = np.array([p["is_harmful"] for p in eval_pairs])

        n_force = int(would_force.sum())
        n_correct = int((would_force & ~true_harm).sum())  # force and not harmful
        n_harmful = int((would_force & true_harm).sum())   # force and harmful (breaks)

        force_precision = n_correct / max(n_force, 1)
        break_rate = n_harmful / max(n_force, 1)

        # Rule of three: 95% upper bound on break rate if 0 breaks observed
        if n_harmful == 0 and n_force > 0:
            break_rate_upper_95 = 3.0 / n_force
        else:
            break_rate_upper_95 = None

        results[f"coverage_{alpha:.2f}"] = {
            "nominal_coverage": alpha,
            "q_alpha": round(q_alpha, 4),
            "empirical_coverage": round(coverage, 4),
            "n_force": n_force,
            "n_correct": n_correct,
            "n_harmful": n_harmful,
            "force_precision": round(force_precision, 4),
            "break_rate": round(break_rate, 4),
            "break_rate_upper_95": round(break_rate_upper_95, 4) if break_rate_upper_95 else None,
        }

    return results


def main():
    # Load Q_res model
    model_path = M4_DIR / "q_res_m4.pkl"
    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_keys = model_data["feature_keys"]

    # Load splits
    cal_records = load_m4_split("calibration")
    struct_ood_records = load_m4_split("structural_ood")
    mech_ood_records = load_m4_split("mechanism_ood")

    print(f"Calibration: {len(cal_records)} records")
    print(f"Structural OOD: {len(struct_ood_records)} records")
    print(f"Mechanism OOD: {len(mech_ood_records)} records")

    # Compute pairwise advantages
    cal_pairs = compute_pairwise_advantages(cal_records, model, feature_keys)
    struct_ood_pairs = compute_pairwise_advantages(struct_ood_records, model, feature_keys)
    mech_ood_pairs = compute_pairwise_advantages(mech_ood_records, model, feature_keys)

    print(f"\nCalibration pairs: {len(cal_pairs)}")
    print(f"Structural OOD pairs: {len(struct_ood_pairs)}")
    print(f"Mechanism OOD pairs: {len(mech_ood_pairs)}")

    # Harm distribution
    for name, pairs in [("calibration", cal_pairs), ("structural_ood", struct_ood_pairs), ("mechanism_ood", mech_ood_pairs)]:
        n_harm = sum(p["is_harmful"] for p in pairs)
        print(f"  {name}: {n_harm}/{len(pairs)} harmful ({100*n_harm/max(len(pairs),1):.1f}%)")

    # Calibrate
    alpha_levels = [0.50, 0.80, 0.90, 0.95, 0.99]

    all_results = {}

    # Self-coverage check: calibrate on calibration set, evaluate on calibration set
    # This should achieve approximately nominal coverage if the method is correct.
    print(f"\n{'='*60}")
    print(f"  Conformal SELF-COVERAGE check → CALIBRATION")
    print(f"{'='*60}")
    cal_self_results = conformal_calibrate(cal_pairs, cal_pairs, alpha_levels)
    all_results["calibration_self"] = cal_self_results
    print(f"\n  Coverage  q_alpha    EmpCov      n_force    Precision    BreakRate    BreakUpper95")
    for key, r in cal_self_results.items():
        bu95 = f"{r['break_rate_upper_95']:.4f}" if r['break_rate_upper_95'] else "N/A"
        print(f"  {r['nominal_coverage']:.2f}     {r['q_alpha']:.4f}     {r['empirical_coverage']:.4f}      {r['n_force']:5d}      {r['force_precision']:.4f}       {r['break_rate']:.4f}       {bu95}")

    for eval_name, eval_pairs in [("structural_ood", struct_ood_pairs), ("mechanism_ood", mech_ood_pairs)]:
        print(f"\n{'='*60}")
        print(f"  Conformal calibration → {eval_name.upper()}")
        print(f"{'='*60}")

        results = conformal_calibrate(cal_pairs, eval_pairs, alpha_levels)
        all_results[eval_name] = results

        print(f"\n  Coverage  q_alpha    EmpCov      n_force    Precision    BreakRate    BreakUpper95")
        for key, r in results.items():
            bu95 = f"{r['break_rate_upper_95']:.4f}" if r['break_rate_upper_95'] else "N/A"
            print(f"  {r['nominal_coverage']:.2f}     {r['q_alpha']:.4f}     {r['empirical_coverage']:.4f}      {r['n_force']:5d}      {r['force_precision']:.4f}       {r['break_rate']:.4f}       {bu95}")

    # Save results
    output = {
        "calibration_pairs": len(cal_pairs),
        "calibration_harmful": sum(p["is_harmful"] for p in cal_pairs),
        "alpha_levels": alpha_levels,
        "results": all_results,
    }
    output_path = M4_DIR / "conformal_calibration_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
