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

    Also extracts stratification features for stratified conformal calibration.
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

        # Extract stratification features from the executive action's record
        exec_rec = group[exec_idx][1]
        exec_feats = extract_m4_features(exec_rec)
        graph_feats = exec_rec.get("graph_features", {})

        # Stratification variables (all pre-decision)
        action_type = exec_rec.get("first_action_type", "UNKNOWN")
        belief_entropy = graph_feats.get("belief_entropy", 0.0)
        has_competition = graph_feats.get("topo_has_competition", 0.0)
        n_hyp = float(exec_feats.get("n_hyp", 0))
        n_ev = float(exec_feats.get("n_ev", 0))
        verify_remaining = float(exec_feats.get("verify_remaining", 0))

        # Coarse strata for stratified conformal
        # Stratum key: action_class | entropy_bin | competition_bin
        action_class = "ANSWER" if action_type == "ANSWER" else "OTHER"
        entropy_bin = "low" if belief_entropy < 1.0 else "high"
        competition_bin = "comp" if has_competition > 0.5 else "nocomp"
        stratum = f"{action_class}_{entropy_bin}_{competition_bin}"

        pairs.append({
            "group_id": gid,
            "exec_action": group[exec_idx][1]["first_action"],
            "base_action": group[base_idx][1]["first_action"],
            "delta_q_hat": float(delta_q_hat),
            "delta_u": float(delta_u),
            "residual": float(residual),
            "is_harmful": int(delta_u < 0),
            # Stratification features
            "stratum": stratum,
            "action_type": action_type,
            "action_class": action_class,
            "belief_entropy": float(belief_entropy),
            "has_competition": float(has_competition),
            "n_hyp": n_hyp,
            "n_ev": n_ev,
            "verify_remaining": verify_remaining,
            "topo_complexity": n_hyp * n_ev,
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


def stratified_conformal_calibrate(
    cal_pairs: list[dict],
    eval_pairs: list[dict],
    alpha_levels: list[float] = [0.50, 0.80, 0.90, 0.95, 0.99],
    min_stratum_size: int = 10,
) -> dict:
    """Stratified conformal calibration.

    Computes q_alpha(c) for each calibration stratum c, then applies the
    stratum-specific quantile to eval pairs in the same stratum.

    This improves coverage under distribution shift because within each
    stratum, the residual distribution is more homogeneous, and the shift
    is partially accounted for by the stratification.

    Strata are defined by pre-decision observables:
      - action_class (ANSWER vs OTHER)
      - belief_entropy bin (low < 1.0, high >= 1.0)
      - has_competition (comp vs nocomp)

    If a stratum has too few calibration samples (< min_stratum_size),
    it falls back to the global quantile.
    """
    # Group calibration pairs by stratum
    cal_by_stratum = defaultdict(list)
    for p in cal_pairs:
        cal_by_stratum[p["stratum"]].append(p)

    # Compute global quantile as fallback
    cal_residuals_global = np.array([p["residual"] for p in cal_pairs])
    n_cal_global = len(cal_residuals_global)

    results = {}
    for alpha in alpha_levels:
        # Global fallback quantile
        q_level_global = np.ceil(alpha * (n_cal_global + 1)) / n_cal_global
        q_level_global = min(q_level_global, 1.0)
        q_alpha_global = float(np.quantile(cal_residuals_global, q_level_global))

        # Per-stratum quantiles
        stratum_quantiles = {}
        for stratum, cal_stratum_pairs in cal_by_stratum.items():
            if len(cal_stratum_pairs) < min_stratum_size:
                stratum_quantiles[stratum] = q_alpha_global
                continue
            residuals = np.array([p["residual"] for p in cal_stratum_pairs])
            n = len(residuals)
            q_level = np.ceil(alpha * (n + 1)) / n
            q_level = min(q_level, 1.0)
            stratum_quantiles[stratum] = float(np.quantile(residuals, q_level))

        # Apply stratum-specific quantiles to eval pairs
        eval_residuals = np.array([p["residual"] for p in eval_pairs])
        would_force = np.zeros(len(eval_pairs), dtype=bool)
        true_harm = np.array([p["is_harmful"] for p in eval_pairs])

        covered = 0
        for i, p in enumerate(eval_pairs):
            stratum = p["stratum"]
            q_alpha = stratum_quantiles.get(stratum, q_alpha_global)
            if eval_residuals[i] <= q_alpha:
                covered += 1
            if p["delta_q_hat"] > q_alpha:
                would_force[i] = True

        coverage = covered / len(eval_pairs) if eval_pairs else 0.0

        n_force = int(would_force.sum())
        n_correct = int((would_force & ~true_harm).sum())
        n_harmful = int((would_force & true_harm).sum())

        force_precision = n_correct / max(n_force, 1)
        break_rate = n_harmful / max(n_force, 1)

        if n_harmful == 0 and n_force > 0:
            break_rate_upper_95 = 3.0 / n_force
        else:
            break_rate_upper_95 = None

        # Per-stratum coverage breakdown
        stratum_details = {}
        eval_by_stratum = defaultdict(list)
        for p in eval_pairs:
            eval_by_stratum[p["stratum"]].append(p)

        for stratum, eval_stratum_pairs in eval_by_stratum.items():
            q_alpha = stratum_quantiles.get(stratum, q_alpha_global)
            n_s = len(eval_stratum_pairs)
            n_cal_s = len(cal_by_stratum.get(stratum, []))
            cov_s = sum(1 for p in eval_stratum_pairs if p["residual"] <= q_alpha) / max(n_s, 1)
            stratum_details[stratum] = {
                "n_cal": n_cal_s,
                "n_eval": n_s,
                "q_alpha": round(q_alpha, 4),
                "coverage": round(cov_s, 4),
                "fallback": n_cal_s < min_stratum_size,
            }

        results[f"coverage_{alpha:.2f}"] = {
            "nominal_coverage": alpha,
            "q_alpha_global": round(q_alpha_global, 4),
            "empirical_coverage": round(coverage, 4),
            "n_force": n_force,
            "n_correct": n_correct,
            "n_harmful": n_harmful,
            "force_precision": round(force_precision, 4),
            "break_rate": round(break_rate, 4),
            "break_rate_upper_95": round(break_rate_upper_95, 4) if break_rate_upper_95 else None,
            "n_strata": len(stratum_quantiles),
            "n_strata_fallback": sum(1 for s in cal_by_stratum if len(cal_by_stratum[s]) < min_stratum_size),
            "strata": stratum_details,
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
        print(f"  GLOBAL Conformal calibration → {eval_name.upper()}")
        print(f"{'='*60}")

        results = conformal_calibrate(cal_pairs, eval_pairs, alpha_levels)
        all_results[eval_name + "_global"] = results

        print(f"\n  Coverage  q_alpha    EmpCov      n_force    Precision    BreakRate    BreakUpper95")
        for key, r in results.items():
            bu95 = f"{r['break_rate_upper_95']:.4f}" if r['break_rate_upper_95'] else "N/A"
            print(f"  {r['nominal_coverage']:.2f}     {r['q_alpha']:.4f}     {r['empirical_coverage']:.4f}      {r['n_force']:5d}      {r['force_precision']:.4f}       {r['break_rate']:.4f}       {bu95}")

        print(f"\n{'='*60}")
        print(f"  STRATIFIED Conformal calibration → {eval_name.upper()}")
        print(f"{'='*60}")

        strat_results = stratified_conformal_calibrate(cal_pairs, eval_pairs, alpha_levels)
        all_results[eval_name + "_stratified"] = strat_results

        print(f"\n  Coverage  q_global   EmpCov      n_force    Precision    BreakRate    BreakUpper95  nStrata  nFallback")
        for key, r in strat_results.items():
            bu95 = f"{r['break_rate_upper_95']:.4f}" if r['break_rate_upper_95'] else "N/A"
            print(f"  {r['nominal_coverage']:.2f}     {r['q_alpha_global']:.4f}     {r['empirical_coverage']:.4f}      {r['n_force']:5d}      {r['force_precision']:.4f}       {r['break_rate']:.4f}       {bu95}      {r['n_strata']:5d}    {r['n_strata_fallback']:5d}")

        # Print per-stratum details for 90% level
        print(f"\n  Per-stratum details (90% coverage):")
        r90 = strat_results.get("coverage_0.90", {})
        for s_name, s_detail in sorted(r90.get("strata", {}).items()):
            fb = " (fallback)" if s_detail["fallback"] else ""
            print(f"    {s_name:30s}  n_cal={s_detail['n_cal']:3d}  n_eval={s_detail['n_eval']:3d}  q={s_detail['q_alpha']:.2f}  cov={s_detail['coverage']:.4f}{fb}")

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
