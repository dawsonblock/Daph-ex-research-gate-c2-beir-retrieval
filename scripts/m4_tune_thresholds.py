#!/usr/bin/env python3
"""Tune shadow authority thresholds using calibration data only.

Sweeps over alpha (conformal coverage), tau_delta (LCB threshold), and
rho (risk threshold) to find the configuration that maximizes rescue
recall subject to a safety constraint on the calibration set.

The safety constraint is: UCB_95(break_rate) < target_break_upper.

OOD splits are NOT used during tuning — only calibration.

Usage:
    python scripts/m4_tune_thresholds.py
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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_m4_q_res import extract_m4_features, compute_q_mb_from_record, load_m4_split


def compute_intervention_data(records, q_res_model, q_res_feature_keys,
                               pairwise_model, pairwise_feature_keys,
                               risk_model, risk_feature_keys,
                               stratum_quantiles, q_alpha_global):
    """Compute all pre-decision data for each intervention group."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["counterfactual_group_id"]].append((i, r))

    interventions = []
    for gid, group in groups.items():
        if len(group) < 2:
            continue

        q_x_scores = []
        q_mb_scores = []
        utilities = []
        for idx, rec in group:
            feats = extract_m4_features(rec)
            x = np.array([[feats[k] for k in q_res_feature_keys]])
            q_mb = compute_q_mb_from_record(rec)
            q_res = q_res_model.predict(x)[0]
            q_x = q_mb + q_res
            q_x_scores.append(q_x)
            q_mb_scores.append(q_mb)
            utilities.append(rec["utility"])

        exec_idx = np.argmax(q_x_scores)

        base_idx = None
        for i, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = i
                break
        if base_idx is None:
            base_idx = 0

        delta_q_hat = q_x_scores[exec_idx] - q_x_scores[base_idx]
        delta_u = utilities[exec_idx] - utilities[base_idx]

        exec_rec = group[exec_idx][1]
        exec_feats = extract_m4_features(exec_rec)
        graph_feats = exec_rec.get("graph_features", {})

        # Stratum
        action_type = exec_rec.get("first_action_type", "UNKNOWN")
        action_class = "ANSWER" if action_type == "ANSWER" else "OTHER"
        belief_entropy = graph_feats.get("belief_entropy", 0.0)
        has_competition = graph_feats.get("topo_has_competition", 0.0)
        entropy_bin = "low" if belief_entropy < 1.0 else "high"
        competition_bin = "comp" if has_competition > 0.5 else "nocomp"
        stratum = f"{action_class}_{entropy_bin}_{competition_bin}"
        q_alpha_local = stratum_quantiles.get(stratum, q_alpha_global)

        # Risk prediction
        risk_feats = {}
        risk_feats.update(exec_feats)
        risk_feats["q_mb_exec"] = float(q_mb_scores[exec_idx])
        risk_feats["q_mb_base"] = float(q_mb_scores[base_idx])
        risk_feats["delta_q_mb"] = float(q_mb_scores[exec_idx] - q_mb_scores[base_idx])
        x_risk = np.array([[risk_feats.get(k, 0.0) for k in risk_feature_keys]])
        risk_prob = float(risk_model.predict_proba(x_risk)[0, 1])

        # Pairwise prediction
        pairwise_pred = 0.0
        if pairwise_model is not None:
            pw_feats = dict(exec_feats)
            pw_feats["delta_q_mb"] = float(q_mb_scores[exec_idx] - q_mb_scores[base_idx])
            exec_x = np.array([[exec_feats[k] for k in q_res_feature_keys]])
            base_rec = group[base_idx][1]
            base_feats_dict = extract_m4_features(base_rec)
            base_x = np.array([[base_feats_dict[k] for k in q_res_feature_keys]])
            pw_feats["q_res_exec_pred"] = float(q_res_model.predict(exec_x)[0])
            pw_feats["q_res_base_pred"] = float(q_res_model.predict(base_x)[0])
            pw_feats["delta_q_res_pred"] = pw_feats["q_res_exec_pred"] - pw_feats["q_res_base_pred"]
            pw_x = np.array([[pw_feats.get(k, 0.0) for k in pairwise_feature_keys]])
            pairwise_pred = float(pairwise_model.predict(pw_x)[0])

        interventions.append({
            "group_id": gid,
            "delta_q_hat": float(delta_q_hat),
            "q_alpha_local": float(q_alpha_local),
            "risk_prob": risk_prob,
            "pairwise_pred": pairwise_pred,
            "delta_u": float(delta_u),
            "is_harmful": int(delta_u < 0),
            "stratum": stratum,
        })

    return interventions


def evaluate_config(interventions, tau_delta, rho, use_pairwise=True,
                    pairwise_threshold=0.0, gate_mode="all"):
    """Evaluate a configuration.

    gate_mode:
      "all" — AND of LCB, risk, pairwise (conservative)
      "pairwise_risk" — pairwise AND risk (no conformal LCB gate)
      "pairwise_only" — pairwise only
      "lcb_risk" — LCB AND risk (no pairwise)
    """
    n_force = 0
    rescues = 0
    breaks = 0
    n_beneficial = 0

    for iv in interventions:
        if not iv["is_harmful"]:
            n_beneficial += 1

        lcb = iv["delta_q_hat"] - iv["q_alpha_local"]
        lcb_pass = lcb > tau_delta
        risk_pass = iv["risk_prob"] < rho
        pw_pass = iv["pairwise_pred"] > pairwise_threshold

        if gate_mode == "all":
            conditions = [lcb_pass, risk_pass]
            if use_pairwise:
                conditions.append(pw_pass)
        elif gate_mode == "pairwise_risk":
            conditions = [risk_pass, pw_pass] if use_pairwise else [risk_pass]
        elif gate_mode == "pairwise_only":
            conditions = [pw_pass] if use_pairwise else []
        elif gate_mode == "lcb_risk":
            conditions = [lcb_pass, risk_pass]
        else:
            conditions = [lcb_pass, risk_pass]

        would_force = all(conditions) if conditions else False

        if would_force:
            n_force += 1
            if iv["is_harmful"]:
                breaks += 1
            else:
                rescues += 1

    break_rate = breaks / max(n_force, 1)
    force_precision = rescues / max(n_force, 1)
    rescue_recall = rescues / max(n_beneficial, 1)

    if breaks == 0 and n_force > 0:
        break_rate_upper_95 = 3.0 / n_force
    else:
        break_rate_upper_95 = None

    return {
        "n_force": n_force,
        "rescues": rescues,
        "breaks": breaks,
        "break_rate": break_rate,
        "force_precision": force_precision,
        "rescue_recall": rescue_recall,
        "break_rate_upper_95": break_rate_upper_95,
        "n_beneficial": n_beneficial,
    }


def main():
    # Load models
    q_res_data = joblib.load(M4_DIR / "q_res_m4.pkl")
    q_res_model = q_res_data["model"]
    q_res_feature_keys = q_res_data["feature_keys"]

    risk_data = joblib.load(M4_DIR / "risk_model_m4.pkl")
    risk_model = risk_data["model"]
    risk_feature_keys = risk_data["feature_keys"]

    pairwise_model = None
    pairwise_feature_keys = None
    pairwise_path = M4_DIR / "pairwise_model_m4.pkl"
    if pairwise_path.exists():
        pairwise_data = joblib.load(pairwise_path)
        pairwise_model = pairwise_data["model"]
        pairwise_feature_keys = pairwise_data["feature_keys"]

    # Load conformal calibration for stratum quantiles at different alpha levels
    cal_data = json.loads(open(M4_DIR / "conformal_calibration_m4.json").read())

    # Load calibration records
    cal_records = load_m4_split("calibration")
    print(f"Calibration: {len(cal_records)} records")

    # Get stratum quantiles for each alpha level from the stratified calibration
    # We use the calibration_self stratified results (computed on calibration itself)
    # Actually, we need to compute stratum quantiles from calibration data directly
    # The conformal calibration output has structural_ood_stratified which uses cal→struct
    # For tuning, we need cal→cal (self-coverage) stratum quantiles

    # Compute stratum quantiles from calibration data directly
    cal_pairs_data = []
    groups = defaultdict(list)
    for i, r in enumerate(cal_records):
        groups[r["counterfactual_group_id"]].append((i, r))

    for gid, group in groups.items():
        if len(group) < 2:
            continue
        q_x_scores = []
        utilities = []
        for idx, rec in group:
            feats = extract_m4_features(rec)
            x = np.array([[feats[k] for k in q_res_feature_keys]])
            q_mb = compute_q_mb_from_record(rec)
            q_res = q_res_model.predict(x)[0]
            q_x = q_mb + q_res
            q_x_scores.append(q_x)
            utilities.append(rec["utility"])

        exec_idx = np.argmax(q_x_scores)
        base_idx = None
        for j, (_, rec) in enumerate(group):
            if "DEFER" in rec["first_action"]:
                base_idx = j
                break
        if base_idx is None:
            base_idx = 0

        delta_q_hat = q_x_scores[exec_idx] - q_x_scores[base_idx]
        delta_u = utilities[exec_idx] - utilities[base_idx]
        residual = abs(delta_q_hat - delta_u)

        exec_rec = group[exec_idx][1]
        exec_feats = extract_m4_features(exec_rec)
        graph_feats = exec_rec.get("graph_features", {})
        action_type = exec_rec.get("first_action_type", "UNKNOWN")
        action_class = "ANSWER" if action_type == "ANSWER" else "OTHER"
        belief_entropy = graph_feats.get("belief_entropy", 0.0)
        has_competition = graph_feats.get("topo_has_competition", 0.0)
        entropy_bin = "low" if belief_entropy < 1.0 else "high"
        competition_bin = "comp" if has_competition > 0.5 else "nocomp"
        stratum = f"{action_class}_{entropy_bin}_{competition_bin}"

        cal_pairs_data.append({
            "stratum": stratum,
            "residual": residual,
            "delta_q_hat": delta_q_hat,
            "delta_u": delta_u,
            "is_harmful": int(delta_u < 0),
        })

    # Compute per-stratum quantiles for each alpha level
    alpha_levels = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    cal_by_stratum = defaultdict(list)
    for p in cal_pairs_data:
        cal_by_stratum[p["stratum"]].append(p)

    all_stratum_quantiles = {}
    for alpha in alpha_levels:
        stratum_qs = {}
        cal_residuals_global = np.array([p["residual"] for p in cal_pairs_data])
        n_cal = len(cal_residuals_global)
        q_level_global = np.ceil(alpha * (n_cal + 1)) / n_cal
        q_level_global = min(q_level_global, 1.0)
        q_alpha_global = float(np.quantile(cal_residuals_global, q_level_global))

        for stratum, pairs in cal_by_stratum.items():
            if len(pairs) < 10:
                stratum_qs[stratum] = q_alpha_global
                continue
            residuals = np.array([p["residual"] for p in pairs])
            n = len(residuals)
            q_level = np.ceil(alpha * (n + 1)) / n
            q_level = min(q_level, 1.0)
            stratum_qs[stratum] = float(np.quantile(residuals, q_level))

        all_stratum_quantiles[alpha] = stratum_qs, q_alpha_global

    # Compute intervention data for calibration
    # We need to compute q_alpha_local for each alpha level
    # So we compute intervention data once, then evaluate for each alpha

    # First, compute base intervention data (without q_alpha_local)
    cal_interventions_raw = compute_intervention_data(
        cal_records, q_res_model, q_res_feature_keys,
        pairwise_model, pairwise_feature_keys,
        risk_model, risk_feature_keys,
        {}, 0.0  # placeholder, will be set per-alpha
    )

    # Sweep over configurations
    tau_deltas = [0.0, 5.0, 10.0, 15.0, 20.0]
    rhos = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    pairwise_thresholds = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
    target_break_upper = 0.05  # 5% upper bound on break rate

    print(f"\nTuning on calibration data only:")
    print(f"  Target: break_rate_upper_95 < {target_break_upper}")
    print(f"  Alpha levels: {alpha_levels}")
    print(f"  tau_delta values: {tau_deltas}")
    print(f"  rho values: {rhos}")
    print(f"  use_pairwise: {pairwise_model is not None}")
    print()

    best_config = None
    best_rescue_recall = -1
    all_configs = []

    gate_modes = ["all", "pairwise_risk", "lcb_risk"]
    if pairwise_model is not None:
        gate_modes.append("pairwise_only")

    for alpha in alpha_levels:
        stratum_qs, q_alpha_global = all_stratum_quantiles[alpha]

        # Set q_alpha_local for each intervention
        cal_interventions = []
        for iv in cal_interventions_raw:
            iv_copy = dict(iv)
            iv_copy["q_alpha_local"] = stratum_qs.get(iv["stratum"], q_alpha_global)
            cal_interventions.append(iv_copy)

        for gate_mode in gate_modes:
            for tau_delta in tau_deltas:
                for rho in rhos:
                    for pw_thresh in pairwise_thresholds:
                        use_pw = pairwise_model is not None and gate_mode != "lcb_risk"
                        result = evaluate_config(cal_interventions, tau_delta, rho,
                                                 use_pw, pw_thresh, gate_mode)

                        safe = (result["break_rate_upper_95"] is not None and
                                result["break_rate_upper_95"] < target_break_upper)
                        if result["n_force"] == 0:
                            safe = True  # vacuously safe

                        config = {
                            "alpha": alpha,
                            "tau_delta": tau_delta,
                            "rho": rho,
                            "use_pairwise": use_pw,
                            "pairwise_threshold": pw_thresh,
                            "gate_mode": gate_mode,
                            **result,
                            "safe": safe,
                        }
                        all_configs.append(config)

                        if safe and result["n_force"] > 0 and result["rescue_recall"] > best_rescue_recall:
                            best_rescue_recall = result["rescue_recall"]
                            best_config = config

    # Print top configurations
    print(f"{'alpha':>6} {'tau':>6} {'rho':>6} {'pw':>4} {'pw_th':>6} {'gate':>16} {'n_force':>8} {'rescues':>8} {'breaks':>7} {'brk_rate':>9} {'brk_upper':>10} {'recall':>7} {'safe':>5}")
    print("-" * 115)

    # Sort: safe configs first (by recall desc, force desc), then unsafe (by recall desc)
    useful_configs = [c for c in all_configs if c["n_force"] > 0]
    useful_configs.sort(key=lambda c: (-int(c["safe"]), -c["rescue_recall"], -c["n_force"]))

    for c in useful_configs[:25]:
        bu = f"{c['break_rate_upper_95']:.4f}" if c['break_rate_upper_95'] else "N/A"
        print(f"{c['alpha']:6.2f} {c['tau_delta']:6.1f} {c['rho']:6.2f} {'Y' if c['use_pairwise'] else 'N':>4} {c['pairwise_threshold']:6.1f} {c['gate_mode']:>16} {c['n_force']:8d} {c['rescues']:8d} {c['breaks']:7d} {c['break_rate']:9.4f} {bu:>10} {c['rescue_recall']:7.4f} {'✓' if c['safe'] else '✗':>5}")

    if best_config:
        print(f"\n{'='*60}")
        print(f"  BEST SAFE CONFIGURATION (calibration)")
        print(f"{'='*60}")
        print(f"  alpha:          {best_config['alpha']}")
        print(f"  tau_delta:      {best_config['tau_delta']}")
        print(f"  rho:            {best_config['rho']}")
        print(f"  use_pairwise:   {best_config['use_pairwise']}")
        print(f"  n_force:        {best_config['n_force']}")
        print(f"  rescues:        {best_config['rescues']}")
        print(f"  breaks:         {best_config['breaks']}")
        print(f"  break_rate:     {best_config['break_rate']:.4f}")
        bu = best_config['break_rate_upper_95']
        print(f"  break_upper_95: {bu:.4f}" if bu else "  break_upper_95: N/A")
        print(f"  rescue_recall:  {best_config['rescue_recall']:.4f}")
        print(f"  force_precision:{best_config['force_precision']:.4f}")
    else:
        print(f"\n  No safe configuration with n_force > 0 found.")
        print(f"  All configurations either have breaks or 0 interventions.")
        # Find the config with max force and 0 breaks
        zero_break_configs = [c for c in all_configs if c["breaks"] == 0 and c["n_force"] > 0]
        if zero_break_configs:
            best_zb = max(zero_break_configs, key=lambda c: c["n_force"])
            print(f"\n  Best zero-break config (max force):")
            print(f"    alpha={best_zb['alpha']}, tau={best_zb['tau_delta']}, rho={best_zb['rho']}, pw={best_zb['use_pairwise']}")
            print(f"    n_force={best_zb['n_force']}, recall={best_zb['rescue_recall']:.4f}")
            bu = best_zb['break_rate_upper_95']
            print(f"    break_upper_95={bu:.4f}" if bu else "    break_upper_95=N/A")
            best_config = best_zb

    # Save tuning results
    output = {
        "target_break_upper": target_break_upper,
        "best_config": best_config,
        "all_configs_count": len(all_configs),
        "calibration_n": len(cal_interventions_raw),
    }
    output_path = M4_DIR / "threshold_tuning_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
