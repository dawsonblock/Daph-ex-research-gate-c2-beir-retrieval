#!/usr/bin/env python3
"""Shadow authority experiment for DAPH-X M4.

Uses the LEARNED stack (Q_res, conformal LCB, risk model) — NOT placeholder
constants — to make FORCE decisions in shadow mode.

force_applied = 0 always (shadow only).
would_force = 1 when LCB_delta > tau_delta AND P(harm) < rho AND a_X in A_safe.

Evaluates against realized counterfactual truth:
  Rescues = #(would_force AND ΔU > 0)
  Breaks  = #(would_force AND ΔU < 0)
  BreakRate = P(ΔU < 0 | would_force)
  ForcePrecision = Rescues / (Rescues + Breaks)

Usage:
    python scripts/m4_shadow_authority.py
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


def run_shadow_authority(
    tau_delta: float = 0.0,  # LCB threshold
    rho: float = 0.05,        # Risk probability threshold (tuned on calibration)
    alpha: float = 0.50,      # Conformal coverage level (tuned)
    use_stratified: bool = True,  # Use stratified conformal quantiles
    use_pairwise: bool = True,    # Use pairwise model as additional gate
    pairwise_threshold: float = 0.0,  # Pairwise advantage threshold
    gate_mode: str = "pairwise_risk",  # Gate combination mode
):
    """Run shadow authority on structural_ood and mechanism_ood.

    Uses the LEARNED stack:
      - Q_res value model (boundary-weighted)
      - Pairwise advantage model (direct ΔU prediction)
      - Stratified conformal LCB
      - Intervention-risk model

    FORCE decision:
      would_force = 1 if LCB_delta > tau_delta
                        AND risk_prob < rho
                        AND pairwise_pred > 0  (if use_pairwise)
    """

    # Load models
    q_res_data = joblib.load(M4_DIR / "q_res_m4.pkl")
    q_res_model = q_res_data["model"]
    q_res_feature_keys = q_res_data["feature_keys"]

    risk_data = joblib.load(M4_DIR / "risk_model_m4.pkl")
    risk_model = risk_data["model"]
    risk_feature_keys = risk_data["feature_keys"]

    # Load pairwise model if available
    pairwise_model = None
    pairwise_feature_keys = None
    pairwise_path = M4_DIR / "pairwise_model_m4.pkl"
    if use_pairwise and pairwise_path.exists():
        pairwise_data = joblib.load(pairwise_path)
        pairwise_model = pairwise_data["model"]
        pairwise_feature_keys = pairwise_data["feature_keys"]

    # Load conformal calibration to get q_alpha
    cal_data = json.loads(open(M4_DIR / "conformal_calibration_m4.json").read())

    # Use stratified or global conformal quantiles
    cal_key = "structural_ood_stratified" if use_stratified else "structural_ood_global"
    struct_cal = cal_data["results"].get(cal_key, cal_data["results"].get("structural_ood_global", {}))
    q_alpha = struct_cal.get(f"coverage_{alpha:.2f}", {}).get("q_alpha_global",
                struct_cal.get(f"coverage_{alpha:.2f}", {}).get("q_alpha", 0.0))

    # For stratified, we need per-stratum quantiles
    stratum_quantiles = {}
    if use_stratified:
        strat_results = cal_data["results"].get("structural_ood_stratified", {})
        strat_90 = strat_results.get("coverage_0.90", {})
        for s_name, s_detail in strat_90.get("strata", {}).items():
            stratum_quantiles[s_name] = s_detail["q_alpha"]

    print(f"Shadow authority configuration:")
    print(f"  tau_delta (LCB threshold): {tau_delta}")
    print(f"  rho (risk threshold): {rho}")
    print(f"  alpha (conformal coverage): {alpha}")
    print(f"  q_alpha (global conformal quantile): {q_alpha}")
    print(f"  use_stratified: {use_stratified} ({len(stratum_quantiles)} strata)")
    print(f"  use_pairwise: {use_pairwise and pairwise_model is not None}")
    print(f"  pairwise_threshold: {pairwise_threshold}")
    print(f"  gate_mode: {gate_mode}")
    print()

    all_results = {}

    for split_name in ["structural_ood", "mechanism_ood"]:
        records = load_m4_split(split_name)
        if not records:
            continue

        # Group by counterfactual group
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r["counterfactual_group_id"]].append((i, r))

        interventions = []
        for gid, group in groups.items():
            if len(group) < 2:
                continue

            # Compute Q_X for each action
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

            # Executive action: argmax Q_X (learned model)
            exec_idx = np.argmax(q_x_scores)

            # Base action: DEFER
            base_idx = None
            for i, (_, rec) in enumerate(group):
                if "DEFER" in rec["first_action"]:
                    base_idx = i
                    break
            if base_idx is None:
                base_idx = 0

            delta_q_hat = q_x_scores[exec_idx] - q_x_scores[base_idx]
            delta_u = utilities[exec_idx] - utilities[base_idx]

            # Determine stratum-specific q_alpha
            exec_rec = group[exec_idx][1]
            exec_feats = extract_m4_features(exec_rec)
            graph_feats = exec_rec.get("graph_features", {})

            if use_stratified and stratum_quantiles:
                action_type = exec_rec.get("first_action_type", "UNKNOWN")
                action_class = "ANSWER" if action_type == "ANSWER" else "OTHER"
                belief_entropy = graph_feats.get("belief_entropy", 0.0)
                has_competition = graph_feats.get("topo_has_competition", 0.0)
                entropy_bin = "low" if belief_entropy < 1.0 else "high"
                competition_bin = "comp" if has_competition > 0.5 else "nocomp"
                stratum = f"{action_class}_{entropy_bin}_{competition_bin}"
                q_alpha_local = stratum_quantiles.get(stratum, q_alpha)
            else:
                q_alpha_local = q_alpha

            lcb_delta = delta_q_hat - q_alpha_local

            # Risk prediction
            risk_feats = {}
            risk_feats.update(exec_feats)
            risk_feats["q_mb_exec"] = float(q_mb_scores[exec_idx])
            risk_feats["q_mb_base"] = float(q_mb_scores[base_idx])
            risk_feats["delta_q_mb"] = float(q_mb_scores[exec_idx] - q_mb_scores[base_idx])

            x_risk = np.array([[risk_feats.get(k, 0.0) for k in risk_feature_keys]])
            risk_prob = float(risk_model.predict_proba(x_risk)[0, 1])

            # Pairwise advantage prediction
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

            # Shadow FORCE decision — gate_mode determines which conditions apply
            lcb_pass = lcb_delta > tau_delta
            risk_pass = risk_prob < rho
            pw_pass = pairwise_pred > pairwise_threshold

            if gate_mode == "all":
                force_conditions = [lcb_pass, risk_pass]
                if use_pairwise and pairwise_model is not None:
                    force_conditions.append(pw_pass)
            elif gate_mode == "pairwise_risk":
                force_conditions = [risk_pass]
                if use_pairwise and pairwise_model is not None:
                    force_conditions.append(pw_pass)
            elif gate_mode == "pairwise_only":
                force_conditions = [pw_pass] if (use_pairwise and pairwise_model is not None) else []
            elif gate_mode == "lcb_risk":
                force_conditions = [lcb_pass, risk_pass]
            else:
                force_conditions = [lcb_pass, risk_pass]

            would_force = int(all(force_conditions) if force_conditions else False)
            is_harmful = int(delta_u < 0)

            interventions.append({
                "group_id": gid,
                "exec_action": exec_rec["first_action"],
                "base_action": group[base_idx][1]["first_action"],
                "delta_q_hat": float(delta_q_hat),
                "lcb_delta": float(lcb_delta),
                "q_alpha_local": float(q_alpha_local),
                "risk_prob": risk_prob,
                "pairwise_pred": float(pairwise_pred),
                "delta_u": float(delta_u),
                "would_force": would_force,
                "is_harmful": is_harmful,
                "force_applied": 0,  # Always shadow
            })

        # Compute metrics
        n = len(interventions)
        n_force = sum(i["would_force"] for i in interventions)
        n_harmful = sum(i["is_harmful"] for i in interventions)

        rescues = sum(1 for i in interventions if i["would_force"] and not i["is_harmful"])
        breaks = sum(1 for i in interventions if i["would_force"] and i["is_harmful"])

        force_precision = rescues / max(n_force, 1)
        break_rate = breaks / max(n_force, 1)
        rescue_recall = n_force / max(n_harmful, 1) if n_harmful > 0 else 0.0
        # Actually recall = #(would_force AND ΔU > 0) / #(ΔU > 0)
        n_beneficial = sum(1 for i in interventions if not i["is_harmful"])
        rescue_recall = rescues / max(n_beneficial, 1) if n_beneficial > 0 else 0.0

        # Rule of three
        if breaks == 0 and n_force > 0:
            break_rate_upper_95 = 3.0 / n_force
        else:
            break_rate_upper_95 = None

        # 2x2 contingency
        both_correct = sum(1 for i in interventions if not i["is_harmful"] and not i["would_force"])
        base_correct_daph_wrong = sum(1 for i in interventions if not i["is_harmful"] and i["would_force"])
        base_wrong_daph_correct = sum(1 for i in interventions if i["is_harmful"] and not i["would_force"])
        both_wrong = sum(1 for i in interventions if i["is_harmful"] and i["would_force"])

        result = {
            "n": n,
            "n_force": n_force,
            "n_harmful": n_harmful,
            "rescues": rescues,
            "breaks": breaks,
            "force_precision": round(force_precision, 4),
            "break_rate": round(break_rate, 4),
            "rescue_recall": round(rescue_recall, 4),
            "break_rate_upper_95": round(break_rate_upper_95, 4) if break_rate_upper_95 else None,
            "contingency": {
                "both_correct": both_correct,
                "base_correct_daph_wrong": base_correct_daph_wrong,
                "base_wrong_daph_correct": base_wrong_daph_correct,
                "both_wrong": both_wrong,
            },
            "force_applied": 0,  # Shadow only
        }
        all_results[split_name] = result

        print(f"{'='*60}")
        print(f"  {split_name.upper()} — Shadow Authority")
        print(f"{'='*60}")
        print(f"  Total interventions: {n}")
        print(f"  Would FORCE: {n_force}/{n}")
        print(f"  Rescues (force & beneficial): {rescues}")
        print(f"  Breaks (force & harmful): {breaks}")
        print(f"  Force Precision: {force_precision:.4f}")
        print(f"  Break Rate: {break_rate:.4f}")
        print(f"  Rescue Recall: {rescue_recall:.4f}")
        if break_rate_upper_95:
            print(f"  Break Rate 95% Upper: {break_rate_upper_95:.4f}")
        print(f"  Contingency:")
        print(f"    Both correct: {both_correct}")
        print(f"    Base correct, DAPH wrong: {base_correct_daph_wrong}")
        print(f"    Base wrong, DAPH correct: {base_wrong_daph_correct}")
        print(f"    Both wrong: {both_wrong}")
        print()

    # Save
    output = {
        "config": {
            "tau_delta": tau_delta,
            "rho": rho,
            "alpha": alpha,
            "q_alpha": q_alpha,
            "use_stratified": use_stratified,
            "use_pairwise": use_pairwise and pairwise_model is not None,
            "pairwise_threshold": pairwise_threshold,
            "gate_mode": gate_mode,
        },
        "results": all_results,
        "force_applied": 0,
        "note": "Shadow authority using boundary-weighted Q_res + stratified conformal + risk model + pairwise advantage gate. Thresholds tuned on calibration only. force_applied=0 always.",
    }
    output_path = M4_DIR / "shadow_authority_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_shadow_authority()
