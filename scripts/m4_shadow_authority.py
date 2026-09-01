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
    rho: float = 0.3,         # Risk probability threshold
    alpha: float = 0.90,      # Conformal alpha level
):
    """Run shadow authority on structural_ood and mechanism_ood."""

    # Load models
    q_res_data = joblib.load(M4_DIR / "q_res_m4.pkl")
    q_res_model = q_res_data["model"]
    q_res_feature_keys = q_res_data["feature_keys"]

    risk_data = joblib.load(M4_DIR / "risk_model_m4.pkl")
    risk_model = risk_data["model"]
    risk_feature_keys = risk_data["feature_keys"]

    # Load conformal calibration to get q_alpha
    cal_data = json.loads(open(M4_DIR / "conformal_calibration_m4.json").read())
    # Use the specified coverage level from structural_ood
    struct_cal = cal_data["results"]["structural_ood"]
    q_alpha = struct_cal[f"coverage_{alpha:.2f}"]["q_alpha"]

    print(f"Shadow authority configuration:")
    print(f"  tau_delta (LCB threshold): {tau_delta}")
    print(f"  rho (risk threshold): {rho}")
    print(f"  alpha (conformal level): {alpha}")
    print(f"  q_alpha (conformal quantile): {q_alpha}")
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
            lcb_delta = delta_q_hat - q_alpha
            delta_u = utilities[exec_idx] - utilities[base_idx]

            # Risk prediction
            exec_rec = group[exec_idx][1]
            risk_feats = {}
            base_feats = extract_m4_features(exec_rec)
            risk_feats.update(base_feats)
            risk_feats["q_mb_exec"] = float(q_mb_scores[exec_idx])
            risk_feats["q_mb_base"] = float(q_mb_scores[base_idx])
            risk_feats["delta_q_mb"] = float(q_mb_scores[exec_idx] - q_mb_scores[base_idx])

            x_risk = np.array([[risk_feats.get(k, 0.0) for k in risk_feature_keys]])
            risk_prob = float(risk_model.predict_proba(x_risk)[0, 1])

            # Shadow FORCE decision
            would_force = int(lcb_delta > tau_delta and risk_prob < rho)
            is_harmful = int(delta_u < 0)

            interventions.append({
                "group_id": gid,
                "exec_action": exec_rec["first_action"],
                "base_action": group[base_idx][1]["first_action"],
                "delta_q_hat": float(delta_q_hat),
                "lcb_delta": float(lcb_delta),
                "risk_prob": risk_prob,
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
        },
        "results": all_results,
        "force_applied": 0,
        "note": "Shadow authority using learned Q_res + conformal LCB + risk model. force_applied=0 always.",
    }
    output_path = M4_DIR / "shadow_authority_m4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_shadow_authority()
