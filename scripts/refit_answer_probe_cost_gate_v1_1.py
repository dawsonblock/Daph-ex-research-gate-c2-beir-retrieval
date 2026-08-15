#!/usr/bin/env python3
"""ANSWER_PROBE_COST_GATE_V1.1 controller refit, per
configs/gate_answer_probe_cost_v1_1_design.json.

DESIGN USE ONLY -- runs against the CONSUMED exec_training_v2 receipts,
which permit diagnosis_only / hypothesis_generation_only. Controller design
is hypothesis generation. NO PROMOTION CLAIM IS MADE HERE.

Fits P(memory improves quality) on TRAIN, selects TAU on CALIBRATION as the
LARGEST operating point whose calibration quality loss stays within the
frozen margin (i.e. the most compute saved subject to the quality
constraint), and then reports a DESIGN-TIME ESTIMATE on a development
holdout against the five frozen PHASE_5' criteria -- so the decision to
spend GPU on a fresh evaluation split is itself evidence-based.

Writes the frozen controller + TAU that PHASE_5 must load unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_probe_cost_separation import FEATURES, load_records  # noqa: E402
from scripts.diagnose_c5_confirmation_stopgate import grouped_lcb  # noqa: E402
from scripts.fit_answer_probe_cost_gate import (  # noqa: E402
    ESCALATE_CLASS_EXTRA_WEIGHT, fit_logistic, logistic_proba, stratified_dev_split)

# --- FROZEN in configs/gate_answer_probe_cost_v1_1_design.json ------------
QUALITY_NONINFERIORITY_MARGIN = 0.02
REQUIRED_RELATIVE_REDUCTION = 0.10
SAFETY_BOUND = 0.30
EVAL_CLASS_FLOORS = {"MEMORY_preferred": 40, "ANSWER_preferred": 40}
#: Half the promotion margin, reserved as generalization headroom -- see the
#: AMENDMENT_..._tau_selection_buffer block in the frozen design doc. Selecting
#: the argmax exactly at a finite-sample constraint boundary violates that
#: constraint out-of-sample ~half the time. Makes C2 harder, loosens nothing.
TAU_SELECTION_BUDGET = QUALITY_NONINFERIORITY_MARGIN / 2.0


def feats(r):
    return np.array([r[f] for f in FEATURES], float)


def policy_outcome(records, decisions):
    """(mean_quality, mean_tokens) for a decision vector."""
    q = sum(r["q_memory"] if d == "ESCALATE" else r["q_direct"] for r, d in zip(records, decisions))
    c = sum(r["c_memory"] if d == "ESCALATE" else r["c_direct"] for r, d in zip(records, decisions))
    n = len(records) or 1
    return q / n, c / n


def select_tau(calib, score_fn, margin):
    """FROZEN RULE: the LARGEST tau (=> most ACCEPTs => most compute saved)
    whose calibration quality loss vs always-escalate stays within margin.
    Higher tau accepts more, so we scan descending and take the first that
    satisfies the constraint."""
    q_escalate, c_escalate = policy_outcome(calib, ["ESCALATE"] * len(calib))
    candidates = sorted({score_fn(r) for r in calib}, reverse=True)
    best = None
    for tau in candidates:
        dec = ["ESCALATE" if score_fn(r) >= tau else "ACCEPT" for r in calib]
        q, c = policy_outcome(calib, dec)
        if q_escalate - q <= margin:
            saving = c_escalate - c
            if best is None or saving > best[1]:
                best = (tau, saving, q, c)
    # tau above every score => accept everything; guaranteed to exist as a
    # fallback only if it satisfies the constraint, else fall back to
    # always-escalate (tau below every score).
    if best is None:
        return min(candidates) if candidates else 0.0, 0.0
    return best[0], best[1]


def evaluate_against_frozen_criteria(records, score_fn, tau, label):
    dec = ["ESCALATE" if score_fn(r) >= tau else "ACCEPT" for r in records]
    q_cand, c_cand = policy_outcome(records, dec)
    q_p0, c_p0 = policy_outcome(records, ["ACCEPT"] * len(records))
    q_p1, c_p1 = policy_outcome(records, ["ESCALATE"] * len(records))

    # best fixed policy under the SAME preference ordering: quality first,
    # cheaper on an exact quality tie.
    if q_p1 > q_p0:
        q_fix, c_fix, fix_name = q_p1, c_p1, "P1_always_escalate"
    elif q_p0 > q_p1:
        q_fix, c_fix, fix_name = q_p0, c_p0, "P0_always_accept"
    else:
        q_fix, c_fix, fix_name = (q_p0, c_p0, "P0_always_accept") if c_p0 <= c_p1 else (q_p1, c_p1, "P1_always_escalate")

    dmap = {r["key"]: d for r, d in zip(records, dec)}
    savings_pairs = []
    for r in records:
        c_c = r["c_memory"] if dmap[r["key"]] == "ESCALATE" else r["c_direct"]
        c_f = r["c_memory"] if fix_name == "P1_always_escalate" else r["c_direct"]
        savings_pairs.append((r["family"], float(c_f - c_c)))
    lcb = grouped_lcb(savings_pairs)

    mem_better = [r for r in records if r["q_memory"] > r["q_direct"]]
    p_acc = (sum(1 for r in mem_better if dmap[r["key"]] == "ACCEPT") / len(mem_better)
             if mem_better else float("nan"))

    rel_red = (c_fix - c_cand) / c_fix if c_fix else 0.0
    c1 = q_cand >= q_fix - QUALITY_NONINFERIORITY_MARGIN
    c2 = rel_red >= REQUIRED_RELATIVE_REDUCTION
    c3 = lcb is not None and lcb > 0.0
    c4 = (p_acc <= SAFETY_BOUND) if mem_better else False
    n_mem_pref = sum(1 for r in records if r["delta_u_cost"] > 0)
    n_ans_pref = len(records) - n_mem_pref
    c5 = (n_mem_pref >= EVAL_CLASS_FLOORS["MEMORY_preferred"]
          and n_ans_pref >= EVAL_CLASS_FLOORS["ANSWER_preferred"])

    res = {
        "label": label, "tau": tau,
        "best_fixed": fix_name, "quality_best_fixed": q_fix, "tokens_best_fixed": c_fix,
        "quality_candidate": q_cand, "tokens_candidate": c_cand,
        "quality_loss": q_fix - q_cand, "relative_token_reduction": rel_red,
        "token_saving_lcb_2p5": lcb,
        "P_accept_given_memory_better": p_acc,
        "n_MEMORY_preferred": n_mem_pref, "n_ANSWER_preferred": n_ans_pref,
        "C1_quality_non_inferiority": bool(c1),
        "C2_material_compute_reduction": bool(c2),
        "C3_reduction_statistically_real": bool(c3),
        "C4_safety_bound": bool(c4),
        "C5_eval_class_floors": bool(c5),
        "ALL_CRITERIA_MET": bool(c1 and c2 and c3 and c4 and c5),
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()

    records = load_records(args.receipts)
    print("=== ANSWER_PROBE_COST_GATE_V1.1  controller refit ===")
    print("    DESIGN USE ONLY -- consumed split. No promotion claim.\n")
    print(f"  frozen: margin={QUALITY_NONINFERIORITY_MARGIN}  min_reduction={REQUIRED_RELATIVE_REDUCTION}"
          f"  safety={SAFETY_BOUND}  floors={EVAL_CLASS_FLOORS}")
    print(f"  tau selection budget (generalization headroom) = {TAU_SELECTION_BUDGET}\n")

    train, calib, holdout = stratified_dev_split(records)
    print(f"  dev split: train={len(train)} calibration={len(calib)} holdout={len(holdout)}")

    X = np.stack([feats(r) for r in train])
    y = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in train])
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    w_pos = (len(y) / (2.0 * n_pos)) * ESCALATE_CLASS_EXTRA_WEIGHT if n_pos else 0.0
    w_neg = len(y) / (2.0 * n_neg) if n_neg else 0.0
    sw = np.array([w_pos if v else w_neg for v in y])
    w, b, m, s = fit_logistic(X, y, sample_weight=sw)
    score = lambda r: logistic_proba(feats(r), w, b, m, s)  # noqa: E731

    tau, calib_saving = select_tau(calib, score, TAU_SELECTION_BUDGET)
    print(f"  TAU selected on CALIBRATION = {tau:.6f}  (budget={TAU_SELECTION_BUDGET}, "
          f"calibration token saving {calib_saving:.1f}/task)\n")

    # --- attainability frontier on the dev holdout (design-time only) -----
    print("  --- achievable frontier on dev holdout (design-time estimate) ---")
    print(f"  {'tau':>10}{'quality':>10}{'q_loss':>9}{'tokens':>9}{'reduction':>11}")
    q_p1, c_p1 = policy_outcome(holdout, ["ESCALATE"] * len(holdout))
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        dec = ["ESCALATE" if score(r) >= t else "ACCEPT" for r in holdout]
        q, c = policy_outcome(holdout, dec)
        print(f"  {t:>10.2f}{q:>10.4f}{q_p1-q:>9.4f}{c:>9.1f}{(c_p1-c)/c_p1:>10.1%}")

    print(f"\n  --- DESIGN-TIME ESTIMATE at the frozen TAU={tau:.6f} ---")
    est = evaluate_against_frozen_criteria(holdout, score, tau, "dev_holdout_estimate")
    for k in ("best_fixed", "quality_best_fixed", "quality_candidate", "quality_loss",
              "tokens_best_fixed", "tokens_candidate", "relative_token_reduction",
              "token_saving_lcb_2p5", "P_accept_given_memory_better"):
        v = est[k]
        print(f"    {k:<34}{v if isinstance(v, str) else f'{v:.4f}' if isinstance(v, float) else v}")
    print()
    for k in ("C1_quality_non_inferiority", "C2_material_compute_reduction",
              "C3_reduction_statistically_real", "C4_safety_bound", "C5_eval_class_floors"):
        print(f"    {k:<34}{est[k]}")
    print(f"    {'ALL_CRITERIA_MET':<34}{est['ALL_CRITERIA_MET']}")
    print("\n    (Development holdout, consumed split -- an ESTIMATE of attainability,")
    print("     NOT a promotion claim. PHASE_5 is one-shot on a fresh split.)")

    out = {
        "artifact": "ANSWER_PROBE_COST_GATE_V1.1",
        "design": "configs/gate_answer_probe_cost_v1_1_design.json",
        "DESIGN_USE_ONLY": True, "promotion_claim_made": False,
        "frozen_criteria": {
            "QUALITY_NONINFERIORITY_MARGIN": QUALITY_NONINFERIORITY_MARGIN,
            "REQUIRED_RELATIVE_REDUCTION": REQUIRED_RELATIVE_REDUCTION,
            "SAFETY_BOUND": SAFETY_BOUND, "EVAL_CLASS_FLOORS": EVAL_CLASS_FLOORS,
        },
        "frozen_controller": {
            "model": "cost_sensitive_logistic (P3 lineage)",
            "features": FEATURES, "weights": w.tolist(), "bias": b,
            "feature_mean": m.tolist(), "feature_std": s.tolist(),
            "class_weights": {"positive": w_pos, "negative": w_neg},
            "TAU": tau, "tau_selected_on": "CALIBRATION",
        },
        "design_time_estimate_on_dev_holdout": est,
    }
    out_path = Path(args.receipts).with_suffix(".cost_gate_v1_1_frozen.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written (controller + TAU FROZEN for PHASE_5): {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
