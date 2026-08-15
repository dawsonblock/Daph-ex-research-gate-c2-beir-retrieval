#!/usr/bin/env python3
"""ANSWER_PROBE_COST_GATE_V1 PHASE_3: fit + freeze the controller.

Per configs/gate_answer_probe_cost_v1_design.json. GPU-free.

DESIGN USE ONLY. This runs against the CONSUMED exec_training_v2 receipts,
so nothing here is a promotion claim -- its purposes are (a) to fit P2/P3/P4
and freeze their decision thresholds, and (b) to produce a DESIGN-TIME
ESTIMATE on a development holdout, so the decision to spend GPU on a fresh
evaluation split (PHASE_4) is itself evidence-based rather than hopeful.

Development split mechanics (60/20/20 train/calibration/dev-holdout,
stratified by family x label, seed 9902) are a design-time implementation
detail, not a promotion-affecting parameter: the actual promotion test in
PHASE_5 runs once against a completely fresh split that does not exist yet.
They are fixed here before any model is fit.

The frozen artifacts this writes (weights, thresholds, feature
standardization) are what PHASE_5 must load unchanged -- PHASE_5 refits
nothing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_probe_cost_separation import (  # noqa: E402
    C_REF, EPSILON, FEATURES, LAMBDA_T_GRID, load_records)
from scripts.diagnose_c5_confirmation_stopgate import grouped_lcb  # noqa: E402

DEV_SPLIT_SEED = 9902
DEV_SHARES = (0.60, 0.20, 0.20)
#: carried over UNCHANGED from configs/gate_answer_probe_v2_design.json
ESCALATE_CLASS_EXTRA_WEIGHT = 2.0
SAFETY_BOUND = 0.30


def stratified_dev_split(records):
    rng = random.Random(DEV_SPLIT_SEED)
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["suite_family"], r["delta_u_cost"] > 0)].append(r)
    train, calib, holdout = [], [], []
    for _key, recs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        recs = recs[:]
        rng.shuffle(recs)
        n_tr = round(len(recs) * DEV_SHARES[0])
        n_ca = round(len(recs) * DEV_SHARES[1])
        train += recs[:n_tr]
        calib += recs[n_tr:n_tr + n_ca]
        holdout += recs[n_tr + n_ca:]
    return train, calib, holdout


def fit_logistic(X, y, sample_weight=None, lr=0.1, iterations=3000, seed=7):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    w_arr = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    w_sum = w_arr.sum()
    rng = np.random.RandomState(seed)
    w, b = rng.normal(scale=0.01, size=Xs.shape[1]), 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        resid = w_arr * (p - y)
        w -= lr * (Xs.T @ resid / w_sum)
        b -= lr * float(resid.sum() / w_sum)
    return w, b, mean, std


def logistic_proba(x, w, b, mean, std):
    return 1.0 / (1.0 + np.exp(-float(((x - mean) / std) @ w + b)))


def fit_ols(X, y):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    coef, *_ = np.linalg.lstsq(np.hstack([Xs, np.ones((len(y), 1))]), y, rcond=None)
    return coef[:-1], float(coef[-1]), mean, std


def ols_predict(x, w, b, mean, std):
    return float(((x - mean) / std) @ w + b)


def utility_of(records, decisions, lambda_t=None):
    """Frozen PRIMARY utility unless lambda_t is given (SECONDARY grid, diagnostic)."""
    tot = 0.0
    for r, d in zip(records, decisions):
        q = r["q_memory"] if d == "ESCALATE" else r["q_direct"]
        c = r["c_memory"] if d == "ESCALATE" else r["c_direct"]
        tot += (q - lambda_t * c / 1000.0) if lambda_t is not None else (q - EPSILON * c / C_REF)
    return tot / len(records) if records else 0.0


def best_threshold(records, score_fn):
    scores = sorted({score_fn(r) for r in records})
    best_t, best_u = (scores[0] if scores else 0.0), -1e18
    for t in scores:
        u = utility_of(records, ["ESCALATE" if score_fn(r) >= t else "ACCEPT" for r in records])
        if u > best_u:
            best_u, best_t = u, t
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()

    records = load_records(args.receipts)
    print("=== ANSWER_PROBE_COST_GATE_V1  PHASE_3: fit + freeze controller ===")
    print("    DESIGN USE ONLY -- consumed split. Numbers below are a DESIGN-TIME")
    print("    ESTIMATE, not a promotion claim. Promotion = PHASE_5, fresh split.\n")

    train, calib, holdout = stratified_dev_split(records)
    print(f"  dev split (seed {DEV_SPLIT_SEED}): train={len(train)} calibration={len(calib)} holdout={len(holdout)}")

    X_tr = np.stack([r["x"] for r in train]) if "x" in train[0] else np.stack(
        [np.array([r[f] for f in FEATURES], float) for r in train])
    y_tr = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in train])
    n_pos, n_neg = int(y_tr.sum()), int((1 - y_tr).sum())
    print(f"  train labels: MEMORY_preferred={n_pos}  ANSWER_preferred={n_neg}\n")

    def feats(r):
        return np.array([r[f] for f in FEATURES], float)

    conf_i = FEATURES.index("mean_token_confidence")

    # --- P2: single confidence threshold, frozen on CALIBRATION -----------
    # escalate when confidence is LOW -> score on negated confidence
    p2_t = best_threshold(calib, lambda r: -r[FEATURES[conf_i]])

    # --- P3: cost-sensitive logistic (PRIMARY), threshold on CALIBRATION --
    w_pos = (len(y_tr) / (2.0 * n_pos)) * ESCALATE_CLASS_EXTRA_WEIGHT if n_pos else 0.0
    w_neg = len(y_tr) / (2.0 * n_neg) if n_neg else 0.0
    sw = np.array([w_pos if y else w_neg for y in y_tr])
    w3, b3, m3, s3 = fit_logistic(X_tr, y_tr, sample_weight=sw)
    p3_t = best_threshold(calib, lambda r: logistic_proba(feats(r), w3, b3, m3, s3))

    # --- P4: value regression (CO-PRIMARY), threshold on CALIBRATION ------
    y_tr_d = np.array([r["delta_u_cost"] for r in train], float)
    w4, b4, m4, s4 = fit_ols(X_tr, y_tr_d)
    p4_t = best_threshold(calib, lambda r: ols_predict(feats(r), w4, b4, m4, s4))

    policies = {
        "P0_always_accept_direct": lambda r: "ACCEPT",
        "P1_always_escalate_to_memory": lambda r: "ESCALATE",
        "P2_frozen_confidence_threshold": lambda r: "ESCALATE" if -r[FEATURES[conf_i]] >= p2_t else "ACCEPT",
        "P3_cost_sensitive_logistic": lambda r: "ESCALATE" if logistic_proba(feats(r), w3, b3, m3, s3) >= p3_t else "ACCEPT",
        "P4_value_regression": lambda r: "ESCALATE" if ols_predict(feats(r), w4, b4, m4, s4) >= p4_t else "ACCEPT",
        "P5_oracle": lambda r: "ESCALATE" if r["delta_u_cost"] > 0 else "ACCEPT",
    }

    print("  --- DESIGN-TIME ESTIMATE on development holdout ---")
    print(f"  {'policy':<34}{'U':>10}{'meanTok':>10}{'quality':>9}")
    dec = {name: [fn(r) for r in holdout] for name, fn in policies.items()}
    utils = {}
    for name, d in dec.items():
        u = utility_of(holdout, d)
        utils[name] = u
        tok = sum(r["c_memory"] if x == "ESCALATE" else r["c_direct"] for r, x in zip(holdout, d)) / len(holdout)
        qual = sum(r["q_memory"] if x == "ESCALATE" else r["q_direct"] for r, x in zip(holdout, d)) / len(holdout)
        print(f"  {name:<34}{u:>10.4f}{tok:>10.1f}{qual:>9.4f}")

    best_fixed = max(utils["P0_always_accept_direct"], utils["P1_always_escalate_to_memory"])
    fixed_is_accept = utils["P0_always_accept_direct"] >= utils["P1_always_escalate_to_memory"]
    print(f"\n  best fixed = {best_fixed:.4f} ({'P0 always-accept' if fixed_is_accept else 'P1 always-escalate'})")

    print(f"\n  {'policy':<34}{'DeltaU_gate':>13}{'LCB2.5':>10}{'P(ACC|mem better)':>19}{'safety':>8}")
    stats = {}
    for name in ("P2_frozen_confidence_threshold", "P3_cost_sensitive_logistic", "P4_value_regression"):
        d = dec[name]
        dmap = {r["key"]: x for r, x in zip(holdout, d)}
        pairs = []
        for r in holdout:
            cu = r["u_memory"] if dmap[r["key"]] == "ESCALATE" else r["u_direct"]
            fu = r["u_direct"] if fixed_is_accept else r["u_memory"]
            pairs.append((r["family"], float(cu - fu)))
        lcb = grouped_lcb(pairs)
        mem_better = [r for r in holdout if r["q_memory"] > r["q_direct"]]
        p_acc = (sum(1 for r in mem_better if dmap[r["key"]] == "ACCEPT") / len(mem_better)
                 if mem_better else float("nan"))
        safe = p_acc <= SAFETY_BOUND if mem_better else False
        dug = utils[name] - best_fixed
        stats[name] = {"delta_u_gate": dug, "lcb": lcb, "p_accept_given_memory_better": p_acc,
                       "safety_bound_met": bool(safe)}
        print(f"  {name:<34}{dug:>13.4f}{str(lcb):>10}{p_acc:>19.4f}{str(bool(safe)):>8}")

    # --- MANDATORY per-family decomposition -------------------------------
    print("\n  --- MANDATORY per-family decomposition (is the gain real routing?) ---")
    per_family = {}
    for fam in sorted({r["suite_family"] for r in holdout}):
        sub = [r for r in holdout if r["suite_family"] == fam]
        if not sub:
            continue
        u_p0 = utility_of(sub, ["ACCEPT"] * len(sub))
        u_p1 = utility_of(sub, ["ESCALATE"] * len(sub))
        bf = max(u_p0, u_p1)
        row = {"n": len(sub), "U_P0": u_p0, "U_P1": u_p1, "best_fixed": bf}
        print(f"    {fam}  (n={len(sub)})   P0={u_p0:.4f}  P1={u_p1:.4f}  best_fixed={bf:.4f}")
        for name in ("P3_cost_sensitive_logistic", "P4_value_regression"):
            u = utility_of(sub, [policies[name](r) for r in sub])
            row[name] = {"U": u, "delta_vs_best_fixed": u - bf}
            print(f"        {name:<32}U={u:.4f}  Delta_vs_best_fixed={u - bf:+.4f}")
        per_family[fam] = row

    # --- SECONDARY sensitivity grid (diagnostic) --------------------------
    print("\n  --- SECONDARY lambda_T sensitivity grid (diagnostic only) ---")
    grid = {}
    for lam in LAMBDA_T_GRID:
        u0 = utility_of(holdout, dec["P0_always_accept_direct"], lambda_t=lam)
        u1 = utility_of(holdout, dec["P1_always_escalate_to_memory"], lambda_t=lam)
        u3 = utility_of(holdout, dec["P3_cost_sensitive_logistic"], lambda_t=lam)
        grid[str(lam)] = {"P0": u0, "P1": u1, "P3": u3, "delta_vs_best_fixed": u3 - max(u0, u1)}
        print(f"    lambda_T={lam:<6} P0={u0:>8.4f} P1={u1:>8.4f} P3={u3:>8.4f}"
              f"  Delta(P3 - best_fixed)={u3 - max(u0,u1):+.4f}")

    frozen = {
        "artifact": "ANSWER_PROBE_COST_GATE_V1",
        "design": "configs/gate_answer_probe_cost_v1_design.json",
        "DESIGN_USE_ONLY": True, "promotion_claim_made": False,
        "dev_split": {"seed": DEV_SPLIT_SEED, "shares": list(DEV_SHARES),
                      "n_train": len(train), "n_calibration": len(calib), "n_holdout": len(holdout)},
        "features": FEATURES,
        "frozen_controllers": {
            "P2_frozen_confidence_threshold": {"score": "-mean_token_confidence", "threshold": p2_t},
            "P3_cost_sensitive_logistic": {
                "weights": w3.tolist(), "bias": b3, "feature_mean": m3.tolist(),
                "feature_std": s3.tolist(), "threshold": p3_t,
                "class_weights": {"positive": w_pos, "negative": w_neg},
                "escalate_class_extra_weight": ESCALATE_CLASS_EXTRA_WEIGHT},
            "P4_value_regression": {
                "weights": w4.tolist(), "bias": b4, "feature_mean": m4.tolist(),
                "feature_std": s4.tolist(), "threshold": p4_t},
        },
        "design_time_estimate_on_dev_holdout": {
            "eval_utilities": utils, "best_fixed": best_fixed,
            "best_fixed_policy": "P0_always_accept_direct" if fixed_is_accept else "P1_always_escalate_to_memory",
            "gate_stats": stats, "per_family_decomposition": per_family,
            "lambda_T_sensitivity_grid": grid,
        },
    }
    out_path = Path(args.receipts).with_suffix(".cost_gate_frozen.json")
    out_path.write_text(json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written (controller FROZEN for PHASE_5): {out_path}")
    print("  PHASE_5 must load these parameters unchanged and refit nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
