#!/usr/bin/env python3
"""ANSWER_PROBE_GATE_V2: outcome-stratified split + model ladder + promotion.

Per configs/gate_answer_probe_v2_design.json PHASE_3/4/5. GPU-free --
operates entirely on the receipts written by
scripts/run_exec_training_v2_collection.py --execute.

No separate separation stop-gate here (unlike V1): V1's own receipts already
established the confidence signal separates MEMORY-strict-win from
ANSWER-strict-win (4/5 features cleared the frozen threshold), which is the
whole reason V2 exists. This script picks up at PHASE_3.

Order matters and mirrors the frozen design exactly:
  1. Load records, compute delta_u_i per task.
  2. PHASE_3: outcome-stratified train/calibration/eval split (seed 9901,
     per-stratum ratios frozen in the design doc).
  3. PHASE_5's minimum-class-floor check, evaluated BEFORE any fitting --
     if unmet, STOP at INCONCLUSIVE_INSUFFICIENT_EVAL_CLASS_SUPPORT and do
     not fit anything. Not a reshuffle-and-retry condition.
  4. PHASE_4: fit V0-V5 + VR1, with V1/V2/V3/VR1's decision thresholds (and
     V3's realized class-weight value) frozen on CALIBRATION, never on
     TRAIN or EVAL.
  5. PHASE_5: promotion requires ALL of the utility condition (Delta_U_gate>0,
     LCB>0), the hard safety bound (P(ACCEPT|delta_u>0)<=0.30), and the
     class floors already checked in step 3.
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

from scripts.diagnose_c5_confirmation_stopgate import grouped_lcb  # noqa: E402

FEATURES = [
    "mean_token_confidence", "min_token_confidence", "sequence_confidence",
    "mean_entropy", "answer_length",
]
SPLIT_SEED = 9901
#: Per-stratum (train, calibration, eval) shares -- frozen in
#: configs/gate_answer_probe_v2_design.json PHASE_3, before any V2 data existed.
STRATUM_SPLIT = {
    "MEMORY_strict_win": (0.60, 0.20, 0.20),
    "TIE": (0.60, 0.20, 0.20),
    "ANSWER_strict_win": (0.40, 0.25, 0.35),
}
MIN_EVAL_FLOOR = {"ANSWER_strict_win": 10, "MEMORY_strict_win": 40}
SAFETY_BOUND_P_ACCEPT_GIVEN_MEMORY_BETTER = 0.30
ESCALATE_CLASS_EXTRA_WEIGHT = 2.0


def load_records(receipts_path: str) -> list[dict]:
    receipts = [json.loads(l) for l in open(receipts_path) if l.strip()]
    by_task: dict[tuple, dict] = defaultdict(dict)
    family_of: dict[tuple, str] = {}
    suite_family_of: dict[tuple, str] = {}
    for r in receipts:
        key = (r["suite_family"], r["task_id"])
        by_task[key][r["action"]] = r
        family_of[key] = r["family"]
        suite_family_of[key] = r["suite_family"]

    records = []
    for key, actions in by_task.items():
        if "A0_ANSWER_NOW" not in actions or "A1_USE_CERTIFIED_MEMORY" not in actions:
            continue
        a0, a1 = actions["A0_ANSWER_NOW"], actions["A1_USE_CERTIFIED_MEMORY"]
        q0, q1 = int(a0["correct"]), int(a1["correct"])
        delta_u = q1 - q0
        stratum = ("MEMORY_strict_win" if delta_u > 0 else
                  "ANSWER_strict_win" if delta_u < 0 else "TIE")
        records.append({
            "key": key, "family": family_of[key], "suite_family": suite_family_of[key],
            "q_direct": q0, "q_memory": q1, "delta_u": delta_u, "stratum": stratum,
            "x": np.array([a0[f] for f in FEATURES], dtype=float),
        })
    return records


def outcome_stratified_split(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """PHASE_3: split each outcome stratum independently per STRATUM_SPLIT,
    seed SPLIT_SEED. Mechanical -- applied to whatever labels Phase 2 produced,
    zero human selection of which examples land where."""
    rng = random.Random(SPLIT_SEED)
    train, calib, eval_ = [], [], []
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_stratum[r["stratum"]].append(r)
    for stratum, recs in by_stratum.items():
        recs = recs[:]
        rng.shuffle(recs)
        train_share, calib_share, _eval_share = STRATUM_SPLIT[stratum]
        n_train = round(len(recs) * train_share)
        n_calib = round(len(recs) * calib_share)
        train.extend(recs[:n_train])
        calib.extend(recs[n_train:n_train + n_calib])
        eval_.extend(recs[n_train + n_calib:])
    return train, calib, eval_


def utility(records: list[dict], decisions: list[str]) -> float:
    total = 0
    for r, d in zip(records, decisions):
        total += r["q_direct"] if d == "ACCEPT" else r["q_memory"]
    return total / len(records) if records else 0.0


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
                            lr: float = 0.1, iterations: int = 2000, seed: int = 7,
                            ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Weighted gradient-descent logistic regression, y in {0,1} (1 = escalate
    improves, i.e. delta_u>0). Standardizes on TRAIN only. Unweighted (V2)
    when sample_weight is None; weighted (V3) otherwise."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std

    n = len(y)
    w_arr = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w_sum = w_arr.sum()

    rng = np.random.RandomState(seed)
    w = rng.normal(scale=0.01, size=Xs.shape[1])
    b = 0.0
    for _ in range(iterations):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        resid = w_arr * (p - y)
        grad_w = Xs.T @ resid / w_sum
        grad_b = float(resid.sum() / w_sum)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mean, std


def logistic_predict_proba(x: np.ndarray, w: np.ndarray, b: float,
                           mean: np.ndarray, std: np.ndarray) -> float:
    xs = (x - mean) / std
    z = float(xs @ w + b)
    return 1.0 / (1.0 + np.exp(-z))


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """Standard inverse-frequency 'balanced' weighting, computed from the
    realized TRAIN partition's own counts -- the RULE is frozen in the design
    doc; only the numeric value is a deterministic function of TRAIN."""
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w_pos = n / (2.0 * n_pos) if n_pos else 0.0
    w_neg = n / (2.0 * n_neg) if n_neg else 0.0
    return {1: w_pos * ESCALATE_CLASS_EXTRA_WEIGHT, 0: w_neg}


def fit_value_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """VR1: ordinary least squares regressing delta_u_i directly (continuous
    target). Standardizes on TRAIN only, closed-form via lstsq."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    design = np.hstack([Xs, np.ones((len(y), 1))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    w, b = coef[:-1], float(coef[-1])
    return w, b, mean, std


def value_predict(x: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> float:
    xs = (x - mean) / std
    return float(xs @ w + b)


class ShallowTree:
    """V4: diagnostic-only depth<=3 CART classifier (Gini impurity), no
    external dependency. Predicts P(escalate helps) as leaf positive-rate."""

    def __init__(self, max_depth: int = 3, min_leaf: int = 5):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.tree = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p ** 2 - (1 - p) ** 2

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best = None
        n = len(y)
        for feat in range(X.shape[1]):
            values = np.unique(X[:, feat])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for t in thresholds:
                left = X[:, feat] <= t
                right = ~left
                if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
                    continue
                g = (left.sum() * self._gini(y[left]) + right.sum() * self._gini(y[right])) / n
                if best is None or g < best[0]:
                    best = (g, feat, t)
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or len(np.unique(y)) < 2:
            return {"leaf": True, "p": float(y.mean()) if len(y) else 0.0}
        split = self._best_split(X, y)
        if split is None:
            return {"leaf": True, "p": float(y.mean()) if len(y) else 0.0}
        _, feat, t = split
        left = X[:, feat] <= t
        right = ~left
        return {"leaf": False, "feat": feat, "t": t,
                "left": self._build(X[left], y[left], depth + 1),
                "right": self._build(X[right], y[right], depth + 1)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShallowTree":
        self.tree = self._build(X, y, 0)
        return self

    def predict_proba(self, x: np.ndarray) -> float:
        node = self.tree
        while not node["leaf"]:
            node = node["left"] if x[node["feat"]] <= node["t"] else node["right"]
        return node["p"]


def best_threshold_on(records: list[dict], score_fn, target: str = "utility") -> float:
    """Scan candidate thresholds (the record's own scores) and return the one
    maximizing utility on `records` -- used to freeze V1/V2/V3/VR1's decision
    threshold on CALIBRATION, never TRAIN or EVAL."""
    scores = sorted({score_fn(r) for r in records})
    best_t, best_u = scores[0] if scores else 0.0, -1.0
    for t in scores:
        decisions = ["ESCALATE" if score_fn(r) < t else "ACCEPT" for r in records]
        u = utility(records, decisions)
        if u > best_u:
            best_u, best_t = u, t
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()

    records = load_records(args.receipts)
    n = len(records)
    print(f"=== ANSWER_PROBE_GATE_V2 split + model ladder ({n} paired tasks) ===\n")

    strata_counts = defaultdict(int)
    for r in records:
        strata_counts[r["stratum"]] += 1
    print(f"  strata (whole pool): {dict(strata_counts)}\n")

    train, calib, eval_ = outcome_stratified_split(records)
    print(f"  train: {len(train)}   calibration: {len(calib)}   eval: {len(eval_)} "
          f"(outcome-stratified, seed={SPLIT_SEED})")
    eval_strata = defaultdict(int)
    for r in eval_:
        eval_strata[r["stratum"]] += 1
    print(f"  eval strata: {dict(eval_strata)}\n")

    # --- PHASE_5 minimum-class-floor check, BEFORE any fitting -------------
    floor_ok = all(eval_strata.get(k, 0) >= v for k, v in MIN_EVAL_FLOOR.items())
    result_base = {
        "n_total": n, "n_train": len(train), "n_calibration": len(calib), "n_eval": len(eval_),
        "split_seed": SPLIT_SEED, "eval_strata_counts": dict(eval_strata),
        "min_eval_floor": MIN_EVAL_FLOOR,
    }
    if not floor_ok:
        print("  MINIMUM EVAL CLASS FLOOR NOT MET -- STOPPING per frozen protocol.")
        print("  Do not reshuffle. This is INCONCLUSIVE, not a second negative result.")
        out = {
            **result_base,
            "outcome": "INCONCLUSIVE_INSUFFICIENT_EVAL_CLASS_SUPPORT",
            "promoted": False,
            "formal_gate_result": {
                "run_valid": True, "scientific_verdict": "NOT_COMPUTABLE",
                "mechanism_status": "PENDING_FURTHER_EVIDENCE",
                "failure_class": "MINIMUM_CLASS_FLOOR_NOT_MET",
                "split_status": "CONSUMED",
            },
        }
        out_path = Path(args.receipts).with_suffix(".gate_result.json")
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"\n  written: {out_path}")
        return 2
    print("  eval class floors MET -- proceeding to model ladder.\n")

    conf_idx = FEATURES.index("mean_token_confidence")
    X_train = np.stack([r["x"] for r in train])
    y_train = np.array([1.0 if r["delta_u"] > 0 else 0.0 for r in train])

    # --- V1: frozen confidence threshold, chosen on CALIBRATION ------------
    v1_thresh = best_threshold_on(calib, lambda r: r["x"][conf_idx])

    # --- V2: ordinary logistic regression, decision cutoff on CALIBRATION --
    w2, b2, mean2, std2 = fit_logistic_regression(X_train, y_train, sample_weight=None, seed=7)
    v2_cutoff = best_threshold_on(
        calib, lambda r, _w=w2, _b=b2, _m=mean2, _s=std2:
            -logistic_predict_proba(r["x"], _w, _b, _m, _s))  # negate: escalate when proba HIGH

    # --- V3: cost-sensitive logistic regression (PRIMARY hypothesis) -------
    class_w = balanced_class_weights(y_train)
    sample_w = np.array([class_w[int(y)] for y in y_train])
    w3, b3, mean3, std3 = fit_logistic_regression(X_train, y_train, sample_weight=sample_w, seed=7)
    v3_cutoff = best_threshold_on(
        calib, lambda r, _w=w3, _b=b3, _m=mean3, _s=std3:
            -logistic_predict_proba(r["x"], _w, _b, _m, _s))

    # --- V4: shallow tree (diagnostic only) ---------------------------------
    tree = ShallowTree(max_depth=3, min_leaf=5).fit(X_train, y_train)

    # --- VR1: continuous value regression, threshold on CALIBRATION --------
    y_train_delta = np.array([float(r["delta_u"]) for r in train])
    wr, br, meanr, stdr = fit_value_regression(X_train, y_train_delta)
    vr1_thresh = best_threshold_on(
        calib, lambda r, _w=wr, _b=br, _m=meanr, _s=stdr:
            -value_predict(r["x"], _w, _b, _m, _s))

    def v1_decision(r): return "ESCALATE" if r["x"][conf_idx] < v1_thresh else "ACCEPT"
    def v2_decision(r): return "ESCALATE" if -logistic_predict_proba(r["x"], w2, b2, mean2, std2) < v2_cutoff else "ACCEPT"
    def v3_decision(r): return "ESCALATE" if -logistic_predict_proba(r["x"], w3, b3, mean3, std3) < v3_cutoff else "ACCEPT"
    def v4_decision(r): return "ESCALATE" if tree.predict_proba(r["x"]) >= 0.5 else "ACCEPT"
    def v5_decision(r): return "ESCALATE" if r["delta_u"] > 0 else "ACCEPT"  # oracle, tie->ACCEPT
    def vr1_decision(r): return "ESCALATE" if -value_predict(r["x"], wr, br, meanr, stdr) < vr1_thresh else "ACCEPT"

    policies = {
        "V0_best_fixed": None,  # computed from V0a/V0b below
        "V0a_always_accept_direct": ["ACCEPT"] * len(eval_),
        "V0b_always_escalate_to_memory": ["ESCALATE"] * len(eval_),
        "V1_frozen_confidence_threshold": [v1_decision(r) for r in eval_],
        "V2_ordinary_logistic": [v2_decision(r) for r in eval_],
        "V3_cost_sensitive_logistic": [v3_decision(r) for r in eval_],
        "V4_shallow_tree_diagnostic": [v4_decision(r) for r in eval_],
        "V5_oracle_escalation": [v5_decision(r) for r in eval_],
        "VR1_value_regression": [vr1_decision(r) for r in eval_],
    }
    del policies["V0_best_fixed"]

    print(f"  {'policy':<34}{'U(eval)':>10}")
    utils = {}
    for name, decisions in policies.items():
        u = utility(eval_, decisions)
        utils[name] = u
        print(f"  {name:<34}{u:>10.4f}")

    best_fixed = max(utils["V0a_always_accept_direct"], utils["V0b_always_escalate_to_memory"])
    print(f"\n  V0_best_fixed = {best_fixed:.4f}")

    def gate_stats(name: str, decisions: list[str]) -> dict:
        u = utils[name]
        delta_u_gate = u - best_fixed
        fixed_is_accept = utils["V0a_always_accept_direct"] >= utils["V0b_always_escalate_to_memory"]
        decisions_by_key = {r["key"]: d for r, d in zip(eval_, decisions)}
        paired_deltas = []
        for r in eval_:
            cand_u = r["q_direct"] if decisions_by_key[r["key"]] == "ACCEPT" else r["q_memory"]
            fixed_u = r["q_direct"] if fixed_is_accept else r["q_memory"]
            paired_deltas.append((r["family"], float(cand_u - fixed_u)))
        lcb = grouped_lcb(paired_deltas)
        memory_would_improve = [r for r in eval_ if r["delta_u"] > 0]
        direct_sufficient = [r for r in eval_ if r["delta_u"] <= 0]
        p_accept_given_memory_better = (
            sum(1 for r in memory_would_improve if decisions_by_key[r["key"]] == "ACCEPT") / len(memory_would_improve)
            if memory_would_improve else float("nan"))
        p_escalate_given_direct_sufficient = (
            sum(1 for r in direct_sufficient if decisions_by_key[r["key"]] == "ESCALATE") / len(direct_sufficient)
            if direct_sufficient else float("nan"))
        oracle_u = utils["V5_oracle_escalation"]
        regret = oracle_u - u
        safety_met = (p_accept_given_memory_better <= SAFETY_BOUND_P_ACCEPT_GIVEN_MEMORY_BETTER
                     if memory_would_improve else False)
        promoted = delta_u_gate > 0 and lcb is not None and lcb > 0.0 and safety_met
        return {
            "utility": u, "delta_u_gate": delta_u_gate, "delta_u_gate_lcb_2p5": lcb,
            "policy_regret_vs_oracle": regret,
            "P_accept_given_memory_would_improve": p_accept_given_memory_better,
            "P_escalate_given_direct_sufficient": p_escalate_given_direct_sufficient,
            "safety_bound_met": safety_met, "promoted": promoted,
        }

    print(f"\n  {'policy':<34}{'DeltaU_gate':>13}{'LCB2.5':>10}{'P(ACC|mem>0)':>14}{'safety':>8}{'promoted':>10}")
    stats = {}
    for name in ("V1_frozen_confidence_threshold", "V2_ordinary_logistic",
                "V3_cost_sensitive_logistic", "V4_shallow_tree_diagnostic", "VR1_value_regression"):
        s = gate_stats(name, policies[name])
        stats[name] = s
        print(f"  {name:<34}{s['delta_u_gate']:>13.4f}{str(s['delta_u_gate_lcb_2p5']):>10}"
              f"{s['P_accept_given_memory_would_improve']:>14.4f}{str(s['safety_bound_met']):>8}{str(s['promoted']):>10}")

    # V4 is diagnostic-only per the frozen model ladder -- report but never eligible.
    stats["V4_shallow_tree_diagnostic"]["promoted"] = False

    any_promoted = any(s["promoted"] for name, s in stats.items() if name != "V4_shallow_tree_diagnostic")
    primary = stats["V3_cost_sensitive_logistic"]
    co_primary = stats["VR1_value_regression"]

    print(f"\n  PRIMARY (V3) promoted: {primary['promoted']}")
    print(f"  CO-PRIMARY (VR1) promoted: {co_primary['promoted']}")
    print(f"  ANY candidate promoted: {any_promoted}")

    if any_promoted:
        promoted_names = [name for name, s in stats.items() if s["promoted"] and name != "V4_shallow_tree_diagnostic"]
        print(f"\n  PROMOTED: {promoted_names}")
        print("  Next: generate a fresh, untouched executive-confirmation split.")
        scientific_verdict, mechanism_status, failure_class = "POSITIVE", "PENDING_FURTHER_EVIDENCE", "NONE"
    else:
        print("\n  NOT PROMOTED. This closes confidence-only escalation-gating specifically "
              "(not the executive concept). Report as a second formal negative result.")
        scientific_verdict, mechanism_status, failure_class = "NEGATIVE", "NOT_PROMOTED", "LEARNED_POLICY_UNDERPERFORMS_FIXED_BASELINE"

    out = {
        **result_base,
        "strata_counts_whole_pool": dict(strata_counts),
        "v0_best_fixed_utility": best_fixed,
        "eval_utilities": utils,
        "gate_stats": stats,
        "primary_hypothesis": "V3_cost_sensitive_logistic",
        "co_primary_hypothesis": "VR1_value_regression",
        "any_promoted": any_promoted,
        "promoted": any_promoted,
        "formal_gate_result": {
            "run_valid": True, "scientific_verdict": scientific_verdict,
            "mechanism_status": mechanism_status, "failure_class": failure_class,
            "split_status": "CONSUMED",
        },
        "fitted_params": {
            "v1_threshold": v1_thresh, "v2_cutoff_neg_proba": v2_cutoff,
            "v3_cutoff_neg_proba": v3_cutoff, "v3_class_weights": class_w,
            "vr1_threshold_neg_pred": vr1_thresh,
        },
    }
    out_path = Path(args.receipts).with_suffix(".gate_result.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
