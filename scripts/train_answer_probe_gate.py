#!/usr/bin/env python3
"""ANSWER_PROBE_GATE_V1: train + evaluate the escalation gate.

Per configs/gate_answer_probe_v1_design.json. Run ONLY after
scripts/analyze_answer_probe_separation.py confirms separation exists --
this script does not re-check that gate itself, by design, so it cannot be
run "just to see" past a STOP verdict.

GPU-free: operates entirely on the receipts written by
scripts/run_exec_training_v1_collection.py --execute. Logistic regression is
implemented directly (gradient descent via numpy) rather than pulling in a
new ML dependency, per the frozen protocol's MODEL_LADDER.
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

from hrm_adaptive_memory.experiment_integrity.executive_bootstrap import (  # noqa: E402
    grouped_lcb_executive_opportunity)

FEATURES = [
    "mean_token_confidence", "min_token_confidence", "sequence_confidence",
    "mean_entropy", "answer_length",
]
SPLIT_SEED = 9801
TRAIN_SHARE = 0.7


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
        records.append({
            "key": key, "family": family_of[key], "suite_family": suite_family_of[key],
            "q_direct": q0, "q_memory": q1, "delta_u": q1 - q0,
            "x": np.array([a0[f] for f in FEATURES], dtype=float),
        })
    return records


def stratified_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SPLIT_SEED)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_family[r["suite_family"]].append(r)
    train, eval_ = [], []
    for fam, recs in by_family.items():
        recs = recs[:]
        rng.shuffle(recs)
        n_train = round(len(recs) * TRAIN_SHARE)
        train.extend(recs[:n_train])
        eval_.extend(recs[n_train:])
    return train, eval_


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                            iterations: int = 2000, seed: int = 7) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Plain gradient-descent logistic regression, y in {0,1} (1 = escalate
    improves, i.e. delta_u>0). Standardizes features on TRAIN only. Returns
    (weights, bias, feature_mean, feature_std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std

    rng = np.random.RandomState(seed)
    w = rng.normal(scale=0.01, size=Xs.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iterations):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xs.T @ (p - y) / n
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mean, std


def logistic_predict_proba(x: np.ndarray, w: np.ndarray, b: float,
                           mean: np.ndarray, std: np.ndarray) -> float:
    xs = (x - mean) / std
    z = float(xs @ w + b)
    return 1.0 / (1.0 + np.exp(-z))


def utility(records: list[dict], decisions: list[str]) -> float:
    """decisions[i] in {'ACCEPT', 'ESCALATE'} for records[i]."""
    total = 0
    for r, d in zip(records, decisions):
        total += r["q_direct"] if d == "ACCEPT" else r["q_memory"]
    return total / len(records) if records else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()

    records = load_records(args.receipts)
    n = len(records)
    print(f"=== ANSWER_PROBE_GATE_V1 training + evaluation ({n} paired tasks) ===\n")

    train, eval_ = stratified_split(records)
    print(f"  train: {len(train)}   eval: {len(eval_)} (stratified 70/30 by suite_family, seed={SPLIT_SEED})\n")

    X_train = np.stack([r["x"] for r in train])
    y_train = np.array([1.0 if r["delta_u"] > 0 else 0.0 for r in train])

    # --- E2: fixed confidence threshold, fit on TRAIN only ---
    conf_idx = FEATURES.index("mean_token_confidence")
    candidate_thresholds = sorted(set(r["x"][conf_idx] for r in train))
    best_thresh, best_train_u = None, -1.0
    for t in candidate_thresholds:
        decisions = ["ESCALATE" if r["x"][conf_idx] < t else "ACCEPT" for r in train]
        u = utility(train, decisions)
        if u > best_train_u:
            best_train_u, best_thresh = u, t

    # --- E3: logistic regression, fit on TRAIN only ---
    w, b, mean, std = fit_logistic_regression(X_train, y_train)

    def e3_decision(r: dict) -> str:
        p_escalate_helps = logistic_predict_proba(r["x"], w, b, mean, std)
        return "ESCALATE" if p_escalate_helps >= 0.5 else "ACCEPT"

    def e4_decision(r: dict) -> str:
        # oracle, tie -> ACCEPT (cheaper action wins ties, same convention as EOB-v1/v2)
        return "ESCALATE" if r["delta_u"] > 0 else "ACCEPT"

    policies = {
        "E0_always_accept_direct": ["ACCEPT"] * len(eval_),
        "E1_always_escalate_to_memory": ["ESCALATE"] * len(eval_),
        "E2_fixed_confidence_threshold": ["ESCALATE" if r["x"][conf_idx] < best_thresh else "ACCEPT" for r in eval_],
        "E3_logistic_value_predictor": [e3_decision(r) for r in eval_],
        "E4_oracle_escalation": [e4_decision(r) for r in eval_],
    }

    print(f"  {'policy':<32}{'U(eval)':>10}")
    utils = {}
    for name, decisions in policies.items():
        u = utility(eval_, decisions)
        utils[name] = u
        print(f"  {name:<32}{u:>10.4f}")

    best_fixed = max(utils["E0_always_accept_direct"], utils["E1_always_escalate_to_memory"])
    delta_u_gate = utils["E3_logistic_value_predictor"] - best_fixed
    regret = utils["E4_oracle_escalation"] - utils["E3_logistic_value_predictor"]

    # grouped bootstrap LCB of Delta_U_gate: per-eval-task (family, q_direct, q_memory)
    # triples, generalizing exactly like EOB's ExecutiveOpportunity bootstrap --
    # but here the "policy" is E3's OWN decision per task, not a max() over two
    # fixed policies, so we bootstrap the per-task E3-vs-best-fixed delta directly.
    e3_decisions_by_key = {r["key"]: d for r, d in zip(eval_, policies["E3_logistic_value_predictor"])}
    fixed_is_accept = utils["E0_always_accept_direct"] >= utils["E1_always_escalate_to_memory"]
    paired_deltas = []
    for r in eval_:
        e3_u = r["q_direct"] if e3_decisions_by_key[r["key"]] == "ACCEPT" else r["q_memory"]
        fixed_u = r["q_direct"] if fixed_is_accept else r["q_memory"]
        paired_deltas.append((r["family"], float(e3_u - fixed_u)))

    from scripts.diagnose_c5_confirmation_stopgate import grouped_lcb
    lcb = grouped_lcb(paired_deltas)

    print(f"\n  Delta_U_gate (E3 - best_fixed) = {delta_u_gate:+.4f}   LCB2.5={lcb}")
    print(f"  policy regret (E4_oracle - E3) = {regret:+.4f}")

    # --- asymmetric error metrics ---
    memory_would_improve = [r for r in eval_ if r["delta_u"] > 0]
    direct_sufficient = [r for r in eval_ if r["delta_u"] <= 0]
    e3_by_key = e3_decisions_by_key
    p_accept_given_memory_better = (
        sum(1 for r in memory_would_improve if e3_by_key[r["key"]] == "ACCEPT") / len(memory_would_improve)
        if memory_would_improve else float("nan"))
    p_escalate_given_direct_sufficient = (
        sum(1 for r in direct_sufficient if e3_by_key[r["key"]] == "ESCALATE") / len(direct_sufficient)
        if direct_sufficient else float("nan"))

    print(f"\n  P(ACCEPT | MEMORY would improve)   = {p_accept_given_memory_better:.4f}  "
          f"(n={len(memory_would_improve)}) -- dangerous: confidently stops while missing needed evidence")
    print(f"  P(ESCALATE | direct was sufficient) = {p_escalate_given_direct_sufficient:.4f}  "
          f"(n={len(direct_sufficient)}) -- wasted/harmful memory calls")

    promoted = delta_u_gate > 0 and lcb is not None and lcb > 0.0
    print(f"\n  PROMOTION_CRITERIA met: {promoted}")
    if promoted:
        print("  Next: generate a fresh, untouched executive-confirmation split.")
    else:
        print("  Do not build an executive-confirmation split. Report as a negative result.")

    out = {
        "n_total": n, "n_train": len(train), "n_eval": len(eval_),
        "split_seed": SPLIT_SEED, "fixed_confidence_threshold": best_thresh,
        "logistic_weights": w.tolist(), "logistic_bias": b,
        "logistic_feature_mean": mean.tolist(), "logistic_feature_std": std.tolist(),
        "eval_utilities": utils, "delta_u_gate": delta_u_gate, "delta_u_gate_lcb_2p5": lcb,
        "policy_regret": regret,
        "P_accept_given_memory_would_improve": p_accept_given_memory_better,
        "P_escalate_given_direct_sufficient": p_escalate_given_direct_sufficient,
        "promoted": promoted,
    }
    out_path = Path(args.receipts).with_suffix(".gate_result.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
