#!/usr/bin/env python3
"""Train coding-specific authority models on collected coding data.

Trains:
  1. Q_res: residual value correction (predicts utility from code features)
  2. Pairwise advantage model (predicts ΔU between two candidates)
  3. Risk model (predicts whether a candidate is harmful vs base)
  4. Conformal calibrator (calibrates uncertainty on held-out calibration data)

Splits data by task (not by candidate) to avoid leakage:
  - Train: 60% of tasks
  - Calibration: 15% of tasks
  - Eval: 25% of tasks

Usage:
    python scripts/train_coding_authority.py \\
        --corpus experiments/daph_x/coding/coding_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.daphx_ranker import extract_code_features, compute_q_mb
from daph_x.coding.tasks import CodingTask, get_all_tasks

M4_DIR = REPO_ROOT / "experiments/daph_x/m4"
CODING_DIR = REPO_ROOT / "experiments/daph_x/coding"


def load_corpus(path: str) -> list[dict]:
    """Load coding corpus from JSONL."""
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def get_feature_keys(records: list[dict]) -> list[str]:
    """Get sorted feature keys from records."""
    all_keys = set()
    for r in records:
        all_keys.update(r["features"].keys())
    return sorted(all_keys)


def build_feature_vector(features: dict, feature_keys: list[str]) -> np.ndarray:
    """Build feature vector in consistent order."""
    return np.array([float(features.get(k, 0.0)) for k in feature_keys])


def split_tasks(tasks: list[dict], train_frac=0.6, cal_frac=0.15, seed=42):
    """Split tasks into train/cal/eval sets."""
    rng = np.random.RandomState(seed)
    n = len(tasks)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    train_idx = indices[:n_train]
    cal_idx = indices[n_train:n_train + n_cal]
    eval_idx = indices[n_train + n_cal:]
    return (
        [tasks[i] for i in train_idx],
        [tasks[i] for i in cal_idx],
        [tasks[i] for i in eval_idx],
    )


def flatten_candidates(tasks: list[dict]) -> list[dict]:
    """Flatten task records into per-candidate records."""
    records = []
    for task in tasks:
        for cand in task["candidates"]:
            records.append(cand)
    return records


def train_q_res(train_records: list[dict], feature_keys: list[str]):
    """Train Q_res model to predict utility from code features.

    Q_res = predicted_utility - Q_MB
    So we train a model to predict utility, then Q_res = model(features) - Q_MB.
    """
    X = np.array([build_feature_vector(r["features"], feature_keys) for r in train_records])
    y = np.array([r["utility"] for r in train_records])
    q_mb = np.array([r["q_mb"] for r in train_records])

    # Train model to predict utility directly
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    # Q_res = predicted_utility - Q_MB
    # So at inference: Q_X = Q_MB + Q_res = Q_MB + (model(features) - Q_MB) = model(features)
    # But we keep the decomposition for interpretability

    return model


def train_pairwise_model(train_tasks: list[dict], feature_keys: list[str]):
    """Train pairwise advantage model.

    For each task, generate all pairs (a, b) where a != b.
    Target: ΔU = U(a) - U(b)
    Features: features(a) - features(b), features(a), features(b)
    """
    X_pairs = []
    y_pairs = []

    for task in train_tasks:
        cands = task["candidates"]
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                a, b = cands[i], cands[j]
                fa = build_feature_vector(a["features"], feature_keys)
                fb = build_feature_vector(b["features"], feature_keys)

                # Pairwise features: difference and both originals
                pair_feats = np.concatenate([
                    fa - fb,  # difference
                    fa,  # candidate a features
                    fb,  # candidate b features
                    [a["q_mb"] - b["q_mb"]],  # Q_MB difference
                    [a["q_mb"]],  # Q_MB of a
                    [b["q_mb"]],  # Q_MB of b
                ])

                delta_u = a["utility"] - b["utility"]
                X_pairs.append(pair_feats)
                y_pairs.append(delta_u)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_pairs, y_pairs)

    return model, len(X_pairs)


def train_risk_model(train_records: list[dict], feature_keys: list[str]):
    """Train risk model to predict whether a candidate is harmful (worse than base).

    A candidate is "harmful" if its utility is meaningfully worse than the base
    candidate's utility (ΔU < -0.5).
    """
    # For each task, label candidates as harmful if they're worse than base
    X = []
    y = []

    # Group by task
    by_task = defaultdict(list)
    for r in train_records:
        by_task[r["task_id"]].append(r)

    for task_id, cands in by_task.items():
        base = cands[0]  # First candidate is base
        base_utility = base["utility"]
        for c in cands:
            fa = build_feature_vector(c["features"], feature_keys)
            X.append(fa)
            y.append(1 if c["utility"] < base_utility - 0.5 else 0)

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2:
        print(f"  Warning: only one class in risk training data ({set(y)}), skipping risk model")
        return None, len(X)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    return model, len(X)


def conformal_calibrate(cal_tasks: list[dict], q_res_model, pairwise_model,
                        feature_keys: list[str]):
    """Compute conformal quantile on calibration data.

    For each task, compute the nonconformity score:
      |Q_X(candidate) - U(candidate)|
    where Q_X = Q_MB + Q_res.

    The conformal quantile q_alpha gives us a calibration interval:
      Q_X(candidate) ± q_alpha
    """
    scores = []

    for task in cal_tasks:
        cands = task["candidates"]
        for c in cands:
            fa = build_feature_vector(c["features"], feature_keys)
            q_mb = c["q_mb"]
            q_res_pred = q_res_model.predict([fa])[0]
            # Q_X = Q_MB + (predicted_utility - Q_MB) = predicted_utility
            # But we keep the decomposition
            q_x = q_mb + (q_res_pred - q_mb)  # = q_res_pred
            actual_u = c["utility"]
            score = abs(q_x - actual_u)
            scores.append(score)

    scores = np.array(sorted(scores))
    n = len(scores)

    # 90% coverage quantile
    alpha = 0.90
    q_90 = scores[min(int(np.ceil(alpha * (n + 1))) - 1, n - 1)]

    # 80% coverage quantile
    alpha_80 = 0.80
    q_80 = scores[min(int(np.ceil(alpha_80 * (n + 1))) - 1, n - 1)]

    # 50% coverage quantile
    alpha_50 = 0.50
    q_50 = scores[min(int(np.ceil(alpha_50 * (n + 1))) - 1, n - 1)]

    return {
        "q_90": float(q_90),
        "q_80": float(q_80),
        "q_50": float(q_50),
        "n_cal": n,
        "scores": scores.tolist(),
    }


def evaluate_authority(eval_tasks: list[dict], q_res_model, pairwise_model,
                       risk_model, conformal: dict, feature_keys: list[str],
                       rho: float = 0.05, tau_delta: float = 0.0,
                       use_pairwise: bool = True, use_risk: bool = True):
    """Evaluate the full authority stack on held-out tasks.

    For each task:
      1. Compute Q_X for each candidate
      2. Identify base action (first candidate) and DAPH-X action (highest Q_X)
      3. If disagreement, apply authority gate:
         - LCB_Δ = ΔQ_hat - q_alpha
         - Risk: risk_prob < rho
         - Pairwise: pairwise_pred > 0
      4. Record would_force, rescue, break
    """
    results = []

    for task in eval_tasks:
        cands = task["candidates"]
        base = cands[0]
        base_utility = base["utility"]

        # Compute Q_X for each candidate
        for c in cands:
            fa = build_feature_vector(c["features"], feature_keys)
            q_mb = c["q_mb"]
            q_res_pred = q_res_model.predict([fa])[0]
            c["q_res"] = q_res_pred - q_mb  # Residual
            c["q_x"] = q_mb + c["q_res"]  # = q_res_pred

        # DAPH-X pick: highest Q_X
        daphx_pick = max(cands, key=lambda c: c["q_x"])
        daphx_id = daphx_pick["candidate_id"]
        base_id = base["candidate_id"]

        disagreement = daphx_id != base_id

        if not disagreement:
            results.append({
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "disagreement": False,
                "would_force": False,
                "base_utility": base_utility,
                "daphx_utility": base_utility,
                "delta_u": 0.0,
                "outcome": "AGREE",
            })
            continue

        # Disagreement — apply authority gate
        delta_q_hat = daphx_pick["q_x"] - base["q_x"]
        lcb_delta = delta_q_hat - conformal["q_90"]

        # Risk probability
        risk_prob = 0.0
        if risk_model is not None and use_risk:
            fa_daphx = build_feature_vector(daphx_pick["features"], feature_keys)
            risk_prob = float(risk_model.predict_proba([fa_daphx])[0, 1])

        # Pairwise prediction
        pairwise_pred = 0.0
        if pairwise_model is not None and use_pairwise:
            fa_daphx = build_feature_vector(daphx_pick["features"], feature_keys)
            fa_base = build_feature_vector(base["features"], feature_keys)
            pair_feats = np.concatenate([
                fa_daphx - fa_base,
                fa_daphx,
                fa_base,
                [daphx_pick["q_mb"] - base["q_mb"]],
                [daphx_pick["q_mb"]],
                [base["q_mb"]],
            ])
            pairwise_pred = float(pairwise_model.predict([pair_feats])[0])

        # Gate: LCB > tau AND risk < rho AND pairwise > 0
        lcb_pass = lcb_delta > tau_delta
        risk_pass = risk_prob < rho
        pw_pass = pairwise_pred > 0

        would_force = lcb_pass and risk_pass and (pw_pass if use_pairwise else True)

        # Actual outcome
        daphx_utility = daphx_pick["utility"]
        delta_u = daphx_utility - base_utility

        if would_force:
            if delta_u > 0.5:
                outcome = "RESCUE"
            elif delta_u < -0.5:
                outcome = "BREAK"
            else:
                outcome = "TIE_FORCE"
        else:
            outcome = "ABSTAIN"

        results.append({
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
            "disagreement": True,
            "would_force": would_force,
            "base_utility": base_utility,
            "daphx_utility": daphx_utility,
            "delta_u": delta_u,
            "delta_q_hat": delta_q_hat,
            "lcb_delta": lcb_delta,
            "risk_prob": risk_prob,
            "pairwise_pred": pairwise_pred,
            "outcome": outcome,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Train coding-specific authority models")
    parser.add_argument("--corpus", default=str(CODING_DIR / "coding_corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--tau_delta", type=float, default=0.0)
    args = parser.parse_args()

    # Load corpus
    tasks = load_corpus(args.corpus)
    print(f"Loaded {len(tasks)} tasks from {args.corpus}")

    # Split
    train_tasks, cal_tasks, eval_tasks = split_tasks(tasks, seed=args.seed)
    print(f"Split: {len(train_tasks)} train, {len(cal_tasks)} cal, {len(eval_tasks)} eval")

    train_records = flatten_candidates(train_tasks)
    cal_records = flatten_candidates(cal_tasks)
    eval_records = flatten_candidates(eval_tasks)
    print(f"Candidates: {len(train_records)} train, {len(cal_records)} cal, {len(eval_records)} eval")

    # Get feature keys
    feature_keys = get_feature_keys(train_records)
    print(f"Features: {len(feature_keys)}")

    # Train Q_res
    print("\nTraining Q_res model...")
    q_res_model = train_q_res(train_records, feature_keys)

    # Evaluate Q_res on train and eval
    for split_name, records in [("train", train_records), ("eval", eval_records)]:
        X = np.array([build_feature_vector(r["features"], feature_keys) for r in records])
        y = np.array([r["utility"] for r in records])
        q_mb = np.array([r["q_mb"] for r in records])
        q_res_pred = q_res_model.predict(X)
        q_x = q_res_pred  # Q_X = predicted utility

        mae_mb = np.mean(np.abs(q_mb - y))
        mae_x = np.mean(np.abs(q_x - y))
        regret_mb = np.mean(np.maximum(y - q_mb, 0))
        regret_x = np.mean(np.maximum(y - q_x, 0))

        # Top-1 accuracy: does the highest Q pick the highest utility?
        print(f"\n  {split_name}:")
        print(f"    MAE(Q_MB) = {mae_mb:.2f}, MAE(Q_X) = {mae_x:.2f}")
        print(f"    Regret(Q_MB) = {regret_mb:.2f}, Regret(Q_X) = {regret_x:.2f}")

    # Train pairwise model
    print("\nTraining pairwise model...")
    pairwise_model, n_pairs = train_pairwise_model(train_tasks, feature_keys)
    print(f"  Trained on {n_pairs} pairs")

    # Train risk model
    print("\nTraining risk model...")
    risk_model, n_risk = train_risk_model(train_records, feature_keys)
    if risk_model:
        print(f"  Trained on {n_risk} samples")

    # Conformal calibration
    print("\nCalibrating conformal...")
    conformal = conformal_calibrate(cal_tasks, q_res_model, pairwise_model, feature_keys)
    print(f"  q_90 = {conformal['q_90']:.2f}")
    print(f"  q_80 = {conformal['q_80']:.2f}")
    print(f"  q_50 = {conformal['q_50']:.2f}")

    # Evaluate authority
    print(f"\nEvaluating authority (rho={args.rho}, tau_delta={args.tau_delta})...")
    results = evaluate_authority(
        eval_tasks, q_res_model, pairwise_model, risk_model,
        conformal, feature_keys,
        rho=args.rho, tau_delta=args.tau_delta,
    )

    # Summary
    n_disagree = sum(1 for r in results if r["disagreement"])
    n_force = sum(1 for r in results if r["would_force"])
    n_rescue = sum(1 for r in results if r["outcome"] == "RESCUE")
    n_break = sum(1 for r in results if r["outcome"] == "BREAK")
    n_tie = sum(1 for r in results if r["outcome"] == "TIE_FORCE")
    n_abstain = sum(1 for r in results if r["outcome"] == "ABSTAIN")

    print(f"\n{'='*60}")
    print(f"  AUTHORITY EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Eval tasks: {len(eval_tasks)}")
    print(f"  Disagreements: {n_disagree}")
    print(f"  Would FORCE: {n_force}")
    print(f"  Rescues: {n_rescue}")
    print(f"  Breaks: {n_break}")
    print(f"  Ties (forced): {n_tie}")
    print(f"  Abstained: {n_abstain}")

    if n_force > 0:
        print(f"  Force precision: {n_rescue / n_force:.4f}")
        print(f"  Break rate: {n_break / n_force:.4f}")
        if n_break == 0:
            print(f"  Break rate 95% upper: {3.0 / n_force:.4f}")

    # Also evaluate Q_MB only (no authority gate)
    print(f"\n  Q_MB-only baseline (no gate):")
    qmb_disagree = 0
    qmb_rescue = 0
    qmb_break = 0
    qmb_tie = 0
    for task in eval_tasks:
        cands = task["candidates"]
        base = cands[0]
        qmb_pick = max(cands, key=lambda c: c["q_mb"])
        if qmb_pick["candidate_id"] != base["candidate_id"]:
            qmb_disagree += 1
            du = qmb_pick["utility"] - base["utility"]
            if du > 0.5: qmb_rescue += 1
            elif du < -0.5: qmb_break += 1
            else: qmb_tie += 1
    print(f"    Disagreements: {qmb_disagree}")
    print(f"    Rescues: {qmb_rescue}, Breaks: {qmb_break}, Ties: {qmb_tie}")
    if qmb_disagree > 0:
        print(f"    Precision: {qmb_rescue / qmb_disagree:.4f}")
        print(f"    Break rate: {qmb_break / qmb_disagree:.4f}")

    # Save models
    import joblib
    model_path = CODING_DIR / "coding_authority_models.pkl"
    joblib.dump({
        "q_res_model": q_res_model,
        "pairwise_model": pairwise_model,
        "risk_model": risk_model,
        "conformal": conformal,
        "feature_keys": feature_keys,
    }, model_path)
    print(f"\n  Models saved to {model_path}")

    # Save results
    results_path = CODING_DIR / "coding_authority_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "rho": args.rho,
                "tau_delta": args.tau_delta,
                "n_train_tasks": len(train_tasks),
                "n_cal_tasks": len(cal_tasks),
                "n_eval_tasks": len(eval_tasks),
                "seed": args.seed,
            },
            "results": results,
            "summary": {
                "n_disagree": n_disagree,
                "n_force": n_force,
                "n_rescue": n_rescue,
                "n_break": n_break,
                "n_tie": n_tie,
                "n_abstain": n_abstain,
                "qmb_baseline": {
                    "n_disagree": qmb_disagree,
                    "n_rescue": qmb_rescue,
                    "n_break": qmb_break,
                    "n_tie": qmb_tie,
                },
            },
            "conformal": conformal,
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
