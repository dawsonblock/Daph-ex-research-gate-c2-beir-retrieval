#!/usr/bin/env python3
"""RETRIEVAL_PROBE_GATE_V1 PHASE_2: incremental information of the probe.

The question, stated so it can only be answered one way:

    Does the cheap retrieval observation add discrimination about
    Delta_U BEYOND what direct-answer confidence already provides,
    WITHIN MEMORY_required?

Restricting to MEMORY_required is the whole point. Pooled over both
families, probe_candidate_count and probe_identity_binding_status_code
separate the classes trivially -- ANSWER_NOW_viable tasks carry zero
evidence, so they can never bind and never retrieve. A large pooled gain
driven by telling an empty corpus from a non-empty one is NOT evidence for a
useful retrieval executive, and configs/gate_retrieval_probe_v1_design.json
says so in advance (KNOWN_LIMITATION_binding_status_may_be_too_good_on_this
_instrument).

FROZEN STOP CONDITION -- declared here BEFORE this script was ever run:

    PROCEED to fitting only if, on MEMORY_required alone,
        Delta_AUC = AUC(confidence + probe) - AUC(confidence only)
    satisfies BOTH
        Delta_AUC >= 0.02
        grouped-bootstrap 2.5% LCB of Delta_AUC > 0
    Otherwise STOP and close the retrieval-probe branch before any
    evaluation split is built and before any GPU is spent on it.

0.02 AUC is a deliberately modest bar -- roughly the smallest
discrimination gain that could plausibly matter -- so a STOP here means the
probe added essentially nothing, not merely that it fell short of a
demanding target. AUC is measured OUT-OF-FOLD (stratified k-fold), because
the richer feature set would otherwise win in-sample by construction.

DEVELOPMENT USE ONLY: consumed split, no promotion claim.
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

from scripts.analyze_answer_probe_cost_separation import load_records  # noqa: E402
from scripts.fit_answer_probe_cost_gate import fit_logistic, logistic_proba  # noqa: E402

CONFIDENCE_FEATURES = [
    "mean_token_confidence", "min_token_confidence", "sequence_confidence",
    "mean_entropy", "answer_length",
]
PROBE_FEATURES = [
    "probe_top1_retrieval_score", "probe_topk_mean_retrieval_score",
    "probe_retrieval_score_margin", "probe_candidate_count",
    "probe_identity_binding_status_code", "probe_relation_extracted",
]

# --- FROZEN before first run ---------------------------------------------
MIN_DELTA_AUC = 0.02
N_FOLDS = 5
CV_SEED = 9903
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 12345


def auc(y_true, scores) -> float:
    """Rank-based AUC with tie handling."""
    pairs = sorted(zip(scores, y_true))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _s, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_s, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, seed: int = CV_SEED) -> np.ndarray:
    """Stratified k-fold out-of-fold predictions, so a richer feature set
    cannot win merely by fitting the training data harder."""
    rng = random.Random(seed)
    idx_pos = [i for i, v in enumerate(y) if v == 1]
    idx_neg = [i for i, v in enumerate(y) if v == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    folds = [[] for _ in range(N_FOLDS)]
    for k, i in enumerate(idx_pos):
        folds[k % N_FOLDS].append(i)
    for k, i in enumerate(idx_neg):
        folds[k % N_FOLDS].append(i)

    oof = np.zeros(len(y))
    for f in range(N_FOLDS):
        test = folds[f]
        train = [i for g in range(N_FOLDS) if g != f for i in folds[g]]
        if not test or not train:
            continue
        ytr = y[train]
        if len(set(ytr.tolist())) < 2:
            oof[test] = float(ytr.mean())
            continue
        w, b, m, s = fit_logistic(X[train], ytr, sample_weight=None)
        for i in test:
            oof[i] = logistic_proba(X[i], w, b, m, s)
    return oof


def grouped_bootstrap_delta_auc(groups, y, oof_a, oof_b) -> tuple[float, float]:
    """CI of AUC(b) - AUC(a), resampling GROUPS with replacement -- the same
    convention every other gate in this project uses."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = sorted(by_group)
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_ITERS):
        idx = []
        for _ in keys:
            idx.extend(by_group[keys[rng.randrange(len(keys))]])
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        a = auc(yy, oof_a[idx])
        b = auc(yy, oof_b[idx])
        if a == a and b == b:
            deltas.append(b - a)
    if not deltas:
        return float("nan"), float("nan")
    deltas.sort()
    return deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]


def evaluate(records, label: str) -> dict:
    y = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in records])
    groups = [r["family"] for r in records]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    row = {"label": label, "n": len(records), "n_positive": n_pos, "n_negative": n_neg}
    if n_pos < 2 or n_neg < 2:
        row["skipped"] = "degenerate class balance"
        return row

    X_conf = np.array([[r[f] for f in CONFIDENCE_FEATURES] for r in records], float)
    X_both = np.array([[r[f] for f in CONFIDENCE_FEATURES + PROBE_FEATURES] for r in records], float)
    X_probe = np.array([[r[f] for f in PROBE_FEATURES] for r in records], float)

    oof_conf = out_of_fold_scores(X_conf, y)
    oof_both = out_of_fold_scores(X_both, y)
    oof_probe = out_of_fold_scores(X_probe, y)

    a_conf, a_both, a_probe = auc(y, oof_conf), auc(y, oof_both), auc(y, oof_probe)
    lo, hi = grouped_bootstrap_delta_auc(groups, y, oof_conf, oof_both)
    row.update({
        "auc_confidence_only": a_conf, "auc_probe_only": a_probe,
        "auc_confidence_plus_probe": a_both,
        "delta_auc": a_both - a_conf,
        "delta_auc_lcb_2p5": lo, "delta_auc_ucb_97p5": hi,
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", default=str(
        ROOT / "evidence/gate_executive/exec_training_v2_execute.receipts.jsonl"))
    ap.add_argument("--probe-features", default=str(
        ROOT / "evidence/gate_executive/retrieval_probe_v1_features.jsonl"))
    args = ap.parse_args()

    base = load_records(args.receipts)
    by_key = {(r["suite_family"], r["key"][1]): r for r in base}
    probe_rows = [json.loads(l) for l in open(args.probe_features) if l.strip()]

    merged, missing = [], 0
    for p in probe_rows:
        k = (p["suite_family"], p["task_id"])
        b = by_key.get(k)
        if b is None:
            missing += 1
            continue
        merged.append({**b, **{f: float(p[f]) for f in PROBE_FEATURES},
                       "identity_status": p["identity_status"],
                       "evidence_pool_size": p["evidence_pool_size"]})

    print("=== RETRIEVAL_PROBE_GATE_V1 PHASE_2: incremental probe information ===")
    print("    DEVELOPMENT USE ONLY -- consumed split, no promotion claim.\n")
    print(f"  joined {len(merged)} tasks (unmatched probe rows: {missing})")
    print(f"  FROZEN stop condition: within MEMORY_required, Delta_AUC >= {MIN_DELTA_AUC} "
          f"AND grouped-bootstrap LCB > 0\n")

    memory_only = [r for r in merged if r["suite_family"] == "MEMORY_required"]
    nonempty = [r for r in merged if r["probe_candidate_count"] > 0]
    bound = [r for r in memory_only if r["probe_identity_binding_status_code"] >= 2]

    strata = [
        ("POOLED_all_tasks", merged),
        ("POOLED_excluding_empty_pool", nonempty),
        ("MEMORY_required_ONLY__PRIMARY", memory_only),
        ("MEMORY_required_bound_only", bound),
    ]

    results = []
    print(f"  {'stratum':<34}{'n':>6}{'AUC_conf':>10}{'AUC_probe':>11}{'AUC_both':>10}{'dAUC':>9}{'LCB':>9}")
    for label, recs in strata:
        row = evaluate(recs, label)
        results.append(row)
        if "skipped" in row:
            print(f"  {label:<34}{row['n']:>6}   {row['skipped']}")
            continue
        print(f"  {label:<34}{row['n']:>6}{row['auc_confidence_only']:>10.4f}"
              f"{row['auc_probe_only']:>11.4f}{row['auc_confidence_plus_probe']:>10.4f}"
              f"{row['delta_auc']:>9.4f}{row['delta_auc_lcb_2p5']:>9.4f}")

    primary = next(r for r in results if r["label"] == "MEMORY_required_ONLY__PRIMARY")
    passed = ("skipped" not in primary
              and primary["delta_auc"] >= MIN_DELTA_AUC
              and primary["delta_auc_lcb_2p5"] > 0.0)

    print(f"\n  PRIMARY (MEMORY_required only): Delta_AUC={primary.get('delta_auc', float('nan')):+.4f}  "
          f"LCB={primary.get('delta_auc_lcb_2p5', float('nan')):+.4f}")
    print(f"  frozen bar: Delta_AUC >= {MIN_DELTA_AUC} AND LCB > 0  ->  {'PASS' if passed else 'FAIL'}")

    out = {
        "phase": "RETRIEVAL_PROBE_GATE_V1 PHASE_2",
        "design": "configs/gate_retrieval_probe_v1_design.json",
        "DEVELOPMENT_ONLY_no_promotion_claim": True,
        "frozen_stop_condition": {"min_delta_auc": MIN_DELTA_AUC,
                                  "require_lcb_positive": True,
                                  "primary_stratum": "MEMORY_required only",
                                  "declared_before_first_run": True},
        "cv": {"folds": N_FOLDS, "seed": CV_SEED, "out_of_fold": True},
        "confidence_features": CONFIDENCE_FEATURES,
        "probe_features": PROBE_FEATURES,
        "n_joined": len(merged), "unmatched_probe_rows": missing,
        "strata": results,
        "PRIMARY_passed": bool(passed),
        "verdict": ("PROCEED_TO_FITTING" if passed else
                    "STOP__PROBE_ADDS_NO_INCREMENTAL_INFORMATION"),
    }
    out_path = ROOT / "evidence/gate_executive/retrieval_probe_v1_phase2.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {out_path}")
    if not passed:
        print("\n  STOP per the frozen condition. Close the retrieval-probe branch")
        print("  before building any evaluation split and before spending GPU on it.")
        return 2
    print("\n  PROCEED to PHASE_3 fitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
