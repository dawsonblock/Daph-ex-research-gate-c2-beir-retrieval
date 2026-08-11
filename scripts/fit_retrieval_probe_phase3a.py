#!/usr/bin/env python3
"""RETRIEVAL_PROBE_GATE_V1 PHASE_3A: predictive signal, CPU only.

Per the research-lead split of PHASE_3:

    PHASE_3A (here, no GPU)  fit models, measure held-out discrimination,
                             run feature ablations, keep a continuous
                             target, measure ranking efficiency.
    PHASE_3B (needs A100)    collect real stage latencies, apply THESE
                             already-fitted score functions, and only then
                             calibrate + freeze the operating threshold
                             under the frozen latency utility.

HARD BOUNDARY ENFORCED BY THIS SCRIPT: it makes NO cost claim, computes NO
token-based or latency-based utility, and freezes NO decision threshold.
The frozen cost axis for this study is wall-clock latency
(configs/gate_retrieval_probe_v1_design.json). Substituting a token proxy
here could select an operating point that differs under real cost, so the
substitution is simply not performed.

All discrimination numbers are OUT-OF-FOLD (stratified 5-fold) -- i.e. an
actual held-out fit, not an in-sample feature screen.

POOLED and MEMORY_required-only analyses are reported side by side and never
combined. MEMORY_required-only is the PRIMARY mechanism diagnostic; pooled is
a deployment diagnostic that must never override it.

DEVELOPMENT USE ONLY: consumed split, no promotion claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_probe_incremental_information import (  # noqa: E402
    CONFIDENCE_FEATURES, PROBE_FEATURES, auc, grouped_bootstrap_delta_auc,
    out_of_fold_scores)
from scripts.analyze_answer_probe_cost_separation import load_records  # noqa: E402
from scripts.fit_answer_probe_cost_gate import fit_ols, ols_predict  # noqa: E402
from scripts.train_answer_probe_gate_v2 import ShallowTree  # noqa: E402

RETRIEVAL_SCORE_FEATURES = [
    "probe_top1_retrieval_score", "probe_topk_mean_retrieval_score",
    "probe_retrieval_score_margin",
]
BINDING_AVAILABILITY_FEATURES = [
    "probe_candidate_count", "probe_identity_binding_status_code",
    "probe_relation_extracted",
]

MODELS = {
    "C0_confidence_only": CONFIDENCE_FEATURES,
    "C1_retrieval_probe_only": PROBE_FEATURES,
    "C2_confidence_plus_retrieval": CONFIDENCE_FEATURES + PROBE_FEATURES,
}
ABLATIONS = {
    "A1_retrieval_scores_only": RETRIEVAL_SCORE_FEATURES,
    "A2_binding_availability_only": BINDING_AVAILABILITY_FEATURES,
    "A3_scores_plus_binding": RETRIEVAL_SCORE_FEATURES + BINDING_AVAILABILITY_FEATURES,
    "A4_confidence_plus_scores": CONFIDENCE_FEATURES + RETRIEVAL_SCORE_FEATURES,
    "A5_confidence_plus_scores_plus_binding": (
        CONFIDENCE_FEATURES + RETRIEVAL_SCORE_FEATURES + BINDING_AVAILABILITY_FEATURES),
}
CAPTURE_POINTS = (0.10, 0.20, 0.30, 0.50, 1.00)


def _X(records, feats):
    return np.array([[r[f] for f in feats] for r in records], float)


def _oof_continuous(X, y, folds_seed=9903):
    """Out-of-fold OLS predictions of the continuous target."""
    import random
    rng = random.Random(folds_seed)
    idx = list(range(len(y)))
    rng.shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    oof = np.zeros(len(y))
    for f in range(5):
        test = folds[f]
        train = [i for g in range(5) if g != f for i in folds[g]]
        if not test or not train:
            continue
        w, b, m, s = fit_ols(X[train], y[train])
        for i in test:
            oof[i] = ols_predict(X[i], w, b, m, s)
    return oof


def _oof_tree(X, y, folds_seed=9903):
    import random
    rng = random.Random(folds_seed)
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    folds = [[] for _ in range(5)]
    for k, i in enumerate(pos):
        folds[k % 5].append(i)
    for k, i in enumerate(neg):
        folds[k % 5].append(i)
    oof = np.zeros(len(y))
    for f in range(5):
        test = folds[f]
        train = [i for g in range(5) if g != f for i in folds[g]]
        if not test or not train or len(set(y[train].tolist())) < 2:
            continue
        t = ShallowTree(max_depth=3, min_leaf=10).fit(X[train], y[train])
        for i in test:
            oof[i] = t.predict_proba(X[i])
    return oof


def capture_curve(scores, delta_q) -> dict:
    """Ranking efficiency: escalate the top-x% by score; what fraction of the
    ORACLE's available quality gain does that capture?

    Oracle gain = sum of delta_q over tasks where delta_q > 0 (escalate only
    where memory actually helps). Captured = sum of delta_q over the
    escalated set, so a model that ranks a HARMFUL task highly is penalized
    by that task's negative delta_q -- as it should be.
    """
    order = np.argsort(-np.asarray(scores))
    dq = np.asarray(delta_q, float)[order]
    oracle_total = float(sum(v for v in delta_q if v > 0))
    n = len(dq)
    out = {"oracle_available_gain": oracle_total}
    if oracle_total <= 0:
        return out
    csum = np.cumsum(dq)
    for frac in CAPTURE_POINTS:
        k = max(1, int(round(frac * n)))
        out[f"captured_at_{int(frac*100)}pct"] = float(csum[k - 1] / oracle_total)
    # a random-ranking reference at the same escalation fractions
    total_dq = float(dq.sum())
    for frac in CAPTURE_POINTS:
        out[f"random_reference_at_{int(frac*100)}pct"] = float(frac * total_dq / oracle_total)
    return out


def spearman(a, b) -> float:
    def rank(v):
        order = np.argsort(np.asarray(v, float))
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def analyze(records, label: str) -> dict:
    y = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in records])
    dq = np.array([float(r["q_memory"] - r["q_direct"]) for r in records])
    groups = [r["family"] for r in records]
    res: dict = {"label": label, "n": len(records),
                 "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                 "oracle_available_gain": float(sum(v for v in dq if v > 0))}
    if res["n_positive"] < 2 or res["n_negative"] < 2:
        res["skipped"] = "degenerate class balance"
        return res

    oof = {}
    for name, feats in MODELS.items():
        oof[name] = out_of_fold_scores(_X(records, feats), y)
    oof["C3_shallow_tree_diagnostic"] = _oof_tree(_X(records, MODELS["C2_confidence_plus_retrieval"]), y)

    res["models"] = {}
    for name, scores in oof.items():
        row = {"auc_heldout": auc(y, scores)}
        row.update(capture_curve(scores, dq))
        res["models"][name] = row

    lo, hi = grouped_bootstrap_delta_auc(groups, y, oof["C0_confidence_only"],
                                         oof["C2_confidence_plus_retrieval"])
    res["primary_comparison_C2_minus_C0"] = {
        "delta_auc": res["models"]["C2_confidence_plus_retrieval"]["auc_heldout"]
                     - res["models"]["C0_confidence_only"]["auc_heldout"],
        "lcb_2p5": lo, "ucb_97p5": hi,
    }

    res["ablations"] = {}
    for name, feats in ABLATIONS.items():
        s = out_of_fold_scores(_X(records, feats), y)
        res["ablations"][name] = {
            "auc_heldout": auc(y, s),
            "delta_vs_C0": auc(y, s) - res["models"]["C0_confidence_only"]["auc_heldout"],
        }

    # continuous target, kept alongside the binary one
    cont = _oof_continuous(_X(records, MODELS["C2_confidence_plus_retrieval"]), dq)
    cont_c0 = _oof_continuous(_X(records, CONFIDENCE_FEATURES), dq)
    res["continuous_target"] = {
        "spearman_C2_pred_vs_actual_delta_q": spearman(cont, dq),
        "spearman_C0_pred_vs_actual_delta_q": spearman(cont_c0, dq),
        "capture_curve_C2_regression": capture_curve(cont, dq),
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", default=str(
        ROOT / "evidence/gate_executive/exec_training_v2_execute.receipts.jsonl"))
    ap.add_argument("--probe-features", default=str(
        ROOT / "evidence/gate_executive/retrieval_probe_v1_features.jsonl"))
    args = ap.parse_args()

    base = load_records(args.receipts)
    by_key = {(r["suite_family"], r["key"][1]): r for r in base}
    merged = []
    for p in (json.loads(l) for l in open(args.probe_features) if l.strip()):
        b = by_key.get((p["suite_family"], p["task_id"]))
        if b is not None:
            merged.append({**b, **{f: float(p[f]) for f in PROBE_FEATURES}})

    print("=== RETRIEVAL_PROBE_GATE_V1 PHASE_3A (CPU) ===")
    print("    Fit + diagnose ONLY. No cost model, no threshold frozen.")
    print("    The frozen cost axis is wall-clock latency; it is deliberately")
    print("    NOT proxied by tokens here. Thresholds are PHASE_3B, post-GPU.\n")

    strata = [
        ("MEMORY_required_ONLY__PRIMARY", [r for r in merged if r["suite_family"] == "MEMORY_required"]),
        ("POOLED_deployment_diagnostic", merged),
    ]
    results = []
    for label, recs in strata:
        r = analyze(recs, label)
        results.append(r)
        print(f"  --- {label}  (n={r['n']}, pos={r['n_positive']}, "
              f"oracle_gain={r['oracle_available_gain']:.0f}) ---")
        if "skipped" in r:
            print(f"      {r['skipped']}\n")
            continue
        print(f"    {'model':<34}{'AUC':>8}{'cap@10%':>9}{'cap@20%':>9}{'cap@30%':>9}{'cap@50%':>9}")
        for name, row in r["models"].items():
            print(f"    {name:<34}{row['auc_heldout']:>8.4f}"
                  f"{row.get('captured_at_10pct', float('nan')):>9.3f}"
                  f"{row.get('captured_at_20pct', float('nan')):>9.3f}"
                  f"{row.get('captured_at_30pct', float('nan')):>9.3f}"
                  f"{row.get('captured_at_50pct', float('nan')):>9.3f}")
        rnd = r["models"]["C0_confidence_only"]
        print(f"    {'(random-ranking reference)':<34}{'--':>8}"
              f"{rnd.get('random_reference_at_10pct', float('nan')):>9.3f}"
              f"{rnd.get('random_reference_at_20pct', float('nan')):>9.3f}"
              f"{rnd.get('random_reference_at_30pct', float('nan')):>9.3f}"
              f"{rnd.get('random_reference_at_50pct', float('nan')):>9.3f}")
        pc = r["primary_comparison_C2_minus_C0"]
        print(f"    C2 - C0 held-out Delta_AUC = {pc['delta_auc']:+.4f}  "
              f"LCB={pc['lcb_2p5']:+.4f}")
        print(f"    {'ablation':<44}{'AUC':>8}{'vs C0':>9}")
        for name, row in r["ablations"].items():
            print(f"    {name:<44}{row['auc_heldout']:>8.4f}{row['delta_vs_C0']:>+9.4f}")
        ct = r["continuous_target"]
        print(f"    continuous target: spearman(C2)={ct['spearman_C2_pred_vs_actual_delta_q']:+.4f}  "
              f"spearman(C0)={ct['spearman_C0_pred_vs_actual_delta_q']:+.4f}")
        print()

    out = {
        "phase": "RETRIEVAL_PROBE_GATE_V1 PHASE_3A",
        "design": "configs/gate_retrieval_probe_v1_design.json",
        "DEVELOPMENT_ONLY_no_promotion_claim": True,
        "no_cost_model_applied": True,
        "no_threshold_frozen": True,
        "cost_axis_deferred_to_PHASE_3B": "wall-clock latency (T_G2 + T_composition + T_A1_generation)",
        "heldout_method": "out-of-fold stratified 5-fold",
        "strata": results,
    }
    p = ROOT / "evidence/gate_executive/retrieval_probe_v1_phase3a.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"  written: {p}")
    print("  PHASE_3A ends here. Threshold calibration requires real latency (PHASE_3B).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
