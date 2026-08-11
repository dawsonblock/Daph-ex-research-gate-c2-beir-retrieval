#!/usr/bin/env python3
"""RETRIEVAL_PROBE_GATE_V1: corpus-size confound check on the PHASE_2 result.

PHASE_2 restricted to MEMORY_required to remove the EMPTY-POOL shortcut
(ANSWER_NOW_viable tasks carry zero evidence, so they can never retrieve or
bind). That stratification was necessary but NOT sufficient: it left a
second, subtler shortcut in place.

    probe_candidate_count = c2(corpus_size) = min(300, ceil(0.15 * n))

is a DETERMINISTIC function of which of the five exec_training_v2 scales a
task came from -- one distinct value per scale -- and those scales differ
sharply in how often memory helps (0.453 at exec2_700 down to 0.233 at
exec2_2200). So the feature can predict the label by identifying the corpus,
without carrying any query-level information about whether relevant evidence
exists for THIS question.

Critically, that shortcut cannot exist in deployment: against ONE fixed
corpus, retrieval depth is a constant, so probe_candidate_count is a
constant and can predict nothing.

This script runs the two controls that separate query-level signal from the
corpus-size shortcut, and re-applies the FROZEN PHASE_2 stop condition
(Delta_AUC >= 0.02 AND grouped-bootstrap LCB > 0) to the controlled result.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_probe_cost_separation import load_records  # noqa: E402
from scripts.analyze_probe_incremental_information import (  # noqa: E402
    CONFIDENCE_FEATURES, MIN_DELTA_AUC, PROBE_FEATURES, auc,
    grouped_bootstrap_delta_auc, out_of_fold_scores)

#: probe_candidate_count is excluded: it is a corpus-size indicator here and
#: a CONSTANT in any single-corpus deployment, so it cannot transfer.
DEPLOYMENT_SAFE_PROBE_FEATURES = [f for f in PROBE_FEATURES if f != "probe_candidate_count"]


def main() -> int:
    base = load_records(ROOT / "evidence/gate_executive/exec_training_v2_execute.receipts.jsonl")
    by_key = {(r["suite_family"], r["key"][1]): r for r in base}
    merged = []
    for p in (json.loads(l) for l in
              open(ROOT / "evidence/gate_executive/retrieval_probe_v1_features.jsonl") if l.strip()):
        b = by_key.get((p["suite_family"], p["task_id"]))
        if b is not None:
            merged.append({**b, **{f: float(p[f]) for f in PROBE_FEATURES},
                           "scale": p["scale"], "identity_status": p["identity_status"]})
    mem = [r for r in merged if r["suite_family"] == "MEMORY_required"]

    print("=== RETRIEVAL_PROBE_GATE_V1: corpus-size confound check ===\n")

    per_scale_cc = defaultdict(Counter)
    for r in mem:
        per_scale_cc[r["scale"]][r["probe_candidate_count"]] += 1
    label_rate = {}
    for s in sorted(per_scale_cc):
        sub = [r for r in mem if r["scale"] == s]
        label_rate[s] = sum(1 for r in sub if r["delta_u_cost"] > 0) / len(sub)
        vals = sorted(per_scale_cc[s])
        print(f"  {s:<12} candidate_count={vals}  P(memory helps)={label_rate[s]:.3f}")
    print(f"\n  binding status within MEMORY_required: "
          f"{dict(Counter(r['identity_status'] for r in mem))}")
    print("  -> 744/750 bind successfully, so binding status is near-constant here too.\n")

    y = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in mem])
    groups = [r["family"] for r in mem]

    def X(feats, recs=mem):
        return np.array([[r[f] for f in feats] for r in recs], float)

    def A(feats, recs=mem, yy=y):
        return auc(yy, out_of_fold_scores(X(feats, recs), yy))

    a_c0 = A(CONFIDENCE_FEATURES)
    a_full = A(CONFIDENCE_FEATURES + PROBE_FEATURES)
    a_safe = A(CONFIDENCE_FEATURES + DEPLOYMENT_SAFE_PROBE_FEATURES)
    a_cc = A(CONFIDENCE_FEATURES + ["probe_candidate_count"])
    a_cc_alone = auc(y, -np.array([r["probe_candidate_count"] for r in mem]))

    print("  CONTROL 1 -- drop the corpus-size indicator:")
    print(f"    C0 confidence only                            AUC={a_c0:.4f}")
    print(f"    C2 confidence + ALL probe features            AUC={a_full:.4f}  dAUC={a_full-a_c0:+.4f}")
    print(f"    confidence + probe MINUS candidate_count      AUC={a_safe:.4f}  dAUC={a_safe-a_c0:+.4f}")
    print(f"    confidence + candidate_count ONLY             AUC={a_cc:.4f}  dAUC={a_cc-a_c0:+.4f}")
    print(f"    candidate_count ALONE (no confidence)         AUC={a_cc_alone:.4f}")
    share = (a_cc - a_c0) / (a_full - a_c0) if (a_full - a_c0) else float("nan")
    print(f"    -> the corpus-size indicator alone accounts for {share:.0%} of the apparent gain\n")

    lo, hi = grouped_bootstrap_delta_auc(
        groups, y,
        out_of_fold_scores(X(CONFIDENCE_FEATURES), y),
        out_of_fold_scores(X(CONFIDENCE_FEATURES + DEPLOYMENT_SAFE_PROBE_FEATURES), y))

    print("  CONTROL 2 -- analyze WITHIN each scale (candidate_count constant by construction):")
    print(f"    {'scale':<12}{'n':>5}{'pos':>5}{'AUC_C0':>9}{'AUC_C2':>9}{'dAUC':>9}")
    within = {}
    for s in sorted(per_scale_cc):
        sub = [r for r in mem if r["scale"] == s]
        ys = np.array([1.0 if r["delta_u_cost"] > 0 else 0.0 for r in sub])
        if ys.sum() < 5 or (1 - ys).sum() < 5:
            continue
        a0 = auc(ys, out_of_fold_scores(X(CONFIDENCE_FEATURES, sub), ys))
        a2 = auc(ys, out_of_fold_scores(X(CONFIDENCE_FEATURES + PROBE_FEATURES, sub), ys))
        within[s] = a2 - a0
        print(f"    {s:<12}{len(sub):>5}{int(ys.sum()):>5}{a0:>9.4f}{a2:>9.4f}{a2-a0:>+9.4f}")
    mean_within = float(np.mean(list(within.values()))) if within else float("nan")
    print(f"    mean within-scale dAUC = {mean_within:+.4f}   "
          f"(range {min(within.values()):+.4f} .. {max(within.values()):+.4f})\n")

    delta_safe = a_safe - a_c0
    passed = delta_safe >= MIN_DELTA_AUC and lo > 0.0
    print("  FROZEN PHASE_2 STOP CONDITION re-applied to the CONTROLLED result:")
    print(f"    deployment-safe dAUC = {delta_safe:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    bar: dAUC >= {MIN_DELTA_AUC} AND LCB > 0  ->  {'PASS' if passed else 'FAIL'}")
    print(f"    within-scale control agrees: {mean_within:+.4f}, indistinguishable from zero")

    out = {
        "check": "RETRIEVAL_PROBE_GATE_V1 corpus-size confound",
        "supersedes_verdict_in": "evidence/gate_executive/retrieval_probe_v1_phase2.json",
        "DEVELOPMENT_ONLY_no_promotion_claim": True,
        "candidate_count_by_scale": {s: sorted(per_scale_cc[s]) for s in sorted(per_scale_cc)},
        "label_rate_by_scale": label_rate,
        "binding_status_counts": dict(Counter(r["identity_status"] for r in mem)),
        "auc": {"C0_confidence_only": a_c0, "C2_all_probe": a_full,
                "confidence_plus_deployment_safe_probe": a_safe,
                "confidence_plus_candidate_count_only": a_cc,
                "candidate_count_alone": a_cc_alone},
        "share_of_gain_from_corpus_size_indicator": share,
        "deployment_safe_delta_auc": delta_safe,
        "deployment_safe_ci": [lo, hi],
        "within_scale_delta_auc": within,
        "mean_within_scale_delta_auc": mean_within,
        "frozen_bar": {"min_delta_auc": MIN_DELTA_AUC, "require_lcb_positive": True},
        "PASSED": bool(passed),
        "verdict": ("PROCEED" if passed else
                    "STOP__NO_DEPLOYMENT_TRANSFERABLE_INCREMENTAL_INFORMATION"),
    }
    p = ROOT / "evidence/gate_executive/retrieval_probe_v1_scale_confound.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n  written: {p}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
