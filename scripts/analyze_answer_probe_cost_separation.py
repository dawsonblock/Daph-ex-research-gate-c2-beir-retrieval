#!/usr/bin/env python3
"""ANSWER_PROBE_COST_GATE_V1 PHASE_1 + PHASE_2, per
configs/gate_answer_probe_cost_v1_design.json.

PHASE_1: recompute action labels from the FROZEN cost-aware utility.
PHASE_2: check whether the cheap confidence features separate
         Delta_U_cost>0 from Delta_U_cost<=0 at the frozen threshold.

GPU-free. Run against the CONSUMED exec_training_v2 receipts for DESIGN USE
ONLY -- this script deliberately makes no promotion claim and prints that
constraint in its own output. Promotion requires the fresh evaluation split
built in PHASE_4.

Reuses the separation statistics from ANSWER_PROBE_GATE_V1 unchanged
(cohens_d, grouped_bootstrap_mean_diff_ci) so the frozen SEPARATION
criterion is applied by identical code, not a re-implementation that could
silently drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_answer_probe_separation import (  # noqa: E402
    cohens_d, grouped_bootstrap_mean_diff_ci)

FEATURES = [
    "mean_token_confidence", "min_token_confidence", "sequence_confidence",
    "mean_entropy", "answer_length",
]
COHEN_D_THRESHOLD = 0.3

# --- FROZEN in configs/gate_answer_probe_cost_v1_design.json PHASE_0 -------
EPSILON = 0.01
C_REF = 1000.0
LAMBDA_T_GRID = [0.01, 0.05, 0.1, 0.25]


def cost_tokens(receipt: dict) -> int:
    """C_i(a) = prompt_tokens + completion_tokens -- the only cost field
    instrumented for BOTH actions (latency_seconds is A1-only)."""
    return int(receipt["prompt_tokens"]) + int(receipt["completion_tokens"])


def utility_lex(q: int, c: int) -> float:
    """PRIMARY frozen utility: U = Q - EPSILON * C / C_REF. Strictly
    lexicographic on this instrument (max cost term ~0.0029 << 1.0)."""
    return q - EPSILON * c / C_REF


def load_records(receipts_path: str) -> list[dict]:
    receipts = [json.loads(l) for l in open(receipts_path) if l.strip()]
    by_task: dict[tuple, dict] = defaultdict(dict)
    for r in receipts:
        by_task[(r["suite_family"], r["task_id"])][r["action"]] = r

    records = []
    for key, actions in by_task.items():
        if "A0_ANSWER_NOW" not in actions or "A1_USE_CERTIFIED_MEMORY" not in actions:
            continue
        a0, a1 = actions["A0_ANSWER_NOW"], actions["A1_USE_CERTIFIED_MEMORY"]
        q0, q1 = int(a0["correct"]), int(a1["correct"])
        c0, c1 = cost_tokens(a0), cost_tokens(a1)
        u0, u1 = utility_lex(q0, c0), utility_lex(q1, c1)
        records.append({
            "key": key, "suite_family": key[0], "family": a0["family"],
            "q_direct": q0, "q_memory": q1, "c_direct": c0, "c_memory": c1,
            "u_direct": u0, "u_memory": u1,
            "delta_u_cost": u1 - u0,
            "delta_q": q1 - q0,
            **{f: a0[f] for f in FEATURES if f in a0},
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    args = ap.parse_args()

    records = load_records(args.receipts)
    n = len(records)
    print("=== ANSWER_PROBE_COST_GATE_V1  PHASE_1 + PHASE_2 ===")
    print("    DESIGN USE ONLY -- consumed split, no promotion claim may be made here.\n")
    print(f"  frozen utility: U = Q - {EPSILON} * C / {C_REF:.0f}   (C = prompt+completion tokens)")
    print(f"  paired tasks: {n}\n")

    # --- PHASE_1: recomputed labels ---------------------------------------
    escalate = [r for r in records if r["delta_u_cost"] > 0]
    accept = [r for r in records if r["delta_u_cost"] <= 0]
    print("  --- PHASE_1: labels recomputed from utility (not inherited) ---")
    print(f"    MEMORY_preferred (Delta_U_cost > 0): {len(escalate)}")
    print(f"    ANSWER_preferred (Delta_U_cost <= 0): {len(accept)}")

    # verify the derived consequence stated in the frozen protocol
    mismatch = [r for r in records if (r["delta_u_cost"] > 0) != (r["delta_q"] > 0)]
    print(f"    consistency check -- tasks where (Delta_U_cost>0) != (Delta_Q>0): {len(mismatch)}"
          f"  {'OK (matches the frozen protocol derivation)' if not mismatch else 'UNEXPECTED'}")

    by_fam = defaultdict(lambda: {"escalate": 0, "accept": 0})
    for r in records:
        by_fam[r["suite_family"]]["escalate" if r["delta_u_cost"] > 0 else "accept"] += 1
    print("\n    per-family (MANDATORY diagnostic -- family/label correlation):")
    for fam, c in sorted(by_fam.items()):
        tot = c["escalate"] + c["accept"]
        print(f"      {fam:<22} MEMORY_preferred={c['escalate']:>4}  ANSWER_preferred={c['accept']:>4}"
              f"   ({100*c['escalate']/tot:.1f}% escalate)")

    # --- cost accounting (mandatory diagnostic) ---------------------------
    mc0 = sum(r["c_direct"] for r in records) / n
    mc1 = sum(r["c_memory"] for r in records) / n
    mq0 = sum(r["q_direct"] for r in records) / n
    mq1 = sum(r["q_memory"] for r in records) / n
    print(f"\n    cost accounting: mean tokens  ANSWER={mc0:.1f}  MEMORY={mc1:.1f}  (+{mc1-mc0:.1f})")
    print(f"                     mean quality ANSWER={mq0:.4f}  MEMORY={mq1:.4f}")

    if not escalate or not accept:
        print("\n  ABORT: one label class is empty -- cannot compute separation.")
        return 1

    # --- PHASE_2: separation check ----------------------------------------
    print("\n  --- PHASE_2: does the cheap probe separate the utility classes? ---")
    print(f"    {'feature':<26}{'mean(MEM_pref)':>16}{'mean(ANS_pref)':>16}{'cohens_d':>11}{'CI_excl_0':>12}")
    any_clears = False
    feature_rows = {}
    for feat in FEATURES:
        a = [r[feat] for r in escalate if feat in r]
        b = [r[feat] for r in accept if feat in r]
        if not a or not b:
            continue
        d = cohens_d(a, b)
        lo, hi = grouped_bootstrap_mean_diff_ci(
            [(r["family"], r[feat]) for r in escalate if feat in r],
            [(r["family"], r[feat]) for r in accept if feat in r])
        ci_excl = not (lo <= 0.0 <= hi)
        clears = abs(d) >= COHEN_D_THRESHOLD or ci_excl
        any_clears = any_clears or clears
        feature_rows[feat] = {"mean_memory_preferred": sum(a)/len(a),
                              "mean_answer_preferred": sum(b)/len(b),
                              "cohens_d": d, "ci_excludes_zero": ci_excl, "clears": clears}
        print(f"    {feat:<26}{sum(a)/len(a):>16.4f}{sum(b)/len(b):>16.4f}{d:>11.4f}"
              f"{str(ci_excl):>12}{'  <-- CLEARS' if clears else ''}")

    print(f"\n    frozen threshold: |Cohen's d| >= {COHEN_D_THRESHOLD} OR grouped-bootstrap CI excludes 0")

    out = {
        "phase": "ANSWER_PROBE_COST_GATE_V1 PHASE_1+PHASE_2",
        "design": "configs/gate_answer_probe_cost_v1_design.json",
        "DESIGN_USE_ONLY": True,
        "promotion_claim_made": False,
        "frozen_utility": {"form": "U = Q - EPSILON*C/C_REF", "EPSILON": EPSILON,
                           "C_REF": C_REF, "cost": "prompt_tokens + completion_tokens"},
        "n_total": n,
        "labels": {"MEMORY_preferred": len(escalate), "ANSWER_preferred": len(accept)},
        "label_derivation_consistency_mismatches": len(mismatch),
        "per_family_labels": {k: dict(v) for k, v in by_fam.items()},
        "cost_accounting": {"mean_tokens_answer": mc0, "mean_tokens_memory": mc1,
                            "mean_quality_answer": mq0, "mean_quality_memory": mq1},
        "separation": feature_rows,
        "separation_exists": any_clears,
    }
    out_path = Path(args.receipts).with_suffix(".cost_separation.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if any_clears:
        print("\n    SEPARATION EXISTS -- PHASE_3 (fit controller on this development data) is permitted.")
        print("    Reminder: promotion still requires the PHASE_4 fresh evaluation split.")
        print(f"\n  written: {out_path}")
        return 0
    print("\n    NO FEATURE CLEARS THE FROZEN THRESHOLD -- STOP per the frozen protocol.")
    print("    The cheap confidence probe carries no information about cost-aware action")
    print("    value either. Do not fit a controller; do not build an evaluation split.")
    print(f"\n  written: {out_path}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
