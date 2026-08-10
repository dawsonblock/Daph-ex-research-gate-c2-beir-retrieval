#!/usr/bin/env python3
"""ANSWER_PROBE_GATE_V1 stop-gate: does confidence separate MEMORY-strict-win
from ANSWER-strict-win tasks at all? Per configs/gate_answer_probe_v1_design.json
SEPARATION_CRITERION, decided BEFORE this script ever saw real data.

GPU-free -- operates entirely on the receipts written by
scripts/run_exec_training_v1_collection.py --execute. Run this FIRST, before
scripts/train_answer_probe_gate.py: if no feature clears the frozen
threshold, STOP per the frozen protocol -- do not train.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.experiment_integrity.executive_bootstrap import (  # noqa: E402
    grouped_lcb_executive_opportunity)

FEATURES = [
    "mean_token_confidence", "min_token_confidence", "sequence_confidence",
    "mean_entropy", "answer_length",
]
COHEN_D_THRESHOLD = 0.3


def cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    return (ma - mb) / pooled_sd if pooled_sd > 1e-12 else 0.0


def grouped_bootstrap_mean_diff_ci(pairs_a: list[tuple[str, float]],
                                   pairs_b: list[tuple[str, float]],
                                   iterations: int = 2000, seed: int = 12345) -> tuple[float, float]:
    """95% CI of mean(a) - mean(b), resampling GROUPS (family) with
    replacement independently for each side -- reuses this project's
    established grouped-resampling convention."""
    import random
    ga: dict[str, list[float]] = defaultdict(list)
    gb: dict[str, list[float]] = defaultdict(list)
    for k, v in pairs_a:
        ga[k].append(v)
    for k, v in pairs_b:
        gb[k].append(v)
    keys_a, keys_b = sorted(ga), sorted(gb)
    if not keys_a or not keys_b:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        picked_a = [ga[keys_a[rng.randrange(len(keys_a))]] for _ in keys_a]
        picked_b = [gb[keys_b[rng.randrange(len(keys_b))]] for _ in keys_b]
        flat_a = [v for g in picked_a for v in g]
        flat_b = [v for g in picked_b for v in g]
        if flat_a and flat_b:
            diffs.append(sum(flat_a) / len(flat_a) - sum(flat_b) / len(flat_b))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True, help="path to the collection run's .receipts.jsonl")
    args = ap.parse_args()

    receipts = [json.loads(l) for l in open(args.receipts) if l.strip()]
    by_task: dict[tuple, dict] = defaultdict(dict)
    family_of: dict[tuple, str] = {}
    for r in receipts:
        key = (r["suite_family"], r["task_id"])
        by_task[key][r["action"]] = r
        family_of[key] = r["family"]

    records = []
    for key, actions in by_task.items():
        if "A0_ANSWER_NOW" not in actions or "A1_USE_CERTIFIED_MEMORY" not in actions:
            continue
        a0, a1 = actions["A0_ANSWER_NOW"], actions["A1_USE_CERTIFIED_MEMORY"]
        if "correct" not in a0 or "correct" not in a1:
            print(f"ABORT: {key} missing 'correct' -- was this receipts file from --execute?")
            return 1
        q0, q1 = int(a0["correct"]), int(a1["correct"])
        delta_u = q1 - q0
        records.append({
            "key": key, "family": family_of[key], "delta_u": delta_u,
            **{f: a0[f] for f in FEATURES if f in a0},
        })

    n = len(records)
    print(f"=== ANSWER_PROBE_GATE_V1 separation check ({n} paired tasks) ===\n")

    memory_win = [r for r in records if r["delta_u"] > 0]
    answer_win = [r for r in records if r["delta_u"] < 0]
    tie = [r for r in records if r["delta_u"] == 0]
    print(f"  MEMORY_strict_win: {len(memory_win)}   ANSWER_strict_win: {len(answer_win)}   TIE: {len(tie)}\n")

    if not memory_win or not answer_win:
        print("  ABORT: at least one of MEMORY_strict_win / ANSWER_strict_win is empty -- "
              "cannot compute separation. STOP per frozen protocol.")
        return 1

    any_clears_threshold = False
    print(f"  {'feature':<30}{'mean(MEM_win)':>15}{'mean(ANS_win)':>15}{'cohens_d':>10}{'CI_excl_0':>12}")
    for feat in FEATURES:
        mem_vals = [r[feat] for r in memory_win if feat in r]
        ans_vals = [r[feat] for r in answer_win if feat in r]
        if not mem_vals or not ans_vals:
            continue
        d = cohens_d(mem_vals, ans_vals)
        pairs_mem = [(r["family"], r[feat]) for r in memory_win if feat in r]
        pairs_ans = [(r["family"], r[feat]) for r in answer_win if feat in r]
        lo, hi = grouped_bootstrap_mean_diff_ci(pairs_mem, pairs_ans)
        ci_excludes_zero = not (lo <= 0.0 <= hi)
        clears = abs(d) >= COHEN_D_THRESHOLD or ci_excludes_zero
        any_clears_threshold = any_clears_threshold or clears
        marker = "  <-- CLEARS" if clears else ""
        print(f"  {feat:<30}{sum(mem_vals)/len(mem_vals):>15.4f}{sum(ans_vals)/len(ans_vals):>15.4f}"
              f"{d:>10.4f}{str(ci_excludes_zero):>12}{marker}")

    print(f"\n  frozen threshold: |Cohen's d| >= {COHEN_D_THRESHOLD} OR grouped-bootstrap CI excludes 0")
    if any_clears_threshold:
        print("\n  SEPARATION EXISTS -- proceed to scripts/train_answer_probe_gate.py")
        return 0
    else:
        print("\n  NO SEPARATION CLEARS THE FROZEN THRESHOLD -- STOP per "
              "configs/gate_answer_probe_v1_design.json. Do not train. This is a "
              "valid, useful negative result: confidence carries no action-value "
              "information for this task set, not a failure requiring feature "
              "engineering after the fact.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
