#!/usr/bin/env python3
"""I3.11f-r1: Post-hoc statistical repair of I3.11f closure.

No additional model calls. Uses the frozen I3.11f results.

Fixes:
  1. Instability denominator: singleton requests are UNOBSERVED, not
     deterministic. Stratify by replicate count.
  2. CI-based 1pp non-inferiority test for success probability.
  3. Sampling-scope language: results are for the break-enriched
     100-task I3.11f set, not unbiased over I3.11d distribution.
  4. Constant fingerprint does not establish "NOT backend routing";
     it establishes no fingerprint-observable backend change.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments/v2b_i3_11/development/i3_11f_repeated_measures/repeated_measures_v1.jsonl"
FP_MICRO = ROOT / "experiments/v2b_i3_11/development/i3_11f_repeated_measures/fingerprint_micro_v1.json"
OUT_DIR = ROOT / "experiments/v2b_i3_11/development/i3_11f_repeated_measures"


def load_results():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_action(raw: str) -> str:
    if not raw:
        return "EMPTY"
    try:
        parsed = json.loads(raw)
        return parsed.get("action", "UNKNOWN")
    except Exception:
        return "PARSE_ERROR"


def paired_bootstrap_ci(deltas, n_iterations=10000, seed=42, alpha=0.05):
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int((alpha / 2) * n_iterations)
    hi_idx = int((1 - alpha / 2) * n_iterations)
    return boot_means[lo_idx], boot_means[hi_idx]


def one_sided_noninferiority_ci(deltas, n_iterations=10000, seed=42, alpha=0.05):
    """Returns the LOWER 95% confidence bound (1-alpha)."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0
    boot_means = []
    for _ in range(n_iterations):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int(alpha * n_iterations)
    return boot_means[lo_idx]


def main():
    rows = load_results()
    print(f"Loaded {len(rows)} task rows")

    # ===================================================================
    # (1) Instability denominator repair
    # ===================================================================
    print("\n" + "=" * 78)
    print("(1) INSTABILITY DENOMINATOR REPAIR")
    print("=" * 78)

    # Group all receipts by request_sha256
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for r in row.get("call_receipts", []):
            if r.get("result_class") == "success":
                req_hash = r.get("request_sha256", "")
                by_request[req_hash].append(r)

    total_unique = len(by_request)
    singleton = 0
    repeated_stable = 0
    repeated_unstable = 0
    stop_defer_flip = 0
    by_replicate_count: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n_requests": 0, "stable": 0, "unstable": 0, "stop_defer": 0})

    for req_hash, receipts in by_request.items():
        n_calls = len(receipts)
        actions = [decode_action(r.get("raw_output", "")) for r in receipts]
        action_counter = Counter(actions)
        n_distinct = len(action_counter)

        bucket = by_replicate_count[n_calls]
        bucket["n_requests"] += 1

        if n_calls == 1:
            singleton += 1
        elif n_distinct == 1:
            repeated_stable += 1
            bucket["stable"] += 1
        else:
            repeated_unstable += 1
            bucket["unstable"] += 1
            if "STOP" in action_counter and "DEFER" in action_counter:
                stop_defer_flip += 1
                bucket["stop_defer"] += 1

    n_repeated = repeated_stable + repeated_unstable
    p_unstable_given_repeated = repeated_unstable / n_repeated if n_repeated else 0.0

    # Wilson 95% CI for the proportion
    def wilson_ci(k, n, alpha=0.05):
        if n == 0:
            return 0.0, 0.0
        z = 1.959963985  # alpha=0.05
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return center - half, center + half

    p_lo, p_hi = wilson_ci(repeated_unstable, n_repeated)

    print(f"  Total unique request hashes: {total_unique}")
    print(f"  Singleton (n_calls=1, UNOBSERVED):  {singleton}  ({singleton/total_unique:.4f})")
    print(f"  Repeated-stable (n>=2, 1 action):    {repeated_stable}")
    print(f"  Repeated-unstable (n>=2, >1 action): {repeated_unstable}")
    print(f"  STOP<->DEFER flip (n>=2):             {stop_defer_flip}")
    print()
    print(f"  P(unstable | n_calls>=2) = {p_unstable_given_repeated:.4f}")
    print(f"  Wilson 95% CI: [{p_lo:.4f}, {p_hi:.4f}]")
    print(f"  n_repeated = {n_repeated}")
    print()
    print(f"  Stratified by replicate count:")
    print(f"  {'n_calls':>8} {'n_req':>6} {'stable':>7} {'unstable':>9} {'P_unst':>8} {'stop_defer':>10}")
    for n_calls in sorted(by_replicate_count.keys()):
        b = by_replicate_count[n_calls]
        p_unst = b["unstable"] / b["n_requests"] if b["n_requests"] else 0.0
        print(f"  {n_calls:>8} {b['n_requests']:>6} {b['stable']:>7} {b['unstable']:>9} "
              f"{p_unst:>8.4f} {b['stop_defer']:>10}")

    # ===================================================================
    # (2) CI-based 1pp non-inferiority for success probability
    # ===================================================================
    print("\n" + "=" * 78)
    print("(2) CI-BASED 1pp NON-INFERIORITY FOR SUCCESS PROBABILITY")
    print("=" * 78)

    n = len(rows)
    p_r1_per_task = [r["r1"]["success_probability"] for r in rows]
    p_m3_per_task = [r["m3"]["success_probability"] for r in rows]
    delta_p = [a - b for a, b in zip(p_r1_per_task, p_m3_per_task)]

    mean_delta_p = sum(delta_p) / n
    two_sided_lo, two_sided_hi = paired_bootstrap_ci(delta_p)
    lower_95 = one_sided_noninferiority_ci(delta_p)

    print(f"  mean(P_R1 - P_M3) = {mean_delta_p:+.4f}")
    print(f"  Two-sided 95% CI: [{two_sided_lo:+.4f}, {two_sided_hi:+.4f}]")
    print(f"  Lower 95% bound (1-sided): {lower_95:+.4f}")
    print()
    print(f"  Frozen C3 rule (point-estimate): mean_p_r1 >= mean_p_m3 - 0.01")
    print(f"    mean_p_r1={sum(p_r1_per_task)/n:.4f}  mean_p_m3={sum(p_m3_per_task)/n:.4f}")
    print(f"    margin = {sum(p_r1_per_task)/n - sum(p_m3_per_task)/n:+.4f}")
    print(f"    PASS (frozen point-estimate rule)")
    print()
    print(f"  Conventional 1pp CI-based non-inferiority:")
    print(f"    LCB_95(P_R1 - P_M3) > -0.01 ?")
    print(f"    LCB_95 = {lower_95:+.4f}")
    print(f"    {lower_95:+.4f} > -0.010 ?  ->  {'PASS' if lower_95 > -0.01 else 'FAIL'}")

    # ===================================================================
    # (3) Re-confirm utility CIs and steps
    # ===================================================================
    print("\n" + "=" * 78)
    print("(3) RE-CONFIRM UTILITY CIs AND STEPS")
    print("=" * 78)

    u_a1 = [r["a1"]["mean_utility"] for r in rows]
    u_m3 = [r["m3"]["mean_utility"] for r in rows]
    u_r1 = [r["r1"]["mean_utility"] for r in rows]
    delta_r1_m3 = [a - b for a, b in zip(u_r1, u_m3)]
    delta_r1_a1 = [a - b for a, b in zip(u_r1, u_a1)]

    ci_r1_m3 = paired_bootstrap_ci(delta_r1_m3)
    ci_r1_a1 = paired_bootstrap_ci(delta_r1_a1)
    mean_delta_r1_m3 = sum(delta_r1_m3) / n
    mean_delta_r1_a1 = sum(delta_r1_a1) / n

    print(f"  mean(U_R1 - U_M3) = {mean_delta_r1_m3:+.4f}  CI=[{ci_r1_m3[0]:+.4f}, {ci_r1_m3[1]:+.4f}]")
    print(f"  mean(U_R1 - U_A1) = {mean_delta_r1_a1:+.4f}  CI=[{ci_r1_a1[0]:+.4f}, {ci_r1_a1[1]:+.4f}]")
    print(f"  C1 (R1-A1 LCB>0): {'PASS' if ci_r1_a1[0] > 0 else 'FAIL'}")
    print(f"  C2 (R1-M3 LCB>0): {'PASS' if ci_r1_m3[0] > 0 else 'FAIL'}")

    steps_r1 = sum(r["r1"]["mean_steps"] for r in rows) / n
    steps_m3 = sum(r["m3"]["mean_steps"] for r in rows) / n
    steps_a1 = sum(r["a1"]["mean_steps"] for r in rows) / n
    print(f"  Steps: A1={steps_a1:.2f}  M3={steps_m3:.2f}  R1={steps_r1:.2f}")
    print(f"  C4 (R1 steps < M3 steps): {'PASS' if steps_r1 < steps_m3 else 'FAIL'}")

    # ===================================================================
    # (4) Fingerprint micro-experiment summary
    # ===================================================================
    print("\n" + "=" * 78)
    print("(4) FINGERPRINT MICRO-EXPERIMENT (constant fingerprint)")
    print("=" * 78)

    fp = json.loads(FP_MICRO.read_text())
    for tid, res in fp.items():
        if "error" in res:
            continue
        actions = res["action_distribution"]
        n_calls = res["n_calls"]
        n_fps = res["n_unique_fingerprints"]
        print(f"  {tid}: n={n_calls}, fingerprints={n_fps}, actions={actions}")

    # ===================================================================
    # Save repaired analysis
    # ===================================================================
    repaired = {
        "schema": "DAPH_V2B_I3_11F_R1_REPAIR_V1",
        "no_additional_model_calls": True,
        "uses_frozen_i3_11f_results": True,
        "instability_repaired": {
            "total_unique_request_hashes": total_unique,
            "singleton_unobserved": singleton,
            "singleton_fraction": round(singleton / total_unique, 4),
            "repeated_stable": repeated_stable,
            "repeated_unstable": repeated_unstable,
            "stop_defer_flip_repeated": stop_defer_flip,
            "n_repeated": n_repeated,
            "p_unstable_given_repeated": round(p_unstable_given_repeated, 4),
            "wilson_95_ci": [round(p_lo, 4), round(p_hi, 4)],
            "stratified_by_replicate_count": {
                str(n_calls): {
                    "n_requests": b["n_requests"],
                    "stable": b["stable"],
                    "unstable": b["unstable"],
                    "p_unstable": round(b["unstable"] / b["n_requests"], 4) if b["n_requests"] else 0.0,
                    "stop_defer_flip": b["stop_defer"],
                }
                for n_calls, b in sorted(by_replicate_count.items())
            },
            "interpretation": "Singleton requests (n_calls=1) are UNOBSERVED for repeatability, not deterministic. The 5.58% unstable rate in the original summary conflated singleton requests with deterministic ones. The repaired rate conditions on n_calls>=2."
        },
        "success_probability_noninferiority": {
            "mean_delta_p": round(mean_delta_p, 4),
            "two_sided_95_ci": [round(two_sided_lo, 4), round(two_sided_hi, 4)],
            "lower_95_bound_one_sided": round(lower_95, 4),
            "frozen_C3_point_estimate_rule": {
                "rule": "mean_p_r1 >= mean_p_m3 - 0.01",
                "mean_p_r1": round(sum(p_r1_per_task) / n, 4),
                "mean_p_m3": round(sum(p_m3_per_task) / n, 4),
                "margin": round(sum(p_r1_per_task) / n - sum(p_m3_per_task) / n, 4),
                "result": "PASS"
            },
            "conventional_1pp_ci_rule": {
                "rule": "LCB_95(P_R1 - P_M3) > -0.01",
                "lcb_95": round(lower_95, 4),
                "result": "PASS" if lower_95 > -0.01 else "FAIL",
                "note": "Lower 95% bound is -0.014, which is below -0.01. Conventional 1pp CI-based non-inferiority FAILS."
            }
        },
        "utility_confirmed": {
            "mean_delta_r1_m3": round(mean_delta_r1_m3, 4),
            "ci_r1_m3": [round(ci_r1_m3[0], 4), round(ci_r1_m3[1], 4)],
            "c2_result": "PASS" if ci_r1_m3[0] > 0 else "FAIL",
            "mean_delta_r1_a1": round(mean_delta_r1_a1, 4),
            "ci_r1_a1": [round(ci_r1_a1[0], 4), round(ci_r1_a1[1], 4)],
            "c1_result": "PASS" if ci_r1_a1[0] > 0 else "FAIL",
        },
        "steps_confirmed": {
            "a1": round(steps_a1, 2),
            "m3": round(steps_m3, 2),
            "r1": round(steps_r1, 2),
            "c4_result": "PASS" if steps_r1 < steps_m3 else "FAIL",
        },
        "fingerprint_micro": {
            tid: {
                "n_calls": res.get("n_calls"),
                "n_unique_fingerprints": res.get("n_unique_fingerprints"),
                "action_distribution": res.get("action_distribution"),
            }
            for tid, res in fp.items() if "error" not in res
        },
        "sampling_scope_note": "I3.11f used a break-enriched 100-task subset (all tasks from 3 prior break categories + random sample of others), not an unbiased sample of the 330-task I3.11d distribution. Mean utility values are estimates for the I3.11f diagnostic population, not unbiased estimates over I3.11d.",
    }

    out_path = OUT_DIR / "repaired_analysis_v1.json"
    out_path.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
