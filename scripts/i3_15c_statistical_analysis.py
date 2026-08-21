"""Statistical analysis module for I3.15c experiments.

Implements:
- Paired bootstrap CIs with median, wins/losses/ties, effect size
- Direct I_phase bootstrap (resample T2+ and DEFER- independently)
- Scientific status classification (CONFIRMED_POSITIVE, POSITIVE_POINT_ESTIMATE, etc.)
- TOST-style equivalence testing with configurable margins
- Cognition-cost receipts and per-cost-contrast analysis

Usage:
    PYTHONPATH=. python3 scripts/i3_15c_statistical_analysis.py \
        --results experiments/v2b_i3_15c/closure/results.jsonl \
        --output experiments/v2b_i3_15c/closure/statistical_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap CI using resampling with replacement."""
    if not values:
        return 0.0, 0.0
    import random
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(n_bootstrap * alpha)]
    hi = means[int(n_bootstrap * (1 - alpha))]
    return lo, hi


def bootstrap_ci_paired(
    deltas: list[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for paired differences (same as bootstrap_ci on deltas)."""
    return bootstrap_ci(deltas, n_bootstrap, confidence, seed)


def bootstrap_i_phase_direct(
    t2_deltas: list[float],
    defer_deltas: list[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Direct bootstrap for I_phase = Delta_T2+ - Delta_DEFER-.

    Resamples T2+ tasks and DEFER- tasks independently,
    computes Delta_T2+ and Delta_DEFER- on each resample,
    then takes the difference.

    Returns (lo, hi, interaction_samples).
    """
    import random
    rng = random.Random(seed)
    if not t2_deltas or not defer_deltas:
        return 0.0, 0.0, []
    n_t2 = len(t2_deltas)
    n_defer = len(defer_deltas)
    interactions = []
    for _ in range(n_bootstrap):
        t2_sample = [t2_deltas[rng.randint(0, n_t2 - 1)] for _ in range(n_t2)]
        defer_sample = [defer_deltas[rng.randint(0, n_defer - 1)] for _ in range(n_defer)]
        delta_t2 = sum(t2_sample) / n_t2
        delta_defer = sum(defer_sample) / n_defer
        interactions.append(delta_t2 - delta_defer)
    interactions.sort()
    alpha = (1 - confidence) / 2
    lo = interactions[int(n_bootstrap * alpha)]
    hi = interactions[int(n_bootstrap * (1 - alpha))]
    return lo, hi, interactions


def tost_equivalence(
    values: list[float],
    margin: float,
    n_bootstrap: int = 5000,
    confidence: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    """TOST-style equivalence test via bootstrap.

    Two one-sided tests:
    H0a: mean <= -margin  vs  H1a: mean > -margin
    H0b: mean >= +margin   vs  H1b: mean < +margin

    Equivalence confirmed if both H0 rejected.
    """
    if not values:
        return {"equivalent": False, "margin": margin, "reason": "no data"}
    lo, hi = bootstrap_ci(values, n_bootstrap, confidence, seed)
    mean_val = sum(values) / len(values)
    # H0a rejected if lower bound of CI > -margin
    reject_h0a = lo > -margin
    # H0b rejected if upper bound of CI < +margin
    reject_h0b = hi < margin
    equivalent = reject_h0a and reject_h0b
    return {
        "equivalent": equivalent,
        "margin": margin,
        "mean": mean_val,
        "ci_90": [lo, hi],
        "reject_h0a_mean_le_neg_margin": reject_h0a,
        "reject_h0b_mean_ge_pos_margin": reject_h0b,
        "status": "EQUIVALENT" if equivalent else "NOT_EQUIVALENT",
    }


# ---------------------------------------------------------------------------
# Scientific status classification
# ---------------------------------------------------------------------------

def classify_status(
    mean: float,
    ci: tuple[float, float],
    equivalence_margin: float | None = None,
) -> str:
    """Classify a contrast into a scientific status."""
    lo, hi = ci
    if lo > 0:
        return "CONFIRMED_POSITIVE"
    if hi < 0:
        return "CONFIRMED_NEGATIVE"
    if equivalence_margin is not None:
        if abs(mean) < equivalence_margin and lo > -equivalence_margin and hi < equivalence_margin:
            return "EQUIVALENT"
    if mean > 0:
        return "POSITIVE_POINT_ESTIMATE"
    if mean < 0:
        return "NEGATIVE_POINT_ESTIMATE"
    return "NO_DIFFERENCE_DETECTED"


# ---------------------------------------------------------------------------
# Contrast computation
# ---------------------------------------------------------------------------

def compute_contrast_detail(
    deltas: list[float],
    name: str,
    equivalence_margin: float | None = None,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a full contrast report with all statistics."""
    if not deltas:
        return {
            "name": name,
            "n": 0,
            "status": "INCONCLUSIVE",
            "mean": 0.0,
            "median": 0.0,
            "ci_95": [0.0, 0.0],
        }
    mean_val = sum(deltas) / len(deltas)
    median_val = statistics.median(deltas)
    ci = bootstrap_ci(deltas, n_bootstrap, 0.95, seed)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    # Cohen's d (paired, using SD of deltas)
    if len(deltas) > 1:
        sd = statistics.stdev(deltas)
        cohen_d = mean_val / sd if sd > 0 else 0.0
    else:
        cohen_d = 0.0
    status = classify_status(mean_val, ci, equivalence_margin)
    result = {
        "name": name,
        "n": len(deltas),
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        "ci_lower": round(ci[0], 4),
        "ci_upper": round(ci[1], 4),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "effect_size_cohen_d": round(cohen_d, 4),
        "scientific_status": status,
    }
    if equivalence_margin is not None:
        tost = tost_equivalence(deltas, equivalence_margin, n_bootstrap, 0.90, seed)
        result["equivalence_test"] = tost
    return result


def compute_all_contrasts(
    results: list[dict[str, Any]],
    equivalence_margin: float = 5.0,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute all pre-registered contrasts with full statistics."""
    # Pair A1 and R1 by task_id
    by_task: dict[str, dict[str, dict]] = {}
    for r in results:
        tid = r.get("task_id")
        arm = r.get("arm")
        if tid and arm:
            by_task.setdefault(tid, {})[arm] = r

    # Build paired deltas by stratum
    strata = {
        "T2_CONFLICT_IMMEDIATE": [],
        "T2_CONFLICT_LATE": [],
        "DEFER_CONTROL": [],
        "ANSWER_CONTROL": [],
    }
    t2_positive_deltas: list[float] = []
    all_deltas: list[float] = []

    for tid, arms in by_task.items():
        if "A1_INFERRED" not in arms or "R1_INFERRED" not in arms:
            continue
        a1 = arms["A1_INFERRED"]
        r1 = arms["R1_INFERRED"]
        u_a1 = a1.get("realized_utility")
        u_r1 = r1.get("realized_utility")
        if u_a1 is None or u_r1 is None:
            continue
        delta = u_r1 - u_a1
        all_deltas.append(delta)

        category = a1.get("category", "")
        if "immediate" in category:
            strata["T2_CONFLICT_IMMEDIATE"].append(delta)
            t2_positive_deltas.append(delta)
        elif "late" in category:
            strata["T2_CONFLICT_LATE"].append(delta)
            t2_positive_deltas.append(delta)
        elif "defer" in category:
            strata["DEFER_CONTROL"].append(delta)
        elif "answer" in category:
            strata["ANSWER_CONTROL"].append(delta)

    # Compute contrasts
    contrasts = {}
    contrasts["Delta_T2+"] = compute_contrast_detail(
        t2_positive_deltas, "Delta_T2+", n_bootstrap=n_bootstrap, seed=seed)
    contrasts["Delta_T2_immediate"] = compute_contrast_detail(
        strata["T2_CONFLICT_IMMEDIATE"], "Delta_T2_immediate",
        n_bootstrap=n_bootstrap, seed=seed)
    contrasts["Delta_T2_late"] = compute_contrast_detail(
        strata["T2_CONFLICT_LATE"], "Delta_T2_late",
        n_bootstrap=n_bootstrap, seed=seed)
    contrasts["Delta_DEFER-"] = compute_contrast_detail(
        strata["DEFER_CONTROL"], "Delta_DEFER-",
        equivalence_margin=equivalence_margin, n_bootstrap=n_bootstrap, seed=seed)
    contrasts["Delta_ANSWER"] = compute_contrast_detail(
        strata["ANSWER_CONTROL"], "Delta_ANSWER",
        equivalence_margin=equivalence_margin, n_bootstrap=n_bootstrap, seed=seed)

    # Direct I_phase bootstrap
    i_phase_lo, i_phase_hi, i_phase_samples = bootstrap_i_phase_direct(
        t2_positive_deltas, strata["DEFER_CONTROL"],
        n_bootstrap=n_bootstrap, seed=seed)
    i_phase_mean = sum(i_phase_samples) / len(i_phase_samples) if i_phase_samples else 0.0
    i_phase_status = classify_status(i_phase_mean, (i_phase_lo, i_phase_hi))
    contrasts["I_phase"] = {
        "name": "I_phase",
        "n_t2": len(t2_positive_deltas),
        "n_defer": len(strata["DEFER_CONTROL"]),
        "mean": round(i_phase_mean, 4),
        "ci_95": [round(i_phase_lo, 4), round(i_phase_hi, 4)],
        "ci_lower": round(i_phase_lo, 4),
        "ci_upper": round(i_phase_hi, 4),
        "scientific_status": i_phase_status,
        "method": "direct_independent_bootstrap",
        "note": "T2+ and DEFER- tasks resampled independently; difference computed per bootstrap iteration",
    }

    # False T2 on controls
    false_t2 = {"DEFER_CONTROL": 0, "ANSWER_CONTROL": 0}
    for tid, arms in by_task.items():
        r1 = arms.get("R1_INFERRED")
        if not r1:
            continue
        category = r1.get("category", "")
        if "defer" in category and r1.get("r1_triggered", False):
            false_t2["DEFER_CONTROL"] += 1
        if "answer" in category and r1.get("r1_triggered", False):
            false_t2["ANSWER_CONTROL"] += 1

    # Cost metrics
    cost = compute_cost_metrics(by_task, t2_positive_deltas, strata, seed)

    return {
        "contrasts": contrasts,
        "false_t2_on_controls": false_t2,
        "cost_metrics": cost,
        "equivalence_margin": equivalence_margin,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_total_trajectories": len(results),
        "n_paired_tasks": len(all_deltas),
    }


def compute_cost_metrics(
    by_task: dict[str, dict[str, dict]],
    t2_deltas: list[float],
    strata: dict[str, list[float]],
    seed: int,
) -> dict[str, Any]:
    """Compute cognition-cost contrasts."""
    # Steps deltas
    steps_t2 = []
    steps_defer = []
    steps_answer = []
    tokens_t2 = []
    redundant_t2 = []

    for tid, arms in by_task.items():
        if "A1_INFERRED" not in arms or "R1_INFERRED" not in arms:
            continue
        a1 = arms["A1_INFERRED"]
        r1 = arms["R1_INFERRED"]
        category = a1.get("category", "")

        s_a1 = a1.get("steps", 0) or 0
        s_r1 = r1.get("steps", 0) or 0
        d_steps = s_r1 - s_a1

        # Token sums
        tok_a1 = sum(c.get("completion_tokens", 0) or 0 for c in a1.get("model_call_log", []))
        tok_r1 = sum(c.get("completion_tokens", 0) or 0 for c in r1.get("model_call_log", []))
        d_tokens = tok_r1 - tok_a1

        # Redundant actions
        red_a1 = a1.get("redundant_action_summary", {}).get("total_redundant", 0)
        red_r1 = r1.get("redundant_action_summary", {}).get("total_redundant", 0)
        d_redundant = red_r1 - red_a1

        if "immediate" in category or "late" in category:
            steps_t2.append(d_steps)
            tokens_t2.append(d_tokens)
            redundant_t2.append(d_redundant)
        elif "defer" in category:
            steps_defer.append(d_steps)
        elif "answer" in category:
            steps_answer.append(d_steps)

    # Step-limit rates
    t2_r1_step_limit = 0
    t2_a1_step_limit = 0
    t2_count = 0
    for tid, arms in by_task.items():
        if "A1_INFERRED" not in arms or "R1_INFERRED" not in arms:
            continue
        category = arms["A1_INFERRED"].get("category", "")
        if "immediate" not in category and "late" not in category:
            continue
        t2_count += 1
        if arms["R1_INFERRED"].get("terminal_result") == "STEP_LIMIT":
            t2_r1_step_limit += 1
        if arms["A1_INFERRED"].get("terminal_result") == "STEP_LIMIT":
            t2_a1_step_limit += 1

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def safe_ci(lst):
        if not lst:
            return [0.0, 0.0]
        lo, hi = bootstrap_ci(lst, 2000, 0.95, seed)
        return [round(lo, 4), round(hi, 4)]

    return {
        "Delta_Steps_T2+": {
            "mean": round(safe_mean(steps_t2), 4),
            "ci_95": safe_ci(steps_t2),
            "n": len(steps_t2),
        },
        "Delta_Tokens_T2+": {
            "mean": round(safe_mean(tokens_t2), 4),
            "ci_95": safe_ci(tokens_t2),
            "n": len(tokens_t2),
        },
        "Delta_RedundantActions_T2+": {
            "mean": round(safe_mean(redundant_t2), 4),
            "ci_95": safe_ci(redundant_t2),
            "n": len(redundant_t2),
        },
        "P_step_limit_R1_T2+": {"rate": t2_r1_step_limit / max(t2_count, 1), "n": t2_count},
        "P_step_limit_A1_T2+": {"rate": t2_a1_step_limit / max(t2_count, 1), "n": t2_count},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="I3.15c statistical analysis")
    parser.add_argument("--results", required=True, help="Path to results.jsonl or results.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--equivalence-margin", type=float, default=5.0,
                        help="Equivalence margin for control contrasts (default: 5.0)")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if results_path.suffix == ".jsonl":
        results = []
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    else:
        with open(results_path) as f:
            results = json.load(f)

    print(f"Loaded {len(results)} trajectories from {results_path}")

    # Compute
    analysis = compute_all_contrasts(
        results,
        equivalence_margin=args.equivalence_margin,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"\nEquivalence margin: ±{args.equivalence_margin}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Paired tasks: {analysis['n_paired_tasks']}")

    print("\nContrasts:")
    for name, info in analysis["contrasts"].items():
        print(f"\n  {name}:")
        print(f"    mean: {info.get('mean', 0):.4f}")
        print(f"    median: {info.get('median', 'N/A')}")
        print(f"    CI_95: [{info.get('ci_lower', 0):.4f}, {info.get('ci_upper', 0):.4f}]")
        print(f"    n: {info.get('n', 'N/A')}")
        if "wins" in info:
            print(f"    wins/losses/ties: {info['wins']}/{info['losses']}/{info['ties']}")
            print(f"    Cohen's d: {info.get('effect_size_cohen_d', 'N/A')}")
        print(f"    status: {info.get('scientific_status', 'N/A')}")
        if "equivalence_test" in info:
            eq = info["equivalence_test"]
            print(f"    equivalence: {eq['status']} (margin={eq['margin']})")

    print(f"\nFalse T2 on controls: {analysis['false_t2_on_controls']}")

    print("\nCost metrics:")
    for name, info in analysis["cost_metrics"].items():
        if isinstance(info, dict):
            if "mean" in info:
                print(f"  {name}: mean={info['mean']:.4f} CI={info.get('ci_95', 'N/A')} n={info.get('n', 'N/A')}")
            elif "rate" in info:
                print(f"  {name}: rate={info['rate']:.4f} n={info.get('n', 'N/A')}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()
