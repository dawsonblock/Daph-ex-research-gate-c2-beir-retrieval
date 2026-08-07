#!/usr/bin/env python3
"""Gate C4 analyzer — comprehensive analysis of the integrated memory pipeline.

Computes:
- Arm quality (partial-credit metric) and binary correct rate
- Paired adjacent deltas (C4-k vs C4-(k-1))
- Task flips: improve / regress / unchanged
- Grouped family bootstrap CI for primary quality delta
- Per-regime breakdown (canonical, abbreviation, alias, description)
- Identity resolution rate (EXACT, RESOLVED, AMBIGUOUS, UNRESOLVED)
- S2c live rate
- Complete Set Retention (CSR)
- Role retention (answer, bridge, identity)
- Selector Gap Capture (SGC)
- Oracle Gap Capture (OGC)
- Arm parity validation

Usage:
    python scripts/analyze_gate_c4.py [--dir evidence/gate_c4/full/development_evaluator_v2]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = ["C4_0", "C4_1", "C4_2", "C4_3", "C4_4", "C4_5", "C4_6"]
GROUPING_KEYS = ("family", "template_id", "source_cluster_id")
BOOTSTRAP = 10000
SEED = 20260807


def load_arm(path: Path) -> list[dict]:
    """Load receipts for one arm."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_all(base_dir: Path) -> dict[str, list[dict]]:
    """Load all arms."""
    arms = {}
    for arm_id in ARMS:
        p = base_dir / f"{arm_id}.jsonl"
        if p.exists():
            arms[arm_id] = load_arm(p)
    return arms


def _quality(receipt: dict) -> float:
    return receipt["evaluator_annotation"]["quality"]


def _correct(receipt: dict) -> bool:
    return receipt["evaluator_annotation"]["correct"]


def _regime(receipt: dict) -> str:
    return receipt["evaluator_annotation"].get("metadata", {}).get("entity_regime", "unknown")


def _family(receipt: dict) -> str:
    return receipt["evaluator_annotation"].get("family", "unknown")


def _cluster(receipt: dict) -> str:
    return receipt["evaluator_annotation"].get("source_cluster_id", "unknown")


def _template(receipt: dict) -> str:
    return receipt["evaluator_annotation"].get("template_id",
        receipt["evaluator_annotation"].get("metadata", {}).get("template_id", "unknown"))


def _task_id(receipt: dict) -> str:
    return receipt.get("task_id", receipt.get("runtime_payload", {}).get("task_id", ""))


def _identity_status(receipt: dict) -> str:
    return receipt["runtime_payload"]["identity"]["status"]


def _selector(receipt: dict) -> str:
    return receipt["runtime_payload"]["selection"]["selector"]


def _csr(receipt: dict) -> float:
    return receipt["evaluator_annotation"].get("csr", 0.0)


def _role_retention(receipt: dict) -> dict:
    return receipt["evaluator_annotation"].get("role_retention", {})


def _second_pass(receipt: dict) -> bool:
    return receipt["runtime_payload"]["query"].get("second_pass_performed", False)


def arm_quality(receipts: list[dict]) -> float:
    return sum(_quality(r) for r in receipts) / len(receipts) if receipts else 0.0


def arm_correct_rate(receipts: list[dict]) -> float:
    return sum(1 for r in receipts if _correct(r)) / len(receipts) if receipts else 0.0


def grouped_bootstrap_ci(
    deltas: list[float],
    groups: list[str],
    *,
    resamples: int = BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Grouped bootstrap: resample whole groups, preserving within-group correlation.
    Returns (mean, lower, upper)."""
    if not deltas:
        return (float("nan"), float("nan"), float("nan"))

    by_group: dict[str, list[float]] = defaultdict(list)
    for d, g in zip(deltas, groups):
        by_group[g].append(d)

    keys = sorted(by_group.keys())
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn: list[float] = []
        for _ in range(len(keys)):
            drawn.extend(by_group[keys[rng.randrange(len(keys))]])
        if drawn:
            means.append(sum(drawn) / len(drawn))

    means.sort()
    if not means:
        return (float("nan"), float("nan"), float("nan"))
    n = len(means)
    return (
        sum(deltas) / len(deltas),
        means[int(0.025 * n)],
        means[int(0.975 * n)],
    )


def paired_deltas(arm_a: list[dict], arm_b: list[dict]) -> dict:
    """Compute paired deltas between arm_b and arm_a (b - a) on same tasks."""
    by_id_a = {_task_id(r): r for r in arm_a}
    by_id_b = {_task_id(r): r for r in arm_b}

    common_ids = sorted(set(by_id_a) & set(by_id_b))
    deltas = []
    groups_family = []
    groups_cluster = []
    groups_template = []
    flips = {"improve": 0, "regress": 0, "unchanged": 0}
    flip_details = {"improve": [], "regress": []}

    for tid in common_ids:
        qa = _quality(by_id_a[tid])
        qb = _quality(by_id_b[tid])
        delta = qb - qa
        deltas.append(delta)
        groups_family.append(_family(by_id_a[tid]))
        groups_cluster.append(_cluster(by_id_a[tid]))
        groups_template.append(_template(by_id_a[tid]))

        if delta > 0:
            flips["improve"] += 1
            flip_details["improve"].append(tid)
        elif delta < 0:
            flips["regress"] += 1
            flip_details["regress"].append(tid)
        else:
            flips["unchanged"] += 1

    mean, lo, hi = grouped_bootstrap_ci(deltas, groups_family)
    _, lo_cluster, hi_cluster = grouped_bootstrap_ci(deltas, groups_cluster)
    _, lo_template, hi_template = grouped_bootstrap_ci(deltas, groups_template)
    return {
        "n": len(common_ids),
        "mean_delta": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_lower_cluster": lo_cluster,
        "ci_upper_cluster": hi_cluster,
        "ci_lower_template": lo_template,
        "ci_upper_template": hi_template,
        "flips": flips,
        "flip_details": flip_details,
    }


def per_regime_breakdown(receipts: list[dict]) -> dict[str, dict]:
    """Break down quality and correct rate by entity regime."""
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for r in receipts:
        by_regime[_regime(r)].append(r)

    result = {}
    for regime, recs in sorted(by_regime.items()):
        result[regime] = {
            "n": len(recs),
            "quality": arm_quality(recs),
            "correct_rate": arm_correct_rate(recs),
        }
    return result


def identity_stats(receipts: list[dict]) -> dict:
    """Compute identity resolution statistics."""
    counts = defaultdict(int)
    for r in receipts:
        counts[_identity_status(r)] += 1
    n = len(receipts)
    return {
        "n": n,
        "EXACT": counts.get("EXACT", 0),
        "RESOLVED": counts.get("RESOLVED", 0),
        "AMBIGUOUS": counts.get("AMBIGUOUS", 0),
        "UNRESOLVED": counts.get("UNRESOLVED", 0),
        "exact_rate": counts.get("EXACT", 0) / n if n else 0,
        "resolved_rate": counts.get("RESOLVED", 0) / n if n else 0,
        "ambiguous_rate": counts.get("AMBIGUOUS", 0) / n if n else 0,
        "unresolved_rate": counts.get("UNRESOLVED", 0) / n if n else 0,
    }


def selector_stats(receipts: list[dict]) -> dict:
    """Compute selector usage statistics."""
    counts = defaultdict(int)
    for r in receipts:
        counts[_selector(r)] += 1
    n = len(receipts)
    return {
        "n": n,
        "s2c": counts.get("s2c", 0),
        "s0": counts.get("s0", 0),
        "srel": counts.get("srel", 0),
        "oracle": counts.get("oracle", 0),
        "oracle_evidence": counts.get("oracle_evidence", 0),
        "s2c_live_rate": counts.get("s2c", 0) / n if n else 0,
    }


def iterative_stats(receipts: list[dict]) -> dict:
    """Compute iterative retrieval statistics."""
    n = len(receipts)
    second_pass = sum(1 for r in receipts if _second_pass(r))
    return {
        "n": n,
        "second_pass_performed": second_pass,
        "second_pass_rate": second_pass / n if n else 0,
    }


def csr_stats(receipts: list[dict]) -> dict:
    """Compute Complete Set Retention statistics."""
    n = len(receipts)
    total_csr = sum(_csr(r) for r in receipts)
    return {
        "n": n,
        "mean_csr": total_csr / n if n else 0,
    }


def role_retention_stats(receipts: list[dict]) -> dict:
    """Compute mean role retention."""
    n = len(receipts)
    if n == 0:
        return {"n": 0}
    answer_rr = sum(_role_retention(r).get("answer_retention", 1.0) for r in receipts) / n
    bridge_rr = sum(_role_retention(r).get("bridge_retention", 1.0) for r in receipts) / n
    identity_rr = sum(_role_retention(r).get("identity_retention", 1.0) for r in receipts) / n
    return {
        "n": n,
        "answer_retention": answer_rr,
        "bridge_retention": bridge_rr,
        "identity_retention": identity_rr,
    }


def selector_gap_capture(arms: dict[str, list[dict]]) -> float | None:
    """SGC = (C4_4 - C4_3) / (C4_5 - C4_3)."""
    if "C4_3" not in arms or "C4_4" not in arms or "C4_5" not in arms:
        return None
    q3 = arm_quality(arms["C4_3"])
    q4 = arm_quality(arms["C4_4"])
    q5 = arm_quality(arms["C4_5"])
    gap = q5 - q3
    if abs(gap) < 1e-9:
        return None
    return (q4 - q3) / gap


def oracle_gap_capture(arms: dict[str, list[dict]]) -> float | None:
    """OGC = (C4_5 - C4_0) / (C4_6 - C4_0)."""
    if "C4_0" not in arms or "C4_5" not in arms or "C4_6" not in arms:
        return None
    q0 = arm_quality(arms["C4_0"])
    q5 = arm_quality(arms["C4_5"])
    q6 = arm_quality(arms["C4_6"])
    gap = q6 - q0
    if abs(gap) < 1e-9:
        return None
    return (q5 - q0) / gap


def validate_parity(arms: dict[str, list[dict]]) -> dict:
    """Validate arm parity: arms differ only where expected."""
    # Check that all arms have the same task set
    task_sets = {arm: set(_task_id(r) for r in recs) for arm, recs in arms.items()}
    common = set.intersection(*task_sets.values()) if task_sets else set()
    return {
        "all_arms_same_tasks": all(s == common for s in task_sets.values()),
        "n_common_tasks": len(common),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Gate C4 results")
    parser.add_argument("--dir", type=Path,
                        default=ROOT / "evidence/gate_c4/full/development_evaluator_v2",
                        help="Directory containing C4_*.jsonl receipts")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    arms = load_all(args.dir)
    if not arms:
        print(f"No receipts found in {args.dir}")
        return

    report = {
        "source_dir": str(args.dir),
        "arms_loaded": list(arms.keys()),
        "parity": validate_parity(arms),
        "arm_summary": {},
        "adjacent_deltas": {},
        "primary_delta": {},
        "per_regime": {},
        "identity_stats": {},
        "selector_stats": {},
        "iterative_stats": {},
        "csr_stats": {},
        "role_retention": {},
        "selector_gap_capture": None,
        "oracle_gap_capture": None,
    }

    # Arm summaries
    for arm_id in sorted(arms):
        recs = arms[arm_id]
        report["arm_summary"][arm_id] = {
            "n": len(recs),
            "quality": arm_quality(recs),
            "correct_rate": arm_correct_rate(recs),
        }

    # Adjacent deltas
    for i in range(1, len(ARMS)):
        a, b = ARMS[i - 1], ARMS[i]
        if a in arms and b in arms:
            report["adjacent_deltas"][f"{b}_vs_{a}"] = paired_deltas(arms[a], arms[b])

    # Primary delta: C4-4 vs C4-0
    if "C4_0" in arms and "C4_4" in arms:
        report["primary_delta"] = paired_deltas(arms["C4_0"], arms["C4_4"])

    # Per-regime breakdown
    for arm_id in sorted(arms):
        report["per_regime"][arm_id] = per_regime_breakdown(arms[arm_id])

    # Identity stats
    for arm_id in sorted(arms):
        report["identity_stats"][arm_id] = identity_stats(arms[arm_id])

    # Selector stats
    for arm_id in sorted(arms):
        report["selector_stats"][arm_id] = selector_stats(arms[arm_id])

    # Iterative retrieval stats
    for arm_id in sorted(arms):
        report["iterative_stats"][arm_id] = iterative_stats(arms[arm_id])

    # CSR stats
    for arm_id in sorted(arms):
        report["csr_stats"][arm_id] = csr_stats(arms[arm_id])

    # Role retention
    for arm_id in sorted(arms):
        report["role_retention"][arm_id] = role_retention_stats(arms[arm_id])

    # Gap capture metrics
    report["selector_gap_capture"] = selector_gap_capture(arms)
    report["oracle_gap_capture"] = oracle_gap_capture(arms)

    # Print summary
    print("=" * 70)
    print("GATE C4 ANALYSIS")
    print("=" * 70)
    print(f"\nSource: {args.dir}")
    print(f"Arms: {list(arms.keys())}")
    print(f"Parity: {report['parity']}")

    print("\n--- Arm Quality ---")
    print(f"{'Arm':<8} {'Q':>8} {'Correct':>10} {'N':>6}")
    for arm_id in sorted(arms):
        s = report["arm_summary"][arm_id]
        print(f"{arm_id:<8} {s['quality']:>8.4f} {s['correct_rate']:>10.4f} {s['n']:>6}")

    print("\n--- Adjacent Deltas ---")
    for key, d in report["adjacent_deltas"].items():
        f = d["flips"]
        print(f"{key}: mean={d['mean_delta']:+.4f} "
              f"CI_family=[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}] "
              f"CI_cluster=[{d.get('ci_lower_cluster', float('nan')):+.4f}, {d.get('ci_upper_cluster', float('nan')):+.4f}] "
              f"CI_template=[{d.get('ci_lower_template', float('nan')):+.4f}, {d.get('ci_upper_template', float('nan')):+.4f}] "
              f"improve={f['improve']} regress={f['regress']} unchanged={f['unchanged']}")

    if report["primary_delta"]:
        d = report["primary_delta"]
        f = d["flips"]
        print(f"\n--- Primary Delta (C4-4 vs C4-0) ---")
        print(f"mean={d['mean_delta']:+.4f} "
              f"CI_family=[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}] "
              f"CI_cluster=[{d.get('ci_lower_cluster', float('nan')):+.4f}, {d.get('ci_upper_cluster', float('nan')):+.4f}] "
              f"CI_template=[{d.get('ci_lower_template', float('nan')):+.4f}, {d.get('ci_upper_template', float('nan')):+.4f}] "
              f"improve={f['improve']} regress={f['regress']} unchanged={f['unchanged']}")
        print(f"Protocol threshold: +0.15 → {'PASS' if d['mean_delta'] >= 0.15 else 'FAIL'}")
        if d.get("flip_details", {}).get("regress"):
            print(f"  Regressed tasks: {d['flip_details']['regress'][:10]}")

    print("\n--- Per-Regime Quality ---")
    for arm_id in sorted(arms):
        for regime, rs in report["per_regime"][arm_id].items():
            print(f"  {arm_id} {regime}: Q={rs['quality']:.4f} correct={rs['correct_rate']:.4f} n={rs['n']}")

    print("\n--- Identity Stats ---")
    for arm_id in sorted(arms):
        s = report["identity_stats"][arm_id]
        print(f"  {arm_id}: EXACT={s['EXACT']} RESOLVED={s['RESOLVED']} "
              f"AMBIGUOUS={s['AMBIGUOUS']} UNRESOLVED={s['UNRESOLVED']}")

    print("\n--- Selector Stats ---")
    for arm_id in sorted(arms):
        s = report["selector_stats"][arm_id]
        print(f"  {arm_id}: s2c={s['s2c']} s0={s['s0']} srel={s['srel']} "
              f"oracle={s['oracle']} oracle_ev={s['oracle_evidence']} "
              f"s2c_live={s['s2c_live_rate']:.4f}")

    print("\n--- Iterative Retrieval Stats ---")
    for arm_id in sorted(arms):
        s = report["iterative_stats"][arm_id]
        print(f"  {arm_id}: second_pass={s['second_pass_performed']}/{s['n']} "
              f"rate={s['second_pass_rate']:.4f}")

    print("\n--- CSR ---")
    for arm_id in sorted(arms):
        s = report["csr_stats"][arm_id]
        print(f"  {arm_id}: mean_csr={s['mean_csr']:.4f}")

    print("\n--- Role Retention ---")
    for arm_id in sorted(arms):
        s = report["role_retention"][arm_id]
        print(f"  {arm_id}: answer={s.get('answer_retention', 0):.4f} "
              f"bridge={s.get('bridge_retention', 0):.4f} "
              f"identity={s.get('identity_retention', 0):.4f}")

    print("\n--- Gap Capture ---")
    sgc = report["selector_gap_capture"]
    ogc = report["oracle_gap_capture"]
    print(f"  Selector Gap Capture (SGC): {sgc:.4f}" if sgc is not None else "  SGC: N/A")
    print(f"  Oracle Gap Capture (OGC): {ogc:.4f}" if ogc is not None else "  OGC: N/A")

    # Write output
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.output}")
    else:
        print("\n--- Full Report (JSON) ---")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
