#!/usr/bin/env python3
"""Failure decomposition for a C4 result bundle. Diagnostic only, no GPU.

Motivated by the qualification result (VALID_RUN=false, primary delta +0.091
vs the pre-registered +0.15): the mechanism kept positive signal but only
~46% of its development gain generalized. Deciding what to fix requires
knowing WHERE each task died, and "quality went down" does not say that.

Every task in the primary arm lands in exactly one bucket of a mutually
exclusive, exhaustive chain -- the pipeline's own stages, in order:

    A_RETRIEVAL   required evidence never reached the candidate pool.
                  No selector can recover this. Retrieval/representation work.
    B_SELECTOR    evidence WAS in the pool and the selector dropped it.
                  Pure selector work; the ceiling is already in hand.
    C_READER      evidence survived into the packet, HRM still answered wrong.
    SUCCESS       evidence survived and the answer was correct.

Two cross-cutting characterizations are reported on top of that chain,
because they name *why* B happens rather than *that* it happens:

    D_EXACT_OVER_EXPANSION  B-bucket tasks whose identity was already EXACT.
                  Hypothesis this measures: when the subject needs no
                  resolution, structural expansion is over-aggressive and
                  displaces the direct answer-bearing record. Reported with
                  the fusion rank of the dropped evidence, since evidence the
                  retriever ranked FIRST being dropped is the strongest form
                  of the claim.
    E_TEMPORAL    per-family view for temporal_update / temporal_chain, which
                  sit at opposite extremes (huge selector headroom vs almost
                  no evidence availability at all).

Nothing here decides a gate or writes into a certified bundle. It reads
frozen receipts and writes one diagnostic JSON. It is deliberately usable on
any split, but note the discipline it exists to serve: a fix motivated by a
qualification diagnosis must still be developed and certified on development
data and confirmed on a fresh untouched split. Sizing a bet against these
receipts is legitimate; tuning against them is not.

The output deliberately does NOT default inside the bundle. BUNDLE.sha256 is
a recursive hash over the entire bundle directory including certification/,
so dropping a new file in there permanently breaks verification of an
already-certified result -- caught exactly that way on first run against the
qualification bundle. Diagnoses are derived artifacts about a bundle, not
part of the measurement it certified, so they live beside it.

Usage:
    python scripts/diagnose_c4_generalization.py \
        [--bundle evidence/gate_c4/full/qualification] \
        [--baseline-arm C4_0] [--primary-arm C4_4] [--oracle-arm C4_5] \
        [--out <path>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = ("A_RETRIEVAL", "B_SELECTOR", "C_READER", "SUCCESS")


def default_out_path(bundle: Path) -> Path:
    """A sibling of the bundle, never inside it.

    evidence/gate_c4/full/qualification
      -> evidence/gate_c4/diagnosis/qualification_generalization.json
    """
    return (bundle.parent.parent / "diagnosis"
            / f"{bundle.name}_generalization.json")


def load_arm(bundle: Path, arm: str) -> list[dict]:
    path = bundle / f"{arm}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing arm receipts: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fusion_ids(receipt: dict) -> list[str]:
    """fusion_ranked is [[id, score], ...]; tolerate a bare-id form too."""
    ranked = receipt["runtime_payload"]["retrieval"].get("fusion_ranked") or []
    return [entry[0] if isinstance(entry, (list, tuple)) else entry for entry in ranked]


def classify(receipt: dict) -> str:
    """Place one task in the first pipeline stage that lost it."""
    runtime = receipt["runtime_payload"]
    required = set(receipt["evaluator_annotation"]["required_evidence_ids"])
    in_pool = required <= set(runtime["retrieval"]["candidate_ids"])
    in_selection = required <= set(runtime["selection"]["selected_ids"])

    if not in_pool:
        return "A_RETRIEVAL"
    if not in_selection:
        return "B_SELECTOR"
    return "SUCCESS" if receipt["evaluator_annotation"]["correct"] else "C_READER"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def decompose(primary: list[dict]) -> dict[str, Any]:
    buckets: dict[str, list[dict]] = {name: [] for name in BUCKETS}
    for receipt in primary:
        buckets[classify(receipt)].append(receipt)

    n = len(primary)
    summary = {
        "task_count": n,
        "buckets": {
            name: {
                "n": len(rows),
                "share": round(len(rows) / n, 4) if n else 0.0,
                "mean_quality": round(_mean(
                    [r["evaluator_annotation"]["quality"] for r in rows]), 4),
            }
            for name, rows in buckets.items()
        },
    }

    # D: the EXACT-identity over-expansion signature, inside the selector bucket.
    dropped = buckets["B_SELECTOR"]
    by_status: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "required_was_fusion_rank_1": 0})
    rank1_examples: list[dict[str, Any]] = []
    for receipt in dropped:
        status = receipt["runtime_payload"]["identity"]["status"]
        entry = by_status[status]
        entry["n"] += 1
        ranked = _fusion_ids(receipt)
        required = set(receipt["evaluator_annotation"]["required_evidence_ids"])
        if ranked and ranked[0] in required:
            entry["required_was_fusion_rank_1"] += 1
            if len(rank1_examples) < 5:
                rank1_examples.append({
                    "task_id": receipt["task_id"],
                    "family": receipt["evaluator_annotation"]["family"],
                    "identity_status": status,
                    "required": sorted(required),
                    "selected": list(
                        receipt["runtime_payload"]["selection"]["selected_ids"]),
                })

    # Keep-rate asymmetry: of tasks whose evidence WAS available, how often did
    # the selector keep it, split by identity status. This is the comparison
    # that distinguishes "selector is weak" from "selector is weak only when
    # identity resolution was unnecessary".
    keep_rate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kept": 0, "dropped": 0})
    for receipt in primary:
        required = set(receipt["evaluator_annotation"]["required_evidence_ids"])
        runtime = receipt["runtime_payload"]
        if not required <= set(runtime["retrieval"]["candidate_ids"]):
            continue  # unavailable: not the selector's fault
        status = runtime["identity"]["status"]
        if required <= set(runtime["selection"]["selected_ids"]):
            keep_rate[status]["kept"] += 1
        else:
            keep_rate[status]["dropped"] += 1
    for stats in keep_rate.values():
        total = stats["kept"] + stats["dropped"]
        stats["available"] = total
        stats["keep_rate"] = round(stats["kept"] / total, 4) if total else None

    summary["D_exact_over_expansion"] = {
        "selector_drops_by_identity_status": dict(by_status),
        "available_evidence_keep_rate_by_identity_status": dict(keep_rate),
        "rank_1_drop_examples": rank1_examples,
        "reads": (
            "A large EXACT/RESOLVED gap in keep_rate supports routing the "
            "selector on information state rather than applying one global "
            "expansion policy. Cases where the dropped evidence was the "
            "retriever's own rank-1 candidate are the least ambiguous form of "
            "over-expansion: retrieval had already solved the task."),
    }
    return summary


def per_family(primary: list[dict], oracle: list[dict] | None) -> dict[str, Any]:
    """Where the ceiling is: availability vs what the selector kept vs oracle."""
    oracle_by_task = {r["task_id"]: r for r in (oracle or [])}
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "candidate_ces": 0, "selected_ces": 0, "oracle_ces": 0})

    for receipt in primary:
        family = receipt["evaluator_annotation"]["family"]
        required = set(receipt["evaluator_annotation"]["required_evidence_ids"])
        runtime = receipt["runtime_payload"]
        row = agg[family]
        row["n"] += 1
        row["candidate_ces"] += required <= set(runtime["retrieval"]["candidate_ids"])
        row["selected_ces"] += required <= set(runtime["selection"]["selected_ids"])
        counterpart = oracle_by_task.get(receipt["task_id"])
        if counterpart:
            row["oracle_ces"] += required <= set(
                counterpart["runtime_payload"]["selection"]["selected_ids"])

    out = {}
    for family, row in agg.items():
        n = row["n"]
        candidate = row["candidate_ces"] / n
        selected = row["selected_ces"] / n
        oracle_ces = row["oracle_ces"] / n
        out[family] = {
            "n": n,
            "candidate_ces": round(candidate, 4),
            "selected_ces": round(selected, 4),
            "oracle_selected_ces": round(oracle_ces, 4),
            # What a perfect selector could still recover from THIS pool.
            "selector_headroom": round(oracle_ces - selected, 4),
            # What no selector can recover: evidence simply is not there.
            "retrieval_deficit": round(1.0 - candidate, 4),
            "bottleneck": ("RETRIEVAL" if (1.0 - candidate) > (oracle_ces - selected)
                           else "SELECTOR"),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["candidate_ces"]))


def subgroup_deltas(baseline: list[dict], primary: list[dict],
                    key: str) -> dict[str, Any]:
    """Paired per-subgroup delta on an arbitrary grouping axis."""
    def group_of(receipt: dict) -> str:
        annotation = receipt["evaluator_annotation"]
        if key == "family":
            return annotation["family"]
        return annotation["metadata"][key]

    base_by_task = {r["task_id"]: r for r in baseline}
    grouped: dict[str, list[float]] = defaultdict(list)
    for receipt in primary:
        counterpart = base_by_task.get(receipt["task_id"])
        if not counterpart:
            continue
        grouped[group_of(receipt)].append(
            receipt["evaluator_annotation"]["quality"]
            - counterpart["evaluator_annotation"]["quality"])

    return {
        group: {"n": len(deltas), "mean_delta": round(_mean(deltas), 4)}
        for group, deltas in sorted(grouped.items(),
                                    key=lambda kv: _mean(kv[1]))
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose where a C4 bundle's tasks fail (no GPU)")
    parser.add_argument("--bundle",
                        default="evidence/gate_c4/full/qualification")
    parser.add_argument("--baseline-arm", default="C4_0")
    parser.add_argument("--primary-arm", default="C4_4")
    parser.add_argument("--oracle-arm", default="C4_5")
    parser.add_argument("--out", default=None,
                        help="defaults to a sibling of the bundle, NOT inside "
                             "it -- writing into a bundle breaks its "
                             "BUNDLE.sha256 recursive hash")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = ROOT / bundle

    baseline = load_arm(bundle, args.baseline_arm)
    primary = load_arm(bundle, args.primary_arm)
    try:
        oracle = load_arm(bundle, args.oracle_arm)
    except SystemExit:
        oracle = None

    report: dict[str, Any] = {
        "schema_version": "c4-generalization-diagnosis-v1",
        "bundle": str(bundle.relative_to(ROOT)) if bundle.is_relative_to(ROOT) else str(bundle),
        "arms": {"baseline": args.baseline_arm, "primary": args.primary_arm,
                 "oracle": args.oracle_arm if oracle else None},
        "diagnostic_only": (
            "Not a gate. Does not decide VALID_RUN and writes no certified "
            "artifact. A fix motivated by these numbers must be developed and "
            "certified on development data, then confirmed on a fresh "
            "untouched split -- sizing a bet against these receipts is "
            "legitimate, tuning against them is not."),
        "failure_decomposition": decompose(primary),
        "per_family_bottleneck": per_family(primary, oracle),
        "subgroup_delta_family": subgroup_deltas(baseline, primary, "family"),
        "subgroup_delta_entity_regime": subgroup_deltas(
            baseline, primary, "entity_regime"),
    }

    out = Path(args.out) if args.out else default_out_path(bundle)
    # Resolve before comparing: a RELATIVE --out pointing into the bundle would
    # otherwise slip past is_relative_to(), which compares path text and not
    # locations, since `bundle` has already been made absolute above.
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    bundle = bundle.resolve()
    if out.is_relative_to(bundle):
        raise SystemExit(
            f"refusing to write inside the bundle: {out}\n"
            f"BUNDLE.sha256 hashes the whole bundle directory recursively, so "
            f"adding a file there would break verification of an already "
            f"certified result. Write beside the bundle instead.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    decomposition = report["failure_decomposition"]
    print(f"=== C4 failure decomposition: {report['bundle']} ===")
    print(f"  tasks: {decomposition['task_count']}  "
          f"(arms {args.baseline_arm} -> {args.primary_arm})\n")
    for name in BUCKETS:
        stats = decomposition["buckets"][name]
        print(f"  {name:<14}{stats['n']:>5}  {stats['share']:>7.1%}  "
              f"meanQ={stats['mean_quality']:.4f}")

    print("\n  available-evidence keep rate by identity status:")
    keep = decomposition["D_exact_over_expansion"][
        "available_evidence_keep_rate_by_identity_status"]
    for status, stats in sorted(keep.items()):
        if stats["keep_rate"] is not None:
            print(f"    {status:<11}{stats['kept']:>4}/{stats['available']:<4} "
                  f"= {stats['keep_rate']:>7.1%} kept")

    print("\n  per-family bottleneck:")
    for family, row in report["per_family_bottleneck"].items():
        print(f"    {family:<22}candCES={row['candidate_ces']:.3f}  "
              f"selCES={row['selected_ces']:.3f}  "
              f"headroom={row['selector_headroom']:.3f}  -> {row['bottleneck']}")

    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
