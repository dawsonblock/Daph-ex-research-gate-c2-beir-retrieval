#!/usr/bin/env python3
"""Sprint B3-B: the A_COMPLETE stop gate. Analysis only -- no mechanism work.

WRITTEN BEFORE THE CONFIRMATION ARTIFACTS EXISTED. The decision rule below is
frozen in code, so the outcome classification cannot be shaped by the numbers
it is applied to. That provenance is the point: the confirmation verdict was
already fixed by the selection-only dry pass before any Q was observed, and
this gate is decided the same way.

The question
------------
Confirmation #1 failed. Two things happened together at confirmation scale:

    candidate CES collapsed          75.8% -> 41.2%
    S2 bridge retention REVERSED     +4.4pp on development -> -4.4pp here

Those are consistent with two very different causes, and the next engineering
effort belongs in different places depending on which it is:

    1. the useful evidence never reaches the candidate pool  -> RETRIEVAL
    2. the evidence IS in the pool and S2 drops it            -> SELECTOR

Conditioning on availability separates them. Two complementary decompositions
are computed, because either alone can mislead:

    availability          P(role in candidate pool)
    conditional retention P(role selected | role available)

A role can look fine on availability while the selector quietly discards it,
and a role can look badly retained simply because it was rarely there.

Strata
------
    A_COMPLETE     every required record is in the candidate pool
    A_PARTIAL      some but not all
    A_NO_BRIDGE    a required bridge record is absent
    A_NO_TERMINAL  terminal answer evidence is absent

A_PARTIAL overlaps A_NO_BRIDGE / A_NO_TERMINAL by construction: the latter two
name WHICH role is missing, and are reported as diagnostic views rather than a
partition. Only A_COMPLETE / A_PARTIAL partition the tasks.

Decision rule (FROZEN, applied to the aggregate conditional effect)
------------------------------------------------------------------
    OUTCOME_A_POSITIVE  J1 materially beats J0 in A_COMPLETE and structural
                        retention is not worse -> preserve S2, B3 justified
    OUTCOME_B_NEUTRAL   approximately equal -> retrieval first, but S2 is not
                        settled; a selector-robustness sprint is required
    OUTCOME_C_HARMFUL   J1 < J0 in A_COMPLETE, or bridge/terminal retention
                        materially degrades despite full availability -> STOP,
                        do not build B3 around this S2

"Materially" reuses the protocol's existing -0.05 subgroup tolerance as a
descriptive boundary rather than inventing a new bar after the fact. The point
estimate and grouped CI are always reported alongside, so the classification
never rests on the threshold alone.

Usage:
    python scripts/diagnose_c5_confirmation_stopgate.py
        [--receipts evidence/gate_c4/diagnosis/confirmation_c5_Jladder_hrm.receipts.jsonl]
        [--split confirmation]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_c4_selector_eligibility import (  # noqa: E402
    task_shape, terminal_records)
from scripts.run_c5_integrated_ladder import (  # noqa: E402
    evaluate_task, use_ladder)
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records, ARMS)

#: Reused from the protocol's subgroup tolerance. NOT a new bar invented here.
MATERIAL = 0.05

BASELINE, PRIMARY = "J0", "J1"


def bridge_records(task: dict) -> set[str]:
    oracle = task.get("_oracle_metadata") or {}
    bridge = oracle.get("latent_bridge")
    if not bridge:
        return set()
    return {edge["record_id"] for edge in (oracle.get("proof_edges") or [])
            if edge.get("target") == bridge}


def identity_records(task: dict, terminals: set[str], bridges: set[str]) -> set[str]:
    return set(task["required_evidence_ids"]) - terminals - bridges


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def grouped_lcb(pairs: list[tuple[str, float]], iterations: int = 2000,
                seed: int = 12345) -> float | None:
    """Lower 2.5% bound on the mean delta, resampling GROUPS not tasks."""
    groups: dict[str, list[float]] = defaultdict(list)
    for key, value in pairs:
        groups[key].append(value)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [v for g in picked for v in g]
        if flat:
            means.append(sum(flat) / len(flat))
    means.sort()
    return round(means[int(0.025 * len(means))], 4) if means else None


def classify_outcome(delta_q: float | None, lcb: float | None,
                     bridge_delta: float | None,
                     terminal_delta: float | None) -> tuple[str, list[str]]:
    """The FROZEN decision rule. Written before any confirmation Q existed."""
    reasons: list[str] = []
    if delta_q is None:
        return "INSUFFICIENT_DATA", ["A_COMPLETE stratum is empty"]

    structural_harm = [
        (name, value) for name, value in
        (("bridge", bridge_delta), ("terminal", terminal_delta))
        if value is not None and value <= -MATERIAL]

    if structural_harm:
        reasons += [f"{name} conditional retention degrades {value:+.4f} "
                    f"(<= -{MATERIAL}) despite full availability"
                    for name, value in structural_harm]
        return "OUTCOME_C_HARMFUL", reasons

    if delta_q <= -MATERIAL:
        reasons.append(f"Q delta {delta_q:+.4f} <= -{MATERIAL} in A_COMPLETE")
        return "OUTCOME_C_HARMFUL", reasons

    if delta_q >= MATERIAL and (lcb is None or lcb > 0):
        reasons.append(f"Q delta {delta_q:+.4f} >= +{MATERIAL} with grouped "
                       f"LCB {lcb}")
        return "OUTCOME_A_POSITIVE", reasons

    reasons.append(f"Q delta {delta_q:+.4f} is inside +/-{MATERIAL} "
                   f"(grouped LCB {lcb}); S2 is not established as robust "
                   f"even where evidence was fully available")
    return "OUTCOME_B_NEUTRAL", reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="B3-B A_COMPLETE stop gate")
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--receipts", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    receipts_path = Path(args.receipts) if args.receipts else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_c5_Jladder_hrm"
               f".receipts.jsonl")
    if not receipts_path.is_file():
        print(f"receipts not found: {receipts_path}")
        return 1

    receipts = {}
    for line in receipts_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            receipts[row["task_id"]] = row

    tasks, evidence, texts = load_split(args.split)
    by_id = {t["task_id"]: t for t in tasks}

    # AVAILABILITY MUST COME FROM THE ACTUAL CANDIDATE POOL.
    #
    # The receipts store `selected`, not the pool, so availability cannot be
    # read from them. A first version of this script proxied it from the union
    # of selections plus the oracle arm J2 -- that was wrong: J2's
    # oracle-over-pool selection is capped at the PACKET budget (6) and so can
    # never represent a 50-candidate pool. It would have understated
    # availability and inflated apparent selector failure, biasing the stop
    # gate toward OUTCOME_C.
    #
    # The pools are instead RECOMPUTED here on CPU through the same
    # evaluate_task path the run used. That is exact rather than approximate:
    # the determinism precondition PASSED on this split (20 tasks x 3 seeds,
    # byte-identical on candidate_ids and candidate_pool_hash), so a replay
    # reproduces the same pools the scored run saw. Recomputed pool hashes are
    # cross-checked against the receipts below, fail-closed.
    use_ladder("J")
    records = to_index_records(evidence)
    query_arm = ARMS["C4_4"]
    pools: dict[str, set[str]] = {}
    pool_hash_mismatches: list[str] = []
    print(f"  recomputing candidate pools for {len(tasks)} tasks (CPU)...")
    for index, task in enumerate(tasks, 1):
        if index % 50 == 0 or index == len(tasks):
            print(f"    {index}/{len(tasks)}", end="\r", flush=True)
        row = evaluate_task(task, query_arm, records, texts, len(records))
        arm_row = row["arms"][BASELINE]
        pools[task["task_id"]] = set(arm_row["pool"])
        receipt = receipts.get(task["task_id"])
        if receipt and receipt["arms"][BASELINE]["candidate_pool_hash"] != \
                arm_row["candidate_pool_hash"]:
            pool_hash_mismatches.append(task["task_id"])
    print(" " * 30, end="\r")
    if pool_hash_mismatches:
        print(f"  ABORT: {len(pool_hash_mismatches)} recomputed pool hash(es) "
              f"do not match the receipts, e.g. {pool_hash_mismatches[:3]}. "
              f"The replay does not reproduce the scored run, so availability "
              f"cannot be trusted.")
        return 1
    print(f"  pool hashes match the receipts for all {len(pools)} tasks")

    strata: dict[str, list[str]] = defaultdict(list)
    # role -> arm -> [available, selected]
    role_stats: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    complete_pairs: list[tuple[str, float]] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for task_id, receipt in receipts.items():
        task = by_id.get(task_id)
        if task is None:
            continue
        arms = receipt["arms"]
        if BASELINE not in arms or PRIMARY not in arms:
            continue
        required = set(task["required_evidence_ids"])
        terminals = set(terminal_records(task))
        bridges = bridge_records(task)
        idents = identity_records(task, terminals, bridges)

        # J0 and J1 share the same frozen fusion, so one recomputed pool is
        # the availability set for both -- which is exactly what makes the
        # J0-vs-J1 comparison inside a stratum a selector comparison.
        available = pools[task_id]

        complete = required <= available
        stratum = "A_COMPLETE" if complete else "A_PARTIAL"
        strata[stratum].append(task_id)
        if bridges and not bridges <= available:
            strata["A_NO_BRIDGE"].append(task_id)
        if terminals and not terminals <= available:
            strata["A_NO_TERMINAL"].append(task_id)

        for role, records in (("identity", idents), ("bridge", bridges),
                              ("terminal", terminals)):
            if not records:
                continue
            role_available = records <= available
            for arm in (BASELINE, PRIMARY):
                if role_available:
                    role_stats[role][arm][0] += 1
                    if records <= set(arms[arm]["selected"]):
                        role_stats[role][arm][1] += 1

        row = {
            "task_id": task_id, "stratum": stratum,
            "family": receipt.get("family"),
            "entity_regime": receipt.get("entity_regime"),
            "shape": task_shape(task),
            "q_J0": arms[BASELINE].get("q"), "q_J1": arms[PRIMARY].get("q"),
            "complete_J0": required <= set(arms[BASELINE]["selected"]),
            "complete_J1": required <= set(arms[PRIMARY]["selected"]),
        }
        rows.append(row)
        if complete and row["q_J0"] is not None and row["q_J1"] is not None:
            complete_pairs.append(
                (receipt.get("family", "?"), row["q_J1"] - row["q_J0"]))

    for stratum in ("A_COMPLETE", "A_PARTIAL", "A_NO_BRIDGE", "A_NO_TERMINAL"):
        ids = set(strata.get(stratum, ()))
        subset = [r for r in rows if r["task_id"] in ids]
        q0 = [r["q_J0"] for r in subset if r["q_J0"] is not None]
        q1 = [r["q_J1"] for r in subset if r["q_J1"] is not None]
        per_stratum[stratum] = {
            "tasks": len(subset),
            "q_J0": _mean(q0), "q_J1": _mean(q1),
            "delta_q": (round(_mean(q1) - _mean(q0), 4)
                        if q0 and q1 else None),
            "selected_ces_J0": _rate(sum(r["complete_J0"] for r in subset), len(subset)),
            "selected_ces_J1": _rate(sum(r["complete_J1"] for r in subset), len(subset)),
        }

    conditional = {}
    for role, arms_stats in role_stats.items():
        entry = {}
        for arm in (BASELINE, PRIMARY):
            avail, sel = arms_stats[arm]
            entry[arm] = {"available": avail, "selected": sel,
                          "conditional_retention": _rate(sel, avail)}
        r0 = entry[BASELINE]["conditional_retention"]
        r1 = entry[PRIMARY]["conditional_retention"]
        entry["delta"] = (round(r1 - r0, 4)
                          if r0 is not None and r1 is not None else None)
        conditional[role] = entry

    complete = per_stratum["A_COMPLETE"]
    lcb = grouped_lcb(complete_pairs)
    outcome, reasons = classify_outcome(
        complete["delta_q"], lcb,
        conditional.get("bridge", {}).get("delta"),
        conditional.get("terminal", {}).get("delta"))

    report = {
        "schema_version": "c5-confirmation-stopgate-v1",
        "split": args.split,
        "decision_rule_frozen_before_data": True,
        "material_threshold": MATERIAL,
        "material_threshold_provenance": (
            "reused from the protocol's existing subgroup tolerance; NOT "
            "invented after seeing the result"),
        "strata": per_stratum,
        "a_complete_grouped_lcb": lcb,
        "role_conditional_retention": conditional,
        "OUTCOME": outcome,
        "reasons": reasons,
        "interpretation": {
            "OUTCOME_A_POSITIVE": "retrieval budget is the primary problem; S2 stays frozen; B3 justified",
            "OUTCOME_B_NEUTRAL": "retrieval first, but S2 is not settled; selector-robustness sprint required",
            "OUTCOME_C_HARMFUL": "STOP; do not build B3 around this S2; open selector generalization",
        }[outcome] if outcome in (
            "OUTCOME_A_POSITIVE", "OUTCOME_B_NEUTRAL", "OUTCOME_C_HARMFUL")
        else "insufficient data",
    }

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_b3/diagnosis/{args.split}_stopgate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"=== B3-B stop gate: {args.split} ===\n")
    print(f"  {'stratum':<16}{'tasks':>7}{'Q(J0)':>9}{'Q(J1)':>9}{'dQ':>9}"
          f"{'selCES J0':>11}{'selCES J1':>11}")
    for name, s in per_stratum.items():
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
        print(f"  {name:<16}{s['tasks']:>7}{fmt(s['q_J0']):>9}{fmt(s['q_J1']):>9}"
              f"{fmt(s['delta_q']):>9}{fmt(s['selected_ces_J0']):>11}"
              f"{fmt(s['selected_ces_J1']):>11}")

    print(f"\n  A_COMPLETE grouped LCB on dQ: {lcb}")
    print("\n  role-conditional retention  P(selected | available):")
    print(f"  {'role':<12}{'avail':>7}{'J0':>9}{'J1':>9}{'delta':>9}")
    for role, entry in sorted(conditional.items()):
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
        print(f"  {role:<12}{entry[BASELINE]['available']:>7}"
              f"{fmt(entry[BASELINE]['conditional_retention']):>9}"
              f"{fmt(entry[PRIMARY]['conditional_retention']):>9}"
              f"{fmt(entry['delta']):>9}")

    print(f"\n  OUTCOME: {outcome}")
    for reason in reasons:
        print(f"    - {reason}")
    print(f"  => {report['interpretation']}")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
