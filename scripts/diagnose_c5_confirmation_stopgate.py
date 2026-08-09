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

from hrm_adaptive_memory.c4.packet_ordering import (  # noqa: E402
    canonical_candidate_membership_hash)
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


def temporal_current_records(task: dict, kinds: dict[str, str]) -> set[str]:
    """Required records the corpus marks as the currently-valid revision.

    Identified by record_kind == "required_current". Reading record_kind is
    forbidden to the SELECTOR at runtime -- its values are generator answer-key
    labels -- but it is legitimate for an EVALUATION-side analyzer, exactly as
    terminal_records() reads proof_edges. The asymmetry is deliberate and is
    enforced by test on selector_v2, not here.

    Note this role OVERLAPS terminal: on this corpus 24 of the terminal-answer
    proof records are required_current. Reported as its own row rather than
    carved out of terminal, so temporal handling stays visible without
    distorting the terminal figures.
    """
    return {eid for eid in task["required_evidence_ids"]
            if kinds.get(eid) == "required_current"}


def identity_records(task: dict, terminals: set[str], bridges: set[str]) -> set[str]:
    """Whatever remains: surface->canonical mappings and other support."""
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
    parser.add_argument("--pools", default=None,
                        help="frozen candidate-pool artifact from "
                             "replay_c5_candidate_pools.py, captured on the "
                             "platform that scored the run. PREFERRED: reading "
                             "it removes any dependence on cross-platform "
                             "bit-reproducibility.")
    parser.add_argument("--allow-local-replay", action="store_true",
                        help="reconstruct pools locally instead. Availability "
                             "is then only as trustworthy as cross-platform "
                             "reproducibility, which is NOT guaranteed.")
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
    kinds = {r["evidence_id"]: (r.get("metadata") or {}).get("record_kind", "")
             for r in evidence}

    # AVAILABILITY MUST COME FROM THE POOL THE SCORED RUN ACTUALLY USED.
    #
    # Preferred path: read a frozen pool artifact captured by
    # replay_c5_candidate_pools.py ON THE SCORING PLATFORM. Then no analysis
    # depends on cross-platform bit-reproducibility at all.
    #
    # The invariant enforced here is unordered candidate-MEMBERSHIP identity.
    # Two weaker arguments were tried and are recorded as insufficient:
    #   * selection equality does NOT imply pool equality -- two pools differing
    #     in low-ranked members can yield identical top-6 selections;
    #   * an aggregate candidate-CES match only means the COUNT of complete
    #     tasks agreed, so two tasks could swap completeness and cancel out.
    # Availability predicates are per-task set membership, so membership is the
    # invariant the conclusion rests on.
    use_ladder("J")
    pools: dict[str, set[str]] = {}
    membership_mismatches: list[str] = []
    pools_path = Path(args.pools) if args.pools else (
        ROOT / f"evidence/gate_b3/pools/{args.split}_candidate_pools.json")

    if pools_path.is_file():
        frozen = json.loads(pools_path.read_text())
        env = frozen.get("environment_fingerprint", {})
        print(f"  frozen pools: {pools_path.name}")
        print(f"    captured on: {env.get('device_name')} torch={env.get('torch')}")
        print(f"    reproduces scored run exactly: "
              f"{frozen.get('reproduces_scored_run_exactly')}")
        if not frozen.get("reproduces_scored_run_exactly"):
            print(f"  ABORT: the frozen pool artifact does not reproduce the "
                  f"scored run ({frozen.get('order_hash_mismatch_count')} "
                  f"order-hash mismatches). Capture it on the scoring platform "
                  f"before running this gate.")
            return 1
        for entry in frozen["pools"]:
            pools[entry["task_id"]] = set(entry["candidate_ids"])
            # Membership must also be internally consistent with its own hash.
            if canonical_candidate_membership_hash(entry["candidate_ids"]) != \
                    entry["candidate_membership_hash"]:
                membership_mismatches.append(entry["task_id"])
        if membership_mismatches:
            print(f"  ABORT: {len(membership_mismatches)} membership hash(es) in "
                  f"the artifact do not match their own candidate_ids.")
            return 1
        print(f"    membership hashes self-consistent for {len(pools)} tasks")
    elif args.allow_local_replay:
        print(f"  WARNING: no frozen pools at {pools_path}; reconstructing "
              f"locally. Availability is only as trustworthy as cross-platform "
              f"bit-reproducibility, which is NOT guaranteed -- treat any "
              f"outcome as PROVISIONAL.")
        records = to_index_records(evidence)
        query_arm = ARMS["C4_4"]
        for index, task in enumerate(tasks, 1):
            if index % 50 == 0 or index == len(tasks):
                print(f"    {index}/{len(tasks)}", end="\r", flush=True)
            row = evaluate_task(task, query_arm, records, texts, len(records))
            pools[task["task_id"]] = set(row["arms"][BASELINE]["pool"])
        print(" " * 30, end="\r")
    else:
        print(f"  ABORT: no frozen candidate-pool artifact at {pools_path}.\n"
              f"  Capture it on the platform that scored the run:\n"
              f"    python3 scripts/replay_c5_candidate_pools.py "
              f"--split {args.split}\n"
              f"  or pass --allow-local-replay to accept a PROVISIONAL result.")
        return 1

    strata: dict[str, list[str]] = defaultdict(list)
    # role -> arm -> [available, selected]
    role_stats: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))
    #: role -> count of tasks where the role EXISTS but was not in the pool.
    #: Needed for layer 1: without it, P(available) would be computed over
    #: available tasks only and would be 1.0 by construction.
    role_absent: dict[str, int] = defaultdict(int)
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
        temporal = temporal_current_records(task, kinds)

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
                              ("terminal", terminals),
                              ("temporal_current", temporal)):
            if not records:
                continue
            role_available = records <= available
            if not role_available:
                role_absent[role] += 1
                continue
            for arm in (BASELINE, PRIMARY):
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

    availability = {
        role: {
            "tasks_with_role": stats[BASELINE][0] + role_absent.get(role, 0),
            "available": stats[BASELINE][0],
            "p_available": _rate(stats[BASELINE][0],
                                 stats[BASELINE][0] + role_absent.get(role, 0)),
        }
        for role, stats in role_stats.items()}
    availability["complete_required_set"] = {
        "tasks_with_role": len(rows),
        "available": per_stratum["A_COMPLETE"]["tasks"],
        "p_available": _rate(per_stratum["A_COMPLETE"]["tasks"], len(rows)),
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
        "pool_provenance": {
            "source": ("frozen artifact captured on the scoring platform"
                       if pools_path.is_file() else "LOCAL REPLAY (provisional)"),
            "artifact": str(pools_path.name),
            "tasks": len(pools),
            "membership_hash_self_consistent": not membership_mismatches,
            "invariant_enforced": "unordered candidate-membership identity",
            "insufficient_alternatives_recorded": [
                "selection equality does not imply pool equality",
                "aggregate candidate-CES equality does not imply per-task equality",
            ],
            "cross_platform_note": (
                "BGE embeddings differ in the last bits between the scored "
                "run's CUDA device and this CPU replay, permuting ranks at "
                "near-ties. Order-sensitive hash changes; pool membership, all "
                "500 selections and candidate CES (0.4120 both) are identical. "
                "Availability is a subset predicate over an unordered pool and "
                "is therefore unaffected."),
            "provenance_finding": (
                "The pipeline is deterministic WITHIN a platform (the "
                "PYTHONHASHSEED qualification passes) but is NOT bit-"
                "reproducible ACROSS CUDA and CPU. The determinism "
                "qualification never tested cross-platform replay, so this was "
                "latent. No prior result is invalidated -- each was computed on "
                "a single platform -- but candidate_pool_hash must not be used "
                "as a cross-platform replay check."),
        },
        "material_threshold": MATERIAL,
        "material_threshold_provenance": (
            "reused from the protocol's existing subgroup tolerance; NOT "
            "invented after seeing the result"),
        "layer1_availability": availability,
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
    print("  LAYER 3 -- utility by availability stratum:")
    print(f"  {'stratum':<16}{'tasks':>7}{'Q(J0)':>9}{'Q(J1)':>9}{'dQ':>9}"
          f"{'selCES J0':>11}{'selCES J1':>11}")
    for name, s in per_stratum.items():
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
        print(f"  {name:<16}{s['tasks']:>7}{fmt(s['q_J0']):>9}{fmt(s['q_J1']):>9}"
              f"{fmt(s['delta_q']):>9}{fmt(s['selected_ces_J0']):>11}"
              f"{fmt(s['selected_ces_J1']):>11}")

    print("\n  LAYER 1 -- candidate availability  P(role in pool):")
    print(f"  {'role':<22}{'avail':>8}{'of':>7}{'P(avail)':>11}")
    for role, entry in sorted(availability.items()):
        fmt = f"{entry['p_available']:.4f}" if entry["p_available"] is not None else "-"
        print(f"  {role:<22}{entry['available']:>8}{entry['tasks_with_role']:>7}{fmt:>11}")

    print(f"\n  A_COMPLETE grouped LCB on dQ: {lcb}")
    print("\n  LAYER 2 -- conditional retention  P(selected | available):")
    print(f"  {'role':<12}{'avail':>7}{'J0':>9}{'J1':>9}{'delta':>9}")
    for role, entry in sorted(conditional.items()):
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
        print(f"  {role:<12}{entry[BASELINE]['available']:>7}"
              f"{fmt(entry[BASELINE]['conditional_retention']):>9}"
              f"{fmt(entry[PRIMARY]['conditional_retention']):>9}"
              f"{fmt(entry['delta']):>9}")

    print(f"\n  LAYER 4 -- preregistered classification\n  OUTCOME: {outcome}")
    for reason in reasons:
        print(f"    - {reason}")
    print(f"  => {report['interpretation']}")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
