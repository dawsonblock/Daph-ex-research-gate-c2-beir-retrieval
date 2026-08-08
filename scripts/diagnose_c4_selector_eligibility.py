#!/usr/bin/env python3
"""Sprint B2 pre-freeze feasibility check for the selector eligibility rule.

Run BEFORE any selector arm exists. Its only job is to answer whether the
proposed direct-answer eligibility rule can actually fire on the population
that carries the measured defect -- because a rule that cannot is an
underpowered arm, and Sprint B1's R4 already demonstrated what that costs.

Two things are measured, both on development only:

  1. Where the defect lives. Keep-rate of AVAILABLE required evidence, cross
     tabulated by identity status and task shape (bridged vs unbridged).
     "Available" means the required set was in the candidate pool, so a loss is
     the selector's and not retrieval's.

  2. Whether the eligibility rule fires there. The originally specified rule is
     canonical_subject_match AND target_relation_match. A bridged task's
     terminal-answer record is anchored on the BRIDGE entity rather than the
     query subject, so the conjunction is structurally unsatisfiable for those
     tasks -- this quantifies that.

Runtime-signal discipline: eligibility is evaluated using only record content,
the target relation PARSED FROM THE QUESTION, and the canonical subject from
the I3 identity stage. The oracle metadata is read here solely to LABEL task
shape and locate the terminal record for measurement -- never as an input to
the rule. record_kind is deliberately not used at all: its values in this
corpus ('required', 'dead_end_link', 'rejected_candidate', ...) are generator
answer-key labels, so a runtime selector reading them would be reading the
answer key.

Usage:
    python scripts/diagnose_c4_selector_eligibility.py [--split development]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.query_stage import (  # noqa: E402
    extract_target_relation)
from hrm_adaptive_memory.retrieval.canonicalization import _norm  # noqa: E402
from scripts.run_gate_c4 import (  # noqa: E402
    _load_split as load_split, _to_index_records as to_index_records,
    run_pre_hrm_stages)


def task_shape(task: dict) -> str:
    """bridged if the task's proof path goes through a latent bridge."""
    return "bridged" if (task.get("_oracle_metadata") or {}).get("latent_bridge") \
        else "unbridged"


def terminal_records(task: dict) -> list[str]:
    """Records whose proof edge terminates at the answer node.

    Oracle metadata, used for MEASUREMENT only -- never as a rule input.
    """
    oracle = task.get("_oracle_metadata") or {}
    answer_node = oracle.get("answer_node")
    return [edge["record_id"] for edge in (oracle.get("proof_edges") or [])
            if edge.get("target") == answer_node]


def rule_fires(task: dict, canonical: str | None,
               texts: dict[str, str]) -> bool:
    """The ORIGINALLY SPECIFIED rule: subject match AND relation match.

    Uses only question-parsed relation and the canonical subject, matched
    against record content. Returns whether an eligible record exists among the
    task's own terminal-answer records.
    """
    relation = extract_target_relation(task["question"]) or ""
    if not relation or not canonical:
        return False
    for record_id in terminal_records(task):
        content = _norm(texts.get(record_id, ""))
        if _norm(canonical) in content and _norm(relation) in content:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B2 pre-freeze eligibility feasibility check")
    parser.add_argument("--split", default="development")
    parser.add_argument("--arm", default="C4_4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks, evidence, texts = load_split(args.split)
    records = to_index_records(evidence)
    arm = ARMS[args.arm]

    cross: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"kept": 0, "dropped": 0})
    defect_shape: Counter = Counter()
    fires_by_shape: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "fires": 0})
    unavailable = 0

    for index, task in enumerate(tasks, 1):
        if index % 25 == 0 or index == len(tasks):
            print(f"  {index}/{len(tasks)}...", end="\r", flush=True)
        result = run_pre_hrm_stages(task, arm, records, texts)
        required = set(task["required_evidence_ids"])
        shape = task_shape(task)

        # Rule fire-rate is a property of the task, independent of outcome.
        stats = fires_by_shape[shape]
        stats["n"] += 1
        stats["fires"] += rule_fires(task, result.identity.canonical, texts)

        if not required <= set(result.retrieval.candidate_ids):
            unavailable += 1
            continue  # retrieval's loss, not the selector's

        dropped = not required <= set(result.selection.selected_ids)
        cross[(result.identity.status, shape)][
            "dropped" if dropped else "kept"] += 1
        if dropped and result.identity.status == "EXACT":
            defect_shape[shape] += 1

    print(" " * 30, end="\r")

    def rate(entry: dict[str, int]) -> float | None:
        total = entry["kept"] + entry["dropped"]
        return round(entry["kept"] / total, 4) if total else None

    defect_total = sum(defect_shape.values())
    report: dict[str, Any] = {
        "schema_version": "c4-selector-eligibility-feasibility-v1",
        "split": args.split,
        "task_count": len(tasks),
        "tasks_with_unavailable_evidence_excluded": unavailable,
        "pre_freeze": (
            "Run before any selector arm exists, to decide whether the "
            "proposed eligibility rule can fire where the defect lives."),
        "keep_rate_by_identity_and_shape": {
            f"{status}_{shape}": {**entry, "keep_rate": rate(entry)}
            for (status, shape), entry in sorted(cross.items())},
        "exact_defect_population_by_shape": dict(defect_shape),
        "exact_defect_share_bridged": (
            round(defect_shape.get("bridged", 0) / defect_total, 4)
            if defect_total else None),
        "original_rule_fire_rate_by_shape": {
            shape: {**stats,
                    "fire_rate": round(stats["fires"] / stats["n"], 4)
                    if stats["n"] else None}
            for shape, stats in sorted(fires_by_shape.items())},
        "conclusion": (
            "If the defect population is predominantly bridged while the rule "
            "fires only on unbridged tasks, the rule as specified cannot "
            "address the defect and must be corrected BEFORE freezing -- not "
            "after seeing arm results."),
    }

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_c4/diagnosis/{args.split}_selector_eligibility.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"=== B2 pre-freeze eligibility check: {args.split} ===")
    print(f"  tasks={len(tasks)}  excluded (evidence unavailable)={unavailable}\n")
    print(f"  {'identity x shape':<26}{'kept':>6}{'dropped':>9}{'keep rate':>11}")
    for key, entry in report["keep_rate_by_identity_and_shape"].items():
        kr = entry["keep_rate"]
        print(f"  {key:<26}{entry['kept']:>6}{entry['dropped']:>9}"
              f"{(f'{kr:.1%}' if kr is not None else '-'):>11}")

    print(f"\n  EXACT-identity defect population by shape: "
          f"{report['exact_defect_population_by_shape']}")
    if report["exact_defect_share_bridged"] is not None:
        print(f"    share bridged: {report['exact_defect_share_bridged']:.1%}")

    print("\n  originally specified rule fire rate:")
    for shape, stats in report["original_rule_fire_rate_by_shape"].items():
        fr = stats["fire_rate"]
        print(f"    {shape:<12}{stats['fires']:>4}/{stats['n']:<4} = "
              f"{(f'{fr:.1%}' if fr is not None else '-')}")

    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
