#!/usr/bin/env python3
"""Build Fresh Confirmation #2 for the grammar_v4 -> G2 -> path-coherent
composition -> HRM stack. Fail-closed, additive, generated/frozen before any
scoring -- per the user's ten-step confirmation sequence.

Why this cannot reuse build_c5_confirmation_split.py or its output
--------------------------------------------------------------------
confirmation_1 (data/hrm/controlled_gate_a_v4/confirmation/, CONSUMED,
CONFIRMATION_1_NEGATIVE_RESULT.md) tested a DIFFERENT mechanism -- frozen_rrf
+ S2 with no runtime graph, no path composer -- and is a different dataset
LINEAGE entirely (controlled_gate_a_v4, seed 9104) from the one this whole
G2/G2-v2/G2-v3/S3/S4/HRM-qualification sequence was built and calibrated on
(b3_calibration_v1, seeds 9201-9205). A split from the wrong lineage, or one
that overlaps cal_700..cal_3000, would not be a confirmation of THIS stack.

This script therefore reuses build_b3_calibration_suite.py's machinery
directly (build_v4_corpus, audit_scale, composition_by_family,
structural_profile, take_filler, build_filler_pool -- imported, not
reimplemented) with two differences:

  1. fresh seeds (9301-9305 for tasks, 9298 for filler) -- outside BOTH the old
     C4/C5 seed range (9101-9104) and the b3 calibration range (9201-9205,
     9299), so there is no possibility of PRNG-state or vocabulary-cycle
     collision, not merely no observed collision;
  2. the consumed-content screen is EXTENDED to also cover every task and
     evidence record already written to cal_700/1000/1500/2200/3000, since
     those are consumed now (they trained every architecture decision from
     B3 through HRM qualification), in addition to the original
     development/qualification/ood/confirmation splits that
     build_b3_calibration_suite.py already screens against.

The main() control loop is necessarily re-stated (build_b3_calibration_suite's
main() is monolithic, not factored into a reusable per-scale function), but
every substantive helper it calls is imported, so a change to audit logic,
composition matching or filler screening in one place applies to both scripts
automatically rather than silently drifting apart.

Written to a NEW directory (data/hrm/g2_confirmation_2/), never touching
b3_calibration_v1/ or its CALIBRATION.sha256 -- additive by construction, the
same invariant build_c5_confirmation_split.py asserts for its own split.

Usage:
    python scripts/build_g2_confirmation2_suite.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

import scripts.build_b3_calibration_suite as b3  # noqa: E402
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (  # noqa: E402
    build_v4_corpus)
from scripts.build_gate_a_v4_dataset import ID_REGIMES, ID_STYLES  # noqa: E402

OUT = ROOT / "data/hrm/g2_confirmation_2"
B3_ROOT = ROOT / "data/hrm/b3_calibration_v1"
B3_SCALES = ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000")

#: Outside BOTH the old C4/C5 range (9101-9104) and the b3 calibration range
#: (9201-9205, filler 9299) by construction, not by observed non-collision.
SCALES: tuple[tuple[str, int, int], ...] = (
    ("confirm2_700", 700, 9301),
    ("confirm2_1000", 1000, 9302),
    ("confirm2_1500", 1500, 9303),
    ("confirm2_2200", 2200, 9304),
    ("confirm2_3000", 3000, 9305),
)
FILLER_SEED_2 = 9298


def load_b3_consumed() -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """The b3_calibration_v1 scales' own questions/contents/pairs, in the SAME
    normalized form build_b3_calibration_suite.load_consumed() uses, so the two
    ledgers merge without a format mismatch."""
    questions: set[str] = set()
    contents: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for scale in B3_SCALES:
        base = B3_ROOT / scale
        for line in (base / "oracle_tasks.jsonl").read_text().splitlines():
            if line.strip():
                task = json.loads(line)
                questions.add(b3._norm(task["question"]))
                pairs.add((b3._norm(task["question"]), b3._norm(task["answer"])))
        for line in (base / "evidence.jsonl").read_text().splitlines():
            if line.strip():
                contents.add(b3._norm(json.loads(line)["content"]))
    return questions, contents, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Fresh Confirmation #2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if OUT.exists():
        print(f"ABORT: refusing to overwrite a frozen suite: {OUT}")
        return 1
    if not B3_ROOT.exists():
        print(f"ABORT: b3_calibration_v1 not found at {B3_ROOT}; cannot screen against it")
        return 1

    print("=== Fresh Confirmation #2 (grammar_v4 -> G2 -> path-composer -> HRM) ===")
    print(f"  scales = {[s[0] for s in SCALES]}  (fresh seeds, outside every prior range)")
    print(f"  fixed tasks/family = {b3.TASKS_PER_FAMILY} at every scale\n")

    print("[1/6] loading OLD C4/C5 consumed splits (development/qualification/ood/confirmation)")
    old_consumed = b3.load_consumed()
    print(f"      {len(old_consumed[0])} questions, {len(old_consumed[1])} evidence contents")

    print("[2/6] loading b3_calibration_v1 (cal_700..cal_3000) -- consumed by this whole sprint")
    b3_consumed = load_b3_consumed()
    print(f"      {len(b3_consumed[0])} questions, {len(b3_consumed[1])} evidence contents")

    consumed = (old_consumed[0] | b3_consumed[0],
               old_consumed[1] | b3_consumed[1],
               old_consumed[2] | b3_consumed[2])
    print(f"      MERGED ledger: {len(consumed[0])} questions, {len(consumed[1])} contents\n")

    print("[3/6] building a FRESH filler pool (seed distinct from the original 9299)")
    b3.FILLER_SEED = FILLER_SEED_2  # module-global; build_filler_pool reads it at call time
    filler = b3.build_filler_pool(consumed[1])
    print(f"      {len(filler)} screened non-required records available\n")

    print("[4/6] generating scales")
    built: dict[str, tuple[list[dict], list[dict]]] = {}
    manifests: dict[str, dict] = {}
    used_filler_ids: set[str] = set()
    reference_composition: dict[str, dict[tuple, int]] | None = None
    seen_questions: set[str] = set(consumed[0])
    seen_contents: set[str] = set(consumed[1])
    for split_id, target, seed in SCALES:
        corpus = build_v4_corpus(split=split_id, seed=seed,
                                 tasks_per_family=b3.OVERGENERATE_PER_FAMILY,
                                 styles=ID_STYLES, regimes=ID_REGIMES)
        raw_tasks, raw_evidence = list(corpus.tasks), list(corpus.evidence)
        evidence_by_task: dict[str, list[dict]] = {}
        for record in raw_evidence:
            evidence_by_task.setdefault(
                record["evidence_id"].split("/", 1)[0], []).append(record)
        survivors: dict[str, list[dict]] = {}
        local_questions: set[str] = set()
        local_pairs: set[tuple[str, str]] = set()
        for task in raw_tasks:
            question = b3._norm(task["question"])
            pair = (question, b3._norm(task["answer"]))
            if question in seen_questions or question in local_questions:
                continue
            if pair in local_pairs:
                continue
            own = evidence_by_task.get(task["task_id"], [])
            if any(b3._norm(r["content"]) in seen_contents for r in own):
                continue
            if len({b3._norm(r["content"]) for r in own}) != len(own):
                continue
            local_questions.add(question)
            local_pairs.add(pair)
            survivors.setdefault(task["family"], []).append(task)
        short = {f: len(v) for f, v in survivors.items() if len(v) < b3.TASKS_PER_FAMILY}
        if len(survivors) < 10 or short:
            print(f"      ABORT: {split_id} has families short of "
                  f"{b3.TASKS_PER_FAMILY} after collision filtering: {short}")
            return 1
        if reference_composition is None:
            tasks = [t for family in sorted(survivors)
                     for t in survivors[family][:b3.TASKS_PER_FAMILY]]
        else:
            tasks = []
            for family in sorted(survivors):
                wanted = Counter(reference_composition.get(family, {}))
                pool_by_key: dict[tuple, list[dict]] = {}
                for task in survivors[family]:
                    pool_by_key.setdefault(b3._composition_key(task), []).append(task)
                chosen: list[dict] = []
                for key, count in sorted(wanted.items()):
                    chosen.extend(pool_by_key.get(key, [])[:count])
                if len(chosen) < b3.TASKS_PER_FAMILY:
                    already = {t["task_id"] for t in chosen}
                    for task in survivors[family]:
                        if len(chosen) >= b3.TASKS_PER_FAMILY:
                            break
                        if task["task_id"] not in already:
                            chosen.append(task)
                tasks.extend(chosen[:b3.TASKS_PER_FAMILY])
        kept_ids = {t["task_id"] for t in tasks}
        base_evidence = [r for r in raw_evidence
                         if r["evidence_id"].split("/", 1)[0] in kept_ids]
        for task in tasks:
            seen_questions.add(b3._norm(task["question"]))
        for record in base_evidence:
            seen_contents.add(b3._norm(record["content"]))
        need = target - len(base_evidence)
        if reference_composition is None:
            reference_composition = b3.composition_by_family(tasks)
        if need < 0:
            print(f"      ABORT: {split_id} base corpus {len(base_evidence)} "
                  f"already exceeds target {target}")
            return 1
        answers = {t["answer"] for t in tasks}
        base_contents = {b3._norm(r["content"]) for r in base_evidence}
        slice_ = b3.take_filler(filler, need, base_contents, answers,
                                used_filler_ids, seen_contents)
        if len(slice_) < need:
            print(f"      ABORT: filler pool exhausted for {split_id} "
                  f"(needed {need}, had {len(slice_)})")
            return 1
        evidence = base_evidence + slice_
        built[split_id] = (tasks, evidence)
        print(f"      {split_id}: {len(tasks)} tasks, {len(base_evidence)} base "
              f"+ {len(slice_)} filler = {len(evidence)} records (target {target})")

    print("\n[5/6] auditing (fail-closed; nothing written unless all pass, "
          "including cross-overlap with b3_calibration_v1)")
    reference = b3.structural_profile(built[SCALES[0][0]][0])
    all_problems: list[str] = []
    audits: dict[str, dict] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        others = {k: v for k, v in built.items() if k != split_id}
        results, problems = b3.audit_scale(
            split_id, tasks, evidence, target, consumed, others,
            None if split_id == SCALES[0][0] else reference)
        audits[split_id] = results
        all_problems += problems
        print(f"      {split_id}: {'PASS' if not problems else 'FAIL'}")
    for line in all_problems:
        print(f"      FAIL {line}")
    if all_problems:
        print("\nABORT: audits failed. Nothing written.")
        return 1

    # NOTE: raw task_id/evidence_id strings are NOT globally unique across
    # independent corpus builds -- generalization_dataset_v4.build_v4_corpus
    # assigns task_id = f"{family}-{ordinal:04d}", scoped only to a single
    # split's own files (confirmed: even the already-frozen cal_700 and
    # cal_1000 scales collide on 114/150 raw task_id strings with EACH OTHER).
    # No consumer merges ids across splits (run_hrm_qualification.py processes
    # `for scale in scales` one split at a time, tagging receipts with `scale`
    # alongside `task_id`), so an id-string overlap check here would be a false
    # invariant. The real leakage guard is content-level, already enforced
    # above via the merged consumed-content ledger and audit_scale's cross-set
    # comparison.
    print("      ALL AUDITS PASS")

    if args.dry_run:
        print("\n[dry run] audits pass; nothing written by request.")
        return 0

    print("\n[6/6] writing and freezing")
    OUT.mkdir(parents=True)
    digests: dict[str, str] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        directory = OUT / split_id
        directory.mkdir()
        task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks)
        evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence)
        (directory / "oracle_tasks.jsonl").write_text(task_text)
        (directory / "evidence.jsonl").write_text(evidence_text)

        profile = audits[split_id]["structural_profile"]
        manifest = {
            "split_id": split_id, "generation_seed": seed,
            "task_count": len(tasks), "evidence_count": len(evidence),
            "tasks_per_family": b3.TASKS_PER_FAMILY,
            "family_counts": profile["family_counts"],
            "entity_regime_counts": profile["entity_regime_counts"],
            "bridged_counts": profile["bridged_counts"],
            "answer_kind_counts": profile["answer_kind_counts"],
            "proof_depth_counts": {str(k): v for k, v in profile["proof_depth_counts"].items()},
            "required_set_size_counts": {str(k): v for k, v in profile["required_set_size_counts"].items()},
            "source_cluster_counts": profile["source_cluster_counts"],
            "task_sha256": b3.sha256_text(task_text),
            "evidence_sha256": b3.sha256_text(evidence_text),
            "generator_revision": "controlled-gate-a-v4",
            "generator_config_hash": b3.sha256_text(json.dumps(
                {"tasks_per_family": b3.TASKS_PER_FAMILY, "seed": seed,
                 "styles": list(ID_STYLES), "regimes": list(ID_REGIMES),
                 "filler_seed": FILLER_SEED_2}, sort_keys=True))[:16],
            "target_evidence_count": target,
            "scale_conformance": audits[split_id]["scale_conformance"],
            "budget_metadata": b3.BUDGET_METADATA,
        }
        manifests[split_id] = manifest
        (directory / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (directory / "CALIBRATION_AUDIT.json").write_text(json.dumps({
            "split_id": split_id, "frozen_before_evaluation": True,
            "audit": audits[split_id],
        }, indent=2, sort_keys=True) + "\n")
        for path in sorted(directory.iterdir()):
            digests[f"{split_id}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    (OUT / "CONFIRMATION_2.sha256").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "g2_confirmation_2",
        "purpose": ("Fresh Confirmation #2 for the grammar_v4 -> G2 -> "
                   "path-coherent composition -> HRM stack qualified in "
                   "evidence/gate_hrm/qualification_execute.json. Never used "
                   "for architecture decisions, mechanism selection or "
                   "threshold selection -- generated after every such decision "
                   "was already frozen."),
        "distinct_lineage_from": "data/hrm/controlled_gate_a_v4/confirmation (confirmation_1, consumed, different mechanism)",
        "screened_against": ["data/hrm/controlled_gate_a_v4/{development,qualification,ood,confirmation}",
                             "data/hrm/b3_calibration_v1/{cal_700,cal_1000,cal_1500,cal_2200,cal_3000}"],
        "fixed_tasks_per_family": b3.TASKS_PER_FAMILY,
        "scales": {s[0]: {"target": s[1], "seed": s[2]} for s in SCALES},
        "filler_seed": FILLER_SEED_2,
        "manifests": manifests,
        "frozen_before_evaluation": True,
        "run_discipline": ("No G2, S2, S4 composer, or HRM execution has "
                           "touched this data. Permitted use: CONFIRMATION only, "
                           "per hrm_adaptive_memory.experiment_integrity.split_lineage."),
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into CONFIRMATION_2.sha256")
    print("\nFRESH CONFIRMATION #2 FROZEN. No evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
