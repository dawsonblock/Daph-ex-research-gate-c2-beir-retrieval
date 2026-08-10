#!/usr/bin/env python3
"""Build the Executive-training split v2, per configs/gate_answer_probe_v2_design.json.

Two families, both freshly generated -- NEITHER resamples the consumed
exec_training_v1 (144 tasks, evidence/gate_executive/exec_training_v1_execute.
receipts.gate_result.json: NOT_PROMOTED) split:

  ANSWER_NOW_viable: 265 hand-verified GK facts across 5 NEW categories
    (atomic_number, currency, continent, state_capital, planet_order) --
    hrm_adaptive_memory.experiments.exec_training_v2_dataset. Distinct
    relations from V1's capitals/element_symbols, by construction: zero
    overlap is a design property, independently re-verified below anyway.

  MEMORY_required: a FRESH b3-style corpus, NOT a resample of the existing
    (now largely consumed) b3_calibration_v1 pool -- that pool is capped at
    750 tasks total (150/scale x 5 scales) and cannot supply 700 NEW,
    disjoint-from-every-prior-consumer tasks. This mirrors
    build_g2_confirmation2_suite.py's precedent exactly: reuses
    build_b3_calibration_suite.py's machinery (build_v4_corpus, audit_scale,
    composition_by_family, structural_profile, take_filler,
    build_filler_pool -- imported, not reimplemented) with fresh seeds
    (9401-9405 tasks, 9398 filler) outside every previously used range
    (9101-9104 old C4/C5; 9201-9205/9299 b3_calibration_v1; 9301-9305/9298
    g2_confirmation_2), and a consumed-content ledger extended to cover
    EVERY split this project has frozen to date, not just the two
    build_g2_confirmation2_suite.py screened against.

Usage:
    python scripts/build_exec_training_v2_suite.py [--dry-run]
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
from hrm_adaptive_memory.experiments.exec_training_v2_dataset import (  # noqa: E402
    build_answer_now_tasks_v2, verify_native_parsing)
from scripts.build_gate_a_v4_dataset import ID_REGIMES, ID_STYLES  # noqa: E402

OUT = ROOT / "data/hrm/exec_training_v2"

#: Outside every previously used generation-seed range in this project's
#: history (9101-9104, 9201-9205/9299, 9301-9305/9298) by construction.
SCALES: tuple[tuple[str, int, int], ...] = (
    ("exec2_700", 700, 9401),
    ("exec2_1000", 1000, 9402),
    ("exec2_1500", 1500, 9403),
    ("exec2_2200", 2200, 9404),
    ("exec2_3000", 3000, 9405),
)
FILLER_SEED_V2 = 9398

PRIOR_SPLITS_TO_SCREEN: tuple[tuple[Path, tuple[str, ...]], ...] = (
    (ROOT / "data/hrm/b3_calibration_v1",
     ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000")),
    (ROOT / "data/hrm/g2_confirmation_2",
     ("confirm2_700", "confirm2_1000", "confirm2_1500", "confirm2_2200", "confirm2_3000")),
    (ROOT / "data/hrm/eob_v1",
     ("D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor")),
    (ROOT / "data/hrm/eob_v2",
     ("D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor")),
    (ROOT / "data/hrm/exec_training_v1",
     ("ANSWER_NOW_viable", "MEMORY_required")),
)


def load_all_prior_consumed() -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Normalized questions/contents/pairs across EVERY split this project
    has frozen to date -- old C4/C5 (via b3.load_consumed()) plus every
    entry in PRIOR_SPLITS_TO_SCREEN above."""
    questions, contents, pairs = b3.load_consumed()
    for root, subdirs in PRIOR_SPLITS_TO_SCREEN:
        for name in subdirs:
            base = root / name
            tasks_p = base / "oracle_tasks.jsonl"
            if not tasks_p.is_file():
                continue
            for line in tasks_p.read_text().splitlines():
                if line.strip():
                    task = json.loads(line)
                    questions.add(b3._norm(task["question"]))
                    pairs.add((b3._norm(task["question"]), b3._norm(task["answer"])))
            ev_p = base / "evidence.jsonl"
            if ev_p.is_file():
                for line in ev_p.read_text().splitlines():
                    if line.strip():
                        contents.add(b3._norm(json.loads(line)["content"]))
    return questions, contents, pairs


def build_memory_required_v2() -> tuple[dict[str, tuple[list[dict], list[dict]]], dict[str, dict], list[str]]:
    """Generate 5 fresh scales mirroring build_g2_confirmation2_suite.py's
    exact control loop. Returns (built, audits, problems)."""
    consumed = load_all_prior_consumed()
    print(f"      merged consumed ledger: {len(consumed[0])} questions, "
          f"{len(consumed[1])} contents, {len(consumed[2])} pairs")

    b3.FILLER_SEED = FILLER_SEED_V2  # module-global; build_filler_pool reads it at call time
    filler = b3.build_filler_pool(consumed[1])
    print(f"      {len(filler)} screened filler records available\n")

    built: dict[str, tuple[list[dict], list[dict]]] = {}
    used_filler_ids: set[str] = set()
    reference_composition: dict[str, dict[tuple, int]] | None = None
    seen_questions: set[str] = set(consumed[0])
    seen_contents: set[str] = set(consumed[1])
    problems: list[str] = []

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
            problems.append(f"{split_id}: families short of {b3.TASKS_PER_FAMILY} after "
                            f"collision filtering: {short}")
            return built, {}, problems
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
            problems.append(f"{split_id}: base corpus {len(base_evidence)} exceeds target {target}")
            return built, {}, problems
        answers = {t["answer"] for t in tasks}
        base_contents = {b3._norm(r["content"]) for r in base_evidence}
        slice_ = b3.take_filler(filler, need, base_contents, answers,
                                used_filler_ids, seen_contents)
        if len(slice_) < need:
            problems.append(f"{split_id}: filler pool exhausted (needed {need}, had {len(slice_)})")
            return built, {}, problems
        evidence = base_evidence + slice_
        built[split_id] = (tasks, evidence)
        print(f"      {split_id}: {len(tasks)} tasks, {len(base_evidence)} base "
              f"+ {len(slice_)} filler = {len(evidence)} records (target {target})")

    print("\n      auditing (cross-checked against every other exec2 scale + full consumed ledger)")
    reference = b3.structural_profile(built[SCALES[0][0]][0])
    audits: dict[str, dict] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        others = {k: v for k, v in built.items() if k != split_id}
        results, scale_problems = b3.audit_scale(
            split_id, tasks, evidence, target, consumed, others,
            None if split_id == SCALES[0][0] else reference)
        audits[split_id] = results
        problems += scale_problems
        print(f"      {split_id}: {'PASS' if not scale_problems else 'FAIL'}")
    return built, audits, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Executive-training split v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if OUT.exists():
        print(f"ABORT: refusing to overwrite a frozen suite: {OUT}")
        return 1

    print("=== Building Executive-training split v2 ===\n")

    print("[1/4] ANSWER_NOW_viable: 5 new hand-verified GK categories")
    gk_tasks = build_answer_now_tasks_v2()
    try:
        verify_native_parsing(gk_tasks)
    except Exception as e:
        print(f"      ABORT: {e}")
        return 1
    print(f"      {len(gk_tasks)} tasks, ALL verified via the real extract_subject/"
          "extract_target_relation -- zero bypass\n")

    print("[2/4] MEMORY_required: 5 fresh b3-style scales (seeds 9401-9405, filler 9398)")
    built, audits, problems = build_memory_required_v2()
    if problems:
        for p in problems:
            print(f"      FAIL {p}")
        print("\nABORT: MEMORY_required construction/audit failed. Nothing written.")
        return 1
    mem_total = sum(len(t) for t, _ in built.values())
    print(f"\n      ALL SCALES PASS -- {mem_total} MEMORY_required tasks across {len(built)} scales\n")

    print("[3/4] auditing ANSWER_NOW_viable zero-overlap against the SAME full consumed ledger")
    consumed_q, _, consumed_pairs = load_all_prior_consumed()
    gk_questions = {t.question for t in gk_tasks}
    gk_norm_questions = {b3._norm(q) for q in gk_questions}
    overlap = gk_norm_questions & consumed_q
    if overlap:
        print(f"      ABORT: {len(overlap)} ANSWER_NOW_viable questions collide with a prior split")
        return 1
    dup_check = len(gk_norm_questions) == len(gk_tasks)
    if not dup_check:
        print("      ABORT: duplicate questions within the new GK table itself")
        return 1
    print(f"      ANSWER_NOW_viable questions ({len(gk_norm_questions)} distinct): zero overlap "
          "against every consumed split in this project's history\n")

    if args.dry_run:
        print("[dry run] all audits pass; nothing written by request.")
        return 0

    print("[4/4] freezing suite")
    OUT.mkdir(parents=True)
    digests: dict[str, str] = {}

    # --- ANSWER_NOW_viable -------------------------------------------------
    gk_dir = OUT / "ANSWER_NOW_viable"
    gk_dir.mkdir()
    gk_task_dicts = [{
        "task_id": t.task_id, "family": "ANSWER_NOW_viable", "domain": t.domain,
        "question": t.question, "answer": t.answer,
        "required_evidence_ids": [], "metadata": {**t.metadata, "subject": t.subject},
    } for t in gk_tasks]
    gk_task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in gk_task_dicts)
    (gk_dir / "oracle_tasks.jsonl").write_text(gk_task_text)
    (gk_dir / "evidence.jsonl").write_text("")
    gk_manifest = {
        "family": "ANSWER_NOW_viable", "task_count": len(gk_task_dicts), "evidence_count": 0,
        "task_sha256": hashlib.sha256(gk_task_text.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(b"").hexdigest(),
        "generator": "hrm_adaptive_memory.experiments.exec_training_v2_dataset",
        "native_parser_verified": True,
        "domains": sorted({t.domain for t in gk_tasks}),
    }
    (gk_dir / "dataset_manifest.json").write_text(json.dumps(gk_manifest, indent=2, sort_keys=True) + "\n")
    for path in sorted(gk_dir.iterdir()):
        digests[f"ANSWER_NOW_viable/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    # --- MEMORY_required: per-scale audit trail + flattened pooled view ----
    mem_dir = OUT / "MEMORY_required"
    mem_dir.mkdir()
    pooled_tasks: list[dict] = []
    pooled_evidence: list[dict] = []
    scale_manifests: dict[str, dict] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        scale_dir = mem_dir / split_id
        scale_dir.mkdir()
        task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks)
        evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence)
        (scale_dir / "oracle_tasks.jsonl").write_text(task_text)
        (scale_dir / "evidence.jsonl").write_text(evidence_text)
        profile = audits[split_id]["structural_profile"]
        manifest = {
            "split_id": split_id, "generation_seed": seed,
            "task_count": len(tasks), "evidence_count": len(evidence),
            "tasks_per_family": b3.TASKS_PER_FAMILY,
            "family_counts": profile["family_counts"],
            "task_sha256": b3.sha256_text(task_text),
            "evidence_sha256": b3.sha256_text(evidence_text),
            "generator_revision": "controlled-gate-a-v4",
            "target_evidence_count": target,
            "scale_conformance": audits[split_id]["scale_conformance"],
        }
        scale_manifests[split_id] = manifest
        (scale_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (scale_dir / "CALIBRATION_AUDIT.json").write_text(json.dumps({
            "split_id": split_id, "frozen_before_evaluation": True,
            "audit": audits[split_id],
        }, indent=2, sort_keys=True) + "\n")
        for path in sorted(scale_dir.iterdir()):
            digests[f"MEMORY_required/{split_id}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

        # Namespace by scale before pooling -- the exact fix applied after
        # EOB-v2's real RUN_VALID failure (raw task_id/evidence_id strings
        # are only unique within a single corpus build, confirmed even
        # cal_700/cal_1000 collide on 114/150 raw task_id strings).
        def ns(record_id: str, _tag=split_id) -> str:
            return f"{_tag}:{record_id}"
        for t in tasks:
            t2 = dict(t)
            t2["task_id"] = ns(t["task_id"])
            t2["required_evidence_ids"] = [ns(e) for e in t["required_evidence_ids"]]
            if isinstance(t2.get("oracle_evidence_ids"), list):
                t2["oracle_evidence_ids"] = [ns(e) for e in t2["oracle_evidence_ids"]]
            t2["family"] = "MEMORY_required"
            pooled_tasks.append(t2)
        for r in evidence:
            r2 = dict(r)
            r2["evidence_id"] = ns(r["evidence_id"])
            if "source_id" in r2:
                r2["source_id"] = ns(r["source_id"])
            pooled_evidence.append(r2)

    pooled_ids = [t["task_id"] for t in pooled_tasks]
    assert len(pooled_ids) == len(set(pooled_ids)), "scale-namespacing failed to achieve global uniqueness"
    pooled_task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in pooled_tasks)
    pooled_evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in pooled_evidence)
    (mem_dir / "oracle_tasks.jsonl").write_text(pooled_task_text)
    (mem_dir / "evidence.jsonl").write_text(pooled_evidence_text)
    pooled_manifest = {
        "family": "MEMORY_required", "task_count": len(pooled_tasks),
        "evidence_count": len(pooled_evidence),
        "task_sha256": hashlib.sha256(pooled_task_text.encode()).hexdigest(),
        "evidence_sha256": hashlib.sha256(pooled_evidence_text.encode()).hexdigest(),
        "generator": "5 fresh scales (exec2_700..exec2_3000), scale-namespaced and pooled",
        "scales": [s[0] for s in SCALES],
        "native_parser_verified": False,
        "note": "task_id/evidence_id are scale-namespaced (f'{scale}:{orig_id}') before pooling; verified globally unique via a hard assertion at build time.",
    }
    (mem_dir / "dataset_manifest.json").write_text(json.dumps(pooled_manifest, indent=2, sort_keys=True) + "\n")
    digests["MEMORY_required/oracle_tasks.jsonl"] = hashlib.sha256(pooled_task_text.encode()).hexdigest()
    digests["MEMORY_required/evidence.jsonl"] = hashlib.sha256(pooled_evidence_text.encode()).hexdigest()
    digests["MEMORY_required/dataset_manifest.json"] = hashlib.sha256(
        json.dumps(pooled_manifest, indent=2, sort_keys=True).encode() + b"\n").hexdigest()

    (OUT / "EXEC_TRAINING_V2.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "exec_training_v2",
        "design": "configs/gate_answer_probe_v2_design.json",
        "families": ["ANSWER_NOW_viable", "MEMORY_required"],
        "tasks_per_family": {"ANSWER_NOW_viable": len(gk_task_dicts), "MEMORY_required": len(pooled_tasks)},
        "total_tasks": len(gk_task_dicts) + len(pooled_tasks),
        "frozen_before_evaluation": True,
        "purpose": "Training data for ANSWER_PROBE_GATE_V2 -- fresh, outcome-stratified follow-up to ANSWER_PROBE_GATE_V1 (NOT_PROMOTED). Zero privileged-parsing bypass. Zero reuse of exec_training_v1's consumed 144 tasks.",
        "role": "development/training -- NOT a confirmation split. A separate, later, untouched executive-confirmation split is required before promotion.",
        "memory_required_scales": {s[0]: {"target_evidence": s[1], "generation_seed": s[2]} for s in SCALES},
        "filler_seed": FILLER_SEED_V2,
        "screened_against": [f"{root.relative_to(ROOT)}/{{{','.join(subdirs)}}}" for root, subdirs in PRIOR_SPLITS_TO_SCREEN]
                            + ["data/hrm/controlled_gate_a_v4/{development,qualification,ood,confirmation}"],
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into EXEC_TRAINING_V2.sha256")
    print(f"      TOTAL: {len(gk_task_dicts)} ANSWER_NOW_viable + {len(pooled_tasks)} MEMORY_required "
          f"= {len(gk_task_dicts) + len(pooled_tasks)} tasks")
    print("\nEXEC-TRAINING-V2 FROZEN. No evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
