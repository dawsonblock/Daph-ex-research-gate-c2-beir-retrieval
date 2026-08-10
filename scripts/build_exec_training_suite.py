#!/usr/bin/env python3
"""Build the Executive-training split v1: 72 ANSWER_NOW-viable (hand-verified
general-knowledge facts, zero evidence, zero bypass) + 72 MEMORY_required
(sampled from b3_calibration_v1, unchanged generation/sampling approach).

Per configs/gate_exec_training_v1_design.json. Fail-closed: nothing is
written unless every check passes, including native-parser verification for
every ANSWER_NOW task and zero overlap against every consumed split
(including EOB-v1/v2, now themselves consumed).

Namespaces b3_calibration_v1 scale-sourced task_ids/evidence_ids by scale
before pooling -- the exact fix applied to build_eob_v2_suite.py after a
real RUN_VALID failure caught task_id collisions from pooling multiple
scales without namespacing.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.experiments.exec_training_dataset import (  # noqa: E402
    build_answer_now_tasks, verify_native_parsing)

OUT = ROOT / "data/hrm/exec_training_v1"
B3_SCALES = tuple((ROOT / "data/hrm/b3_calibration_v1" / s) for s in
                  ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000"))
D1_SAMPLE_SEED = 9701
D1_TARGET = 72


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def build_memory_required_sample() -> tuple[list[dict], list[dict]]:
    all_tasks: list[dict] = []
    all_evidence_by_id: dict[str, dict] = {}
    for base in B3_SCALES:
        scale_tag = base.name

        def ns(record_id: str, _tag=scale_tag) -> str:
            return f"{_tag}:{record_id}"

        for l in (base / "oracle_tasks.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            t["task_id"] = ns(t["task_id"])
            t["required_evidence_ids"] = [ns(e) for e in t["required_evidence_ids"]]
            if isinstance(t.get("oracle_evidence_ids"), list):
                t["oracle_evidence_ids"] = [ns(e) for e in t["oracle_evidence_ids"]]
            all_tasks.append(t)
        for l in (base / "evidence.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            r["evidence_id"] = ns(r["evidence_id"])
            if "source_id" in r:
                r["source_id"] = ns(r["source_id"])
            all_evidence_by_id[r["evidence_id"]] = r

    task_ids = [t["task_id"] for t in all_tasks]
    assert len(task_ids) == len(set(task_ids)), "namespacing failed to achieve global uniqueness"

    rng = random.Random(D1_SAMPLE_SEED)
    sampled = rng.sample(all_tasks, D1_TARGET)

    needed_ids: set[str] = set()
    for t in sampled:
        needed_ids.update(t["required_evidence_ids"])
        prefix = t["task_id"] + "/"
        needed_ids.update(eid for eid in all_evidence_by_id if eid.startswith(prefix))
    sampled_evidence = [all_evidence_by_id[eid] for eid in sorted(needed_ids) if eid in all_evidence_by_id]
    return sampled, sampled_evidence


def load_all_consumed_questions() -> set[str]:
    consumed: set[str] = set()
    for split in ("development", "qualification", "ood", "confirmation"):
        p = ROOT / "data/hrm/controlled_gate_a_v4" / split
        if (p / "oracle_tasks.jsonl").exists():
            for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
                if l.strip():
                    consumed.add(_norm(json.loads(l)["question"]))
    for scale in ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000"):
        p = ROOT / "data/hrm/b3_calibration_v1" / scale
        for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
            if l.strip():
                consumed.add(_norm(json.loads(l)["question"]))
    for scale_dir in (ROOT / "data/hrm/g2_confirmation_2").glob("confirm2_*"):
        for l in (scale_dir / "oracle_tasks.jsonl").read_text().splitlines():
            if l.strip():
                consumed.add(_norm(json.loads(l)["question"]))
    for eob_root, regimes in (
        (ROOT / "data/hrm/eob_v1", ("D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor")),
        (ROOT / "data/hrm/eob_v2", ("D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor")),
    ):
        if eob_root.exists():
            for regime in regimes:
                p = eob_root / regime
                if (p / "oracle_tasks.jsonl").exists():
                    for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
                        if l.strip():
                            consumed.add(_norm(json.loads(l)["question"]))
    return consumed


def main() -> int:
    if OUT.exists():
        print(f"ABORT: refusing to overwrite a frozen suite: {OUT}")
        return 1
    for base in B3_SCALES:
        if not base.exists():
            print(f"ABORT: {base} not found")
            return 1

    print("=== Building Executive-training split v1 ===\n")

    print("[1/4] ANSWER_NOW-viable: hand-verified general-knowledge facts")
    gk_tasks = build_answer_now_tasks()
    try:
        verify_native_parsing(gk_tasks)
    except Exception as e:
        print(f"      ABORT: {e}")
        return 1
    print(f"      {len(gk_tasks)} tasks, ALL verified via the real extract_subject/"
          "extract_target_relation -- zero bypass")

    print("[2/4] MEMORY_required: sampling from b3_calibration_v1 (all 5 scales, namespaced)")
    mem_tasks, mem_evidence = build_memory_required_sample()
    print(f"      {len(mem_tasks)} tasks sampled, {len(mem_evidence)} evidence records pulled in")

    print("[3/4] auditing zero overlap against every consumed split (incl. EOB-v1/v2)")
    consumed_q = load_all_consumed_questions()
    gk_questions = {_norm(t.question) for t in gk_tasks}
    overlap = gk_questions & consumed_q
    if overlap:
        print(f"      ABORT: {len(overlap)} ANSWER_NOW questions collide with a consumed split")
        return 1
    print(f"      ANSWER_NOW questions ({len(gk_questions)} distinct): zero overlap")
    print("      (MEMORY_required is sampled calibration data by design, expected reuse)")

    print("[4/4] freezing suite")
    OUT.mkdir(parents=True)
    digests: dict[str, str] = {}

    gk_task_dicts = [{
        "task_id": t.task_id, "family": "ANSWER_NOW_viable", "domain": t.domain,
        "question": t.question, "answer": t.answer,
        "required_evidence_ids": [], "metadata": {**t.metadata, "subject": t.subject},
    } for t in gk_tasks]
    mem_task_dicts = [{**t, "family": "MEMORY_required"} for t in mem_tasks]

    for family, tasks_out, evidence_out in (
        ("ANSWER_NOW_viable", gk_task_dicts, []),
        ("MEMORY_required", mem_task_dicts, mem_evidence),
    ):
        directory = OUT / family
        directory.mkdir()
        task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks_out)
        evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence_out)
        (directory / "oracle_tasks.jsonl").write_text(task_text)
        (directory / "evidence.jsonl").write_text(evidence_text)
        manifest = {
            "family": family, "task_count": len(tasks_out), "evidence_count": len(evidence_out),
            "task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
            "evidence_sha256": hashlib.sha256(evidence_text.encode()).hexdigest(),
            "generator": ("hrm_adaptive_memory.experiments.exec_training_dataset" if family == "ANSWER_NOW_viable"
                         else "sampled from data/hrm/b3_calibration_v1 (all 5 scales)"),
            "native_parser_verified": family == "ANSWER_NOW_viable",
        }
        (directory / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for path in sorted(directory.iterdir()):
            digests[f"{family}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    (OUT / "EXEC_TRAINING_V1.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "exec_training_v1",
        "design": "configs/gate_exec_training_v1_design.json",
        "families": ["ANSWER_NOW_viable", "MEMORY_required"],
        "tasks_per_family": {"ANSWER_NOW_viable": len(gk_tasks), "MEMORY_required": len(mem_tasks)},
        "total_tasks": len(gk_tasks) + len(mem_tasks),
        "frozen_before_evaluation": True,
        "purpose": "Training data for Executive v0 (binary ANSWER_NOW vs USE_CERTIFIED_MEMORY, PRE_DECISION features only). Zero privileged-parsing bypass anywhere in this split.",
        "role": "development/training -- NOT a confirmation split. A separate, later, untouched executive-confirmation split is required before promotion.",
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into EXEC_TRAINING_V1.sha256")
    print("\nEXEC-TRAINING-V1 FROZEN. No evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
