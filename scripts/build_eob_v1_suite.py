#!/usr/bin/env python3
"""Build EOB-v1 (Executive Opportunity Benchmark v1): D0/D1/D2/D3, 100 tasks
each, per configs/gate_eob_v1_design.json. Frozen mix before any scoring.

D0/D2/D3 are freshly generated (hrm_adaptive_memory.experiments.eob_v1_dataset)
with reference-solver verification already enforced by the generator itself.
D1 is SAMPLED from the already-audited data/hrm/b3_calibration_v1/cal_700 --
that generation problem is solved and frozen, not redone here.

Fail-closed: nothing is written unless every audit check passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.experiments.eob_v1_dataset import (  # noqa: E402
    build_d0_tasks, build_d2_tasks, build_d3_tasks, reference_solve)

OUT = ROOT / "data/hrm/eob_v1"
B3_CAL_700 = ROOT / "data/hrm/b3_calibration_v1/cal_700"
D0_SEED = 9501
D2_SEED = 9502
D3_SEED = 9503
D1_SAMPLE_SEED = 9504
TASKS_PER_FAMILY = 25  # x4 families = 100 D0 tasks; D2/D3 derive 1:1 from D0
D1_SAMPLE_SIZE = 100


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def build_d1_sample() -> tuple[list[dict], list[dict]]:
    """Sample D1_SAMPLE_SIZE tasks from the already-frozen, already-audited
    cal_700, plus every evidence record any sampled task requires (not the
    whole cal_700 evidence pool -- keeps EOB-v1's own corpus bounded)."""
    import random
    tasks = [json.loads(l) for l in (B3_CAL_700 / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (B3_CAL_700 / "evidence.jsonl").read_text().splitlines() if l.strip()]
    evidence_by_id = {r["evidence_id"]: r for r in evidence}

    rng = random.Random(D1_SAMPLE_SEED)
    sampled = rng.sample(tasks, D1_SAMPLE_SIZE)

    needed_ids: set[str] = set()
    for t in sampled:
        needed_ids.update(t["required_evidence_ids"])
        # also keep every OTHER record from the same base task_id prefix
        # (dead-ends / rejected / superseded records the task's own oracle
        # metadata references) so retrieval sees a realistic pool, not just
        # the required set
        prefix = t["task_id"] + "/"
        needed_ids.update(eid for eid in evidence_by_id if eid.startswith(prefix))

    sampled_evidence = [evidence_by_id[eid] for eid in sorted(needed_ids) if eid in evidence_by_id]
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
    if not B3_CAL_700.exists():
        print(f"ABORT: {B3_CAL_700} not found -- D1 sampling requires it")
        return 1

    print("=== Building EOB-v1 (Executive Opportunity Benchmark v1) ===\n")

    print("[1/5] D0: generating direct-sufficient tasks")
    d0_tasks = build_d0_tasks(seed=D0_SEED, tasks_per_family=TASKS_PER_FAMILY)
    print(f"      {len(d0_tasks)} D0 tasks, all reference-solver verified by the generator itself")
    for t in d0_tasks:
        if reference_solve(t.question) != t.answer:
            print(f"      ABORT: {t.task_id} fails independent re-verification at build time")
            return 1
    print("      re-verified independently a second time here: OK")

    print("[2/5] D2: building confirming-evidence tasks from the same D0 set")
    d2_tasks = build_d2_tasks(d0_tasks, seed=D2_SEED)
    for t in d2_tasks:
        if t.question in t.evidence[0]["content"]:
            print(f"      ABORT: {t.task_id} confirming evidence leaks the verbatim question")
            return 1
    print(f"      {len(d2_tasks)} D2 tasks, no verbatim question leakage")

    print("[3/5] D3: building distractor-evidence tasks from the same D0 set")
    try:
        d3_tasks = build_d3_tasks(d0_tasks, seed=D3_SEED)
    except ValueError as e:
        print(f"      ABORT: {e}")
        return 1
    print(f"      {len(d3_tasks)} D3 tasks, distractor values verified != correct answers")

    print("[4/5] D1: sampling from the already-audited b3_calibration_v1/cal_700")
    d1_tasks, d1_evidence = build_d1_sample()
    print(f"      {len(d1_tasks)} D1 tasks sampled, {len(d1_evidence)} evidence records pulled in")

    # cross-regime zero-overlap check: D0/D2/D3 question text vs D1's own
    # question text (different generators entirely, but verify rather than
    # assume) and vs every consumed split
    print("[5/5] auditing zero overlap against consumed splits")
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

    d0_questions = {_norm(t.question) for t in d0_tasks}
    overlap = d0_questions & consumed_q
    if overlap:
        print(f"      ABORT: {len(overlap)} D0 questions collide with a consumed split "
              "(extremely unlikely given disjoint templates -- investigate before proceeding)")
        return 1
    print(f"      D0/D2/D3 questions ({len(d0_questions)} distinct): zero overlap with consumed splits")
    print("      (D1 is a SAMPLE of an already-consumed/calibration split by design -- not novel data)")

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
                         else "sampled from data/hrm/b3_calibration_v1/cal_700"),
            "seed": {"D0_direct_sufficient": D0_SEED, "D1_memory_required": D1_SAMPLE_SEED,
                    "D2_both_sufficient": D2_SEED, "D3_memory_distractor": D3_SEED}[regime],
        }
        (directory / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for path in sorted(directory.iterdir()):
            digests[f"{regime}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    (OUT / "EOB_V1.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "eob_v1",
        "design": "configs/gate_eob_v1_design.json",
        "regimes": ["D0_direct_sufficient", "D1_memory_required", "D2_both_sufficient", "D3_memory_distractor"],
        "tasks_per_regime": {"D0_direct_sufficient": len(d0_tasks), "D1_memory_required": len(d1_tasks),
                             "D2_both_sufficient": len(d2_tasks), "D3_memory_distractor": len(d3_tasks)},
        "total_tasks": len(d0_tasks) + len(d1_tasks) + len(d2_tasks) + len(d3_tasks),
        "frozen_before_evaluation": True,
        "purpose": "Executive Opportunity Study v2 -- a benchmark with controlled ANSWER_NOW-vs-MEMORY heterogeneity, purpose-built after v1 (b3_calibration_v1 alone) found ExecutiveOpportunity=0.0 by construction.",
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into EOB_V1.sha256")
    print("\nEOB-V1 FROZEN. No evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
