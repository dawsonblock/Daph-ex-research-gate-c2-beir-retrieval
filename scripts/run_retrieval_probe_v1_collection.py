#!/usr/bin/env python3
"""RETRIEVAL_PROBE_GATE_V1 PHASE_1: development collection.

Per configs/gate_retrieval_probe_v1_design.json. Runs against the CONSUMED
exec_training_v2 suite, which permits diagnosis_only / hypothesis_generation
_only -- controller design is hypothesis generation. NO PROMOTION CLAIM.

This is a MEASUREMENT REFACTOR, not a behaviour change. It executes the same
two logical actions as scripts/run_exec_training_v2_collection.py --
A0_ANSWER_NOW and A1_USE_CERTIFIED_MEMORY -- over the same suite with the
same certified pipeline, and adds:

  * the full per-stage latency decomposition (CUDA-synchronized), and
  * the cheap retrieval probe's features, captured BETWEEN A0 and the
    escalation-only work.

A1 is produced by escalating from the probe, so the probe's retrieval is
REUSED rather than recomputed; probe_handoff_hash is recorded on both the
probe and the escalation receipt so reuse is verifiable after the fact.
tests/unit/test_retrieval_probe.py proves probe->escalate reproduces the
pre-refactor packet and prompt exactly.

Policy-relevant costs (C_accept / C_escalate / C_avoided) and the true
no-probe deployment baselines are DERIVED from the atomic stages by
StageTimer.derived_costs(), never timed separately, so the decomposition
cannot disagree with its own totals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaker)

from hrm_adaptive_memory.c4.arms import ARMS  # noqa: E402
from hrm_adaptive_memory.c4.contracts import C4_PRIMARY_PACKET_BUDGET  # noqa: E402
from hrm_adaptive_memory.c4.packet_composition import generation_hash  # noqa: E402
from hrm_adaptive_memory.executive.confidence import generate_with_confidence  # noqa: E402
from hrm_adaptive_memory.executive.retrieval_probe import (  # noqa: E402
    PROBE_FEATURE_NAMES, retrieval_probe_features, run_full_memory_from_probe,
    run_retrieval_probe)
from hrm_adaptive_memory.executive.stage_timing import StageTimer  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.certified_memory import (  # noqa: E402
    CertifiedMemoryDriftError, assert_certified_memory_v1_unchanged,
    pin_certified_memory_v1_boundary_policy)
from hrm_adaptive_memory.experiment_integrity.execution_identity import ExecutionIdentity  # noqa: E402
from hrm_adaptive_memory.experiment_integrity.executive_features import (  # noqa: E402
    KNOWN_FEATURES, require_admissible_for_answer_vs_memory)
from scripts.run_exec_training_v2_collection import MEMORY_REQUIRED_SCALES, load_groups  # noqa: E402
from scripts.run_gate_c4 import (  # noqa: E402
    HRM_MAX_NEW_TOKENS, _assert_prompt_binding, _load_hrm, _run_hrm_batch,
    _to_index_records as to_index_records)

M = 50
PACKET = C4_PRIMARY_PACKET_BUDGET
PIPELINE_VERSION = "retrieval_probe_v1_collection"
SUITE_ROOT = ROOT / "data/hrm/exec_training_v2"
FAMILIES = ("ANSWER_NOW_viable", "MEMORY_required")
SCHEMA_VERSION = "retrieval-probe-v1-collection"


def c2(n: int) -> int:
    return max(1, min(300, math.ceil(0.15 * n))) if n else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="RETRIEVAL_PROBE_GATE_V1 PHASE_1 collection")
    ap.add_argument("--arm-for-queries", default="C4_4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--limit-tasks", type=int, default=None)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    pin_certified_memory_v1_boundary_policy()
    try:
        certified_identity = assert_certified_memory_v1_unchanged()
    except CertifiedMemoryDriftError as e:
        print(f"ABORT: CERTIFIED_MEMORY_V1 drift detected before any task ran: {e}")
        return 1
    extractor_hash = certified_identity.graph_compressor_config_hash

    # Fail closed on the feature boundary BEFORE any work happens.
    for name in PROBE_FEATURE_NAMES:
        if name not in KNOWN_FEATURES:
            print(f"ABORT: probe feature {name!r} is not declared in KNOWN_FEATURES")
            return 1
        require_admissible_for_answer_vs_memory(KNOWN_FEATURES[name])

    arm = ARMS[args.arm_for_queries]
    families = args.families or list(FAMILIES)

    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_commit = "non-git-tree"

    print(f"=== RETRIEVAL_PROBE_GATE_V1 PHASE_1 ({'DRY RUN' if args.dry_run else 'EXECUTE'}) ===")
    print("    DEVELOPMENT COLLECTION on a CONSUMED split -- no promotion claim.")
    print(f"  CERTIFIED_MEMORY_V1 identity OK  extractor_hash={extractor_hash}  M={M}  packet={PACKET}")
    print(f"  probe features admissible: {', '.join(PROBE_FEATURE_NAMES)}\n")

    adapter, condition = (None, None)
    if args.execute:
        adapter, condition = _load_hrm()

    receipts: list[dict[str, Any]] = []

    for family in families:
        for group_label, tasks, evidence, texts in load_groups(family):
            if args.limit_tasks:
                tasks = tasks[:args.limit_tasks]
            records = to_index_records(evidence)
            depth = c2(len(records))
            prefix = f"{group_label}: " if family == "MEMORY_required" else ""

            for i, task in enumerate(tasks, 1):
                if i % 10 == 0 or i == len(tasks):
                    print(f"  {prefix}{i}/{len(tasks)}", end="\r", flush=True)
                q = task["question"]
                fam = task.get("family", family)
                required = set(task.get("required_evidence_ids", []))
                rid = f"{group_label}:{task['task_id']}" if family == "MEMORY_required" else task["task_id"]
                timer = StageTimer()

                # --- A0: cheap answer probe, now timed -------------------
                a0_prompt = (f"[OBJECTIVE]\n{q}\n[EVIDENCE]\n[NO EXTERNAL EVIDENCE]\n"
                             "[RESPONSE REQUIREMENT]\nAnswer directly, concisely.")
                a0_receipt: dict[str, Any] = {
                    "task_id": rid, "action": "A0_ANSWER_NOW", "family": fam,
                    "suite_family": family, "scale": group_label,
                    "prompt_hash": hashlib.sha256(a0_prompt.encode()).hexdigest(),
                }
                if args.execute:
                    with timer.stage("T_A0_generation"):
                        conf = generate_with_confidence(
                            adapter, condition, a0_prompt, max_new_tokens=HRM_MAX_NEW_TOKENS)
                    q0 = task.get("answer", "").strip().lower() in conf.text.strip().lower()
                    a0_receipt.update({
                        "output": conf.text, "correct": q0,
                        "prompt_tokens": conf.prompt_tokens,
                        "completion_tokens": conf.completion_tokens,
                        "mean_token_confidence": conf.mean_token_confidence,
                        "min_token_confidence": conf.min_token_confidence,
                        "sequence_confidence": conf.sequence_confidence,
                        "mean_entropy": conf.mean_entropy,
                        "answer_length": conf.answer_length,
                        "T_A0_generation": timer.stages.get("T_A0_generation"),
                        "A0_cuda_sync_used": timer.cuda_synchronized,
                    })
                else:
                    a0_receipt["output"] = None
                receipts.append(a0_receipt)

                # --- cheap retrieval probe (between A0 and escalation) ---
                probe = run_retrieval_probe(q, arm, records, texts, depth, timer=timer)
                pfeat = retrieval_probe_features(probe)
                probe_receipt: dict[str, Any] = {
                    "task_id": rid, "action": "PROBE_RETRIEVAL", "family": fam,
                    "suite_family": family, "scale": group_label,
                    "probe_handoff_hash": probe.handoff_hash(),
                    "identity_status": probe.identity_status,
                    "T_probe_retrieval": timer.stages.get("T_probe_retrieval"),
                    "T_probe_identity_binding": timer.stages.get("T_probe_identity_binding"),
                    "T_probe_total": (timer.stages.get("T_probe_retrieval", 0.0)
                                      + timer.stages.get("T_probe_identity_binding", 0.0)),
                    **pfeat,
                }
                receipts.append(probe_receipt)

                # --- A1: escalation-only work, consuming the probe -------
                mem = run_full_memory_from_probe(probe, arm, texts, M, PACKET,
                                                 selector_label="retrieval_probe_v1",
                                                 timer=timer)
                a1_identity = ExecutionIdentity(
                    task_id=rid, arm_id="A1_USE_CERTIFIED_MEMORY",
                    prompt_hash=mem.packet.prompt_hash, retrieval_config_hash="C2",
                    selector_config_hash="s2_v2+s4_composer_v1",
                    graph_compressor_config_hash=extractor_hash,
                    model_revision="sapientinc/HRM-Text-1B@9f082d68",
                    pipeline_version=PIPELINE_VERSION, source_commit=source_commit,
                    extra_config_hashes={
                        "graph_hash": mem.graph_hash, "path_set_hash": mem.path_set_hash,
                        "composed_packet_hash": mem.composed_packet_hash})
                a1_receipt: dict[str, Any] = {
                    "task_id": rid, "action": "A1_USE_CERTIFIED_MEMORY", "family": fam,
                    "suite_family": family, "scale": group_label,
                    "packet_ids": list(mem.packet.packet_ids),
                    "candidate_pool_hash": mem.packet.candidate_pool_hash,
                    "graph_hash": mem.graph_hash, "path_set_hash": mem.path_set_hash,
                    "composed_packet_hash": mem.composed_packet_hash,
                    "membership_hash": mem.packet.membership_hash,
                    "order_hash": mem.packet.order_hash,
                    "prompt_hash": mem.packet.prompt_hash,
                    "execution_identity_sha256": a1_identity.canonical_sha256(),
                    "required_in_packet": bool(required) and required <= set(mem.packet.packet_ids),
                    "identity_status": mem.identity_status,
                    "probe_handoff_hash": mem.consumed_handoff_hash,
                    "probe_reused_on_escalation": mem.consumed_handoff_hash == probe.handoff_hash(),
                    "T_G2": timer.stages.get("T_G2"),
                    "T_composition": timer.stages.get("T_composition"),
                }
                if args.execute:
                    with timer.stage("T_A1_generation"):
                        hrm_result = _run_hrm_batch(adapter, condition, [mem.prompt])[0]

                    class _Pre:
                        pass
                    pre = _Pre(); pre.task_id = rid
                    pre.arm_id = "A1_USE_CERTIFIED_MEMORY"; pre.packet = mem.packet
                    _assert_prompt_binding(pre, hrm_result)
                    q1 = task.get("answer", "").strip().lower() in hrm_result.output.strip().lower()
                    a1_receipt.update({
                        "output": hrm_result.output, "correct": q1,
                        "generation_hash": generation_hash(hrm_result.output),
                        "prompt_tokens": hrm_result.prompt_tokens,
                        "completion_tokens": hrm_result.completion_tokens,
                        "T_A1_generation": timer.stages.get("T_A1_generation"),
                        "latency_seconds": hrm_result.latency_seconds,
                    })
                else:
                    a1_receipt["output"] = None
                a1_receipt.update(timer.derived_costs())
                a1_receipt["T_total"] = float(sum(timer.stages.values()))
                a1_receipt["cuda_synchronized"] = timer.cuda_synchronized
                receipts.append(a1_receipt)

            print(" " * 44, end="\r")
            print(f"  {prefix}done  tasks={len(tasks)}  evidence_pool={len(evidence)}")

    out = Path(args.out) if args.out else (
        ROOT / f"evidence/gate_executive/retrieval_probe_v1_"
               f"{'dry_run' if args.dry_run else 'collection'}.receipts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in receipts) + "\n")

    reused = [r for r in receipts if r["action"] == "A1_USE_CERTIFIED_MEMORY"]
    all_reused = all(r.get("probe_reused_on_escalation") for r in reused)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run" if args.dry_run else "execute",
        "design": "configs/gate_retrieval_probe_v1_design.json",
        "DEVELOPMENT_ONLY_no_promotion_claim": True,
        "certified_memory_v1_identity_hash": certified_identity.canonical_sha256(),
        "source_commit": source_commit, "families_run": families,
        "receipts_written": len(receipts),
        "tasks": len(reused),
        "probe_reused_on_every_escalation": all_reused,
        "probe_features": list(PROBE_FEATURE_NAMES),
        "measured_stages": list(StageTimer.REQUIRED_STAGES),
        "derived_costs": ["T_probe_total", "C_accept", "C_escalate", "C_avoided",
                          "C_noprobe_answer_only", "C_noprobe_full_memory"],
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n  written: {out.with_suffix('.json')}\n  receipts: {out}")
    print(f"  probe reused on every escalation: {all_reused}")
    if not all_reused:
        print("  ABORT-WORTHY: probe reuse invariant violated on at least one task.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
