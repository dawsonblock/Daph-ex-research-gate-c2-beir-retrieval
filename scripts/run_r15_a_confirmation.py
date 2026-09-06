#!/usr/bin/env python3
"""R15-A: Run STOP + COT_REFLECT on 419 confirmation checkpoints.

Uses the frozen manifest. STOP is free (current_answer from manifest).
COT_REFLECT requires live OptiLLM service.

Outputs: experiments/daph_x/r15/r15_a_confirmation_executions.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from daph_x.backends.openai_compat import OpenAICompatibleBackend
from daph_x.coding.reasoning_tasks import check_answer, get_all_reasoning_tasks
from daph_x.operators.external.optillm import PROFILES as OPT_PROFILES, OptiLLMOperator
from daph_x.operators.types import EvaluationLabels, RuntimeState, Candidate, TrajectoryPoint

R15_DIR = PROJECT_ROOT / "experiments/daph_x/r15"
MANIFEST_PATH = R15_DIR / "r15_a_confirmation_manifest.jsonl"
OUTPUT_PATH = R15_DIR / "r15_a_confirmation_executions.jsonl"


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return [json.loads(l) for l in f]


def load_labels():
    labels = {}
    for t in get_all_reasoning_tasks():
        labels[t.task_id] = EvaluationLabels(
            task_id=t.task_id, correct_answer=t.answer, answer_type=t.answer_type,
        )
    return labels


def manifest_to_runtime_state(entry):
    rs = entry["runtime_state"]
    candidates = tuple(
        Candidate(
            candidate_id=c["candidate_id"], answer=c["answer"],
            reasoning_trace=c.get("reasoning_trace", ""),
            temperature=c.get("temperature", 0.0), seed=c.get("seed", 42),
            generation_index=c.get("generation_index", 0),
            metadata=c.get("metadata", {}),
        ) for c in rs.get("candidates", [])
    )
    trajectory = tuple(
        TrajectoryPoint(
            k=t["k"], top_answer=t["top_answer"], p_top1=t["p_top1"],
            p_top2=t["p_top2"], margin=t["margin"], entropy=t["entropy"],
            n_unique=t["n_unique"],
        ) for t in rs.get("trajectory", [])
    )
    return RuntimeState(
        checkpoint_id=rs["checkpoint_id"], task_id=rs["task_id"],
        task_prompt=rs["task_prompt"], answer_type=rs.get("answer_type", "default"),
        category=rs.get("category", ""), difficulty=rs.get("difficulty", ""),
        candidates=candidates, trajectory=trajectory, k=rs.get("k", 1),
        current_answer=rs.get("current_answer", ""),
        observable_features=rs.get("observable_features", {}),
        state_hash=rs.get("state_hash", ""),
    )


def main():
    print("R15-A: Running STOP + COT_REFLECT on 419 confirmation checkpoints")
    print("=" * 70)

    manifest = load_manifest()
    labels = load_labels()
    print(f"Manifest: {len(manifest)} checkpoints")
    print(f"Labels: {len(labels)}")

    backend = OpenAICompatibleBackend(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen",
        api_key="no_key",
        provider_name="optillm",
        timeout_s=300.0,
    )
    cot_op = OptiLLMOperator(OPT_PROFILES["OPT_COT_REFLECT"], backend)

    results = []
    total = len(manifest) * 2  # STOP + COT

    for i, entry in enumerate(manifest):
        state = manifest_to_runtime_state(entry)
        lbl = labels.get(state.task_id)

        # STOP
        stop_record = {
            "operator_id": "STOP", "operator_id_canonical": "STOP",
            "checkpoint_id": state.checkpoint_id, "task_id": state.task_id,
            "terminal_answer": state.current_answer, "reasoning_trace": "",
            "status": "SUCCESS", "error_code": None, "error_message": None,
            "cost": {"gateway_calls": 0, "underlying_model_calls": 0, "wall_ms": 0.0, "total_tokens": 0},
            "wall_ms_observed": 0.0,
            "provenance": {"operator": "STOP", "source": "current_state"},
        }
        if lbl:
            stop_record["correct_answer"] = lbl.correct_answer
            stop_record["answer_type"] = lbl.answer_type
            stop_record["correct"] = check_answer(stop_record["terminal_answer"], lbl.correct_answer, lbl.answer_type)
        else:
            stop_record["correct"] = None
        results.append(stop_record)
        print(f"  [{i*2+1}/{total}] {state.checkpoint_id} × STOP ... {stop_record['status']} correct={stop_record['correct']}")

        # COT_REFLECT
        t0 = time.monotonic()
        cot_result = cot_op.execute(state, replicate_id=42)
        wall_ms = (time.monotonic() - t0) * 1000
        cot_record = {
            "operator_id": cot_result.provenance.get("optillm_slug", "cot_reflection"),
            "operator_id_canonical": "OPT_COT_REFLECT",
            "checkpoint_id": state.checkpoint_id, "task_id": state.task_id,
            "terminal_answer": cot_result.terminal_answer,
            "reasoning_trace": cot_result.reasoning_artifacts.get("raw_text", "")[:500],
            "status": cot_result.status, "error_code": cot_result.error_code,
            "error_message": cot_result.error_message,
            "cost": cot_result.cost.to_dict(), "wall_ms_observed": wall_ms,
            "provenance": cot_result.provenance,
        }
        if lbl:
            cot_record["correct_answer"] = lbl.correct_answer
            cot_record["answer_type"] = lbl.answer_type
            cot_record["correct"] = check_answer(cot_record["terminal_answer"], lbl.correct_answer, lbl.answer_type)
        else:
            cot_record["correct"] = None
        results.append(cot_record)
        print(f"  [{i*2+2}/{total}] {state.checkpoint_id} × COT ... {cot_record['status']} correct={cot_record['correct']} wall={wall_ms:.0f}ms")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} executions to {OUTPUT_PATH}")

    # Quick summary
    from collections import defaultdict
    by_op = defaultdict(list)
    for r in results:
        by_op[r.get("operator_id_canonical", r["operator_id"])].append(r)
    for op_id, recs in sorted(by_op.items()):
        n = len(recs)
        n_correct = sum(1 for r in recs if r.get("correct") is True)
        n_eval = sum(1 for r in recs if r.get("correct") is not None)
        walls = [r.get("wall_ms_observed", 0) for r in recs if r["status"] == "SUCCESS"]
        acc = n_correct / n_eval if n_eval > 0 else None
        mean_wall = sum(walls) / len(walls) if walls else 0
        print(f"  {op_id}: n={n}, acc={acc:.4f}, mean_wall={mean_wall:.0f}ms" if acc else f"  {op_id}: n={n}")


if __name__ == "__main__":
    main()
