#!/usr/bin/env python3
"""R14-C: Replicated qualification with 3 seeds.

90 checkpoints × {STOP, OPT_RE2, OPT_COT_REFLECT} × seeds {42, 123, 2024}
= 810 action cells

Usage:
    python scripts/run_r14_c.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from daph_x.backends.openai_compat import OpenAICompatibleBackend
from daph_x.coding.reasoning_tasks import check_answer, get_all_reasoning_tasks
from daph_x.evaluation.answer_extractor import extract_answer
from daph_x.operators.external.optillm import PROFILES as OPT_PROFILES, OptiLLMOperator
from daph_x.operators.types import EvaluationLabels, RuntimeState, Candidate, TrajectoryPoint

R13_CHECKPOINTS = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
R14_OUTPUT_DIR = PROJECT_ROOT / "experiments/daph_x/r14"

R14C_OPERATORS = ["STOP", "OPT_RE2", "OPT_COT_REFLECT"]
R14C_SEEDS = [42, 123, 2024]


def load_labels():
    labels = {}
    for t in get_all_reasoning_tasks():
        labels[t.task_id] = EvaluationLabels(
            task_id=t.task_id, correct_answer=t.answer, answer_type=t.answer_type,
        )
    return labels


def load_checkpoints(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def checkpoint_to_runtime_state(cp):
    rs = cp["runtime_state"]
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


def execute_stop(state):
    return {
        "operator_id": "STOP", "operator_id_canonical": "STOP",
        "checkpoint_id": state.checkpoint_id, "task_id": state.task_id,
        "terminal_answer": state.current_answer, "reasoning_trace": "",
        "status": "SUCCESS", "error_code": None, "error_message": None,
        "cost": {"gateway_calls": 0, "underlying_model_calls": 0, "wall_ms": 0.0, "total_tokens": 0},
        "wall_ms_observed": 0.0,
        "provenance": {"operator": "STOP", "source": "current_state"},
    }


def execute_optillm(op, state, seed):
    t0 = time.monotonic()
    result = op.execute(state, replicate_id=seed)
    wall_ms = (time.monotonic() - t0) * 1000
    return {
        "operator_id": result.provenance.get("optillm_slug", op.spec.operator_id),
        "operator_id_canonical": op.spec.operator_id,
        "checkpoint_id": state.checkpoint_id, "task_id": state.task_id,
        "terminal_answer": result.terminal_answer,
        "reasoning_trace": result.reasoning_artifacts.get("raw_text", "")[:500],
        "status": result.status, "error_code": result.error_code,
        "error_message": result.error_message,
        "cost": result.cost.to_dict(), "wall_ms_observed": wall_ms,
        "provenance": result.provenance,
    }


def evaluate_result(record, labels):
    lbl = labels.get(record["task_id"])
    if lbl is None:
        record["correct"] = None
        record["correct_answer"] = None
        return record
    record["correct_answer"] = lbl.correct_answer
    record["answer_type"] = lbl.answer_type
    record["correct"] = check_answer(record["terminal_answer"], lbl.correct_answer, lbl.answer_type)
    return record


def run_seed(checkpoints, labels, optillm_url, model, seed):
    backend = OpenAICompatibleBackend(
        base_url=optillm_url, model=model, api_key="no_key",
        provider_name="optillm", timeout_s=300.0,
    )
    operators = [("STOP", None)]
    for pid in ["OPT_RE2", "OPT_COT_REFLECT"]:
        operators.append((pid, OptiLLMOperator(OPT_PROFILES[pid], backend)))

    results = []
    total = len(checkpoints) * len(operators)
    done = 0

    for cp in checkpoints:
        state = checkpoint_to_runtime_state(cp)
        for op_id, op in operators:
            done += 1
            if op_id == "STOP":
                record = execute_stop(state)
            else:
                record = execute_optillm(op, state, seed)
            record["seed"] = seed
            record = evaluate_result(record, labels)
            results.append(record)
            status = record["status"]
            correct = record.get("correct")
            wall = record.get("wall_ms_observed", 0)
            print(f"  [{done}/{total}] seed={seed} {cp['checkpoint_id']} × {op_id} ... {status} correct={correct} wall={wall:.0f}ms")
    return results


def compute_summary(results):
    by_op_seed = defaultdict(list)
    for r in results:
        key = (r.get("operator_id_canonical", r["operator_id"]), r["seed"])
        by_op_seed[key].append(r)

    summary = {}
    for (op_id, seed), records in sorted(by_op_seed.items()):
        n = len(records)
        n_success = sum(1 for r in records if r["status"] == "SUCCESS")
        n_correct = sum(1 for r in records if r.get("correct") is True)
        n_eval = sum(1 for r in records if r.get("correct") is not None)
        walls = [r.get("wall_ms_observed", 0) for r in records if r["status"] == "SUCCESS"]
        ws = sorted(walls) if walls else []
        summary[f"{op_id}_seed{seed}"] = {
            "operator": op_id, "seed": seed,
            "n": n, "n_success": n_success, "n_correct": n_correct,
            "n_evaluable": n_eval,
            "accuracy": n_correct / n_eval if n_eval > 0 else None,
            "success_rate": n_success / n if n > 0 else None,
            "wall_ms_mean": sum(walls) / len(walls) if walls else None,
            "wall_ms_median": ws[len(ws)//2] if ws else None,
            "wall_ms_p90": ws[int(len(ws)*0.9)] if ws else None,
            "wall_ms_p95": ws[int(len(ws)*0.95)] if ws else None,
            "wall_ms_max": max(walls) if walls else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="R14-C: 3-seed replicated qualification")
    parser.add_argument("--optillm-url", default=os.environ.get("DAPH_OPTILLM_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("DAPH_R14_MODEL", "qwen"))
    args = parser.parse_args()

    print("Loading checkpoints and labels...")
    checkpoints = load_checkpoints(R13_CHECKPOINTS)
    labels = load_labels()
    print(f"  {len(checkpoints)} checkpoints, {len(labels)} labels")

    all_results = []
    for seed in R14C_SEEDS:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}: 90 checkpoints × {len(R14C_OPERATORS)} operators")
        print(f"{'='*60}")
        seed_results = run_seed(checkpoints, labels, args.optillm_url, args.model, seed)
        all_results.extend(seed_results)

    output_dir = R14_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_path = output_dir / "r14_c_executions.jsonl"
    with open(exec_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nExecutions saved to {exec_path}")

    summary = compute_summary(all_results)
    summary_path = output_dir / "r14_c_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "R14-C",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_checkpoints": 90,
            "n_operators": 3,
            "seeds": R14C_SEEDS,
            "total_cells": len(all_results),
            "operators": R14C_OPERATORS,
            "optillm_url": args.optillm_url,
            "model": args.model,
            "results": summary,
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'Operator':<25} {'Seed':>5} {'N':>4} {'Acc':>6} {'Mean(s)':>8} {'Med(s)':>8} {'P90(s)':>8} {'P95(s)':>8}")
    print("-" * 80)
    for key, stats in sorted(summary.items()):
        acc = f"{stats['accuracy']:.3f}" if stats["accuracy"] is not None else "  N/A"
        mean = f"{stats['wall_ms_mean']/1000:.1f}" if stats["wall_ms_mean"] is not None else "  N/A"
        med = f"{stats['wall_ms_median']/1000:.1f}" if stats["wall_ms_median"] is not None else "  N/A"
        p90 = f"{stats['wall_ms_p90']/1000:.1f}" if stats["wall_ms_p90"] is not None else "  N/A"
        p95 = f"{stats['wall_ms_p95']/1000:.1f}" if stats["wall_ms_p95"] is not None else "  N/A"
        print(f"{stats['operator']:<25} {stats['seed']:>5} {stats['n']:>4} {acc:>6} {mean:>8} {med:>8} {p90:>8} {p95:>8}")


if __name__ == "__main__":
    main()
