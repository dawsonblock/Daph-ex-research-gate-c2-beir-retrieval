#!/usr/bin/env python3
"""R14-B1: Broad OptiLLM screening against R13-A checkpoints.

Executes admissible OptiLLM profiles + DAPH native controls on a subset
of R13-A checkpoints. Single seed. Uses wall_ms as primary cost axis
because token accounting for multi-call strategies is FINAL_RESPONSE_ONLY.

Usage:
    python scripts/run_r14_b1.py --checkpoints 30 --seed 42

Output:
    experiments/daph_x/r14/r14_b1_executions.jsonl
    experiments/daph_x/r14/r14_b1_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from daph_x.backends.openai_compat import OpenAICompatibleBackend
from daph_x.coding.reasoning_tasks import check_answer, get_all_reasoning_tasks
from daph_x.evaluation.answer_extractor import extract_answer
from daph_x.operators.external.optillm import PROFILES as OPT_PROFILES, OptiLLMOperator
from daph_x.operators.types import EvaluationLabels, RuntimeState, Candidate, TrajectoryPoint

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

R13_CHECKPOINTS = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
R14_OUTPUT_DIR = PROJECT_ROOT / "experiments/daph_x/r14"

# OptiLLM profiles admissible with llama-server (no multi_sample)
ADMISSIBLE_OPTILLM_PROFILES = [
    "OPT_COT_REFLECT",
    "OPT_RE2",
    "OPT_PLANSEARCH_LOW",
    "OPT_SC_LOW",
]

# DAPH native controls (STOP = use current answer, no external call)
NATIVE_CONTROLS = ["STOP"]


def load_labels() -> dict[str, EvaluationLabels]:
    labels = {}
    for t in get_all_reasoning_tasks():
        labels[t.task_id] = EvaluationLabels(
            task_id=t.task_id,
            correct_answer=t.answer,
            answer_type=t.answer_type,
        )
    return labels


def load_checkpoints(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def checkpoint_to_runtime_state(cp: dict) -> RuntimeState:
    rs = cp["runtime_state"]
    candidates = tuple(
        Candidate(
            candidate_id=c["candidate_id"],
            answer=c["answer"],
            reasoning_trace=c.get("reasoning_trace", ""),
            temperature=c.get("temperature", 0.0),
            seed=c.get("seed", 42),
            generation_index=c.get("generation_index", 0),
            metadata=c.get("metadata", {}),
        )
        for c in rs.get("candidates", [])
    )
    trajectory = tuple(
        TrajectoryPoint(
            k=t["k"],
            top_answer=t["top_answer"],
            p_top1=t["p_top1"],
            p_top2=t["p_top2"],
            margin=t["margin"],
            entropy=t["entropy"],
            n_unique=t["n_unique"],
        )
        for t in rs.get("trajectory", [])
    )
    return RuntimeState(
        checkpoint_id=rs["checkpoint_id"],
        task_id=rs["task_id"],
        task_prompt=rs["task_prompt"],
        answer_type=rs.get("answer_type", "default"),
        category=rs.get("category", ""),
        difficulty=rs.get("difficulty", ""),
        candidates=candidates,
        trajectory=trajectory,
        k=rs.get("k", 1),
        current_answer=rs.get("current_answer", ""),
        observable_features=rs.get("observable_features", {}),
        state_hash=rs.get("state_hash", ""),
    )


def execute_stop(state: RuntimeState) -> dict:
    """STOP control: use current answer, zero cost."""
    return {
        "operator_id": "STOP",
        "operator_version": "2",
        "checkpoint_id": state.checkpoint_id,
        "task_id": state.task_id,
        "terminal_answer": state.current_answer,
        "reasoning_trace": "",
        "status": "SUCCESS",
        "cost": {"gateway_calls": 0, "underlying_model_calls": 0, "wall_ms": 0.0, "total_tokens": 0},
        "provenance": {"operator": "STOP", "source": "current_state"},
    }


def execute_optillm(op: OptiLLMOperator, state: RuntimeState, seed: int) -> dict:
    """Execute an OptiLLM operator and return a result record."""
    t0 = time.monotonic()
    result = op.execute(state, replicate_id=seed)
    wall_ms = (time.monotonic() - t0) * 1000

    return {
        "operator_id": result.provenance.get("optillm_slug", op.spec.operator_id),
        "operator_id_canonical": op.spec.operator_id,
        "checkpoint_id": state.checkpoint_id,
        "task_id": state.task_id,
        "terminal_answer": result.terminal_answer,
        "reasoning_trace": result.reasoning_artifacts.get("raw_text", "")[:500],
        "status": result.status,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "cost": result.cost.to_dict(),
        "wall_ms_observed": wall_ms,
        "provenance": result.provenance,
    }


def evaluate_result(record: dict, labels: dict[str, EvaluationLabels]) -> dict:
    """Add correctness evaluation to a result record."""
    task_id = record["task_id"]
    lbl = labels.get(task_id)
    if lbl is None:
        record["correct"] = None
        record["correct_answer"] = None
        return record

    record["correct_answer"] = lbl.correct_answer
    record["answer_type"] = lbl.answer_type
    record["correct"] = check_answer(record["terminal_answer"], lbl.correct_answer, lbl.answer_type)
    return record


def run_screening(
    checkpoints: list[dict],
    labels: dict[str, EvaluationLabels],
    optillm_url: str,
    model: str,
    seed: int,
) -> list[dict]:
    """Run all operators on all checkpoints."""
    # Build OptiLLM backend and operators
    backend = OpenAICompatibleBackend(
        base_url=optillm_url,
        model=model,
        api_key="no_key",
        provider_name="optillm",
    )

    operators: list[tuple[str, OptiLLMOperator | None]] = []
    # Native controls
    for ctrl in NATIVE_CONTROLS:
        operators.append((ctrl, None))
    # OptiLLM profiles
    for pid in ADMISSIBLE_OPTILLM_PROFILES:
        op = OptiLLMOperator(OPT_PROFILES[pid], backend)
        operators.append((pid, op))

    results = []
    total = len(checkpoints) * len(operators)
    done = 0

    for cp in checkpoints:
        state = checkpoint_to_runtime_state(cp)

        for op_id, op in operators:
            done += 1
            print(f"  [{done}/{total}] {cp['checkpoint_id']} × {op_id} ... ", end="", flush=True)

            if op_id == "STOP":
                record = execute_stop(state)
            elif op is not None:
                record = execute_optillm(op, state, seed)
            else:
                continue

            record = evaluate_result(record, labels)
            results.append(record)

            status = record["status"]
            correct = record.get("correct")
            wall = record.get("wall_ms_observed", record.get("cost", {}).get("wall_ms", 0))
            print(f"{status} correct={correct} wall={wall:.0f}ms")

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute per-operator summary statistics."""
    from collections import defaultdict

    by_op = defaultdict(list)
    for r in results:
        by_op[r["operator_id_canonical"] if "operator_id_canonical" in r else r["operator_id"]].append(r)

    summary = {}
    for op_id, records in by_op.items():
        n = len(records)
        n_success = sum(1 for r in records if r["status"] == "SUCCESS")
        n_correct = sum(1 for r in records if r.get("correct") is True)
        n_evaluable = sum(1 for r in records if r.get("correct") is not None)
        wall_times = [r.get("wall_ms_observed", 0) for r in records if r["status"] == "SUCCESS"]
        tokens = [r.get("cost", {}).get("total_tokens") for r in records
                  if r["status"] == "SUCCESS" and r.get("cost", {}).get("total_tokens")]

        summary[op_id] = {
            "n": n,
            "n_success": n_success,
            "n_correct": n_correct,
            "n_evaluable": n_evaluable,
            "accuracy": n_correct / n_evaluable if n_evaluable > 0 else None,
            "success_rate": n_success / n if n > 0 else None,
            "wall_ms_mean": sum(wall_times) / len(wall_times) if wall_times else None,
            "wall_ms_median": sorted(wall_times)[len(wall_times)//2] if wall_times else None,
            "total_tokens_mean": sum(tokens) / len(tokens) if tokens else None,
            "total_tokens_reported": len(tokens),
            "token_accounting_note": "FINAL_RESPONSE_ONLY for multi-call strategies" if op_id != "STOP" else None,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="R14-B1: Broad OptiLLM screening")
    parser.add_argument("--checkpoints", type=int, default=30,
                        help="Number of checkpoints to screen (default 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optillm-url", default=os.environ.get("DAPH_OPTILLM_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("DAPH_R14_MODEL", "qwen"))
    parser.add_argument("--output-dir", default=str(R14_OUTPUT_DIR))
    args = parser.parse_args()

    # Load data
    print("Loading checkpoints and labels...")
    all_checkpoints = load_checkpoints(R13_CHECKPOINTS)
    labels = load_labels()
    print(f"  {len(all_checkpoints)} checkpoints, {len(labels)} labels")

    # Select subset (evenly spaced)
    n = min(args.checkpoints, len(all_checkpoints))
    step = len(all_checkpoints) / n
    selected = [all_checkpoints[int(i * step)] for i in range(n)]
    print(f"  Selected {len(selected)} checkpoints")

    # Run screening
    print(f"\nRunning R14-B1 screening: {len(selected)} checkpoints × {len(NATIVE_CONTROLS) + len(ADMISSIBLE_OPTILLM_PROFILES)} operators, seed={args.seed}")
    print(f"  OptiLLM URL: {args.optillm_url}")
    print(f"  Model: {args.model}")
    print(f"  Admissible OptiLLM: {ADMISSIBLE_OPTILLM_PROFILES}")
    print()

    results = run_screening(selected, labels, args.optillm_url, args.model, args.seed)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exec_path = output_dir / "r14_b1_executions.jsonl"
    with open(exec_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nExecutions saved to {exec_path}")

    # Compute and save summary
    summary = compute_summary(results)
    summary_path = output_dir / "r14_b1_summary.json"
    summary_meta = {
        "experiment": "R14-B1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_checkpoints": len(selected),
        "n_operators": len(NATIVE_CONTROLS) + len(ADMISSIBLE_OPTILLM_PROFILES),
        "seed": args.seed,
        "optillm_url": args.optillm_url,
        "model": args.model,
        "token_accounting_note": "usage.total_tokens is FINAL_RESPONSE_ONLY for multi-call strategies. Use wall_ms for Pareto analysis.",
        "operators": summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_meta, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'Operator':<25} {'N':>4} {'Acc':>6} {'Wall(ms)':>10} {'Tokens':>8}")
    print("-" * 60)
    for op_id, stats in sorted(summary.items()):
        acc = f"{stats['accuracy']:.3f}" if stats["accuracy"] is not None else "  N/A"
        wall = f"{stats['wall_ms_mean']:.0f}" if stats["wall_ms_mean"] is not None else "  N/A"
        tok = f"{stats['total_tokens_mean']:.0f}" if stats["total_tokens_mean"] is not None else "  N/A"
        print(f"{op_id:<25} {stats['n']:>4} {acc:>6} {wall:>10} {tok:>8}")


if __name__ == "__main__":
    main()
