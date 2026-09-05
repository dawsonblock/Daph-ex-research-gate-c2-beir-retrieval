#!/usr/bin/env python3
"""Analyze R13-A v2 tournament: join execution receipts with oracle labels.

Runs outside the operator path. Evaluation happens after execution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import get_reasoning_task
from daph_x.evaluation.answer_judge import is_correct
from daph_x.operators.types import EvaluationLabels


def load_labels():
    labels = {}
    for t in get_reasoning_task.__module__:  # not iterable, use get_all_reasoning_tasks
        pass
    from daph_x.coding.reasoning_tasks import get_all_reasoning_tasks
    for t in get_all_reasoning_tasks():
        labels[t.task_id] = EvaluationLabels(
            task_id=t.task_id,
            correct_answer=t.answer,
            answer_type=t.answer_type,
        )
    return labels


def load_checkpoints(checkpoints_path: Path) -> dict:
    checkpoints = {}
    with open(checkpoints_path) as f:
        for line in f:
            cp = json.loads(line)
            checkpoints[cp["checkpoint_id"]] = cp
    return checkpoints


def analyze(executions_path: Path, checkpoints_path: Path):
    labels = load_labels()
    checkpoints = load_checkpoints(checkpoints_path)

    events = []
    with open(executions_path) as f:
        for line in f:
            e = json.loads(line)
            if not e.get("admissible") or e.get("status") != "SUCCESS":
                continue

            cp = checkpoints.get(e.get("checkpoint_id"))
            if not cp:
                continue

            task_id = e["task_id"]
            lbl = labels.get(task_id)
            if not lbl:
                continue

            baseline_answer = cp["runtime_state"]["current_answer"]
            terminal_answer = e["candidate_answer"]

            baseline_correct = is_correct(baseline_answer, lbl)
            terminal_correct = is_correct(terminal_answer, lbl)

            if e["operator_id"] == "STOP":
                event = "NEUTRAL"
            elif terminal_correct and not baseline_correct:
                event = "RESCUE"
            elif not terminal_correct and baseline_correct:
                event = "BREAK"
            else:
                event = "WASTE"

            cost = e.get("cost", {})
            tokens = cost.get("tokens", 0) + cost.get("completion_tokens", 0)
            calls = cost.get("model_calls", 0)
            wall_ms = cost.get("wall_ms", 0)

            events.append({
                "execution_id": e["execution_id"],
                "task_id": task_id,
                "k": e["k"],
                "operator_id": e["operator_id"],
                "replicate_id": e["replicate_id"],
                "baseline_correct": baseline_correct,
                "terminal_correct": terminal_correct,
                "event": event,
                "tokens": tokens,
                "calls": calls,
                "wall_ms": wall_ms,
                "prompt_tokens": cost.get("tokens", 0),
                "completion_tokens": cost.get("completion_tokens", 0),
            })

    # Per-operator summary
    by_op = defaultdict(Counter)
    cost_by_op = defaultdict(lambda: {"tokens": 0, "calls": 0, "wall_ms": 0.0, "n": 0})
    for ev in events:
        op = ev["operator_id"]
        by_op[op][ev["event"]] += 1
        cost_by_op[op]["tokens"] += ev["tokens"]
        cost_by_op[op]["calls"] += ev["calls"]
        cost_by_op[op]["wall_ms"] += ev["wall_ms"]
        cost_by_op[op]["n"] += 1

    # Per-K summary
    by_k = defaultdict(Counter)
    for ev in events:
        by_k[ev["k"]][ev["event"]] += 1

    print("=" * 60)
    print("R13-A v2 Tournament Analysis")
    print("=" * 60)
    print(f"Total successful executions: {len(events)}")

    print("\nBy operator:")
    print(f"  {'Operator':20s} {'RESCUE':>6s} {'BREAK':>6s} {'WASTE':>6s} {'NEUTRAL':>7s} {'n':>5s} {'Rescue%':>8s}")
    for op in sorted(by_op.keys()):
        counts = by_op[op]
        total = sum(counts.values())
        rescue = counts.get("RESCUE", 0)
        break_ = counts.get("BREAK", 0)
        waste = counts.get("WASTE", 0)
        neutral = counts.get("NEUTRAL", 0)
        print(f"  {op:20s} {rescue:6d} {break_:6d} {waste:6d} {neutral:7d} {total:5d} {rescue/total*100:7.1f}%")

    print("\nBy K:")
    print(f"  {'K':>3s} {'RESCUE':>6s} {'BREAK':>6s} {'WASTE':>6s} {'n':>5s}")
    for k in sorted(by_k.keys()):
        counts = by_k[k]
        total = sum(counts.values())
        print(f"  {k:3d} {counts.get('RESCUE',0):6d} {counts.get('BREAK',0):6d} {counts.get('WASTE',0):6d} {total:5d}")

    print("\nPer-operator mean cost:")
    for op in sorted(cost_by_op.keys()):
        c = cost_by_op[op]
        n = c["n"]
        if n > 0:
            print(f"  {op:20s}: tokens={c['tokens']/n:.0f} calls={c['calls']/n:.1f} wall_ms={c['wall_ms']/n:.0f}")

    # Save detail
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executions", default="experiments/daph_x/r13/v2/executions.jsonl")
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    events = analyze(Path(args.executions), Path(args.checkpoints))
    if args.output:
        with open(args.output, "w") as f:
            for ev in events:
                f.write(json.dumps(ev, default=str) + "\n")
        print(f"\nSaved {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
