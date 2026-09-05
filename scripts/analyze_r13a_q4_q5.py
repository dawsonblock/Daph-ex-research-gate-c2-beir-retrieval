#!/usr/bin/env python3
"""R13-A v2 detailed Q4/Q5 analysis.

Q4: Does any nontrivial operator materially beat vanilla resampling?
Q5: Does oracle heterogeneous routing beat the best fixed operator?
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.reasoning_tasks import get_all_reasoning_tasks
from daph_x.evaluation.answer_judge import is_correct
from daph_x.operators.types import EvaluationLabels


def load_labels():
    labels = {}
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


def load_events(executions_path: Path, checkpoints_path: Path):
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
                "checkpoint_id": e["checkpoint_id"],
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
            })

    return events


def cluster_by_checkpoint(events):
    by_cp = defaultdict(list)
    for ev in events:
        by_cp[ev["checkpoint_id"]].append(ev)
    return by_cp


def analyze_q4(events):
    """Q4: does any nontrivial operator materially beat vanilla resampling?"""
    print("\n" + "=" * 70)
    print("Q4 — Operator causal utility")
    print("=" * 70)

    by_op = defaultdict(Counter)
    by_op_conditional = defaultdict(Counter)
    cost_by_op = defaultdict(list)

    for ev in events:
        if ev["operator_id"] == "STOP":
            continue
        op = ev["operator_id"]
        by_op[op][ev["event"]] += 1
        if not ev["baseline_correct"]:
            by_op_conditional[op][ev["event"]] += 1
        cost_by_op[op].append(ev["tokens"])

    print("\nAll-state event rates:")
    print(f"  {'Operator':20s} {'n':>5s} {'Rescue%':>8s} {'Break%':>8s} {'Waste%':>8s} {'MeanTok':>9s}")
    for op in sorted(by_op.keys()):
        counts = by_op[op]
        total = sum(counts.values())
        r = counts.get("RESCUE", 0)
        b = counts.get("BREAK", 0)
        w = counts.get("WASTE", 0)
        mean_tok = sum(cost_by_op[op]) / len(cost_by_op[op]) if cost_by_op[op] else 0
        print(f"  {op:20s} {total:5d} {r/total*100:7.1f}% {b/total*100:7.1f}% {w/total*100:7.1f}% {mean_tok:9.0f}")

    print("\nOpportunity-conditional rescue (baseline wrong only):")
    print(f"  {'Operator':20s} {'n_wrong':>8s} {'Rescue%':>10s} {'Waste%':>10s}")
    for op in sorted(by_op_conditional.keys()):
        counts = by_op_conditional[op]
        total = sum(counts.values())
        r = counts.get("RESCUE", 0)
        w = counts.get("WASTE", 0)
        print(f"  {op:20s} {total:8d} {r/total*100:9.1f}% {w/total*100:9.1f}%")

    # Standard baseline
    std_counts = by_op.get("SAMPLE_STANDARD", Counter())
    print("\nQ4 gate: each non-standard operator vs SAMPLE_STANDARD on opportunity-conditional rescue")
    for op in sorted(by_op_conditional.keys()):
        if op == "SAMPLE_STANDARD":
            continue
        op_r = by_op_conditional[op].get("RESCUE", 0)
        op_total = sum(by_op_conditional[op].values())
        std_r = by_op_conditional["SAMPLE_STANDARD"].get("RESCUE", 0)
        std_total = sum(by_op_conditional["SAMPLE_STANDARD"].values())
        delta = (op_r / op_total) - (std_r / std_total) if op_total and std_total else 0
        print(f"  {op:20s}: Δrescue = {delta:+.3f}  ({op_r}/{op_total} vs {std_r}/{std_total})")


def analyze_q5(events, lambda_values=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2)):
    """Q5 — Oracle heterogeneous routing headroom."""
    print("\n" + "=" * 70)
    print("Q5 — Oracle heterogeneous routing headroom")
    print("=" * 70)

    by_cp = cluster_by_checkpoint(events)

    # Per-state oracle action value
    oracle_results = {}
    best_fixed_results = {}
    action_counts = Counter()

    for lambda_ in lambda_values:
        state_values = {}
        for cp_id, evs in by_cp.items():
            op_values = {}
            for ev in evs:
                # Utility = terminal_correct - lambda * (tokens / 1000)
                u = float(ev["terminal_correct"]) - lambda_ * (ev["tokens"] / 1000)
                op_values[ev["operator_id"]] = u

            # Oracle picks best action (tie-break: fewer tokens, then lexical)
            best_u = -1e9
            best_op = "STOP"
            best_cost = 0
            for op in sorted(op_values.keys()):
                u = op_values[op]
                cost = next(ev["tokens"] for ev in evs if ev["operator_id"] == op)
                if u > best_u + 1e-9:
                    best_u = u
                    best_op = op
                    best_cost = cost
                elif abs(u - best_u) < 1e-9:
                    if cost < best_cost:
                        best_op = op
                        best_cost = cost
            state_values[cp_id] = (best_op, best_u, op_values)
            action_counts[best_op] += 1

        oracle_results[lambda_] = state_values

        # Best fixed operator
        fixed_scores = defaultdict(list)
        for op in set(ev["operator_id"] for ev in events):
            for cp_id, evs in by_cp.items():
                ev = next((e for e in evs if e["operator_id"] == op), None)
                if ev:
                    u = float(ev["terminal_correct"]) - lambda_ * (ev["tokens"] / 1000)
                    fixed_scores[op].append(u)

        best_fixed = None
        best_fixed_mean = -1e9
        for op, vals in fixed_scores.items():
            mean_u = sum(vals) / len(vals)
            if mean_u > best_fixed_mean:
                best_fixed_mean = mean_u
                best_fixed = op

        best_fixed_results[lambda_] = (best_fixed, best_fixed_mean, fixed_scores[best_fixed])

        oracle_mean = sum(v[1] for v in state_values.values()) / len(state_values)
        delta = oracle_mean - best_fixed_mean

        print(f"\nλ = {lambda_:.3f}:")
        print(f"  Oracle mean utility: {oracle_mean:.4f}")
        print(f"  Best fixed operator: {best_fixed:20s} mean U = {best_fixed_mean:.4f}")
        print(f"  Routing headroom Δ:  {delta:+.4f}")
        print(f"  Oracle action distribution:")
        total = sum(action_counts.values())
        for op in sorted(action_counts.keys()):
            print(f"    {op:20s}: {action_counts[op]/total*100:5.1f}%")


def paired_bootstrap(events, n_boot=1000, seed=42):
    """Clustered bootstrap by checkpoint."""
    print("\n" + "=" * 70)
    print("Paired bootstrap CIs (clustered by checkpoint)")
    print("=" * 70)

    by_cp = cluster_by_checkpoint(events)
    cp_ids = list(by_cp.keys())
    rng = random.Random(seed)

    # Focus on rescue rates for each operator
    op_rescue_rates = defaultdict(list)
    for _ in range(n_boot):
        sampled_cps = [rng.choice(cp_ids) for _ in range(len(cp_ids))]
        by_op = defaultdict(Counter)
        for cp_id in sampled_cps:
            for ev in by_cp[cp_id]:
                if ev["operator_id"] == "STOP":
                    continue
                by_op[ev["operator_id"]][ev["event"]] += 1
        for op, counts in by_op.items():
            total = sum(counts.values())
            if total > 0:
                op_rescue_rates[op].append(counts.get("RESCUE", 0) / total)

    print(f"  {'Operator':20s} {'Mean%':>8s} {'P5%':>8s} {'P95%':>8s}")
    for op in sorted(op_rescue_rates.keys()):
        rates = op_rescue_rates[op]
        if rates:
            rates_sorted = sorted(rates)
            mean = sum(rates) / len(rates)
            p5 = rates_sorted[int(0.05 * len(rates_sorted))]
            p95 = rates_sorted[int(0.95 * len(rates_sorted))]
            print(f"  {op:20s} {mean*100:7.1f}% {p5*100:7.1f}% {p95*100:7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executions", default="experiments/daph_x/r13/v2/executions.jsonl")
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    args = parser.parse_args()

    events = load_events(Path(args.executions), Path(args.checkpoints))
    print(f"Loaded {len(events)} successful executions")

    analyze_q4(events)
    analyze_q5(events)
    paired_bootstrap(events)


if __name__ == "__main__":
    main()
