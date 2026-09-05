#!/usr/bin/env python3
"""R13-A v2.2 — corrected binary ceiling and cost normalization.

Blind analysis: seed-42 only until replicates finish.

Corrections:
1. Binary ceiling now uses best globally fixed continuation (J_bin_best).
2. V0 is the λ=0 optimal fixed continuation (J_bin_V0).
3. Heterogeneous oracle = max over all actions per state.
4. Cost normalized by operator version:
   - SAMPLE_STANDARD v2, SAMPLE_DIVERSE v2: tokens + completion_tokens.
   - CRITIQUE_RETRY v2, VERIFY_TARGETED v2: tokens (already total).
   - STOP: 0.
5. Task-clustered bootstrap with one-sided 95% UCB for the het - bin_best gap.
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
from daph_x.evaluation.r12_selector import select_r12_maxcal
from daph_x.operators.types import EvaluationLabels, Candidate


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


def _make_candidate(answer: str, response: str, meta: dict, index: int) -> Candidate:
    return Candidate(
        candidate_id=f"new_{index}",
        answer=answer,
        reasoning_trace=response,
        temperature=meta.get("temperature", 0.0),
        seed=meta.get("seed", 0),
        generation_index=index,
        metadata=meta,
    )


def _recompute_terminal_answer(e: dict, cp: dict) -> str:
    op = e["operator_id"]
    evidence = e.get("evidence", {})

    if op in ("SAMPLE_STANDARD", "SAMPLE_DIVERSE"):
        originals = [
            Candidate(
                candidate_id=c["candidate_id"],
                answer=c["answer"],
                reasoning_trace=c.get("reasoning_trace", ""),
                temperature=c.get("temperature", 0.0),
                seed=c.get("seed", 0),
                generation_index=c["generation_index"],
                metadata=c.get("metadata", {}),
            )
            for c in cp["runtime_state"]["candidates"]
        ]
        new_cands = evidence.get("new_candidates", [])
        new = [
            _make_candidate(
                c.get("answer", ""),
                c.get("response", ""),
                {k: v for k, v in c.items() if k not in ("answer", "response")},
                i,
            )
            for i, c in enumerate(new_cands)
        ]
        combined = originals + new
        if not combined:
            return ""
        return select_r12_maxcal(combined).answer

    return e.get("candidate_answer", "")


def _normalize_tokens(e: dict) -> int:
    """Normalize tokens from historical receipts by operator version."""
    op = e.get("operator_id", "")
    version = e.get("operator_version", "2")
    cost = e.get("cost", {})
    tokens = cost.get("tokens", 0)
    completion = cost.get("completion_tokens", 0)

    if op == "STOP":
        return 0
    if op in ("SAMPLE_STANDARD", "SAMPLE_DIVERSE") and version == "2":
        return tokens + completion
    if op in ("CRITIQUE_RETRY", "VERIFY_TARGETED") and version == "2":
        return tokens
    # Fallback for unknown versions
    return tokens


def load_events(executions_path: Path, checkpoints_path: Path, replicate_filter: list = None):
    labels = load_labels()
    checkpoints = load_checkpoints(checkpoints_path)

    events = []
    with open(executions_path) as f:
        for line in f:
            e = json.loads(line)
            if not e.get("admissible") or e.get("status") != "SUCCESS":
                continue
            if replicate_filter and e.get("replicate_id") not in replicate_filter:
                continue

            cp = checkpoints.get(e.get("checkpoint_id"))
            if not cp:
                continue

            task_id = e["task_id"]
            lbl = labels.get(task_id)
            if not lbl:
                continue

            baseline_answer = cp["runtime_state"]["current_answer"]
            terminal_answer = _recompute_terminal_answer(e, cp)

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

            total_tokens = _normalize_tokens(e)

            events.append({
                "execution_id": e["execution_id"],
                "checkpoint_id": e["checkpoint_id"],
                "task_id": task_id,
                "k": e["k"],
                "operator_id": e["operator_id"],
                "operator_version": e.get("operator_version", "?"),
                "replicate_id": e.get("replicate_id", 0),
                "baseline_correct": baseline_correct,
                "terminal_correct": terminal_correct,
                "event": event,
                "terminal_answer": terminal_answer,
                "tokens": total_tokens,
                "calls": e.get("cost", {}).get("model_calls", 0),
                "wall_ms": e.get("cost", {}).get("wall_ms", 0.0),
            })

    return events


def keep_complete_checkpoints(events):
    by_cp = defaultdict(list)
    for ev in events:
        by_cp[ev["checkpoint_id"]].append(ev)

    complete = []
    for cp_id, evs in by_cp.items():
        ops = {e["operator_id"] for e in evs}
        if ops >= {"STOP", "SAMPLE_STANDARD", "SAMPLE_DIVERSE", "CRITIQUE_RETRY", "VERIFY_TARGETED"}:
            complete.extend(evs)

    return complete


def analyze_q4(events):
    print("\n" + "=" * 70)
    print("Q4 — Operator causal utility (v2.2 corrected cost + selector)")
    print("=" * 70)

    complete = keep_complete_checkpoints(events)
    by_op = defaultdict(Counter)
    by_op_conditional = defaultdict(Counter)
    cost_by_op = defaultdict(list)

    for ev in complete:
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
    print(f"  {'Operator':20s} {'n_wrong':>8s} {'Rescue%':>10s} {'Waste%':>10s} {'Break%':>10s}")
    for op in sorted(by_op_conditional.keys()):
        counts = by_op_conditional[op]
        total = sum(counts.values())
        r = counts.get("RESCUE", 0)
        w = counts.get("WASTE", 0)
        b = counts.get("BREAK", 0)
        print(f"  {op:20s} {total:8d} {r/total*100:9.1f}% {w/total*100:9.1f}% {b/total*100:9.1f}%")


def _utility(ev: dict, lambda_: float) -> float:
    return float(ev["terminal_correct"]) - lambda_ * (ev["tokens"] / 1000.0)


def analyze_q5(events, lambda_values=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)):
    print("\n" + "=" * 70)
    print("Q5 — Three ceilings: heterogeneous, bin-best, bin-V0 (v2.2)")
    print("=" * 70)

    complete = keep_complete_checkpoints(events)
    by_cp = defaultdict(list)
    for ev in complete:
        by_cp[ev["checkpoint_id"]].append(ev)

    for lambda_ in lambda_values:
        # Compute per-state utilities for every operator
        op_utils_by_cp = {}
        all_operators = set()
        for cp_id, evs in by_cp.items():
            op_utils = {ev["operator_id"]: _utility(ev, lambda_) for ev in evs}
            op_utils_by_cp[cp_id] = op_utils
            all_operators.update(op_utils.keys())

        all_operators = sorted(all_operators)
        cont_operators = [op for op in all_operators if op != "STOP"]

        # Heterogeneous oracle: best action per state
        het_utilities = []
        het_actions = Counter()
        for cp_id, op_utils in op_utils_by_cp.items():
            best_het = max(op_utils.values())
            best_het_op = max(op_utils.items(),
                             key=lambda x: (x[1], -next(e["tokens"] for e in by_cp[cp_id] if e["operator_id"] == x[0]), x[0]))[0]
            het_utilities.append(best_het)
            het_actions[best_het_op] += 1

        # Best globally fixed continuation (paired with STOP per state)
        bin_best_utility = -1e9
        bin_best_op = None
        for cont in cont_operators:
            bin_utilities = []
            for cp_id, op_utils in op_utils_by_cp.items():
                stop_u = op_utils.get("STOP", -1e9)
                cont_u = op_utils.get(cont, -1e9)
                bin_utilities.append(max(stop_u, cont_u))
            mean_bin = sum(bin_utilities) / len(bin_utilities)
            if mean_bin > bin_best_utility:
                bin_best_utility = mean_bin
                bin_best_op = cont

        # V0: best fixed continuation at λ=0
        q_means = {}
        for cont in cont_operators:
            vals = []
            for cp_id, evs in by_cp.items():
                ev = next((e for e in evs if e["operator_id"] == cont), None)
                if ev:
                    vals.append(float(ev["terminal_correct"]))
            q_means[cont] = sum(vals) / len(vals) if vals else -1e9
        v0 = max(q_means, key=lambda cont: (q_means[cont], -next(e["tokens"] for e in next(iter(by_cp.values())) if e["operator_id"] == cont), cont))

        bin_v0_utilities = []
        for cp_id, op_utils in op_utils_by_cp.items():
            stop_u = op_utils.get("STOP", -1e9)
            v0_u = op_utils.get(v0, -1e9)
            bin_v0_utilities.append(max(stop_u, v0_u))

        het_mean = sum(het_utilities) / len(het_utilities)
        bin_v0_mean = sum(bin_v0_utilities) / len(bin_v0_utilities)

        print(f"\nλ = {lambda_:.3f}:")
        print(f"  Heterogeneous oracle:  U = {het_mean:.4f}, actions: {dict(het_actions)}")
        print(f"  Best fixed binary:     {bin_best_op:20s} U = {bin_best_utility:.4f}")
        print(f"  V0 fixed binary:       {v0:20s} U = {bin_v0_mean:.4f}")
        print(f"  Δ het vs bin-best:     {het_mean - bin_best_utility:+.4f}")
        print(f"  Δ het vs bin-V0:       {het_mean - bin_v0_mean:+.4f}")


def _bootstrap_ceilings(events, lambda_values=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2), n_boot=10000, seed=99):
    print("\n" + "=" * 70)
    print("Bootstrap: J_het - J_bin-best task-clustered UCB (v2.2)")
    print("=" * 70)

    complete = keep_complete_checkpoints(events)
    by_task = defaultdict(list)
    for ev in complete:
        by_task[ev["task_id"]].append(ev)

    task_ids = list(by_task.keys())
    rng = random.Random(seed)

    for lambda_ in lambda_values:
        gaps = []
        for _ in range(n_boot):
            sampled_tasks = [rng.choice(task_ids) for _ in range(len(task_ids))]

            # Build sampled per-checkpoint utilities
            op_utils_by_cp = defaultdict(dict)
            for tid in sampled_tasks:
                for ev in by_task[tid]:
                    op_utils_by_cp[ev["checkpoint_id"]][ev["operator_id"]] = _utility(ev, lambda_)

            conts = set()
            for cp_id, ou in op_utils_by_cp.items():
                conts.update(ou.keys())
            conts = sorted(c for c in conts if c != "STOP")

            if not conts:
                continue

            # Heterogeneous
            het_vals = []
            for cp_id, ou in op_utils_by_cp.items():
                het_vals.append(max(ou.values()))

            # Best binary
            bin_best = -1e9
            for cont in conts:
                bin_vals = []
                for cp_id, ou in op_utils_by_cp.items():
                    bin_vals.append(max(ou.get("STOP", -1e9), ou.get(cont, -1e9)))
                mean_bin = sum(bin_vals) / len(bin_vals)
                if mean_bin > bin_best:
                    bin_best = mean_bin

            het_mean = sum(het_vals) / len(het_vals) if het_vals else 0
            gaps.append(het_mean - bin_best)

        gaps_sorted = sorted(gaps)
        mean_gap = sum(gaps) / len(gaps)
        ucb95 = gaps_sorted[int(0.95 * len(gaps_sorted))]
        lcb5 = gaps_sorted[int(0.05 * len(gaps_sorted))]
        print(f"  λ={lambda_:.3f}: mean={mean_gap:+.4f}, UCB95={ucb95:+.4f}, LCB5={lcb5:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executions", default="experiments/daph_x/r13/v2/executions.jsonl")
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    parser.add_argument("--replicate_filter", default="42", help="Comma-separated replicate seeds to use")
    args = parser.parse_args()

    rep_filter = [int(x) for x in args.replicate_filter.split(",")] if args.replicate_filter else None
    events = load_events(Path(args.executions), Path(args.checkpoints), rep_filter)
    complete = keep_complete_checkpoints(events)
    print(f"Loaded {len(events)} events; {len(complete)} from matched complete checkpoints")
    print(f"Replicate filter: {rep_filter}")

    analyze_q4(events)
    analyze_q5(events)
    _bootstrap_ceilings(events)


if __name__ == "__main__":
    main()
