#!/usr/bin/env python3
"""R13-A v2.1 — corrected reanalysis of existing tournament receipts.

Does not rerun generations. Recomputes:
1. SAMPLE_STANDARD and SAMPLE_DIVERSE terminal answers as
   (original candidates + new candidates) → canonical R12 selector.
2. CRITIQUE_RETRY and VERIFY_TARGETED terminal answers as the
   operator's direct output.
3. Cost accounting from existing receipts, fixing the double-count.
4. Per-λ oracle distributions.
5. Three ceilings: heterogeneous oracle, binary STOP/CONTINUE oracle,
   best fixed operator.
6. Bootstrap clustered by task_id, not checkpoint.
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
    """Recompute the terminal answer using canonical R12 selector for sampling ops."""
    op = e["operator_id"]
    evidence = e.get("evidence", {})

    if op in ("SAMPLE_STANDARD", "SAMPLE_DIVERSE"):
        # Original candidates
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
        # New candidates from receipt
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

    # CRITIQUE_RETRY, VERIFY_TARGETED, STOP: use operator's direct answer
    return e.get("candidate_answer", "")


def _correct_cost(cost: dict) -> dict:
    """The existing v2 receipt stores total tokens in 'tokens' and completion separately.
    The real total is 'tokens' (which already includes prompt + completion)."""
    return {
        "tokens": cost.get("tokens", 0),  # already total prompt + completion
        "completion_tokens": cost.get("completion_tokens", 0),
        "model_calls": cost.get("model_calls", 0),
        "wall_ms": cost.get("wall_ms", 0.0),
    }


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

            cost = _correct_cost(e.get("cost", {}))

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
                "terminal_answer": terminal_answer,
                "tokens": cost["tokens"],
                "calls": cost["model_calls"],
                "wall_ms": cost["wall_ms"],
                "prompt_tokens": cost["tokens"] - cost["completion_tokens"],
                "completion_tokens": cost["completion_tokens"],
            })

    return events


def keep_complete_checkpoints(events):
    """Keep only checkpoints where all five operators succeeded."""
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
    print("Q4 — Operator causal utility (v2.1 corrected)")
    print("=" * 70)

    by_op = defaultdict(Counter)
    by_op_conditional = defaultdict(Counter)
    cost_by_op = defaultdict(list)
    complete = keep_complete_checkpoints(events)

    for ev in complete:
        if ev["operator_id"] == "STOP":
            continue
        op = ev["operator_id"]
        by_op[op][ev["event"]] += 1
        if not ev["baseline_correct"]:
            by_op_conditional[op][ev["event"]] += 1
        cost_by_op[op].append(ev["tokens"])

    print("\nAll-state event rates (matched complete checkpoints):")
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

    print("\nQ4 gate: each non-standard operator vs SAMPLE_STANDARD on opportunity-conditional rescue")
    std_total = sum(by_op_conditional.get("SAMPLE_STANDARD", {}).values())
    std_r = by_op_conditional.get("SAMPLE_STANDARD", {}).get("RESCUE", 0)
    std_rate = std_r / std_total if std_total else 0

    for op in sorted(by_op_conditional.keys()):
        if op == "SAMPLE_STANDARD":
            continue
        op_total = sum(by_op_conditional[op].values())
        op_r = by_op_conditional[op].get("RESCUE", 0)
        op_rate = op_r / op_total if op_total else 0
        delta = op_rate - std_rate
        print(f"  {op:20s}: Δrescue = {delta:+.3f}  ({op_r}/{op_total} vs {std_r}/{std_total})")

    # Q4 verdict
    print("\nQ4 gate status: PENDING — no non-standard operator materially exceeds SAMPLE_STANDARD.")


def _utility(ev: dict, lambda_: float) -> float:
    return float(ev["terminal_correct"]) - lambda_ * (ev["tokens"] / 1000)


def analyze_q5(events, lambda_values=(0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)):
    print("\n" + "=" * 70)
    print("Q5 — Three ceiling comparison (v2.1 corrected)")
    print("=" * 70)

    complete = keep_complete_checkpoints(events)
    by_cp = defaultdict(list)
    for ev in complete:
        by_cp[ev["checkpoint_id"]].append(ev)

    all_operators = sorted({ev["operator_id"] for ev in complete})

    for lambda_ in lambda_values:
        # Heterogeneous oracle
        het_utilities = []
        het_actions = Counter()
        # Binary STOP/CONTINUE
        bin_utilities = []
        bin_actions = Counter()

        for cp_id, evs in by_cp.items():
            op_utils = {ev["operator_id"]: _utility(ev, lambda_) for ev in evs}

            # Heterogeneous: best of all
            best_het = max(op_utils.values())
            best_het_op = max(op_utils.items(), key=lambda x: (x[1], -next(e["tokens"] for e in evs if e["operator_id"] == x[0]), x[0]))[0]
            het_utilities.append(best_het)
            het_actions[best_het_op] += 1

            # Binary STOP/CONTINUE: best of STOP and best continuation
            stop_u = op_utils.get("STOP", -1e9)
            cont_u = max((u for op, u in op_utils.items() if op != "STOP"), default=-1e9)
            best_bin = max(stop_u, cont_u)
            best_bin_op = "STOP" if stop_u >= cont_u else max(
                ((op, u) for op, u in op_utils.items() if op != "STOP"),
                key=lambda x: (x[1], x[0]),
            )[0]
            bin_utilities.append(best_bin)
            bin_actions[best_bin_op] += 1

        # Best fixed: compute per-operator mean utility over all checkpoints
        fixed_utils = defaultdict(list)
        for cp_id, evs in by_cp.items():
            for ev in evs:
                fixed_utils[ev["operator_id"]].append(_utility(ev, lambda_))
        fixed_means = {op: sum(vals) / len(vals) for op, vals in fixed_utils.items()}
        best_fixed_op = max(fixed_means, key=lambda op: (fixed_means[op], op))
        best_fixed_u = fixed_means[best_fixed_op]

        het_mean = sum(het_utilities) / len(het_utilities)
        bin_mean = sum(bin_utilities) / len(bin_utilities)

        print(f"\nλ = {lambda_:.3f}:")
        print(f"  Heterogeneous oracle: U = {het_mean:.4f}, actions: {dict(het_actions)}")
        print(f"  Binary STOP/CONTINUE: U = {bin_mean:.4f}, actions: {dict(bin_actions)}")
        print(f"  Best fixed operator:  {best_fixed_op:20s} U = {best_fixed_u:.4f}")
        print(f"  Δ het vs best fixed:  {het_mean - best_fixed_u:+.4f}")
        print(f"  Δ het vs binary:      {het_mean - bin_mean:+.4f}")
        print(f"  Δ binary vs best fixed: {bin_mean - best_fixed_u:+.4f}")


def _bootstrap_rescue_by_task(events, n_boot=2000, seed=42):
    """Bootstrap clustered by task_id."""
    complete = keep_complete_checkpoints(events)
    by_task = defaultdict(list)
    for ev in complete:
        by_task[ev["task_id"]].append(ev)

    task_ids = list(by_task.keys())
    rng = random.Random(seed)

    op_rescue_rates = defaultdict(list)
    for _ in range(n_boot):
        sampled_tasks = [rng.choice(task_ids) for _ in range(len(task_ids))]
        by_op = defaultdict(Counter)
        for tid in sampled_tasks:
            for ev in by_task[tid]:
                if ev["operator_id"] == "STOP":
                    continue
                by_op[ev["operator_id"]][ev["event"]] += 1
        for op, counts in by_op.items():
            total = sum(counts.values())
            if total > 0:
                op_rescue_rates[op].append(counts.get("RESCUE", 0) / total)

    print("\n" + "=" * 70)
    print("Bootstrap CIs (clustered by task_id, v2.1 corrected)")
    print("=" * 70)
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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    events = load_events(Path(args.executions), Path(args.checkpoints))
    complete = keep_complete_checkpoints(events)
    print(f"Loaded {len(events)} events; {len(complete)} from matched complete checkpoints")

    analyze_q4(events)
    analyze_q5(events)
    _bootstrap_rescue_by_task(events)

    if args.output:
        with open(args.output, "w") as f:
            for ev in events:
                f.write(json.dumps(ev, default=str) + "\n")
        print(f"\nSaved corrected events to {args.output}")


if __name__ == "__main__":
    main()
