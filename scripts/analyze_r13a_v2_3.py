#!/usr/bin/env python3
"""R13-A v2.3 — replicated state-action averaging and multiplicity-correct bootstrap.

Blind analyzer committed before inspecting seeds 123/2024 completed results.

Key fixes over v2.2:
1. Averages Q(s,a) and c(s,a) over required replicates (42, 123, 2024).
2. Drops checkpoint unless all 5 operators × 3 seeds = 15 receipts present.
3. Bootstrap resampling with explicit task weights (multiplicity preserved).
4. Fail-closed cost decoder.
5. Correct v0 tie-breaking.
6. Uses the full frozen λ grid from the preregistration.
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

REQUIRED_REPLICATES = {42, 123, 2024}
REQUIRED_OPERATORS = {"STOP", "SAMPLE_STANDARD", "SAMPLE_DIVERSE", "CRITIQUE_RETRY", "VERIFY_TARGETED"}
LAMBDA_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


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
    op = e.get("operator_id", "")
    version = e.get("operator_version", "?")
    cost = e.get("cost", {})
    tokens = cost.get("tokens", 0)
    completion = cost.get("completion_tokens", 0)

    if op == "STOP":
        return 0
    if version == "2":
        if op in ("SAMPLE_STANDARD", "SAMPLE_DIVERSE"):
            return tokens + completion
        if op in ("CRITIQUE_RETRY", "VERIFY_TARGETED"):
            return tokens
    raise ValueError(f"No frozen cost decoder for {op} v{version}")


def load_events(executions_path: Path, checkpoints_path: Path):
    labels = load_labels()
    checkpoints = load_checkpoints(checkpoints_path)

    events = []
    with open(executions_path) as f:
        for line in f:
            e = json.loads(line)
            if not e.get("admissible") or e.get("status") != "SUCCESS":
                continue
            if e.get("replicate_id") not in REQUIRED_REPLICATES:
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


def average_replicates(events):
    """Average Q and cost over replicates for each (checkpoint, operator)."""
    by_cp_op = defaultdict(list)
    for ev in events:
        key = (ev["checkpoint_id"], ev["operator_id"])
        by_cp_op[key].append(ev)

    averaged = []
    for (cp_id, op), evs in by_cp_op.items():
        replicates = {ev["replicate_id"] for ev in evs}
        if replicates != REQUIRED_REPLICATES:
            continue

        mean_q = sum(ev["terminal_correct"] for ev in evs) / len(evs)
        mean_c = sum(ev["tokens"] for ev in evs) / len(evs)
        mean_calls = sum(ev["calls"] for ev in evs) / len(evs)

        averaged.append({
            "checkpoint_id": cp_id,
            "operator_id": op,
            "task_id": evs[0]["task_id"],
            "k": evs[0]["k"],
            "baseline_correct": evs[0]["baseline_correct"],
            "mean_q": mean_q,
            "mean_c": mean_c,
            "mean_calls": mean_calls,
        })

    return averaged


def keep_complete_checkpoints(averaged_events):
    """Require 5 operators × 3 seeds = 15 receipts per checkpoint before averaging."""
    by_cp = defaultdict(list)
    for ev in averaged_events:
        by_cp[ev["checkpoint_id"]].append(ev)

    complete = []
    for cp_id, evs in by_cp.items():
        ops = {ev["operator_id"] for ev in evs}
        if ops == REQUIRED_OPERATORS and len(evs) == 5:
            complete.extend(evs)

    return complete


def analyze_q4(raw_events):
    """Q4 descriptive event rates computed on the raw replicated events (all 1,350)."""
    print("\n" + "=" * 70)
    print("Q4 — Operator causal utility (v2.3, raw replicated events)")
    print("=" * 70)

    by_op = defaultdict(Counter)
    by_op_conditional = defaultdict(Counter)
    cost_by_op = defaultdict(list)

    for ev in raw_events:
        if ev["operator_id"] == "STOP":
            continue
        op = ev["operator_id"]
        if ev["event"] in ("RESCUE", "BREAK", "WASTE", "NEUTRAL"):
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
    return ev["mean_q"] - lambda_ * (ev["mean_c"] / 1000.0)


def _select_v0(by_cp: dict) -> str:
    """Select fixed continuation at lambda=0: best mean Q, tie-break cost/calls/id."""
    cont_q = defaultdict(list)
    cont_c = defaultdict(list)
    cont_calls = defaultdict(list)

    for cp_id, evs in by_cp.items():
        for ev in evs:
            if ev["operator_id"] == "STOP":
                continue
            cont_q[ev["operator_id"]].append(ev["mean_q"])
            cont_c[ev["operator_id"]].append(ev["mean_c"])
            cont_calls[ev["operator_id"]].append(ev["mean_calls"])

    if not cont_q:
        raise ValueError("No continuation operators found")

    stats = {}
    for op in cont_q:
        stats[op] = (
            sum(cont_q[op]) / len(cont_q[op]),
            sum(cont_c[op]) / len(cont_c[op]),
            sum(cont_calls[op]) / len(cont_calls[op]),
        )

    # Minimize negative q, then cost, then calls, then id
    best = min(stats.keys(), key=lambda op: (-stats[op][0], stats[op][1], stats[op][2], op))
    return best


def analyze_q5(averaged_events):
    print("\n" + "=" * 70)
    print("Q5 — Three ceilings: heterogeneous, bin-best, bin-V0 (v2.3)")
    print("=" * 70)

    complete = keep_complete_checkpoints(averaged_events)
    by_cp = defaultdict(list)
    for ev in complete:
        by_cp[ev["checkpoint_id"]].append(ev)

    v0 = _select_v0(by_cp)
    print(f"v0 (λ=0 best continuation): {v0}")

    for lambda_ in LAMBDA_GRID:
        op_utils_by_cp = {}
        for cp_id, evs in by_cp.items():
            op_utils_by_cp[cp_id] = {ev["operator_id"]: _utility(ev, lambda_) for ev in evs}

        cont_operators = sorted({op for ou in op_utils_by_cp.values() for op in ou if op != "STOP"})

        # Heterogeneous oracle
        het_utilities = []
        het_actions = Counter()
        for cp_id, ou in op_utils_by_cp.items():
            best_het = max(ou.values())
            best_het_op = max(ou.items(), key=lambda x: (x[1], -next(e["mean_c"] for e in by_cp[cp_id] if e["operator_id"] == x[0]), x[0]))[0]
            het_utilities.append(best_het)
            het_actions[best_het_op] += 1

        # Best fixed binary
        bin_best_utility = -1e9
        bin_best_op = None
        for cont in cont_operators:
            bin_vals = []
            for cp_id, ou in op_utils_by_cp.items():
                bin_vals.append(max(ou.get("STOP", -1e9), ou.get(cont, -1e9)))
            mean_bin = sum(bin_vals) / len(bin_vals)
            if mean_bin > bin_best_utility:
                bin_best_utility = mean_bin
                bin_best_op = cont

        # V0 fixed binary
        bin_v0_vals = []
        for cp_id, ou in op_utils_by_cp.items():
            bin_v0_vals.append(max(ou.get("STOP", -1e9), ou.get(v0, -1e9)))
        bin_v0_utility = sum(bin_v0_vals) / len(bin_v0_vals)

        het_mean = sum(het_utilities) / len(het_utilities)

        print(f"\nλ = {lambda_:.3f}:")
        print(f"  Heterogeneous oracle:  U = {het_mean:.4f}, actions: {dict(het_actions)}")
        print(f"  Best fixed binary:     {bin_best_op:20s} U = {bin_best_utility:.4f}")
        print(f"  V0 fixed binary:       {v0:20s} U = {bin_v0_utility:.4f}")
        print(f"  Δ het vs bin-best:     {het_mean - bin_best_utility:+.4f}")
        print(f"  Δ het vs bin-V0:       {het_mean - bin_v0_utility:+.4f}")


def _bootstrap_ceilings(averaged_events, n_boot=10000, seed=99):
    print("\n" + "=" * 70)
    print("Bootstrap: J_het - J_bin_best with multiplicity-correct task-cluster (v2.3.1)")
    print("=" * 70)

    complete = keep_complete_checkpoints(averaged_events)

    # Preserve all checkpoint states per task so multi-K tasks are not collapsed.
    states_by_task = defaultdict(list)
    state_records = {}
    for ev in complete:
        cp_id = ev["checkpoint_id"]
        if cp_id not in state_records:
            state_records[cp_id] = {
                "task_id": ev["task_id"],
                "operators": {},
            }
        state_records[cp_id]["operators"][ev["operator_id"]] = (ev["mean_q"], ev["mean_c"])

    for cp_id, rec in state_records.items():
        states_by_task[rec["task_id"]].append(rec["operators"])

    task_ids = sorted(states_by_task.keys())
    rng = random.Random(seed)

    for lambda_ in LAMBDA_GRID:
        gaps = []
        for _ in range(n_boot):
            sampled_tasks = [rng.choice(task_ids) for _ in range(len(task_ids))]
            weights = Counter(sampled_tasks)

            # Identify continuation operators present across all sampled states
            conts = set()
            for tid in states_by_task:
                for ops in states_by_task[tid]:
                    conts.update(ops.keys())
            conts = sorted(c for c in conts if c != "STOP")
            if not conts:
                continue

            # Heterogeneous values: per-checkpoint max, weighted by task multiplicity
            het_num = 0.0
            denom = 0
            for tid, weight in weights.items():
                for ops in states_by_task[tid]:
                    utilities = {op: q - lambda_ * (c / 1000.0) for op, (q, c) in ops.items()}
                    best = max(utilities.values())
                    het_num += weight * best
                    denom += weight

            # Best fixed binary: for each continuation, per-checkpoint max(STOP, cont)
            bin_best = -1e9
            for cont in conts:
                bin_num = 0.0
                bin_den = 0
                for tid, weight in weights.items():
                    for ops in states_by_task[tid]:
                        utilities = {op: q - lambda_ * (c / 1000.0) for op, (q, c) in ops.items()}
                        stop_u = utilities.get("STOP", -1e9)
                        cont_u = utilities.get(cont, -1e9)
                        val = max(stop_u, cont_u)
                        bin_num += weight * val
                        bin_den += weight
                mean_bin = bin_num / bin_den if bin_den else 0
                if mean_bin > bin_best:
                    bin_best = mean_bin

            het_mean = het_num / denom if denom else 0
            gaps.append(het_mean - bin_best)

        gaps_sorted = sorted(gaps)
        mean_gap = sum(gaps) / len(gaps)
        ucb95 = gaps_sorted[int(0.95 * len(gaps_sorted))]
        lcb5 = gaps_sorted[int(0.05 * len(gaps_sorted))]

        # Eligibility notation: pass if UCB95 < 0.005
        status = "PASS" if ucb95 < 0.005 else "FAIL"
        print(f"  λ={lambda_:.3f}: Δ mean={mean_gap:+.4f}, UCB95={ucb95:+.4f}, LCB5={lcb5:+.4f}, eligibility {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executions", default="experiments/daph_x/r13/v2/executions.jsonl")
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    args = parser.parse_args()

    events = load_events(Path(args.executions), Path(args.checkpoints))
    print(f"Loaded {len(events)} raw events")

    # Integrity: each complete checkpoint should have exactly 5 operators × 3 seeds = 15 raw receipts.
    complete_checkpoints = set()
    for ev in events:
        complete_checkpoints.add((ev["checkpoint_id"], ev["operator_id"], ev["replicate_id"]))
    n_cells = len(complete_checkpoints)
    print(f"Total (checkpoint, operator, replicate) cells: {n_cells}")
    if n_cells != 0 and n_cells != 90 * 5 * 3:
        print(f"WARNING: expected {90 * 5 * 3} cells, got {n_cells}")

    averaged = average_replicates(events)
    print(f"Averaged to {len(averaged)} (checkpoint, operator) records")

    complete = keep_complete_checkpoints(averaged)
    n_cp = len(complete) // 5
    print(f"Complete checkpoints: {n_cp}")
    if n_cp != 0 and n_cp != 90:
        print(f"WARNING: expected 90 complete checkpoints, got {n_cp}")

    analyze_q4(events)
    analyze_q5(averaged)
    _bootstrap_ceilings(averaged)


if __name__ == "__main__":
    main()
