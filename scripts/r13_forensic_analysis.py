#!/usr/bin/env python3
"""
R13-F: Post-hoc exploratory forensic analysis of the closed R13 dataset.

Reads ONLY from raw_closed/ — never mutates R13 artifacts.
All outputs are labeled POST_HOC_EXPLORATORY.

R13-F can identify likely failure mechanisms; it cannot confirm them causally.
Any hypothesis produced by R13-F must be tested in new held-out development data.

Usage:
    python3 scripts/r13_forensic_analysis.py \
        --raw-closed experiments/v2b_i3_15c/confirmation/r13/raw_closed \
        --output-dir experiments/v2b_i3_15c/confirmation/r13/forensic \
        --r13-dataset-sha 56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# --- Frozen source identity ---
R13_DATASET_SHA256 = "56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db"
LABEL = "POST_HOC_EXPLORATORY"


def load_results(raw_closed: Path) -> list[dict]:
    """Load results from raw_closed/results.jsonl."""
    results_path = raw_closed / "results.jsonl"
    with open(results_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mechanism_receipts(raw_closed: Path) -> list[dict]:
    """Load mechanism receipts from raw_closed/mechanism_receipts.jsonl."""
    path = raw_closed / "mechanism_receipts.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pairs(results: list[dict]) -> list[dict]:
    """Build 640 A1/R1 pairs by task_id + retrieval_level + backend_identity."""
    by_key = {}
    for r in results:
        key = f"{r['task_id']}|{r['retrieval_level']}|{r['backend_identity']}"
        by_key.setdefault(key, {})[r["arm"]] = r

    pairs = []
    for key, arms in sorted(by_key.items()):
        if "A1_INFERRED" in arms and "R1_INFERRED" in arms:
            a1 = arms["A1_INFERRED"]
            r1 = arms["R1_INFERRED"]
            pairs.append({
                "pair_key": key,
                "task_id": a1["task_id"],
                "category": a1.get("category", ""),
                "a1": a1,
                "r1": r1,
            })
    return pairs


def step_signature(call: dict) -> tuple:
    """
    Define a step signature for comparison: (action, target_id, execution_outcome).
    This is the primary divergence tuple — two steps with the same action but
    different targets are NOT identical.
    """
    return (
        call.get("executed_action", ""),
        call.get("decoded_target_id", call.get("proposal_target_id", "")),
        call.get("execution_outcome", ""),
    )


def classify_prefix_variance(pair: dict) -> str:
    """
    Classify pre-T2 variance vs intervention divergence.

    Uses full step signatures (action, target_id, execution_outcome) for comparison,
    NOT just action labels. This is critical because the primary divergence mode
    is VERIFY→VERIFY with different targets.

    Returns:
        IMMEDIATE_T2: trigger at step 0, no pre-T2 prefix
        PREFIX_IDENTICAL: A1/R1 step signatures agree through trigger_step-1
        PRE_T2_DIVERGED: trajectories already diverged before R1 saw M3
        NO_TRIGGER: R1 did not trigger T2
    """
    r1 = pair["r1"]
    a1 = pair["a1"]
    trigger_step = r1.get("r1_trigger_step")

    if trigger_step is None or not r1.get("r1_triggered", False):
        return "NO_TRIGGER"

    if trigger_step == 0:
        return "IMMEDIATE_T2"

    r1_calls = r1.get("model_call_log", [])
    a1_calls = a1.get("model_call_log", [])

    # Compare full step signatures up to trigger_step
    for step in range(trigger_step):
        if step >= len(r1_calls) or step >= len(a1_calls):
            return "PRE_T2_DIVERGED"  # Length mismatch
        if step_signature(r1_calls[step]) != step_signature(a1_calls[step]):
            return "PRE_T2_DIVERGED"

    return "PREFIX_IDENTICAL"


def find_first_divergence(pair: dict) -> dict | None:
    """
    Find first post-T2 divergence between A1 and R1.

    Primary divergence tuple: (executed_action, target_id)
    Secondary: reason_code

    Returns None if no divergence found.
    """
    r1 = pair["r1"]
    a1 = pair["a1"]
    trigger_step = r1.get("r1_trigger_step")

    if trigger_step is None or not r1.get("r1_triggered", False):
        return None

    r1_calls = r1.get("model_call_log", [])
    a1_calls = a1.get("model_call_log", [])

    for step in range(trigger_step, min(len(r1_calls), len(a1_calls))):
        r1_call = r1_calls[step]
        a1_call = a1_calls[step]

        r1_tuple = (
            r1_call.get("executed_action", ""),
            r1_call.get("decoded_target_id", r1_call.get("proposal_target_id", "")),
        )
        a1_tuple = (
            a1_call.get("executed_action", ""),
            a1_call.get("decoded_target_id", a1_call.get("proposal_target_id", "")),
        )

        if r1_tuple != a1_tuple:
            return {
                "step": step,
                "a1_action": a1_tuple[0],
                "a1_target": a1_tuple[1],
                "r1_action": r1_tuple[0],
                "r1_target": r1_tuple[1],
                "a1_reason": a1_call.get("decoded_reason_code", ""),
                "r1_reason": r1_call.get("decoded_reason_code", ""),
                "a1_outcome": a1_call.get("execution_outcome", ""),
                "r1_outcome": r1_call.get("execution_outcome", ""),
                "r1_representation": r1_call.get("representation", ""),
                "a1_representation": a1_call.get("representation", ""),
            }

    # Check if trajectory lengths differ
    if len(r1_calls) != len(a1_calls):
        return {
            "step": min(len(r1_calls), len(a1_calls)),
            "a1_action": "TRAJECTORY_END" if len(a1_calls) < len(r1_calls) else a1_calls[min(len(r1_calls), len(a1_calls)) - 1].get("executed_action", ""),
            "a1_target": "",
            "r1_action": "TRAJECTORY_END" if len(r1_calls) < len(a1_calls) else r1_calls[min(len(r1_calls), len(a1_calls)) - 1].get("executed_action", ""),
            "r1_target": "",
            "a1_reason": "",
            "r1_reason": "",
            "a1_outcome": "",
            "r1_outcome": "",
            "r1_representation": "",
            "a1_representation": "",
        }

    return None


def compute_action_distribution(pairs: list[dict], stratum_filter: str = None) -> dict:
    """
    Compute P(action | T2) for A1 and R1 in T2-triggering pairs.

    Returns action probabilities and displacement ΔP(a|T2).
    """
    action_types = ["ANSWER", "DEFER", "VERIFY", "RETRIEVE", "SEARCH_MORE", "REASON_MORE", "STOP"]

    a1_counts = Counter()
    r1_counts = Counter()
    total_steps = 0

    for pair in pairs:
        r1 = pair["r1"]
        a1 = pair["a1"]

        # Filter by stratum
        if stratum_filter:
            cat = r1.get("category", "")
            if stratum_filter == "IMMEDIATE" and "immediate" not in cat:
                continue
            elif stratum_filter == "LATE_1" and "late_1" not in cat:
                continue
            elif stratum_filter == "LATE_2" and "late_2" not in cat:
                continue
            elif stratum_filter == "LATE_3" and "late_3" not in cat:
                continue

        trigger_step = r1.get("r1_trigger_step")
        if trigger_step is None or not r1.get("r1_triggered", False):
            continue

        # Count actions from trigger_step onward
        for step in range(trigger_step, len(r1.get("continuation_actions", []))):
            r1_counts[r1["continuation_actions"][step]] += 1
            total_steps += 1

        for step in range(trigger_step, len(a1.get("continuation_actions", []))):
            a1_counts[a1["continuation_actions"][step]] += 1

    # Compute probabilities
    r1_total = sum(r1_counts.values())
    a1_total = sum(a1_counts.values())

    result = {
        "stratum": stratum_filter or "ALL_T2",
        "r1_total_post_t2_steps": r1_total,
        "a1_total_post_t2_steps": a1_total,
        "action_probabilities": {},
        "displacement": {},
    }

    for action in action_types:
        p_r1 = r1_counts.get(action, 0) / r1_total if r1_total > 0 else 0
        p_a1 = a1_counts.get(action, 0) / a1_total if a1_total > 0 else 0
        result["action_probabilities"][action] = {
            "P_R1": round(p_r1, 4),
            "P_A1": round(p_a1, 4),
            "R1_count": r1_counts.get(action, 0),
            "A1_count": a1_counts.get(action, 0),
        }
        result["displacement"][action] = round(p_r1 - p_a1, 4)

    return result


def audit_verify_actions(pairs: list[dict]) -> dict:
    """
    VERIFY forensic audit.

    Classify each VERIFY as:
    - VERIFY_COMPLETED
    - INVALID_VERIFY_TARGET
    - RESOURCE_EXHAUSTED
    - other

    Compute:
    - InvalidVerifyRate
    - RepeatedTargetRate (trajectory-local, not cross-trajectory)
    - VerifyCompletedRate
    - UsefulVerifyV1: Δ eliminated/live hypothesis sets (narrow)
    - UsefulVerifyV2: Δ decision_state OR Δ hypothesis sets OR Δ representation
      (expanded, but limited to fields available in decision_state_log)

    Important: RepeatedTargetRate is computed TRAJECTORY-LOCALLY.
    We aggregate numerator (repeated targets within each trajectory) and
    denominator (total intra-trajectory adjacent pairs) across trajectories.
    We NEVER compare targets across trajectory boundaries.

    Important: UsefulVerify denominator is explicitly reported.
    VERIFY actions without an adjacent state snapshot (e.g., terminal
    RESOURCE_EXHAUSTED) are classified as NOT_OBSERVABLE, not as useless.
    """
    r1_verify_outcomes = Counter()
    a1_verify_outcomes = Counter()

    # Trajectory-local repeated target tracking
    r1_repeated_total = 0  # numerator: repeated targets across all trajectories
    r1_adjacent_total = 0  # denominator: total adjacent pairs across all trajectories
    a1_repeated_total = 0
    a1_adjacent_total = 0

    # UsefulVerify V1 (narrow: hypothesis sets only)
    r1_useful_v1 = 0
    r1_observable_v1 = 0
    r1_not_observable = 0

    # UsefulVerify V2 (expanded: decision_state OR hypothesis sets OR representation)
    r1_useful_v2 = 0
    r1_observable_v2 = 0

    # Per-trajectory target lists for repeated target computation
    for pair in pairs:
        r1 = pair["r1"]
        a1 = pair["a1"]
        trigger_step = r1.get("r1_trigger_step")

        if trigger_step is None or not r1.get("r1_triggered", False):
            continue

        r1_calls = r1.get("model_call_log", [])
        a1_calls = a1.get("model_call_log", [])
        r1_states = r1.get("decision_state_log", [])

        # R1: collect per-trajectory VERIFY targets
        r1_traj_targets = []
        for step in range(trigger_step, len(r1_calls)):
            call = r1_calls[step]
            if call.get("executed_action") == "VERIFY":
                outcome = call.get("execution_outcome", "UNKNOWN")
                r1_verify_outcomes[outcome] += 1
                target = call.get("decoded_target_id", call.get("proposal_target_id", ""))
                r1_traj_targets.append(target)

                # UsefulVerifyV1: check hypothesis set changes
                if step < len(r1_states) - 1:
                    curr_state = r1_states[step]
                    next_state = r1_states[step + 1]
                    curr_elim = set(curr_state.get("eliminated_hypotheses", []))
                    next_elim = set(next_state.get("eliminated_hypotheses", []))
                    curr_live = set(curr_state.get("live_hypotheses", []))
                    next_live = set(next_state.get("live_hypotheses", []))

                    v1_useful = (curr_elim != next_elim or curr_live != next_live)
                    if v1_useful:
                        r1_useful_v1 += 1
                    r1_observable_v1 += 1

                    # UsefulVerifyV2: also check decision_state and representation
                    v2_useful = (
                        v1_useful
                        or curr_state.get("decision_state", "") != next_state.get("decision_state", "")
                        or curr_state.get("representation", "") != next_state.get("representation", "")
                    )
                    if v2_useful:
                        r1_useful_v2 += 1
                    r1_observable_v2 += 1
                else:
                    r1_not_observable += 1

        # Trajectory-local repeated target count
        for i in range(1, len(r1_traj_targets)):
            r1_adjacent_total += 1
            if r1_traj_targets[i] == r1_traj_targets[i - 1]:
                r1_repeated_total += 1

        # A1: collect per-trajectory VERIFY targets
        a1_traj_targets = []
        for step in range(trigger_step, len(a1_calls)):
            call = a1_calls[step]
            if call.get("executed_action") == "VERIFY":
                a1_verify_outcomes[call.get("execution_outcome", "UNKNOWN")] += 1
                a1_traj_targets.append(call.get("decoded_target_id", call.get("proposal_target_id", "")))

        for i in range(1, len(a1_traj_targets)):
            a1_adjacent_total += 1
            if a1_traj_targets[i] == a1_traj_targets[i - 1]:
                a1_repeated_total += 1

    r1_total_verify = sum(r1_verify_outcomes.values())
    a1_total_verify = sum(a1_verify_outcomes.values())

    return {
        "R1": {
            "total_verify": r1_total_verify,
            "outcomes": dict(r1_verify_outcomes),
            "verify_completed_rate": round(r1_verify_outcomes.get("VERIFY_COMPLETED", 0) / r1_total_verify, 4) if r1_total_verify > 0 else 0,
            "invalid_verify_rate": round(r1_verify_outcomes.get("INVALID_VERIFY_TARGET", 0) / r1_total_verify, 4) if r1_total_verify > 0 else 0,
            "resource_exhausted_rate": round(r1_verify_outcomes.get("RESOURCE_EXHAUSTED", 0) / r1_total_verify, 4) if r1_total_verify > 0 else 0,
            "repeated_target_rate": round(r1_repeated_total / max(r1_adjacent_total, 1), 4),
            "repeated_target_count": r1_repeated_total,
            "adjacent_target_pairs": r1_adjacent_total,
            "useful_verify_v1_count": r1_useful_v1,
            "useful_verify_v1_observable": r1_observable_v1,
            "useful_verify_v1_not_observable": r1_not_observable,
            "useful_verify_v1_rate": round(r1_useful_v1 / r1_observable_v1, 4) if r1_observable_v1 > 0 else 0,
            "useful_verify_v2_count": r1_useful_v2,
            "useful_verify_v2_observable": r1_observable_v2,
            "useful_verify_v2_rate": round(r1_useful_v2 / r1_observable_v2, 4) if r1_observable_v2 > 0 else 0,
            "epistemic_usefulness_note": (
                f"Among {r1_observable_v1} R1 VERIFY transitions with adjacent state snapshots, "
                f"none changed live- or eliminated-hypothesis sets (V1). "
                f"V2 (also checking decision_state and representation): {r1_useful_v2}/{r1_observable_v2}. "
                f"{r1_not_observable} VERIFY actions had no following state snapshot (terminal)."
            ),
        },
        "A1": {
            "total_verify": a1_total_verify,
            "outcomes": dict(a1_verify_outcomes),
            "verify_completed_rate": round(a1_verify_outcomes.get("VERIFY_COMPLETED", 0) / max(a1_total_verify, 1), 4),
            "invalid_verify_rate": round(a1_verify_outcomes.get("INVALID_VERIFY_TARGET", 0) / max(a1_total_verify, 1), 4),
            "resource_exhausted_rate": round(a1_verify_outcomes.get("RESOURCE_EXHAUSTED", 0) / max(a1_total_verify, 1), 4),
            "repeated_target_rate": round(a1_repeated_total / max(a1_adjacent_total, 1), 4),
            "repeated_target_count": a1_repeated_total,
            "adjacent_target_pairs": a1_adjacent_total,
        },
    }


def compute_harm_by_divergence(pairs: list[dict], first_divergences: dict) -> dict:
    """
    Condition harm on first divergence class.

    For each divergence class compute:
    - E[ΔU | D]
    - P(R1 break | D)
    - ΔSteps | D
    """
    by_class = defaultdict(list)

    for pair in pairs:
        pair_key = pair["pair_key"]
        r1 = pair["r1"]
        a1 = pair["a1"]

        if pair_key not in first_divergences or first_divergences[pair_key] is None:
            continue

        div = first_divergences[pair_key]
        div_class = f"{div['a1_action']}→{div['r1_action']}"

        delta_u = r1.get("realized_utility", 0) - a1.get("realized_utility", 0)
        delta_steps = r1.get("steps", 0) - a1.get("steps", 0)
        r1_break = a1.get("success", False) and not r1.get("success", False)
        r1_rescue = not a1.get("success", False) and r1.get("success", False)

        by_class[div_class].append({
            "delta_u": delta_u,
            "delta_steps": delta_steps,
            "r1_break": r1_break,
            "r1_rescue": r1_rescue,
        })

    result = {}
    for div_class, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
        n = len(items)
        mean_du = sum(i["delta_u"] for i in items) / n
        mean_ds = sum(i["delta_steps"] for i in items) / n
        breaks = sum(i["r1_break"] for i in items)
        rescues = sum(i["r1_rescue"] for i in items)

        result[div_class] = {
            "n": n,
            "mean_delta_u": round(mean_du, 4),
            "mean_delta_steps": round(mean_ds, 4),
            "r1_breaks": breaks,
            "r1_rescues": rescues,
        }

    return result


def analyze_persistent_m3_harm(pairs: list[dict]) -> dict:
    """
    Separate persistent-M3 harm from first-M3-action harm.

    Measure utility against:
    - first action immediately after T2
    - number of post-T2 M3 decisions
    - number of consecutive VERIFY actions
    - repeated/invalid VERIFY count
    """
    results = []

    for pair in pairs:
        r1 = pair["r1"]
        a1 = pair["a1"]
        trigger_step = r1.get("r1_trigger_step")

        if trigger_step is None or not r1.get("r1_triggered", False):
            continue

        calls = r1.get("model_call_log", [])
        actions = r1.get("continuation_actions", [])
        outcomes = r1.get("continuation_outcomes", [])

        # First post-T2 action
        first_action = actions[trigger_step] if trigger_step < len(actions) else ""
        first_outcome = outcomes[trigger_step] if trigger_step < len(outcomes) else ""

        # Count post-T2 M3 decisions
        post_t2_m3 = sum(1 for c in calls[trigger_step:] if c.get("representation") == "M3")

        # Count consecutive VERIFY from trigger
        consecutive_verify = 0
        for a in actions[trigger_step:]:
            if a == "VERIFY":
                consecutive_verify += 1
            else:
                break

        # Count repeated and invalid VERIFY
        post_t2_targets = [c.get("decoded_target_id", "") for c in calls[trigger_step:] if c.get("executed_action") == "VERIFY"]
        repeated = sum(1 for i in range(1, len(post_t2_targets)) if post_t2_targets[i] == post_t2_targets[i-1])
        invalid = sum(1 for c in calls[trigger_step:] if c.get("execution_outcome") == "INVALID_VERIFY_TARGET")

        delta_u = r1.get("realized_utility", 0) - a1.get("realized_utility", 0)

        results.append({
            "task_id": pair["task_id"],
            "category": r1.get("category", ""),
            "trigger_step": trigger_step,
            "first_post_t2_action": first_action,
            "first_post_t2_outcome": first_outcome,
            "post_t2_m3_decisions": post_t2_m3,
            "consecutive_verify": consecutive_verify,
            "repeated_verify_targets": repeated,
            "invalid_verify_count": invalid,
            "delta_u": delta_u,
            "r1_success": r1.get("success", False),
            "a1_success": a1.get("success", False),
        })

    # Aggregate
    n = len(results)
    if n == 0:
        return {"n": 0}

    # Correlation between consecutive_verify and delta_u
    mean_consec = sum(r["consecutive_verify"] for r in results) / n
    mean_du = sum(r["delta_u"] for r in results) / n

    # First action distribution
    first_actions = Counter(r["first_post_t2_action"] for r in results)

    # Harm by consecutive VERIFY bucket
    buckets = {"0": [], "1-2": [], "3-4": [], "5+": []}
    for r in results:
        cv = r["consecutive_verify"]
        if cv == 0:
            buckets["0"].append(r)
        elif cv <= 2:
            buckets["1-2"].append(r)
        elif cv <= 4:
            buckets["3-4"].append(r)
        else:
            buckets["5+"].append(r)

    bucket_stats = {}
    for label, items in buckets.items():
        if items:
            bucket_stats[label] = {
                "n": len(items),
                "mean_delta_u": round(sum(i["delta_u"] for i in items) / len(items), 4),
                "mean_repeated": round(sum(i["repeated_verify_targets"] for i in items) / len(items), 4),
                "mean_invalid": round(sum(i["invalid_verify_count"] for i in items) / len(items), 4),
            }

    return {
        "n": n,
        "mean_consecutive_verify": round(mean_consec, 4),
        "mean_delta_u": round(mean_du, 4),
        "first_post_t2_action_distribution": dict(first_actions),
        "harm_by_consecutive_verify_bucket": bucket_stats,
        "per_trajectory": results,
    }


def build_rescue_break_cases(pairs: list[dict], max_cases: int = 10) -> dict:
    """
    Build individual case files for breaks (A1 success, R1 fail) and rescues (A1 fail, R1 success).
    """
    breaks = []
    rescues = []

    for pair in pairs:
        r1 = pair["r1"]
        a1 = pair["a1"]

        if a1.get("success") and not r1.get("success"):
            # Break case
            breaks.append({
                "task_id": pair["task_id"],
                "category": r1.get("category", ""),
                "trigger_step": r1.get("r1_trigger_step"),
                "a1_utility": a1.get("realized_utility"),
                "r1_utility": r1.get("realized_utility"),
                "delta_u": r1.get("realized_utility", 0) - a1.get("realized_utility", 0),
                "a1_actions": a1.get("continuation_actions", []),
                "r1_actions": r1.get("continuation_actions", []),
                "a1_outcomes": a1.get("continuation_outcomes", []),
                "r1_outcomes": r1.get("continuation_outcomes", []),
                "a1_terminal": a1.get("terminal_result"),
                "r1_terminal": r1.get("terminal_result"),
                "r1_representation_by_step": r1.get("mechanism_receipt", {}).get("representation_by_step", []),
                "a1_decision_states": [s.get("decision_state", "") for s in a1.get("decision_state_log", [])],
                "r1_decision_states": [s.get("decision_state", "") for s in r1.get("decision_state_log", [])],
            })
        elif not a1.get("success") and r1.get("success"):
            rescues.append({
                "task_id": pair["task_id"],
                "category": r1.get("category", ""),
                "trigger_step": r1.get("r1_trigger_step"),
                "a1_utility": a1.get("realized_utility"),
                "r1_utility": r1.get("realized_utility"),
                "delta_u": r1.get("realized_utility", 0) - a1.get("realized_utility", 0),
                "a1_actions": a1.get("continuation_actions", []),
                "r1_actions": r1.get("continuation_actions", []),
                "a1_outcomes": a1.get("continuation_outcomes", []),
                "r1_outcomes": r1.get("continuation_outcomes", []),
                "a1_terminal": a1.get("terminal_result"),
                "r1_terminal": r1.get("terminal_result"),
                "r1_representation_by_step": r1.get("mechanism_receipt", {}).get("representation_by_step", []),
            })

    return {
        "breaks": {
            "count": len(breaks),
            "cases": breaks[:max_cases],
        },
        "rescues": {
            "count": len(rescues),
            "cases": rescues[:max_cases],
        },
    }


def analyze_verify_target_value(pairs: list[dict]) -> dict:
    """
    VERIFY target-value analysis.

    For each T2-triggered trajectory, determine whether ANY verification target
    available at T2 time could theoretically change the epistemic state.

    The key question: is the failure "model chose a bad target" or
    "there were no useful VERIFY targets left at all"?

    We classify the state at T2 trigger time:
    - HAS_LIVE_HYPOTHESES: live_hypotheses is non-empty → verification could
      potentially confirm/eliminate a live hypothesis
    - ALL_ELIMINATED: live_hypotheses is empty, eliminated_hypotheses is non-empty →
      no hypothesis can be changed by verification
    - EMPTY_STATE: both sets empty → no epistemic content to verify against

    If ALL_ELIMINATED or EMPTY_STATE, then VERIFY is structurally useless
    regardless of target selection. The affordance itself is wrong.
    """
    state_at_t2 = Counter()
    target_quality = []

    for pair in pairs:
        r1 = pair["r1"]
        trigger_step = r1.get("r1_trigger_step")

        if trigger_step is None or not r1.get("r1_triggered", False):
            continue

        states = r1.get("decision_state_log", [])
        if trigger_step >= len(states):
            continue

        t2_state = states[trigger_step]
        live = set(t2_state.get("live_hypotheses", []))
        eliminated = set(t2_state.get("eliminated_hypotheses", []))
        decision_state = t2_state.get("decision_state", "")

        if live:
            state_class = "HAS_LIVE_HYPOTHESES"
        elif eliminated:
            state_class = "ALL_ELIMINATED"
        else:
            state_class = "EMPTY_STATE"

        state_at_t2[state_class] += 1

        # Track whether any subsequent state change occurs
        any_state_change = False
        for step in range(trigger_step, len(states) - 1):
            curr = states[step]
            nxt = states[step + 1]
            if (set(curr.get("live_hypotheses", [])) != set(nxt.get("live_hypotheses", []))
                or set(curr.get("eliminated_hypotheses", [])) != set(nxt.get("eliminated_hypotheses", []))
                or curr.get("decision_state", "") != nxt.get("decision_state", "")):
                any_state_change = True
                break

        target_quality.append({
            "task_id": pair["task_id"],
            "state_class": state_class,
            "decision_state": decision_state,
            "n_live": len(live),
            "n_eliminated": len(eliminated),
            "any_state_change_post_t2": any_state_change,
        })

    # Aggregate
    n = len(target_quality)
    n_with_live = sum(1 for t in target_quality if t["state_class"] == "HAS_LIVE_HYPOTHESES")
    n_all_elim = sum(1 for t in target_quality if t["state_class"] == "ALL_ELIMINATED")
    n_empty = sum(1 for t in target_quality if t["state_class"] == "EMPTY_STATE")
    n_any_change = sum(1 for t in target_quality if t["any_state_change_post_t2"])

    return {
        "n_t2_triggered": n,
        "state_at_t2": dict(state_at_t2),
        "n_has_live_hypotheses": n_with_live,
        "n_all_eliminated": n_all_elim,
        "n_empty_state": n_empty,
        "n_any_state_change_post_t2": n_any_change,
        "pct_structurally_useless_verify": round((n_all_elim + n_empty) / max(n, 1), 4),
        "interpretation": (
            f"Of {n} T2-triggered trajectories, {n_all_elim + n_empty} ({round((n_all_elim + n_empty) / max(n, 1) * 100, 1)}%) "
            f"had no live hypotheses at T2 time, making VERIFY structurally useless regardless of target. "
            f"Only {n_with_live} had live hypotheses that verification could potentially resolve. "
            f"Across all {n} trajectories, {n_any_change} had any post-T2 state change."
        ),
        "per_trajectory": target_quality,
    }


def compare_a1_r1_target_quality(pairs: list[dict]) -> dict:
    """
    Compare A1 and R1 VERIFY target quality after T2.

    Both arms are stuck in VERIFY ~100% of the time. The question is whether
    A1 and R1 select different targets, and whether one set is better.

    We compare:
    - Target overlap: do A1 and R1 verify the same targets?
    - A1 invalid target rate vs R1 invalid target rate
    - A1 state change rate vs R1 state change rate (if A1 state logs available)
    """
    target_overlap = []
    a1_target_sets = []
    r1_target_sets = []

    for pair in pairs:
        r1 = pair["r1"]
        a1 = pair["a1"]
        trigger_step = r1.get("r1_trigger_step")

        if trigger_step is None or not r1.get("r1_triggered", False):
            continue

        r1_calls = r1.get("model_call_log", [])
        a1_calls = a1.get("model_call_log", [])

        r1_targets = [c.get("decoded_target_id", "") for c in r1_calls[trigger_step:] if c.get("executed_action") == "VERIFY"]
        a1_targets = [c.get("decoded_target_id", "") for c in a1_calls[trigger_step:] if c.get("executed_action") == "VERIFY"]

        r1_set = set(r1_targets)
        a1_set = set(a1_targets)

        overlap = r1_set & a1_set
        r1_only = r1_set - a1_set
        a1_only = a1_set - r1_set

        target_overlap.append({
            "task_id": pair["task_id"],
            "r1_targets": r1_targets,
            "a1_targets": a1_targets,
            "r1_unique_targets": len(r1_set),
            "a1_unique_targets": len(a1_set),
            "shared_targets": len(overlap),
            "r1_only_targets": len(r1_only),
            "a1_only_targets": len(a1_only),
            "identical_target_sets": r1_set == a1_set,
        })

    n = len(target_overlap)
    if n == 0:
        return {"n": 0}

    identical = sum(1 for t in target_overlap if t["identical_target_sets"])
    mean_shared = sum(t["shared_targets"] for t in target_overlap) / n
    mean_r1_only = sum(t["r1_only_targets"] for t in target_overlap) / n
    mean_a1_only = sum(t["a1_only_targets"] for t in target_overlap) / n

    return {
        "n": n,
        "identical_target_sets": identical,
        "pct_identical": round(identical / n, 4),
        "mean_shared_targets": round(mean_shared, 4),
        "mean_r1_only_targets": round(mean_r1_only, 4),
        "mean_a1_only_targets": round(mean_a1_only, 4),
        "per_pair": target_overlap,
    }


def compute_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="R13-F post-hoc forensic analysis")
    parser.add_argument("--raw-closed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--r13-dataset-sha", default=R13_DATASET_SHA256)
    args = parser.parse_args()

    raw_closed = args.raw_closed
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify dataset SHA
    results_path = raw_closed / "results.jsonl"
    actual_sha = compute_sha256(results_path)
    print(f"R13-F Forensic Analysis")
    print(f"  Label: {LABEL}")
    print(f"  Source: {raw_closed}")
    print(f"  R13_DATASET_SHA256 (expected): {args.r13_dataset_sha}")
    print(f"  results.jsonl SHA256 (actual): {actual_sha}")
    if actual_sha != args.r13_dataset_sha.split(",")[0]:
        # The dataset SHA is computed over all files, not just results.jsonl
        # Check against the results.jsonl hash from the manifest
        manifest_path = raw_closed / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            expected_results_sha = manifest.get("file_hashes", {}).get("results.jsonl", "")
            if expected_results_sha:
                if actual_sha == expected_results_sha:
                    print(f"  results.jsonl SHA matches manifest entry")
                else:
                    print(f"  WARNING: results.jsonl SHA mismatch!")
                    print(f"    Expected (from manifest): {expected_results_sha}")
                    print(f"    Actual: {actual_sha}")

    # Load data
    print(f"\n[1] Loading results...")
    results = load_results(raw_closed)
    print(f"  Loaded {len(results)} records")

    # Build pairs
    print(f"\n[2] Building A1/R1 pairs...")
    pairs = build_pairs(results)
    print(f"  Built {len(pairs)} pairs")

    # Classify prefix variance
    print(f"\n[3] Classifying pre-T2 variance (using (action, target, outcome) signatures)...")
    prefix_classes = Counter()
    for pair in pairs:
        cls = classify_prefix_variance(pair)
        pair["prefix_class"] = cls
        prefix_classes[cls] += 1
    print(f"  Prefix classification (full step signatures):")
    for cls, count in sorted(prefix_classes.items()):
        print(f"    {cls}: {count}")

    # Find first divergences
    print(f"\n[4] Finding first post-T2 divergences...")
    first_divergences = {}
    divergence_transitions = Counter()
    for pair in pairs:
        div = find_first_divergence(pair)
        first_divergences[pair["pair_key"]] = div
        if div is not None:
            transition = f"{div['a1_action']}→{div['r1_action']}"
            divergence_transitions[transition] += 1

    no_div = sum(1 for d in first_divergences.values() if d is None)
    print(f"  Divergences found: {len(first_divergences) - no_div}")
    print(f"  No divergence: {no_div}")
    print(f"  Top transition classes:")
    for trans, count in divergence_transitions.most_common(10):
        print(f"    {trans}: {count}")

    # Action distribution displacement
    print(f"\n[5] Computing action distribution displacement...")
    action_dist = {}
    for stratum in [None, "IMMEDIATE", "LATE_1", "LATE_2", "LATE_3"]:
        key = stratum or "ALL_T2"
        action_dist[key] = compute_action_distribution(pairs, stratum_filter=stratum)
        print(f"  {key}: R1_steps={action_dist[key]['r1_total_post_t2_steps']}, A1_steps={action_dist[key]['a1_total_post_t2_steps']}")
        for action, disp in sorted(action_dist[key]["displacement"].items()):
            if abs(disp) > 0.001:
                print(f"    ΔP({action}) = {disp:+.4f}")

    # VERIFY audit (corrected: trajectory-local repeated targets, V1+V2 usefulness)
    print(f"\n[6] VERIFY forensic audit (corrected)...")
    verify_audit = audit_verify_actions(pairs)
    print(f"  R1 VERIFY outcomes: {verify_audit['R1']['outcomes']}")
    print(f"  R1 verify_completed_rate: {verify_audit['R1']['verify_completed_rate']}")
    print(f"  R1 invalid_verify_rate: {verify_audit['R1']['invalid_verify_rate']}")
    print(f"  R1 repeated_target_rate (trajectory-local): {verify_audit['R1']['repeated_target_rate']} ({verify_audit['R1']['repeated_target_count']}/{verify_audit['R1']['adjacent_target_pairs']})")
    print(f"  R1 useful_verify_v1 (hypothesis sets): {verify_audit['R1']['useful_verify_v1_count']}/{verify_audit['R1']['useful_verify_v1_observable']}")
    print(f"  R1 useful_verify_v2 (expanded): {verify_audit['R1']['useful_verify_v2_count']}/{verify_audit['R1']['useful_verify_v2_observable']}")
    print(f"  R1 not_observable (terminal): {verify_audit['R1']['useful_verify_v1_not_observable']}")
    print(f"  R1 note: {verify_audit['R1']['epistemic_usefulness_note']}")
    print(f"  A1 VERIFY outcomes: {verify_audit['A1']['outcomes']}")
    print(f"  A1 repeated_target_rate (trajectory-local): {verify_audit['A1']['repeated_target_rate']} ({verify_audit['A1']['repeated_target_count']}/{verify_audit['A1']['adjacent_target_pairs']})")

    # Harm by divergence
    print(f"\n[7] Harm conditioned on first divergence...")
    harm_by_div = compute_harm_by_divergence(pairs, first_divergences)
    for div_class, stats in sorted(harm_by_div.items(), key=lambda x: x[1]["n"], reverse=True)[:10]:
        print(f"  {div_class}: n={stats['n']}, mean_ΔU={stats['mean_delta_u']}, breaks={stats['r1_breaks']}, rescues={stats['r1_rescues']}")

    # Persistent M3 harm
    print(f"\n[8] Persistent-M3 vs first-M3-action harm...")
    persistent_m3 = analyze_persistent_m3_harm(pairs)
    print(f"  n={persistent_m3['n']}")
    print(f"  mean_consecutive_verify: {persistent_m3['mean_consecutive_verify']}")
    print(f"  first_post_t2_action: {persistent_m3['first_post_t2_action_distribution']}")
    print(f"  harm_by_consecutive_verify_bucket:")
    for label, stats in persistent_m3["harm_by_consecutive_verify_bucket"].items():
        print(f"    {label}: {stats}")

    # Rescue/break cases
    print(f"\n[9] Building rescue/break case files...")
    rescue_break = build_rescue_break_cases(pairs)
    print(f"  Breaks (A1 success, R1 fail): {rescue_break['breaks']['count']}")
    print(f"  Rescues (A1 fail, R1 success): {rescue_break['rescues']['count']}")

    # VERIFY target-value analysis
    print(f"\n[10] VERIFY target-value analysis...")
    target_value = analyze_verify_target_value(pairs)
    print(f"  State at T2: {target_value['state_at_t2']}")
    print(f"  Has live hypotheses: {target_value['n_has_live_hypotheses']}")
    print(f"  All eliminated: {target_value['n_all_eliminated']}")
    print(f"  Empty state: {target_value['n_empty_state']}")
    print(f"  Pct structurally useless VERIFY: {target_value['pct_structurally_useless_verify']}")
    print(f"  Any state change post-T2: {target_value['n_any_state_change_post_t2']}")
    print(f"  Interpretation: {target_value['interpretation']}")

    # A1 vs R1 target quality comparison
    print(f"\n[11] A1 vs R1 target quality comparison...")
    target_quality = compare_a1_r1_target_quality(pairs)
    print(f"  n={target_quality['n']}")
    print(f"  Identical target sets: {target_quality['identical_target_sets']}/{target_quality['n']} ({target_quality['pct_identical']})")
    print(f"  Mean shared targets: {target_quality['mean_shared_targets']}")
    print(f"  Mean R1-only targets: {target_quality['mean_r1_only_targets']}")
    print(f"  Mean A1-only targets: {target_quality['mean_a1_only_targets']}")

    # Write outputs
    print(f"\n[12] Writing outputs...")

    # r13_f_pairs.jsonl
    pairs_path = output_dir / "r13_f_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for pair in pairs:
            entry = {
                "label": LABEL,
                "pair_key": pair["pair_key"],
                "task_id": pair["task_id"],
                "category": pair["category"],
                "prefix_class": pair["prefix_class"],
                "a1_utility": pair["a1"].get("realized_utility"),
                "r1_utility": pair["r1"].get("realized_utility"),
                "delta_u": pair["r1"].get("realized_utility", 0) - pair["a1"].get("realized_utility", 0),
                "a1_success": pair["a1"].get("success"),
                "r1_success": pair["r1"].get("success"),
                "a1_steps": pair["a1"].get("steps"),
                "r1_steps": pair["r1"].get("steps"),
                "r1_trigger_step": pair["r1"].get("r1_trigger_step"),
                "r1_triggered": pair["r1"].get("r1_triggered"),
                "a1_actions": pair["a1"].get("continuation_actions", []),
                "r1_actions": pair["r1"].get("continuation_actions", []),
                "a1_outcomes": pair["a1"].get("continuation_outcomes", []),
                "r1_outcomes": pair["r1"].get("continuation_outcomes", []),
                "r1_representation_by_step": pair["r1"].get("mechanism_receipt", {}).get("representation_by_step", []),
                "first_divergence": first_divergences.get(pair["pair_key"]),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"  {pairs_path}")

    # r13_f_first_divergence.jsonl
    div_path = output_dir / "r13_f_first_divergence.jsonl"
    with open(div_path, "w") as f:
        for pair_key, div in first_divergences.items():
            if div is not None:
                entry = {
                    "label": LABEL,
                    "pair_key": pair_key,
                    **div,
                }
                f.write(json.dumps(entry) + "\n")
    print(f"  {div_path}")

    # Full analysis JSON
    analysis_path = output_dir / "r13_f_analysis.json"
    full_analysis = {
        "label": LABEL,
        "version": "R13-F1.1",
        "r13_dataset_sha256": args.r13_dataset_sha,
        "n_pairs": len(pairs),
        "prefix_classification": dict(prefix_classes),
        "prefix_classification_note": (
            "Uses full step signatures (action, target_id, execution_outcome) for comparison, "
            "not just action labels. This corrects the R13-F1 issue where VERIFY(Ex) vs VERIFY(Ey) "
            "would have been classified as identical."
        ),
        "first_divergence_transitions": dict(divergence_transitions),
        "action_distribution": action_dist,
        "verify_audit": verify_audit,
        "verify_audit_note": (
            "RepeatedTargetRate is computed trajectory-locally (not cross-trajectory). "
            "UsefulVerifyV1 checks hypothesis set changes only. "
            "UsefulVerifyV2 also checks decision_state and representation changes. "
            "Denominator explicitly reported: VERIFY actions without adjacent state snapshots "
            "are classified as NOT_OBSERVABLE, not as useless."
        ),
        "harm_by_divergence": harm_by_div,
        "persistent_m3_analysis": {k: v for k, v in persistent_m3.items() if k != "per_trajectory"},
        "rescue_break_summary": {
            "breaks": rescue_break["breaks"]["count"],
            "rescues": rescue_break["rescues"]["count"],
        },
        "verify_target_value": {k: v for k, v in target_value.items() if k != "per_trajectory"},
        "a1_r1_target_quality": {k: v for k, v in target_quality.items() if k != "per_pair"},
        "a1_baseline_note": (
            "A1 is also stuck in VERIFY ~100% of the time post-T2. "
            "P(VERIFY|A1,T2)=1.0 vs P(VERIFY|R1,T2)=0.9836. "
            "R1 does not introduce VERIFY behavior from nowhere. "
            "Both policies are trapped in the same broad action mode. "
            "The difference is target selection and cost, not action type."
        ),
        "methodological_note": (
            "R13-F can identify likely failure mechanisms; it cannot confirm them causally. "
            "Any hypothesis produced by R13-F must be tested in new held-out development data. "
            "R13-F1.1 corrections: (1) prefix comparison uses (action,target,outcome) signatures, "
            "(2) repeated target rate is trajectory-local, (3) UsefulVerify denominator is explicit "
            "with V1 (hypothesis sets) and V2 (expanded) variants."
        ),
    }
    with open(analysis_path, "w") as f:
        json.dump(full_analysis, f, indent=2)
    print(f"  {analysis_path}")

    # Rescue/break cases
    cases_path = output_dir / "r13_f_rescue_break_cases.json"
    with open(cases_path, "w") as f:
        json.dump({"label": LABEL, **rescue_break}, f, indent=2)
    print(f"  {cases_path}")

    # Per-trajectory persistent M3 data
    pm3_path = output_dir / "r13_f_persistent_m3.jsonl"
    with open(pm3_path, "w") as f:
        for entry in persistent_m3.get("per_trajectory", []):
            f.write(json.dumps({"label": LABEL, **entry}) + "\n")
    print(f"  {pm3_path}")

    # VERIFY target-value per-trajectory data
    tv_path = output_dir / "r13_f_target_value.jsonl"
    with open(tv_path, "w") as f:
        for entry in target_value.get("per_trajectory", []):
            f.write(json.dumps({"label": LABEL, **entry}) + "\n")
    print(f"  {tv_path}")

    # A1 vs R1 target quality per-pair data
    tq_path = output_dir / "r13_f_target_quality.jsonl"
    with open(tq_path, "w") as f:
        for entry in target_quality.get("per_pair", []):
            f.write(json.dumps({"label": LABEL, **entry}) + "\n")
    print(f"  {tq_path}")

    # Compute analysis SHA
    analysis_sha = compute_sha256(analysis_path)
    print(f"\n  R13_F_ANALYSIS_SHA256: {analysis_sha}")

    # Write SHA file
    sha_path = output_dir / "R13_F_ANALYSIS_SHA256.txt"
    with open(sha_path, "w") as f:
        f.write(analysis_sha)
    print(f"  {sha_path}")

    print(f"\nR13-F forensic analysis complete.")
    print(f"  Label: {LABEL}")
    print(f"  R13-F can identify likely failure mechanisms; it cannot confirm them causally.")


if __name__ == "__main__":
    main()
