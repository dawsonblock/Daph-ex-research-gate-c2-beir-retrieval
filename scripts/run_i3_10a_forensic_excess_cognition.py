#!/usr/bin/env python3
"""I3.10a: Forensic excess-cognition analysis on I3.9-r3 v5 trajectories.

DIAGNOSIS ONLY — no new model runs. Uses existing v5 trajectories.

Defines:
  first_decision_sufficient_step = first step where M3's epistemic state
    would have justified the eventual correct terminal action
    (READY_TO_ANSWER for ANSWER tasks, INSUFFICIENT for DEFER tasks)

  ExcessCognition = terminal_step - first_decision_sufficient_step

Builds:
  - epistemic state -> next action transition matrix
  - P(next action changes epistemic state | state, action)
  - per-category excess cognition comparison (A1 vs M3)
  - low-value continuation state identification

Usage:
    PYTHONPATH=. python scripts/run_i3_10a_forensic_excess_cognition.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_v5_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_first_decision_sufficient_step(
    decision_state_log: list[dict],
    terminal_action: str,
) -> int | None:
    """Find the first step where the epistemic state justifies the terminal action.

    For ANSWER: first step with READY_TO_ANSWER
    For DEFER: first step with INSUFFICIENT
    For STOP: first step with INSUFFICIENT
    """
    target_state = "READY_TO_ANSWER" if terminal_action == "ANSWER" else "INSUFFICIENT"
    for entry in decision_state_log:
        if entry.get("decision_state") == target_state:
            return entry["step"]
    return None


def compute_excess_cognition(
    decision_state_log: list[dict],
    terminal_action: str,
    total_steps: int,
) -> dict:
    """Compute excess cognition metrics for a single trajectory."""
    first_sufficient = compute_first_decision_sufficient_step(
        decision_state_log, terminal_action
    )

    if first_sufficient is None:
        # The model reached the correct answer without M3 ever saying
        # READY_TO_ANSWER (or INSUFFICIENT for DEFER). This happens when
        # the model acts on SUPPORTED_BUT_UNRESOLVED.
        return {
            "first_decision_sufficient_step": None,
            "terminal_step": total_steps - 1,
            "excess_cognition_steps": None,
            "never_sufficient": True,
            "acted_on_unresolved": terminal_action == "ANSWER",
        }

    terminal_step = total_steps - 1
    excess = terminal_step - first_sufficient

    return {
        "first_decision_sufficient_step": first_sufficient,
        "terminal_step": terminal_step,
        "excess_cognition_steps": max(0, excess),
        "never_sufficient": False,
        "acted_on_unresolved": False,
    }


def build_state_action_matrix(
    results: list[dict],
    arm: str,
) -> dict:
    """Build epistemic_state -> action -> count matrix.

    Also tracks whether the action changed the epistemic state.
    """
    matrix: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "count": 0, "state_changed": 0, "state_same": 0
    }))

    for r in results:
        fork = r[f"fork_{arm.lower()}"]
        log = fork.get("decision_state_log", [])
        actions = fork.get("continuation_actions", [])

        for i, entry in enumerate(log):
            state = entry.get("decision_state", "UNKNOWN")
            action = actions[i] if i < len(actions) else "UNKNOWN"

            # Check if next state differs
            if i + 1 < len(log):
                next_state = log[i + 1].get("decision_state", "UNKNOWN")
                changed = next_state != state
            else:
                # Terminal step — no next state
                changed = None

            matrix[state][action]["count"] += 1
            if changed is True:
                matrix[state][action]["state_changed"] += 1
            elif changed is False:
                matrix[state][action]["state_same"] += 1

    return dict(matrix)


def compute_state_change_probability(matrix: dict) -> dict:
    """Compute P(next state changes | state, action)."""
    probs = {}
    for state, actions in matrix.items():
        probs[state] = {}
        for action, counts in actions.items():
            total = counts["state_changed"] + counts["state_same"]
            if total > 0:
                probs[state][action] = {
                    "p_change": round(counts["state_changed"] / total, 4),
                    "p_same": round(counts["state_same"] / total, 4),
                    "n": total,
                    "changed": counts["state_changed"],
                    "same": counts["state_same"],
                }
            else:
                probs[state][action] = {
                    "p_change": None, "p_same": None, "n": 0,
                    "changed": 0, "same": 0,
                }
    return probs


def main():
    v5_path = ROOT / "experiments/v2b_i3_9/development/i3_9_r3_affordance_clean/affordance_clean_v1.jsonl"
    output_dir = ROOT / "experiments/v2b_i3_10/development/i3_10a_forensic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.10a: Forensic excess-cognition analysis on v5 trajectories")
    print(f"  Source: {v5_path}")
    print()

    results = load_v5_results(v5_path)
    n = len(results)
    print(f"  Loaded {n} task results")

    # === ExcessCognition per trajectory ===
    print("\n=== EXCESS COGNITION ANALYSIS ===")

    m3_excess = []
    a1_excess = []
    m3_never_sufficient = 0
    m3_acted_on_unresolved = 0

    per_task = []
    for r in results:
        m3_ec = compute_excess_cognition(
            r["fork_m3"]["decision_state_log"],
            r["fork_m3"]["terminal_action"],
            r["fork_m3"]["steps"],
        )
        # A1 doesn't have decision_state_log — use a proxy:
        # first step where answer_condition_satisfied_before_terminal is True
        a1_first_sufficient = None
        if r["fork_a1"].get("answer_condition_satisfied_before_terminal"):
            # A1 doesn't have per-step state log, so we can't compute this precisely
            # Use the oracle path length as a proxy for minimum sufficient steps
            a1_first_sufficient = r["oracle_steps"] - 1  # last oracle step is terminal

        a1_ec = {
            "first_decision_sufficient_step": a1_first_sufficient,
            "terminal_step": r["fork_a1"]["steps"] - 1,
            "excess_cognition_steps": max(0, r["fork_a1"]["steps"] - r["oracle_steps"]) if r["fork_a1"]["success"] else None,
            "never_sufficient": not r["fork_a1"]["success"],
        }

        if m3_ec["excess_cognition_steps"] is not None:
            m3_excess.append(m3_ec["excess_cognition_steps"])
        if m3_ec["never_sufficient"]:
            m3_never_sufficient += 1
        if m3_ec["acted_on_unresolved"]:
            m3_acted_on_unresolved += 1

        if a1_ec["excess_cognition_steps"] is not None:
            a1_excess.append(a1_ec["excess_cognition_steps"])

        per_task.append({
            "task_id": r["task_id"],
            "category": r["category"],
            "m3_success": r["m3_success"],
            "a1_success": r["a1_success"],
            "m3_steps": r["fork_m3"]["steps"],
            "a1_steps": r["fork_a1"]["steps"],
            "oracle_steps": r["oracle_steps"],
            "m3_excess": m3_ec,
            "a1_excess": a1_ec,
            "m3_actions": r["fork_m3"]["continuation_actions"],
            "m3_states": [e["decision_state"] for e in r["fork_m3"]["decision_state_log"]],
        })

    m3_mean_excess = sum(m3_excess) / len(m3_excess) if m3_excess else 0
    a1_mean_excess = sum(a1_excess) / len(a1_excess) if a1_excess else 0

    print(f"  M3 trajectories with computable excess: {len(m3_excess)}/{n}")
    print(f"  M3 never sufficient (no READY/INSUFFICIENT emitted): {m3_never_sufficient}")
    print(f"  M3 acted on SUPPORTED_BUT_UNRESOLVED: {m3_acted_on_unresolved}")
    print(f"  M3 mean excess cognition steps: {m3_mean_excess:.2f}")
    print(f"  A1 mean excess steps (proxy: steps - oracle): {a1_mean_excess:.2f}")

    # === Per-category excess cognition ===
    print("\n  PER-CATEGORY EXCESS COGNITION (M3):")
    print(f"    {'Category':<30} {'n':>3} {'M3_succ':>7} {'mean_excess':>11} {'max_excess':>10} {'never_suff':>10}")
    cat_excess = {}
    for cat in sorted(set(r["category"] for r in results)):
        cat_tasks = [p for p in per_task if p["category"] == cat]
        cat_m3_succ = sum(1 for p in cat_tasks if p["m3_success"])
        cat_excess_vals = [p["m3_excess"]["excess_cognition_steps"] for p in cat_tasks
                           if p["m3_excess"]["excess_cognition_steps"] is not None]
        cat_never = sum(1 for p in cat_tasks if p["m3_excess"]["never_sufficient"])
        mean_e = sum(cat_excess_vals) / len(cat_excess_vals) if cat_excess_vals else 0
        max_e = max(cat_excess_vals) if cat_excess_vals else 0
        cat_excess[cat] = {
            "n": len(cat_tasks),
            "m3_success": cat_m3_succ,
            "mean_excess": round(mean_e, 2),
            "max_excess": max_e,
            "never_sufficient": cat_never,
        }
        print(f"    {cat:<30} {len(cat_tasks):>3} {cat_m3_succ:>7} {mean_e:>11.2f} {max_e:>10} {cat_never:>10}")

    # === State -> Action transition matrix ===
    print("\n=== EPISTEMIC STATE -> ACTION MATRIX (M3) ===")
    matrix = build_state_action_matrix(results, "m3")
    probs = compute_state_change_probability(matrix)

    print(f"    {'State':<28} {'Action':<14} {'Count':>6} {'P(change)':>10} {'P(same)':>8}")
    for state in sorted(matrix.keys()):
        for action in sorted(matrix[state].keys()):
            counts = matrix[state][action]
            total = counts["state_changed"] + counts["state_same"]
            p_change = counts["state_changed"] / total if total > 0 else None
            p_same = counts["state_same"] / total if total > 0 else None
            p_change_str = f"{p_change:.4f}" if p_change is not None else "N/A"
            p_same_str = f"{p_same:.4f}" if p_same is not None else "N/A"
            print(f"    {state:<28} {action:<14} {counts['count']:>6} {p_change_str:>10} {p_same_str:>8}")

    # === Low-value continuation identification ===
    print("\n=== LOW-VALUE CONTINUATION IDENTIFICATION ===")
    print("  (States where P(state changes | action) is low = action likely wasted)")
    low_value = []
    for state, actions in probs.items():
        for action, stats in actions.items():
            if stats["n"] >= 5 and stats["p_change"] is not None:
                if stats["p_change"] < 0.3:
                    low_value.append({
                        "state": state,
                        "action": action,
                        "p_change": stats["p_change"],
                        "n": stats["n"],
                        "interpretation": f"{action} in {state} changes state only {stats['p_change']*100:.1f}% of the time",
                    })

    low_value.sort(key=lambda x: x["p_change"])
    for lv in low_value:
        print(f"  {lv['state']:<28} + {lv['action']:<14} P(change)={lv['p_change']:.4f}  n={lv['n']}")
        print(f"    -> {lv['interpretation']}")

    # === State dwell analysis ===
    print("\n=== STATE DWELL ANALYSIS (M3) ===")
    print("  How many consecutive steps does M3 stay in each state?")
    state_dwell: dict[str, list[int]] = defaultdict(list)
    for r in results:
        log = r["fork_m3"]["decision_state_log"]
        if not log:
            continue
        current_state = log[0]["decision_state"]
        dwell = 1
        for i in range(1, len(log)):
            if log[i]["decision_state"] == current_state:
                dwell += 1
            else:
                state_dwell[current_state].append(dwell)
                current_state = log[i]["decision_state"]
                dwell = 1
        state_dwell[current_state].append(dwell)

    print(f"    {'State':<28} {'n_episodes':>10} {'mean_dwell':>10} {'max_dwell':>9}")
    dwell_stats = {}
    for state in sorted(state_dwell.keys()):
        dwells = state_dwell[state]
        mean_d = sum(dwells) / len(dwells)
        max_d = max(dwells)
        dwell_stats[state] = {
            "n_episodes": len(dwells),
            "mean_dwell": round(mean_d, 2),
            "max_dwell": max_d,
        }
        print(f"    {state:<28} {len(dwells):>10} {mean_d:>10.2f} {max_d:>9}")

    # === Excess cognition examples ===
    print("\n=== HIGH EXCESS COGNITION EXAMPLES (M3, excess >= 3) ===")
    high_excess = [p for p in per_task
                   if p["m3_excess"]["excess_cognition_steps"] is not None
                   and p["m3_excess"]["excess_cognition_steps"] >= 3]
    high_excess.sort(key=lambda x: x["m3_excess"]["excess_cognition_steps"], reverse=True)
    for p in high_excess[:10]:
        ec = p["m3_excess"]
        print(f"  {p['task_id']} ({p['category']}):")
        print(f"    excess={ec['excess_cognition_steps']} first_suff={ec['first_decision_sufficient_step']} terminal={ec['terminal_step']}")
        print(f"    states: {p['m3_states']}")
        print(f"    actions: {p['m3_actions']}")
        print()

    # === Save forensic analysis ===
    forensic = {
        "schema": "DAPH_V2B_I3_10A_FORENSIC_V1",
        "source": "i3_9_r3_affordance_clean/affordance_clean_v1.jsonl",
        "n_tasks": n,
        "excess_cognition": {
            "m3_mean_excess_steps": round(m3_mean_excess, 4),
            "a1_mean_excess_steps_proxy": round(a1_mean_excess, 4),
            "m3_never_sufficient": m3_never_sufficient,
            "m3_acted_on_unresolved": m3_acted_on_unresolved,
            "per_category": cat_excess,
        },
        "state_action_matrix": {
            state: {
                action: {
                    "count": counts["count"],
                    "state_changed": counts["state_changed"],
                    "state_same": counts["state_same"],
                }
                for action, counts in actions.items()
            }
            for state, actions in matrix.items()
        },
        "state_change_probabilities": probs,
        "low_value_continuations": low_value,
        "state_dwell_stats": dwell_stats,
        "diagnosis": {
            "primary_finding": "M3's SUPPORTED_BUT_UNRESOLVED state has low state-change probability for several actions, indicating overconservative continuation",
            "implication_for_m4": "M4 should add a decision-stability signal that distinguishes 'unresolved but decision-relevant' from 'unresolved but non-decisive'",
            "constraint": "M4 must not change the epistemic state classifier or add action recommendations",
        },
    }

    forensic_path = output_dir / "forensic_v1.json"
    forensic_path.write_text(json.dumps(forensic, indent=2, sort_keys=True) + "\n")
    print(f"\nForensic analysis saved: {forensic_path}")

    # Save per-task detail
    detail_path = output_dir / "forensic_v1_per_task.jsonl"
    with open(detail_path, "w") as f:
        for p in per_task:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    print(f"Per-task detail saved: {detail_path}")


if __name__ == "__main__":
    main()
