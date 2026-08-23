#!/usr/bin/env python3
"""
R2-DEV-V2 Immutable Analysis Pipeline.

11-step analysis, run in order. Each step reads from the previous
step's output and writes to a separate file. The pipeline is immutable:
once R2-DEV-V2 trajectories are collected, this analysis runs
deterministically and does not modify the trajectories.

Steps:
 1. Integrity/provenance
 2. Qualification invariants
 3. Gold vs inferred gate confusion matrix
 4. T2 frequency by arm/task
 5. Replacement-action distribution
 6. VERIFY usefulness in C0/E
 7. Loop migration
 8. Terminal-action distribution
 9. Success/rescue/break
10. Utility contrasts
11. D×E interaction

Usage:
    PYTHONPATH=scripts:. python3 scripts/r2_dev_v2_analysis.py \
        --trajectories /path/to/trajectories.jsonl \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/analysis/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def step1_integrity_provenance(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 1: Integrity and provenance checks."""
    print("Step 1: Integrity/provenance")

    result = {
        "n_trajectories": len(trajectories),
        "n_tasks": len(dataset),
        "expected_trajectories": len(dataset) * 4,  # 4 arms
        "arms": sorted(set(t.get("condition", t.get("arm", "?")) for t in trajectories)),
        "task_ids_in_trajectories": len(set(t.get("task_id") for t in trajectories)),
        "task_ids_in_dataset": len(set(t["task_id"] for t in dataset)),
    }

    result["trajectory_count_matches"] = (
        result["n_trajectories"] == result["expected_trajectories"]
    )
    result["task_ids_match"] = (
        result["task_ids_in_trajectories"] == result["task_ids_in_dataset"]
    )

    # Check for decoder errors
    decoder_errors = sum(1 for t in trajectories if t.get("decoder_error", False))
    result["decoder_errors"] = decoder_errors
    result["no_decoder_errors"] = decoder_errors == 0

    save_json(output / "01_integrity.json", result)
    return result


def step2_qualification_invariants(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 2: Qualification invariants."""
    print("Step 2: Qualification invariants")

    result = {
        "decoder_valid_rate": 0.0,
        "schema_valid_rate": 0.0,
        "schema_gate_violations": 0,
        "executor_admissibility_violations": 0,
    }

    total_calls = 0
    decoder_valid = 0
    schema_valid = 0
    schema_violations = 0
    admissibility_violations = 0

    for traj in trajectories:
        calls = traj.get("model_calls", [])
        for call in calls:
            total_calls += 1
            if call.get("decoder_valid"):
                decoder_valid += 1
            if call.get("schema_valid"):
                schema_valid += 1
            if call.get("schema_gate_violation"):
                schema_violations += 1
            if call.get("executor_admissibility_violation"):
                admissibility_violations += 1

    result["total_model_calls"] = total_calls
    result["decoder_valid"] = decoder_valid
    result["schema_valid"] = schema_valid
    result["decoder_valid_rate"] = decoder_valid / total_calls if total_calls else 0.0
    result["schema_valid_rate"] = schema_valid / total_calls if total_calls else 0.0
    result["schema_gate_violations"] = schema_violations
    result["executor_admissibility_violations"] = admissibility_violations

    result["all_invariants_hold"] = (
        result["decoder_valid_rate"] == 1.0
        and result["schema_valid_rate"] == 1.0
        and schema_violations == 0
        and admissibility_violations == 0
    )

    save_json(output / "02_qualification_invariants.json", result)
    return result


def step3_gate_confusion_matrix(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 3: Gold vs inferred gate confusion matrix."""
    print("Step 3: Gold vs inferred gate confusion matrix")

    # Build task lookup
    task_lookup = {t["task_id"]: t for t in dataset}

    # For each trajectory, check if the gate fired (VERIFY was gated)
    # and whether it should have (gold)
    tp = fp = fn = tn = 0
    per_arm: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        gold_should_gate = task.get("gold_should_gate_verify", False)

        # Check if gate fired: was VERIFY ever gated in this trajectory?
        gate_fired = any(
            call.get("schema_gate_violation") or
            (call.get("allowed_actions") and "VERIFY" not in call.get("allowed_actions", []))
            for call in traj.get("model_calls", [])
        )

        # Only count D and DE arms for gate assessment
        if arm not in ("D", "DE"):
            continue

        if gate_fired and gold_should_gate:
            tp += 1
            per_arm[arm]["tp"] += 1
        elif gate_fired and not gold_should_gate:
            fp += 1
            per_arm[arm]["fp"] += 1
        elif not gate_fired and gold_should_gate:
            fn += 1
            per_arm[arm]["fn"] += 1
        else:
            tn += 1
            per_arm[arm]["tn"] += 1

    fgr = fp / (fp + tn) if (fp + tn) > 0 else None
    mgr = fn / (fn + tp) if (fn + tp) > 0 else None

    result = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_gate_rate": fgr,
        "missed_gate_rate": mgr,
        "false_gate_rate_estimable": (fp + tn) > 0,
        "missed_gate_rate_estimable": (fn + tp) > 0,
        "per_arm": dict(per_arm),
    }

    save_json(output / "03_gate_confusion.json", result)
    return result


def step4_t2_frequency(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 4: T2 frequency by arm/task."""
    print("Step 4: T2 frequency by arm/task")

    task_lookup = {t["task_id"]: t for t in dataset}
    per_arm: dict[str, dict] = defaultdict(lambda: {
        "gold_t2_count": 0, "t2_count": 0, "total": 0,
    })

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        gold_t2 = task.get("gold_t2", False)

        # Count inferred T2 occurrences during the trajectory
        t2_count = sum(
            1 for call in traj.get("model_calls", [])
            if call.get("t2", False)
        )

        per_arm[arm]["total"] += 1
        if gold_t2:
            per_arm[arm]["gold_t2_count"] += 1
        if t2_count > 0:
            per_arm[arm]["t2_count"] += 1

    result = {"per_arm": dict(per_arm)}
    save_json(output / "04_t2_frequency.json", result)
    return result


def step5_replacement_action_distribution(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 5: Replacement-action distribution.

    For D/DE, when VERIFY is removed, what does the model choose instead?
    """
    print("Step 5: Replacement-action distribution")

    task_lookup = {t["task_id"]: t for t in dataset}
    replacements: dict[str, Counter] = defaultdict(Counter)

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        if arm not in ("D", "DE"):
            continue

        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        gold_t2 = task.get("gold_t2", False)

        # Find the first action chosen when VERIFY was gated (at T2)
        calls = traj.get("model_calls", [])
        for call in calls:
            allowed = call.get("allowed_actions", [])
            if allowed and "VERIFY" not in allowed and call.get("selected_action"):
                action = call["selected_action"]
                if gold_t2:
                    replacements[arm][action] += 1
                break  # first replacement only

    result = {
        arm: dict(counter) for arm, counter in replacements.items()
    }
    save_json(output / "05_replacement_actions.json", result)
    return result


def step6_verify_usefulness(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 6: VERIFY usefulness in C0/E.

    For each VERIFY in C0/E:
        UsefulVerify = Δdecision_state OR Δlive/eliminated OR ΔT2
    """
    print("Step 6: VERIFY usefulness in C0/E")

    task_lookup = {t["task_id"]: t for t in dataset}
    verify_events: list[dict] = []

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        if arm not in ("C0", "E"):
            continue

        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        calls = traj.get("model_calls", [])

        for i, call in enumerate(calls):
            if call.get("selected_action") == "VERIFY":
                # Check if state changed after this VERIFY.
                # calls[i] is the state when VERIFY was chosen.
                # calls[i+1] is the state after VERIFY was executed.
                state_before = call.get("decision_state_exposed")
                state_after = calls[i + 1].get("decision_state_exposed") if i + 1 < len(calls) else None
                t2_before = call.get("t2")
                t2_after = calls[i + 1].get("t2") if i + 1 < len(calls) else None
                live_before = call.get("n_live_hypotheses")
                live_after = calls[i + 1].get("n_live_hypotheses") if i + 1 < len(calls) else None
                elim_before = call.get("n_eliminated_hypotheses")
                elim_after = calls[i + 1].get("n_eliminated_hypotheses") if i + 1 < len(calls) else None

                useful = (
                    (state_before != state_after)
                    or (live_before != live_after)
                    or (t2_before != t2_after)
                    or (elim_before != elim_after)
                )

                verify_events.append({
                    "task_id": task_id,
                    "arm": arm,
                    "stratum": task.get("stratum"),
                    "useful": useful,
                    "state_changed": state_before != state_after,
                    "live_changed": live_before != live_after,
                    "t2_changed": t2_before != t2_after,
                    "elim_changed": elim_before != elim_after,
                    "state_before": state_before,
                    "state_after": state_after,
                    "live_before": live_before,
                    "live_after": live_after,
                })

    useful_count = sum(1 for e in verify_events if e["useful"])
    total = len(verify_events)

    result = {
        "total_verify_events": total,
        "useful_verify_events": useful_count,
        "useful_verify_rate": useful_count / total if total > 0 else None,
        "events": verify_events,
    }
    save_json(output / "06_verify_usefulness.json", result)
    return result


def step7_loop_migration(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 7: Loop migration — how trajectories move through epistemic phases."""
    print("Step 7: Loop migration")

    task_lookup = {t["task_id"]: t for t in dataset}
    per_arm: dict[str, dict] = defaultdict(lambda: {
        "avg_steps": 0.0,
        "avg_verify_count": 0.0,
        "avg_retrieve_count": 0.0,
        "avg_search_count": 0.0,
        "avg_reason_count": 0.0,
    })

    arm_counts: dict[str, list] = defaultdict(list)

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        calls = traj.get("model_calls", [])
        actions = [c.get("selected_action") for c in calls if c.get("selected_action")]

        arm_counts[arm].append({
            "n_steps": len(calls),
            "n_verify": actions.count("VERIFY"),
            "n_retrieve": actions.count("RETRIEVE"),
            "n_search": actions.count("SEARCH_MORE"),
            "n_reason": actions.count("REASON_MORE"),
        })

    for arm, counts in arm_counts.items():
        n = len(counts)
        if n > 0:
            per_arm[arm] = {
                "avg_steps": sum(c["n_steps"] for c in counts) / n,
                "avg_verify_count": sum(c["n_verify"] for c in counts) / n,
                "avg_retrieve_count": sum(c["n_retrieve"] for c in counts) / n,
                "avg_search_count": sum(c["n_search"] for c in counts) / n,
                "avg_reason_count": sum(c["n_reason"] for c in counts) / n,
            }

    result = {"per_arm": dict(per_arm)}
    save_json(output / "07_loop_migration.json", result)
    return result


def step8_terminal_action_distribution(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 8: Terminal-action distribution."""
    print("Step 8: Terminal-action distribution")

    per_arm: dict[str, Counter] = defaultdict(Counter)

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        terminal = traj.get("terminal_action") or traj.get("final_action")
        if terminal:
            per_arm[arm][terminal] += 1
        else:
            per_arm[arm]["NONE"] += 1

    result = {
        arm: dict(counter) for arm, counter in per_arm.items()
    }
    save_json(output / "08_terminal_actions.json", result)
    return result


def step9_success_rescue_break(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 9: Success/rescue/break classification."""
    print("Step 9: Success/rescue/break")

    task_lookup = {t["task_id"]: t for t in dataset}
    per_arm: dict[str, dict] = defaultdict(lambda: {
        "success": 0, "rescue": 0, "break": 0, "total": 0,
    })

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        expected = task.get("expected_terminal")

        terminal = traj.get("terminal_action") or traj.get("final_action")
        success = (terminal == expected) if expected and terminal else False
        decoder_error = traj.get("decoder_error", False)

        per_arm[arm]["total"] += 1
        if decoder_error:
            per_arm[arm]["break"] += 1
        elif success:
            per_arm[arm]["success"] += 1
        else:
            per_arm[arm]["rescue"] += 1  # completed but wrong terminal

    result = {"per_arm": dict(per_arm)}
    save_json(output / "09_success_rescue_break.json", result)
    return result


def step10_utility_contrasts(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 10: Utility contrasts (D vs C0, E vs C0, DE vs C0)."""
    print("Step 10: Utility contrasts")

    task_lookup = {t["task_id"]: t for t in dataset}

    # Compute utility per arm
    arm_utilities: dict[str, list[float]] = defaultdict(list)

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        expected = task.get("expected_terminal")
        terminal = traj.get("terminal_action") or traj.get("final_action")
        # Use realized_utility from the trajectory if available
        if "realized_utility" in traj:
            utility = float(traj["realized_utility"])
        else:
            utility = 1.0 if (terminal == expected) else 0.0
        arm_utilities[arm].append(utility)

    # Compute mean utility per arm
    mean_utility = {}
    for arm, utils in arm_utilities.items():
        mean_utility[arm] = sum(utils) / len(utils) if utils else 0.0

    # Contrasts
    c0_util = mean_utility.get("C0", 0.0)
    d_util = mean_utility.get("D", 0.0)
    e_util = mean_utility.get("E", 0.0)
    de_util = mean_utility.get("DE", 0.0)

    result = {
        "mean_utility": mean_utility,
        "contrasts": {
            "delta_D": d_util - c0_util,
            "delta_E": e_util - c0_util,
            "delta_DE": de_util - c0_util,
            "U_DE_minus_U_E": de_util - e_util,
            "U_DE_minus_U_D": de_util - d_util,
        },
        "n_per_arm": {arm: len(utils) for arm, utils in arm_utilities.items()},
    }
    save_json(output / "10_utility_contrasts.json", result)
    return result


def step11_dxe_interaction(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Step 11: D×E interaction."""
    print("Step 11: D×E interaction")

    task_lookup = {t["task_id"]: t for t in dataset}
    arm_utilities: dict[str, list[float]] = defaultdict(list)

    for traj in trajectories:
        arm = traj.get("condition", traj.get("arm", "?"))
        task_id = traj.get("task_id")
        task = task_lookup.get(task_id, {})
        expected = task.get("expected_terminal")
        terminal = traj.get("terminal_action") or traj.get("final_action")
        # Use realized_utility from the trajectory if available
        if "realized_utility" in traj:
            utility = float(traj["realized_utility"])
        else:
            utility = 1.0 if (terminal == expected) else 0.0
        arm_utilities[arm].append(utility)

    c0 = sum(arm_utilities.get("C0", [0])) / max(len(arm_utilities.get("C0", [1])), 1)
    d = sum(arm_utilities.get("D", [0])) / max(len(arm_utilities.get("D", [1])), 1)
    e = sum(arm_utilities.get("E", [0])) / max(len(arm_utilities.get("E", [1])), 1)
    de = sum(arm_utilities.get("DE", [0])) / max(len(arm_utilities.get("DE", [1])), 1)

    # I_{D×E} = (U_DE - U_E) - (U_D - U_C0)
    #        = U_DE - U_E - U_D + U_C0
    interaction = de - e - d + c0

    result = {
        "U_C0": c0,
        "U_D": d,
        "U_E": e,
        "U_DE": de,
        "I_DxE": interaction,
        "interpretation": (
            "Positive interaction: D mitigates E's harm (or floor effect)" if interaction > 0
            else "Negative interaction: D worsens E's harm" if interaction < 0
            else "No interaction"
        ),
        "U_DE_equals_U_E": abs(de - e) < 1e-10,
        "U_DE_equals_U_D": abs(de - d) < 1e-10,
    }
    save_json(output / "11_dxe_interaction.json", result)
    return result


def run_full_analysis(
    trajectories: list[dict],
    dataset: list[dict],
    output: Path,
) -> dict:
    """Run all 11 analysis steps in order."""
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    results["step1"] = step1_integrity_provenance(trajectories, dataset, output)
    results["step2"] = step2_qualification_invariants(trajectories, dataset, output)
    results["step3"] = step3_gate_confusion_matrix(trajectories, dataset, output)
    results["step4"] = step4_t2_frequency(trajectories, dataset, output)
    results["step5"] = step5_replacement_action_distribution(trajectories, dataset, output)
    results["step6"] = step6_verify_usefulness(trajectories, dataset, output)
    results["step7"] = step7_loop_migration(trajectories, dataset, output)
    results["step8"] = step8_terminal_action_distribution(trajectories, dataset, output)
    results["step9"] = step9_success_rescue_break(trajectories, dataset, output)
    results["step10"] = step10_utility_contrasts(trajectories, dataset, output)
    results["step11"] = step11_dxe_interaction(trajectories, dataset, output)

    # Summary
    summary = {
        "steps_completed": 11,
        "integrity_ok": results["step1"].get("trajectory_count_matches", False),
        "invariants_hold": results["step2"].get("all_invariants_hold", False),
        "gate_assessment": results["step3"],
        "utility_contrasts": results["step10"]["contrasts"],
        "dxe_interaction": results["step11"],
        "verify_usefulness": {
            "total": results["step6"]["total_verify_events"],
            "useful": results["step6"]["useful_verify_events"],
            "rate": results["step6"]["useful_verify_rate"],
        },
    }
    save_json(output / "summary.json", summary)

    print("\n=== Analysis Summary ===")
    print(f"  Steps completed: 11")
    print(f"  Integrity OK: {summary['integrity_ok']}")
    print(f"  Invariants hold: {summary['invariants_hold']}")
    print(f"  Utility contrasts: {summary['utility_contrasts']}")
    print(f"  D×E interaction: {summary['dxe_interaction']['I_DxE']:.4f}")
    print(f"  VERIFY usefulness: {summary['verify_usefulness']}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="R2-DEV-V2 Immutable Analysis Pipeline")
    parser.add_argument("--trajectories", type=Path, required=True,
                        help="Path to trajectories JSONL (results.jsonl)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to balanced dataset JSONL")
    parser.add_argument("--receipts", type=Path, default=None,
                        help="Path to mechanism_receipts.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for analysis")
    args = parser.parse_args()

    trajectories = load_jsonl(args.trajectories)
    dataset = load_jsonl(args.dataset)

    # Load mechanism receipts and join with trajectories
    receipts_by_key: dict[str, list[dict]] = defaultdict(list)
    if args.receipts and args.receipts.exists():
        for receipt in load_jsonl(args.receipts):
            key = receipt.get("trajectory_key", "")
            receipts_by_key[key].append(receipt)
        print(f"Mechanism receipts: {sum(len(v) for v in receipts_by_key.values())}")

    # Attach receipts to trajectories
    for traj in trajectories:
        key = traj.get("trajectory_key", "")
        traj["model_calls"] = receipts_by_key.get(key, [])

    print(f"Trajectories: {len(trajectories)}")
    print(f"Dataset tasks: {len(dataset)}")
    print()

    run_full_analysis(trajectories, dataset, args.output)


if __name__ == "__main__":
    main()
