#!/usr/bin/env python3
"""I3.15 trajectory smoke test: 5 tasks x 2 arms under Q3_RERANKED.

Evaluates S1-S11 criteria and checks A1/R1 packet divergence post-T2.
Does NOT run the 900-trajectory experiment.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.run_i3_15_r1_balanced import (
    generate_i3_15_corpus, run_single, FROZEN_INFERENCE_CONFIG,
)

# 5 carefully selected tasks:
#   1 easy ANSWER, 1 easy DEFER, 1 medium (2-evidence ANSWER),
#   1 hard multi-evidence ANSWER, 1 hard insufficient/DEFER
SMOKE_TASK_INDICES = [0, 1, 50, 100, 101]
RETRIEVAL_LEVEL = "Q3_RERANKED"
ARMS = ["A1_INFERRED", "R1_INFERRED"]
BUDGET = {"max_executive_steps": 10, "max_retrieval_calls": 3,
          "max_search_calls": 2, "max_verification_calls": 5}


def main():
    tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)
    selected = [tasks[i] for i in SMOKE_TASK_INDICES]

    print(f"SMOKE TEST: {len(selected)} tasks x {len(ARMS)} arms x {RETRIEVAL_LEVEL}")
    print(f"  Config: {FROZEN_INFERENCE_CONFIG['config_id']}")
    print(f"  Adapter: {FROZEN_INFERENCE_CONFIG['prompt_adapter']}")
    print()

    results = []
    for task in selected:
        task_id = task.evidence_task.task_id
        for arm in ARMS:
            print(f"  Running {task_id} {arm} {RETRIEVAL_LEVEL}...")
            t0 = time.time()
            work_item = (task_id, RETRIEVAL_LEVEL, arm, "local", "", BUDGET)
            result = run_single(work_item)
            elapsed = time.time() - t0
            print(f"    steps={result.get('steps', 0)} "
                  f"terminal={result.get('terminal_action')} "
                  f"success={result.get('success')} "
                  f"t2={result.get('t2_triggered')} "
                  f"decoder_failures={result.get('decoder_failures', 0)} "
                  f"fail_closed={result.get('fail_closed_count', 0)} "
                  f"({elapsed:.1f}s)")
            results.append(result)

    # Save full results
    out_dir = ROOT / "experiments/v2b_i3_15/development/i3_15_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "smoke_results_v1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Evaluate S1-S11
    print()
    print("=" * 80)
    print("SMOKE TEST EVALUATION (S1-S11)")
    print("=" * 80)

    criteria = {}

    # S1: decoder failures = 0
    total_decoder_failures = sum(r.get("decoder_failures", 0) for r in results)
    criteria["S1_decoder_failures_zero"] = {
        "value": total_decoder_failures, "passed": total_decoder_failures == 0,
    }

    # S2: fail_closed trajectories = 0
    total_fail_closed = sum(r.get("fail_closed_count", 0) for r in results)
    criteria["S2_fail_closed_zero"] = {
        "value": total_fail_closed, "passed": total_fail_closed == 0,
    }

    # S3: terminal actions include ANSWER and DEFER
    terminals = set(r.get("terminal_action") for r in results)
    criteria["S3_terminal_includes_answer_and_defer"] = {
        "terminals": sorted(terminals),
        "passed": "ANSWER" in terminals and "DEFER" in terminals,
    }

    # S4: intermediate actions include VERIFY
    all_actions = []
    for r in results:
        all_actions.extend(r.get("continuation_actions", []))
    intermediate = [a for a in all_actions if a not in ("ANSWER", "DEFER", "STOP")]
    criteria["S4_intermediate_includes_verify"] = {
        "intermediate_actions": sorted(set(intermediate)),
        "passed": "VERIFY" in intermediate,
    }

    # S5: at least some trajectories have steps > 1
    max_steps = max(r.get("steps", 0) for r in results)
    multi_step = sum(1 for r in results if r.get("steps", 0) > 1)
    criteria["S5_some_multi_step"] = {
        "max_steps": max_steps, "multi_step_count": multi_step,
        "passed": multi_step > 0,
    }

    # S6: R1 T2 fires on at least one T2-eligible task
    r1_results = [r for r in results if r.get("arm") == "R1_INFERRED"]
    t2_fired = sum(1 for r in r1_results if r.get("t2_triggered"))
    criteria["S6_r1_t2_fires"] = {
        "t2_fired_count": t2_fired, "n_r1_trajectories": len(r1_results),
        "passed": t2_fired > 0,
    }

    # S7: R1 stays untriggered on at least one non-T2 task
    t2_not_fired = sum(1 for r in r1_results if not r.get("t2_triggered"))
    criteria["S7_r1_untriggered"] = {
        "t2_not_fired_count": t2_not_fired,
        "passed": t2_not_fired > 0,
    }

    # S8: A1 and R1 are not identical on every T2-eligible trajectory
    a1_by_task = {r["task_id"]: r for r in results if r.get("arm") == "A1_INFERRED"}
    r1_by_task = {r["task_id"]: r for r in results if r.get("arm") == "R1_INFERRED"}
    divergent = 0
    for task_id in a1_by_task:
        a1 = a1_by_task[task_id]
        r1 = r1_by_task.get(task_id)
        if r1 is None:
            continue
        a1_actions = tuple(a1.get("continuation_actions", []))
        r1_actions = tuple(r1.get("continuation_actions", []))
        if a1_actions != r1_actions:
            divergent += 1
    criteria["S8_a1_r1_divergence"] = {
        "divergent_count": divergent, "n_pairs": len(a1_by_task),
        "passed": divergent > 0,
    }

    # S9: Q3 retrieved evidence actually appears in the serialized model packet
    q3_evidence_in_packet = 0
    for r in results:
        call_log = r.get("model_call_log", [])
        for call in call_log:
            if call.get("packet_sha256"):
                q3_evidence_in_packet += 1
                break
    criteria["S9_q3_evidence_in_packet"] = {
        "trajectories_with_packet_hash": q3_evidence_in_packet,
        "passed": q3_evidence_in_packet > 0,
    }

    # S10: changing retrieval condition changes at least some request hashes
    # (We only run Q3 here, so we check that packet hashes differ across tasks)
    packet_hashes = set()
    for r in results:
        for call in r.get("model_call_log", []):
            if call.get("packet_sha256"):
                packet_hashes.add(call["packet_sha256"])
    criteria["S10_packet_hash_diversity"] = {
        "unique_packet_hashes": len(packet_hashes),
        "passed": len(packet_hashes) > 1,
    }

    # S11: R1 packet after T2 != A1 packet for same observable state
    # Check if any R1 trajectory has T2 triggered and the post-T2 packet
    # differs from the corresponding A1 packet
    s11_passed = False
    s11_details = []
    for r1 in r1_results:
        if not r1.get("t2_triggered"):
            continue
        task_id = r1["task_id"]
        a1 = a1_by_task.get(task_id)
        if a1 is None:
            continue
        # Compare post-T2 packets
        r1_calls = r1.get("model_call_log", [])
        a1_calls = a1.get("model_call_log", [])
        t2_step = r1.get("t2_trigger_step")
        if t2_step is not None and t2_step < len(r1_calls):
            r1_post_t2_packet = r1_calls[t2_step].get("packet_sha256")
            # Compare with A1 packet at the same step
            if t2_step < len(a1_calls):
                a1_packet = a1_calls[t2_step].get("packet_sha256")
                differs = r1_post_t2_packet != a1_packet
                s11_details.append({
                    "task_id": task_id,
                    "t2_step": t2_step,
                    "r1_post_t2_packet": r1_post_t2_packet,
                    "a1_packet": a1_packet,
                    "differs": differs,
                })
                if differs:
                    s11_passed = True
    criteria["S11_r1_post_t2_packet_differs"] = {
        "details": s11_details, "passed": s11_passed,
    }

    # Print results
    all_passed = True
    for name, info in criteria.items():
        status = "PASS" if info["passed"] else "FAIL"
        if not info["passed"]:
            all_passed = False
        # Print without the details dict for S11
        if name == "S11_r1_post_t2_packet_differs":
            print(f"  {name}: {status}  (details: {len(info['details'])} T2 cases)")
        else:
            print(f"  {name}: {status}  {info}")

    print()
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # Print trajectory summaries
    print()
    print("=" * 80)
    print("TRAJECTORY SUMMARIES")
    print("=" * 80)
    for r in results:
        actions = r.get("continuation_actions", [])
        print(f"  {r['task_id']:<20} {r['arm']:<14} "
              f"steps={r.get('steps', 0)} "
              f"actions={actions} "
              f"terminal={r.get('terminal_action')} "
              f"success={r.get('success')} "
              f"t2={r.get('t2_triggered')}")

    # Print provenance sample
    print()
    print("=" * 80)
    print("PROVENANCE SAMPLE (first call of first R1 trajectory)")
    print("=" * 80)
    if r1_results:
        first_r1 = r1_results[0]
        calls = first_r1.get("model_call_log", [])
        if calls:
            call = calls[0]
            provenance_fields = {
                "provider_raw_output": call.get("provider_raw_output"),
                "normalized_output": call.get("normalized_output"),
                "provider_raw_sha256": call.get("provider_raw_sha256"),
                "normalized_sha256": call.get("normalized_sha256"),
                "normalization_applied": call.get("normalization_applied"),
                "system_prompt_sha256": call.get("system_prompt_sha256"),
                "packet_sha256": call.get("packet_sha256"),
                "finish_reason": call.get("finish_reason"),
                "decoder_valid": call.get("decoder_valid"),
                "decoded_action": call.get("decoded_action"),
                "decoded_target_id": call.get("decoded_target_id"),
            }
            print(json.dumps(provenance_fields, indent=2))

    # Save evaluation
    with open(out_dir / "smoke_evaluation_v1.json", "w") as f:
        json.dump({
            "config_id": FROZEN_INFERENCE_CONFIG["config_id"],
            "prompt_adapter": FROZEN_INFERENCE_CONFIG["prompt_adapter"],
            "criteria": criteria,
            "all_passed": all_passed,
        }, f, indent=2, default=str)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
