#!/usr/bin/env python3
"""I3.5-PQ: Collect pinned-policy causal action data.

For each of the 1056 interventions in the frozen schedule:
  1. Restore the checkpoint
  2. Force the specified action
  3. For terminal actions (ANSWER, DEFER, STOP): record outcome immediately
  4. For non-terminal actions (RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE):
     return control to the pinned Qwen2.5-7B policy and continue until
     terminal, step limit, backend failure, or decoder failure.
  5. Record Q^{pi_Qwen}(s,a) = realized utility under pinned downstream policy

This is the critical dataset that will separate RETRIEVE/VERIFY/SEARCH_MORE
because Qwen will make different downstream decisions (and mistakes) depending
on which action was forced first.

The pinned-policy rollout mirrors the canonical control loop in
``run_r2_dev_v2.py`` (the P0/no-governor arm) so the downstream Qwen
sees exactly the same packet, schema, and decoder path it would see live.

Usage (Colab):
  PYTHONPATH=. python3 scripts/collect_i3_5_pinned_policy_causal.py \
    --gguf-path /content/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/pinned_policy \
    --resume

Output:
  experiments/i3_5/pinned_policy/pinned_causal_actions_v1.jsonl
  experiments/i3_5/pinned_policy/pinned_causal_receipts_v1.jsonl
  experiments/i3_5/pinned_policy/pinned_model_calls_v1.jsonl
  experiments/i3_5/pinned_policy/pinned_causal_errors_v1.jsonl
  experiments/i3_5/pinned_policy/pinned_causal_manifest_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_checkpoints(path: Path) -> list[dict]:
    checkpoints: list[dict] = []
    with open(path) as f:
        for line in f:
            checkpoints.append(json.loads(line))
    return checkpoints


def load_causal_oracle(path: Path) -> dict[str, dict[str, dict]]:
    """Load oracle causal data for direct comparison.

    Returns: {checkpoint_id: {action: record}}
    """
    by_cp_action: dict[str, dict[str, dict]] = defaultdict(dict)
    if not path.exists():
        return by_cp_action
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cp_action[r["checkpoint_id"]][r["forced_action"]] = r
    return by_cp_action


@dataclass
class PinnedPolicyWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = open(self.output_dir / "pinned_causal_actions_v1.jsonl", "a")
        self.receipts_file = open(self.output_dir / "pinned_causal_receipts_v1.jsonl", "a")
        self.model_calls_file = open(self.output_dir / "pinned_model_calls_v1.jsonl", "a")
        self.errors_file = open(self.output_dir / "pinned_causal_errors_v1.jsonl", "a")

    def _write(self, fh, record: dict):
        fh.write(json.dumps(record, sort_keys=True, default=bool) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write_result(self, record: dict):
        self._write(self.results_file, record)

    def write_receipt(self, record: dict):
        self._write(self.receipts_file, record)

    def write_model_call(self, record: dict):
        self._write(self.model_calls_file, record)

    def write_error(self, record: dict):
        self._write(self.errors_file, record)

    def load_completed_keys(self) -> set[str]:
        completed: set[str] = set()
        results_path = self.output_dir / "pinned_causal_actions_v1.jsonl"
        if not results_path.exists():
            return completed
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(f"{r['checkpoint_id']}:{r['forced_action']}")
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed


# ---------------------------------------------------------------------------
# Pinned-policy rollout (mirrors run_r2_dev_v2.py P0 arm)
# ---------------------------------------------------------------------------

def pinned_policy_rollout(
    runtime,
    task,
    backend,
    i3_7e,
    utility,
    max_steps: int = 8,
    checkpoint_id: str = "",
    forced_action: str = "",
    writer: PinnedPolicyWriter | None = None,
) -> tuple[float, bool, str | None, list[str], list[dict], bool, bool, bool, bool, str]:
    """Continue with the pinned Qwen policy after a forced non-terminal action.

    Mirrors the canonical control loop in run_r2_dev_v2.py:
      1. build_evidence_snapshot(runtime, prior_actions, prior_outcomes)
      2. _classify_from_snapshot -> T2
      3. ActionState -> compute_allowed_actions (C0 arm, no gates)
      4. build_mdsg_state_with_affordances_packet -> packet
      5. MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT -> system_prompt
      6. evidence_packet_json(packet) -> user_prompt
      7. build_action_schema(allowed) -> schema
      8. backend.generate(system_prompt, user_prompt, temperature=0, max_tokens=256,
                          allowed_actions=allowed)
      9. decode_output_strict(raw_output)
     10. executor.execute(runtime, action, target_evidence_id=target_id)
     11. utility.action_cost + utility.terminal_reward

    Returns:
        (terminal_utility, success, terminal_action, downstream_actions,
         model_call_receipts, premature_defer, premature_answer,
         resource_exhaustion, loop, status)
    """
    from hrm_adaptive_memory.executive.evidence_benchmark import build_evidence_snapshot
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from r2_schema import build_action_schema, schema_sha256, schema_action_enum
    from r2_allowed_actions import (
        compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0,
    )

    executor = EvidenceExecutor()
    downstream_actions: list[str] = []
    model_call_receipts: list[dict] = []
    prior_actions: list[str] = list(runtime.task.task_id and [] or [])
    prior_actions = []
    prior_outcomes: list[str] = []
    n_hypotheses = len(task.hypotheses)
    realized = 0.0
    premature_defer = False
    premature_answer = False
    loop_detected = False
    status = "completed"

    for step_id in range(max_steps):
        # 1. Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # 2. Compute T2 from snapshot
        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

        # 3. Build action state and compute allowed actions (C0 = neutral, no gates)
        action_state = ActionState(
            t2=t2,
            executive_steps_remaining=max_steps - step_id,
            can_retrieve=evidence_snapshot.can_retrieve,
            can_search=evidence_snapshot.can_search,
            can_verify=evidence_snapshot.can_verify,
        )

        try:
            allowed_decision = compute_allowed_actions(action_state, C0)
        except EmptyAllowedActionSet:
            status = "empty_allowed_set"
            break

        # 4-6. Build packet and prompts
        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT
        user_prompt = i3_7e.evidence_packet_json(packet)

        # 7. Build dynamic schema from allowed actions
        schema = build_action_schema(allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        schema_enum = set(schema_action_enum(schema))

        # 8. Call the pinned Qwen model
        try:
            call_result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            model_call_receipts.append({
                "checkpoint_id": checkpoint_id,
                "forced_action": forced_action,
                "step": step_id,
                "result_class": "backend_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_output": "",
                "decoder_valid": False,
            })
            status = "backend_error"
            break

        raw_output = call_result.raw_output

        # 9. Strict decoding (no markdown stripping, no repair)
        outcome = decode_output_strict(raw_output)
        schema_valid = (
            outcome.valid
            and outcome.parsed_json is not None
            and outcome.parsed_json.get("action") in schema_enum
        )

        # Record model call receipt
        receipt: dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "forced_action": forced_action,
            "step": step_id,
            "raw_output": raw_output,
            "provider_raw_output": getattr(call_result, "provider_raw_output", raw_output),
            "decoder_valid": bool(outcome.valid),
            "decoder_rejection_code": getattr(outcome, "rejection_code", None),
            "schema_valid": bool(schema_valid),
            "schema_sha256": schema_sha,
            "json_schema_sha256": getattr(call_result, "json_schema_sha256", ""),
            "system_prompt_sha256": getattr(call_result, "system_prompt_sha256", ""),
            "user_packet_sha256": getattr(call_result, "user_packet_sha256", ""),
            "request_sha256": getattr(call_result, "request_sha256", ""),
            "prompt_tokens": getattr(call_result, "prompt_tokens", 0),
            "completion_tokens": getattr(call_result, "completion_tokens", 0),
            "latency_ms": getattr(call_result, "latency_ms", 0),
            "model_name": getattr(call_result, "model_name", ""),
            "finish_reason": getattr(call_result, "finish_reason", ""),
            "allowed_actions": sorted(allowed_decision.allowed),
            "t2": bool(t2),
            "n_live_hypotheses": n_hypotheses - len(eliminated),
            "n_eliminated_hypotheses": len(eliminated),
        }

        if not outcome.valid or not outcome.proposal:
            receipt["result_class"] = "decoder_error"
            receipt["selected_action"] = None
            model_call_receipts.append(receipt)
            if writer:
                writer.write_model_call(receipt)
            status = "decoder_error"
            break

        proposal = outcome.proposal
        action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)
        target_id = getattr(proposal, "target_id", None)

        # Admissibility check (defense in depth)
        admissibility_passed = action_str in allowed_decision.allowed
        receipt["result_class"] = "success"
        receipt["selected_action"] = action_str
        receipt["selected_reason_code"] = proposal.reason_code
        receipt["selected_target_id"] = target_id
        receipt["admissibility_passed"] = bool(admissibility_passed)
        model_call_receipts.append(receipt)
        if writer:
            writer.write_model_call(receipt)

        if not admissibility_passed:
            status = "admissibility_violation"
            break

        # 10. Execute the action
        action = DecisionAction(action_str)
        resources_before = runtime.resources
        try:
            exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        except Exception as exc:
            status = "executor_error"
            break

        runtime = exec_res.runtime
        resources_after = runtime.resources

        # 11. Compute step cost
        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        downstream_actions.append(action_str)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            if action is DecisionAction.DEFER and not exec_res.task_success:
                premature_defer = True
            if action is DecisionAction.ANSWER and not exec_res.task_success:
                premature_answer = True
            return (realized, bool(exec_res.task_success), action_str,
                    downstream_actions, model_call_receipts,
                    premature_defer, premature_answer,
                    exec_res.outcome_code == "RESOURCE_EXHAUSTED", False,
                    status)

        # Loop detection (same action 3+ times)
        if len(downstream_actions) >= 3:
            recent = downstream_actions[-3:]
            if all(a == recent[0] for a in recent):
                loop_detected = True
                status = "loop_detected"
                break

    # Step limit reached without terminal
    return (realized - 0.5, False, None, downstream_actions,
            model_call_receipts, premature_defer, premature_answer,
            False, loop_detected, status)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ pinned-policy causal data collection")
    parser.add_argument("--gguf-path", required=True, help="Path to Qwen2.5-7B GGUF file")
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/pinned_policy")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-rollout-steps", type=int, default=8)
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load i3_7e module (system prompt, packet builder, snapshot classifier)
    print("Loading i3_7e module...")
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)
    print("  i3_7e loaded.")

    # Load benchmark tasks
    print("Loading benchmark tasks...")
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_state_discrimination_generator import (
        generate_i3_5_state_discrimination_benchmark,
    )
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=24, n_per_two_live_subtype=20, seed=9137,
    )
    task_lookup = {t.task_id: t for t in tasks}
    print(f"  Loaded {len(tasks)} tasks")

    # Load checkpoints
    checkpoints_path = REPO_ROOT / "experiments/i3_5/datasets/checkpoints_v1.jsonl"
    checkpoint_recs = load_checkpoints(checkpoints_path)
    print(f"  Loaded {len(checkpoint_recs)} checkpoints")

    # Load schedule
    from daph.intervention.schedule import load_schedule
    schedule_path = REPO_ROOT / "experiments/i3_5/datasets/intervention_schedule_v1.json"
    schedule = load_schedule(schedule_path)
    print(f"  Loaded schedule: {schedule.schedule_id[:16]}... ({len(schedule.interventions)} interventions)")

    # Load oracle causal data for comparison
    oracle_path = REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1.jsonl"
    oracle_lookup = load_causal_oracle(oracle_path)
    print(f"  Loaded oracle causal data for {len(oracle_lookup)} checkpoints")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"
    )
    print("  Utility loaded.")

    # Initialize backend
    print(f"\nInitializing R2DirectLlamaBackend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Compute backend identity SHA
    backend_id_content = json.dumps({
        "model_name": args.model_name,
        "gguf_path": args.gguf_path,
        "n_ctx": args.n_ctx,
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 42,
        "top_p": 1.0,
        "top_k": 40,
        "repeat_penalty": 1.0,
    }, sort_keys=True)
    backend_sha = hashlib.sha256(backend_id_content.encode()).hexdigest()

    # Setup writer
    writer = PinnedPolicyWriter(output_dir)
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} interventions already completed")

    # Group interventions by checkpoint
    by_checkpoint: dict[str, list] = defaultdict(list)
    for iv in schedule.interventions:
        by_checkpoint[iv.checkpoint_id].append(iv)

    # Imports for the main loop
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from daph.intervention.checkpoint import StateCheckpoint
    from daph.intervention.force_action import force_action, ForcedActionResult

    executor = EvidenceExecutor()
    timestamp = datetime.now(timezone.utc).isoformat()

    n_total = len(schedule.interventions)
    n_done = len(completed_keys)
    n_success = 0
    n_terminal = 0
    n_premature_defer = 0
    n_premature_answer = 0
    n_backend_errors = 0
    n_decoder_errors = 0
    start_time = time.time()

    print(f"\nStarting collection: {n_total - n_done} interventions remaining")
    print(f"  Backend SHA: {backend_sha[:16]}...")
    print(f"  Schedule ID: {schedule.schedule_id[:16]}...")
    print()

    for cp_rec in checkpoint_recs:
        cp_id = cp_rec["checkpoint_id"]
        task_id = cp_rec["task_id"]
        task = task_lookup.get(task_id)
        if task is None:
            print(f"WARNING: task {task_id} not found")
            continue

        interventions = by_checkpoint.get(cp_id, [])
        if not interventions:
            continue

        for iv in interventions:
            intervention_key = f"{cp_id}:{iv.action}"
            if intervention_key in completed_keys:
                continue

            action = DecisionAction(iv.action)
            target_eid = iv.target_evidence_id

            # Reconstruct checkpoint object
            cp = StateCheckpoint(
                checkpoint_id=cp_rec["checkpoint_id"],
                task_id=cp_rec["task_id"],
                step=cp_rec["step"],
                phase=cp_rec["phase"],
                hypotheses=tuple(cp_rec["hypotheses"]),
                evidence=tuple(cp_rec["evidence"]),
                state_features=cp_rec["state_features"],
                resources=cp_rec["resources"],
                legal_actions=tuple(cp_rec["legal_actions"]),
                state_sha256=cp_rec["state_sha256"],
                prior_actions=tuple(cp_rec["prior_actions"]),
                prior_outcomes=tuple(cp_rec["prior_outcomes"]),
            )

            # Force the action
            try:
                forced_result, post_runtime = force_action(cp, task, action, target_evidence_id=target_eid)
            except Exception as exc:
                writer.write_error({
                    "checkpoint_id": cp_id,
                    "task_id": task_id,
                    "forced_action": iv.action,
                    "error": "ForceActionError",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timestamp": timestamp,
                })
                n_done += 1
                continue

            # Compute forced action step cost
            forced_resources_before = type(post_runtime.resources)(**cp_rec["resources"]) if False else None
            # The force_action already consumed resources; we track cost from the
            # post_runtime resources vs a reconstructed "before" state.
            # For simplicity and consistency with the oracle script, we compute
            # the total utility from the rollout including all step costs.
            # The forced action's step cost is included in the rollout's
            # realized utility calculation (we start from 0.0 and subtract
            # each step's cost, including the first downstream step).
            # For terminal forced actions, we compute utility directly.

            if forced_result.terminal:
                # Terminal forced action — compute utility directly
                tr = utility.terminal_reward(action, bool(forced_result.success))
                terminal_utility = tr
                success = bool(forced_result.success)
                terminal_action = iv.action
                downstream_actions = []
                model_call_receipts = []
                premature_defer = forced_result.premature_defer
                premature_answer = forced_result.premature_answer
                resource_exhaustion = forced_result.resource_exhaustion
                loop = False
                rollout_status = "terminal_forced"
                steps_to_terminal = 1
            else:
                # Non-terminal — continue with pinned Qwen policy
                (terminal_utility, success, terminal_action, downstream_actions,
                 model_call_receipts, premature_defer, premature_answer,
                 resource_exhaustion, loop, rollout_status) = pinned_policy_rollout(
                    post_runtime, task, backend, i3_7e, utility,
                    max_steps=args.max_rollout_steps,
                    checkpoint_id=cp_id,
                    forced_action=iv.action,
                    writer=writer,
                )
                # Add forced action step cost
                # The rollout starts after the forced action, so we need to
                # account for the forced action's resource cost separately.
                # We reconstruct the "before" resource state from the checkpoint.
                from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
                budget = ResourceBudget(
                    max_executive_steps=10, max_retrieval_calls=3,
                    max_search_calls=2, max_verification_calls=5,
                )
                forced_resources_before = ResourceState(
                    budget=budget,
                    executive_steps_used=cp_rec["resources"].get("executive_steps_used", 0),
                    reasoning_tokens_used=cp_rec["resources"].get("reasoning_tokens_used", 0),
                    retrieval_calls_used=cp_rec["resources"].get("retrieval_calls_used", 0),
                    verification_calls_used=cp_rec["resources"].get("verification_calls_used", 0),
                    search_calls_used=cp_rec["resources"].get("search_calls_used", 0),
                    elapsed_ms=cp_rec["resources"].get("elapsed_ms", 0),
                    monetary_cost_microusd=cp_rec["resources"].get("monetary_cost_microusd", 0),
                    policy_rejections_used=cp_rec["resources"].get("policy_rejections_used", 0),
                )
                forced_step_cost = utility.action_cost(forced_resources_before, post_runtime.resources)
                terminal_utility -= forced_step_cost
                steps_to_terminal = len(downstream_actions) + 1

                if rollout_status == "backend_error":
                    n_backend_errors += 1
                elif rollout_status == "decoder_error":
                    n_decoder_errors += 1

            # Get oracle Q for comparison
            oracle_rec = oracle_lookup.get(cp_id, {}).get(iv.action, {})
            oracle_q = oracle_rec.get("terminal_utility")
            oracle_success = oracle_rec.get("success")

            # Build causal record
            record = {
                "checkpoint_id": cp_id,
                "task_id": task_id,
                "category": cp_rec["category"],
                "correct_first_action": cp_rec["correct_first_action"],
                "expected_terminal": cp_rec["expected_terminal"],
                "forced_action": iv.action,
                "target_evidence_id": iv.target_evidence_id,
                "intervention_type": iv.intervention_type,
                "state_features": cp_rec["state_features"],
                "state_sha256": cp_rec["state_sha256"],
                "pinned_policy_utility": round(float(terminal_utility), 4),
                "pinned_policy_success": bool(success),
                "pinned_policy_terminal": terminal_action is not None,
                "pinned_policy_terminal_action": terminal_action,
                "pinned_policy_steps": steps_to_terminal,
                "pinned_policy_downstream_actions": downstream_actions,
                "pinned_policy_premature_defer": bool(premature_defer),
                "pinned_policy_premature_answer": bool(premature_answer),
                "pinned_policy_resource_exhaustion": bool(resource_exhaustion),
                "pinned_policy_loop": bool(loop),
                "pinned_policy_status": rollout_status,
                "pinned_policy_n_model_calls": len(model_call_receipts),
                "oracle_q": oracle_q,
                "oracle_success": oracle_success,
                "downstream_policy": "QWEN_PINNED_V1",
                "schedule_id": schedule.schedule_id,
                "backend_identity_sha256": backend_sha,
                "timestamp": timestamp,
            }
            writer.write_result(record)

            # Write receipt
            writer.write_receipt({
                "checkpoint_id": cp_id,
                "task_id": task_id,
                "forced_action": iv.action,
                "intervention_type": iv.intervention_type,
                "state_sha_before": cp_rec["state_sha256"],
                "pinned_policy_utility": record["pinned_policy_utility"],
                "pinned_policy_success": record["pinned_policy_success"],
                "pinned_policy_terminal_action": terminal_action,
                "pinned_policy_downstream_actions": downstream_actions,
                "pinned_policy_status": rollout_status,
                "backend_identity_sha256": backend_sha,
                "timestamp": timestamp,
            })

            n_done += 1
            if success:
                n_success += 1
            if terminal_action is not None:
                n_terminal += 1
            if premature_defer:
                n_premature_defer += 1
            if premature_answer:
                n_premature_answer += 1

            if n_done % 25 == 0:
                elapsed = time.time() - start_time
                rate = (n_done - len(completed_keys)) / elapsed if elapsed > 0 else 0
                remaining = n_total - n_done
                eta = remaining / rate if rate > 0 else 0
                print(f"  Progress: {n_done}/{n_total} "
                      f"({rate:.1f}/s, ETA {eta:.0f}s) "
                      f"success={n_success} pd={n_premature_defer} pa={n_premature_answer} "
                      f"be={n_backend_errors} de={n_decoder_errors}")

    # Compute manifest
    print(f"\n{'='*60}")
    print(f"Pinned-Policy Causal Data Collection Complete")
    print(f"{'='*60}")
    print(f"  Records:           {n_done}")
    print(f"  Success rate:      {n_success/max(n_done,1):.4f}")
    print(f"  Terminal rate:     {n_terminal/max(n_done,1):.4f}")
    print(f"  Premature DEFER:   {n_premature_defer/max(n_done,1):.4f}")
    print(f"  Premature ANSWER:  {n_premature_answer/max(n_done,1):.4f}")
    print(f"  Backend errors:    {n_backend_errors}")
    print(f"  Decoder errors:    {n_decoder_errors}")

    # Compute dataset SHA
    results_path = output_dir / "pinned_causal_actions_v1.jsonl"
    dataset_sha = hashlib.sha256()
    with open(results_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            dataset_sha.update(chunk)

    manifest = {
        "binding_id": "I3_5_PINNED_POLICY_V1",
        "n_records": n_done,
        "n_success": n_success,
        "n_terminal": n_terminal,
        "n_premature_defer": n_premature_defer,
        "n_premature_answer": n_premature_answer,
        "n_backend_errors": n_backend_errors,
        "n_decoder_errors": n_decoder_errors,
        "overall_success_rate": round(n_success / max(n_done, 1), 4),
        "terminal_rate": round(n_terminal / max(n_done, 1), 4),
        "premature_defer_rate": round(n_premature_defer / max(n_done, 1), 4),
        "premature_answer_rate": round(n_premature_answer / max(n_done, 1), 4),
        "schedule_id": schedule.schedule_id,
        "backend_identity_sha256": backend_sha,
        "downstream_policy": "QWEN_PINNED_V1",
        "dataset_sha256": dataset_sha.hexdigest(),
        "timestamp": timestamp,
        "max_rollout_steps": args.max_rollout_steps,
        "n_ctx": args.n_ctx,
    }
    manifest_path = output_dir / "pinned_causal_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  Manifest:          {manifest_path}")
    print(f"  Dataset SHA:       {dataset_sha.hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
