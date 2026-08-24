#!/usr/bin/env python3
"""DAPH I3.4c — P0/P1/P2 Executive Experiment Runner.

Runs the phase-aware executive experiment with three arms:
  P0: Base LLM only (no phase, no values) — identical to R2 C0
  P1: Phase exposed to LLM
  P2: Phase + learned action-value prior (B1 phase×action table)

Uses the same Qwen2.5-7B-Instruct backend, same dataset, same executor,
same allowed-action computation (C0 arm — no R2 interventions) as R2.
Only the packet differs: P1 adds epistemic_phase, P2 adds
epistemic_phase + action_value_estimates.

Usage:
    PYTHONPATH=scripts:. python3 scripts/run_i3_4_development.py \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/output/ \
        --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --qualification-bundle /path/to/bundle.json
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from daph.phase.ontology import Phase, classify_transition
from daph.phase.classifier import classify_phase
from daph.executive.packet_builder import build_packet, get_phase_from_packet
from daph.value.empirical import GlobalActionMean, PhaseActionTable
from daph.value.dataset import load_transitions, get_action_value_target

from r2_backend_identity import R2_POLICY_BACKEND_V2
from r2_schema import build_action_schema, schema_sha256, verify_schema_invariant, schema_action_enum
from r2_allowed_actions import (
    ACTION_VOCABULARY, compute_allowed_actions,
    AllowedActionDecision, ActionState, R2Arm, C0 as NEUTRAL_ARM,
    EmptyAllowedActionSet, allowed_actions_sha256,
)


class ExecutiveArm(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


ALL_ARMS = (ExecutiveArm.P0, ExecutiveArm.P1, ExecutiveArm.P2)


def load_value_table() -> PhaseActionTable:
    """Load the B1 phase×action table from R2 transitions."""
    transitions_path = REPO_ROOT / "experiments/i3_4/datasets/transitions_r2_dev_v2.jsonl"
    with open(transitions_path) as f:
        transitions = [json.loads(line) for line in f]
    b0 = GlobalActionMean()
    b0.fit(transitions, get_action_value_target)
    b1 = PhaseActionTable(min_samples=3, fallback=b0)
    b1.fit(transitions, get_action_value_target)
    return b1


def _load_i3_7e():
    """Load the i3_7e module."""
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)
    return i3_7e


def _decode_output_strict(raw_output: str):
    """Decode model output using the strict decoder."""
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    return decode_output_strict(raw_output)


@dataclass
class IncrementalWriter:
    """Incremental writer with fsync for crash safety."""
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = open(self.output_dir / "results.jsonl", "a")
        self.model_calls_file = open(self.output_dir / "model_calls.jsonl", "a")
        self.receipts_file = open(self.output_dir / "mechanism_receipts.jsonl", "a")
        self.errors_file = open(self.output_dir / "errors.jsonl", "a")

    def write_result(self, record: dict):
        self.results_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.results_file.flush()
        os.fsync(self.results_file.fileno())

    def write_model_call(self, record: dict):
        self.model_calls_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.model_calls_file.flush()
        os.fsync(self.model_calls_file.fileno())

    def write_receipt(self, record: dict):
        self.receipts_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.receipts_file.flush()
        os.fsync(self.receipts_file.fileno())

    def write_error(self, record: dict):
        self.errors_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.errors_file.flush()
        os.fsync(self.errors_file.fileno())

    def write_progress(self, record: dict):
        with open(self.output_dir / "progress.json", "w") as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def load_completed_keys(self) -> set[str]:
        """Load trajectory keys that have been completed."""
        keys = set()
        results_path = self.output_dir / "results.jsonl"
        if not results_path.exists():
            return keys
        with open(results_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = record.get("trajectory_key")
                    if key and record.get("status") == "completed":
                        keys.add(key)
                except json.JSONDecodeError:
                    continue
        return keys

    def close(self):
        for f in [self.results_file, self.model_calls_file,
                  self.receipts_file, self.errors_file]:
            if f and not f.closed:
                f.close()


@dataclass
class TrajectoryResult:
    trajectory_key: str
    task_id: str
    arm: str
    realized_utility: float
    success: bool
    steps: int
    terminal_action: str | None
    terminal_result: str
    model_calls: int
    decoder_errors: int
    status: str


def run_trajectory(
    task_record: dict,
    arm: ExecutiveArm,
    backend,
    writer: IncrementalWriter,
    value_table: PhaseActionTable,
    *,
    trajectory_key: str,
    max_tokens: int = 128,
    i3_7e=None,
    task_id: str = "",
) -> TrajectoryResult:
    """Run a single trajectory for one executive arm.

    Closely mirrors run_r2_dev_v2.run_trajectory but with I3.4 packet
    modifications (phase + values) instead of R2 interventions.
    """
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
        EvidenceExecutor, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
        initial_evidence_runtime,
    )
    from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    from r2_dataset_generator import R2GoldLabels, generate_r2_dataset

    # Reconstruct task from record (same as R2 runner)
    gold = R2GoldLabels(
        gold_t2=task_record["gold_t2"],
        gold_all_eliminated=task_record["gold_all_eliminated"],
        gold_verify_epistemically_relevant=task_record["gold_verify_epistemically_relevant"],
        gold_should_gate_verify=task_record["gold_should_gate_verify"],
        gold_n_live=task_record["gold_n_live"],
        gold_n_eliminated=task_record["gold_n_eliminated"],
        semantic_error_class=task_record["semantic_error_class"],
        expected_terminal=task_record["expected_terminal"],
        stratum=task_record["stratum"],
        retrieval_budget_case=task_record["retrieval_budget_case"],
        search_budget_case=task_record["search_budget_case"],
    )

    # Load the actual evidence task
    all_tasks = generate_r2_dataset(n_per_stratum=40, seed=137)
    task_lookup = {t.task_id: t for t in all_tasks}
    task = task_lookup.get(task_id)
    if task is None:
        base_task_id = task_id.split("__")[0]
        task = task_lookup.get(base_task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found in dataset")

    et = task.evidence_task
    if et is None:
        raise ValueError(f"Task {task_id} has no evidence_task")

    # Use the SAME budget as R2
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"
    )
    executor = EvidenceExecutor()

    retrieval_used = budget.max_retrieval_calls if task_record.get("retrieval_budget_case") == "exhausted" else 0
    search_used = budget.max_search_calls if task_record.get("search_budget_case") == "exhausted" else 0

    resources = ResourceState(
        budget,
        retrieval_calls_used=retrieval_used,
        search_calls_used=search_used,
    )
    runtime = initial_evidence_runtime(et, resources)

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal_action = None
    terminal_result = "STEP_LIMIT"
    decoder_errors = 0
    status = "completed"

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps
    n_hypotheses = len(et.hypotheses)

    for step_id in range(max_steps):
        # Build snapshot
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Compute T2 from snapshot
        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

        # Build action state
        action_state = ActionState(
            t2=t2,
            executive_steps_remaining=max_steps - step_id,
            can_retrieve=evidence_snapshot.can_retrieve,
            can_search=evidence_snapshot.can_search,
            can_verify=evidence_snapshot.can_verify,
        )

        # Compute allowed actions (always NEUTRAL_ARM = C0 — no R2 interventions)
        try:
            allowed_decision = compute_allowed_actions(action_state, NEUTRAL_ARM)
        except EmptyAllowedActionSet as exc:
            writer.write_error({
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.value,
                "step": step_id,
                "error": "EmptyAllowedActionSet",
                "error_detail": str(exc),
            })
            terminal_result = "EMPTY_ALLOWED_ACTION_SET"
            status = "empty_allowed_set"
            break

        # Build base packet (MDSG state with affordances)
        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

        # Classify phase from packet
        phase = get_phase_from_packet(packet)

        # Build arm-specific packet
        legal_actions = sorted(allowed_decision.allowed)
        features = {
            "n_live": n_hypotheses - len(eliminated),
            "n_eliminated": len(eliminated),
            "n_total": n_hypotheses,
            "decision_state": packet.get("decision_state_summary", {}).get("decision_state", ""),
            "t2": t2,
            "can_retrieve": evidence_snapshot.can_retrieve,
            "can_search": evidence_snapshot.can_search,
            "can_verify": evidence_snapshot.can_verify,
        }

        modified_packet = build_packet(
            packet,
            arm=arm.value,
            phase=phase,
            value_table=value_table if arm == ExecutiveArm.P2 else None,
            legal_actions=legal_actions,
            features=features,
        )

        # Build dynamic schema from allowed actions
        schema = build_action_schema(allowed_decision.allowed)
        verify_schema_invariant(schema, allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        allowed_sha = allowed_actions_sha256(allowed_decision.allowed)

        user_prompt = i3_7e.evidence_packet_json(modified_packet)

        # Pre-state
        ds_internal = packet.get("decision_state_summary", {}).get("decision_state", "UNKNOWN")
        pre_state = {
            "decision_state": ds_internal,
            "t2": t2,
            "n_live": n_hypotheses - len(eliminated),
            "n_eliminated": len(eliminated),
            "phase": phase.value,
        }
        pre_state_sha = hashlib.sha256(
            json.dumps(pre_state, sort_keys=True).encode()
        ).hexdigest()

        # Receipt
        receipt: dict[str, Any] = {
            "task_id": task_id,
            "arm": arm.value,
            "trajectory_key": trajectory_key,
            "step": step_id,
            "phase_before": phase.value,
            "decision_state_internal": ds_internal,
            "decision_state_exposed": ds_internal,  # no relabeling in I3.4
            "t2": t2,
            "n_live_hypotheses": n_hypotheses - len(eliminated),
            "n_eliminated_hypotheses": len(eliminated),
            "legal_actions": sorted(allowed_decision.legal),
            "epistemically_admissible_actions": sorted(allowed_decision.epistemically_admissible),
            "allowed_actions": legal_actions,
            "allowed_actions_sha256": allowed_sha,
            "verify_gate_condition_active": allowed_decision.verify_gate_condition_active,
            "verify_removed_by_epistemic_gate": allowed_decision.verify_removed_by_epistemic_gate,
            "verify_gate_reason": allowed_decision.verify_gate_reason,
            "schema_sha256": schema_sha,
            "schema_action_enum": schema_action_enum(schema),
            "pre_state_sha256": pre_state_sha,
            "pre_state": pre_state,
            "packet_has_phase": "epistemic_phase" in modified_packet,
            "packet_has_values": "action_value_estimates" in modified_packet,
        }

        # Generate
        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=max_tokens,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            receipt.update({
                "result_class": "backend_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_output": "",
                "decoder_valid": False,
                "selected_action": None,
                "execution_outcome": "BACKEND_ERROR",
            })
            writer.write_receipt(receipt)
            writer.write_error({
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.value,
                "step": step_id,
                "error": "BackendError",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            terminal_result = "BACKEND_ERROR"
            status = "backend_error"
            break

        raw_output = call_result.raw_output

        # Decode
        outcome = _decode_output_strict(raw_output)

        # Check schema validity
        schema_enum = set(schema_action_enum(schema))
        schema_valid = (
            outcome.valid
            and outcome.parsed_json is not None
            and outcome.parsed_json.get("action") in schema_enum
        )

        schema_gate_violation = False
        if outcome.valid and outcome.parsed_json is not None:
            parsed_action = outcome.parsed_json.get("action")
            if parsed_action and parsed_action not in allowed_decision.allowed:
                schema_gate_violation = True

        if outcome.valid and outcome.proposal:
            proposal = outcome.proposal
            action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)

            admissibility_passed = action_str in allowed_decision.allowed
            executor_admissibility_violation = not admissibility_passed

            post_state = {
                "decision_state": ds_internal,
                "t2": t2,
                "n_live": n_hypotheses - len(eliminated),
                "n_eliminated": len(eliminated),
            }
            post_state_sha = hashlib.sha256(
                json.dumps(post_state, sort_keys=True).encode()
            ).hexdigest()

            receipt.update({
                "result_class": "success",
                "raw_output": raw_output,
                "provider_raw_output": getattr(call_result, "provider_raw_output", raw_output),
                "decoder_valid": outcome.valid,
                "decoder_rejection_code": outcome.rejection_code,
                "schema_valid": schema_valid,
                "schema_gate_violation": schema_gate_violation,
                "executor_admissibility_violation": executor_admissibility_violation,
                "selected_action": action_str,
                "selected_reason_code": proposal.reason_code,
                "selected_target_id": getattr(proposal, "target_id", None),
                "admissibility_assertion_passed": admissibility_passed,
                "json_schema_sha256": getattr(call_result, "json_schema_sha256", ""),
                "prompt_tokens": getattr(call_result, "prompt_tokens", 0),
                "completion_tokens": getattr(call_result, "completion_tokens", 0),
                "latency_ms": getattr(call_result, "latency_ms", 0),
                "model_name": getattr(call_result, "model_name", ""),
                "finish_reason": getattr(call_result, "finish_reason", ""),
                "post_state_sha256": post_state_sha,
                "post_state": post_state,
                "execution_outcome": "SUCCESS",
            })
            writer.write_receipt(receipt)
            writer.write_model_call(receipt)

            if not admissibility_passed:
                writer.write_error({
                    "trajectory_key": trajectory_key,
                    "task_id": task_id,
                    "arm": arm.value,
                    "step": step_id,
                    "error": "EpistemicAdmissibilityViolation",
                    "action": action_str,
                    "allowed": legal_actions,
                })
                terminal_result = "ADMISSIBILITY_VIOLATION"
                status = "admissibility_violation"
                break
        else:
            decoder_errors += 1
            receipt.update({
                "result_class": "decoder_error",
                "raw_output": raw_output,
                "decoder_valid": False,
                "decoder_rejection_code": outcome.rejection_code,
                "schema_valid": schema_valid,
                "schema_gate_violation": schema_gate_violation,
                "executor_admissibility_violation": False,
                "selected_action": None,
                "admissibility_assertion_passed": False,
                "json_schema_sha256": getattr(call_result, "json_schema_sha256", ""),
                "post_state_sha256": None,
                "execution_outcome": "DECODER_ERROR",
            })
            writer.write_receipt(receipt)
            writer.write_model_call(receipt)
            writer.write_error({
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.value,
                "step": step_id,
                "error": "DecoderError",
                "raw_output": raw_output[:200],
                "rejection_code": outcome.rejection_code,
            })
            terminal_result = "DECODER_ERROR"
            status = "decoder_error"
            break

        # Execute action
        action = proposal.action
        target_id = getattr(proposal, "target_id", None)

        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        resources_after = exec_res.runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        action_str = action.value if hasattr(action, "value") else str(action)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)
        runtime = exec_res.runtime
        steps_taken += 1

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            success = bool(exec_res.task_success)
            terminal_result = exec_res.outcome_code
            terminal_action = action_str
            break

    result = TrajectoryResult(
        trajectory_key=trajectory_key,
        task_id=task_id,
        arm=arm.value,
        realized_utility=round(realized, 4),
        success=success,
        steps=steps_taken,
        terminal_action=terminal_action,
        terminal_result=terminal_result,
        model_calls=model_calls,
        decoder_errors=decoder_errors,
        status=status,
    )

    result_record = {
        "trajectory_key": trajectory_key,
        "task_id": task_id,
        "arm": arm.value,
        "realized_utility": result.realized_utility,
        "success": result.success,
        "steps": result.steps,
        "terminal_action": result.terminal_action,
        "terminal_result": result.terminal_result,
        "model_calls": result.model_calls,
        "decoder_errors": result.decoder_errors,
        "status": result.status,
        "stratum": task_record.get("stratum", ""),
    }
    writer.write_result(result_record)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="I3.4c P0/P1/P2 Executive Experiment")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--qualification-bundle", type=Path, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results (skip completed keys)")
    args = parser.parse_args()

    # Preflight
    bundle_path = args.qualification_bundle or (
        REPO_ROOT / "experiments/v2b_i3_15c/development/r2-dev-v2/qualification_evidence/bundle.json"
    )

    from run_r2_dev_v2 import run_preflight
    preflight = run_preflight(
        dataset_path=args.dataset,
        gguf_path=args.gguf_path,
        qualification_bundle_path=bundle_path,
    )
    if not preflight["passed"]:
        sys.exit(1)

    dataset_sha = preflight["dataset_sha"]
    backend_sha = R2_POLICY_BACKEND_V2.identity_sha256()

    # Load dataset
    with open(args.dataset) as f:
        task_records = [json.loads(line) for line in f]
    print(f"\nDataset: {len(task_records)} tasks")
    print(f"Dataset SHA: {dataset_sha[:16]}...")
    print(f"Backend identity SHA: {backend_sha[:16]}...")

    # Load value table
    print("Loading B1 phase×action value table...")
    value_table = load_value_table()
    print(f"  {len(value_table._values)} entries")

    # Load i3_7e
    print("Loading i3_7e...")
    i3_7e = _load_i3_7e()

    # Initialize backend
    print("Initializing R2DirectLlamaBackend...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
    )

    # Output directory
    args.output.mkdir(parents=True, exist_ok=True)
    writer = IncrementalWriter(args.output)

    # Resume logic
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} trajectories already completed")

    # Build schedule (Latin-square ordering)
    n_tasks = len(task_records)
    n_arms = len(ALL_ARMS)
    total = n_tasks * n_arms
    print(f"\nTotal trajectories: {total} ({n_tasks} tasks × {n_arms} arms)")

    schedule = []
    for task_idx in range(n_tasks):
        for arm_idx in range(n_arms):
            arm = ALL_ARMS[(task_idx + arm_idx) % n_arms]
            task_id = task_records[task_idx]["task_id"]
            key = f"{dataset_sha}|{task_id}|{arm.value}|{backend_sha}|i3_4c"
            schedule.append({
                "task_index": task_idx,
                "task_id": task_id,
                "arm": arm.value,
                "trajectory_key": key,
            })

    with open(args.output / "execution_schedule.json", "w") as f:
        json.dump({"schedule": schedule, "total": total}, f, indent=2)

    # Run
    print(f"\nStarting I3.4c experiment...")
    print("=" * 60)

    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, entry in enumerate(schedule):
        task_idx = entry["task_index"]
        task_id = entry["task_id"]
        arm_name = entry["arm"]
        traj_key = entry["trajectory_key"]

        # Resume check
        if traj_key in completed_keys:
            skipped += 1
            continue

        arm = ExecutiveArm(arm_name)
        task_record = task_records[task_idx]

        elapsed = time.time() - start_time
        if completed > 0:
            eta = elapsed / completed * (total - completed - skipped - failed)
        else:
            eta = 0
        print(f"[{i+1}/{total}] task={task_id} arm={arm_name} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        try:
            result = run_trajectory(
                task_record=task_record,
                arm=arm,
                backend=backend,
                writer=writer,
                value_table=value_table,
                trajectory_key=traj_key,
                max_tokens=args.max_tokens,
                i3_7e=i3_7e,
                task_id=task_id,
            )
            completed += 1
            if result.status != "completed":
                failed += 1
                print(f"  → {result.status}: {result.terminal_result}")
            else:
                print(f"  → utility={result.realized_utility:.2f}, "
                      f"terminal={result.terminal_action}, "
                      f"steps={result.steps}, success={result.success}")
        except Exception as exc:
            failed += 1
            print(f"  → ERROR: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            writer.write_error({
                "trajectory_key": traj_key,
                "task_id": task_id,
                "arm": arm_name,
                "error": type(exc).__name__,
                "error_message": str(exc),
            })

        writer.write_progress({
            "completed": completed,
            "failed": failed,
            "total": total,
            "remaining": total - completed - failed,
            "elapsed_seconds": time.time() - start_time,
        })

    writer.close()

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("I3.4c EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Total:     {total}")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {total_time:.0f}s")

    manifest = {
        "experiment": "i3_4c_p0_p1_p2",
        "dataset_sha256": dataset_sha,
        "backend_identity_sha256": backend_sha,
        "total_trajectories": total,
        "completed": completed,
        "failed": failed,
        "run_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "run_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_time_seconds": total_time,
        "arms": [a.value for a in ALL_ARMS],
        "value_model": "B1_phase_action_table",
    }
    with open(args.output / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nResults in: {args.output}")


if __name__ == "__main__":
    main()
