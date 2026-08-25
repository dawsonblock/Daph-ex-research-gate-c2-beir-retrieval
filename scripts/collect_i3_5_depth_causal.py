#!/usr/bin/env python3
"""I3.5-PQ Phase 22: Collect depth-varying pinned-policy causal data.

Generates checkpoint states at retrieval depths 0, 1, 2 by executing
RETRIEVE actions from initial states, then forces all legal actions
at each depth to collect pinned-policy utility.

This produces matched action-depth panels:
  Q(s_{d=0}, RETRIEVE), Q(s_{d=0}, VERIFY), Q(s_{d=0}, SEARCH_MORE), ...
  Q(s_{d=1}, RETRIEVE), Q(s_{d=1}, VERIFY), Q(s_{d=1}, SEARCH_MORE), ...
  Q(s_{d=2}, RETRIEVE), Q(s_{d=2}, VERIFY), Q(s_{d=2}, SEARCH_MORE), ...

The frozen Qwen2.5-7B-Instruct-Q4_K_M policy is used for downstream
rollouts, same as the original V1 collection.

Also collects verify-depth panels (depths 0, 1) where possible.

Usage:
  PYTHONPATH=. python3 scripts/collect_i3_5_depth_causal.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/pinned_policy_depth \
    --resume
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class DepthWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actions_file = open(self.output_dir / "depth_causal_actions_v1.jsonl", "a")
        self.checkpoints_file = open(self.output_dir / "depth_checkpoints_v1.jsonl", "a")
        self.model_call_file = open(self.output_dir / "model_calls_v1.jsonl", "a")
        self.error_file = open(self.output_dir / "errors_v1.jsonl", "a")

    def _write(self, fh, record: dict):
        fh.write(json.dumps(record, sort_keys=True, default=bool) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def write_action(self, r: dict): self._write(self.actions_file, r)
    def write_checkpoint(self, r: dict): self._write(self.checkpoints_file, r)
    def write_model_call(self, r: dict): self._write(self.model_call_file, r)
    def write_error(self, r: dict): self._write(self.error_file, r)

    def load_completed_keys(self) -> set[str]:
        completed = set()
        path = self.output_dir / "depth_causal_actions_v1.jsonl"
        if not path.exists():
            return completed
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(f"{r['checkpoint_id']}:{r['forced_action']}")
                except (json.JSONDecodeError, KeyError):
                    continue
        return completed


def create_depth_checkpoints(task, initial_runtime, max_depth: int = 2):
    """Create checkpoints at retrieval depths 0, 1, ..., max_depth.

    Executes RETRIEVE actions from the initial state to create
    states at increasing retrieval depths.

    Returns a list of (depth, checkpoint) pairs.
    """
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from daph.intervention.checkpoint import create_checkpoint

    executor = EvidenceExecutor()
    checkpoints = []
    runtime = initial_runtime
    prior_actions = []

    for depth in range(max_depth + 1):
        # Create checkpoint at current depth
        cp = create_checkpoint(
            runtime,
            step=depth,
            phase=f"RETRIEVE_DEPTH_{depth}",
            prior_actions=tuple(prior_actions),
            prior_outcomes=(),
        )
        checkpoints.append((depth, cp))

        # If we haven't reached max_depth, execute RETRIEVE to advance.
        # We allow no-op RETRIEVEs (retrieval_calls_remaining > 0 but no
        # hidden evidence left) because that's exactly what the live LLM
        # does — it calls RETRIEVE even when nothing new can be found.
        # This is the behavior we want V2 to learn to penalize.
        if depth < max_depth:
            sf = cp.state_features
            retrieval_remaining = sf.get("retrieval_remaining", 0)
            if retrieval_remaining <= 0:
                break  # No retrieval budget left at all

            try:
                result = executor.execute(runtime, DecisionAction.RETRIEVE)
                runtime = result.runtime
                prior_actions.append("RETRIEVE")
            except Exception:
                break  # Can't advance further

    return checkpoints


def force_action_from_runtime(runtime, task, action, target_evidence_id=None):
    """Force an action from a runtime state (not a checkpoint)."""
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor

    executor = EvidenceExecutor()
    result = executor.execute(runtime, action, target_evidence_id=target_evidence_id)
    return result


def pinned_policy_rollout(
    post_runtime,
    task,
    backend,
    i3_7e,
    utility,
    max_steps: int = 8,
    checkpoint_id: str = "",
    forced_action: str = "",
    writer: DepthWriter | None = None,
):
    """Continue from post-forced-action runtime with pinned Qwen policy."""
    from hrm_adaptive_memory.executive.evidence_benchmark import build_evidence_snapshot
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0

    executor = EvidenceExecutor()
    runtime = post_runtime
    prior_actions = []
    prior_outcomes = []
    realized = 0.0
    success = False
    terminal = False
    terminal_action = None
    terminal_result = "STEP_LIMIT"
    premature_defer = False
    premature_answer = False
    downstream_actions = []
    system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

    for step_id in range(max_steps):
        evidence_snapshot = build_evidence_snapshot(
            runtime,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        viability = i3_7e._classify_from_snapshot(evidence_snapshot)
        eliminated = [h_id for h_id, info in viability.items()
                      if info["status"] == "ELIMINATED"]
        n_hypotheses = len(task.hypotheses)
        t2 = (len(eliminated) == n_hypotheses and n_hypotheses > 0)

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
            terminal_result = "EMPTY_ALLOWED_SET"
            break

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        user_prompt = i3_7e.evidence_packet_json(packet)

        try:
            call_result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            if writer:
                writer.write_error({
                    "checkpoint_id": checkpoint_id,
                    "forced_action": forced_action,
                    "step": step_id,
                    "error": "BackendError",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
            terminal_result = "BACKEND_ERROR"
            break

        raw_output = call_result.raw_output
        outcome = decode_output_strict(raw_output)

        if writer:
            writer.write_model_call({
                "checkpoint_id": checkpoint_id,
                "forced_action": forced_action,
                "step": step_id,
                "raw_output": raw_output,
                "decoder_valid": bool(outcome.valid),
                "schema_sha256": schema_sha,
            })

        if not outcome.valid or not outcome.proposal:
            terminal_result = "DECODER_ERROR"
            break

        proposal = outcome.proposal
        action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)
        target_id = getattr(proposal, "target_id", None)

        if action_str not in allowed_decision.allowed:
            terminal_result = "ADMISSIBILITY_VIOLATION"
            break

        action = DecisionAction(action_str)
        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        runtime = exec_res.runtime
        resources_after = runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        downstream_actions.append(action_str)
        prior_actions.append(action_str)
        prior_outcomes.append(exec_res.outcome_code)

        if exec_res.terminal:
            tr = utility.terminal_reward(exec_res.action, bool(exec_res.task_success))
            realized += tr
            success = bool(exec_res.task_success)
            terminal = True
            terminal_action = action_str
            terminal_result = exec_res.outcome_code
            if action is DecisionAction.DEFER and not exec_res.task_success:
                premature_defer = True
            if action is DecisionAction.ANSWER and not exec_res.task_success:
                premature_answer = True
            break

    if not terminal:
        realized -= 0.5

    return (realized, success, terminal_action, downstream_actions,
            premature_defer, premature_answer, terminal_result)


def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 22 Depth-Varying Causal Collection")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/pinned_policy_depth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-rollout-steps", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--n-per-subtype", type=int, default=24)
    parser.add_argument("--n-per-two-live", type=int, default=20)
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load i3_7e
    print("Loading i3_7e module...")
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    sys.modules["i3_7e"] = i3_7e
    spec.loader.exec_module(i3_7e)

    # Load tasks
    print("Loading benchmark tasks...")
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_5_state_discrimination_generator import (
        generate_i3_5_state_discrimination_benchmark,
    )
    tasks = generate_i3_5_state_discrimination_benchmark(
        n_per_subtype=args.n_per_subtype,
        n_per_two_live_subtype=args.n_per_two_live,
        seed=9137,
    )
    print(f"  {len(tasks)} tasks")

    # Load utility
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Load budget
    from hrm_adaptive_memory.executive.resources import ResourceBudget
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )

    # Initialize backend
    print(f"\nInitializing R2DirectLlamaBackend with {args.gguf_path}...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
        n_ctx=args.n_ctx,
    )
    print("  Backend initialized.")

    # Build run manifest
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = hashlib.sha256(f"i3_5_depth_causal:{timestamp}".encode()).hexdigest()[:16]

    # Compute SHAs for binding
    source_shas = {}
    for fname in ["run_i3_7e_compact_governor.py", "r2_schema.py", "r2_allowed_actions.py"]:
        path = REPO_ROOT / "scripts" / fname
        if path.exists():
            source_shas[fname] = sha256_bytes(path.read_bytes())

    model_sha = hashlib.sha256(
        Path(args.gguf_path).read_bytes()[:65536]  # first 64KB for speed
    ).hexdigest()

    manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 22 Depth-Varying Causal Collection",
        "timestamp": timestamp,
        "parameters": {
            "max_depth": args.max_depth,
            "max_rollout_steps": args.max_rollout_steps,
            "n_per_subtype": args.n_per_subtype,
            "n_per_two_live": args.n_per_two_live,
        },
        "binding": {
            "model_name": args.model_name,
            "gguf_path": args.gguf_path,
            "model_sha256_prefix": model_sha,
            "temperature": 0.0,
        },
        "source_shas": source_shas,
        "hypothesis": (
            "Adding causal intervention coverage at repeated-action depths "
            "will teach the existing state-conditioned estimator diminishing "
            "marginal returns, reducing redundant RETRIEVE behavior without "
            "degrading action selection on original states."
        ),
        "target_panels": {
            "retrieval_depths": [0, 1, 2],
            "actions_per_depth": ["RETRIEVE", "VERIFY", "SEARCH_MORE", "DEFER", "ANSWER"],
            "note": "Only legal actions are forced at each depth",
        },
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Max depth: {args.max_depth}")

    # Setup writer
    writer = DepthWriter(output_dir)
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} actions already completed")

    # Load initial checkpoints
    from daph.intervention.checkpoint import StateCheckpoint, compute_state_features, compute_legal_actions
    from daph.intervention.restore import restore_runtime
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor, valid_verify_targets
    from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime

    # Generate depth checkpoints and collect causal data
    all_actions = ["RETRIEVE", "VERIFY", "SEARCH_MORE", "DEFER", "ANSWER"]
    n_done = len(completed_keys)
    n_total_estimate = len(tasks) * (args.max_depth + 1) * len(all_actions)
    start_time = time.time()

    print(f"\nStarting depth-varying causal collection...")
    print(f"  Estimated total interventions: ~{n_total_estimate}")
    print()

    for task in tasks:
        # Create initial runtime
        from hrm_adaptive_memory.executive.resources import ResourceState
        resources = ResourceState(budget=budget)
        initial_runtime = initial_evidence_runtime(task, resources)

        # Create depth checkpoints by executing RETRIEVEs
        depth_cps = create_depth_checkpoints(task, initial_runtime, max_depth=args.max_depth)

        for depth, cp in depth_cps:
            # Override legal actions to match the live runner's definition.
            # The live runner uses the snapshot's can_retrieve (budget-only),
            # not the checkpoint's can_retrieve (budget + hidden evidence).
            # This is critical: we need to force RETRIEVE at depth 1 and 2
            # even when there's no hidden evidence, because that's exactly
            # what the live LLM does.
            sf = cp.state_features
            legal = set(cp.legal_actions)
            # Add RETRIEVE if budget allows (matching live runner behavior)
            if sf.get("retrieval_remaining", 0) > 0:
                legal.add("RETRIEVE")
            # Remove RETRIEVE if no budget
            if sf.get("retrieval_remaining", 0) <= 0:
                legal.discard("RETRIEVE")
            legal = sorted(legal)

            # Write checkpoint
            cp_dict = {
                "checkpoint_id": cp.checkpoint_id,
                "task_id": task.task_id,
                "depth": depth,
                "step": cp.step,
                "phase": cp.phase,
                "hypotheses": list(cp.hypotheses),
                "evidence": list(cp.evidence),
                "state_features": cp.state_features,
                "resources": cp.resources,
                "legal_actions": legal,
                "state_sha256": cp.state_sha256,
                "prior_actions": list(cp.prior_actions),
                "prior_outcomes": list(cp.prior_outcomes),
                "category": task.task_id.split("_")[2] + "_" + task.task_id.split("_")[3],
            }
            writer.write_checkpoint(cp_dict)

            # For each legal action, force it and collect utility
            for action_str in all_actions:
                if action_str not in legal:
                    continue

                action_key = f"{cp.checkpoint_id}:{action_str}"
                if action_key in completed_keys:
                    continue

                action = DecisionAction(action_str)

                # For VERIFY, need a target evidence ID
                target_eid = None
                if action is DecisionAction.VERIFY:
                    runtime = restore_runtime(cp, task)
                    valid_targets = valid_verify_targets(runtime)
                    if valid_targets:
                        target_eid = valid_targets[0]
                    else:
                        continue  # No valid verify target

                # Restore runtime from checkpoint
                runtime = restore_runtime(cp, task)
                executor = EvidenceExecutor()

                # Execute forced action
                try:
                    forced_result = executor.execute(runtime, action, target_evidence_id=target_eid)
                except Exception as exc:
                    writer.write_error({
                        "checkpoint_id": cp.checkpoint_id,
                        "task_id": task.task_id,
                        "depth": depth,
                        "forced_action": action_str,
                        "error": "ForceActionError",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": timestamp,
                    })
                    n_done += 1
                    continue

                post_runtime = forced_result.runtime

                # Compute utility
                if forced_result.terminal:
                    # Terminal action (ANSWER, DEFER)
                    tr = utility.terminal_reward(action, bool(forced_result.task_success))
                    realized_utility = tr
                    success = bool(forced_result.task_success)
                    terminal_action = action_str
                    downstream_actions = []
                    premature_defer = (action is DecisionAction.DEFER and not forced_result.task_success)
                    premature_answer = (action is DecisionAction.ANSWER and not forced_result.task_success)
                    terminal_result = forced_result.outcome_code
                else:
                    # Non-terminal — continue with pinned Qwen policy
                    (realized_utility, success, terminal_action, downstream_actions,
                     premature_defer, premature_answer, terminal_result) = pinned_policy_rollout(
                        post_runtime, task, backend, i3_7e, utility,
                        max_steps=args.max_rollout_steps,
                        checkpoint_id=cp.checkpoint_id,
                        forced_action=action_str,
                        writer=writer,
                    )

                # Write causal record
                record = {
                    "checkpoint_id": cp.checkpoint_id,
                    "task_id": task.task_id,
                    "depth": depth,
                    "category": task.task_id.split("_")[2] + "_" + task.task_id.split("_")[3],
                    "forced_action": action_str,
                    "target_evidence_id": target_eid,
                    "pinned_policy_utility": round(realized_utility, 4),
                    "success": bool(success),
                    "terminal": forced_result.terminal,
                    "terminal_action": terminal_action,
                    "downstream_actions": downstream_actions,
                    "premature_defer": bool(premature_defer),
                    "premature_answer": bool(premature_answer),
                    "terminal_result": terminal_result,
                    "state_features": sf,
                    "prior_actions": list(cp.prior_actions),
                    "legal_actions": list(legal),
                    "run_id": run_id,
                    "timestamp": timestamp,
                }
                writer.write_action(record)
                n_done += 1

                if n_done % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (n_done - len(completed_keys)) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {n_done} interventions ({rate:.1f}/s) "
                          f"last: {task.task_id}:{action_str} depth={depth} "
                          f"U={realized_utility:.1f} success={success}")

    print(f"\n{'='*70}")
    print(f"Depth-Varying Causal Collection Complete")
    print(f"{'='*70}")
    print(f"  Total interventions: {n_done}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
