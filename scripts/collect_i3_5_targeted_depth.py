#!/usr/bin/env python3
"""I3.5-PQ Phase 22 Targeted: Temporal-effect collection for V2 decision.

Collects depth-varying causal data only for the categories where
temporal effects could matter, using only the key continuation actions.

Categories (skip ol_retrieve — already collected):
  ol_defer, ol_verify, ol_search, tl_retrieve, tl_verify, tl_search, tl_defer

Actions (only continuation actions that could show depth effects):
  RETRIEVE, VERIFY, SEARCH_MORE

Depths: 0, 1, 2

Target: ~20-25 matched checkpoints per category.

Uses H=8 downstream rollout (matching V1 collection).
Uses frozen Qwen2.5-7B-Instruct-Q4_K_M at temperature=0.

Usage:
  PYTHONPATH=. python3 scripts/collect_i3_5_targeted_depth.py \
    --gguf-path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_5/pinned_policy_targeted_depth \
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


# Categories to collect (skip ol_retrieve — already have it)
TARGET_CATEGORIES = {
    "ol_defer", "ol_verify", "ol_search",
    "tl_retrieve", "tl_verify", "tl_search", "tl_defer",
}

# Only continuation actions that could show depth effects
TARGET_ACTIONS = {"RETRIEVE", "VERIFY", "SEARCH_MORE"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class TargetedWriter:
    output_dir: Path

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actions_file = open(self.output_dir / "targeted_depth_actions_v1.jsonl", "a")
        self.checkpoints_file = open(self.output_dir / "targeted_depth_checkpoints_v1.jsonl", "a")
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
        path = self.output_dir / "targeted_depth_actions_v1.jsonl"
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
    """Create checkpoints at retrieval depths 0, 1, ..., max_depth."""
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from daph.intervention.checkpoint import create_checkpoint

    executor = EvidenceExecutor()
    checkpoints = []
    runtime = initial_runtime
    prior_actions = []

    for depth in range(max_depth + 1):
        cp = create_checkpoint(
            runtime,
            step=depth,
            phase=f"RETRIEVE_DEPTH_{depth}",
            prior_actions=tuple(prior_actions),
            prior_outcomes=(),
        )
        checkpoints.append((depth, cp))

        if depth < max_depth:
            sf = cp.state_features
            retrieval_remaining = sf.get("retrieval_remaining", 0)
            if retrieval_remaining <= 0:
                break
            try:
                result = executor.execute(runtime, DecisionAction.RETRIEVE)
                runtime = result.runtime
                prior_actions.append("RETRIEVE")
            except Exception:
                break

    return checkpoints


def pinned_policy_rollout(
    post_runtime,
    task,
    backend,
    i3_7e,
    utility,
    max_steps: int = 8,
    checkpoint_id: str = "",
    forced_action: str = "",
    writer: TargetedWriter | None = None,
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


def get_category(task_id: str) -> str:
    parts = task_id.split("_")
    return "_".join(parts[2:4])


def main():
    parser = argparse.ArgumentParser(description="I3.5-PQ Phase 22 Targeted Depth Collection")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--model-name", default="qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="experiments/i3_5/pinned_policy_targeted_depth")
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
    # Filter to target categories only
    tasks = [t for t in tasks if get_category(t.task_id) in TARGET_CATEGORIES]
    print(f"  {len(tasks)} tasks (target categories only)")

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
    run_id = hashlib.sha256(f"i3_5_targeted_depth:{timestamp}".encode()).hexdigest()[:16]

    source_shas = {}
    for fname in ["run_i3_7e_compact_governor.py", "r2_schema.py", "r2_allowed_actions.py"]:
        path = REPO_ROOT / "scripts" / fname
        if path.exists():
            source_shas[fname] = sha256_bytes(path.read_bytes())

    model_sha = hashlib.sha256(
        Path(args.gguf_path).read_bytes()[:65536]
    ).hexdigest()

    manifest = {
        "run_id": run_id,
        "experiment": "I3.5-PQ Phase 22 Targeted Temporal-Effect Collection",
        "timestamp": timestamp,
        "parameters": {
            "max_depth": args.max_depth,
            "max_rollout_steps": args.max_rollout_steps,
            "target_categories": sorted(TARGET_CATEGORIES),
            "target_actions": sorted(TARGET_ACTIONS),
        },
        "binding": {
            "model_name": args.model_name,
            "gguf_path": args.gguf_path,
            "model_sha256_prefix": model_sha,
            "temperature": 0.0,
        },
        "source_shas": source_shas,
        "v2_decision_rule": {
            "build_v2_only_if": [
                ">=30% of tested category/action pairs show |Delta_Q| > 5",
                "the effect is reproducible across matched states",
                "V1 fails to reflect those effects",
                "that V1 error causes live-policy harm that I2 does not already prevent",
            ],
            "note": "Condition 4 matters most. A richer Q model is only justified if it solves a remaining failure.",
        },
    }
    manifest_path = output_dir / "run_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nRun manifest: {manifest_path}")
    print(f"  Run ID: {run_id}")
    print(f"  Target categories: {sorted(TARGET_CATEGORIES)}")
    print(f"  Target actions: {sorted(TARGET_ACTIONS)}")
    print(f"  Max depth: {args.max_depth}")

    # Setup writer
    writer = TargetedWriter(output_dir)
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} actions already completed")

    # Generate depth checkpoints and collect causal data
    from daph.intervention.checkpoint import StateCheckpoint, compute_state_features
    from daph.intervention.restore import restore_runtime
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor, valid_verify_targets
    from hrm_adaptive_memory.executive.evidence_benchmark import initial_evidence_runtime
    from hrm_adaptive_memory.executive.resources import ResourceState

    n_done = len(completed_keys)
    start_time = time.time()

    print(f"\nStarting targeted depth collection...")

    for task in tasks:
        resources = ResourceState(budget=budget)
        initial_runtime = initial_evidence_runtime(task, resources)
        depth_cps = create_depth_checkpoints(task, initial_runtime, max_depth=args.max_depth)

        for depth, cp in depth_cps:
            sf = cp.state_features
            legal = set(cp.legal_actions)
            if sf.get("retrieval_remaining", 0) > 0:
                legal.add("RETRIEVE")
            legal = sorted(legal)

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
                "category": get_category(task.task_id),
            }
            writer.write_checkpoint(cp_dict)

            for action_str in sorted(TARGET_ACTIONS):
                if action_str not in legal:
                    continue

                action_key = f"{cp.checkpoint_id}:{action_str}"
                if action_key in completed_keys:
                    continue

                action = DecisionAction(action_str)
                target_eid = None
                if action is DecisionAction.VERIFY:
                    runtime = restore_runtime(cp, task)
                    valid_targets = valid_verify_targets(runtime)
                    if valid_targets:
                        target_eid = valid_targets[0]
                    else:
                        continue

                runtime = restore_runtime(cp, task)
                executor = EvidenceExecutor()

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

                if forced_result.terminal:
                    tr = utility.terminal_reward(action, bool(forced_result.task_success))
                    realized_utility = tr
                    success = bool(forced_result.task_success)
                    terminal_action = action_str
                    downstream_actions = []
                    premature_defer = (action is DecisionAction.DEFER and not forced_result.task_success)
                    premature_answer = (action is DecisionAction.ANSWER and not forced_result.task_success)
                    terminal_result = forced_result.outcome_code
                else:
                    (realized_utility, success, terminal_action, downstream_actions,
                     premature_defer, premature_answer, terminal_result) = pinned_policy_rollout(
                        post_runtime, task, backend, i3_7e, utility,
                        max_steps=args.max_rollout_steps,
                        checkpoint_id=cp.checkpoint_id,
                        forced_action=action_str,
                        writer=writer,
                    )

                record = {
                    "checkpoint_id": cp.checkpoint_id,
                    "task_id": task.task_id,
                    "depth": depth,
                    "category": get_category(task.task_id),
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

                if n_done % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = (n_done - len(completed_keys)) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {n_done} interventions ({rate:.1f}/s) "
                          f"last: {task.task_id}:{action_str} d={depth} "
                          f"U={realized_utility:.1f} success={success}")

    print(f"\n{'='*70}")
    print(f"Targeted Depth Collection Complete")
    print(f"{'='*70}")
    print(f"  Total interventions: {n_done}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
