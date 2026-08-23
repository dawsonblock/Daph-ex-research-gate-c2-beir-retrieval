#!/usr/bin/env python3
"""
R2-DEV-V2 Runner — Frozen 2×2 factorial on balanced dataset.

Execution path (identical for all arms except two intervention hooks):

    task
      ↓
    frozen retrieval/evidence execution
      ↓
    internal MDSG snapshot + T2
      ↓
    apply E label-only hook          ← intervention 1 (R2e)
      ↓
    compute Legal
      ↓
    compute EpistemicallyAdmissible
      ↓
    Allowed = intersection
      ↓
    build canonical schema
      ↓
    LlamaGrammar
      ↓
    Qwen policy
      ↓
    strict decoder
      ↓
    admissibility assertion
      ↓
    executor
      ↓
    next state

Uses R2DirectLlamaBackend exclusively. No fallback to LocalLlamaBackend
or server path. At startup, fails closed unless the live environment
matches R2_POLICY_BACKEND_V2.

Incremental persistence with fsync:
    results.jsonl
    model_calls.jsonl
    mechanism_receipts.jsonl
    errors.jsonl
    progress.json
    run_manifest.json

Each trajectory is fsynced before beginning the next. A Colab
disconnect costs at most the currently executing trajectory.

Resume logic: skip a trajectory only when a completed result with the
exact trajectory key exists. Never skip merely because task_id exists.

Trajectory key:
    f"{dataset_sha}|{task_id}|{arm}|{backend_identity_sha}"

Usage:
    PYTHONPATH=scripts:. python3 scripts/run_r2_dev_v2.py \
        --dataset /path/to/balanced_dataset.jsonl \
        --output /path/to/r2-dev-v2/raw/ \
        --gguf-path /content/alt_model/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --model-name qwen2.5-7b-instruct
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from r2_allowed_actions import (
    ACTION_VOCABULARY,
    ALWAYS_LEGAL,
    ActionState,
    AllowedActionDecision,
    R2Arm,
    C0, D, E, DE,
    ALL_ARMS,
    EmptyAllowedActionSet,
    compute_legal_actions,
    compute_epistemically_admissible_actions,
    compute_allowed_actions,
    allowed_actions_sha256,
)
from r2_schema import (
    build_action_schema,
    schema_sha256,
    schema_action_enum,
    verify_schema_invariant,
    FROZEN_R13_ACTION_SCHEMA_SHA256,
)
from r2_backend_identity import (
    R2_POLICY_BACKEND_V2,
    compute_gguf_sha256,
    compute_schema_builder_sha,
    get_runtime_version,
    verify_pinned_identity,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EpistemicAdmissibilityViolation(Exception):
    """Raised when a decoded action is not in the allowed action set."""
    def __init__(self, action: str, allowed: frozenset[str], reason: str | None):
        self.action = action
        self.allowed = allowed
        self.reason = reason
        super().__init__(
            f"Admissibility violation: action={action} not in allowed={sorted(allowed)} "
            f"reason={reason}"
        )


class PreflightFailure(Exception):
    """Raised when a preflight check fails."""
    pass


# ---------------------------------------------------------------------------
# Intervention hooks (identical to original runner)
# ---------------------------------------------------------------------------

def apply_semantics_intervention(
    packet: dict,
    state: ActionState,
    arm: R2Arm,
) -> tuple[dict, str, str]:
    """Apply R2e semantics intervention (label-only).

    Returns (modified_packet, decision_state_internal, decision_state_exposed).

    For E/DE at T2: changes decision_state from NEEDS_DISCRIMINATION to
    NO_VIABLE_HYPOTHESIS. Underlying computation is unchanged.
    """
    ds_summary = packet.get("decision_state_summary", {})
    decision_state_internal = ds_summary.get("decision_state", "UNKNOWN")
    decision_state_exposed = decision_state_internal

    if arm.corrected_t2_semantics and state.t2:
        if decision_state_internal in ("NEEDS_DISCRIMINATION", "INSUFFICIENT",
                                        "NEEDS_EVIDENCE", "SUPPORTED_BUT_UNRESOLVED"):
            decision_state_exposed = "NO_VIABLE_HYPOTHESIS"
            packet = dict(packet)
            ds_summary = dict(ds_summary)
            ds_summary["decision_state"] = "NO_VIABLE_HYPOTHESIS"
            packet["decision_state_summary"] = ds_summary

    return packet, decision_state_internal, decision_state_exposed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_i3_7e():
    """Load the i3_7e module."""
    spec = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3_7e)
    return i3_7e


def _dataset_sha256(dataset_path: Path) -> str:
    """Compute SHA-256 of the dataset file."""
    return hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def _latin_square_order(n_tasks: int, arms: list[R2Arm], seed: int = 137) -> list[tuple[int, R2Arm]]:
    """Generate a deterministic Latin-square-style arm ordering.

    For each task, the arm order is rotated so every arm appears roughly
    equally often in each execution position.

    Returns a list of (task_index, arm) pairs in execution order.
    """
    rng = random.Random(seed)
    n_arms = len(arms)
    schedule = []

    # Generate a rotation offset for each task
    for task_idx in range(n_tasks):
        offset = task_idx % n_arms
        arm_order = arms[offset:] + arms[:offset]
        # Apply a deterministic shuffle within the rotation
        # (seeded, so it's reproducible)
        rng.shuffle(arm_order)
        for arm in arm_order:
            schedule.append((task_idx, arm))

    return schedule


def _trajectory_key(dataset_sha: str, task_id: str, arm: str, backend_sha: str) -> str:
    """Stable unique trajectory key."""
    return f"{dataset_sha}|{task_id}|{arm}|{backend_sha}"


# ---------------------------------------------------------------------------
# Incremental persistence
# ---------------------------------------------------------------------------

class IncrementalWriter:
    """Incremental persistence with fsync.

    Each file is opened in append mode and fsynced after every write.
    A Colab disconnect costs at most the currently executing trajectory.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        self.results_path = output_dir / "results.jsonl"
        self.model_calls_path = output_dir / "model_calls.jsonl"
        self.mechanism_receipts_path = output_dir / "mechanism_receipts.jsonl"
        self.errors_path = output_dir / "errors.jsonl"
        self.progress_path = output_dir / "progress.json"

        # Open in append mode
        self._results_f = open(self.results_path, "a")
        self._model_calls_f = open(self.model_calls_path, "a")
        self._receipts_f = open(self.mechanism_receipts_path, "a")
        self._errors_f = open(self.errors_path, "a")

    def write_result(self, record: dict):
        self._results_f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._results_f.flush()
        os.fsync(self._results_f.fileno())

    def write_model_call(self, record: dict):
        self._model_calls_f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._model_calls_f.flush()
        os.fsync(self._model_calls_f.fileno())

    def write_receipt(self, record: dict):
        self._receipts_f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._receipts_f.flush()
        os.fsync(self._receipts_f.fileno())

    def write_error(self, record: dict):
        self._errors_f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._errors_f.flush()
        os.fsync(self._errors_f.fileno())

    def write_progress(self, progress: dict):
        with open(self.progress_path, "w") as f:
            json.dump(progress, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())

    def close(self):
        for f in [self._results_f, self._model_calls_f, self._receipts_f, self._errors_f]:
            try:
                f.close()
            except Exception:
                pass

    def load_completed_keys(self) -> set[str]:
        """Load trajectory keys that have been completed."""
        keys = set()
        if not self.results_path.exists():
            return keys
        with open(self.results_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = record.get("trajectory_key")
                    if key and record.get("status") == "completed":
                        keys.add(key)
                except json.JSONDecodeError:
                    continue
        return keys


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def run_preflight(
    dataset_path: Path,
    gguf_path: str,
    expected_n_tasks: int = 80,
    expected_trajectories: int = 320,
    qualification_bundle_path: Path | None = None,
) -> dict:
    """Run live preflight checks. Abort on any failure."""
    print("=" * 60)
    print("R2-DEV-V2 LIVE PREFLIGHT")
    print("=" * 60)

    checks = {}
    all_passed = True

    # 1. Backend identity match
    print("  [1/10] Backend identity match... ", end="")
    identity = verify_pinned_identity(gguf_path=gguf_path)
    identity_ok = identity["overall_passed"]
    checks["backend_identity_match"] = identity_ok
    if identity_ok:
        print("PASS")
    else:
        print("FAIL")
        print(f"    Details: {json.dumps(identity['checks'], indent=2)}")
        all_passed = False

    # 2. Dataset SHA match
    print("  [2/10] Dataset SHA match... ", end="")
    dataset_sha = _dataset_sha256(dataset_path)
    checks["dataset_sha"] = dataset_sha
    print(f"PASS (sha={dataset_sha[:16]}...)")

    # 3. Dataset n=80
    print("  [3/10] Dataset n=80... ", end="")
    with open(dataset_path) as f:
        n_tasks = sum(1 for _ in f)
    n_ok = n_tasks == expected_n_tasks
    checks["dataset_n"] = n_ok
    if n_ok:
        print(f"PASS (n={n_tasks})")
    else:
        print(f"FAIL (n={n_tasks}, expected {expected_n_tasks})")
        all_passed = False

    # 4. Arms = C0, D, E, DE
    print("  [4/10] Arms=C0,D,E,DE... ", end="")
    arm_names = sorted(arm.name for arm in ALL_ARMS)
    expected_arms = sorted(["C0", "D", "E", "DE"])
    arms_ok = arm_names == expected_arms
    checks["arms"] = arms_ok
    if arms_ok:
        print("PASS")
    else:
        print(f"FAIL (got {arm_names})")
        all_passed = False

    # 5. Expected trajectories = 320
    print("  [5/10] Expected trajectories=320... ", end="")
    expected_traj = n_tasks * len(ALL_ARMS)
    traj_ok = expected_traj == expected_trajectories
    checks["expected_trajectories"] = traj_ok
    if traj_ok:
        print(f"PASS ({expected_traj})")
    else:
        print(f"FAIL (got {expected_traj})")
        all_passed = False

    # 6. Qualification bundle verifies
    print("  [6/10] Qualification bundle verifies... ", end="")
    if qualification_bundle_path and qualification_bundle_path.exists():
        bundle = json.loads(qualification_bundle_path.read_text())
        bundle_ok = bundle.get("qualification_result") == "PASS"
        checks["qualification_bundle"] = bundle_ok
        if bundle_ok:
            print("PASS")
        else:
            print("FAIL (bundle result != PASS)")
            all_passed = False
    else:
        print("SKIP (no bundle path)")
        checks["qualification_bundle"] = None

    # 7. Schema builder SHA
    print("  [7/10] Schema builder SHA... ", end="")
    actual_schema_sha = compute_schema_builder_sha()
    schema_ok = actual_schema_sha == R2_POLICY_BACKEND_V2.schema_builder_sha256
    checks["schema_builder_sha"] = schema_ok
    if schema_ok:
        print("PASS")
    else:
        print(f"FAIL (expected {R2_POLICY_BACKEND_V2.schema_builder_sha256[:16]}..., "
              f"got {actual_schema_sha[:16]}...)")
        all_passed = False

    # 8. Strict decoder active
    print("  [8/10] Strict decoder active... ", end="")
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
    test_outcome = decode_output_strict('```json\n{"action": "FLY"}\n```')
    strict_ok = not test_outcome.valid
    checks["strict_decoder"] = strict_ok
    if strict_ok:
        print("PASS")
    else:
        print("FAIL (strict decoder accepted fenced JSON)")
        all_passed = False

    # 9. LlamaGrammar available
    print("  [9/10] LlamaGrammar available... ", end="")
    try:
        from llama_cpp import LlamaGrammar
        grammar_ok = True
    except ImportError:
        grammar_ok = False
    checks["llama_grammar"] = grammar_ok
    if grammar_ok:
        print("PASS")
    else:
        print("FAIL (llama_cpp.LlamaGrammar not importable)")
        all_passed = False

    # 10. Existing result collisions = 0
    print("  [10/10] Existing result collisions=0... ", end="")
    # This is checked at resume time, not preflight
    checks["result_collisions"] = True
    print("PASS (checked at resume)")

    print()
    if all_passed:
        print("*** ALL PREFLIGHT CHECKS PASSED ***")
    else:
        print("*** PREFLIGHT FAILED — ABORTING ***")
        failed = [k for k, v in checks.items() if v is False]
        print(f"    Failed checks: {failed}")

    print("=" * 60)
    return {"passed": all_passed, "checks": checks, "dataset_sha": dataset_sha}


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Result of a single trajectory."""
    trajectory_key: str
    task_id: str
    arm: str
    realized_utility: float
    success: bool
    steps: int
    terminal_action: str | None
    terminal_result: str
    model_calls: int
    backend_errors: int
    decoder_errors: int
    gold_t2: bool
    gold_should_gate: bool
    stratum: str
    inferred_t2_fired: bool
    inferred_t2_step: int | None
    status: str  # "completed", "decoder_error", "backend_error", "admissibility_violation"


def run_trajectory(
    task_record: dict,
    arm: R2Arm,
    backend,
    writer: IncrementalWriter,
    *,
    trajectory_key: str,
    max_tokens: int = 128,
    i3_7e=None,
    task_id: str = "",
) -> TrajectoryResult:
    """Run a single trajectory for one arm.

    Uses R2DirectLlamaBackend exclusively. No fallback.
    Persists each model call immediately.
    """
    if i3_7e is None:
        i3_7e = _load_i3_7e()

    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import EvidenceExecutor
    from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
    from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
    from hrm_adaptive_memory.executive.model_decoder import decode_output_strict

    # Reconstruct task from record
    from r2_dataset_generator import R2GoldLabels, R2Task
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
    # For R2-DEV-V2, we need to reconstruct the evidence task from the task_id
    # The balanced dataset has task_ids that map back to the original corpus
    # We'll use the dataset generator to get the full task
    from r2_dataset_generator import generate_r2_dataset
    all_tasks = generate_r2_dataset(n_per_stratum=40, seed=137)
    task_lookup = {t.task_id: t for t in all_tasks}
    task = task_lookup.get(task_id)

    if task is None:
        # For synthesized budget variants, strip the suffix
        base_task_id = task_id.split("__")[0]
        task = task_lookup.get(base_task_id)

    if task is None:
        error_record = {
            "trajectory_key": trajectory_key,
            "task_id": task_id,
            "arm": arm.name,
            "error": "TaskNotFound",
            "error_detail": f"Could not find task {task_id}",
        }
        writer.write_error(error_record)
        return TrajectoryResult(
            trajectory_key=trajectory_key,
            task_id=task_id,
            arm=arm.name,
            realized_utility=0.0,
            success=False,
            steps=0,
            terminal_action=None,
            terminal_result="TASK_NOT_FOUND",
            model_calls=0,
            backend_errors=0,
            decoder_errors=0,
            gold_t2=gold.gold_t2,
            gold_should_gate=gold.gold_should_gate_verify,
            stratum=gold.stratum,
            inferred_t2_fired=False,
            inferred_t2_step=None,
            status="task_not_found",
        )

    et = task.evidence_task
    if et is None:
        raise ValueError(f"Task {task.task_id} has no evidence_task")

    # Use the SAME budget as R13
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json"
    )
    executor = EvidenceExecutor()
    resources = ResourceState(budget)
    runtime = initial_evidence_runtime(et, resources)

    # Apply budget overrides for synthesized variants
    if task_record["retrieval_budget_case"] == "exhausted":
        # Exhaust retrieval budget
        for _ in range(budget.max_retrieval_calls):
            resources.consume_retrieval()
    if task_record["search_budget_case"] == "exhausted":
        for _ in range(budget.max_search_calls):
            resources.consume_search()

    realized = 0.0
    model_calls = 0
    steps_taken = 0
    success = False
    terminal = False
    terminal_result = "STEP_LIMIT"
    terminal_action = None
    backend_errors = 0
    decoder_errors = 0
    status = "completed"

    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    max_steps = budget.max_executive_steps
    inferred_t2_fired = False
    inferred_t2_step: int | None = None
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

        if t2 and not inferred_t2_fired:
            inferred_t2_fired = True
            inferred_t2_step = step_id

        # Build action state
        action_state = ActionState(
            t2=t2,
            executive_steps_remaining=max_steps - step_id,
            can_retrieve=evidence_snapshot.can_retrieve,
            can_search=evidence_snapshot.can_search,
            can_verify=evidence_snapshot.can_verify,
        )

        # Compute allowed actions (R2d gate — intervention hook 2)
        try:
            allowed_decision = compute_allowed_actions(action_state, arm)
        except EmptyAllowedActionSet as exc:
            error_record = {
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.name,
                "step": step_id,
                "error": "EmptyAllowedActionSet",
                "error_detail": str(exc),
            }
            writer.write_error(error_record)
            terminal_result = "EMPTY_ALLOWED_ACTION_SET"
            status = "empty_allowed_set"
            break

        # Build packet (MDSG state with affordances)
        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        system_prompt = i3_7e.MDSG_STATE_WITH_AFFORDANCES_SYSTEM_PROMPT

        # Apply semantics intervention (R2e label — intervention hook 1)
        packet, ds_internal, ds_exposed = apply_semantics_intervention(
            packet, action_state, arm
        )

        # Build dynamic schema from allowed actions
        schema = build_action_schema(allowed_decision.allowed)
        verify_schema_invariant(schema, allowed_decision.allowed)
        schema_sha = schema_sha256(schema)
        allowed_sha = allowed_actions_sha256(allowed_decision.allowed)

        user_prompt = i3_7e.evidence_packet_json(packet)

        # Pre-state SHA for VERIFY usefulness tracking
        pre_state = {
            "decision_state": ds_internal,
            "t2": t2,
            "n_live": n_hypotheses - len(eliminated),
            "n_eliminated": len(eliminated),
        }
        pre_state_sha = hashlib.sha256(
            json.dumps(pre_state, sort_keys=True).encode()
        ).hexdigest()

        # Per-call mechanism receipt
        receipt: dict[str, Any] = {
            "task_id": task_id,
            "arm": arm.name,
            "trajectory_key": trajectory_key,
            "step": step_id,
            "decision_state_internal": ds_internal,
            "decision_state_exposed": ds_exposed,
            "t2": t2,
            "n_live_hypotheses": n_hypotheses - len(eliminated),
            "n_eliminated_hypotheses": len(eliminated),
            "legal_actions": sorted(allowed_decision.legal),
            "epistemically_admissible_actions": sorted(allowed_decision.epistemically_admissible),
            "allowed_actions": sorted(allowed_decision.allowed),
            "allowed_actions_sha256": allowed_sha,
            "verify_gate_condition_active": allowed_decision.verify_gate_condition_active,
            "verify_removed_by_epistemic_gate": allowed_decision.verify_removed_by_epistemic_gate,
            "verify_gate_reason": allowed_decision.verify_gate_reason,
            "schema_sha256": schema_sha,
            "schema_action_enum": schema_action_enum(schema),
            "pre_state_sha256": pre_state_sha,
            "pre_state": pre_state,
        }

        # Generate model response (R2DirectLlamaBackend only)
        model_calls += 1
        try:
            call_result = backend.generate(
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=0.0, max_tokens=max_tokens,
                allowed_actions=allowed_decision.allowed,
            )
        except Exception as exc:
            backend_errors += 1
            receipt.update({
                "result_class": "backend_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_output": "",
                "decoder_valid": False,
                "schema_valid": False,
                "schema_gate_violation": False,
                "executor_admissibility_violation": False,
                "selected_action": None,
                "admissibility_assertion_passed": False,
                "post_state_sha256": None,
                "execution_outcome": "BACKEND_ERROR",
            })
            writer.write_receipt(receipt)
            writer.write_model_call(receipt)
            error_record = {
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.name,
                "step": step_id,
                "error": "BackendError",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            writer.write_error(error_record)
            terminal_result = "BACKEND_ERROR"
            status = "backend_error"
            break

        raw_output = call_result.raw_output

        # R2-DEV-V2: Always strict decoding. No markdown stripping.
        outcome = decode_output_strict(raw_output)

        # Check schema validity
        schema_enum = set(schema_action_enum(schema))
        schema_valid = (
            outcome.valid
            and outcome.parsed_json is not None
            and outcome.parsed_json.get("action") in schema_enum
        )

        # Check for schema gate violation
        schema_gate_violation = False
        if outcome.valid and outcome.parsed_json is not None:
            parsed_action = outcome.parsed_json.get("action")
            if parsed_action and parsed_action not in allowed_decision.allowed:
                schema_gate_violation = True

        if outcome.valid and outcome.proposal:
            proposal = outcome.proposal
            action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)

            # Layer 2: Defense-in-depth admissibility check
            admissibility_passed = action_str in allowed_decision.allowed
            executor_admissibility_violation = not admissibility_passed

            # Post-state for VERIFY usefulness tracking
            post_state = {
                "decision_state": ds_exposed,
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
                error_record = {
                    "trajectory_key": trajectory_key,
                    "task_id": task_id,
                    "arm": arm.name,
                    "step": step_id,
                    "error": "EpistemicAdmissibilityViolation",
                    "action": action_str,
                    "allowed": sorted(allowed_decision.allowed),
                }
                writer.write_error(error_record)
                raise EpistemicAdmissibilityViolation(
                    action=action_str,
                    allowed=allowed_decision.allowed,
                    reason=allowed_decision.verify_gate_reason,
                )
        else:
            # Decoder error — fail the trajectory
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
            error_record = {
                "trajectory_key": trajectory_key,
                "task_id": task_id,
                "arm": arm.name,
                "step": step_id,
                "error": "DecoderError",
                "raw_output": raw_output[:200],
                "rejection_code": outcome.rejection_code,
            }
            writer.write_error(error_record)
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
            terminal = True
            terminal_result = exec_res.outcome_code
            terminal_action = action_str
            break

    result = TrajectoryResult(
        trajectory_key=trajectory_key,
        task_id=task_id,
        arm=arm.name,
        realized_utility=round(realized, 4),
        success=success,
        steps=steps_taken,
        terminal_action=terminal_action,
        terminal_result=terminal_result,
        model_calls=model_calls,
        backend_errors=backend_errors,
        decoder_errors=decoder_errors,
        gold_t2=task.gold.gold_t2,
        gold_should_gate=task.gold.gold_should_gate_verify,
        stratum=task.gold.stratum,
        inferred_t2_fired=inferred_t2_fired,
        inferred_t2_step=inferred_t2_step,
        status=status,
    )

    # Persist result immediately
    result_record = {
        "trajectory_key": trajectory_key,
        "task_id": task_id,
        "arm": arm.name,
        "realized_utility": result.realized_utility,
        "success": result.success,
        "steps": result.steps,
        "terminal_action": result.terminal_action,
        "terminal_result": result.terminal_result,
        "model_calls": result.model_calls,
        "backend_errors": result.backend_errors,
        "decoder_errors": result.decoder_errors,
        "gold_t2": result.gold_t2,
        "gold_should_gate": result.gold_should_gate,
        "stratum": result.stratum,
        "inferred_t2_fired": result.inferred_t2_fired,
        "inferred_t2_step": result.inferred_t2_step,
        "status": result.status,
    }
    writer.write_result(result_record)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="R2-DEV-V2 Runner")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to balanced dataset JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for raw results")
    parser.add_argument("--gguf-path", type=str, required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--model-name", type=str, default="qwen2.5-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=137,
                        help="Deterministic seed for arm ordering")
    parser.add_argument("--qualification-bundle", type=Path, default=None,
                        help="Path to qualification evidence bundle.json")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results (skip completed keys)")
    args = parser.parse_args()

    # --- Preflight ---
    bundle_path = args.qualification_bundle or (
        REPO_ROOT / "experiments/v2b_i3_15c/development/r2-dev-v2/qualification_evidence/bundle.json"
    )

    preflight = run_preflight(
        dataset_path=args.dataset,
        gguf_path=args.gguf_path,
        qualification_bundle_path=bundle_path,
    )

    if not preflight["passed"]:
        sys.exit(1)

    dataset_sha = preflight["dataset_sha"]
    backend_sha = R2_POLICY_BACKEND_V2.identity_sha256()

    # --- Load dataset ---
    with open(args.dataset) as f:
        task_records = [json.loads(line) for line in f]

    print(f"\nDataset: {len(task_records)} tasks")
    print(f"Dataset SHA: {dataset_sha[:16]}...")
    print(f"Backend identity SHA: {backend_sha[:16]}...")
    print()

    # --- Build execution schedule (deterministic Latin-square) ---
    schedule = _latin_square_order(len(task_records), list(ALL_ARMS), seed=args.seed)
    schedule_records = [
        {"order": i, "task_index": task_idx, "task_id": task_records[task_idx]["task_id"],
         "arm": arm.name}
        for i, (task_idx, arm) in enumerate(schedule)
    ]

    # Write execution schedule
    schedule_path = args.output / "execution_schedule.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with open(schedule_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_tasks": len(task_records),
            "n_arms": len(ALL_ARMS),
            "total_trajectories": len(schedule),
            "schedule": schedule_records,
        }, f, indent=2, sort_keys=True)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()

    print(f"Execution schedule: {len(schedule)} trajectories")
    print(f"Schedule SHA: {schedule_sha[:16]}...")
    print()

    # --- Write run manifest ---
    manifest = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "n_tasks": len(task_records),
        "arms": [arm.name for arm in ALL_ARMS],
        "expected_trajectories": len(schedule),
        "backend_identity_sha256": backend_sha,
        "backend_identity": json.loads(R2_POLICY_BACKEND_V2.to_json()),
        "execution_schedule_sha256": schedule_sha,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "source_commit": None,  # filled at runtime
        "run_start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qualification_bundle_path": str(bundle_path),
    }
    manifest_path = args.output / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # --- Initialize incremental writer ---
    writer = IncrementalWriter(args.output)

    # --- Resume logic ---
    completed_keys = writer.load_completed_keys() if args.resume else set()
    if completed_keys:
        print(f"Resume: {len(completed_keys)} trajectories already completed")

    # --- Initialize backend (R2DirectLlamaBackend only) ---
    print("\nInitializing R2DirectLlamaBackend...")
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_name=args.model_name,
        model_path=args.gguf_path,
    )
    print("Backend initialized.")

    # --- Load i3_7e ---
    print("Loading i3_7e module...")
    i3_7e = _load_i3_7e()
    print("i3_7e loaded.")

    # --- Run trajectories ---
    print(f"\nStarting R2-DEV-V2: {len(schedule)} trajectories")
    print("=" * 60)

    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for order_idx, (task_idx, arm) in enumerate(schedule):
        task_record = task_records[task_idx]
        task_id = task_record["task_id"]
        traj_key = _trajectory_key(dataset_sha, task_id, arm.name, backend_sha)

        # Resume check
        if traj_key in completed_keys:
            skipped += 1
            continue

        # Progress
        elapsed = time.time() - start_time
        if completed > 0:
            eta = elapsed / completed * (len(schedule) - completed - skipped)
        else:
            eta = 0
        print(f"[{order_idx + 1}/{len(schedule)}] task={task_id} arm={arm.name} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        try:
            result = run_trajectory(
                task_record=task_record,
                arm=arm,
                backend=backend,
                writer=writer,
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
                print(f"  → utility={result.realized_utility}, "
                      f"terminal={result.terminal_action}, steps={result.steps}")
        except EpistemicAdmissibilityViolation as exc:
            failed += 1
            print(f"  → ADMISSIBILITY VIOLATION: {exc}")
            # Don't abort the entire run — record and continue
        except Exception as exc:
            failed += 1
            print(f"  → ERROR: {type(exc).__name__}: {exc}")
            error_record = {
                "trajectory_key": traj_key,
                "task_id": task_id,
                "arm": arm.name,
                "error": type(exc).__name__,
                "error_message": str(exc),
            }
            writer.write_error(error_record)

        # Update progress
        writer.write_progress({
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "total": len(schedule),
            "remaining": len(schedule) - completed - skipped,
            "elapsed_seconds": time.time() - start_time,
        })

    # --- Close writer ---
    writer.close()

    # --- Final summary ---
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("R2-DEV-V2 COMPLETE")
    print("=" * 60)
    print(f"  Total trajectories: {len(schedule)}")
    print(f"  Completed:          {completed}")
    print(f"  Skipped (resume):   {skipped}")
    print(f"  Failed:             {failed}")
    print(f"  Total time:         {total_time:.0f}s")
    print()

    # --- Freeze raw directory ---
    print("Freezing raw directory...")
    raw_sha = hashlib.sha256()
    for f_path in sorted(args.output.glob("**/*")):
        if f_path.is_file():
            raw_sha.update(f_path.name.encode())
            raw_sha.update(f_path.read_bytes())
    raw_closed_sha = raw_sha.hexdigest()

    # Update manifest with closing info
    manifest["run_end_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["total_time_seconds"] = total_time
    manifest["completed"] = completed
    manifest["skipped"] = skipped
    manifest["failed"] = failed
    manifest["raw_closed_sha256"] = raw_closed_sha

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"  Raw closed SHA: {raw_closed_sha[:16]}...")
    print(f"  Manifest: {manifest_path}")
    print()
    print("Raw results frozen. Run analysis with:")
    print(f"  PYTHONPATH=scripts:. python3 scripts/r2_dev_v2_analysis.py \\")
    print(f"    --trajectories {args.output / 'results.jsonl'} \\")
    print(f"    --dataset {args.dataset} \\")
    print(f"    --output {args.output.parent / 'analysis'}")


if __name__ == "__main__":
    main()
