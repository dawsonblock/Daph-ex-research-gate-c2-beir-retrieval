#!/usr/bin/env python3
"""I3.30: Frozen Targeted Live Study (D1-D5, V1 vs V3).

FROZEN MANIFEST: This run uses frozen artifacts. No changes to model,
representation, threshold, positive certificates, prompts, or runner
once trajectory generation begins.

Five strata:
  D1: DEFER — resource-exhausted safe defer
  D2: DEFER — post-verification elimination
  D3: CONTINUE — unresolved/competing evidence
  D4: ANSWER — post-verification resolved answer
  D5: CONTINUE — ambiguous post-verification state (competing verified support)

Two arms:
  V1 = confirmed champion (ANSWER-only hard authority)
  V3 = experimental candidate (positive structural certificate authority)

Pre-registered gates (G1-G12):
  G1  success_V3 >= success_V1
  G2  rescues > breaks
  G3  FAR_DEFER_D3 = 0
  G4  FAR_ANSWER = 0
  G5  false terminal authority D5 = 0
  G6  hard-authority breaks = 0
  G7  DEFER coverage D1/D2 materially > 0
  G8  ANSWER coverage D4 materially > 0
  G9  mean paired ΔU > 0
  G10 premature ANSWER/DEFER <= V1
  G11 reliability errors = 0
  G12 positive structural certificate required for every hard authority event

Usage:
  PYTHONPATH=. python3 scripts/run_i3_30_live_study_frozen.py \
    --gguf-path /Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --output-dir experiments/i3_30/live_study \
    --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState
from hrm_adaptive_memory.executive.evidence_benchmark import (
    initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
    EvidenceExecutor, valid_verify_targets,
)
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.model_decoder import decode_output_strict
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from daph.intervention.checkpoint import compute_state_features
from daph.authority.policy import (
    AuthorityMode, StructuralState, decide_authority, build_receipt,
    AUTHORITY_THRESHOLD,
)
from daph.authority.policy_v3 import (
    StructuralStateV3, decide_authority_v3,
    answer_structural_certificate, defer_structural_certificate,
)

from run_i3_29_live_safety import (
    QModelV1, classify_phase_simple, compute_near_optimal_set,
    compute_progress_scores, apply_progress_tiebreak,
    compute_q_gap, get_structural_state, _make_result,
)
from run_i3_28_rep_repair import (
    extract_v1_features, compute_structural_features,
)
from run_i3_30r_train_v3r2 import extract_v3r2_features, get_v3r2_feature_keys
from daph.epistemic.v3_features import compute_v3_features_canonical


# ============================================================
# Q_V3R model
# ============================================================

class QModelV3R:
    """Q_CAUSAL_V3R2 — V3R2 model with canonical topology features (53 features)."""
    def __init__(self, model, feature_keys):
        self.model = model
        self.feature_keys = feature_keys

    @classmethod
    def load(cls, model_path: Path, schema_path: Path):
        model = pickle.loads(model_path.read_bytes())
        with open(schema_path) as f:
            schema = json.load(f)
        return cls(model, schema["featurekeys"])

    def predict_q(self, sf: dict, legal_actions: list[str],
                  v3_struct: dict) -> dict[str, float]:
        X = np.array([[extract_v3r2_features(sf, a, v3_struct)[k]
                      for k in self.feature_keys]
                      for a in legal_actions])
        preds = self.model.predict(X)
        return dict(zip(legal_actions, [float(p) for p in preds]))


# ============================================================
# V3 structural state
# ============================================================

def get_structural_state_v3(runtime) -> StructuralStateV3:
    """Build StructuralStateV3 from runtime."""
    visible_ev = []
    for ev in runtime.visible_evidence:
        visible_ev.append({
            "evidence_id": ev.evidence_id,
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "verification_state": ev.verification_state.name,
            "retrieved": ev.retrieved,
        })
    v2_struct = compute_structural_features(visible_ev)

    hyps_list = [{"hypothesis_id": h.hypothesis_id, "answer_action": h.answer_action.value}
                 for h in runtime.task.hypotheses]
    v3 = compute_v3_features_canonical(visible_ev, hyps_list)

    can_verify = bool(valid_verify_targets(runtime))
    verify_budget_exhausted = (
        runtime.resources.verification_calls_used >= runtime.resources.budget.max_verification_calls
    )
    all_evidence_verified = all(
        ev.verification_state != VerificationState.UNVERIFIED
        for ev in runtime.visible_evidence
        if ev.retrieved
    )

    return StructuralStateV3(
        has_competing_unverified_support=bool(v2_struct["has_competing_unverified_support"]),
        n_hyp_unverified_support=v2_struct["n_hyp_unverified_support"],
        n_hyp_unverified_contradiction=v2_struct["n_hyp_unverified_contradiction"],
        can_verify=can_verify,
        verify_budget_exhausted=verify_budget_exhausted,
        all_evidence_verified=all_evidence_verified,
        n_hyp_with_verified_support=v3["n_hyp_with_verified_support"],
        n_hyp_with_verified_contradiction=v3["n_hyp_with_verified_contradiction"],
        n_hyp_with_mixed_verified=v3["n_hyp_with_mixed_verified"],
        n_viable_hypotheses=v3["n_viable_hypotheses"],
        n_eliminated_hypotheses=v3["n_eliminated_hypotheses"],
        has_unique_verified_supported_hypothesis=bool(v3["has_unique_verified_supported_hypothesis"]),
        has_verified_unresolved_competition=bool(v3["has_verified_unresolved_competition"]),
        verified_hyp_action_is_answer=bool(v3["verified_hyp_action_is_answer"]),
        verified_hyp_action_is_defer=bool(v3["verified_hyp_action_is_defer"]),
    )


# ============================================================
# Certificate type classification
# ============================================================

def classify_certificate(structural_v3: StructuralStateV3, action: str) -> dict:
    """Classify the positive structural certificate type and components."""
    if action == "ANSWER":
        if structural_v3.has_unique_verified_supported_hypothesis and structural_v3.verified_hyp_action_is_answer:
            return {
                "certificate_type": "unique_verified_support_answer",
                "components": {
                    "has_unique_verified_supported_hypothesis": True,
                    "verified_hyp_action_is_answer": True,
                },
            }
        if structural_v3.all_evidence_verified and structural_v3.n_hyp_with_verified_contradiction == 0:
            return {
                "certificate_type": "all_verified_no_contradiction",
                "components": {
                    "all_evidence_verified": True,
                    "n_hyp_with_verified_contradiction": 0,
                },
            }
    elif action == "DEFER":
        if structural_v3.has_unique_verified_supported_hypothesis and structural_v3.verified_hyp_action_is_defer:
            return {
                "certificate_type": "unique_verified_support_defer",
                "components": {
                    "has_unique_verified_supported_hypothesis": True,
                    "verified_hyp_action_is_defer": True,
                },
            }
        if structural_v3.n_eliminated_hypotheses > 0 and structural_v3.n_viable_hypotheses <= 1:
            return {
                "certificate_type": "elimination",
                "components": {
                    "n_eliminated_hypotheses": structural_v3.n_eliminated_hypotheses,
                    "n_viable_hypotheses": structural_v3.n_viable_hypotheses,
                },
            }
        if (structural_v3.verify_budget_exhausted
            and structural_v3.n_hyp_with_verified_support == 0
            and structural_v3.n_hyp_with_verified_contradiction == 0):
            return {
                "certificate_type": "resource_exhaustion_no_verified",
                "components": {
                    "verify_budget_exhausted": True,
                    "n_hyp_with_verified_support": 0,
                    "n_hyp_with_verified_contradiction": 0,
                },
            }
    return {
        "certificate_type": "NONE",
        "components": {},
    }


# ============================================================
# Trajectory runner
# ============================================================

def run_trajectory(task, backend, i3_7e, utility,
                   q_v1: QModelV1, q_v3r: QModelV3R, arm: str,
                   d2_pre_verify: bool = False,
                   d5_pre_verify: bool = False):
    """Run a single trajectory.

    arm: "V1" or "V3"
    d2_pre_verify: if True, pre-execute VERIFY for D2 stratum
    d5_pre_verify: if True, pre-execute VERIFY for D5 stratum (creates competing verified support)
    """
    from r2_schema import build_action_schema, schema_sha256
    from r2_allowed_actions import compute_allowed_actions, ActionState, EmptyAllowedActionSet, C0
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import get_budget_for_task

    budget = get_budget_for_task(task)
    max_steps = budget.max_executive_steps
    resources = ResourceState(budget=budget)
    runtime = initial_evidence_runtime(task, resources)
    executor = EvidenceExecutor()

    # D2: pre-verify first evidence
    if d2_pre_verify:
        valid = valid_verify_targets(runtime)
        if valid:
            res = executor.execute(runtime, DecisionAction.VERIFY,
                                   target_evidence_id=valid[0])
            runtime = res.runtime
            if res.terminal:
                return _make_result(task, arm, 0.0, False, "PRE_VERIFY_TERMINAL",
                                    [], [], [], False, False)

    # D5: evidence is already pre-verified in the task definition
    # (competing verified support is baked into the evidence items)

    prior_actions = []
    prior_outcomes = []
    realized = 0.0
    success = False
    terminal = False
    terminal_action = None
    terminal_result = "STEP_LIMIT"
    premature_defer = False
    premature_answer = False
    resource_exhaustion = False
    actions_taken = []
    authority_log = []
    receipts = []
    hard_authority_events = []  # mechanism analysis records
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

        legal_actions = sorted(allowed_decision.allowed)
        sf = compute_state_features(runtime, tuple(prior_actions))
        phase = classify_phase_simple(sf)

        if sf.get("steps_remaining", 0) <= 0:
            resource_exhaustion = True

        # === Q prediction ===
        if arm == "V1":
            q_values = q_v1.predict_q(sf, legal_actions)
        else:
            structural_v3 = get_structural_state_v3(runtime)
            v3_struct_dict = structural_v3.as_dict()
            v3_struct_dict["n_hyp_unverified_support"] = structural_v3.n_hyp_unverified_support
            v3_struct_dict["n_hyp_unverified_contradiction"] = structural_v3.n_hyp_unverified_contradiction
            v3_struct_dict["has_competing_unverified_support"] = int(structural_v3.has_competing_unverified_support)
            q_values = q_v3r.predict_q(sf, legal_actions, v3_struct_dict)

        near_optimal, lower_value = compute_near_optimal_set(q_values, 3.0)
        progress_scores = compute_progress_scores(runtime, legal_actions, utility, executor)
        refined_set, confidence = apply_progress_tiebreak(near_optimal, progress_scores, 0.05)
        q_gap = compute_q_gap(q_values)

        # === Authority decision ===
        schema_actions = allowed_decision.allowed
        authority_mode_str = "A0_advisory"
        forced_action = None
        certificate_info = None

        if arm == "V1":
            AUTHORITATIVE = frozenset({"ANSWER"})
            if (confidence == "clear" and len(refined_set) == 1
                    and q_gap >= AUTHORITY_THRESHOLD
                    and refined_set[0] in AUTHORITATIVE):
                schema_actions = frozenset({refined_set[0]}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2A_hard_select"
                    forced_action = refined_set[0]
                else:
                    schema_actions = allowed_decision.allowed

        elif arm == "V3":
            structural_v3 = get_structural_state_v3(runtime)
            decision = decide_authority_v3(
                q_values=q_values,
                legal_actions=legal_actions,
                structural=structural_v3,
            )

            if decision.mode == AuthorityMode.HARD_ANSWER:
                schema_actions = frozenset({"ANSWER"}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2AD_hard_ANSWER"
                    forced_action = "ANSWER"
                    certificate_info = classify_certificate(structural_v3, "ANSWER")
                else:
                    schema_actions = allowed_decision.allowed
            elif decision.mode == AuthorityMode.HARD_DEFER:
                schema_actions = frozenset({"DEFER"}) & allowed_decision.allowed
                if schema_actions:
                    authority_mode_str = "A2AD_hard_DEFER"
                    forced_action = "DEFER"
                    certificate_info = classify_certificate(structural_v3, "DEFER")
                else:
                    schema_actions = allowed_decision.allowed
            else:
                schema_actions = allowed_decision.allowed

            receipt = build_receipt(
                state_features=sf,
                legal_actions=legal_actions,
                q_values=q_values,
                structural=structural_v3.to_v2(),
                decision=decision,
            )
            receipt["step"] = step_id
            receipt["arm"] = arm
            receipt["authority_mode"] = authority_mode_str
            receipt["v3_structural"] = structural_v3.as_dict()
            if certificate_info:
                receipt["certificate"] = certificate_info
            receipts.append(receipt)

            # Record hard authority event for mechanism analysis
            if forced_action and authority_mode_str.startswith("A2AD_hard"):
                q_sorted = sorted(q_values.items(), key=lambda x: -x[1])
                state_sha = hashlib.sha256(
                    json.dumps(sf, sort_keys=True).encode()
                ).hexdigest()[:16]
                hard_authority_events.append({
                    "task_id": task.task_id,
                    "stratum": task.category,
                    "step": step_id,
                    "state_sha": state_sha,
                    "q_argmax": q_sorted[0][0] if q_sorted else "",
                    "q_second_best": q_sorted[1][0] if len(q_sorted) > 1 else "",
                    "q_gap": round(q_gap, 4),
                    "certificate_type": certificate_info["certificate_type"] if certificate_info else "NONE",
                    "certificate_components": certificate_info["components"] if certificate_info else {},
                    "forced_action": forced_action,
                    "v1_action": None,  # filled in analysis
                    "causal_best_action": None,  # filled in analysis if replayable
                    "terminal_outcome": None,  # filled after trajectory
                    "utility_delta": None,  # filled in analysis
                    "rescue_break_neutral": None,  # filled in analysis
                })

        authority_log.append({
            "step": step_id,
            "authority_mode": authority_mode_str,
            "legal_actions": legal_actions,
            "schema_actions": sorted(schema_actions),
            "refined_set": refined_set,
            "confidence": confidence,
            "q_gap": round(q_gap, 2),
            "forced_action": forced_action,
        })

        # === Build packet ===
        extra_fields = {
            "near_optimal_actions": refined_set,
            "lower_value_actions": lower_value,
            "guidance_confidence": confidence,
            "epistemic_phase": phase,
        }

        packet = i3_7e.build_mdsg_state_with_affordances_packet(evidence_snapshot)
        schema = build_action_schema(schema_actions)
        schema_sha = schema_sha256(schema)

        packet_dict = json.loads(i3_7e.evidence_packet_json(packet))
        packet_dict["executive_guidance"] = extra_fields
        user_prompt = json.dumps(packet_dict, indent=2)

        # === Call Qwen ===
        try:
            call_result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                allowed_actions=schema_actions,
            )
        except Exception:
            terminal_result = "BACKEND_ERROR"
            break

        raw_output = call_result.raw_output
        outcome = decode_output_strict(raw_output)

        if not outcome.valid or not outcome.proposal:
            terminal_result = "DECODER_ERROR"
            break

        proposal = outcome.proposal
        action_str = proposal.action.value if hasattr(proposal.action, "value") else str(proposal.action)
        target_id = getattr(proposal, "target_id", None)

        if action_str not in schema_actions:
            terminal_result = "ADMISSIBILITY_VIOLATION"
            break

        action = DecisionAction(action_str)
        resources_before = runtime.resources
        exec_res = executor.execute(runtime, action, target_evidence_id=target_id)
        runtime = exec_res.runtime
        resources_after = runtime.resources

        step_cost = utility.action_cost(resources_before, resources_after)
        realized -= step_cost

        prior_actions.append(action_str)
        prior_outcomes.append("EXECUTED")
        actions_taken.append(action_str)

        if exec_res.terminal:
            terminal = True
            terminal_action = action_str
            success = bool(exec_res.task_success)
            tr = utility.terminal_reward(action, success)
            realized += tr
            if success:
                terminal_result = "SUCCESS"
            else:
                terminal_result = "TERMINAL_WRONG"
                if action_str == "DEFER":
                    premature_defer = (sf.get("n_verified", 0) == 0)
                elif action_str == "ANSWER":
                    premature_answer = (sf.get("n_verified", 0) == 0)
            break

    if not terminal:
        realized -= 0.5

    # Update hard authority events with terminal outcome
    for evt in hard_authority_events:
        evt["terminal_outcome"] = terminal_result
        evt["trajectory_success"] = success

    result = _make_result(
        task, arm, realized, success, terminal_result,
        actions_taken, authority_log, receipts,
        premature_defer, premature_answer,
    )
    result["hard_authority_events"] = hard_authority_events
    return result


# ============================================================
# Manifest freezing
# ============================================================

def compute_manifest() -> dict:
    """Compute the frozen manifest with all SHAs."""
    def sha256_file(path):
        p = REPO_ROOT / path
        if not p.exists():
            return "MISSING"
        return hashlib.sha256(p.read_bytes()).hexdigest()

    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=REPO_ROOT).decode().strip()

    manifest = {
        "source_commit": commit,
        "source_commit_short": commit[:12],
        "branch": subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT).decode().strip(),

        # Q models
        "Q_V1_model_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl"),
        "Q_V1_schema_sha256": sha256_file("experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json"),
        "Q_V3R_model_sha256": sha256_file("experiments/i3_30r/Q_V3R2_A.pkl"),
        "Q_V3R_schema_sha256": sha256_file("experiments/i3_30r/v3r2_feature_schema.json"),

        # Schema
        "Q_STATE_SCHEMA_V3_sha256": sha256_file("experiments/i3_30r/v3r2_feature_schema.json"),

        # Authority policies
        "authority_policy_v2_sha256": sha256_file("daph/authority/policy.py"),
        "authority_policy_v3_sha256": sha256_file("daph/authority/policy_v3.py"),
        "authority_init_sha256": sha256_file("daph/authority/__init__.py"),

        # V3 tests
        "authority_v3_tests_sha256": sha256_file("tests/unit/test_authority_v3.py"),

        # Utility
        "utility_config_sha256": sha256_file("configs/v2b_i3_1_utility_v1.json"),

        # Runner
        "runner_sha256": sha256_file("scripts/run_i3_30_live_study_frozen.py"),
        "i3_29_runner_sha256": sha256_file("scripts/run_i3_29_live_safety.py"),

        # I3.30B boundary data
        "i3_30b_boundary_data_sha256": sha256_file("experiments/i3_30b/post_verify_causal_actions_v1.jsonl"),

        # Offline gates
        "offline_gates_sha256": sha256_file("experiments/i3_30r/structural_holdout_gates.json"),

        # Frozen constants
        "authority_threshold": 5.0,
        "near_optimal_epsilon": 3.0,

        # Learner (frozen)
        "learner": "GradientBoostingRegressor",
        "n_estimators": 200,
        "max_depth": 4,
        "random_state": 42,

        # Qwen
        "qwen_gguf_sha256": sha256_file("/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        "qwen_gguf_path": "/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",

        # Benchmark
        "i3_29_generator_sha256": sha256_file("hrm_adaptive_memory/executive/evidence_benchmark/i3_29_safety_generator.py"),
        "i3_30_d5_generator_sha256": sha256_file("hrm_adaptive_memory/executive/evidence_benchmark/i3_30_d5_generator.py"),
        "benchmark_seed": 9817,

        # Pre-registered gates
        "gates": {
            "G1": "success_V3 >= success_V1",
            "G2": "rescues > breaks",
            "G3": "FAR_DEFER_D3 = 0",
            "G4": "FAR_ANSWER = 0",
            "G5": "false_terminal_authority_D5 = 0",
            "G6": "hard_authority_breaks = 0",
            "G7": "DEFER_coverage_D1_D2 materially > 0",
            "G8": "ANSWER_coverage_D4 materially > 0",
            "G9": "mean_paired_delta_U > 0",
            "G10": "premature_ANSWER_DEFER <= V1",
            "G11": "reliability_errors = 0",
            "G12": "positive_structural_certificate_required_for_every_hard_authority_event",
        },

        # Promotion target
        "promotion_target": "TerminalAuthorityPrecision = 1.0",
        "promotion_name_if_passed": "DAPH_ADAPTIVE_AUTHORITY_EXECUTIVE_V3_CONFIRMATION_CANDIDATE",
    }

    # Compute benchmark hash
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        generate_i3_29_benchmark, compute_benchmark_hash,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import (
        generate_d5_tasks,
    )
    tasks = generate_i3_29_benchmark(seed=9817)
    d5_tasks = generate_d5_tasks(seed=9817, n_tasks=35)
    all_tasks = tasks + d5_tasks

    bench_hash = compute_benchmark_hash(tasks)
    manifest["benchmark_sha256"] = bench_hash
    manifest["d5_benchmark_sha256"] = hashlib.sha256(
        json.dumps([t.task_id for t in d5_tasks], sort_keys=True).encode()
    ).hexdigest()

    from collections import Counter
    strata_counts = Counter(t.category for t in all_tasks)
    manifest["stratum_counts"] = dict(strata_counts)
    manifest["task_count"] = len(all_tasks)
    manifest["trajectory_count"] = len(all_tasks) * 2  # V1 + V3

    return manifest


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I3.30 Frozen Live Study")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--output-dir", default="experiments/i3_30/live_study")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-manifest-only", action="store_true",
                        help="Only compute and save the manifest, don't run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute and freeze manifest
    manifest = compute_manifest()
    manifest_path = output_dir / "frozen_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest frozen at {manifest_path}")
    print(f"  source_commit: {manifest['source_commit_short']}")
    print(f"  Q_V3R model SHA: {manifest['Q_V3R_model_sha256'][:16]}...")
    print(f"  authority_policy_v3 SHA: {manifest['authority_policy_v3_sha256'][:16]}...")
    print(f"  task_count: {manifest['task_count']}")
    print(f"  trajectory_count: {manifest['trajectory_count']}")
    print(f"  gates: {len(manifest['gates'])}")

    if args.freeze_manifest_only:
        print("\nManifest frozen. Run without --freeze-manifest-only to execute.")
        return

    # Load models
    q_v1 = QModelV1.load(
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/QCAUSAL_gbt.pkl",
        REPO_ROOT / "experiments/i3_5/pinned_policy/frozen_estimators/feature_schema.json",
    )
    q_v3r = QModelV3R.load(
        REPO_ROOT / "experiments/i3_30r/Q_V3R2_A.pkl",
        REPO_ROOT / "experiments/i3_30r/v3r2_feature_schema.json",
    )

    # Generate benchmark
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_29_safety_generator import (
        generate_i3_29_benchmark,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.i3_30_d5_generator import (
        generate_d5_tasks,
    )
    tasks = generate_i3_29_benchmark(seed=9817)
    d5_tasks = generate_d5_tasks(seed=9817, n_tasks=35)
    all_tasks = tasks + d5_tasks

    print(f"\nTasks: {len(all_tasks)} (D1-D4: {len(tasks)}, D5: {len(d5_tasks)})")

    # Load backend
    from hrm_adaptive_memory.executive.model_backend import R2DirectLlamaBackend
    backend = R2DirectLlamaBackend(
        model_path=args.gguf_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Full GPU offload (Metal/CUDA)
    )

    # Load I3.7e for snapshot building
    import run_i3_7e_compact_governor as i3_7e

    # Load utility
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    # Trajectory output paths
    traj_path_v1 = output_dir / "trajectories_v1.jsonl"
    traj_path_v3 = output_dir / "trajectories_v3.jsonl"
    error_path = output_dir / "errors.jsonl"
    hard_auth_path = output_dir / "hard_authority_events.jsonl"

    # Resume logic
    done_v1 = set()
    done_v3 = set()
    if args.resume:
        for path, done_set in [(traj_path_v1, done_v1), (traj_path_v3, done_v3)]:
            if path.exists():
                with open(path) as f:
                    for line in f:
                        r = json.loads(line)
                        done_set.add(r["task_id"])
        print(f"  Resuming: V1 done={len(done_v1)}, V3 done={len(done_v3)}")

    # Run trajectories
    traj_file_v1 = open(traj_path_v1, "a")
    traj_file_v3 = open(traj_path_v3, "a")
    error_file = open(error_path, "a")
    hard_auth_file = open(hard_auth_path, "a")

    total = len(all_tasks) * 2
    completed = 0

    for task in all_tasks:
        d2_pre = "_d2_" in task.task_id
        d5_pre = "_d5_" in task.task_id

        for arm, q_model, traj_file, done_set in [
            ("V1", q_v1, traj_file_v1, done_v1),
            ("V3", q_v3r, traj_file_v3, done_v3),
        ]:
            if task.task_id in done_set:
                completed += 1
                continue

            try:
                if arm == "V1":
                    result = run_trajectory(task, backend, i3_7e, utility,
                                           q_v1, q_v3r, arm,
                                           d2_pre_verify=d2_pre)
                else:
                    result = run_trajectory(task, backend, i3_7e, utility,
                                           q_v1, q_v3r, arm,
                                           d2_pre_verify=d2_pre)

                traj_file.write(json.dumps(result, default=str) + "\n")
                traj_file.flush()

                # Write hard authority events
                for evt in result.get("hard_authority_events", []):
                    evt["arm"] = arm
                    hard_auth_file.write(json.dumps(evt, default=str) + "\n")
                hard_auth_file.flush()

                done_set.add(task.task_id)
                completed += 1

                if completed % 10 == 0:
                    print(f"  [{completed}/{total}] {task.task_id} {arm}: "
                          f"success={result['success']} util={result['utility']:.1f}")

            except Exception as e:
                error_file.write(json.dumps({
                    "task_id": task.task_id,
                    "arm": arm,
                    "error": str(e),
                    "timestamp": time.time(),
                }) + "\n")
                error_file.flush()
                completed += 1

    traj_file_v1.close()
    traj_file_v3.close()
    error_file.close()
    hard_auth_file.close()

    print(f"\nDone. {completed}/{total} trajectories completed.")
    print(f"  V1 trajectories: {traj_path_v1}")
    print(f"  V3 trajectories: {traj_path_v3}")
    print(f"  Hard authority events: {hard_auth_path}")
    print(f"  Errors: {error_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
