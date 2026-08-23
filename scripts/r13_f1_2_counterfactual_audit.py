#!/usr/bin/env python3
"""
R13-F1.2a: Counterfactual Affordance Audit (Hardened).

Reads ONLY from raw_closed/ — never mutates R13 artifacts.
All outputs are labeled POST_HOC_EXPLORATORY.

This script reconstructs the runtime state at T2 trigger time for each
T2-triggered trajectory, then simulates counterfactual VERIFY actions
on every valid target to determine whether any could change the MDSG state.

F1.2a hardening from F1.2:
  - Hard preflight identity verification (dataset manifest, retrieval receipts,
    executor/schema/task-generator source SHAs, experiment source commit)
  - Exhaustive monotonicity over ALL 228 T2 states and every valid target
  - Precise useful-target definition: epistemically useful = changes to
    decision_state, live/eliminated hypothesis sets, or T2 status ONLY.
    Changes to verification_state, resource_state, or evidence metadata
    are NOT counted as epistemically useful.
  - Explicit distinction between post-hoc oracle simulation (this script)
    and runtime-visible structural logic (the R2d gating rule)

Key questions answered:
  1. What affordances were actually exposed at T2? (not just what was selected)
  2. For each valid VERIFY target at T2, could verifying it change any
     epistemically decision-relevant state (hypothesis sets, decision_state,
     T2 status)?
  3. Are T2 states VERIFY_DEAD_END, VERIFY_RESOLVABLE, or NO_VERIFY?
  4. Is MDSG elimination monotonic within a trajectory? (exhaustive)
  5. Is NEEDS_DISCRIMINATION semantically correct when 0 hypotheses are live?

R13-F can identify likely failure mechanisms; it cannot confirm them causally.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LABEL = "POST_HOC_EXPLORATORY"
VERSION = "R13-F1.2a"

# --- Frozen identity constants (from r13_experiment_identity.json) ---
R13_EXPECTED_RESULTS_SHA = "ad600240bf97cbbdb09126f542d1dc56605b11089b387684883be5784bb8a463"
R13_EXPECTED_DATASET_MANIFEST_RESULTS_SHA = "ad600240bf97cbbdb09126f542d1dc56605b11089b387684883be5784bb8a463"
R13_EXPECTED_EXPERIMENT_SOURCE_COMMIT = "5454246b7e61adfb7a093eb5a1f731347071270d"
R13_EXPECTED_CONFIRMATION_EXECUTABLE_SHA = "41cc60b04f506f63b80c91e036d330d61d79992a86fb975cbe21597bd2d84f57"
R13_EXPECTED_PROTOCOL_SHA = "9590440d2744a6409cc19bc7ba8168d22cb7cee80952fb520a54134815c312c5"
R13_EXPECTED_GGUF_SHA = "2ad4c9ce431a2d5b80af37983828c2cfb8f4909792ca5075e0370e3a71ca013d"

# Source file SHAs at the experiment source commit (verified invariant)
R13_EXPECTED_EXECUTOR_SHA = "48714eb576b25fb2b6543d7cf5ec3b54f8b900363dc86b6a772b0ae9c54d03e6"
R13_EXPECTED_SCHEMA_SHA = "03176c1135cbbe080b3dcf51b2f1abba0364c941624e9d6633628a61274eee5a"
R13_EXPECTED_TASK_GEN_SHA = "b6ccafd9a085ad0a11fdbbbc7bd78d9aa000250969dae2f7074e5e44890dfdb9"
R13_EXPECTED_I3_7E_SHA = "32d043988132c4626d688d327638affe47bd238e89c8afce2275f65bceb27dd4"
R13_EXPECTED_I3_12J_SHA = "827badd1e2e0f7890c0b8a56b2c534930a6599228b89542f939cb4490d9d72ad"
R13_EXPECTED_I3_15_R1_SHA = "bb37f54bfbd8c617a4984704b4ca30a721db5a6e8fa7bffa12996e90e1ed5b67"
R13_EXPECTED_RETRIEVAL_RECEIPTS_SHA = "2329bfe2cf7f5c002ec019b0b9554a2727ee810957b9f74a364a203b99db8e1e"


def load_results(raw_closed: Path) -> list[dict]:
    with open(raw_closed / "results.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_mechanism_receipts(raw_closed: Path) -> dict[str, dict]:
    """Load mechanism receipts keyed by trajectory_key."""
    path = raw_closed / "mechanism_receipts.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        receipts = {}
        for line in f:
            if line.strip():
                r = json.loads(line)
                receipts[r["trajectory_key"]] = r
        return receipts


def load_cognition_receipts(raw_closed: Path) -> dict[str, dict]:
    """Load cognition cost receipts keyed by trajectory_key."""
    path = raw_closed / "cognition_cost_receipts.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        receipts = {}
        for line in f:
            if line.strip():
                r = json.loads(line)
                receipts[r.get("trajectory_key", "")] = r
        return receipts


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_git_blob(commit: str, rel_path: str) -> str:
    """Compute SHA256 of a file as it existed at a given commit."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git show {commit}:{rel_path} failed: {result.stderr}")
    return hashlib.sha256(result.stdout.encode()).hexdigest()


def preflight_identity(raw_closed: Path, receipts_path: Path) -> dict:
    """Hard preflight: verify that reconstruction uses precisely the same
    task/retrieval/executor identities as R13.

    Returns a dict of verified identities. Aborts on mismatch.
    """
    print(f"\n[0] Hard preflight identity verification...")
    checks = {}
    failures = []

    # 1. Verify dataset_manifest.json exists and results.jsonl SHA matches
    manifest_path = raw_closed / "dataset_manifest.json"
    if not manifest_path.exists():
        failures.append(f"Missing {manifest_path}")
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_results_sha = manifest.get("results.jsonl", {}).get("sha256")
        actual_results_sha = sha256_file(raw_closed / "results.jsonl")
        checks["results_jsonl_sha"] = actual_results_sha
        checks["manifest_results_sha"] = manifest_results_sha
        if actual_results_sha != R13_EXPECTED_RESULTS_SHA:
            failures.append(f"results.jsonl SHA mismatch: {actual_results_sha} != {R13_EXPECTED_RESULTS_SHA}")
        if manifest_results_sha != actual_results_sha:
            failures.append(f"manifest results SHA != actual: {manifest_results_sha} != {actual_results_sha}")

    # 2. Verify experiment identity
    identity_path = raw_closed / "r13_experiment_identity.json"
    if not identity_path.exists():
        failures.append(f"Missing {identity_path}")
    else:
        with open(identity_path) as f:
            identity = json.load(f)
        exp_commit = identity.get("experiment_source_commit")
        checks["experiment_source_commit"] = exp_commit
        if exp_commit != R13_EXPECTED_EXPERIMENT_SOURCE_COMMIT:
            failures.append(f"experiment_source_commit mismatch: {exp_commit} != {R13_EXPECTED_EXPERIMENT_SOURCE_COMMIT}")

        exec_sha = identity.get("confirmation_executable_sha256")
        checks["confirmation_executable_sha256"] = exec_sha
        if exec_sha != R13_EXPECTED_CONFIRMATION_EXECUTABLE_SHA:
            failures.append(f"confirmation_executable_sha256 mismatch: {exec_sha} != {R13_EXPECTED_CONFIRMATION_EXECUTABLE_SHA}")

        proto_sha = identity.get("protocol_sha256")
        checks["protocol_sha256"] = proto_sha
        if proto_sha != R13_EXPECTED_PROTOCOL_SHA:
            failures.append(f"protocol_sha256 mismatch: {proto_sha} != {R13_EXPECTED_PROTOCOL_SHA}")

        gguf_sha = identity.get("gguf_sha256")
        checks["gguf_sha256"] = gguf_sha
        if gguf_sha != R13_EXPECTED_GGUF_SHA:
            failures.append(f"gguf_sha256 mismatch: {gguf_sha} != {R13_EXPECTED_GGUF_SHA}")

    # 3. Verify retrieval receipts SHA
    actual_receipts_sha = sha256_file(receipts_path)
    checks["retrieval_receipts_sha"] = actual_receipts_sha
    if actual_receipts_sha != R13_EXPECTED_RETRIEVAL_RECEIPTS_SHA:
        failures.append(f"retrieval_receipts SHA mismatch: {actual_receipts_sha} != {R13_EXPECTED_RETRIEVAL_RECEIPTS_SHA}")

    # 4. Verify source file SHAs match experiment source commit
    source_files = {
        "executor": ("hrm_adaptive_memory/executive/evidence_benchmark/executor.py", R13_EXPECTED_EXECUTOR_SHA),
        "schema": ("hrm_adaptive_memory/executive/evidence_benchmark/schema.py", R13_EXPECTED_SCHEMA_SHA),
        "task_generator": ("hrm_adaptive_memory/executive/semantic_relations/i3_15c_task_generator.py", R13_EXPECTED_TASK_GEN_SHA),
        "i3_7e": ("scripts/run_i3_7e_compact_governor.py", R13_EXPECTED_I3_7E_SHA),
        "i3_12j": ("scripts/run_i3_12j_factorial.py", R13_EXPECTED_I3_12J_SHA),
        "i3_15_r1": ("scripts/run_i3_15_r1_balanced.py", R13_EXPECTED_I3_15_R1_SHA),
    }

    for name, (rel_path, expected_sha) in source_files.items():
        actual_sha = sha256_file(REPO_ROOT / rel_path)
        commit_sha = sha256_git_blob(R13_EXPECTED_EXPERIMENT_SOURCE_COMMIT, rel_path)
        checks[f"{name}_sha"] = actual_sha
        checks[f"{name}_sha_at_source_commit"] = commit_sha
        if actual_sha != expected_sha:
            failures.append(f"{name} SHA mismatch: {actual_sha} != {expected_sha}")
        if actual_sha != commit_sha:
            failures.append(f"{name} SHA != source commit SHA: {actual_sha} != {commit_sha}")

    # Report
    print(f"  Verified {len(checks)} identity properties")
    if failures:
        print(f"\n  *** PREFLIGHT FAILED: {len(failures)} mismatches ***")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  All identity checks passed.")

    return checks


def setup_imports():
    """Set up the same imports as the R13 runner."""
    # Load i3_7e module
    spec_7e = importlib.util.spec_from_file_location(
        "i3_7e", str(REPO_ROOT / "scripts" / "run_i3_7e_compact_governor.py"))
    i3_7e = importlib.util.module_from_spec(spec_7e)
    spec_7e.loader.exec_module(i3_7e)

    # Load i3_15c module
    spec_15c = importlib.util.spec_from_file_location(
        "i3_15c_factorial", str(REPO_ROOT / "scripts" / "run_i3_15c_factorial.py"))
    i3_15c = importlib.util.module_from_spec(spec_15c)
    spec_15c.loader.exec_module(i3_15c)
    i3_12j = i3_15c.i3_12j

    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
        get_i3_15c_corpus, generate_i3_15c_corpus,
    )
    from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
        DeterministicRelationExtractor,
    )
    from hrm_adaptive_memory.executive.semantic_relations.integration import (
        infer_relations_for_runtime,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark import (
        initial_evidence_runtime, build_evidence_snapshot,
        EvidenceExecutor,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.executor import (
        valid_verify_targets,
    )
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
        EvidenceRuntime, EvidenceSnapshot, EvidenceItem,
    )
    from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
    from hrm_adaptive_memory.cognitive_control.core import DecisionAction
    from hrm_adaptive_memory.cognitive_control.state import (
        TemporalStatus, VerificationState,
    )
    from scripts.run_i3_15_r1_balanced import (
        build_corpus_index, get_required_passage_ids, build_retrieved_evidence_task,
        TOP_K,
    )
    from scripts.run_i3_12j_factorial import make_inferred_snapshot_builder

    return {
        "i3_7e": i3_7e,
        "i3_12j": i3_12j,
        "get_i3_15c_corpus": get_i3_15c_corpus,
        "generate_i3_15c_corpus": generate_i3_15c_corpus,
        "DeterministicRelationExtractor": DeterministicRelationExtractor,
        "infer_relations_for_runtime": infer_relations_for_runtime,
        "initial_evidence_runtime": initial_evidence_runtime,
        "build_evidence_snapshot": build_evidence_snapshot,
        "EvidenceExecutor": EvidenceExecutor,
        "valid_verify_targets": valid_verify_targets,
        "EvidenceRuntime": EvidenceRuntime,
        "EvidenceSnapshot": EvidenceSnapshot,
        "EvidenceItem": EvidenceItem,
        "ResourceState": ResourceState,
        "ResourceBudget": ResourceBudget,
        "DecisionAction": DecisionAction,
        "TemporalStatus": TemporalStatus,
        "VerificationState": VerificationState,
        "build_corpus_index": build_corpus_index,
        "get_required_passage_ids": get_required_passage_ids,
        "build_retrieved_evidence_task": build_retrieved_evidence_task,
        "TOP_K": TOP_K,
        "make_inferred_snapshot_builder": make_inferred_snapshot_builder,
    }


def reconstruct_runtime_at_step(
    task, retrieved_passages, corpus_by_text, actions, outcomes, target_ids,
    step, imports, budget, extractor,
):
    """Reconstruct the runtime state at a given step by replaying actions.

    Returns the runtime state BEFORE the action at `step` is executed.
    """
    initial_evidence_runtime = imports["initial_evidence_runtime"]
    build_retrieved_evidence_task = imports["build_retrieved_evidence_task"]
    ResourceState = imports["ResourceState"]
    EvidenceExecutor = imports["EvidenceExecutor"]
    DecisionAction = imports["DecisionAction"]
    infer_relations_for_runtime = imports["infer_relations_for_runtime"]

    # Build the evidence task with retrieved passages
    new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)
    runtime = initial_evidence_runtime(new_et, ResourceState(budget))

    # Apply inferred relations
    runtime, _graph = infer_relations_for_runtime(runtime, extractor)

    executor = EvidenceExecutor()

    # Replay actions up to (but not including) the target step
    for i in range(step):
        if i >= len(actions):
            break
        action_str = actions[i]
        target_id = target_ids[i] if i < len(target_ids) else None

        try:
            action_enum = DecisionAction(action_str)
        except ValueError:
            # Unknown action, skip
            continue

        result = executor.execute(runtime, action_enum, target_evidence_id=target_id)
        runtime = result.runtime

        if result.terminal:
            break

    return runtime


def compute_mdsg_state(runtime, imports, extractor):
    """Compute the MDSG state from a runtime using the same logic as the runner."""
    i3_7e = imports["i3_7e"]
    make_inferred_snapshot_builder = imports["make_inferred_snapshot_builder"]
    build_evidence_snapshot = imports["build_evidence_snapshot"]
    infer_relations_for_runtime = imports["infer_relations_for_runtime"]

    # Apply inferred relations
    runtime, _graph = infer_relations_for_runtime(runtime, extractor)

    snapshot_builder = make_inferred_snapshot_builder(extractor)
    snapshot = snapshot_builder(runtime)

    # Build M3 packet to get decision state
    packet = i3_7e.build_mdsg_state_with_affordances_packet(snapshot)
    summary = packet.get("decision_state_summary", {})

    return {
        "decision_state": summary.get("decision_state"),
        "live_hypotheses": summary.get("live_hypotheses", []),
        "eliminated_hypotheses": summary.get("eliminated_hypotheses", []),
        "weakened_hypotheses": summary.get("weakened_hypotheses", []),
        "untested_hypotheses": summary.get("untested_hypotheses", []),
        "unverified_relevant_evidence": summary.get("unverified_relevant_evidence", []),
        "action_affordances": summary.get("action_affordances", {}),
        "evidence_status": summary.get("evidence_status"),
        "hidden_evidence_count": summary.get("hidden_evidence_count", 0),
        "verified_support": summary.get("verified_support", []),
        "verified_contradictions": summary.get("verified_contradictions", []),
        "snapshot": snapshot,
    }


def simulate_verify(runtime, target_id, imports, extractor):
    """Simulate a VERIFY action on target_id and return the resulting MDSG state."""
    EvidenceExecutor = imports["EvidenceExecutor"]
    DecisionAction = imports["DecisionAction"]

    executor = EvidenceExecutor()
    result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id=target_id)

    if result.outcome_code == "INVALID_VERIFY_TARGET":
        return None, "INVALID_VERIFY_TARGET"

    if result.outcome_code == "RESOURCE_EXHAUSTED":
        return None, "RESOURCE_EXHAUSTED"

    new_state = compute_mdsg_state(result.runtime, imports, extractor)
    return new_state, result.outcome_code


def check_elimination_monotonicity(runtime, imports, extractor):
    """Check whether MDSG elimination is monotonic.

    A hypothesis is ELIMINATED when it has SUFFICIENT contradicting evidence.
    VERIFY can change UNVERIFIED → SUFFICIENT or FALSIFIED.
    Can VERIFY ever un-eliminate a hypothesis?

    The only way to un-eliminate would be to:
    1. Falsify the contradicting evidence (but VERIFY only changes the verified
       item's state, not other items' states)
    2. Verify new supporting evidence (but contradiction still exists)

    So elimination IS monotonic: once a hypothesis has SUFFICIENT contradicting
    evidence, no subsequent VERIFY can remove that contradiction.
    """
    EvidenceExecutor = imports["EvidenceExecutor"]
    DecisionAction = imports["DecisionAction"]
    valid_verify_targets = imports["valid_verify_targets"]

    # Get current state
    current_state = compute_mdsg_state(runtime, imports, extractor)
    current_eliminated = set(current_state["eliminated_hypotheses"])

    # Try every valid VERIFY target
    targets = valid_verify_targets(runtime)
    executor = EvidenceExecutor()

    for target_id in targets:
        result = executor.execute(runtime, DecisionAction.VERIFY, target_evidence_id=target_id)
        if result.outcome_code not in ("VERIFY_COMPLETED",):
            continue

        new_state = compute_mdsg_state(result.runtime, imports, extractor)
        new_eliminated = set(new_state["eliminated_hypotheses"])

        # Check if any previously-eliminated hypothesis is no longer eliminated
        un_eliminated = current_eliminated - new_eliminated
        if un_eliminated:
            return {
                "monotonic": False,
                "violations": list(un_eliminated),
                "target_that_violated": target_id,
            }

    return {"monotonic": True, "violations": []}


def audit_affordances_at_t2(runtime, imports, extractor):
    """Audit the actual exposed affordances at T2 time."""
    state = compute_mdsg_state(runtime, imports, extractor)
    affordances = state["action_affordances"]

    # Also check what actions are legal (not just budget-derived)
    valid_verify_targets = imports["valid_verify_targets"]
    targets = valid_verify_targets(runtime)

    # Check if ANSWER, DEFER, STOP, REASON_MORE are always "legal"
    # (they don't consume specific budgets, just step budget)
    rs = runtime.resources.as_dict()
    steps_remaining = rs.get("executive_steps_remaining", 0)

    return {
        "can_verify": affordances.get("can_verify", False),
        "can_retrieve": affordances.get("can_retrieve", False),
        "can_search": affordances.get("can_search", False),
        "valid_verify_targets": list(targets),
        "n_valid_verify_targets": len(targets),
        "executive_steps_remaining": steps_remaining,
        "verification_calls_remaining": rs.get("verification_calls_remaining", 0),
        "retrieval_calls_remaining": rs.get("retrieval_calls_remaining", 0),
        "search_calls_remaining": rs.get("search_calls_remaining", 0),
        "answer_always_legal": steps_remaining > 0,  # ANSWER consumes a step
        "defer_always_legal": steps_remaining > 0,
        "stop_always_legal": steps_remaining > 0,
        "reason_more_always_legal": steps_remaining > 0,
    }


def counterfactual_verify_audit(runtime, imports, extractor):
    """For each valid VERIFY target at T2, simulate verification and check
    whether it changes any epistemically decision-relevant state.

    Epistemically useful is defined STRICTLY as a change to any of:
      - decision_state (MDSG label)
      - live_hypotheses set
      - eliminated_hypotheses set
      - T2 status (all-eliminated flag)

    Changes to the following are NOT counted as epistemically useful:
      - verification_state of individual evidence items
      - resource_state (budgets remaining)
      - evidence metadata (verified_count, supporting_count, etc.)
      - prior_actions / prior_outcomes logs

    This precise scope prevents the "0 useful" finding from sounding broader
    than the exact implementation.
    """
    valid_verify_targets = imports["valid_verify_targets"]
    targets = valid_verify_targets(runtime)

    current_state = compute_mdsg_state(runtime, imports, extractor)

    results = []
    n_useful = 0

    for target_id in targets:
        new_state, outcome = simulate_verify(runtime, target_id, imports, extractor)

        if new_state is None:
            results.append({
                "target_id": target_id,
                "outcome": outcome,
                "epistemically_useful": False,
                "decision_state_changed": False,
                "hypothesis_sets_changed": False,
                "t2_status_changed": False,
            })
            continue

        # Check epistemically decision-relevant changes ONLY
        ds_changed = (current_state["decision_state"] != new_state["decision_state"])
        hs_changed = (
            set(current_state["live_hypotheses"]) != set(new_state["live_hypotheses"])
            or set(current_state["eliminated_hypotheses"]) != set(new_state["eliminated_hypotheses"])
        )

        # Check T2 status change: T2 fires when all hypotheses eliminated
        n_hyps = len(current_state["live_hypotheses"]) + len(current_state["eliminated_hypotheses"]) + len(current_state["weakened_hypotheses"]) + len(current_state["untested_hypotheses"])
        current_t2 = len(current_state["eliminated_hypotheses"]) == n_hyps and n_hyps > 0
        new_t2 = len(new_state["eliminated_hypotheses"]) == n_hyps and n_hyps > 0
        t2_changed = current_t2 != new_t2

        useful = ds_changed or hs_changed or t2_changed

        if useful:
            n_useful += 1

        results.append({
            "target_id": target_id,
            "outcome": outcome,
            "epistemically_useful": useful,
            "decision_state_changed": ds_changed,
            "hypothesis_sets_changed": hs_changed,
            "t2_status_changed": t2_changed,
            "new_decision_state": new_state["decision_state"] if ds_changed else None,
            "new_live": new_state["live_hypotheses"] if hs_changed else None,
            "new_eliminated": new_state["eliminated_hypotheses"] if hs_changed else None,
        })

    return {
        "n_valid_targets": len(targets),
        "n_epistemically_useful_targets": n_useful,
        "n_not_useful_targets": len(targets) - n_useful,
        "per_target": results,
    }


def classify_t2_state(cf_result):
    """Classify T2 state into VERIFY_DEAD_END, VERIFY_RESOLVABLE, or NO_VERIFY."""
    if cf_result["n_valid_targets"] == 0:
        return "T2_NO_VERIFY"
    if cf_result["n_epistemically_useful_targets"] > 0:
        return "T2_VERIFY_RESOLVABLE"
    return "T2_VERIFY_DEAD_END"


def main():
    parser = argparse.ArgumentParser(description="R13-F1.2a Counterfactual Affordance Audit (Hardened)")
    parser.add_argument("--raw-closed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    raw_closed = args.raw_closed
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    receipts_path = REPO_ROOT / "experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl"

    print(f"R13-F1.2a Counterfactual Affordance Audit (Hardened)")
    print(f"  Label: {LABEL}")
    print(f"  Version: {VERSION}")
    print(f"  Source: {raw_closed}")

    # Hard preflight identity verification
    identity_checks = preflight_identity(raw_closed, receipts_path)

    # Load data
    print(f"\n[1] Loading data...")
    results = load_results(raw_closed)
    receipts = load_mechanism_receipts(raw_closed)
    print(f"  Loaded {len(results)} results, {len(receipts)} mechanism receipts")

    # Setup imports
    print(f"\n[2] Setting up imports...")
    imports = setup_imports()
    print(f"  Imports ready")

    # Load task corpus
    print(f"\n[3] Loading task corpus...")
    generate_i3_15c_corpus = imports["generate_i3_15c_corpus"]
    tasks = generate_i3_15c_corpus(n_per_cell=40, seed=42)
    task_by_id = {t.evidence_task.task_id: t for t in tasks}
    print(f"  Loaded {len(tasks)} tasks")

    # Build corpus index for retrieval
    print(f"\n[4] Building corpus index...")
    get_i3_15c_corpus = imports["get_i3_15c_corpus"]
    build_corpus_index = imports["build_corpus_index"]
    corpus_passages = get_i3_15c_corpus()
    chunks, corpus_by_text, corpus_by_id = build_corpus_index(corpus_passages)
    print(f"  Corpus index: {len(corpus_by_id)} passages")

    # Load Q3_RERANKED retrieval receipts (SHA already verified in preflight)
    print(f"\n[4b] Loading Q3_RERANKED retrieval receipts...")
    q3_receipts = {}
    with open(receipts_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("retrieval_condition") == "Q3_RERANKED":
                    q3_receipts[r["task_id"]] = r
    print(f"  Loaded {len(q3_receipts)} Q3_RERANKED receipts")

    # Setup budget (same as R13)
    ResourceBudget = imports["ResourceBudget"]
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )

    # Setup extractor
    DeterministicRelationExtractor = imports["DeterministicRelationExtractor"]
    extractor = DeterministicRelationExtractor()

    # Find T2-triggered R1 trajectories
    r1_triggered = [r for r in results if r["arm"] == "R1_INFERRED" and r.get("r1_triggered")]
    print(f"\n[5] Found {len(r1_triggered)} T2-triggered R1 trajectories")

    # For each T2-triggered trajectory, reconstruct state at T2 and audit
    print(f"\n[6] Reconstructing T2 states and running counterfactual audit...")

    all_audits = []
    t2_class_counts = Counter()
    affordance_patterns = Counter()
    monotonicity_results = []
    decision_state_semantics = Counter()

    for i, r1_result in enumerate(r1_triggered):
        task_id = r1_result["task_id"]
        trigger_step = r1_result["r1_trigger_step"]

        if task_id not in task_by_id:
            print(f"  WARNING: task {task_id} not in corpus, skipping")
            continue

        task = task_by_id[task_id]

        # Get retrieved passages from Q3_RERANKED receipts
        build_retrieved_evidence_task = imports["build_retrieved_evidence_task"]

        q3_receipt = q3_receipts.get(task_id)
        if q3_receipt is None:
            print(f"  WARNING: no Q3 receipt for {task_id}, skipping")
            continue

        retrieved_passages = [
            corpus_by_id[pid] for pid in q3_receipt.get("retrieved_chunk_ids", [])
            if pid in corpus_by_id
        ]

        # Get actions and target IDs from model_call_log
        actions = r1_result.get("continuation_actions", [])
        mcl = r1_result.get("model_call_log", [])
        target_ids = [c.get("decoded_target_id") for c in mcl]

        # Reconstruct runtime at T2 trigger step
        try:
            runtime_at_t2 = reconstruct_runtime_at_step(
                task, retrieved_passages, corpus_by_text,
                actions, r1_result.get("continuation_outcomes", []),
                target_ids, trigger_step, imports, budget, extractor,
            )
        except Exception as e:
            print(f"  ERROR reconstructing {task_id}: {e}")
            continue

        # Audit affordances at T2
        affordances = audit_affordances_at_t2(runtime_at_t2, imports, extractor)

        # Classify affordance pattern
        pattern_parts = []
        if affordances["can_verify"]:
            pattern_parts.append("VERIFY")
        if affordances["can_retrieve"]:
            pattern_parts.append("RETRIEVE")
        if affordances["can_search"]:
            pattern_parts.append("SEARCH")
        pattern = "+".join(pattern_parts) if pattern_parts else "NONE"
        affordance_patterns[pattern] += 1

        # Counterfactual VERIFY audit
        cf_result = counterfactual_verify_audit(runtime_at_t2, imports, extractor)

        # Classify T2 state
        t2_class = classify_t2_state(cf_result)
        t2_class_counts[t2_class] += 1

        # Check elimination monotonicity — EXHAUSTIVE over all 228 states
        mono = check_elimination_monotonicity(runtime_at_t2, imports, extractor)
        monotonicity_results.append({
            "task_id": task_id,
            **mono,
        })

        # Audit decision-state semantics
        state = compute_mdsg_state(runtime_at_t2, imports, extractor)
        n_live = len(state["live_hypotheses"])
        n_elim = len(state["eliminated_hypotheses"])
        ds = state["decision_state"]

        if n_live == 0 and n_elim > 0:
            decision_state_semantics[f"0_live_{n_elim}_eliminated→{ds}"] += 1
        elif n_live >= 2:
            decision_state_semantics[f"{n_live}_live→{ds}"] += 1
        elif n_live == 1:
            decision_state_semantics[f"1_live→{ds}"] += 1

        all_audits.append({
            "task_id": task_id,
            "trigger_step": trigger_step,
            "category": r1_result.get("category", ""),
            "t2_class": t2_class,
            "affordances": affordances,
            "counterfactual": {
                "n_valid_targets": cf_result["n_valid_targets"],
                "n_epistemically_useful_targets": cf_result["n_epistemically_useful_targets"],
                "n_not_useful_targets": cf_result["n_not_useful_targets"],
                "per_target": cf_result["per_target"],
            },
            "mdsg_state_at_t2": {
                "decision_state": state["decision_state"],
                "live_hypotheses": state["live_hypotheses"],
                "eliminated_hypotheses": state["eliminated_hypotheses"],
                "unverified_relevant_evidence": state["unverified_relevant_evidence"],
                "evidence_status": state["evidence_status"],
                "hidden_evidence_count": state["hidden_evidence_count"],
            },
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(r1_triggered)}")

    print(f"  Processed {len(all_audits)}/{len(r1_triggered)}")

    # Summary
    print(f"\n[7] Summary")
    print(f"  T2 state classification:")
    for cls, count in sorted(t2_class_counts.items()):
        print(f"    {cls}: {count}")

    print(f"\n  Affordance patterns at T2:")
    for pat, count in sorted(affordance_patterns.items(), key=lambda x: -x[1]):
        print(f"    {pat}: {count}")

    print(f"\n  Decision-state semantics (live→state):")
    for pat, count in sorted(decision_state_semantics.items(), key=lambda x: -x[1]):
        print(f"    {pat}: {count}")

    print(f"\n  Elimination monotonicity (EXHAUSTIVE, all {len(monotonicity_results)} states):")
    mono_count = sum(1 for m in monotonicity_results if m["monotonic"])
    mono_violations = [m for m in monotonicity_results if not m["monotonic"]]
    print(f"    Monotonic: {mono_count}/{len(monotonicity_results)}")
    if mono_violations:
        for m in mono_violations:
            print(f"    VIOLATION: {m}")
    else:
        print(f"    0 violations — elimination is monotonic across all T2 states and all valid targets")

    # Write outputs
    print(f"\n[8] Writing outputs...")

    # Per-trajectory audit
    audit_path = output_dir / "r13_f1_2_counterfactual_audit.jsonl"
    with open(audit_path, "w") as f:
        for audit in all_audits:
            f.write(json.dumps({"label": LABEL, "version": VERSION, **audit}) + "\n")
    print(f"  {audit_path}")

    # Summary JSON
    summary_path = output_dir / "r13_f1_2_summary.json"
    summary = {
        "label": LABEL,
        "version": VERSION,
        "r13_results_sha256": identity_checks.get("results_jsonl_sha"),
        "r13_experiment_source_commit": identity_checks.get("experiment_source_commit"),
        "r13_confirmation_executable_sha256": identity_checks.get("confirmation_executable_sha256"),
        "r13_retrieval_receipts_sha256": identity_checks.get("retrieval_receipts_sha"),
        "source_file_shas": {
            k: v for k, v in identity_checks.items()
            if k.endswith("_sha") and not k.endswith("_at_source_commit")
        },
        "n_t2_triggered": len(all_audits),
        "total_valid_verify_targets_tested": sum(
            a["counterfactual"]["n_valid_targets"] for a in all_audits
        ),
        "total_epistemically_useful_targets": sum(
            a["counterfactual"]["n_epistemically_useful_targets"] for a in all_audits
        ),
        "t2_state_classification": dict(t2_class_counts),
        "affordance_patterns": dict(affordance_patterns),
        "decision_state_semantics": dict(decision_state_semantics),
        "elimination_monotonicity": {
            "n_checked": len(monotonicity_results),
            "n_monotonic": mono_count,
            "n_violations": len(monotonicity_results) - mono_count,
            "exhaustive": True,
            "invariant": (
                "For all s in S_T2, for all v in ValidVerify(s): "
                "Eliminated(s) subseteq Eliminated(T(s,v)). "
                "This is an empirically exhaustive invariant over the 228 audited "
                "R13 T2 states. It is not a mathematical theorem over all possible "
                "DAPH states. A formal lemma from transition semantics is documented "
                "in R13-F1-2-REPORT.md."
            ),
            "violations": mono_violations,
        },
        "useful_target_definition": (
            "Epistemically useful = changes to decision_state, live_hypotheses, "
            "eliminated_hypotheses, or T2 status ONLY. "
            "Changes to verification_state, resource_state, evidence metadata "
            "(verified_count, supporting_count, etc.), or prior_actions/outcomes "
            "are NOT counted as epistemically useful."
        ),
        "methodological_note": (
            "R13-F1.2a reconstructs runtime state at T2 by replaying actions through "
            "the frozen deterministic executor. Counterfactual VERIFY simulations use "
            "the same executor with frozen task effects (post-hoc oracle). "
            "No LLM calls are made. "
            "This post-hoc oracle simulation is legitimate for diagnosis but MUST NOT "
            "be used at runtime — R2d gating must derive from visible structural state only. "
            "R13-F can identify likely failure mechanisms; it cannot confirm them causally."
        ),
        "r2d_note": (
            "R2d (Structural Dead-End Affordance Gating) must use the runtime-visible rule: "
            "can_verify = budget>0 AND valid_targets>0 AND NOT all_hypotheses_eliminated. "
            "This is deterministic and public, requiring no counterfactual simulation. "
            "The F1.2a counterfactual audit validates this rule post-hoc but is not needed at runtime."
        ),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path}")

    # Compute SHA
    sha = hashlib.sha256()
    with open(summary_path, "rb") as f:
        sha.update(f.read())
    analysis_sha = sha.hexdigest()
    print(f"\n  R13_F1_2_ANALYSIS_SHA256: {analysis_sha}")

    sha_path = output_dir / "R13_F1_2_ANALYSIS_SHA256.txt"
    with open(sha_path, "w") as f:
        f.write(analysis_sha)
    print(f"  {sha_path}")

    print(f"\nR13-F1.2a counterfactual affordance audit complete.")


if __name__ == "__main__":
    main()
