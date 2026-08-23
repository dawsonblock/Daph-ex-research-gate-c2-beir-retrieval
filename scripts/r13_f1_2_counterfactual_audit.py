#!/usr/bin/env python3
"""
R13-F1.2: Counterfactual Affordance Audit.

Reads ONLY from raw_closed/ — never mutates R13 artifacts.
All outputs are labeled POST_HOC_EXPLORATORY.

This script reconstructs the runtime state at T2 trigger time for each
T2-triggered trajectory, then simulates counterfactual VERIFY actions
on every valid target to determine whether any could change the MDSG state.

Key questions answered:
  1. What affordances were actually exposed at T2? (not just what was selected)
  2. For each valid VERIFY target at T2, could verifying it change any
     decision-relevant state (hypothesis sets, decision_state, T2 status)?
  3. Are T2 states VERIFY_DEAD_END, VERIFY_RESOLVABLE, or NO_VERIFY?
  4. Is MDSG elimination monotonic within a trajectory?
  5. Is NEEDS_DISCRIMINATION semantically correct when 0 hypotheses are live?

R13-F can identify likely failure mechanisms; it cannot confirm them causally.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LABEL = "POST_HOC_EXPLORATORY"
R13_DATASET_SHA256 = "56cff26a4f13d519810a77f61f7a8280cb6d665e729270ff421966cdeccb62db"


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
    whether it changes any decision-relevant state."""
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
                "state_changed": False,
                "decision_state_changed": False,
                "hypothesis_sets_changed": False,
                "t2_status_changed": False,
                "useful": False,
            })
            continue

        # Check what changed
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
            "state_changed": useful,
            "decision_state_changed": ds_changed,
            "hypothesis_sets_changed": hs_changed,
            "t2_status_changed": t2_changed,
            "useful": useful,
            "new_decision_state": new_state["decision_state"] if ds_changed else None,
            "new_live": new_state["live_hypotheses"] if hs_changed else None,
            "new_eliminated": new_state["eliminated_hypotheses"] if hs_changed else None,
        })

    return {
        "n_valid_targets": len(targets),
        "n_useful_targets": n_useful,
        "n_useless_targets": len(targets) - n_useful,
        "per_target": results,
    }


def classify_t2_state(cf_result):
    """Classify T2 state into VERIFY_DEAD_END, VERIFY_RESOLVABLE, or NO_VERIFY."""
    if cf_result["n_valid_targets"] == 0:
        return "T2_NO_VERIFY"
    if cf_result["n_useful_targets"] > 0:
        return "T2_VERIFY_RESOLVABLE"
    return "T2_VERIFY_DEAD_END"


def main():
    parser = argparse.ArgumentParser(description="R13-F1.2 Counterfactual Affordance Audit")
    parser.add_argument("--raw-closed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--r13-dataset-sha", default=R13_DATASET_SHA256)
    args = parser.parse_args()

    raw_closed = args.raw_closed
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"R13-F1.2 Counterfactual Affordance Audit")
    print(f"  Label: {LABEL}")
    print(f"  Source: {raw_closed}")

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

    # Load Q3_RERANKED retrieval receipts
    print(f"\n[4b] Loading Q3_RERANKED retrieval receipts...")
    receipts_path = REPO_ROOT / "experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl"
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

        # Check elimination monotonicity (only for first few to save time)
        if i < 10:
            mono = check_elimination_monotonicity(runtime_at_t2, imports, extractor)
            monotonicity_results.append(mono)

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
                "n_useful_targets": cf_result["n_useful_targets"],
                "n_useless_targets": cf_result["n_useless_targets"],
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

    print(f"\n  Elimination monotonicity (first 10):")
    mono_count = sum(1 for m in monotonicity_results if m["monotonic"])
    print(f"    Monotonic: {mono_count}/{len(monotonicity_results)}")
    for m in monotonicity_results:
        if not m["monotonic"]:
            print(f"    VIOLATION: {m}")

    # Write outputs
    print(f"\n[8] Writing outputs...")

    # Per-trajectory audit
    audit_path = output_dir / "r13_f1_2_counterfactual_audit.jsonl"
    with open(audit_path, "w") as f:
        for audit in all_audits:
            f.write(json.dumps({"label": LABEL, **audit}) + "\n")
    print(f"  {audit_path}")

    # Summary JSON
    summary_path = output_dir / "r13_f1_2_summary.json"
    summary = {
        "label": LABEL,
        "version": "R13-F1.2",
        "r13_dataset_sha256": args.r13_dataset_sha,
        "n_t2_triggered": len(all_audits),
        "t2_state_classification": dict(t2_class_counts),
        "affordance_patterns": dict(affordance_patterns),
        "decision_state_semantics": dict(decision_state_semantics),
        "elimination_monotonicity": {
            "n_checked": len(monotonicity_results),
            "n_monotonic": mono_count,
            "n_violations": len(monotonicity_results) - mono_count,
            "results": monotonicity_results,
        },
        "methodological_note": (
            "R13-F1.2 reconstructs runtime state at T2 by replaying actions through "
            "the frozen deterministic executor. Counterfactual VERIFY simulations use "
            "the same executor with frozen task effects. No LLM calls are made. "
            "R13-F can identify likely failure mechanisms; it cannot confirm them causally."
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

    print(f"\nR13-F1.2 counterfactual affordance audit complete.")


if __name__ == "__main__":
    main()
