#!/usr/bin/env python3
"""R12.9: Production Trajectory Qualification — Forensic Analysis.

Uses the completed 32-trajectory smoke results to diagnose:
  A. False-T2 activation on matched-negative controls
  B. VERIFY loop behavior

Does NOT modify any scientific protocol.
Does NOT launch R13.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load modules
spec_12j = importlib.util.spec_from_file_location(
    "i3_12j", str(REPO_ROOT / "scripts" / "run_i3_12j_factorial.py"))
i3_12j = importlib.util.module_from_spec(spec_12j)
spec_12j.loader.exec_module(i3_12j)
i3_7e = i3_12j.i3_7e

from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
    generate_i3_15c_corpus, get_i3_15c_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.integration import (
    infer_relations_for_runtime,
)
from hrm_adaptive_memory.executive.semantic_relations.schema import RelationType
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.evidence_benchmark import (
    initial_evidence_runtime, build_evidence_snapshot, EvidenceExecutor,
)
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.model_decoder import decode_output
from scripts.run_i3_15_r1_balanced import (
    get_required_passage_ids, build_retrieved_evidence_task,
    adapt_local_system_prompt, TOP_K,
)
from scripts.run_i3_15c_factorial import _get_cached_corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stratum_from_category(cat: str) -> str:
    if cat.startswith("t2_conflict_immediate"):
        return "T2_CONFLICT_IMMEDIATE"
    elif cat.startswith("t2_conflict_late_1"):
        return "T2_CONFLICT_LATE_1"
    elif cat.startswith("t2_conflict_late_2"):
        return "T2_CONFLICT_LATE_2"
    elif cat.startswith("t2_conflict_late_3"):
        return "T2_CONFLICT_LATE_3"
    elif cat.startswith("matched_neg_immediate"):
        return "MATCHED_NEG_IMMEDIATE"
    elif cat.startswith("matched_neg_late"):
        return "MATCHED_NEG_LATE"
    elif cat.startswith("defer_control"):
        return "DEFER_CONTROL"
    elif cat.startswith("answer_control"):
        return "ANSWER_CONTROL"
    return "UNKNOWN"


def load_smoke_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_smoke_model_calls(path: Path) -> list[dict]:
    calls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(json.loads(line))
    return calls


# ---------------------------------------------------------------------------
# R12.9A: False-T2 Forensic Analysis
# ---------------------------------------------------------------------------

def reconstruct_elimination_proof(
    task, corpus_by_text, corpus_by_id, receipts, extractor,
) -> dict[str, Any]:
    """Reconstruct the exact hypothesis-elimination proof for a task.

    Simulates the live pipeline: infer relations, build snapshot,
    check which hypotheses are eliminated and why.
    """
    et = task.evidence_task
    task_id = et.task_id

    # Get pre-retrieved passages from receipts
    receipt = receipts.get(task_id)
    if receipt is None:
        return {"error": "no receipt", "task_id": task_id}

    corpus_passages, corpus_by_text_, corpus_by_id_, chunks, corpus_sha = (
        _get_cached_corpus())
    retrieved_passages = [
        corpus_by_id_[pid] for pid in receipt.get("retrieved_chunk_ids", [])
        if pid in corpus_by_id_
    ]
    new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text_)

    # Build initial runtime
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    runtime = initial_evidence_runtime(new_et, ResourceState(budget))

    # Get GOLD relations from the task
    gold_relations = {}
    if hasattr(task, 'gold_relations'):
        for gr in task.gold_relations:
            gold_relations[(gr.evidence_id, gr.hypothesis_id)] = gr.relation

    # Get INFERRED relations
    new_runtime, graph = infer_relations_for_runtime(runtime, extractor)

    inferred_relations = {}
    for rel in graph.relations:
        inferred_relations[(rel.evidence_id, rel.hypothesis_id)] = rel.relation.value
        inferred_relations[f"{rel.evidence_id}_{rel.hypothesis_id}_reason"] = rel.reason_code.value

    # Compare gold vs inferred
    relation_comparison = []
    all_pairs = set(gold_relations.keys()) | set(
        (k[0], k[1]) for k in inferred_relations.keys()
        if isinstance(k, tuple)
    )
    for (eid, hid) in sorted(all_pairs):
        gold = gold_relations.get((eid, hid), "UNKNOWN")
        inferred = inferred_relations.get((eid, hid), "UNKNOWN")
        reason = inferred_relations.get(f"{eid}_{hid}_reason", "")
        match = gold == inferred
        relation_comparison.append({
            "evidence_id": eid,
            "hypothesis_id": hid,
            "gold": gold,
            "inferred": inferred,
            "match": match,
            "reason_code": reason,
        })

    # Check elimination at initial state
    snapshot = build_evidence_snapshot(new_runtime)
    viability = i3_7e._classify_from_snapshot(snapshot)

    elimination_proof = {}
    for h_id, info in viability.items():
        if info["status"] == "ELIMINATED":
            elimination_proof[h_id] = {
                "status": "ELIMINATED",
                "contradicting_evidence": info["contradicting_evidence"],
                "supporting_evidence": info["supporting_evidence"],
            }
        else:
            elimination_proof[h_id] = {
                "status": info["status"],
                "contradicting_evidence": info["contradicting_evidence"],
                "supporting_evidence": info["supporting_evidence"],
            }

    # Check T2
    n_hypotheses = len(new_et.hypotheses)
    eliminated = [h_id for h_id, info in viability.items()
                  if info["status"] == "ELIMINATED"]
    t2_fires = len(eliminated) == n_hypotheses and n_hypotheses > 0

    # Also check evidence states
    evidence_states = []
    for ev in snapshot.visible_evidence:
        evidence_states.append({
            "evidence_id": ev.evidence_id,
            "verification_state": ev.verification_state.value if hasattr(ev.verification_state, 'value') else str(ev.verification_state),
            "temporal_status": ev.temporal_status.value if hasattr(ev.temporal_status, 'value') else str(ev.temporal_status),
            "supports": list(ev.supports),
            "contradicts": list(ev.contradicts),
            "proposition": ev.proposition[:200],
        })

    # Hypothesis propositions
    hypothesis_props = {}
    for h in new_et.hypotheses:
        hypothesis_props[h.hypothesis_id] = h.proposition[:200]

    return {
        "task_id": task_id,
        "category": et.category,
        "stratum": stratum_from_category(et.category),
        "n_hypotheses": n_hypotheses,
        "t2_fires_initial": t2_fires,
        "eliminated_at_initial": eliminated,
        "elimination_proof": elimination_proof,
        "relation_comparison": relation_comparison,
        "evidence_states": evidence_states,
        "hypothesis_propositions": hypothesis_props,
    }


def classify_false_t2(forensic: dict) -> str:
    """Classify the root cause of a false T2 activation."""
    if not forensic["t2_fires_initial"]:
        return "NO_FALSE_T2"

    stratum = forensic["stratum"]
    if not stratum.startswith("MATCHED_NEG") and not stratum.endswith("CONTROL"):
        return "EXPECTED_T2"  # T2-positive stratum, not a false positive

    # Check for semantic false contradictions
    has_semantic_error = False
    has_verification_bug = False
    has_generator_bug = False

    for cmp in forensic["relation_comparison"]:
        if not cmp["match"]:
            if cmp["gold"] == "NEUTRAL" and cmp["inferred"] == "CONTRADICT":
                has_semantic_error = True
            elif cmp["gold"] == "SUPPORT" and cmp["inferred"] == "CONTRADICT":
                has_semantic_error = True
            elif cmp["gold"] == "CONTRADICT" and cmp["inferred"] == "NEUTRAL":
                has_semantic_error = True  # missed contradiction

    # Check if elimination is caused by inferred CONTRADICT on evidence
    # that should be NEUTRAL
    for h_id, proof in forensic["elimination_proof"].items():
        if proof["status"] == "ELIMINATED":
            for eid in proof["contradicting_evidence"]:
                # Find the relation comparison for this evidence-hypothesis pair
                for cmp in forensic["relation_comparison"]:
                    if cmp["evidence_id"] == eid and cmp["hypothesis_id"] == h_id:
                        if cmp["gold"] == "NEUTRAL" and cmp["inferred"] == "CONTRADICT":
                            return "SEMANTIC_FALSE_CONTRADICTION"
                        if cmp["gold"] != "CONTRADICT" and cmp["inferred"] == "CONTRADICT":
                            return "SEMANTIC_FALSE_CONTRADICTION"

    # Check if evidence verification state is wrong
    for ev in forensic["evidence_states"]:
        if ev["verification_state"] == "SUFFICIENT":
            # Check if this evidence should be UNVERIFIED
            for cmp in forensic["relation_comparison"]:
                if cmp["evidence_id"] == ev["evidence_id"]:
                    # If evidence has CONTRADICT but should be NEUTRAL,
                    # and it's SUFFICIENT, that's the issue
                    pass

    return "IMPLEMENTATION_BUG"


def run_false_t2_forensics(
    results: list[dict],
    model_calls: list[dict],
    output_dir: Path,
) -> dict:
    """Run complete false-T2 forensic analysis."""
    print("\n" + "=" * 80)
    print("R12.9A: FALSE-T2 FORENSIC ANALYSIS")
    print("=" * 80)

    # Generate tasks
    tasks = generate_i3_15c_corpus(n_per_cell=1, seed=42)
    task_by_id = {t.evidence_task.task_id: t for t in tasks}

    # Load receipts
    receipts = {}
    receipts_path = REPO_ROOT / "experiments/v2b_i3_15c/confirmation/retrieval_receipts.jsonl"
    with open(receipts_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("retrieval_condition") == "Q3_RERANKED":
                receipts[r["task_id"]] = r

    # Load corpus
    corpus_passages, corpus_by_text, corpus_by_id, chunks, corpus_sha = (
        _get_cached_corpus())

    extractor = DeterministicRelationExtractor()

    # Find all R1 trajectories where T2 triggered
    r1_triggered = [r for r in results if r.get("arm") == "R1_INFERRED" and r.get("r1_triggered")]
    print(f"\nR1 trajectories with T2 triggered: {len(r1_triggered)}")

    # Find false T2 on controls
    false_t2_cases = []
    for r in r1_triggered:
        cat = r.get("category", "")
        stratum = stratum_from_category(cat)
        if stratum.startswith("MATCHED_NEG") or stratum.endswith("CONTROL"):
            false_t2_cases.append(r)
            print(f"\n  FALSE T2: {r['task_id']} stratum={stratum}")

    # Reconstruct elimination proofs for ALL R1-triggered tasks
    forensics = []
    for r in r1_triggered:
        task = task_by_id.get(r["task_id"])
        if task is None:
            continue
        forensic = reconstruct_elimination_proof(
            task, corpus_by_text, corpus_by_id, receipts, extractor)
        classification = classify_false_t2(forensic)
        forensic["classification"] = classification
        forensics.append(forensic)
        print(f"\n  {r['task_id']} ({forensic['stratum']}): "
              f"T2={forensic['t2_fires_initial']} → {classification}")

    # Also check tasks where T2 should fire (T2-positive)
    t2_positive_tasks = [r for r in results if r.get("arm") == "R1_INFERRED"
                         and stratum_from_category(r.get("category", "")).startswith("T2_CONFLICT")]
    for r in t2_positive_tasks:
        if r.get("r1_triggered"):
            continue  # Already analyzed
        task = task_by_id.get(r["task_id"])
        if task is None:
            continue
        forensic = reconstruct_elimination_proof(
            task, corpus_by_text, corpus_by_id, receipts, extractor)
        forensic["classification"] = "MISSED_T2" if not forensic["t2_fires_initial"] else "NO_FALSE_T2"
        forensics.append(forensic)

    # Write forensics
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "false_t2_forensics.jsonl", "w") as f:
        for f_rec in forensics:
            f.write(json.dumps(f_rec, default=str) + "\n")

    # Summary
    classifications = Counter(f["classification"] for f in forensics)
    print(f"\nClassification summary: {dict(classifications)}")

    # Detailed analysis of false T2 cases
    print("\n--- FALSE T2 DETAILED ANALYSIS ---")
    for f in forensics:
        if f["classification"] in ("SEMANTIC_FALSE_CONTRADICTION", "IMPLEMENTATION_BUG"):
            print(f"\n  Task: {f['task_id']} ({f['stratum']})")
            print(f"  Classification: {f['classification']}")
            print(f"  T2 fires: {f['t2_fires_initial']}")
            print(f"  Eliminated: {f['eliminated_at_initial']}")
            print(f"  Hypothesis propositions:")
            for h_id, prop in f["hypothesis_propositions"].items():
                print(f"    {h_id}: {prop}")
            print(f"  Evidence states:")
            for ev in f["evidence_states"]:
                print(f"    {ev['evidence_id']}: verify={ev['verification_state']} "
                      f"supports={ev['supports']} contradicts={ev['contradicts']}")
                print(f"      proposition: {ev['proposition'][:100]}")
            print(f"  Relation comparison (gold vs inferred):")
            for cmp in f["relation_comparison"]:
                marker = " ← MISMATCH" if not cmp["match"] else ""
                print(f"    {cmp['evidence_id']}→{cmp['hypothesis_id']}: "
                      f"gold={cmp['gold']} inferred={cmp['inferred']} "
                      f"reason={cmp['reason_code']}{marker}")
            print(f"  Elimination proof:")
            for h_id, proof in f["elimination_proof"].items():
                print(f"    {h_id}: {proof['status']} "
                      f"contradicted_by={proof['contradicting_evidence']} "
                      f"supported_by={proof['supporting_evidence']}")

    summary = {
        "total_r1_trajectories": len(r1_triggered),
        "false_t2_count": len(false_t2_cases),
        "classifications": dict(classifications),
        "false_t2_task_ids": [r["task_id"] for r in false_t2_cases],
    }
    with open(output_dir / "false_t2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# R12.9B: VERIFY Loop Forensics
# ---------------------------------------------------------------------------

def run_verify_forensics(
    results: list[dict],
    model_calls: list[dict],
    output_dir: Path,
) -> dict:
    """Analyze every VERIFY transition in the smoke results."""
    print("\n" + "=" * 80)
    print("R12.9B: VERIFY-LOOP FORENSIC ANALYSIS")
    print("=" * 80)

    # Group model calls by trajectory
    calls_by_trajectory = defaultdict(list)
    for c in model_calls:
        key = c.get("trajectory_key")
        if key:
            calls_by_trajectory[key].append(c)

    verify_records = []
    verify_stats = {
        "total_verify_calls": 0,
        "useful_verify": 0,
        "noop_verify": 0,
        "repeated_target_verify": 0,
        "already_verified_target": 0,
        "verify_trajectories": 0,
        "verify_loop_step_limit": 0,
    }

    for r in results:
        key = r.get("trajectory_key", "")
        calls = calls_by_trajectory.get(key, [])
        verify_calls = [c for c in calls if c.get("executed_action") == "VERIFY"
                        or c.get("proposal_action") == "VERIFY"]

        if not verify_calls:
            continue

        verify_stats["verify_trajectories"] += 1
        actions = r.get("continuation_actions", [])
        verify_targets = []
        prev_target = None
        prev_targets_set = set()

        for i, call in enumerate(verify_calls):
            target = call.get("target_id") or call.get("proposal_target_id")
            step = call.get("step", i)

            # Check if this target was already verified
            already_verified = target in prev_targets_set

            # Check if this is a repeated consecutive target
            repeated_consecutive = target == prev_target

            # Check if the MDSG state changed (decision-relevant)
            # We can infer this from the routing_log or decision_state_log
            routing_log = r.get("routing_log", [])
            decision_state_log = r.get("decision_state_log", [])

            state_changed = False
            if i < len(decision_state_log) - 1:
                curr_state = decision_state_log[i].get("decision_state", "")
                next_state = decision_state_log[i + 1].get("decision_state", "")
                state_changed = curr_state != next_state

            # Also check hypothesis status changes
            hyp_changed = False
            if i < len(decision_state_log) - 1:
                curr_elim = set(decision_state_log[i].get("eliminated_hypotheses", []))
                next_elim = set(decision_state_log[i + 1].get("eliminated_hypotheses", []))
                hyp_changed = curr_elim != next_elim

            useful = state_changed or hyp_changed

            record = {
                "trajectory_key": key,
                "task_id": r.get("task_id"),
                "arm": r.get("arm"),
                "stratum": stratum_from_category(r.get("category", "")),
                "step": step,
                "target_id": target,
                "already_verified_target": already_verified,
                "repeated_consecutive": repeated_consecutive,
                "decision_state_changed": state_changed,
                "hypothesis_status_changed": hyp_changed,
                "useful": useful,
                "verification_budget_before": call.get("verification_budget_before"),
                "verification_budget_after": call.get("verification_budget_after"),
            }
            verify_records.append(record)

            verify_stats["total_verify_calls"] += 1
            if useful:
                verify_stats["useful_verify"] += 1
            else:
                verify_stats["noop_verify"] += 1
            if repeated_consecutive:
                verify_stats["repeated_target_verify"] += 1
            if already_verified:
                verify_stats["already_verified_target"] += 1

            prev_target = target
            prev_targets_set.add(target)

        # Check if trajectory hit step limit due to verify loop
        if r.get("terminal_result") == "RESOURCE_EXHAUSTED" or r.get("terminal_result") == "STEP_LIMIT":
            if all(a == "VERIFY" for a in actions):
                verify_stats["verify_loop_step_limit"] += 1

    # Calculate rates
    total = max(verify_stats["total_verify_calls"], 1)
    summary = {
        **verify_stats,
        "useful_verify_rate": verify_stats["useful_verify"] / total,
        "noop_verify_rate": verify_stats["noop_verify"] / total,
        "repeated_target_rate": verify_stats["repeated_target_verify"] / total,
        "already_verified_rate": verify_stats["already_verified_target"] / total,
        "mean_verify_per_trajectory": (
            verify_stats["total_verify_calls"] /
            max(verify_stats["verify_trajectories"], 1)
        ),
    }

    print(f"\n  Total VERIFY calls: {summary['total_verify_calls']}")
    print(f"  Useful VERIFY: {summary['useful_verify']} ({summary['useful_verify_rate']:.1%})")
    print(f"  No-op VERIFY: {summary['noop_verify']} ({summary['noop_verify_rate']:.1%})")
    print(f"  Repeated consecutive target: {summary['repeated_target_verify']} ({summary['repeated_target_rate']:.1%})")
    print(f"  Already-verified target: {summary['already_verified_target']} ({summary['already_verified_rate']:.1%})")
    print(f"  Trajectories with VERIFY: {summary['verify_trajectories']}")
    print(f"  Mean VERIFY per trajectory: {summary['mean_verify_per_trajectory']:.1f}")
    print(f"  Verify-loop step-limit: {summary['verify_loop_step_limit']}")

    # Write records
    with open(output_dir / "verify_forensics.jsonl", "w") as f:
        for rec in verify_records:
            f.write(json.dumps(rec, default=str) + "\n")

    with open(output_dir / "verify_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Detailed per-trajectory report
    print("\n  Per-trajectory VERIFY analysis:")
    by_traj = defaultdict(list)
    for rec in verify_records:
        by_traj[rec["trajectory_key"]].append(rec)

    for key, recs in sorted(by_traj.items()):
        task_id = recs[0]["task_id"]
        arm = recs[0]["arm"]
        stratum = recs[0]["stratum"]
        targets = [r["target_id"] for r in recs]
        useful = sum(r["useful"] for r in recs)
        noop = sum(not r["useful"] for r in recs)
        already = sum(r["already_verified_target"] for r in recs)
        print(f"    {task_id} {arm} ({stratum}): "
              f"{len(recs)} VERIFY calls, targets={targets}, "
              f"useful={useful}, noop={noop}, already_verified={already}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    smoke_dir = Path(os.environ.get(
        "SMOKE_ARTIFACTS_DIR",
        "/tmp/r12_9_smoke_artifacts"))
    output_dir = REPO_ROOT / "experiments" / "v2b_i3_15c" / "production_qualification"

    print("=" * 80)
    print("R12.9: PRODUCTION TRAJECTORY QUALIFICATION")
    print("=" * 80)
    print(f"  Smoke artifacts: {smoke_dir}")
    print(f"  Output: {output_dir}")

    # Load smoke results
    results = load_smoke_results(smoke_dir / "results.jsonl")
    model_calls = load_smoke_model_calls(smoke_dir / "model_calls.jsonl")
    print(f"\n  Loaded {len(results)} trajectory results")
    print(f"  Loaded {len(model_calls)} model call records")

    # R12.9A: False-T2 forensics
    false_t2_summary = run_false_t2_forensics(results, model_calls, output_dir)

    # R12.9B: VERIFY loop forensics
    verify_summary = run_verify_forensics(results, model_calls, output_dir)

    # Combined summary
    qualification = {
        "R12.9A_false_t2": false_t2_summary,
        "R12.9B_verify_loops": verify_summary,
    }
    with open(output_dir / "qualification.json", "w") as f:
        json.dump(qualification, f, indent=2, default=str)

    print(f"\n  Qualification report: {output_dir / 'qualification.json'}")
    print(f"\nR12.9 FORENSIC ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()
