#!/usr/bin/env python3
"""I3.13: Retrieved real-world evidence experiment.

Tests whether the metacognitive architecture operates over actually
retrieved evidence, measuring the full causal chain:

  RetrievalRecall -> RelationF1 -> EliminationAccuracy
  -> T2Precision/Recall -> RoutingAccuracy -> TaskUtility

Three retrieval conditions:
  R0_ORACLE: All required passages supplied directly
  R1_REAL: BM25 retriever supplies top-k passages
  R2_DISTRACTORS: BM25 + guaranteed relevant + distractors

Representations per condition (priority subset for cost control):
  A1_GOLD, A1_INFERRED, R1_GOLD, R1_INFERRED

Per-task telemetry tracks the complete failure attribution.

Usage:
    PYTHONPATH=. python3 -u scripts/run_i3_13_retrieved_evidence.py --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util

# Load I3.12j runner for the trajectory infrastructure
_spec = importlib.util.spec_from_file_location(
    "i3_12j", ROOT / "scripts" / "run_i3_12j_factorial.py")
i3_12j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_12j)

from hrm_adaptive_memory.executive.semantic_relations.i3_13_task_generator import (
    generate_i3_13_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_13_document_corpus import (
    get_corpus, corpus_sha256,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    SemanticTask,
)
from hrm_adaptive_memory.retrieval.lexical import BM25Retriever
from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis, EvidenceRuntime,
    initial_evidence_runtime,
)
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.resources import ResourceBudget, ResourceState
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility


# ---------------------------------------------------------------------------
# Retrieval conditions
# ---------------------------------------------------------------------------

def build_retrieval_index(corpus_passages):
    """Build a BM25 index over the document corpus."""
    chunks = [
        Chunk(
            chunk_id=p.passage_id,
            source_id=p.source,
            source_type="document",
            title=p.domain,
            section="",
            content=p.text,
            token_count=len(p.text.split()),
            metadata={"domain": p.domain, "source": p.source},
        )
        for p in corpus_passages
    ]
    return BM25Retriever(chunks)


def get_required_passage_ids(task, corpus_by_text):
    """Get the passage IDs that contain the evidence for this task."""
    required = set()
    for ev in task.evidence_task.evidence_items:
        pid = corpus_by_text.get(ev.proposition)
        if pid:
            required.add(pid)
    return required


def retrieve_oracle(task, corpus_by_text, corpus_by_id):
    """R0_ORACLE: return exactly the passages used in the task."""
    passages = []
    for ev in task.evidence_task.evidence_items:
        pid = corpus_by_text.get(ev.proposition)
        if pid and pid in corpus_by_id:
            passages.append(corpus_by_id[pid])
    return passages


def retrieve_real(task, retriever, corpus_by_id, k=5):
    """R1_REAL: BM25 retrieval over the corpus."""
    query = task.evidence_task.task_summary
    results = retriever.search(query, top_k=k)
    passages = []
    for chunk, score in results:
        p = corpus_by_id.get(chunk.chunk_id)
        if p:
            passages.append(p)
    return passages


def retrieve_distractors(task, retriever, corpus_by_text, corpus_by_id, k=5, n_distractors=3):
    """R2_DISTRACTORS: guaranteed relevant + BM25 distractors."""
    oracle_passages = retrieve_oracle(task, corpus_by_text, corpus_by_id)
    oracle_ids = {p["passage_id"] for p in oracle_passages}

    query = task.evidence_task.task_summary
    results = retriever.search(query, top_k=k + n_distractors)

    distractors = []
    for chunk, score in results:
        p = corpus_by_id.get(chunk.chunk_id)
        if p and p["passage_id"] not in oracle_ids:
            distractors.append(p)
        if len(distractors) >= n_distractors:
            break

    # If not enough from BM25, add random
    import random
    rng = random.Random(hash(task.task_id) % 2**32)
    all_non_oracle = [p for pid, p in corpus_by_id.items() if pid not in oracle_ids]
    while len(distractors) < n_distractors and all_non_oracle:
        p = rng.choice(all_non_oracle)
        if p not in distractors:
            distractors.append(p)
        all_non_oracle = [x for x in all_non_oracle if x is not p]

    return oracle_passages + distractors


def compute_retrieval_recall(retrieved_passages, required_ids):
    if not required_ids:
        return 1.0
    retrieved_ids = {p["passage_id"] for p in retrieved_passages}
    found = len(retrieved_ids & required_ids)
    return found / len(required_ids)


# ---------------------------------------------------------------------------
# Build evidence task from retrieved passages
# ---------------------------------------------------------------------------

def build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text):
    """Build an EvidenceTask with evidence from retrieved passages.

    The task's original evidence items are kept if their passages were retrieved.
    Additional retrieved passages (distractors) are added as neutral evidence.
    """
    et = task.evidence_task
    retrieved_texts = {p["text"] for p in retrieved_passages}

    # Keep original evidence items whose passages were retrieved
    evidence_items = []
    for ev in et.evidence_items:
        if ev.proposition in retrieved_texts or ev.retrieved:
            evidence_items.append(ev)

    # Add distractor passages as neutral evidence
    existing_texts = {ev.proposition for ev in evidence_items}
    for p in retrieved_passages:
        if p["text"] not in existing_texts:
            evidence_items.append(EvidenceItem(
                evidence_id=f"D{p['passage_id']}",
                proposition=p["text"],
                source_class="search",
                supports=(),
                contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True,
                verify_result="MISSING",
            ))

    if not evidence_items:
        return None

    return EvidenceTask(
        task_id=et.task_id,
        split=et.split,
        category=et.category,
        task_summary=et.task_summary,
        high_stakes=et.high_stakes,
        budget_profile=et.budget_profile,
        hypotheses=et.hypotheses,
        evidence_items=tuple(evidence_items),
        retrieve_exposes=(),
        search_exposes=(),
        oracle_resolution_path=et.oracle_resolution_path,
        expected_terminal=et.expected_terminal,
        correct_hypothesis_id=et.correct_hypothesis_id,
    )


# ---------------------------------------------------------------------------
# Relation accuracy computation
# ---------------------------------------------------------------------------

def compute_relation_accuracy(new_et, task, extractor):
    """Compute per-edge relation accuracy for the retrieved evidence task."""
    correct = True
    error_type = None
    n_edges = 0
    n_correct = 0

    for ev in [e for e in new_et.evidence_items if e.retrieved]:
        for hyp in new_et.hypotheses:
            result = extractor.extract(
                evidence_id=ev.evidence_id,
                evidence_proposition=ev.proposition,
                hypothesis_id=hyp.hypothesis_id,
                hypothesis_proposition=hyp.proposition,
            )
            inferred = result.relation.relation.value

            # Find gold relation
            gold_rel = "NEUTRAL"
            if not ev.evidence_id.startswith("D"):
                for gr in task.gold_relations:
                    if gr.evidence_id == ev.evidence_id and gr.hypothesis_id == hyp.hypothesis_id:
                        gold_rel = gr.relation
                        break

            n_edges += 1
            if inferred == gold_rel:
                n_correct += 1
            else:
                correct = False
                if error_type is None:
                    if gold_rel == "SUPPORT" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"
                    elif gold_rel == "SUPPORT" and inferred == "NEUTRAL":
                        error_type = "MISSED_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "NEUTRAL":
                        error_type = "MISSED_CONTRADICTION"
                    elif gold_rel == "NEUTRAL" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "NEUTRAL" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"

    return {
        "correct": correct,
        "error_type": error_type,
        "n_edges": n_edges,
        "n_correct": n_correct,
        "accuracy": n_correct / n_edges if n_edges > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# Run a single (task, condition, arm) trajectory
# ---------------------------------------------------------------------------

def run_single(args_tuple):
    """Run a single task-condition-arm combination.

    args_tuple is picklable for ProcessPoolExecutor.
    """
    (semantic_task_dict, retrieval_condition, arm,
     corpus_data, api_key, budget_dict) = args_tuple

    # Reconstruct objects
    extractor = DeterministicRelationExtractor()
    corpus_by_id = {p["passage_id"]: p for p in corpus_data}
    corpus_by_text = {p["text"]: p["passage_id"] for p in corpus_data}

    # Reconstruct SemanticTask
    from hrm_adaptive_memory.executive.semantic_relations.i3_13_task_generator import (
        generate_i3_13_corpus,
    )
    # We need the full SemanticTask - regenerate from seed
    # This is deterministic so it's safe
    all_tasks = generate_i3_13_corpus(n_per_category=25, seed=42)
    task = None
    for t in all_tasks:
        if t.task_id == semantic_task_dict["task_id"]:
            task = t
            break
    if task is None:
        raise ValueError(f"Task not found: {semantic_task_dict['task_id']}")

    # Rebuild retriever
    chunks = [
        Chunk(chunk_id=p["passage_id"], source_id=p["source"], source_type="document",
              title=p["domain"], section="", content=p["text"],
              token_count=len(p["text"].split()), metadata={})
        for p in corpus_data
    ]
    retriever = BM25Retriever(chunks)

    budget = ResourceBudget(**budget_dict)
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")

    # 1. Retrieve passages
    required_ids = get_required_passage_ids(task, corpus_by_text)
    if retrieval_condition == "R0_ORACLE":
        passages = retrieve_oracle(task, corpus_by_text, corpus_by_id)
    elif retrieval_condition == "R1_REAL":
        passages = retrieve_real(task, retriever, corpus_by_id, k=5)
    elif retrieval_condition == "R2_DISTRACTORS":
        passages = retrieve_distractors(task, retriever, corpus_by_text, corpus_by_id, k=5, n_distractors=3)
    else:
        raise ValueError(f"Unknown condition: {retrieval_condition}")

    retrieval_recall = compute_retrieval_recall(passages, required_ids)
    required_retrieved = retrieval_recall == 1.0

    # 2. Build evidence task from retrieved passages
    new_et = build_retrieved_evidence_task(task, passages, corpus_by_text)
    if new_et is None:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "retrieval_condition": retrieval_condition,
            "arm": arm,
            "retrieval_recall": 0.0,
            "required_evidence_retrieved": False,
            "n_evidence": 0,
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "NO_EVIDENCE",
            "t2_triggered": False,
            "t2_trigger_step": None,
            "relation_correct": False,
            "relation_error": None,
            "failure_attribution": "RETRIEVAL_ERROR",
        }

    # 3. Compute relation accuracy
    rel_info = compute_relation_accuracy(new_et, task, extractor)

    # 4. Run trajectory
    use_gold = "GOLD" in arm
    arch = arm.split("_")[0]

    if use_gold:
        sb = i3_12j.make_gold_snapshot_builder()
    else:
        sb = i3_12j.make_inferred_snapshot_builder(extractor)

    try:
        if arch == "R1":
            result = i3_12j.run_r1_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
            )
        else:
            mode = "BASELINE_WITH_AFFORDANCES" if arch == "A1" else "MDSG_STATE_WITH_AFFORDANCES"
            result = i3_12j.run_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
            )
    except Exception as e:
        return {
            "task_id": task.task_id,
            "category": task.category,
            "retrieval_condition": retrieval_condition,
            "arm": arm,
            "retrieval_recall": round(retrieval_recall, 4),
            "required_evidence_retrieved": required_retrieved,
            "n_evidence": len(new_et.evidence_items),
            "success": False,
            "utility": -150.0,
            "steps": 0,
            "terminal_action": "ERROR",
            "t2_triggered": False,
            "t2_trigger_step": None,
            "relation_correct": rel_info["correct"],
            "relation_error": rel_info["error_type"],
            "relation_accuracy": round(rel_info["accuracy"], 4),
            "failure_attribution": "POLICY_ERROR",
            "error": str(e)[:200],
        }

    success = result.get("success", False)
    utility_val = result.get("realized_utility", 0.0)
    steps = result.get("steps", 0)
    terminal = result.get("terminal_action", "UNKNOWN")
    t2_triggered = result.get("r1_triggered", False)

    # 5. Determine failure attribution
    if success:
        failure_attribution = "NO_ERROR"
    elif not required_retrieved:
        failure_attribution = "RETRIEVAL_ERROR"
    elif not rel_info["correct"]:
        failure_attribution = "SEMANTIC_EXTRACTION"
    else:
        failure_attribution = "POLICY_ERROR"

    return {
        "task_id": task.task_id,
        "category": task.category,
        "retrieval_condition": retrieval_condition,
        "arm": arm,
        "retrieval_recall": round(retrieval_recall, 4),
        "required_evidence_retrieved": required_retrieved,
        "n_evidence": len(new_et.evidence_items),
        "n_required": len(required_ids),
        "success": success,
        "utility": utility_val,
        "steps": steps,
        "terminal_action": terminal,
        "t2_triggered": t2_triggered,
        "t2_trigger_step": result.get("r1_trigger_step"),
        "relation_correct": rel_info["correct"],
        "relation_error": rel_info["error_type"],
        "relation_accuracy": round(rel_info["accuracy"], 4),
        "failure_attribution": failure_attribution,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-per-category", type=int, default=25)
    parser.add_argument("--retrieval-conditions", nargs="+",
                        default=["R0_ORACLE", "R1_REAL", "R2_DISTRACTORS"])
    parser.add_argument("--arms", nargs="+",
                        default=["A1_GOLD", "A1_INFERRED", "R1_GOLD", "R1_INFERRED"])
    parser.add_argument("--output-dir", type=str,
                        default="experiments/v2b_i3_13/development/i3_13_retrieved")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Generate corpus
    tasks = generate_i3_13_corpus(n_per_category=args.n_per_category, seed=42)
    corpus = get_corpus()
    corpus_data = [
        {"passage_id": p.passage_id, "text": p.text, "source": p.source,
         "domain": p.domain, "gold_relations": [list(r) for r in p.gold_relations]}
        for p in corpus
    ]

    extractor = DeterministicRelationExtractor()
    src = (ROOT / "hrm_adaptive_memory/executive/semantic_relations/deterministic_rules.py").read_bytes()
    extractor_sha = hashlib.sha256(src).hexdigest()

    n_tasks = len(tasks)
    n_conditions = len(args.retrieval_conditions)
    n_arms = len(args.arms)
    total = n_tasks * n_conditions * n_arms

    print(f"I3.13: Retrieved Real-World Evidence Experiment")
    print(f"  {n_tasks} tasks x {n_conditions} conditions x {n_arms} arms = {total} trajectories")
    print(f"  Retrieval conditions: {args.retrieval_conditions}")
    print(f"  Arms: {args.arms}")
    print(f"  Extractor: v2.6.0 (FROZEN), SHA256: {extractor_sha[:16]}...")
    print(f"  Corpus: {len(corpus)} passages, SHA256: {corpus_sha256()[:16]}...")
    print(f"  Primary: LCB_95(U_R1_INFERRED_REAL - U_A1_INFERRED_REAL) > 0")
    print()

    # Build work items
    budget_dict = {"max_executive_steps": 10, "max_retrieval_calls": 3,
                   "max_search_calls": 2, "max_verification_calls": 5}
    task_dicts = [{"task_id": t.task_id, "category": t.category} for t in tasks]

    work_items = []
    for td in task_dicts:
        for cond in args.retrieval_conditions:
            for arm in args.arms:
                work_items.append((td, cond, arm, corpus_data, api_key, budget_dict))

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v1.jsonl"
    log_path = out_dir / "run_log.txt"

    start = time.time()
    results = []
    completed = 0

    with open(log_path, "w") as logf, open(results_path, "w") as rf:
        logf.write(f"I3.13: Retrieved Real-World Evidence Experiment\n")
        logf.write(f"  {n_tasks} tasks x {n_conditions} conditions x {n_arms} arms = {total} trajectories\n")
        logf.write(f"  Extractor: v2.6.0 (FROZEN), SHA256: {extractor_sha[:16]}...\n")
        logf.write(f"  Corpus: {len(corpus)} passages, SHA256: {corpus_sha256()[:16]}...\n\n")
        logf.write(f"Processing {total} trajectories with {args.workers} workers...\n")
        logf.flush()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_single, wi): wi for wi in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    rf.write(json.dumps(result) + "\n")
                    rf.flush()
                except Exception as e:
                    logf.write(f"ERROR: {e}\n{traceback.format_exc()}\n")
                    logf.flush()
                completed += 1
                if completed % 50 == 0 or completed == total:
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    msg = f"  Completed {completed}/{total} ({rate:.1f}/s, ETA {eta:.0f}s)"
                    print(msg)
                    logf.write(msg + "\n")
                    logf.flush()

    elapsed = time.time() - start
    print(f"\nCompleted {total} trajectories in {elapsed:.1f}s")
    print(f"  Results: {results_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"I3.13 Results Summary ({n_tasks} tasks, {total} trajectories)")
    print(f"{'='*80}")

    by_ca = defaultdict(list)
    for r in results:
        by_ca[(r["retrieval_condition"], r["arm"])].append(r)

    print(f"\n{'Condition':<16} {'Arm':<16} {'N':>5} {'Recall':>8} {'Success':>8} {'MeanU':>8} {'RelAcc':>8} {'TopFail':>20}")
    for (cond, arm), rs in sorted(by_ca.items()):
        n = len(rs)
        recall = sum(r["retrieval_recall"] for r in rs) / n
        success = sum(r["success"] for r in rs) / n
        mean_u = sum(r["utility"] for r in rs) / n
        rel_acc = sum(r.get("relation_accuracy", 1.0) for r in rs) / n
        fa = Counter(r["failure_attribution"] for r in rs)
        top_fa = fa.most_common(1)[0][0] if fa else "N/A"
        print(f"  {cond:<16} {arm:<16} {n:>5} {recall:>8.4f} {success:>8.4f} {mean_u:>8.2f} {rel_acc:>8.4f} {top_fa:>20}")

    # Primary criterion
    r1_real = [r for r in results if r["retrieval_condition"] == "R1_REAL" and r["arm"] == "R1_INFERRED"]
    a1_real = [r for r in results if r["retrieval_condition"] == "R1_REAL" and r["arm"] == "A1_INFERRED"]
    if r1_real and a1_real:
        r_by_task = {r["task_id"]: r for r in r1_real}
        a_by_task = {r["task_id"]: r for r in a1_real}
        deltas = []
        for tid in r_by_task:
            if tid in a_by_task:
                deltas.append(r_by_task[tid]["utility"] - a_by_task[tid]["utility"])
        if deltas:
            import statistics
            mean_delta = sum(deltas) / len(deltas)
            if len(deltas) > 1:
                std = statistics.stdev(deltas)
                se = std / (len(deltas) ** 0.5)
                lcb = mean_delta - 1.96 * se
            else:
                lcb = mean_delta
            print(f"\nPrimary criterion (R1_REAL):")
            print(f"  LCB_95(U_R1_INF - U_A1_INF) = {lcb:.4f}")
            print(f"  Mean delta = {mean_delta:.4f}")
            print(f"  PASSES: {lcb > 0}")

    # Failure attribution
    print(f"\nFailure attribution (all results):")
    fa_counts = Counter(r["failure_attribution"] for r in results)
    for fa, count in fa_counts.most_common():
        print(f"  {fa}: {count}")

    # Gap decomposition
    print(f"\nGap decomposition (mean utility):")
    for cond in args.retrieval_conditions:
        cond_results = [r for r in results if r["retrieval_condition"] == cond]
        if not cond_results:
            continue
        by_arm = defaultdict(list)
        for r in cond_results:
            by_arm[r["arm"]].append(r)
        r1_gold_u = sum(r["utility"] for r in by_arm.get("R1_GOLD", [])) / len(by_arm.get("R1_GOLD", [1])) if by_arm.get("R1_GOLD") else 0
        r1_inf_u = sum(r["utility"] for r in by_arm.get("R1_INFERRED", [])) / len(by_arm.get("R1_INFERRED", [1])) if by_arm.get("R1_INFERRED") else 0
        a1_inf_u = sum(r["utility"] for r in by_arm.get("A1_INFERRED", [])) / len(by_arm.get("A1_INFERRED", [1])) if by_arm.get("A1_INFERRED") else 0
        a1_gold_u = sum(r["utility"] for r in by_arm.get("A1_GOLD", [])) / len(by_arm.get("A1_GOLD", [1])) if by_arm.get("A1_GOLD") else 0
        semantic_gap = r1_gold_u - r1_inf_u
        routing_gap = r1_inf_u - a1_inf_u
        print(f"  {cond}: R1_GOLD={r1_gold_u:.2f} R1_INF={r1_inf_u:.2f} A1_INF={a1_inf_u:.2f} A1_GOLD={a1_gold_u:.2f} SemGap={semantic_gap:.2f} RouteGap={routing_gap:.2f}")

    # Save analysis
    analysis = {
        "n_tasks": n_tasks,
        "n_trajectories": total,
        "elapsed_s": round(elapsed, 1),
        "extractor_sha256": extractor_sha,
        "corpus_sha256": corpus_sha256(),
        "retrieval_conditions": args.retrieval_conditions,
        "arms": args.arms,
        "failure_attribution": dict(fa_counts),
    }
    if r1_real and a1_real and deltas:
        analysis["primary_criterion"] = {
            "lcb_95": round(lcb, 4),
            "mean_delta": round(mean_delta, 4),
            "passes": lcb > 0,
        }
    analysis_path = out_dir / "analysis_v1.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
