#!/usr/bin/env python3
"""I3.15-r1-balanced: Retrieval-Fair Epistemically-Hard Benchmark (Balanced).

New benchmark identity — does NOT overwrite I3.15.

3x2 factorial: Q0_BM25 / Q3_RERANKED / Q4_ORACLE x A1_INFERRED / R1_INFERRED
= 150 tasks x 3 retrieval x 2 arms = 900 trajectories

Improvements over I3.15:
  1. Balanced outcomes (50/50 ANSWER/DEFER per cell)
  2. LocalLlamaBackend support (LFM2.5-2.6B Q5_K_M)
  3. Paired bootstrap CI for Delta_R(Q) and interaction
  4. Pre-specified hypothesis: Delta_R(Q3 | epistemic_hard) > 0
  5. Q3 vs Q4 comparability metrics (candidate count, context tokens, distractor load)
  6. Nuisance variable verification
  7. Frozen inference config recording

Usage:
    PYTHONPATH=. python3 -u scripts/run_i3_15_r1_balanced.py [--workers N] [--backend local|deepseek]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import random as rng_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
_env_file = ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Load I3.12j runner for the trajectory infrastructure
_spec_12j = importlib.util.spec_from_file_location(
    "i3_12j", str(ROOT / "scripts" / "run_i3_12j_factorial.py"))
i3_12j = importlib.util.module_from_spec(_spec_12j)
_spec_12j.loader.exec_module(i3_12j)

# Load I3.13 module for shared utilities
_spec_13 = importlib.util.spec_from_file_location(
    "run_i3_13_retrieved_evidence", str(ROOT / "scripts" / "run_i3_13_retrieved_evidence.py"))
i3_13 = importlib.util.module_from_spec(_spec_13)
_spec_13.loader.exec_module(i3_13)

from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
    get_corpus as get_i3_15_corpus, corpus_sha256 as i3_15_corpus_sha256,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_15_task_generator import (
    generate_i3_15_corpus,
)
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever
from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.cognitive_control.state import (
    VerificationState, TemporalStatus,
)
from hrm_adaptive_memory.executive.evidence_benchmark import EvidenceItem, EvidenceTask
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.resources import ResourceBudget
from hrm_adaptive_memory.executive.model_backend import (
    DeepSeekBackend, LocalLlamaBackend,
)

RETRIEVAL_LEVELS = ["Q0_BM25", "Q3_RERANKED", "Q4_ORACLE"]
ARMS = ["A1_INFERRED", "R1_INFERRED"]
TOP_K = 15
BENCHMARK_ID = "i3_15_r1_balanced"


# ---------------------------------------------------------------------------
# Frozen inference config
# ---------------------------------------------------------------------------

FROZEN_INFERENCE_CONFIG = {
    "benchmark_id": BENCHMARK_ID,
    "model_repository": "LiquidAI/LFM2.5-2.6B-GGUF",
    "quantization": "Q5_K_M",
    "model_name": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
    "llama_cpp_version": "b10217-ddd4ec142",
    "context_size": 131072,
    "temperature": 0.0,
    "max_tokens": 2048,
    "top_p": 1.0,
    "top_k": 40,
    "repeat_penalty": 1.0,
    "seed": 42,
    "threads": "auto",
    "gpu_layers": "all (Metal)",
    "base_url": "http://127.0.0.1:8080/v1",
    "determinism_verified": True,
    "determinism_unique_outputs": 1,
    "determinism_n_calls": 25,
}


def make_backend_factory(backend_type: str, api_key: str):
    """Return a factory callable that creates the appropriate backend."""
    if backend_type == "local":
        def factory():
            return LocalLlamaBackend()
        return factory
    elif backend_type == "deepseek":
        def factory():
            return DeepSeekBackend()
        return factory
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


# ---------------------------------------------------------------------------
# Retrieval and evidence construction
# ---------------------------------------------------------------------------

def build_corpus_index(corpus_passages):
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
    corpus_by_text = {p.text: p.passage_id for p in corpus_passages}
    corpus_by_id = {p.passage_id: p for p in corpus_passages}
    return chunks, corpus_by_text, corpus_by_id


def get_required_passage_ids(task, corpus_by_text):
    required = set()
    for ev in task.evidence_task.evidence_items:
        if not ev.retrieved:
            continue
        pid = corpus_by_text.get(ev.proposition)
        if pid:
            required.add(pid)
    return required


def build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text):
    et = task.evidence_task
    retrieved_texts = {p.text for p in retrieved_passages}
    evidence_items = [ev for ev in et.evidence_items if ev.proposition in retrieved_texts]
    existing_texts = {ev.proposition for ev in evidence_items}
    for p in retrieved_passages:
        if p.text not in existing_texts:
            evidence_items.append(EvidenceItem(
                evidence_id=f"D{p.passage_id}",
                proposition=p.text,
                source_class="search",
                supports=(), contradicts=(),
                verification_state=VerificationState.UNVERIFIED,
                temporal_status=TemporalStatus.CURRENT,
                retrieved=True, verify_result="MISSING",
            ))
    new_et = EvidenceTask(
        task_id=et.task_id, split=et.split, category=et.category,
        task_summary=et.task_summary,
        high_stakes=et.high_stakes, budget_profile=et.budget_profile,
        hypotheses=et.hypotheses,
        evidence_items=tuple(evidence_items),
        retrieve_exposes=et.retrieve_exposes,
        search_exposes=et.search_exposes,
        oracle_resolution_path=et.oracle_resolution_path,
        expected_terminal=et.expected_terminal,
        correct_hypothesis_id=et.correct_hypothesis_id,
    )
    return new_et


def compute_retrieval_recall(retrieved_passages, required_ids):
    if not required_ids:
        return 1.0
    retrieved_ids = {p.passage_id for p in retrieved_passages}
    return len(retrieved_ids & required_ids) / len(required_ids)


# ---------------------------------------------------------------------------
# Single trajectory runner (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def run_single(work_item):
    (task_id, retrieval_level, arm, backend_type, api_key, budget_dict) = work_item

    from hrm_adaptive_memory.executive.semantic_relations.i3_15_task_generator import (
        generate_i3_15_corpus,
    )
    from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
        get_corpus,
    )
    all_tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)
    task = next(t for t in all_tasks if t.evidence_task.task_id == task_id)

    corpus = get_corpus()
    chunks, corpus_by_text, corpus_by_id = build_corpus_index(corpus)
    required_ids = get_required_passage_ids(task, corpus_by_text)

    # Retrieval
    if retrieval_level == "Q4_ORACLE":
        retrieved_passages = [p for p in corpus if p.passage_id in required_ids]
        n_candidates = len(required_ids)
    else:
        retriever = build_retriever(retrieval_level, chunks)
        query = task.evidence_task.task_summary
        results = retriever.search(query, top_k=TOP_K)
        retrieved_passages = [corpus_by_id[chunk.chunk_id] for chunk, _ in results
                              if chunk.chunk_id in corpus_by_id]
        n_candidates = len(results)

    recall = compute_retrieval_recall(retrieved_passages, required_ids)
    recall_any = recall > 0 if required_ids else True
    recall_all = recall == 1.0 if required_ids else True

    # Context token count (for Q3 vs Q4 comparability)
    context_tokens = sum(len(p.text.split()) for p in retrieved_passages)
    n_distractors = len(retrieved_passages) - len(required_ids & {p.passage_id for p in retrieved_passages})

    new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)

    if not new_et.evidence_items:
        return {
            "task_id": task_id,
            "category": task.evidence_task.category,
            "expected_terminal": task.evidence_task.expected_terminal.value,
            "retrieval_level": retrieval_level,
            "arm": arm,
            "retrieval_recall": 0.0, "recall_any": False, "recall_all": False,
            "n_required": len(required_ids), "n_retrieved": 0,
            "n_candidates": n_candidates,
            "context_tokens": context_tokens,
            "n_distractors": n_distractors,
            "success": False, "utility": -150.0, "steps": 0,
            "terminal_action": "NO_EVIDENCE",
            "t2_triggered": False,
            "relation_correct": False, "relation_error": None,
            "relation_accuracy": 0.0,
            "failure_attribution": "RETRIEVAL_ERROR",
        }

    extractor = DeterministicRelationExtractor()
    rel_info = i3_13.compute_relation_accuracy(new_et, task, extractor)

    budget = ResourceBudget(**budget_dict)
    utility = MetareasoningUtility.from_file(ROOT / "configs/v2b_i3_1_utility_v1.json")

    use_gold = "GOLD" in arm
    arch = arm.split("_")[0]

    if use_gold:
        sb = i3_12j.make_gold_snapshot_builder()
    else:
        sb = i3_12j.make_inferred_snapshot_builder(extractor)

    backend_factory = make_backend_factory(backend_type, api_key)

    try:
        if arch == "R1":
            result = i3_12j.run_r1_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
                backend_factory=backend_factory,
            )
        else:
            mode = "BASELINE_WITH_AFFORDANCES" if arch == "A1" else "MDSG_STATE_WITH_AFFORDANCES"
            result = i3_12j.run_trajectory_i3_12(
                task=new_et, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm,
                snapshot_builder=sb,
                backend_factory=backend_factory,
            )
    except Exception as e:
        return {
            "task_id": task_id,
            "category": task.evidence_task.category,
            "expected_terminal": task.evidence_task.expected_terminal.value,
            "retrieval_level": retrieval_level,
            "arm": arm,
            "retrieval_recall": round(recall, 4),
            "recall_any": recall_any, "recall_all": recall_all,
            "n_required": len(required_ids),
            "n_retrieved": len(required_ids & {p.passage_id for p in retrieved_passages}),
            "n_candidates": n_candidates,
            "context_tokens": context_tokens,
            "n_distractors": n_distractors,
            "success": False, "utility": -150.0, "steps": 0,
            "terminal_action": "ERROR",
            "t2_triggered": False,
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

    if success:
        failure_attribution = "NO_ERROR"
    elif not recall_all:
        failure_attribution = "RETRIEVAL_ERROR"
    elif not rel_info["correct"]:
        failure_attribution = "SEMANTIC_EXTRACTION"
    else:
        failure_attribution = "POLICY_ERROR"

    return {
        "task_id": task_id,
        "category": task.evidence_task.category,
        "expected_terminal": task.evidence_task.expected_terminal.value,
        "retrieval_level": retrieval_level,
        "arm": arm,
        "retrieval_recall": round(recall, 4),
        "recall_any": recall_any, "recall_all": recall_all,
        "n_required": len(required_ids),
        "n_retrieved": len(required_ids & {p.passage_id for p in retrieved_passages}),
        "n_candidates": n_candidates,
        "context_tokens": context_tokens,
        "n_distractors": n_distractors,
        "success": success, "utility": utility_val, "steps": steps,
        "terminal_action": terminal,
        "t2_triggered": t2_triggered,
        "t2_trigger_step": result.get("r1_trigger_step"),
        "relation_correct": rel_info["correct"],
        "relation_error": rel_info["error_type"],
        "relation_accuracy": round(rel_info["accuracy"], 4),
        "failure_attribution": failure_attribution,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    r1_values: list[float],
    a1_values: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI for the mean difference R1 - A1.

    Both lists must be the same length and paired by task.
    """
    assert len(r1_values) == len(a1_values)
    n = len(r1_values)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    diffs = [r - a for r, a in zip(r1_values, a1_values)]
    point_estimate = sum(diffs) / n

    rng = rng_module.Random(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_diffs = [diffs[i] for i in indices]
        bootstrap_means.append(sum(sample_diffs) / n)

    bootstrap_means.sort()
    alpha = (1 - confidence) / 2
    ci_lower = bootstrap_means[int(n_bootstrap * alpha)]
    ci_upper = bootstrap_means[int(n_bootstrap * (1 - alpha))]

    return {
        "mean_diff": round(point_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "significant": ci_lower > 0 or ci_upper < 0,
    }


# ---------------------------------------------------------------------------
# Nuisance variable verification
# ---------------------------------------------------------------------------

def verify_nuisance_variables(tasks) -> dict:
    """Check that nuisance variables are balanced across cells."""
    from collections import Counter

    by_cell = defaultdict(list)
    for t in tasks:
        by_cell[t.evidence_task.category].append(t)

    checks = {}
    for cell, cell_tasks in sorted(by_cell.items()):
        terminals = Counter(t.evidence_task.expected_terminal.value for t in cell_tasks)
        n_hyps = [len(t.evidence_task.hypotheses) for t in cell_tasks]
        n_evidence = [len(t.evidence_task.evidence_items) for t in cell_tasks]
        query_lengths = [len(t.evidence_task.task_summary.split()) for t in cell_tasks]
        domains = Counter(t.evidence_task.task_summary for t in cell_tasks)

        # Gold relation counts
        support_count = 0
        contradict_count = 0
        neutral_count = 0
        for t in cell_tasks:
            for gr in t.gold_relations:
                if gr.relation == "SUPPORT":
                    support_count += 1
                elif gr.relation == "CONTRADICT":
                    contradict_count += 1
                else:
                    neutral_count += 1

        checks[cell] = {
            "n": len(cell_tasks),
            "expected_terminal": dict(terminals),
            "n_hypotheses_mean": round(sum(n_hyps) / len(n_hyps), 2),
            "n_evidence_mean": round(sum(n_evidence) / len(n_evidence), 2),
            "query_length_mean": round(sum(query_lengths) / len(query_lengths), 2),
            "gold_support_edges": support_count,
            "gold_contradict_edges": contradict_count,
            "gold_neutral_edges": neutral_count,
        }

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local", "deepseek"],
                        help="Model backend: local llama.cpp or DeepSeek API")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if args.backend == "deepseek" and not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set for deepseek backend")
        sys.exit(1)

    corpus = get_i3_15_corpus()
    tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)

    print(f"I3.15-r1-balanced: Retrieval-Fair Epistemically-Hard Benchmark")
    print(f"  Benchmark ID: {BENCHMARK_ID}")
    print(f"  Backend: {args.backend}")
    if args.backend == "local":
        print(f"  Model: {FROZEN_INFERENCE_CONFIG['model_name']}")
        print(f"  llama.cpp: {FROZEN_INFERENCE_CONFIG['llama_cpp_version']}")
    print(f"  Corpus: {len(corpus)} passages, SHA256: {i3_15_corpus_sha256()[:16]}...")
    print(f"  Tasks: {len(tasks)} (seed=42, 25 per cell, 6 cells)")
    print(f"  Retrieval levels: {RETRIEVAL_LEVELS}")
    print(f"  Arms: {ARMS}")
    print(f"  Top-k: {TOP_K}")
    n_traj = len(tasks) * len(RETRIEVAL_LEVELS) * len(ARMS)
    print(f"  Trajectories: {len(tasks)} x {len(RETRIEVAL_LEVELS)} x {len(ARMS)} = {n_traj}")
    print(f"  Workers: {args.workers}")
    print()

    # Nuisance variable verification
    print(f"{'='*80}")
    print("NUISANCE VARIABLE VERIFICATION")
    print(f"{'='*80}")
    nuisance = verify_nuisance_variables(tasks)
    print(f"{'Cell':<35} {'N':>4} {'ANSWER':>7} {'DEFER':>7} {'nHyp':>5} {'nEv':>5} {'qLen':>5} {'SUP':>5} {'CON':>5}")
    print("-" * 90)
    for cell, c in sorted(nuisance.items()):
        print(f"  {cell:<33} {c['n']:>4} {c['expected_terminal'].get('ANSWER', 0):>7} "
              f"{c['expected_terminal'].get('DEFER', 0):>7} {c['n_hypotheses_mean']:>5} "
              f"{c['n_evidence_mean']:>5} {c['query_length_mean']:>5} "
              f"{c['gold_support_edges']:>5} {c['gold_contradict_edges']:>5}")
    print()

    budget_dict = {"max_executive_steps": 10, "max_retrieval_calls": 3,
                   "max_search_calls": 2, "max_verification_calls": 5}

    work_items = []
    for task in tasks:
        task_id = task.evidence_task.task_id
        for retrieval_level in RETRIEVAL_LEVELS:
            for arm in ARMS:
                work_items.append((task_id, retrieval_level, arm,
                                   args.backend, api_key, budget_dict))

    print(f"  Total work items: {len(work_items)}")
    print()

    out_dir = ROOT / f"experiments/v2b_i3_15/development/{BENCHMARK_ID}_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_v1.jsonl"

    t0 = time.time()
    n_completed = 0

    with open(results_path, "w") as f_out:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single, wi): wi for wi in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush()
                    n_completed += 1
                    if n_completed % 50 == 0 or n_completed == len(work_items):
                        elapsed = time.time() - t0
                        rate = n_completed / elapsed
                        eta = (len(work_items) - n_completed) / rate if rate > 0 else 0
                        print(f"  Completed {n_completed}/{len(work_items)} "
                              f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
                except Exception as e:
                    n_completed += 1
                    print(f"  ERROR: {e}", flush=True)
                    f_out.write(json.dumps({"error": str(e)}) + "\n")
                    f_out.flush()

    elapsed = time.time() - t0
    print(f"\nCompleted {n_completed} trajectories in {elapsed:.1f}s")
    print(f"  Results: {results_path}")

    # Load results
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    results.append(r)

    # ====================================================================
    # Summary table
    # ====================================================================
    print(f"\n{'='*80}")
    print(f"I3.15-r1-balanced Results ({len(tasks)} tasks, {len(results)} trajectories)")
    print(f"{'='*80}")

    print(f"\n{'Retrieval':<14} {'Arm':<14} {'N':>4} {'Recall':>8} {'RecAll':>8} {'Success':>8} {'MeanU':>8} {'RelAcc':>8}")
    print("-" * 80)
    for retrieval in RETRIEVAL_LEVELS:
        for arm in ARMS:
            rs = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == arm]
            if not rs: continue
            recall = sum(r["retrieval_recall"] for r in rs) / len(rs)
            recall_all = sum(r["recall_all"] for r in rs) / len(rs)
            success = sum(r["success"] for r in rs) / len(rs)
            mean_u = sum(r["utility"] for r in rs) / len(rs)
            rel_acc = sum(r["relation_accuracy"] for r in rs) / len(rs)
            print(f"  {retrieval:<14} {arm:<14} {len(rs):>4} {recall:>8.4f} {recall_all:>8.4f} {success:>8.4f} {mean_u:>8.2f} {rel_acc:>8.4f}")

    # ====================================================================
    # Delta_R(Q) with paired bootstrap CI
    # ====================================================================
    print(f"\n{'='*80}")
    print("DELTA_R(Q) = U(R1,Q) - U(A1,Q) with paired bootstrap 95% CI")
    print(f"{'='*80}")

    deltas = {}
    cis = {}
    for retrieval in RETRIEVAL_LEVELS:
        # Pair by task_id
        r1_by_task = {r["task_id"]: r["utility"] for r in results
                      if r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"}
        a1_by_task = {r["task_id"]: r["utility"] for r in results
                      if r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"}
        common_ids = sorted(set(r1_by_task.keys()) & set(a1_by_task.keys()))
        if not common_ids: continue

        r1_vals = [r1_by_task[tid] for tid in common_ids]
        a1_vals = [a1_by_task[tid] for tid in common_ids]
        ci = paired_bootstrap_ci(r1_vals, a1_vals, n_bootstrap=10000, seed=42)
        deltas[retrieval] = ci["mean_diff"]
        cis[retrieval] = ci

        sig = "*" if ci["significant"] else ""
        print(f"  {retrieval}: Delta_R = {ci['mean_diff']:+.2f} "
              f"[{ci['ci_lower']:+.2f}, {ci['ci_upper']:+.2f}] {sig} "
              f"(n={ci['n']})")

    # ====================================================================
    # Interaction with CI
    # ====================================================================
    if "Q0_BM25" in cis and "Q3_RERANKED" in cis:
        # Bootstrap the interaction directly
        r1_q0 = {r["task_id"]: r["utility"] for r in results
                 if r["retrieval_level"] == "Q0_BM25" and r["arm"] == "R1_INFERRED"}
        a1_q0 = {r["task_id"]: r["utility"] for r in results
                 if r["retrieval_level"] == "Q0_BM25" and r["arm"] == "A1_INFERRED"}
        r1_q3 = {r["task_id"]: r["utility"] for r in results
                 if r["retrieval_level"] == "Q3_RERANKED" and r["arm"] == "R1_INFERRED"}
        a1_q3 = {r["task_id"]: r["utility"] for r in results
                 if r["retrieval_level"] == "Q3_RERANKED" and r["arm"] == "A1_INFERRED"}
        common = sorted(set(r1_q0) & set(a1_q0) & set(r1_q3) & set(a1_q3))
        if common:
            diffs_q0 = [r1_q0[t] - a1_q0[t] for t in common]
            diffs_q3 = [r1_q3[t] - a1_q3[t] for t in common]
            interaction_diffs = [d3 - d0 for d3, d0 in zip(diffs_q3, diffs_q0)]
            interaction_ci = paired_bootstrap_ci(
                interaction_diffs, [0]*len(interaction_diffs),
                n_bootstrap=10000, seed=42)
            print(f"\n  Interaction E_retrieval x E_MDSG = "
                  f"{interaction_ci['mean_diff']:+.2f} "
                  f"[{interaction_ci['ci_lower']:+.2f}, {interaction_ci['ci_upper']:+.2f}]"
                  f" {'*' if interaction_ci['significant'] else ''}")

    # ====================================================================
    # Pre-specified hypothesis: Delta_R(Q3 | epistemic_hard) > 0
    # ====================================================================
    print(f"\n{'='*80}")
    print("PRE-SPECIFIED HYPOTHESIS: Delta_R(Q3 | epistemic_hard) > 0")
    print(f"{'='*80}")

    hard_tasks = [t for t in tasks if "epistemic_hard" in t.evidence_task.category]
    hard_ids = {t.evidence_task.task_id for t in hard_tasks}

    for retrieval in RETRIEVAL_LEVELS:
        r1_hard = {r["task_id"]: r["utility"] for r in results
                   if r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"
                   and r["task_id"] in hard_ids}
        a1_hard = {r["task_id"]: r["utility"] for r in results
                   if r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"
                   and r["task_id"] in hard_ids}
        common = sorted(set(r1_hard) & set(a1_hard))
        if not common: continue
        r1_vals = [r1_hard[t] for t in common]
        a1_vals = [a1_hard[t] for t in common]
        ci = paired_bootstrap_ci(r1_vals, a1_vals, n_bootstrap=10000, seed=42)
        sig = "*" if ci["significant"] and ci["mean_diff"] > 0 else ""
        print(f"  {retrieval} | epistemic_hard: Delta_R = {ci['mean_diff']:+.2f} "
              f"[{ci['ci_lower']:+.2f}, {ci['ci_upper']:+.2f}] {sig} "
              f"(n={ci['n']})")

    # ====================================================================
    # Per-cell breakdown
    # ====================================================================
    print(f"\n{'='*80}")
    print("PER-CELL (2x3 matrix, R1 vs A1)")
    print(f"{'='*80}")
    print(f"{'Cell':<32} {'Retr':>5} {'N':>4} {'R1_S':>8} {'A1_S':>8} {'R1_U':>8} {'A1_U':>8} {'Delta':>8}")
    print("-" * 90)

    cells = sorted(set(r["category"] for r in results))
    for cell in cells:
        for retrieval in RETRIEVAL_LEVELS:
            r1 = [r for r in results if r["category"] == cell and r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"]
            a1 = [r for r in results if r["category"] == cell and r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"]
            if not r1 or not a1: continue
            r1_s = sum(r["success"] for r in r1) / len(r1)
            a1_s = sum(r["success"] for r in a1) / len(a1)
            r1_u = sum(r["utility"] for r in r1) / len(r1)
            a1_u = sum(r["utility"] for r in a1) / len(a1)
            delta = r1_u - a1_u
            print(f"  {cell[:30]:<31} {retrieval[:5]:>5} {len(r1):>4} {r1_s:>8.4f} {a1_s:>8.4f} {r1_u:>8.2f} {a1_u:>8.2f} {delta:>+8.2f}")

    # ====================================================================
    # Q3 vs Q4 comparability
    # ====================================================================
    print(f"\n{'='*80}")
    print("Q3 vs Q4 COMPARABILITY METRICS")
    print(f"{'='*80}")
    for retrieval in ["Q3_RERANKED", "Q4_ORACLE"]:
        rs = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == "A1_INFERRED"]
        if not rs: continue
        ctx = sum(r["context_tokens"] for r in rs) / len(rs)
        cands = sum(r["n_candidates"] for r in rs) / len(rs)
        distr = sum(r["n_distractors"] for r in rs) / len(rs)
        print(f"  {retrieval}: mean_context_tokens={ctx:.0f}, mean_candidates={cands:.1f}, mean_distractors={distr:.1f}")

    # ====================================================================
    # Conditional survival
    # ====================================================================
    print(f"\n{'='*80}")
    print("CONDITIONAL SURVIVAL (R1_INFERRED)")
    print(f"{'='*80}")
    for retrieval in RETRIEVAL_LEVELS:
        rs = [r for r in results if r["retrieval_level"] == retrieval and r["arm"] == "R1_INFERRED"]
        n = len(rs)
        if n == 0: continue
        p_succ = sum(r["success"] for r in rs) / n
        all_r = [r for r in rs if r["recall_all"]]
        all_and_rel = [r for r in all_r if r["relation_correct"]]
        print(f"  {retrieval}:")
        print(f"    P(success) = {p_succ:.4f} (n={n})")
        if all_r:
            print(f"    P(success | RecallAll=1) = {sum(r['success'] for r in all_r)/len(all_r):.4f} (n={len(all_r)})")
        if all_and_rel:
            print(f"    P(success | RecallAll=1, RelCorrect=1) = {sum(r['success'] for r in all_and_rel)/len(all_and_rel):.4f} (n={len(all_and_rel)})")

    # ====================================================================
    # Failure attribution
    # ====================================================================
    print(f"\n{'='*80}")
    print("FAILURE ATTRIBUTION")
    print(f"{'='*80}")
    fa = defaultdict(int)
    for r in results:
        fa[r["failure_attribution"]] += 1
    for k, v in sorted(fa.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # ====================================================================
    # Save analysis
    # ====================================================================
    analysis = {
        "benchmark_id": BENCHMARK_ID,
        "n_tasks": len(tasks),
        "n_trajectories": len(results),
        "corpus_sha256": i3_15_corpus_sha256(),
        "top_k": TOP_K,
        "frozen_inference_config": FROZEN_INFERENCE_CONFIG if args.backend == "local" else {"backend": "deepseek"},
        "deltas": deltas,
        "confidence_intervals": cis,
        "nuisance_variables": nuisance,
    }
    analysis_path = out_dir / "analysis_v1.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
