#!/usr/bin/env python3
"""R12 two-stage data collection.

Stage 1: Generate 12 raw candidates per task (immutable).
  - Only model generation calls
  - Records: response, answer, is_correct, temperature, seed, latency
  - Output: experiments/daph_x/r12/r12_raw_candidates.jsonl

Stage 2: Enrichment (deterministic, restartable).
  - Self-evaluation confidence
  - One-round verification (versioned as verification_v2)
  - Semantic features, pairwise features (computed later in evaluation)
  - Output: experiments/daph_x/r12/r12_enriched_corpus.jsonl

The two-stage design ensures:
  1. An enrichment failure doesn't waste generation work
  2. Verification can be rerun/ablated without regenerating candidates
  3. Raw candidates are immutable and auditable

Usage:
    # Stage 1: Raw generation
    python scripts/run_r12_collection.py --stage 1 --n_tasks 500 --n_candidates 12

    # Stage 2: Enrichment
    python scripts/run_r12_collection.py --stage 2

    # Both stages
    python scripts/run_r12_collection.py --stage both --n_tasks 500 --n_candidates 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.reasoning_tasks import get_all_reasoning_tasks, check_answer, ReasoningTask

R12_DIR = REPO_ROOT / "experiments/daph_x/r12"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_reasoning_collection import extract_answer, extract_self_confidence
from run_cv_collection_v2 import extract_verification_score, VERIFICATION_PROMPTS


# Temperature schedule for 12 candidates
TEMP_SCHEDULE = [0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.2]

# R12 verification protocol: 1 round, versioned separately from R9/R11
R12_VERIFICATION_ROUNDS = 1
R12_VERIFICATION_VERSION = "v2"


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


# ─── Stage 1: Raw Candidate Generation ───

def generate_raw_candidate(
    model: CodingModelInterface, task: ReasoningTask,
    temperature: float, max_tokens: int, seed: int,
) -> dict:
    """Generate one raw reasoning candidate (no enrichment)."""
    prompt = f"{task.prompt}\n\nThink step by step. At the very end, write 'Answer: <your answer>' on a new line."

    t0 = time.monotonic()
    response = model.generate_raw(
        prompt=prompt, temperature=temperature,
        max_tokens=max_tokens, seed=seed,
    )
    gen_latency_ms = (time.monotonic() - t0) * 1000

    answer = extract_answer(response)
    is_correct = check_answer(answer, task.answer, task.answer_type)

    return {
        "answer": answer,
        "is_correct": is_correct,
        "response": response,
        "temperature": temperature,
        "seed": seed,
        "gen_latency_ms": gen_latency_ms,
        "gen_tokens_approx": count_tokens_approx(response),
    }


def collect_raw_task(
    model: CodingModelInterface, task: ReasoningTask,
    n_candidates: int, max_tokens: int, seed: int = 42,
) -> dict:
    """Stage 1: Collect raw candidates for one task."""
    print(f"\n  Task: {task.task_id} ({task.difficulty}, {task.category})")
    print(f"  {task.description}")

    candidates = []
    t0 = time.monotonic()

    for i in range(n_candidates):
        temp = TEMP_SCHEDULE[i % len(TEMP_SCHEDULE)]
        cand_seed = seed + i * 137

        cand = generate_raw_candidate(
            model=model, task=task,
            temperature=temp, max_tokens=max_tokens,
            seed=cand_seed,
        )
        cand["candidate_id"] = f"{task.task_id}_c{i}"
        cand["task_id"] = task.task_id
        candidates.append(cand)

        status = "OK" if cand["is_correct"] else "X "
        print(f"    c{i:2d} (T={temp:.1f}): {status} ans={cand['answer'][:20]:>20s}")

    gen_time = time.monotonic() - t0

    # Cross-candidate statistics (computed from raw data only)
    answers = [c["answer"] for c in candidates]
    answer_counts = {}
    for a in answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    agreement_rate = answer_counts[majority_answer] / len(answers)

    any_correct = any(c["is_correct"] for c in candidates)
    base = candidates[0]

    summary = {
        "task_id": task.task_id,
        "description": task.description,
        "difficulty": task.difficulty,
        "category": task.category,
        "correct_answer": task.answer,
        "answer_type": task.answer_type,
        "n_candidates": len(candidates),
        "base_correct": base["is_correct"],
        "any_correct": any_correct,
        "rescue_available": any_correct and not base["is_correct"],
        "agreement_rate": agreement_rate,
        "n_unique_answers": len(answer_counts),
        "majority_answer": majority_answer,
        "majority_correct": check_answer(majority_answer, task.answer, task.answer_type),
        "gen_time_s": gen_time,
        "candidates": candidates,
        "stage": "raw",
        "evaluator_version": "r12_v1",
    }

    print(f"  Base: {'OK' if base['is_correct'] else 'X '} | Any correct: {any_correct} | Agreement: {agreement_rate:.2f} | Time: {gen_time:.1f}s")
    return summary


# ─── Stage 2: Enrichment ───

def enrich_candidate(
    model: CodingModelInterface, task: ReasoningTask, cand: dict,
) -> dict:
    """Stage 2: Enrich a single candidate with self-eval and verification."""
    answer = cand["answer"]
    seed = cand["seed"]

    # Self-evaluation
    eval_prompt = (
        f"Question: {task.prompt}\n\n"
        f"Your answer: {answer}\n\n"
        f"On a scale of 0-100, how confident are you that this answer is correct? "
        f"Respond with just a number."
    )
    t0 = time.monotonic()
    eval_response = model.generate_raw(
        prompt=eval_prompt, temperature=0.0,
        max_tokens=10, seed=seed + 1000,
    )
    eval_latency_ms = (time.monotonic() - t0) * 1000
    self_confidence = extract_self_confidence(eval_response)

    # One-round verification (versioned as v2)
    verification_rounds = []
    for round_idx in range(R12_VERIFICATION_ROUNDS):
        vprompt_fn = VERIFICATION_PROMPTS[round_idx]
        v_prompt = vprompt_fn(task.prompt, answer)
        t0 = time.monotonic()
        v_response = model.generate_raw(
            prompt=v_prompt, temperature=0.0,
            max_tokens=100, seed=seed + 2000 + round_idx * 100,
        )
        v_latency_ms = (time.monotonic() - t0) * 1000
        v_score, _ = extract_verification_score(v_response)
        verification_rounds.append({
            "round": round_idx,
            "score": v_score,
            "response": v_response[:200],
            "latency_ms": v_latency_ms,
        })

    v_scores = [r["score"] for r in verification_rounds]
    avg_v_score = sum(v_scores) / len(v_scores) if v_scores else 0.5
    v_std = 0.0  # Single round, no std
    v_consistent = True  # Single round, trivially consistent

    # Features (text-based, no model calls)
    response = cand["response"]
    features = {
        "response_length": len(response),
        "response_lines": response.count("\n") + 1,
        "answer_length": len(answer),
        "self_confidence": self_confidence,
        "has_step_by_step": 1 if "step" in response.lower() else 0,
        "has_therefore": 1 if "therefore" in response.lower() else 0,
        "has_thus": 1 if "thus" in response.lower() else 0,
        "has_hence": 1 if "hence" in response.lower() else 0,
        "has_so": 1 if "\nso " in response.lower() or " so " in response.lower() else 0,
        "has_calculat": 1 if "calculat" in response.lower() else 0,
        "has_formula": 1 if "formula" in response.lower() else 0,
        "has_equation": 1 if "equation" in response.lower() else 0,
        "has_let_x": 1 if "let x" in response.lower() else 0,
        "has_substitut": 1 if "substitut" in response.lower() else 0,
        "has_simplify": 1 if "simplif" in response.lower() else 0,
        "has_verify": 1 if "verif" in response.lower() else 0,
        "has_check": 1 if "check" in response.lower() else 0,
        "n_numbers": len(re.findall(r"\d+\.?\d*", response)),
        "n_equals_signs": response.count("="),
        "n_plus_signs": response.count("+"),
        "n_minus_signs": response.count("-"),
        "n_multiply": response.count("*") + response.count("×"),
        "n_divide": response.count("/") + response.count("÷"),
        "n_parentheses": response.count("("),
        "n_decimal_points": response.count("."),
        "n_commas": response.count(","),
        "has_fraction": 1 if "/" in answer else 0,
        "answer_is_number": 1 if re.match(r"^[+-]?\d+\.?\d*$", answer) else 0,
        "answer_is_yes_no": 1 if answer.lower() in ("yes", "no") else 0,
        "temperature": cand["temperature"],
        "n_words": len(response.split()),
        "n_sentences": response.count(".") + response.count("?") + response.count("!"),
        "avg_word_length": sum(len(w) for w in response.split()) / max(len(response.split()), 1),
        "has_first_person": 1 if any(w in response.lower() for w in ["i think", "i believe", "i'll", "i will"]) else 0,
        "has_uncertainty": 1 if any(w in response.lower() for w in ["maybe", "might", "could be", "not sure", "uncertain"]) else 0,
        "has_definite": 1 if any(w in response.lower() for w in ["definitely", "certainly", "absolutely", "must be"]) else 0,
        "n_steps_indicated": len(re.findall(r"step \d+", response.lower())),
        "has_numbered_list": 1 if re.search(r"\n\d+\.", response) else 0,
        "has_bullets": 1 if "•" in response or "- " in response else 0,
        "response_starts_with_number": 1 if response.strip() and response.strip()[0].isdigit() else 0,
        "answer_in_last_line": 1 if answer.strip().lower() in response.strip().split("\n")[-1].lower() else 0,
        "n_digits_in_answer": sum(c.isdigit() for c in answer),
    }

    # Add verification_v2 fields (versioned separately from R9/R11)
    cand_enriched = dict(cand)  # Copy raw fields
    cand_enriched["self_confidence"] = self_confidence
    cand_enriched["features"] = features
    cand_enriched["verification_v2"] = {
        "verification_score": avg_v_score,
        "verification_std": v_std,
        "verification_rounds": verification_rounds,
        "verification_consistent": v_consistent,
        "version": R12_VERIFICATION_VERSION,
        "n_rounds": R12_VERIFICATION_ROUNDS,
    }
    # Also keep a top-level verification key for compatibility,
    # but mark it as v2
    cand_enriched["verification"] = cand_enriched["verification_v2"]
    cand_enriched["metrics"] = {
        "gen_latency_ms": cand.get("gen_latency_ms", 0),
        "eval_latency_ms": eval_latency_ms,
        "total_latency_ms": cand.get("gen_latency_ms", 0) + eval_latency_ms + sum(r["latency_ms"] for r in verification_rounds),
        "gen_tokens_approx": cand.get("gen_tokens_approx", 0),
        "eval_tokens_approx": count_tokens_approx(eval_response),
        "total_tokens_approx": cand.get("gen_tokens_approx", 0) + count_tokens_approx(eval_response) + sum(count_tokens_approx(r["response"]) for r in verification_rounds),
    }

    return cand_enriched


def enrich_task(model: CodingModelInterface, task: ReasoningTask, raw_task: dict) -> dict:
    """Stage 2: Enrich all candidates for one task."""
    print(f"\n  Enriching: {task.task_id} ({task.difficulty}, {task.category})")

    t0 = time.monotonic()
    enriched_candidates = []
    for i, cand in enumerate(raw_task["candidates"]):
        ec = enrich_candidate(model, task, cand)
        enriched_candidates.append(ec)
        status = "OK" if ec["is_correct"] else "X "
        print(f"    c{i:2d}: {status} conf={ec['self_confidence']:3d} v={ec['verification_v2']['verification_score']:.2f}")

    enrich_time = time.monotonic() - t0

    # Add cross-candidate features
    answers = [c["answer"] for c in enriched_candidates]
    answer_counts = {}
    for a in answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    agreement_rate = answer_counts[majority_answer] / len(answers)

    for c in enriched_candidates:
        c["features"]["agreement_rate"] = agreement_rate
        c["features"]["n_agreeing"] = answer_counts.get(c["answer"], 0)
        c["features"]["is_majority"] = 1 if c["answer"] == majority_answer else 0
        c["features"]["n_unique_answers"] = len(answer_counts)

    enriched_task = dict(raw_task)  # Copy task-level fields
    enriched_task["candidates"] = enriched_candidates
    enriched_task["stage"] = "enriched"
    enriched_task["enrich_time_s"] = enrich_time
    enriched_task["verification_version"] = R12_VERIFICATION_VERSION
    enriched_task["verification_rounds"] = R12_VERIFICATION_ROUNDS
    enriched_task["evaluator_version"] = "r12_v1"

    print(f"  Enriched {len(enriched_candidates)} candidates in {enrich_time:.1f}s")
    return enriched_task


# ─── Common utilities ───

def load_existing(path: Path) -> dict:
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                t = json.loads(line)
                existing[t["task_id"]] = t
    return existing


def save_tasks(path: Path, tasks: dict):
    with open(path, "w") as f:
        for tid in sorted(tasks.keys()):
            f.write(json.dumps(tasks[tid], default=str) + "\n")
        f.flush()


# ─── Stage 1 Main ───

def run_stage1(args):
    """Stage 1: Raw candidate generation."""
    raw_path = R12_DIR / "r12_raw_candidates.jsonl"
    R12_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = get_all_reasoning_tasks()
    tasks = all_tasks[args.start_idx:args.start_idx + args.n_tasks]

    print(f"R12 Stage 1: Raw Candidate Generation")
    print(f"  Tasks: {len(tasks)} (from {len(all_tasks)} available)")
    print(f"  Candidates per task: {args.n_candidates}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output: {raw_path}")
    print()

    existing = load_existing(raw_path)
    n_done = sum(1 for t in existing.values() if len(t["candidates"]) >= args.n_candidates)
    n_to_do = sum(1 for task in tasks if not (task.task_id in existing and len(existing[task.task_id]["candidates"]) >= args.n_candidates))
    print(f"  Resuming: {n_done} tasks already complete, {n_to_do} to do")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    for idx, task in enumerate(tasks):
        task_id = task.task_id
        if task_id in existing and len(existing[task_id]["candidates"]) >= args.n_candidates:
            continue

        print(f"[{idx+1}/{len(tasks)}] {task_id}")

        try:
            summary = collect_raw_task(
                model=model, task=task,
                n_candidates=args.n_candidates,
                max_tokens=args.max_tokens,
                seed=args.seed + idx * 10000,
            )
            existing[task_id] = summary
            save_tasks(raw_path, existing)

            # Interim diagnostic at checkpoints
            n_complete = sum(1 for t in existing.values() if len(t["candidates"]) >= args.n_candidates)
            if n_complete in (100, 200, 350, 500):
                print(f"\n{'='*60}")
                print(f"  INTERIM DIAGNOSTIC: N={n_complete}")
                print(f"{'='*60}")
                print_interim_diagnostics(existing)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print_final_summary(existing)


# ─── Stage 2 Main ───

def run_stage2(args):
    """Stage 2: Enrichment."""
    raw_path = R12_DIR / "r12_raw_candidates.jsonl"
    enriched_path = R12_DIR / "r12_enriched_corpus.jsonl"

    if not raw_path.exists():
        print(f"ERROR: Raw candidates not found at {raw_path}")
        print("  Run stage 1 first: python scripts/run_r12_collection.py --stage 1")
        return

    raw_tasks = load_existing(raw_path)
    print(f"R12 Stage 2: Enrichment")
    print(f"  Loaded {len(raw_tasks)} raw tasks")
    print(f"  Verification: {R12_VERIFICATION_ROUNDS} round(s), version={R12_VERIFICATION_VERSION}")
    print(f"  Output: {enriched_path}")
    print()

    # Load already-enriched tasks for resume
    enriched_existing = load_existing(enriched_path) if enriched_path.exists() else {}

    all_tasks = get_all_reasoning_tasks()
    task_lookup = {t.task_id: t for t in all_tasks}

    n_to_enrich = sum(1 for tid, rt in raw_tasks.items()
                      if len(rt["candidates"]) >= 12
                      and tid not in enriched_existing)
    n_already = len(enriched_existing)
    print(f"  Resuming: {n_already} tasks already enriched, {n_to_enrich} to do")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    for idx, (task_id, raw_task) in enumerate(sorted(raw_tasks.items())):
        if task_id in enriched_existing:
            continue
        if len(raw_task["candidates"]) < 12:
            continue
        if task_id not in task_lookup:
            print(f"  WARNING: {task_id} not found in task registry, skipping")
            continue

        task = task_lookup[task_id]
        print(f"[{idx+1}/{len(raw_tasks)}] {task_id}")

        try:
            enriched = enrich_task(model, task, raw_task)
            enriched_existing[task_id] = enriched
            save_tasks(enriched_path, enriched_existing)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Enrichment complete: {len(enriched_existing)} tasks")
    print(f"  Output: {enriched_path}")


# ─── Diagnostics ───

def print_interim_diagnostics(existing: dict):
    """Print descriptive diagnostics at interim checkpoints.

    IMPORTANT: Only descriptive statistics, no threshold tuning or model changes.
    The same examples may later enter confirmation.
    """
    tasks = list(existing.values())
    n = len(tasks)
    if n == 0:
        return

    base_correct = sum(1 for t in tasks if t.get("base_correct"))
    any_correct = sum(1 for t in tasks if t.get("any_correct"))
    rescue = sum(1 for t in tasks if t.get("rescue_available"))

    # Category balance
    from collections import Counter
    cat_counts = Counter(t.get("category", "unknown") for t in tasks)
    diff_counts = Counter(t.get("difficulty", "unknown") for t in tasks)

    # Oracle@K (using raw correctness, no MaxCal)
    for k in [2, 4, 6, 8, 12]:
        oracle_k = sum(1 for t in tasks
                       if any(c["is_correct"] for c in t["candidates"][:k])
                       if len(t["candidates"]) >= k)
        print(f"  Oracle@{k:2d}: {oracle_k}/{n} = {oracle_k/n:.1%}")

    print(f"  Base correct: {base_correct}/{n} = {base_correct/n:.1%}")
    print(f"  Any correct:   {any_correct}/{n} = {any_correct/n:.1%}")
    print(f"  Rescue avail:  {rescue}/{n} = {rescue/n:.1%}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  Difficulty:  {dict(diff_counts)}")


def print_final_summary(existing: dict):
    total = len(existing)
    if total == 0:
        print("  No tasks collected.")
        return

    base_correct = sum(1 for t in existing.values() if t.get("base_correct"))
    any_correct = sum(1 for t in existing.values() if t.get("any_correct"))
    rescue = sum(1 for t in existing.values() if t.get("rescue_available"))

    print(f"\n{'='*60}")
    print(f"  STAGE 1 SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {total}")
    print(f"  Base correct: {base_correct} ({base_correct/total*100:.1f}%)")
    print(f"  Any candidate correct: {any_correct} ({any_correct/total*100:.1f}%)")
    print(f"  Rescue available: {rescue} ({rescue/total*100:.1f}%)")
    print_interim_diagnostics(existing)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both")
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_candidates", type=int, default=12)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage in ("1", "both"):
        run_stage1(args)
    if args.stage in ("2", "both"):
        run_stage2(args)


if __name__ == "__main__":
    main()
