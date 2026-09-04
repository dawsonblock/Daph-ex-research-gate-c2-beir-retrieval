#!/usr/bin/env python3
"""R12 data collection: generate 12 candidates per task for 500+ reasoning tasks.

For each task:
  1. Generate 12 candidates at varied temperatures (0.0-1.2)
  2. For each candidate, collect self-evaluation confidence
  3. For each candidate, run 3-round verification
  4. Record: response, answer, correctness, confidence, verification, tokens, latency
  5. Preserve candidate order: a_1, ..., a_12

Output: experiments/daph_x/r12/r12_corpus_12.jsonl

Usage:
    python scripts/run_r12_data_collection.py \\
        --n_tasks 500 --n_candidates 12 \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf
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

# Import extract_answer from collection script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_reasoning_collection import extract_answer, extract_self_confidence
from run_cv_collection_v2 import extract_verification_score, VERIFICATION_PROMPTS


# Temperature schedule for 12 candidates
TEMP_SCHEDULE = [0.0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.2]


def count_tokens_approx(text: str) -> int:
    """Approximate token count (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def generate_candidate_with_metrics(
    model: CodingModelInterface, task: ReasoningTask,
    temperature: float, max_tokens: int, seed: int,
) -> dict:
    """Generate one reasoning candidate with full metrics."""
    prompt = f"{task.prompt}\n\nThink step by step. At the very end, write 'Answer: <your answer>' on a new line."

    t0 = time.monotonic()
    response = model.generate_raw(
        prompt=prompt, temperature=temperature,
        max_tokens=max_tokens, seed=seed,
    )
    gen_latency_ms = (time.monotonic() - t0) * 1000

    answer = extract_answer(response)
    is_correct = check_answer(answer, task.answer, task.answer_type)

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

    # 1-round verification (reduced from 3 for speed)
    verification_rounds = []
    for round_idx, vprompt_fn in enumerate(VERIFICATION_PROMPTS[:1]):
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
            "response": v_response[:200],  # Truncate for storage
            "latency_ms": v_latency_ms,
        })

    v_scores = [r["score"] for r in verification_rounds]
    avg_v_score = sum(v_scores) / len(v_scores) if v_scores else 0.5
    v_std = (sum((s - avg_v_score) ** 2 for s in v_scores) / len(v_scores)) ** 0.5 if len(v_scores) > 1 else 0.0
    v_consistent = all(abs(s - v_scores[0]) < 0.2 for s in v_scores) if v_scores else False

    # Features
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
        "answer_is_letter": 1 if re.match(r"^[a-d]$", answer.lower()) else 0,
        "temperature": temperature,
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

    return {
        "answer": answer,
        "is_correct": is_correct,
        "self_confidence": self_confidence,
        "response": response,
        "features": features,
        "temperature": temperature,
        "seed": seed,
        "verification": {
            "verification_score": avg_v_score,
            "verification_std": v_std,
            "verification_rounds": verification_rounds,
            "verification_consistent": v_consistent,
        },
        "metrics": {
            "gen_latency_ms": gen_latency_ms,
            "eval_latency_ms": eval_latency_ms,
            "total_latency_ms": gen_latency_ms + eval_latency_ms + sum(r["latency_ms"] for r in verification_rounds),
            "gen_tokens_approx": count_tokens_approx(response),
            "eval_tokens_approx": count_tokens_approx(eval_response),
            "total_tokens_approx": count_tokens_approx(response) + count_tokens_approx(eval_response) + sum(count_tokens_approx(r["response"]) for r in verification_rounds),
        },
    }


def collect_task_data(
    model: CodingModelInterface, task: ReasoningTask,
    n_candidates: int, max_tokens: int, seed: int = 42,
) -> dict:
    """Collect complete data for one reasoning task with n_candidates."""
    print(f"\n  Task: {task.task_id} ({task.difficulty}, {task.category})")
    print(f"  {task.description}")

    candidates = []
    t0 = time.monotonic()

    for i in range(n_candidates):
        temp = TEMP_SCHEDULE[i % len(TEMP_SCHEDULE)]
        cand_seed = seed + i * 137

        cand = generate_candidate_with_metrics(
            model=model, task=task,
            temperature=temp, max_tokens=max_tokens,
            seed=cand_seed,
        )
        cand["candidate_id"] = f"{task.task_id}_c{i}"
        cand["task_id"] = task.task_id
        candidates.append(cand)

        status = "OK" if cand["is_correct"] else "X "
        print(f"    c{i:2d} (T={temp:.1f}): {status} ans={cand['answer'][:20]:>20s} conf={cand['self_confidence']:3d} v={cand['verification']['verification_score']:.2f}")

    gen_time = time.monotonic() - t0

    # Cross-candidate statistics
    answers = [c["answer"] for c in candidates]
    answer_counts = {}
    for a in answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    agreement_rate = answer_counts[majority_answer] / len(answers)

    for c in candidates:
        c["features"]["agreement_rate"] = agreement_rate
        c["features"]["n_agreeing"] = answer_counts.get(c["answer"], 0)
        c["features"]["is_majority"] = 1 if c["answer"] == majority_answer else 0
        c["features"]["n_unique_answers"] = len(answer_counts)

    any_correct = any(c["is_correct"] for c in candidates)
    base = candidates[0]

    # Aggregate metrics
    total_tokens = sum(c["metrics"]["total_tokens_approx"] for c in candidates)
    total_latency = sum(c["metrics"]["total_latency_ms"] for c in candidates)

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
        "total_tokens_approx": total_tokens,
        "total_latency_ms": total_latency,
        "candidates": candidates,
    }

    print(f"  Base: {'OK' if base['is_correct'] else 'X '} | Any correct: {any_correct} | Agreement: {agreement_rate:.2f} | Time: {gen_time:.1f}s")
    return summary


def load_existing(path: Path) -> dict:
    """Load existing tasks from output file for resume."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                t = json.loads(line)
                existing[t["task_id"]] = t
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_candidates", type=int, default=12)
    parser.add_argument("--n_tasks", type=int, default=500)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    R12_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = get_all_reasoning_tasks()
    tasks = all_tasks[args.start_idx:args.start_idx + args.n_tasks]

    print(f"R12 Data Collection")
    print(f"  Tasks: {len(tasks)} (from {len(all_tasks)} available)")
    print(f"  Candidates per task: {args.n_candidates}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model: {args.model_path}")
    print()

    output_path = Path(args.output) if args.output else R12_DIR / "r12_corpus_12.jsonl"

    # Resume logic: skip tasks that already have enough candidates
    existing = load_existing(output_path)
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

        # Skip if already done
        if task_id in existing and len(existing[task_id]["candidates"]) >= args.n_candidates:
            continue

        print(f"[{idx+1}/{len(tasks)}] {task_id}")

        try:
            summary = collect_task_data(
                model=model, task=task,
                n_candidates=args.n_candidates,
                max_tokens=args.max_tokens,
                seed=args.seed + idx * 10000,
            )
            existing[task_id] = summary

            # Rewrite entire file to avoid duplicates
            with open(output_path, "w") as f:
                for tid in sorted(existing.keys()):
                    f.write(json.dumps(existing[tid], default=str) + "\n")
                f.flush()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    total = len(existing)
    base_correct = sum(1 for t in existing.values() if t.get("base_correct"))
    any_correct = sum(1 for t in existing.values() if t.get("any_correct"))
    rescue = sum(1 for t in existing.values() if t.get("rescue_available"))

    print(f"\n{'='*60}")
    print(f"  R12 DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {total}")
    print(f"  Base correct: {base_correct} ({base_correct/max(total,1)*100:.1f}%)")
    print(f"  Any candidate correct: {any_correct} ({any_correct/max(total,1)*100:.1f}%)")
    print(f"  Rescue available: {rescue} ({rescue/max(total,1)*100:.1f}%)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
