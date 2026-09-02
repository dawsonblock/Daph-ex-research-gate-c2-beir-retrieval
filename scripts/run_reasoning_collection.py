#!/usr/bin/env python3
"""Collect reasoning task data with self-evaluation features.

For each task:
  1. Generate N candidates at different temperatures
  2. For each candidate, ask the model to self-evaluate:
     "How confident are you in this answer? (0-100)"
  3. Record: response, self-confidence, reasoning length, answer
  4. Check correctness against ground truth

No probe tests — this is the no-execution-feedback regime.

Usage:
    python scripts/run_reasoning_collection.py \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        --n_candidates 4 \\
        --max_tokens 300
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.reasoning_tasks import (
    get_all_reasoning_tasks, check_answer, ReasoningTask
)

REASONING_DIR = REPO_ROOT / "experiments/daph_x/reasoning"


def extract_answer(response: str) -> str:
    """Extract the final answer from a reasoning response."""
    # Strategy: look for the answer in the LAST occurrence of common patterns
    # This avoids picking up intermediate calculations

    # Try "Therefore, ... X" or "The answer is X" or "Answer: X"
    patterns = [
        r"(?:therefore|thus|hence|so the answer is|the answer is|answer:|final answer:)\s*[^0-9\-]*([+-]?\d+\.?\d*[/\d]*)",
        r"(?:therefore|thus|hence|so the answer is|the answer is|answer:|final answer:)\s*[:\s]*([^\n,]+)",
        r"\\boxed\{([^}]+)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip().rstrip(".")

    # Fallback: look at the last few lines for a number
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    # Search from the end backwards
    for line in reversed(lines):
        # Look for "X miles" or "X hours" or just "X" at the end
        num_match = re.search(r"=\s*([+-]?\d+\.?\d*[/\d]*)", line)
        if num_match:
            return num_match.group(1).strip()
        # Look for a standalone number
        num_match = re.search(r"\b([+-]?\d+\.?\d*[/\d]*)\b\s*(?:miles?|hours?|dollars?|cents?|percent|%|degrees?|marbles?|ways?|coins?|faces?|primes?|terms?|elements?|people|students|books?|items?|candies?|children|positions?|rows?|crossings?|solutions?|letters?|digits?|numbers?|years?|minutes?|seconds?|feet?|inches?|meters?|grams?|pounds?|ounces?|liters?|gallons?)?\s*$", line, re.IGNORECASE)
        if num_match:
            return num_match.group(1).strip()

    # Last resort: last number in the response
    all_nums = re.findall(r"[+-]?\d+\.?\d*[/\d]*", response)
    if all_nums:
        return all_nums[-1].strip()

    return response.strip()


def extract_self_confidence(response: str) -> int:
    """Extract self-confidence score from self-evaluation response."""
    # Look for a number 0-100
    match = re.search(r"\b(\d{1,3})\b", response)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score
    return 50  # Default if can't parse


def generate_reasoning_candidate(
    model: CodingModelInterface, task: ReasoningTask,
    temperature: float, max_tokens: int, seed: int,
) -> dict:
    """Generate one reasoning candidate with self-evaluation."""
    # Generate the answer
    prompt = f"{task.prompt}\n\nThink step by step. At the very end, write 'Answer: <your answer>' on a new line."
    response = model.generate_raw(
        prompt=prompt, temperature=temperature,
        max_tokens=max_tokens, seed=seed,
    )

    # Extract answer
    answer = extract_answer(response)
    is_correct = check_answer(answer, task.answer, task.answer_type)

    # Self-evaluation: ask the model to rate its confidence
    eval_prompt = (
        f"Question: {task.prompt}\n\n"
        f"Your answer: {answer}\n\n"
        f"On a scale of 0-100, how confident are you that this answer is correct? "
        f"Respond with just a number."
    )
    eval_response = model.generate_raw(
        prompt=eval_prompt, temperature=0.0,
        max_tokens=20, seed=seed + 1000,
    )
    self_confidence = extract_self_confidence(eval_response)

    # Compute features
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
    }


def collect_task_data(
    model: CodingModelInterface, task: ReasoningTask,
    n_candidates: int, max_tokens: int,
) -> dict:
    """Collect complete data for one reasoning task."""
    print(f"\n{'='*60}")
    print(f"  Task: {task.task_id} ({task.difficulty}, {task.category})")
    print(f"  {task.description}")
    print(f"  Correct answer: {task.answer}")
    print(f"{'='*60}")

    # Temperature schedule
    temps = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 0.2, 0.6][:n_candidates]

    candidates = []
    t0 = time.monotonic()
    for i, temp in enumerate(temps):
        cand = generate_reasoning_candidate(
            model=model, task=task,
            temperature=temp, max_tokens=max_tokens,
            seed=42 + i,
        )
        cand["candidate_id"] = f"{task.task_id}_c{i}"
        cand["task_id"] = task.task_id
        candidates.append(cand)

        status = "CORRECT" if cand["is_correct"] else "WRONG"
        print(f"    c{i} (T={temp}): {status} answer={cand['answer']} conf={cand['self_confidence']}")

    gen_time = time.monotonic() - t0

    # Base = first candidate (temp=0)
    base = candidates[0]

    # Compute cross-candidate consistency
    answers = [c["answer"] for c in candidates]
    answer_counts = {}
    for a in answers:
        answer_counts[a] = answer_counts.get(a, 0) + 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    agreement_rate = answer_counts[majority_answer] / len(answers)

    # Add cross-candidate features
    for c in candidates:
        c["features"]["agreement_rate"] = agreement_rate
        c["features"]["n_agreeing"] = answer_counts.get(c["answer"], 0)
        c["features"]["is_majority"] = 1 if c["answer"] == majority_answer else 0
        c["features"]["n_unique_answers"] = len(answer_counts)

    # Best candidate (correct if any)
    any_correct = any(c["is_correct"] for c in candidates)
    best = max(candidates, key=lambda c: (c["is_correct"], c["self_confidence"]))

    # Utility: 100 if correct, 0 if wrong (binary)
    base_utility = 100.0 if base["is_correct"] else 0.0
    best_utility = 100.0 if best["is_correct"] else 0.0
    rescue_available = best["is_correct"] and not base["is_correct"]

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
        "rescue_available": rescue_available,
        "agreement_rate": agreement_rate,
        "n_unique_answers": len(answer_counts),
        "majority_answer": majority_answer,
        "majority_correct": check_answer(majority_answer, task.answer, task.answer_type),
        "gen_time_s": gen_time,
        "candidates": candidates,
    }

    print(f"\n  Base: {'CORRECT' if base['is_correct'] else 'WRONG'} (conf={base['self_confidence']})")
    print(f"  Best: {'CORRECT' if best['is_correct'] else 'WRONG'} (conf={best['self_confidence']})")
    print(f"  Rescue available: {rescue_available}")
    print(f"  Agreement rate: {agreement_rate:.2f} ({len(answer_counts)} unique answers)")
    print(f"  Majority answer: {majority_answer} ({'correct' if summary['majority_correct'] else 'wrong'})")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_candidates", type=int, default=4)
    parser.add_argument("--n_tasks", type=int, default=60)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    REASONING_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = get_all_reasoning_tasks()
    tasks = all_tasks[args.start_idx:args.start_idx + args.n_tasks]

    print(f"Reasoning Data Collection")
    print(f"  Tasks: {len(tasks)} (from {len(all_tasks)} available)")
    print(f"  Candidates per task: {args.n_candidates}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model: {args.model_path}")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else REASONING_DIR / "reasoning_corpus.jsonl"

    all_summaries = []
    with open(output_path, "w") as f:
        for idx, task in enumerate(tasks):
            print(f"\n[Task {idx+1}/{len(tasks)}]")
            try:
                summary = collect_task_data(
                    model=model, task=task,
                    n_candidates=args.n_candidates,
                    max_tokens=args.max_tokens,
                )
                all_summaries.append(summary)
                f.write(json.dumps(summary, default=str) + "\n")
                f.flush()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    total = len(all_summaries)
    base_correct = sum(1 for s in all_summaries if s["base_correct"])
    any_correct = sum(1 for s in all_summaries if s["any_correct"])
    rescue = sum(1 for s in all_summaries if s["rescue_available"])
    majority_correct = sum(1 for s in all_summaries if s["majority_correct"])

    print(f"\n{'='*60}")
    print(f"  REASONING DATA COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks: {total}")
    print(f"  Base correct: {base_correct} ({base_correct/max(total,1)*100:.0f}%)")
    print(f"  Any candidate correct: {any_correct} ({any_correct/max(total,1)*100:.0f}%)")
    print(f"  Rescue available: {rescue} ({rescue/max(total,1)*100:.0f}%)")
    print(f"  Majority vote correct: {majority_correct} ({majority_correct/max(total,1)*100:.0f}%)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
