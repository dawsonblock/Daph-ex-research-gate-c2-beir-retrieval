#!/usr/bin/env python3
"""Collect comprehensive cross-verification data: multi-round + pairwise.

For each task:
  1. Generate 6 candidates (reuse existing corpus)
  2. Multi-round verification (3 rounds per candidate with different phrasings)
  3. Pairwise comparisons (all C(6,2)=15 pairs)

Multi-round verification uses 3 different prompt phrasings:
  Round 1: "Is this answer correct?"
  Round 2: "Verify this answer step by step. Is it right or wrong?"
  Round 3: "A student gave this answer. Grade it as correct or incorrect."

The average verification score across rounds is more robust than single-round.

Usage:
    python scripts/run_cv_collection_v2.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus_v2.jsonl \\
        --n_tasks 200 --n_candidates 6 --n_rounds 3
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
from daph_x.coding.reasoning_tasks import get_reasoning_task, check_answer

CV_DIR = REPO_ROOT / "experiments/daph_x/cross_verification"


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def extract_verification_score(response: str) -> tuple[float, str]:
    """Extract verification score from model response."""
    response_lower = response.lower().strip()

    # Strong yes patterns
    if any(w in response_lower for w in [
        "yes, it is correct", "yes, this is correct", "the answer is correct",
        "this answer is correct", "yes, that is correct", "yes, correct",
        "the answer is right", "this is the correct answer", "grade: correct",
        "correct.", "correct answer"
    ]):
        return 1.0, response
    # Strong no patterns
    if any(w in response_lower for w in [
        "no, it is not correct", "no, this is not correct", "the answer is incorrect",
        "this answer is incorrect", "no, that is not correct", "no, incorrect",
        "the answer is wrong", "this is wrong", "grade: incorrect",
        "incorrect.", "incorrect answer", "it is not right"
    ]):
        return 0.0, response

    # Look for confidence percentage
    match = re.search(r"(\d{1,3})\s*%", response)
    if match:
        score = int(match.group(1)) / 100.0
        return score, response

    # Look for 0-10 scale
    match = re.search(r"\b([0-9]|10)\s*/\s*10\b", response)
    if match:
        score = int(match.group(1)) / 10.0
        return score, response

    # Look for yes/no at start
    if response_lower.startswith("yes"):
        return 0.8, response
    if response_lower.startswith("no"):
        return 0.2, response

    # Look for "correct" or "incorrect" anywhere
    if "correct" in response_lower and "incorrect" not in response_lower:
        return 0.7, response
    if "incorrect" in response_lower or "wrong" in response_lower:
        return 0.3, response

    return 0.5, response


def extract_pairwise_choice(response: str, answer_a: str, answer_b: str) -> tuple[str, float]:
    """Extract which answer the model prefers."""
    response_lower = response.lower()

    # Check for explicit preference patterns
    if any(w in response_lower for w in [
        "answer a is", "a is correct", "a is more likely", "a is right",
        "first answer is", "the first is correct", "i choose a", "option a",
        "answer a", "a is the correct"
    ]):
        return "a", 0.8
    if any(w in response_lower for w in [
        "answer b is", "b is correct", "b is more likely", "b is right",
        "second answer is", "the second is correct", "i choose b", "option b",
        "answer b", "b is the correct"
    ]):
        return "b", 0.8

    # Check for "both" / "tie"
    if any(w in response_lower for w in ["both are", "either", "same", "tie", "equally"]):
        return "tie", 0.5

    # Look for standalone A or B
    a_pattern = re.search(r"\bA\b(?!\w)", response)
    b_pattern = re.search(r"\bB\b(?!\w)", response)
    if a_pattern and not b_pattern:
        return "a", 0.6
    if b_pattern and not a_pattern:
        return "b", 0.6

    # Check for actual answer values
    a_in = answer_a.lower() in response_lower
    b_in = answer_b.lower() in response_lower
    if a_in and not b_in:
        return "a", 0.6
    if b_in and not a_in:
        return "b", 0.6

    return "tie", 0.5


# Multi-round verification prompts
VERIFICATION_PROMPTS = [
    # Round 1: Direct
    lambda q, a: f"Question: {q}\n\nProposed answer: {a}\n\nIs this answer correct? Think step by step, then say 'Yes' or 'No'.",
    # Round 2: Grading
    lambda q, a: f"Question: {q}\n\nA student answered: {a}\n\nGrade this answer as 'correct' or 'incorrect'. Show your work, then give the grade.",
    # Round 3: Verification with skepticism
    lambda q, a: f"Question: {q}\n\nSomeone claims the answer is: {a}\n\nVerify this claim carefully. Is it right or wrong? Explain, then conclude with 'right' or 'wrong'.",
]


def verify_candidate_multi_round(model, task_prompt, candidate_answer, seed, n_rounds=3):
    """Run multi-round verification with different prompt phrasings."""
    rounds = []
    scores = []
    for r in range(min(n_rounds, len(VERIFICATION_PROMPTS))):
        prompt_fn = VERIFICATION_PROMPTS[r]
        verify_prompt = prompt_fn(task_prompt, candidate_answer)
        response = model.generate_raw(
            prompt=verify_prompt, temperature=0.0,
            max_tokens=200, seed=seed + r * 1000,
            system_prompt="You are a careful math tutor. Verify answers step by step."
        )
        score, reasoning = extract_verification_score(response)
        rounds.append({"round": r, "score": score, "response": response})
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.5
    score_std = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0.0

    return {
        "verification_score": avg_score,
        "verification_std": score_std,
        "verification_rounds": rounds,
        "verification_consistent": 1.0 if score_std < 0.1 else 0.0,
    }


def pairwise_compare(model, task_prompt, answer_a, answer_b, seed):
    """Ask the model to compare two answers."""
    compare_prompt = (
        f"Question: {task_prompt}\n\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n\n"
        f"Which answer is more likely correct? Think step by step, then say 'A' or 'B'."
    )
    response = model.generate_raw(
        prompt=compare_prompt, temperature=0.0,
        max_tokens=200, seed=seed,
        system_prompt="You are a careful math tutor. Compare answers step by step."
    )
    choice, confidence = extract_pairwise_choice(response, answer_a, answer_b)
    return {"pairwise_choice": choice, "pairwise_confidence": confidence,
            "pairwise_response": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CV_DIR / "input_corpus.jsonl"))
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--n_candidates", type=int, default=6)
    parser.add_argument("--n_rounds", type=int, default=3)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip_pairs", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    CV_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    tasks = tasks[args.start_idx:args.start_idx + args.n_tasks]
    print(f"Cross-verification collection v2")
    print(f"  Tasks: {len(tasks)} (from idx {args.start_idx})")
    print(f"  Candidates per task: {args.n_candidates}")
    print(f"  Verification rounds: {args.n_rounds}")
    print(f"  Skip pairs: {args.skip_pairs}")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else CV_DIR / "cv_corpus_v2.jsonl"

    with open(output_path, "w") as f:
        for idx, task in enumerate(tasks):
            task_id = task["task_id"]
            task_obj = get_reasoning_task(task_id)
            if task_obj is None:
                print(f"  SKIP {task_id}: task not found")
                continue

            cands = task["candidates"][:args.n_candidates]
            n_c = len(cands)
            n_pairs = n_c * (n_c - 1) // 2
            total_calls = n_c * args.n_rounds + (0 if args.skip_pairs else n_pairs)
            print(f"[{idx+1}/{len(tasks)}] {task_id}: {n_c} cands × {args.n_rounds} rounds + {n_pairs if not args.skip_pairs else 0} pairs = {total_calls} calls")

            t0 = time.monotonic()

            # Multi-round verification
            for i, cand in enumerate(cands):
                cv = verify_candidate_multi_round(
                    model=model,
                    task_prompt=task_obj.prompt,
                    candidate_answer=cand["answer"],
                    seed=args.seed + 10000 + i * 100,
                    n_rounds=args.n_rounds,
                )
                cand["verification"] = cv

            # Pairwise comparisons
            if not args.skip_pairs:
                pairs = []
                for i in range(n_c):
                    for j in range(i + 1, n_c):
                        pw = pairwise_compare(
                            model=model,
                            task_prompt=task_obj.prompt,
                            answer_a=cands[i]["answer"],
                            answer_b=cands[j]["answer"],
                            seed=args.seed + 20000 + i * 100 + j,
                        )
                        if pw["pairwise_choice"] == "a":
                            pref = 1
                        elif pw["pairwise_choice"] == "b":
                            pref = -1
                        else:
                            pref = 0
                        pairs.append({
                            "i": i, "j": j,
                            "preference": pref,
                            "confidence": pw["pairwise_confidence"],
                        })
                task["pairwise_comparisons"] = pairs

            task["candidates"] = cands

            elapsed = time.monotonic() - t0
            print(f"  Done in {elapsed:.1f}s")

            f.write(json.dumps(task, default=str) + "\n")
            f.flush()

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
