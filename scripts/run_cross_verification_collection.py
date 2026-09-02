#!/usr/bin/env python3
"""Collect cross-verification data: ask the model to verify each candidate's answer.

This adds a genuinely new signal beyond self-confidence:
  - Self-confidence: "How confident are you in YOUR answer?"
  - Cross-verification: "Is THIS specific answer correct?"

The cross-verification signal is different because:
  - It evaluates a specific answer, not the model's own output
  - It can catch cases where the model is confidently wrong
  - It provides an independent assessment of each candidate

Also collects pairwise comparisons:
  "Which answer is more likely correct: X or Y?"

Usage:
    python scripts/run_cross_verification_collection.py \\
        --corpus experiments/daph_x/reasoning/reasoning_corpus_v2.jsonl \\
        --model_path /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf
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
    """Extract verification score from model response.
    Returns (score 0-1, reasoning).
    """
    response_lower = response.lower().strip()

    # Check for yes/no/correct/incorrect
    if any(w in response_lower for w in ["yes, it is correct", "yes, this is correct",
                                          "the answer is correct", "this answer is correct",
                                          "yes, that is correct", "yes, correct"]):
        return 1.0, response
    if any(w in response_lower for w in ["no, it is not correct", "no, this is not correct",
                                          "the answer is incorrect", "this answer is incorrect",
                                          "no, that is not correct", "no, incorrect",
                                          "no, it's not correct", "this is wrong"]):
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

    # Default: neutral
    return 0.5, response


def extract_pairwise_choice(response: str, answer_a: str, answer_b: str) -> tuple[str, float]:
    """Extract which answer the model prefers in pairwise comparison.
    Returns (choice: 'a'/'b'/'tie', confidence).
    """
    response_lower = response.lower()

    # Check for explicit preference
    a_mentioned = answer_a.lower() in response_lower
    b_mentioned = answer_b.lower() in response_lower

    # Look for "answer A" / "answer B" / "first" / "second"
    if any(w in response_lower for w in ["answer a is", "a is correct", "a is more likely",
                                          "first answer is", "the first is correct",
                                          "i would choose a", "option a"]):
        return "a", 0.8
    if any(w in response_lower for w in ["answer b is", "b is correct", "b is more likely",
                                          "second answer is", "the second is correct",
                                          "i would choose b", "option b"]):
        return "b", 0.8

    # Look for the actual answer values
    if a_mentioned and not b_mentioned:
        return "a", 0.6
    if b_mentioned and not a_mentioned:
        return "b", 0.6

    # Check for "both" / "either" / "same" / "tie"
    if any(w in response_lower for w in ["both are", "either", "same", "tie", "equally"]):
        return "tie", 0.5

    # Look for "A" or "B" as standalone
    if re.search(r"\bA\b(?!\w)", response) and not re.search(r"\bB\b(?!\w)", response):
        return "a", 0.6
    if re.search(r"\bB\b(?!\w)", response) and not re.search(r"\bA\b(?!\w)", response):
        return "b", 0.6

    return "tie", 0.5


def verify_candidate(model: CodingModelInterface, task_prompt: str,
                     candidate_answer: str, seed: int) -> dict:
    """Ask the model to verify a specific answer."""
    verify_prompt = (
        f"Question: {task_prompt}\n\n"
        f"Proposed answer: {candidate_answer}\n\n"
        f"Is this answer correct? Think step by step, then say 'Yes' or 'No'."
    )
    response = model.generate_raw(
        prompt=verify_prompt, temperature=0.0,
        max_tokens=200, seed=seed,
        system_prompt="You are a careful math tutor. Verify answers step by step."
    )
    score, reasoning = extract_verification_score(response)
    return {"verification_score": score, "verification_response": response}


def pairwise_compare(model: CodingModelInterface, task_prompt: str,
                     answer_a: str, answer_b: str, seed: int) -> dict:
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
    parser.add_argument("--n_tasks", type=int, default=120)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip_pairs", action="store_true",
                        help="Skip pairwise comparisons (faster)")
    args = parser.parse_args()

    CV_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    tasks = tasks[:args.n_tasks]
    print(f"Cross-verification collection")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Skip pairs: {args.skip_pairs}")
    print()

    model = CodingModelInterface(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else CV_DIR / "cv_corpus.jsonl"

    with open(output_path, "w") as f:
        for idx, task in enumerate(tasks):
            task_id = task["task_id"]
            task_obj = get_reasoning_task(task_id)
            if task_obj is None:
                continue

            print(f"[{idx+1}/{len(tasks)}] {task_id}: verifying {len(task['candidates'])} candidates...")

            t0 = time.monotonic()

            # Verify each candidate
            for i, cand in enumerate(task["candidates"]):
                cv = verify_candidate(
                    model=model,
                    task_prompt=task_obj.prompt,
                    candidate_answer=cand["answer"],
                    seed=args.seed + 10000 + i,
                )
                cand["verification"] = cv

            # Pairwise comparisons (all pairs)
            if not args.skip_pairs:
                pairs = []
                cands = task["candidates"]
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        pw = pairwise_compare(
                            model=model,
                            task_prompt=task_obj.prompt,
                            answer_a=cands[i]["answer"],
                            answer_b=cands[j]["answer"],
                            seed=args.seed + 20000 + i * 100 + j,
                        )
                        # Convert to preference: 1 if prefers i, -1 if prefers j, 0 if tie
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

            elapsed = time.monotonic() - t0
            n_verify = len(task["candidates"])
            n_pairs = len(task.get("pairwise_comparisons", []))
            print(f"  Done: {n_verify} verifications + {n_pairs} pairs in {elapsed:.1f}s")

            f.write(json.dumps(task, default=str) + "\n")
            f.flush()

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
