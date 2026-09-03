#!/usr/bin/env python3
"""Oracle@K study: measure the absolute reranking headroom.

For each task, compute:
  Acc_base       = P(candidate[0] correct)
  Acc_MaxCal@K   = P(MaxCal pick correct) with K candidates
  Acc_Majority@K = P(majority vote correct) with K candidates
  Acc_Oracle@K   = P(any of K candidates correct)

The headroom over MaxCal is:
  H_K = Acc_Oracle@K - Acc_MaxCal@K

This is the number DAPH-X cannot exceed by reranking.

We have 6 candidates per task in the existing corpus. To study K > 6,
we need to generate additional candidates. This script:
  1. Uses existing 6-candidate data for K ∈ {1,2,4,6}
  2. Generates additional candidates for K ∈ {8,12,16,24,32}

For K ≤ 6, we subsample from existing candidates.
For K > 6, we generate new candidates with different temperatures/seeds.

Usage:
    python scripts/run_oracle_at_k.py \\
        --corpus experiments/daph_x/cross_verification/cv_corpus_v2.jsonl \\
        --max_k 32 --n_tasks 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ORACLE_DIR = REPO_ROOT / "experiments/daph_x/oracle"

from daph_x.coding.model_interface import CodingModelInterface
from daph_x.coding.reasoning_tasks import get_reasoning_task, check_answer

# Import extract_answer from the collection script
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_reasoning_collection import extract_answer


def load_corpus(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def oracle_at_k(candidates: list[dict], k: int) -> bool:
    """Is any of the first k candidates correct?"""
    return any(c["is_correct"] for c in candidates[:k])


def maxcal_at_k(candidates: list[dict], k: int, corr_model=None, corr_cal=None, fk=None) -> bool:
    """MaxCal pick among first k candidates correct?"""
    if corr_model is None:
        # Without calibration, use self-confidence as proxy
        pick = max(candidates[:k], key=lambda c: c["self_confidence"])
    else:
        # With calibration (from R9)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from run_r9_evaluation import predict_correctness_r9
        pick = max(candidates[:k], key=lambda c: predict_correctness_r9(
            corr_model, corr_cal, c.get("enriched_features", {}), c, fk))
    return pick["is_correct"]


def majority_at_k(candidates: list[dict], k: int) -> bool:
    """Majority vote among first k candidates correct?"""
    answers = [c["answer"] for c in candidates[:k]]
    mv = Counter(answers).most_common(1)[0][0]
    return any(c["is_correct"] for c in candidates[:k] if c["answer"] == mv)


def compute_oracle_curve(tasks, max_k, corr_model=None, corr_cal=None, fk=None):
    """Compute oracle@K, MaxCal@K, majority@K, base@K for all K values."""
    ks = [k for k in [1, 2, 4, 6, 8, 12, 16, 24, 32] if k <= max_k]

    results = {}
    for k in ks:
        oracle_accs = []
        maxcal_accs = []
        majority_accs = []
        base_accs = []

        for task in tasks:
            cands = task["candidates"]
            if len(cands) < k:
                continue

            oracle_accs.append(1.0 if oracle_at_k(cands, k) else 0.0)
            maxcal_accs.append(1.0 if maxcal_at_k(cands, k, corr_model, corr_cal, fk) else 0.0)
            majority_accs.append(1.0 if majority_at_k(cands, k) else 0.0)
            base_accs.append(1.0 if cands[0]["is_correct"] else 0.0)

        n = len(oracle_accs)
        if n == 0:
            continue

        results[k] = {
            "n": n,
            "oracle": float(np.mean(oracle_accs)),
            "maxcal": float(np.mean(maxcal_accs)),
            "majority": float(np.mean(majority_accs)),
            "base": float(np.mean(base_accs)),
            "headroom_oracle_vs_maxcal": float(np.mean(oracle_accs) - np.mean(maxcal_accs)),
            "headroom_oracle_vs_base": float(np.mean(oracle_accs) - np.mean(base_accs)),
        }

    return results


def generate_extra_candidates(model, task, existing_answers, n_extra, base_seed):
    """Generate additional candidates with different temperatures/seeds."""
    task_obj = get_reasoning_task(task["task_id"])
    if task_obj is None:
        return []

    # Use varied temperatures to get diversity
    temps = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 0.4, 0.6, 0.8, 1.1]
    new_cands = []

    for i in range(n_extra):
        temp = temps[i % len(temps)]
        seed = base_seed + i * 137

        try:
            response = model.generate_raw(
                prompt=task_obj.prompt,
                temperature=temp,
                max_tokens=300,
                seed=seed,
            )
            if response:
                answer = extract_answer(response)
                is_correct = check_answer(answer, task_obj.answer, task_obj.answer_type)
                # Estimate confidence from response
                conf_match = __import__('re').search(r'(\d{1,3})\s*%', response)
                conf = float(conf_match.group(1)) if conf_match else 50.0
                new_cands.append({
                    "answer": answer,
                    "is_correct": is_correct,
                    "self_confidence": conf,
                    "temperature": temp,
                    "seed": seed,
                    "candidate_id": f"extra_{i}",
                    "response": response,
                })
        except Exception as e:
            print(f"    Error generating candidate {i}: {e}")

    return new_cands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(ORACLE_DIR / "oracle_corpus.jsonl"))
    parser.add_argument("--model_path",
        default="/Users/dawsonblock/Downloads/qwen_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--max_k", type=int, default=32)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_extra", action="store_true",
                        help="Generate extra candidates for K > 6")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = load_corpus(args.corpus)
    tasks = tasks[:args.n_tasks]
    print(f"Oracle@K study")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Max K: {args.max_k}")
    print(f"  Generate extra: {args.generate_extra}")
    print()

    # If we need K > 6 and generation is enabled, generate extra candidates
    if args.max_k > 6 and args.generate_extra:
        max_existing = max(len(t["candidates"]) for t in tasks)
        print(f"  Existing max candidates per task: {max_existing}")

        if max_existing < args.max_k:
            n_extra_needed = args.max_k - max_existing
            print(f"  Need to generate {n_extra_needed} extra candidates per task")

            model = CodingModelInterface(
                model_path=args.model_path,
                n_gpu_layers=args.n_gpu_layers,
                seed=args.seed,
            )

            output_path = ORACLE_DIR / "oracle_corpus_extended.jsonl"
            with open(output_path, "w") as f:
                for idx, task in enumerate(tasks):
                    n_current = len(task["candidates"])
                    if n_current < args.max_k:
                        existing_answers = [c["answer"] for c in task["candidates"]]
                        extra = generate_extra_candidates(
                            model, task, existing_answers,
                            args.max_k - n_current,
                            base_seed=args.seed + idx * 10000,
                        )
                        task["candidates"].extend(extra)

                    if (idx + 1) % 10 == 0:
                        print(f"  Generated extra for {idx+1}/{len(tasks)} tasks")

                    f.write(json.dumps(task, default=str) + "\n")
                    f.flush()

            # Reload extended corpus
            tasks = load_corpus(str(output_path))
            print(f"  Extended corpus: {len(tasks)} tasks, max {max(len(t['candidates']) for t in tasks)} candidates")

    # Compute oracle curve
    print(f"\n{'K':>4} {'N':>5} {'Base':>7} {'Maj':>7} {'MaxCal':>7} {'Oracle':>7} {'H_O-M':>7} {'H_O-B':>7}")
    print("-" * 60)

    # Also compute with calibrated MaxCal if we have features
    # For simplicity, use self-confidence as MaxCal proxy here
    # (the full calibrated version requires training, which we skip for the oracle study)
    results = compute_oracle_curve(tasks, args.max_k)

    for k in sorted(results.keys()):
        r = results[k]
        print(f"{k:>4} {r['n']:>5} {r['base']:>6.1%} {r['majority']:>6.1%} "
              f"{r['maxcal']:>6.1%} {r['oracle']:>6.1%} "
              f"{r['headroom_oracle_vs_maxcal']:>+6.1%} {r['headroom_oracle_vs_base']:>+6.1%}")

    # Analysis
    print(f"\n{'='*80}")
    print(f"  ORACLE@K ANALYSIS")
    print(f"{'='*80}")

    if 6 in results and 32 in results:
        r6 = results[6]
        r32 = results[32]
        print(f"\n  Oracle@6:   {r6['oracle']:.1%}")
        print(f"  Oracle@32:  {r32['oracle']:.1%}")
        print(f"  MaxCal@6:   {r6['maxcal']:.1%}")
        print(f"  MaxCal@32:  {r32['maxcal']:.1%}")
        print(f"  Headroom@6  (Oracle-MaxCal): {r6['headroom_oracle_vs_maxcal']:.1%}")
        print(f"  Headroom@32 (Oracle-MaxCal): {r32['headroom_oracle_vs_maxcal']:.1%}")

        if r32['oracle'] - r6['oracle'] > 0.05:
            print(f"\n  → Oracle rises substantially with more candidates ({r6['oracle']:.1%} → {r32['oracle']:.1%})")
            print(f"  → Generator is NOT the bottleneck; more candidates = more correct answers")
            print(f"  → This is the regime where DAPH-X should operate (acquire more cognition)")
        else:
            print(f"\n  → Oracle saturates quickly ({r6['oracle']:.1%} → {r32['oracle']:.1%})")
            print(f"  → Generator IS the bottleneck; more candidates don't help much")
            print(f"  → DAPH-X needs better generation, not better selection")

        if r6['headroom_oracle_vs_maxcal'] < 0.03:
            print(f"\n  → Headroom@6 is only {r6['headroom_oracle_vs_maxcal']:.1%}")
            print(f"  → MaxCal captures almost all available signal with 6 candidates")
            print(f"  → Reranking headroom is negligible")
        elif r32['headroom_oracle_vs_maxcal'] > 0.05:
            print(f"\n  → Headroom@32 is {r32['headroom_oracle_vs_maxcal']:.1%}")
            print(f"  → MaxCal leaves significant signal on the table with more candidates")
            print(f"  → Authority has meaningful problem to solve")

    # Rescue opportunity analysis
    print(f"\n  RESCUE OPPORTUNITY ANALYSIS")
    print(f"  {'K':>4} {'P(any correct)':>15} {'P(MaxCal wrong)':>16} {'P(rescue available)':>20}")
    print("  " + "-" * 60)

    for k in sorted(results.keys()):
        cands_per_task = [t["candidates"][:k] for t in tasks if len(t["candidates"]) >= k]
        if not cands_per_task:
            continue
        p_any_correct = np.mean([any(c["is_correct"] for c in cands) for cands in cands_per_task])
        # MaxCal wrong = max confidence pick wrong
        p_maxcal_wrong = np.mean([
            not max(cands, key=lambda c: c["self_confidence"])["is_correct"]
            for cands in cands_per_task
        ])
        # Rescue available = MaxCal wrong AND some candidate correct
        p_rescue = np.mean([
            (not max(cands, key=lambda c: c["self_confidence"])["is_correct"])
            and any(c["is_correct"] for c in cands)
            for cands in cands_per_task
        ])
        print(f"  {k:>4} {p_any_correct:>14.1%} {p_maxcal_wrong:>15.1%} {p_rescue:>19.1%}")

    output_path = Path(args.output) if args.output else ORACLE_DIR / "oracle_results.json"
    with open(output_path, "w") as f:
        json.dump({"results": results, "max_k": args.max_k, "n_tasks": len(tasks)}, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
