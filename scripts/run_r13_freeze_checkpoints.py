#!/usr/bin/env python3
"""R13 Stage 5: Freeze checkpoint states from R12 corpus.

Samples 200-300 frozen checkpoints from R12 using a prospective
stratified rule. Each checkpoint is immutable and all operators
must start from the exact same serialized state.

Stratification:
  - K ∈ {2, 4, 6} (early checkpoints where rescue is most likely)
  - Correct/incorrect MaxCal pick
  - High/low confidence (p_top1)
  - Stable/unstable (answer changed in last 2 checkpoints)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.operators.base import CheckpointState


def load_r12_corpus(path: Path) -> list:
    """Load the R12 enriched corpus."""
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def compute_maxcal_pick(candidates: list, k: int) -> dict:
    """Compute MaxCal pick at checkpoint K.

    MaxCal = majority vote among first K candidates,
    with confidence = fraction agreeing with majority.
    """
    if k == 0 or len(candidates) == 0:
        return {"answer": "", "correct": False, "confidence": 0.0}

    cands_k = candidates[:k]
    answers = [c["answer"] for c in cands_k]

    # Majority vote
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0]
    agreement = answer_counts[majority_answer] / len(answers)

    # Check correctness
    any_correct = any(c["is_correct"] for c in cands_k)
    # Check if majority answer is correct
    # Use the first candidate with majority answer
    majority_cand = next((c for c in cands_k if c["answer"] == majority_answer), None)
    majority_correct = majority_cand["is_correct"] if majority_cand else False

    return {
        "answer": majority_answer,
        "correct": majority_correct,
        "confidence": agreement,
        "agreement_rate": agreement,
    }


def compute_state_features(candidates: list, k: int, prev_features: dict = None) -> dict:
    """Compute observable state features at checkpoint K."""
    cands_k = candidates[:k]
    if not cands_k:
        return {}

    answers = [c["answer"] for c in cands_k]
    answer_counts = Counter(answers)
    n_unique = len(answer_counts)

    # Confidence features
    maxcal = compute_maxcal_pick(candidates, k)
    p_top1 = maxcal["confidence"]
    # p_top2 = second most common answer fraction
    if len(answer_counts) > 1:
        p_top2 = answer_counts.most_common(2)[1][1] / len(answers)
    else:
        p_top2 = 0.0
    margin = p_top1 - p_top2

    # Entropy
    import math
    probs = [c / len(answers) for c in answer_counts.values()]
    entropy = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0.0

    # Stability
    if prev_features:
        prev_answer = prev_features.get("maxcal_answer", "")
        curr_answer = maxcal["answer"]
        answer_changed = prev_answer != curr_answer
        delta_p_top1 = p_top1 - prev_features.get("p_top1", 0.0)
        delta_entropy = entropy - prev_features.get("answer_entropy", 0.0)
    else:
        answer_changed = False
        delta_p_top1 = 0.0
        delta_entropy = 0.0

    return {
        "k": k,
        "p_top1": p_top1,
        "p_top2": p_top2,
        "margin": margin,
        "answer_entropy": entropy,
        "n_unique_answers": n_unique,
        "agreement_rate": maxcal["agreement_rate"],
        "maxcal_answer": maxcal["answer"],
        "maxcal_correct": maxcal["correct"],
        "maxcal_confidence": maxcal["confidence"],
        "answer_changed": answer_changed,
        "delta_p_top1": delta_p_top1,
        "delta_entropy": delta_entropy,
    }


def classify_state(features: dict, prev_features: dict = None) -> dict:
    """Classify a state for stratification."""
    p_top1 = features.get("p_top1", 0.0)
    maxcal_correct = features.get("maxcal_correct", False)
    answer_changed = features.get("answer_changed", False)

    return {
        "k": features["k"],
        "correctness": "correct" if maxcal_correct else "incorrect",
        "confidence": "high" if p_top1 > 0.6 else "low",
        "stability": "stable" if not answer_changed else "unstable",
        # Combined stratum
        "stratum": f"{features['k']}_{maxcal_correct}_{p_top1 > 0.6}_{not answer_changed}",
    }


def freeze_checkpoints(tasks: list, checkpoints: list = None,
                       target_per_stratum: int = 15) -> list:
    """Freeze checkpoint states from R12 tasks.

    Stratifies by K, correctness, confidence, and stability.
    Targets ~200-300 total checkpoints.
    """
    if checkpoints is None:
        checkpoints = [2, 4, 6]

    # Build all candidate checkpoints with classification
    all_checkpoints = []
    for task in tasks:
        cands = task["candidates"]
        if len(cands) < 12:
            continue

        prev_features = None
        for k in checkpoints:
            features = compute_state_features(cands, k, prev_features)
            classification = classify_state(features, prev_features)

            # Build the frozen state
            state = CheckpointState(
                task_id=task["task_id"],
                task_prompt=task.get("description", ""),  # Use description as prompt
                correct_answer=task["correct_answer"],
                answer_type=task["answer_type"],
                difficulty=task["difficulty"],
                category=task["category"],
                candidates=cands[:k],
                k=k,
                features=features,
                maxcal_answer=features["maxcal_answer"],
                maxcal_correct=features["maxcal_correct"],
                maxcal_confidence=features["maxcal_confidence"],
                prev_state=prev_features,
            )

            all_checkpoints.append({
                "state": state.serialize(),
                "classification": classification,
            })

            prev_features = features

    # Stratify and sample
    strata = {}
    for cp in all_checkpoints:
        s = cp["classification"]["stratum"]
        strata.setdefault(s, []).append(cp)

    # Sample target_per_stratum from each stratum
    import random
    rng = random.Random(42)

    selected = []
    for stratum, cps in sorted(strata.items()):
        n_sample = min(target_per_stratum, len(cps))
        sampled = rng.sample(cps, n_sample)
        selected.extend(sampled)
        print(f"  Stratum {stratum}: {len(cps)} available, {n_sample} selected")

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="experiments/daph_x/r12/r12_enriched_corpus.jsonl")
    parser.add_argument("--output", default="experiments/daph_x/r13/r13a_checkpoints.jsonl")
    parser.add_argument("--target_per_stratum", type=int, default=15)
    parser.add_argument("--checkpoints", default="2,4,6")
    args = parser.parse_args()

    corpus_path = REPO_ROOT / args.corpus
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"R13 Stage 5: Freezing checkpoints from R12")
    print(f"  Corpus: {corpus_path}")
    print(f"  Output: {output_path}")

    tasks = load_r12_corpus(corpus_path)
    print(f"  Loaded {len(tasks)} tasks")

    checkpoints = [int(k) for k in args.checkpoints.split(",")]
    print(f"  Checkpoints: K={checkpoints}")
    print(f"  Target per stratum: {args.target_per_stratum}")
    print()

    selected = freeze_checkpoints(tasks, checkpoints, args.target_per_stratum)

    # Save
    with open(output_path, "w") as f:
        for cp in selected:
            f.write(json.dumps({
                "state": cp["state"],
                "classification": cp["classification"],
            }, default=str) + "\n")

    print(f"\n  Frozen {len(selected)} checkpoints to {output_path}")

    # Summary
    from collections import Counter
    k_counts = Counter(cp["classification"]["k"] for cp in selected)
    corr_counts = Counter(cp["classification"]["correctness"] for cp in selected)
    conf_counts = Counter(cp["classification"]["confidence"] for cp in selected)
    stab_counts = Counter(cp["classification"]["stability"] for cp in selected)

    print(f"\n  Distribution:")
    print(f"    K: {dict(sorted(k_counts.items()))}")
    print(f"    Correctness: {dict(corr_counts)}")
    print(f"    Confidence: {dict(conf_counts)}")
    print(f"    Stability: {dict(stab_counts)}")


if __name__ == "__main__":
    main()
