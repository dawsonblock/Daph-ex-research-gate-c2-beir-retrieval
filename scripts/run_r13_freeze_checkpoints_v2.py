#!/usr/bin/env python3
"""R13-A v2: Freeze immutable checkpoints from R12 corpus."""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import math
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.operators.types import RuntimeState, Candidate, TrajectoryPoint
from daph_x.receipts.reasoning_checkpoint import ReasoningCheckpoint
from daph_x.evaluation.r12_selector import select_r12_maxcal
from daph_x.coding.reasoning_tasks import get_reasoning_task


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_prefix_trajectory(candidates: list, k_values: list) -> list:
    """Build TrajectoryPoint for each prefix 1..max(k_values)."""
    points = []
    for k in range(1, max(k_values) + 1):
        prefix = candidates[:k]
        answers = [c["answer"] for c in prefix]
        answer_counts = Counter(answers)
        top = answer_counts.most_common(1)[0]
        top_answer = top[0]
        support = top[1]
        p_top1 = support / k
        if len(answer_counts) > 1:
            p_top2 = answer_counts.most_common(2)[1][1] / k
        else:
            p_top2 = 0.0
        margin = p_top1 - p_top2
        probs = [c / k for c in answer_counts.values()]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs) if probs else 0.0
        points.append(TrajectoryPoint(
            k=k,
            top_answer=top_answer,
            p_top1=p_top1,
            p_top2=p_top2,
            margin=margin,
            entropy=entropy,
            n_unique=len(answer_counts),
        ))
    return points


def compute_observable_features(trajectory: list, target_k: int) -> dict:
    """Compute router-visible features from immutable observable state."""
    if not trajectory or target_k > len(trajectory):
        return {}
    point = trajectory[target_k - 1]
    prev = trajectory[target_k - 2] if target_k > 1 else None

    uncertainty = point.entropy / math.log(max(2, point.k)) if point.k > 0 else 0.0
    uncertainty_delta = (uncertainty - (prev and (prev.entropy / math.log(max(2, prev.k)) or 0.0))) if prev else 0.0
    margin_delta = (point.margin - prev.margin) if prev else 0.0
    answer_changed = 1 if (prev and point.top_answer != prev.top_answer) else 0
    stable_count = 0
    for i in range(target_k - 1, -1, -1):
        if i == 0:
            stable_count += 1
        elif trajectory[i].top_answer == trajectory[i - 1].top_answer:
            stable_count += 1
        else:
            break

    return {
        "k": float(point.k),
        "p_top1": point.p_top1,
        "p_top2": point.p_top2,
        "margin": point.margin,
        "entropy": point.entropy,
        "n_unique_answers": float(point.n_unique),
        "agreement_rate": point.p_top1,
        "uncertainty_current": uncertainty,
        "uncertainty_delta": uncertainty_delta,
        "margin_delta": margin_delta,
        "answer_changed": float(answer_changed),
        "stable_prefix_count": float(stable_count),
    }


def freeze_checkpoints_v2(
    corpus_path: Path,
    output_path: Path,
    k_values: list,
    target_per_stratum: int = 15,
    seed: int = 42,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    with open(corpus_path) as f:
        for line in f:
            tasks.append(json.loads(line))

    corpus_sha256 = sha256_file(corpus_path)
    all_checkpoints = []

    for task in tasks:
        if len(task["candidates"]) < max(k_values):
            continue

        cands_raw = task["candidates"]
        cands = [
            Candidate(
                candidate_id=f"{task['task_id']}_c{i}",
                answer=c["answer"],
                reasoning_trace=c.get("response", ""),
                temperature=c.get("temperature", 0.0),
                seed=c.get("seed", 0),
                generation_index=i,
                metadata={"temperature": c.get("temperature"), "gen_latency_ms": c.get("gen_latency_ms")},
            )
            for i, c in enumerate(cands_raw)
        ]

        trajectory = build_prefix_trajectory(cands_raw, k_values)

        reasoning_task = get_reasoning_task(task["task_id"])
        task_prompt = reasoning_task.prompt if reasoning_task else task.get("description", "")

        for k in k_values:
            prefix_cands = cands[:k]
            selection = select_r12_maxcal(prefix_cands)
            features = compute_observable_features(trajectory, k)

            runtime_state = RuntimeState(
                checkpoint_id=f"{task['task_id']}_k{k}",
                task_id=task["task_id"],
                task_prompt=task_prompt,
                answer_type=task["answer_type"],
                category=task["category"],
                difficulty=task["difficulty"],
                candidates=tuple(prefix_cands),
                trajectory=tuple(trajectory[:k]),
                k=k,
                current_answer=selection.answer,
                observable_features=features,
                state_hash="",  # filled after
            )

            # Recompute state hash
            state_hash = runtime_state.sha256()
            # dataclass is frozen, so we need to create a new one
            runtime_state = RuntimeState(
                checkpoint_id=runtime_state.checkpoint_id,
                task_id=runtime_state.task_id,
                task_prompt=runtime_state.task_prompt,
                answer_type=runtime_state.answer_type,
                category=runtime_state.category,
                difficulty=runtime_state.difficulty,
                candidates=runtime_state.candidates,
                trajectory=runtime_state.trajectory,
                k=runtime_state.k,
                current_answer=runtime_state.current_answer,
                observable_features=runtime_state.observable_features,
                state_hash=state_hash,
            )

            # Compute stratum for sampling
            p_top1 = features["p_top1"]
            is_correct = False  # not observable
            stratum = f"{k}_{p_top1 > 0.6}"

            checkpoint = ReasoningCheckpoint(
                checkpoint_id=runtime_state.checkpoint_id,
                runtime_state=runtime_state,
                dataset_id="r12_corpus_v1",
                corpus_sha256=corpus_sha256,
                selector_version="r12-selector-v1",
                feature_version="r13-state-v2",
            )

            all_checkpoints.append({
                "checkpoint": checkpoint,
                "stratum": stratum,
            })

    # Stratified sample
    import random
    rng = random.Random(seed)
    strata = {}
    for cp in all_checkpoints:
        s = cp["stratum"]
        strata.setdefault(s, []).append(cp)

    selected = []
    for s, cps in sorted(strata.items()):
        n_sample = min(target_per_stratum, len(cps))
        sampled = rng.sample(cps, n_sample)
        selected.extend(sampled)

    with open(output_path, "w") as f:
        for cp in selected:
            rs_dict = json.loads(cp["checkpoint"].runtime_state.canonical_bytes())
            rs_dict["state_hash"] = cp["checkpoint"].runtime_state.state_hash
            f.write(json.dumps({
                "checkpoint_id": cp["checkpoint"].checkpoint_id,
                "state_hash": cp["checkpoint"].runtime_state.state_hash,
                "checkpoint_sha256": cp["checkpoint"].sha256(),
                "runtime_state": rs_dict,
                "dataset_id": cp["checkpoint"].dataset_id,
                "corpus_sha256": cp["checkpoint"].corpus_sha256,
                "selector_version": cp["checkpoint"].selector_version,
                "feature_version": cp["checkpoint"].feature_version,
                "stratum": cp["stratum"],
            }, default=str) + "\n")

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="experiments/daph_x/r12/r12_enriched_corpus.jsonl")
    parser.add_argument("--output", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    parser.add_argument("--k_values", default="2,4,6")
    parser.add_argument("--target_per_stratum", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = freeze_checkpoints_v2(
        REPO_ROOT / args.corpus,
        REPO_ROOT / args.output,
        [int(k) for k in args.k_values.split(",")],
        args.target_per_stratum,
        args.seed,
    )
    print(f"Frozen {len(selected)} v2 checkpoints to {args.output}")


if __name__ == "__main__":
    main()
