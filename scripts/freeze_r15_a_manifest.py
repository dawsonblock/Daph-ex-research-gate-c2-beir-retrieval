#!/usr/bin/env python3
"""Freeze the R15-A confirmation corpus manifest.

419 held-out R12 tasks (not in R13 checkpoints).
One checkpoint per task.
K assigned deterministically using hash(task_id) mod 3 -> {2, 4, 6}.

This manifest is committed BEFORE any R15-A COT inference.
No methodological changes after this point.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from daph_x.operators.types import Candidate, TrajectoryPoint, RuntimeState
from daph_x.receipts.reasoning_checkpoint import ReasoningCheckpoint
from daph_x.evaluation.r12_selector import select_r12_maxcal
from daph_x.coding.reasoning_tasks import get_reasoning_task

R12_CORPUS = PROJECT_ROOT / "experiments/daph_x/r12/r12_enriched_corpus.jsonl"
R13_CHECKPOINTS = PROJECT_ROOT / "experiments/daph_x/r13/v2/checkpoints.jsonl"
R15_DIR = PROJECT_ROOT / "experiments/daph_x/r15"
MANIFEST_PATH = R15_DIR / "r15_a_confirmation_manifest.jsonl"

K_VALUES = [2, 4, 6]


def deterministic_k(task_id: str) -> int:
    """Assign K deterministically: hash(task_id) mod 3 -> {2, 4, 6}."""
    h = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % 3
    return K_VALUES[idx]


def build_prefix_trajectory(candidates: list, k_values: list) -> list:
    from collections import Counter
    import math
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
            k=k, top_answer=top_answer, p_top1=p_top1, p_top2=p_top2,
            margin=margin, entropy=entropy, n_unique=len(answer_counts),
        ))
    return points


def compute_observable_features(trajectory: list, target_k: int) -> dict:
    import math
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # Load R13 task IDs (to exclude)
    with open(R13_CHECKPOINTS) as f:
        r13_task_ids = set(json.loads(l)["runtime_state"]["task_id"] for l in f)
    print(f"R13 task IDs to exclude: {len(r13_task_ids)}")

    # Load R12 corpus
    corpus_sha256 = sha256_file(R12_CORPUS)
    all_tasks = []
    with open(R12_CORPUS) as f:
        for line in f:
            t = json.loads(line)
            if t["task_id"] not in r13_task_ids:
                if len(t.get("candidates", [])) >= 6:
                    all_tasks.append(t)
    print(f"Held-out R12 tasks with >= 6 candidates: {len(all_tasks)}")

    # Assign K deterministically and build checkpoints
    manifest = []
    k_counts = Counter()
    for task in all_tasks:
        task_id = task["task_id"]
        k = deterministic_k(task_id)
        k_counts[k] += 1

        cands_raw = task["candidates"]
        cands = [
            Candidate(
                candidate_id=f"{task_id}_c{i}",
                answer=c["answer"],
                reasoning_trace=c.get("response", ""),
                temperature=c.get("temperature", 0.0),
                seed=c.get("seed", 0),
                generation_index=i,
                metadata={"temperature": c.get("temperature"), "gen_latency_ms": c.get("gen_latency_ms")},
            )
            for i, c in enumerate(cands_raw)
        ]

        trajectory = build_prefix_trajectory(cands_raw, K_VALUES)
        reasoning_task = get_reasoning_task(task_id)
        task_prompt = reasoning_task.prompt if reasoning_task else task.get("description", "")

        prefix_cands = cands[:k]
        selection = select_r12_maxcal(prefix_cands)
        features = compute_observable_features(trajectory, k)

        runtime_state = RuntimeState(
            checkpoint_id=f"{task_id}_k{k}",
            task_id=task_id,
            task_prompt=task_prompt,
            answer_type=task["answer_type"],
            category=task["category"],
            difficulty=task["difficulty"],
            candidates=tuple(prefix_cands),
            trajectory=tuple(trajectory[:k]),
            k=k,
            current_answer=selection.answer,
            observable_features=features,
            state_hash="",
        )
        state_hash = runtime_state.sha256()
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

        checkpoint = ReasoningCheckpoint(
            checkpoint_id=runtime_state.checkpoint_id,
            runtime_state=runtime_state,
            dataset_id="r12_corpus_v1",
            corpus_sha256=corpus_sha256,
            selector_version="r12-selector-v1",
            feature_version="r13-state-v2",
        )

        rs_dict = json.loads(runtime_state.canonical_bytes())
        rs_dict["state_hash"] = state_hash

        manifest.append({
            "checkpoint_id": checkpoint.checkpoint_id,
            "task_id": task_id,
            "k": k,
            "state_hash": state_hash,
            "checkpoint_sha256": checkpoint.sha256(),
            "runtime_state": rs_dict,
            "dataset_id": "r12_corpus_v1",
            "corpus_sha256": corpus_sha256,
            "selector_version": "r12-selector-v1",
            "feature_version": "r13-state-v2",
        })

    # Sort by checkpoint_id for determinism
    manifest.sort(key=lambda x: x["checkpoint_id"])

    # Write manifest
    R15_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Total checkpoints: {len(manifest)}")
    print(f"K distribution: {dict(sorted(k_counts.items()))}")

    # Verify determinism
    for entry in manifest:
        expected_k = deterministic_k(entry["task_id"])
        assert entry["k"] == expected_k, f"K mismatch for {entry['task_id']}"
    print("Determinism check: PASSED")

    # Category and difficulty distribution
    cats = Counter(e["runtime_state"]["category"] for e in manifest)
    diffs = Counter(e["runtime_state"]["difficulty"] for e in manifest)
    types = Counter(e["runtime_state"]["answer_type"] for e in manifest)
    print(f"Categories: {dict(sorted(cats.items()))}")
    print(f"Difficulties: {dict(sorted(diffs.items()))}")
    print(f"Answer types: {dict(sorted(types.items()))}")

    # Manifest hash
    manifest_bytes = Path(MANIFEST_PATH).read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Manifest SHA-256: {manifest_sha}")


if __name__ == "__main__":
    main()
