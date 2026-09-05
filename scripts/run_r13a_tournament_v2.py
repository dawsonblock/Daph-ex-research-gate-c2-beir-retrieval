#!/usr/bin/env python3
"""R13-A v2 resumable operator tournament.

Each execution gets a deterministic execution_id. Already-completed
executions are skipped on resume. Output is append-only JSONL of
execution receipts. Evaluation (correctness labels) is applied
afterward, outside the operator path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.operators.types import RuntimeState, Candidate, TrajectoryPoint, Observation
from daph_x.receipts.reasoning_checkpoint import ReasoningCheckpoint
from daph_x.backends.llama_cpp_backend import LlamaCppBackend, CognitiveBackend
from daph_x.operators.stop_v2 import StopV2
from daph_x.operators.sample_v2 import SampleStandardV2
from daph_x.operators.diverse_v2 import SampleDiverseV2
from daph_x.operators.critique_v2 import CritiqueRetryV2
from daph_x.operators.verify_v2 import VerifyTargetedV2


def _restore_runtime_state(state_dict: dict) -> RuntimeState:
    candidates = tuple(Candidate(
        candidate_id=c["candidate_id"],
        answer=c["answer"],
        reasoning_trace=c["reasoning_trace"],
        temperature=c["temperature"],
        seed=c["seed"],
        generation_index=c["generation_index"],
        metadata=c.get("metadata", {}),
    ) for c in state_dict["candidates"])

    trajectory = tuple(TrajectoryPoint(
        k=t["k"],
        top_answer=t["top_answer"],
        p_top1=t["p_top1"],
        p_top2=t["p_top2"],
        margin=t["margin"],
        entropy=t["entropy"],
        n_unique=t["n_unique"],
    ) for t in state_dict["trajectory"])

    return RuntimeState(
        checkpoint_id=state_dict["checkpoint_id"],
        task_id=state_dict["task_id"],
        task_prompt=state_dict["task_prompt"],
        answer_type=state_dict["answer_type"],
        category=state_dict["category"],
        difficulty=state_dict["difficulty"],
        candidates=candidates,
        trajectory=trajectory,
        k=state_dict["k"],
        current_answer=state_dict["current_answer"],
        observable_features=state_dict["observable_features"],
        state_hash=state_dict["state_hash"],
    )


def _restore_checkpoint(line: str) -> ReasoningCheckpoint:
    data = json.loads(line)
    runtime_state = _restore_runtime_state(data["runtime_state"])
    return ReasoningCheckpoint(
        checkpoint_id=data["checkpoint_id"],
        runtime_state=runtime_state,
        dataset_id=data["dataset_id"],
        corpus_sha256=data["corpus_sha256"],
        selector_version=data["selector_version"],
        feature_version=data["feature_version"],
    )


def _execution_id(checkpoint: ReasoningCheckpoint, operator_id: str, operator_version: str, replicate_id: int, backend_sha: str) -> str:
    s = f"{checkpoint.sha256()}|{operator_id}|{operator_version}|{replicate_id}|{backend_sha}"
    return hashlib.sha256(s.encode()).hexdigest()


def _load_completed_ids(output_path: Path) -> set:
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                completed.add(r["execution_id"])
            except Exception:
                pass
    return completed


def _run_operator(operator, state: RuntimeState, backend: CognitiveBackend, replicate_id: int) -> Observation:
    return operator.execute(state, backend, replicate_id)


def run_tournament_v2(
    checkpoints_path: Path,
    output_path: Path,
    operators: Sequence,
    backend: CognitiveBackend,
    backend_manifest_hash: str,
    replicates: Sequence[int] = (42,),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_ids(output_path)
    print(f"Resuming: {len(completed)} executions already completed")

    checkpoints = []
    with open(checkpoints_path) as f:
        for line in f:
            checkpoints.append(_restore_checkpoint(line))

    n_total = len(checkpoints) * len(operators) * len(replicates)
    n_remaining = n_total - len(completed)
    print(f"Checkpoints: {len(checkpoints)}, operators: {len(operators)}, replicates: {len(replicates)}")
    print(f"Total: {n_total}, remaining: {n_remaining}")

    with open(output_path, "a", buffering=1) as f:
        for ci, checkpoint in enumerate(checkpoints):
            for op in operators:
                for replicate_id in replicates:
                    exec_id = _execution_id(checkpoint, op.operator_id, op.operator_version, replicate_id, backend_manifest_hash)
                    if exec_id in completed:
                        continue

                    if not op.is_admissible(checkpoint.runtime_state):
                        receipt = {
                            "execution_id": exec_id,
                            "checkpoint_hash": checkpoint.sha256(),
                            "task_id": checkpoint.runtime_state.task_id,
                            "k": checkpoint.runtime_state.k,
                            "operator_id": op.operator_id,
                            "operator_version": op.operator_version,
                            "replicate_id": replicate_id,
                            "admissible": False,
                            "status": "SKIPPED",
                        }
                        f.write(json.dumps(receipt, default=str) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        continue

                    hash_before = checkpoint.runtime_state.sha256()
                    t0 = time.monotonic()
                    try:
                        obs = _run_operator(op, checkpoint.runtime_state, backend, replicate_id)
                        status = "SUCCESS"
                        error_code = None
                        error_message = None
                    except Exception as e:
                        import traceback
                        obs = None
                        status = "ERROR"
                        error_code = type(e).__name__
                        error_message = str(e)
                    wall_ms = (time.monotonic() - t0) * 1000
                    hash_after = checkpoint.runtime_state.sha256()

                    receipt = {
                        "execution_id": exec_id,
                        "checkpoint_hash": checkpoint.sha256(),
                        "state_hash_before": hash_before,
                        "state_hash_after": hash_after,
                        "state_unchanged": hash_before == hash_after,
                        "task_id": checkpoint.runtime_state.task_id,
                        "k": checkpoint.runtime_state.k,
                        "operator_id": op.operator_id,
                        "operator_version": op.operator_version,
                        "replicate_id": replicate_id,
                        "admissible": True,
                        "status": status,
                        "error_code": error_code,
                        "error_message": error_message,
                        "wall_ms": wall_ms,
                        "backend_model_id": backend.model_id,
                        "backend_model_sha256": backend.model_sha256,
                    }

                    if obs:
                        receipt.update({
                            "candidate_answer": obs.candidate_answer,
                            "reasoning_trace": obs.reasoning_trace[:500],
                            "confidence": obs.confidence,
                            "verification_score": obs.verification_score,
                            "evidence": obs.evidence,
                            "cost": obs.cost,
                            "metadata": obs.metadata,
                            "success": obs.success,
                            "failure_reason": obs.failure_reason,
                        })

                    f.write(json.dumps(receipt, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                    if ci % 10 == 0:
                        print(f"  checkpoint {ci+1}/{len(checkpoints)} {op.operator_id} r={replicate_id} {status}")

    print(f"Tournament v2 complete: {n_remaining} new executions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="experiments/daph_x/r13/v2/checkpoints.jsonl")
    parser.add_argument("--output", default="experiments/daph_x/r13/v2/executions.jsonl")
    parser.add_argument("--replicates", default="42")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--gpu_layers", type=int, default=-1)
    args = parser.parse_args()

    cp_path = REPO_ROOT / args.checkpoints
    out_path = REPO_ROOT / args.output
    replicates = [int(x) for x in args.replicates.split(",")]

    backend = LlamaCppBackend(model_path=args.model_path, n_gpu_layers=args.gpu_layers)
    backend_manifest = {
        "model_id": backend.model_id,
        "model_sha256": backend.model_sha256,
        "gpu_layers": args.gpu_layers,
    }
    backend_manifest_hash = hashlib.sha256(
        json.dumps(backend_manifest, sort_keys=True).encode()
    ).hexdigest()

    operators = [
        StopV2(),
        SampleStandardV2(),
        SampleDiverseV2(),
        CritiqueRetryV2(),
        VerifyTargetedV2(),
    ]

    run_tournament_v2(cp_path, out_path, operators, backend, backend_manifest_hash, replicates)


if __name__ == "__main__":
    main()
