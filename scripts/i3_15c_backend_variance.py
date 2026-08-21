"""Backend response variance characterization for I3.15c.

Takes 20 representative serialized requests (5 per stratum) and runs each
K times against the same backend. Records whether the decoded action,
target_id, and reason_code are stable across repeated identical requests.

Usage:
    OPENROUTER_API_KEY=... PYTHONPATH=. python3 scripts/i3_15c_backend_variance.py \
        --backend openrouter \
        --model nvidia/nemotron-3.5-lightning:free \
        --k 5 \
        --output experiments/v2b_i3_15c/closure/backend_variance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def build_representative_requests(n_per_stratum: int = 5, seed: int = 42):
    """Build 20 representative requests: 5 per stratum."""
    from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
        generate_i3_15c_corpus, get_i3_15c_corpus,
    )
    from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
        DeterministicRelationExtractor,
    )
    from scripts.run_i3_12j_factorial import make_inferred_snapshot_builder
    from scripts.run_i3_15_r1_balanced import (
        build_retrieved_evidence_task, adapt_local_system_prompt, TOP_K,
    )
    from scripts.run_i3_7e_compact_governor import (
        build_baseline_with_affordances_packet,
        BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT,
        evidence_packet_json,
    )
    from hrm_adaptive_memory.memory.chunking import Chunk
    from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever
    from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
    from hrm_adaptive_memory.executive.evidence_benchmark.schema import (
        initial_evidence_runtime,
    )

    tasks = generate_i3_15c_corpus(n_per_cell=10, seed=seed)
    corpus_passages = get_i3_15c_corpus()
    chunks = [
        Chunk(
            chunk_id=p.passage_id, source_id=p.source, source_type="doc",
            title=p.passage_id, section="", content=p.text,
            token_count=len(p.text.split()),
        )
        for p in corpus_passages
    ]
    corpus_by_text = {p.text: p for p in corpus_passages}
    corpus_by_id = {p.passage_id: p for p in corpus_passages}

    retriever = build_retriever("Q3_RERANKED", chunks)

    strata = {
        "T2_CONFLICT_IMMEDIATE": [],
        "T2_CONFLICT_LATE": [],
        "DEFER_CONTROL": [],
        "ANSWER_CONTROL": [],
    }
    for t in tasks:
        cat = t.evidence_task.category
        for key in strata:
            if key.lower().replace("t2_conflict_", "t2_conflict_") in cat or key.lower() in cat:
                strata[key].append(t)
                break

    requests = []
    for stratum_name, stratum_tasks in strata.items():
        selected = stratum_tasks[:n_per_stratum]
        for task in selected:
            et = task.evidence_task
            retrieved = retriever.search(et.task_summary, top_k=TOP_K)
            retrieved_passages = [
                corpus_by_id[c.chunk_id] for c, _ in retrieved
                if c.chunk_id in corpus_by_id
            ]
            new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)
            budget = ResourceBudget(
                max_executive_steps=10, max_retrieval_calls=3,
                max_search_calls=2, max_verification_calls=5,
            )
            runtime = initial_evidence_runtime(new_et, ResourceState(budget))
            snapshot = make_inferred_snapshot_builder(DeterministicRelationExtractor())(runtime)
            packet = build_baseline_with_affordances_packet(snapshot)
            sys_prompt = adapt_local_system_prompt(BASELINE_WITH_AFFORDANCES_SYSTEM_PROMPT)
            user_prompt = evidence_packet_json(packet)

            req_hash = hashlib.sha256(
                (sys_prompt + "||" + user_prompt).encode()
            ).hexdigest()

            requests.append({
                "task_id": et.task_id,
                "stratum": stratum_name,
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "request_sha256": req_hash,
            })

    return requests


def call_backend(
    backend,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Make a single backend call and return structured result.
    Retries on HTTP 429 with exponential backoff."""
    import urllib.error
    t0 = time.time()
    for attempt in range(max_retries):
        try:
            result = backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            latency_ms = (time.time() - t0) * 1000
            from hrm_adaptive_memory.executive.model_decoder import decode_output
            outcome = decode_output(result.raw_output, strict=True)
            return {
                "raw_output": result.raw_output,
                "decoded_action": outcome.proposal.action.value if outcome.proposal else None,
                "decoded_reason_code": outcome.proposal.reason_code if outcome.proposal else None,
                "decoded_target_id": outcome.proposal.target_id if outcome.proposal else None,
                "decoder_valid": outcome.valid,
                "finish_reason": result.finish_reason,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "latency_ms": round(latency_ms, 1),
                "error": None,
                "attempts": attempt + 1,
            }
        except RuntimeError as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = (2 ** attempt) * 5 + (hash(system_prompt) % 3)
                time.sleep(wait)
                continue
            latency_ms = (time.time() - t0) * 1000
            return {
                "raw_output": "",
                "decoded_action": None,
                "decoded_reason_code": None,
                "decoded_target_id": None,
                "decoder_valid": False,
                "finish_reason": None,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "latency_ms": round(latency_ms, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt + 1,
            }
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            return {
                "raw_output": "",
                "decoded_action": None,
                "decoded_reason_code": None,
                "decoded_target_id": None,
                "decoder_valid": False,
                "finish_reason": None,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "latency_ms": round(latency_ms, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": attempt + 1,
            }


def main():
    parser = argparse.ArgumentParser(description="Backend variance characterization")
    parser.add_argument("--backend", default="openrouter")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning:free")
    parser.add_argument("--k", type=int, default=5, help="Repetitions per request")
    parser.add_argument("--n-per-stratum", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building representative requests...")
    requests = build_representative_requests(args.n_per_stratum, args.seed)
    print(f"Built {len(requests)} requests ({args.n_per_stratum} per stratum)")

    # Create backend
    if args.backend == "openrouter":
        from scripts.run_i3_15c_factorial import OpenRouterBackend
        backend = OpenRouterBackend(model_name=args.model)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Build work items: (request_idx, repetition_idx, request)
    work_items = []
    for ri, req in enumerate(requests):
        for ki in range(args.k):
            work_items.append((ri, ki, req))

    print(f"Running {len(work_items)} calls ({len(requests)} requests x {args.k} reps)...")
    print(f"Using {args.n_workers} parallel workers")

    results_by_request: dict[int, list[dict]] = {i: [] for i in range(len(requests))}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {}
        for ri, ki, req in work_items:
            future = executor.submit(
                call_backend, backend,
                req["system_prompt"], req["user_prompt"],
                args.max_tokens,
            )
            futures[future] = (ri, ki)

        completed = 0
        for future in as_completed(futures):
            ri, ki = futures[future]
            result = future.result()
            result["repetition"] = ki
            results_by_request[ri].append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  {completed}/{len(work_items)} calls done "
                      f"({time.time() - t_start:.0f}s)")

    print(f"All {len(work_items)} calls done in {time.time() - t_start:.0f}s")

    # Analyze variance
    analysis = {
        "backend": args.backend,
        "model": args.model,
        "k": args.k,
        "n_requests": len(requests),
        "max_tokens": args.max_tokens,
        "total_calls": len(work_items),
        "wall_time_s": round(time.time() - t_start, 1),
        "per_request": [],
        "summary": {},
    }

    action_stable = 0
    action_unstable = 0
    terminal_stable = 0
    terminal_unstable = 0
    all_valid = 0
    any_invalid = 0

    for ri, req in enumerate(requests):
        reps = results_by_request[ri]
        actions = [r["decoded_action"] for r in reps if r["decoder_valid"]]
        targets = [r["decoded_target_id"] for r in reps if r["decoder_valid"]]
        reasons = [r["decoded_reason_code"] for r in reps if r["decoder_valid"]]
        valid_count = sum(1 for r in reps if r["decoder_valid"])
        invalid_count = len(reps) - valid_count

        # Action stability: all same action?
        unique_actions = set(actions)
        action_is_stable = len(unique_actions) <= 1 and len(actions) > 0

        # Terminal stability: would the terminal action be the same?
        # (We only have step-0 action, so "terminal" here means the first action)
        terminal_is_stable = action_is_stable

        if action_is_stable:
            action_stable += 1
        else:
            action_unstable += 1

        if terminal_is_stable:
            terminal_stable += 1
        else:
            terminal_unstable += 1

        if invalid_count == 0:
            all_valid += 1
        else:
            any_invalid += 1

        latencies = [r["latency_ms"] for r in reps if r["latency_ms"] > 0]
        token_counts = [r["completion_tokens"] for r in reps if r["completion_tokens"]]

        analysis["per_request"].append({
            "task_id": req["task_id"],
            "stratum": req["stratum"],
            "request_sha256": req["request_sha256"],
            "k": len(reps),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "unique_actions": sorted(list(unique_actions)),
            "action_is_stable": action_is_stable,
            "actions_by_rep": actions,
            "targets_by_rep": targets,
            "reasons_by_rep": reasons,
            "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "latency_stdev_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0,
            "token_mean": round(statistics.mean(token_counts), 1) if token_counts else 0,
            "token_stdev": round(statistics.stdev(token_counts), 1) if len(token_counts) > 1 else 0,
            "errors": [r["error"] for r in reps if r["error"]],
        })

    n = len(requests)
    analysis["summary"] = {
        "P_action_instability": round(action_unstable / n, 4),
        "P_terminal_instability": round(terminal_unstable / n, 4),
        "P_any_invalid": round(any_invalid / n, 4),
        "action_stable_count": action_stable,
        "action_unstable_count": action_unstable,
        "all_valid_count": all_valid,
        "any_invalid_count": any_invalid,
        "n_requests": n,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("BACKEND RESPONSE VARIANCE CHARACTERIZATION")
    print("=" * 80)
    print(f"\nBackend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Requests: {n}, Repetitions: {args.k}, Total calls: {len(work_items)}")
    print(f"\nSummary:")
    print(f"  P(action instability | identical request): {analysis['summary']['P_action_instability']:.4f}")
    print(f"  P(terminal instability | identical request): {analysis['summary']['P_terminal_instability']:.4f}")
    print(f"  P(any invalid | identical request): {analysis['summary']['P_any_invalid']:.4f}")
    print(f"  Action stable: {action_stable}/{n}")
    print(f"  All valid: {all_valid}/{n}")

    print("\nPer-request details:")
    for req_info in analysis["per_request"]:
        status = "STABLE" if req_info["action_is_stable"] else "UNSTABLE"
        print(f"  {req_info['task_id']} ({req_info['stratum']}): "
              f"actions={req_info['unique_actions']} "
              f"valid={req_info['valid_count']}/{req_info['k']} "
              f"latency={req_info['latency_mean_ms']}±{req_info['latency_stdev_ms']}ms "
              f"tokens={req_info['token_mean']}±{req_info['token_stdev']} "
              f"{status}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
