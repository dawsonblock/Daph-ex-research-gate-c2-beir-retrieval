"""R9 parallel: Reasoning-budget qualification with concurrent inference.

Architecture:
  - ONE llama-server stays resident for the entire experiment
  - For each reasoning budget, restart server with --reasoning-budget
  - Queue all 20 states concurrently using ThreadPoolExecutor
  - Workers = server slots (no over-subscription)
  - No retrieval models loaded — R9 consumes frozen serialized policy states only

Usage (on Colab, after llama.cpp is built):
    PYTHONPATH=. python3 tools/colab/run_r9_parallel.py \
        --model-path /content/models/LFM2.5-2.6B-Q5_K_M.gguf \
        --llama-server /content/llama.cpp/build/bin/llama-server \
        --output /content/r9_results.json \
        --parallel 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output

# Import test states from the existing R9 script
from tools.colab.r9_reasoning_budget import (
    SYSTEM_PROMPT, make_test_states, build_user_prompt,
    _compute_metrics,
)


def start_server(llama_server_bin: str, model_path: str, reasoning_budget: int,
                 parallel: int, port: int) -> subprocess.Popen:
    """Start llama-server with --reasoning-budget and parallel slots."""
    cmd = [
        llama_server_bin, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "99", "-fa", "on",
        "--reasoning-budget", str(reasoning_budget),
        "--parallel", str(parallel), "--cont-batching",
        "--ctx-size", "4096", "--batch-size", "2048", "--ubatch-size", "512",
        "--temp", "0.0", "--seed", "42", "--threads", "4", "--no-mmap",
    ]
    print(f"  Starting server: reasoning_budget={reasoning_budget}, parallel={parallel}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            print(f"  Server ready", flush=True)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    raise RuntimeError(f"Server failed to start: {stderr[:500]}")


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server stopped", flush=True)


def run_single_request(backend: LocalLlamaBackend, state: dict,
                       reasoning_budget: int, max_tokens: int) -> dict:
    """Run a single inference request."""
    user_prompt = build_user_prompt(state)
    t0 = time.time()
    try:
        call = backend.generate(
            system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
            temperature=0.0, max_tokens=max_tokens,
        )
        decoded = decode_output(call.raw_output, strict=True)
        action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
        correct = (action == state["expected_action"])
        return {
            "state_id": state["id"],
            "representation": state["representation"],
            "expected_action": state["expected_action"],
            "actual_action": action,
            "correct": correct,
            "decoder_valid": decoded.valid,
            "finish_reason": call.finish_reason,
            "completion_tokens": call.completion_tokens,
            "reasoning_tokens": call.reasoning_tokens,
            "latency_ms": call.latency_ms,
            "reasoning_budget": reasoning_budget,
            "raw_output": call.raw_output[:200] if call.raw_output else "",
        }
    except Exception as e:
        return {
            "state_id": state["id"],
            "representation": state["representation"],
            "expected_action": state["expected_action"],
            "actual_action": "BACKEND_ERROR",
            "correct": False,
            "decoder_valid": False,
            "finish_reason": None,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "latency_ms": int((time.time() - t0) * 1000),
            "reasoning_budget": reasoning_budget,
            "raw_output": str(e)[:200],
        }


def run_budget_parallel(model_path: str, reasoning_budget: int, max_tokens: int,
                        test_states: list[dict], parallel: int, port: int,
                        llama_server_bin: str) -> dict:
    """Run qualification for a single reasoning budget with parallel inference."""
    print(f"\n{'='*80}", flush=True)
    print(f"REASONING BUDGET = {reasoning_budget} (parallel={parallel})", flush=True)
    print(f"{'='*80}", flush=True)

    proc = start_server(llama_server_bin, model_path, reasoning_budget, parallel, port)
    try:
        backend = LocalLlamaBackend(
            model_name="LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=300,
        )

        # Warmup
        print("  Warmup...", flush=True)
        _ = backend.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(test_states[0]),
            temperature=0.0, max_tokens=max_tokens,
        )

        # Run all states concurrently
        print(f"  Running {len(test_states)} states with {parallel} workers...", flush=True)
        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(run_single_request, backend, state, reasoning_budget, max_tokens): state
                for state in test_states
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "OK" if result["correct"] else "MISS"
                print(f"  {result['state_id']:20s}  expected={result['expected_action']:12s}  "
                      f"got={result['actual_action']:12s}  {status}  "
                      f"tokens={result['completion_tokens']}  reasoning={result['reasoning_tokens']}  "
                      f"latency={result['latency_ms']}ms", flush=True)

        wall_time = time.time() - t0
        print(f"  Wall time: {wall_time:.1f}s for {len(test_states)} requests", flush=True)

    finally:
        stop_server(proc)

    # Sort results by state_id for deterministic output
    results.sort(key=lambda r: r["state_id"])

    metrics = _compute_metrics(results, reasoning_budget, max_tokens)
    metrics["wall_time_s"] = round(wall_time, 2)
    metrics["parallel_slots"] = parallel
    return metrics


def main():
    parser = argparse.ArgumentParser(description="R9: Parallel reasoning-budget qualification")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="0,64,128,256,512,1024")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel server slots and client workers")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    test_states = make_test_states()
    print(f"Test states: {len(test_states)}", flush=True)
    print(f"Budgets: {budgets}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Parallel slots: {args.parallel}", flush=True)
    print(f"Total requests: {len(budgets) * len(test_states)}", flush=True)

    # Run qualification for each budget
    all_results = []
    for budget in budgets:
        metrics = run_budget_parallel(
            args.model_path, budget, args.max_tokens, test_states,
            args.parallel, args.port, args.llama_server
        )
        all_results.append(metrics)

    # Compute action agreement vs 1024 (reference)
    reference = next((r for r in all_results if r["reasoning_budget"] == 1024), all_results[-1])
    ref_actions = {r["state_id"]: r["actual_action"] for r in reference["results"]}

    print(f"\n{'='*80}", flush=True)
    print("ACTION AGREEMENT vs 1024 (reference)", flush=True)
    print(f"{'='*80}", flush=True)
    for metrics in all_results:
        budget = metrics["reasoning_budget"]
        if budget == 1024:
            agreement = 1.0
        else:
            agree = sum(1 for r in metrics["results"]
                       if r["actual_action"] == ref_actions.get(r["state_id"]))
            agreement = agree / len(metrics["results"])
        metrics["action_agreement_vs_1024"] = round(agreement, 4)
        print(f"  Budget {budget:5d}: agreement = {agreement:.1%}", flush=True)

    # Gate evaluation
    print(f"\n{'='*80}", flush=True)
    print("GATE EVALUATION", flush=True)
    print(f"{'='*80}", flush=True)
    gate_results = []
    for metrics in all_results:
        gates = {
            "decoder_success_100": metrics["decoder_success"] == 1.0,
            "fail_closed_0": metrics["fail_closed_rate"] == 0.0,
            "core_accuracy_80": metrics["core_action_accuracy"] >= 0.80,
            "length_failures_0": metrics["length_failure_rate"] == 0.0,
            "a1_m3_asymmetry_20pp": metrics["a1_m3_asymmetry_pp"] <= 20.0,
            "non_defer_to_defer_20": metrics["non_defer_to_defer_rate"] <= 0.20,
        }
        for action in ["DEFER", "ANSWER", "VERIFY", "RETRIEVE"]:
            acc = metrics["per_action_accuracy"].get(action, 0)
            gates[f"{action.lower()}_75"] = acc >= 0.75

        all_pass = all(gates.values())
        metrics["gates"] = gates
        metrics["all_gates_pass"] = all_pass
        gate_results.append({
            "budget": metrics["reasoning_budget"],
            "all_pass": all_pass,
            "failed_gates": [k for k, v in gates.items() if not v],
        })
        status = "PASS" if all_pass else "FAIL"
        fails = "(" + ", ".join(k for k, v in gates.items() if not v) + ")" if not all_pass else ""
        print(f"  Budget {metrics['reasoning_budget']:5d}: {status} {fails}", flush=True)

    # Find minimum passing budget
    passing = [g for g in gate_results if g["all_pass"]]
    if passing:
        min_budget = min(g["budget"] for g in passing)
        print(f"\n  MINIMUM PASSING BUDGET: {min_budget}", flush=True)
    else:
        min_budget = None
        print(f"\n  NO BUDGET PASSED ALL GATES", flush=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "R9a_parallel_reasoning_budget_qualification",
        "model": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
        "n_test_states": len(test_states),
        "budgets_tested": budgets,
        "max_tokens": args.max_tokens,
        "parallel_slots": args.parallel,
        "total_requests": len(budgets) * len(test_states),
        "results": all_results,
        "gate_results": gate_results,
        "minimum_passing_budget": min_budget,
        "reference_budget": 1024,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)

    # Print summary table
    print(f"\n{'='*80}", flush=True)
    print("R9a SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Budget':>7} | {'Decoder':>8} | {'Core Acc':>9} | {'Len Fail':>9} | "
          f"{'Tokens':>7} | {'Reason':>7} | {'Latency':>8} | {'Wall':>6} | {'Agree':>6}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['reasoning_budget']:7d} | {r['decoder_success']:8.0%} | {r['core_action_accuracy']:9.0%} | "
              f"{r['length_failure_rate']:9.0%} | {r['mean_completion_tokens']:7.0f} | "
              f"{r['mean_reasoning_tokens']:7.0f} | {r['mean_latency_ms']:8.0f}ms | "
              f"{r.get('wall_time_s', 0):6.1f}s | {r.get('action_agreement_vs_1024', 0):6.0%}")


if __name__ == "__main__":
    main()
