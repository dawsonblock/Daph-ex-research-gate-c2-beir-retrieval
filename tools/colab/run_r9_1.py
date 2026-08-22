"""R9.1: Uncensored reasoning-budget curve with per-request budget control.

Key improvements over R9:
  - ONE server stays resident for ALL budgets (no restarts)
  - Per-request thinking_budget_tokens controls reasoning budget
  - --reasoning-format deepseek captures reasoning_content separately
  - max_tokens=2048 ensures no truncation (finish_reason=length must be 0)
  - Reports reasoning tokens separately from answer tokens

Usage (on Colab, after llama.cpp is built):
    PYTHONPATH=. python3 tools/colab/run_r9_1.py \
        --model-path /content/models/LFM2.5-2.6B-Q5_K_M.gguf \
        --llama-server /content/llama.cpp/build/bin/llama-server \
        --output /content/r9_1_results.json \
        --parallel 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from hrm_adaptive_memory.executive.model_decoder import decode_output

from tools.colab.r9_reasoning_budget import (
    SYSTEM_PROMPT, make_test_states, build_user_prompt,
)


def start_server(llama_server_bin: str, model_path: str,
                 parallel: int, port: int) -> subprocess.Popen:
    """Start ONE llama-server with no fixed reasoning budget.

    Uses --reasoning-format deepseek to capture reasoning_content.
    Per-request thinking_budget_tokens controls the reasoning budget.
    """
    # Kill any existing server on this port
    subprocess.run("pkill -f llama-server 2>/dev/null || true", shell=True)
    time.sleep(2)

    cmd = [
        llama_server_bin, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "99", "-fa", "on",
        "--parallel", str(parallel), "--cont-batching",
        "--ctx-size", "4096", "--batch-size", "2048", "--ubatch-size", "512",
        "--temp", "0.0", "--seed", "42", "--threads", "4",
        "--reasoning-format", "deepseek",
        "-lv", "1",
    ]
    print(f"  Starting server: parallel={parallel}, reasoning-format=deepseek", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for i in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            print(f"  Server ready after {i}s", flush=True)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"Server failed to start within 120s")


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server stopped", flush=True)


def run_single_request(port: int, state: dict, reasoning_budget: int,
                       max_tokens: int) -> dict:
    """Run a single inference request with per-request thinking_budget_tokens."""
    user_prompt = build_user_prompt(state)
    t0 = time.time()

    req_data = json.dumps({
        "model": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_budget_tokens": reasoning_budget,
    }).encode()

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        latency_ms = int((time.time() - t0) * 1000)

        choice = result["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        reasoning_content = msg.get("reasoning_content", "") or ""
        finish_reason = choice.get("finish_reason")
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if usage.get("completion_tokens_details") else 0

        # Estimate reasoning tokens from reasoning_content if usage doesn't report it
        if reasoning_tokens == 0 and reasoning_content:
            # Rough estimate: ~4 chars per token
            reasoning_tokens = len(reasoning_content) // 4

        decoded = decode_output(content, strict=True)
        action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
        correct = (action == state["expected_action"])

        # Compute hashes for provenance
        request_hash = hashlib.sha256(req_data).hexdigest()[:16]
        response_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        return {
            "state_id": state["id"],
            "representation": state["representation"],
            "expected_action": state["expected_action"],
            "actual_action": action,
            "correct": correct,
            "decoder_valid": decoded.valid,
            "finish_reason": finish_reason,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": completion_tokens - reasoning_tokens,
            "latency_ms": latency_ms,
            "reasoning_budget": reasoning_budget,
            "request_hash": request_hash,
            "response_hash": response_hash,
            "raw_output": content[:200] if content else "",
            "reasoning_content_preview": reasoning_content[:200] if reasoning_content else "",
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
            "answer_tokens": 0,
            "latency_ms": int((time.time() - t0) * 1000),
            "reasoning_budget": reasoning_budget,
            "request_hash": "",
            "response_hash": "",
            "raw_output": str(e)[:200],
            "reasoning_content_preview": "",
        }


def compute_metrics(results: list[dict], reasoning_budget: int, max_tokens: int) -> dict:
    """Compute qualification metrics."""
    n = len(results)
    decoder_success = sum(1 for r in results if r["decoder_valid"]) / n
    fail_closed = sum(1 for r in results if r["actual_action"] == "FAIL_CLOSED") / n
    core_accuracy = sum(1 for r in results if r["correct"]) / n
    length_failures = sum(1 for r in results if r["finish_reason"] == "length") / n
    mean_tokens = sum(r["completion_tokens"] for r in results) / n
    mean_reasoning = sum(r["reasoning_tokens"] for r in results) / n
    mean_answer = sum(r["answer_tokens"] for r in results) / n
    mean_latency = sum(r["latency_ms"] for r in results) / n

    action_acc = {}
    for action in ["DEFER", "ANSWER", "VERIFY", "RETRIEVE", "SEARCH_MORE", "REASON_MORE"]:
        relevant = [r for r in results if r["expected_action"] == action]
        if relevant:
            action_acc[action] = sum(1 for r in relevant if r["correct"]) / len(relevant)

    a1_results = [r for r in results if r["representation"] == "A1"]
    m3_results = [r for r in results if r["representation"] == "M3"]
    a1_acc = sum(1 for r in a1_results if r["correct"]) / len(a1_results) if a1_results else 0
    m3_acc = sum(1 for r in m3_results if r["correct"]) / len(m3_results) if m3_results else 0
    a1_m3_asymmetry = abs(a1_acc - m3_acc)

    non_defer_expected = [r for r in results if r["expected_action"] != "DEFER"]
    non_defer_to_defer = sum(1 for r in non_defer_expected if r["actual_action"] == "DEFER") / len(non_defer_expected) if non_defer_expected else 0

    metrics = {
        "reasoning_budget": reasoning_budget,
        "max_tokens": max_tokens,
        "n_states": n,
        "decoder_success": round(decoder_success, 4),
        "fail_closed_rate": round(fail_closed, 4),
        "core_action_accuracy": round(core_accuracy, 4),
        "length_failure_rate": round(length_failures, 4),
        "mean_completion_tokens": round(mean_tokens, 1),
        "mean_reasoning_tokens": round(mean_reasoning, 1),
        "mean_answer_tokens": round(mean_answer, 1),
        "mean_latency_ms": round(mean_latency, 1),
        "per_action_accuracy": {k: round(v, 4) for k, v in action_acc.items()},
        "a1_accuracy": round(a1_acc, 4),
        "m3_accuracy": round(m3_acc, 4),
        "a1_m3_asymmetry_pp": round(a1_m3_asymmetry * 100, 1),
        "non_defer_to_defer_rate": round(non_defer_to_defer, 4),
        "results": results,
    }

    print(f"\n  Summary (budget={reasoning_budget}):", flush=True)
    print(f"    Decoder success:      {decoder_success:.1%}", flush=True)
    print(f"    Core action accuracy: {core_accuracy:.1%}", flush=True)
    print(f"    Fail-closed rate:     {fail_closed:.1%}", flush=True)
    print(f"    Length failures:      {length_failures:.1%}", flush=True)
    print(f"    Mean total tokens:    {mean_tokens:.0f}", flush=True)
    print(f"    Mean reasoning tokens:{mean_reasoning:.0f}", flush=True)
    print(f"    Mean answer tokens:   {mean_answer:.0f}", flush=True)
    print(f"    Mean latency:         {mean_latency:.0f}ms", flush=True)
    print(f"    A1 accuracy:          {a1_acc:.1%}", flush=True)
    print(f"    M3 accuracy:          {m3_acc:.1%}", flush=True)
    print(f"    Per-action: {action_acc}", flush=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="R9.1: Uncensored reasoning-budget curve")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="256,384,512,768,1024",
                        help="Comma-separated reasoning budgets")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens — high enough to prevent truncation")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    test_states = make_test_states()
    print(f"Test states: {len(test_states)}", flush=True)
    print(f"Budgets: {budgets}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Parallel: {args.parallel}", flush=True)
    print(f"Total requests: {len(budgets) * len(test_states)}", flush=True)
    print(f"Mode: per-request thinking_budget_tokens (single server)", flush=True)

    # Start ONE server for all budgets
    proc = start_server(args.llama_server, args.model_path, args.parallel, args.port)
    try:
        # Warmup
        print("  Warmup...", flush=True)
        warmup_data = json.dumps({
            "model": "test",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(test_states[0])},
            ],
            "max_tokens": 100, "temperature": 0,
            "thinking_budget_tokens": 0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=warmup_data, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60)
        print("  Warmup done", flush=True)

        all_results = []
        for budget in budgets:
            print(f"\n{'='*80}", flush=True)
            print(f"REASONING BUDGET = {budget} (per-request, parallel={args.parallel})", flush=True)
            print(f"{'='*80}", flush=True)

            t0 = time.time()
            results = []
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {
                    pool.submit(run_single_request, args.port, state, budget, args.max_tokens): state
                    for state in test_states
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "OK" if result["correct"] else "MISS"
                    print(f"  {result['state_id']:20s}  expected={result['expected_action']:12s}  "
                          f"got={result['actual_action']:12s}  {status}  "
                          f"tokens={result['completion_tokens']}  "
                          f"reason={result['reasoning_tokens']}  "
                          f"answer={result['answer_tokens']}  "
                          f"latency={result['latency_ms']}ms  "
                          f"finish={result['finish_reason']}", flush=True)

            wall_time = time.time() - t0
            results.sort(key=lambda r: r["state_id"])
            metrics = compute_metrics(results, budget, args.max_tokens)
            metrics["wall_time_s"] = round(wall_time, 2)
            all_results.append(metrics)
            print(f"  Wall time: {wall_time:.1f}s", flush=True)

    finally:
        stop_server(proc)

    # Compute action agreement vs highest budget with 100% decoder success
    valid_results = [r for r in all_results if r["decoder_success"] == 1.0 and r["length_failure_rate"] == 0.0]
    if valid_results:
        reference = max(valid_results, key=lambda r: r["reasoning_budget"])
        ref_actions = {r["state_id"]: r["actual_action"] for r in reference["results"]}
        ref_budget = reference["reasoning_budget"]
    else:
        reference = all_results[-1]
        ref_actions = {r["state_id"]: r["actual_action"] for r in reference["results"]}
        ref_budget = reference["reasoning_budget"]

    print(f"\n{'='*80}", flush=True)
    print(f"ACTION AGREEMENT vs {ref_budget} (reference, decoder={reference['decoder_success']:.0%})", flush=True)
    print(f"{'='*80}", flush=True)
    for metrics in all_results:
        budget = metrics["reasoning_budget"]
        if budget == ref_budget:
            agreement = 1.0
        else:
            agree = sum(1 for r in metrics["results"]
                       if r["actual_action"] == ref_actions.get(r["state_id"]))
            agreement = agree / len(metrics["results"])
        metrics["action_agreement_vs_reference"] = round(agreement, 4)
        metrics["reference_budget"] = ref_budget
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

    passing = [g for g in gate_results if g["all_pass"]]
    if passing:
        min_budget = min(g["budget"] for g in passing)
        print(f"\n  MINIMUM PASSING BUDGET: {min_budget}", flush=True)
    else:
        min_budget = None
        print(f"\n  NO BUDGET PASSED ALL GATES", flush=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "R9.1_uncensored_reasoning_budget_curve",
        "model": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
        "n_test_states": len(test_states),
        "budgets_tested": budgets,
        "max_tokens": args.max_tokens,
        "parallel_slots": args.parallel,
        "total_requests": len(budgets) * len(test_states),
        "per_request_budget": True,
        "reasoning_format": "deepseek",
        "results": all_results,
        "gate_results": gate_results,
        "minimum_passing_budget": min_budget,
        "reference_budget": ref_budget,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("R9.1 SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Budget':>7} | {'Decoder':>8} | {'Core Acc':>9} | {'Len Fail':>9} | "
          f"{'Tokens':>7} | {'Reason':>7} | {'Answer':>7} | {'Latency':>8} | {'Wall':>6} | {'Agree':>6}")
    print("-" * 105)
    for r in all_results:
        print(f"{r['reasoning_budget']:7d} | {r['decoder_success']:8.0%} | {r['core_action_accuracy']:9.0%} | "
              f"{r['length_failure_rate']:9.0%} | {r['mean_completion_tokens']:7.0f} | "
              f"{r['mean_reasoning_tokens']:7.0f} | {r['mean_answer_tokens']:7.0f} | "
              f"{r['mean_latency_ms']:8.0f}ms | {r.get('wall_time_s', 0):6.1f}s | "
              f"{r.get('action_agreement_vs_reference', 0):6.0%}")


if __name__ == "__main__":
    main()
