"""MODEL_POLICY_SHOOTOUT: Fast model qualification for the R13 policy role.

Runs the same 20 frozen qualification states against a candidate model
with a cheap staged funnel:

  Stage 1: 20 states, budget=128 → if core < 75%, EARLY_REJECT
  Stage 2: if promising, budgets 128, 256, 384
  Stage 3: only if >=80%, full qualification + repeats

Usage (on Colab, after llama.cpp is built):
    PYTHONPATH=. python3 tools/colab/policy_model_shootout.py \
        --model-path /content/models/LFM2.5-8B-A1B-Q5_K_M.gguf \
        --model-name "LiquidAI/LFM2.5-8B-A1B-GGUF:Q5_K_M" \
        --llama-server /content/llama.cpp/build/bin/llama-server \
        --output /content/shootout_results.json \
        --budget 128

For multi-budget:
    ... --budgets 128,256,384
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

# JSON schema for response_format — same as R9.1
ACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ANSWER", "RETRIEVE", "VERIFY",
                             "SEARCH_MORE", "REASON_MORE",
                             "DEFER", "STOP"],
                },
                "reason_code": {
                    "type": "string",
                    "pattern": "^[A-Z][A-Z0-9_]*$",
                },
                "target_id": {"type": ["string", "null"]},
            },
            "required": ["action", "reason_code", "target_id"],
            "additionalProperties": False,
        },
    },
}

# Qualification gates
PROMOTION_GATES = {
    "decoder_success_100": lambda m: m["decoder_success"] == 1.0,
    "fail_closed_0": lambda m: m["fail_closed_rate"] == 0.0,
    "core_accuracy_80": lambda m: m["core_action_accuracy"] >= 0.80,
    "length_failures_0": lambda m: m["length_failure_rate"] == 0.0,
    "a1_m3_asymmetry_20pp": lambda m: m["a1_m3_asymmetry_pp"] <= 20.0,
    "non_defer_to_defer_20": lambda m: m["non_defer_to_defer_rate"] <= 0.20,
    "defer_75": lambda m: m["per_action_accuracy"].get("DEFER", 0) >= 0.75,
    "answer_75": lambda m: m["per_action_accuracy"].get("ANSWER", 0) >= 0.75,
    "verify_75": lambda m: m["per_action_accuracy"].get("VERIFY", 0) >= 0.75,
    "retrieve_75": lambda m: m["per_action_accuracy"].get("RETRIEVE", 0) >= 0.75,
}

EARLY_GATE_THRESHOLD = 0.75


def start_server(llama_server_bin: str, model_path: str,
                 parallel: int, port: int) -> subprocess.Popen:
    """Start ONE llama-server with per-request budget support."""
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
    print(f"  Starting server: parallel={parallel}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for i in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            print(f"  Server ready after {i}s", flush=True)
            return proc
        except Exception:
            if i > 0 and i % 30 == 0:
                if proc.poll() is not None:
                    output = proc.stdout.read().decode() if proc.stdout else ""
                    print(f"  Server died: {output[:500]}", flush=True)
                    raise RuntimeError(f"Server died: {output[:300]}")
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"Server failed to start within 180s")


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server stopped", flush=True)


def run_single_request(port: int, state: dict, reasoning_budget: int,
                       max_tokens: int, model_name: str) -> dict:
    """Run a single inference request with per-request thinking_budget_tokens."""
    user_prompt = build_user_prompt(state)
    t0 = time.time()

    req_data = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking_budget_tokens": reasoning_budget,
        "response_format": ACTION_SCHEMA,
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
        reasoning_tokens_reported = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if usage.get("completion_tokens_details") else 0

        # Independent estimates (not subtracted from completion_tokens)
        reasoning_tokens_estimated = len(reasoning_content) // 4 if reasoning_content else 0
        answer_tokens_estimated = len(content) // 4 if content else 0

        decoded = decode_output(content, strict=True)
        action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
        correct = (action == state["expected_action"])

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
            "reasoning_tokens_reported": reasoning_tokens_reported,
            "reasoning_tokens_estimated": reasoning_tokens_estimated,
            "answer_tokens_estimated": answer_tokens_estimated,
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
            "reasoning_tokens_reported": 0,
            "reasoning_tokens_estimated": 0,
            "answer_tokens_estimated": 0,
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
    mean_reasoning = sum(r["reasoning_tokens_estimated"] for r in results) / n
    mean_answer = sum(r["answer_tokens_estimated"] for r in results) / n
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
        "mean_reasoning_tokens_estimated": round(mean_reasoning, 1),
        "mean_answer_tokens_estimated": round(mean_answer, 1),
        "mean_latency_ms": round(mean_latency, 1),
        "per_action_accuracy": {k: round(v, 4) for k, v in action_acc.items()},
        "a1_accuracy": round(a1_acc, 4),
        "m3_accuracy": round(m3_acc, 4),
        "a1_m3_asymmetry_pp": round(a1_m3_asymmetry * 100, 1),
        "non_defer_to_defer_rate": round(non_defer_to_defer, 4),
        "results": results,
    }

    # Evaluate gates
    gates = {name: check(metrics) for name, check in PROMOTION_GATES.items()}
    metrics["gates"] = gates
    metrics["all_gates_pass"] = all(gates.values())
    metrics["failed_gates"] = [k for k, v in gates.items() if not v]

    return metrics


def print_shootout_table(model_name: str, all_metrics: list[dict]):
    """Print the shootout comparison table."""
    print(f"\n{'='*120}", flush=True)
    print(f"MODEL POLICY SHOOTOUT: {model_name}", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"{'Budget':>7} | {'Decoder':>8} | {'Core Acc':>9} | {'Len Fail':>9} | "
          f"{'DEFER':>6} | {'ANSWER':>7} | {'VERIFY':>7} | {'RETRIEVE':>9} | "
          f"{'A1':>5} | {'M3':>5} | {'Asym':>5} | {'Latency':>8} | {'Status':>12}")
    print("-" * 130)
    for m in all_metrics:
        status = "PASS" if m["all_gates_pass"] else (
            "EARLY_REJECT" if m["core_action_accuracy"] < EARLY_GATE_THRESHOLD else "FAIL"
        )
        print(f"{m['reasoning_budget']:7d} | {m['decoder_success']:8.0%} | {m['core_action_accuracy']:9.0%} | "
              f"{m['length_failure_rate']:9.0%} | "
              f"{m['per_action_accuracy'].get('DEFER', 0):6.0%} | "
              f"{m['per_action_accuracy'].get('ANSWER', 0):7.0%} | "
              f"{m['per_action_accuracy'].get('VERIFY', 0):7.0%} | "
              f"{m['per_action_accuracy'].get('RETRIEVE', 0):9.0%} | "
              f"{m['a1_accuracy']:5.0%} | {m['m3_accuracy']:5.0%} | "
              f"{m['a1_m3_asymmetry_pp']:5.1f} | "
              f"{m['mean_latency_ms']:8.0f}ms | {status:>12}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MODEL_POLICY_SHOOTOUT")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True,
                        help="Human-readable model name for the results table")
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="Single budget for Stage 1 early gate")
    parser.add_argument("--budgets", default=None,
                        help="Comma-separated budgets for Stage 2+ (overrides --budget)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--states", default="v1",
                        help="State set: 'v1' (20 states) or 'v2' (80 expanded states)")
    args = parser.parse_args()

    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",")]
    elif args.budget is not None:
        budgets = [args.budget]
    else:
        budgets = [128]  # Default: Stage 1 early gate

    if args.states == "v2":
        from tools.colab.policy_qualification_v2_states import make_qualification_v2_states
        test_states = make_qualification_v2_states()
        print(f"Using POLICY_QUALIFICATION_V2 states ({len(test_states)} states)", flush=True)
    else:
        test_states = make_test_states()
        print(f"Using V1 states ({len(test_states)} states)", flush=True)
    print(f"MODEL: {args.model_name}", flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Test states: {len(test_states)}", flush=True)
    print(f"Budgets: {budgets}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Parallel: {args.parallel}", flush=True)
    print(f"Total requests: {len(budgets) * len(test_states)}", flush=True)

    # Start ONE server
    proc = start_server(args.llama_server, args.model_path, args.parallel, args.port)
    try:
        # Warmup
        print("  Warmup...", flush=True)
        warmup_data = json.dumps({
            "model": args.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(test_states[0])},
            ],
            "max_tokens": 100, "temperature": 0,
            "thinking_budget_tokens": 0,
            "response_format": ACTION_SCHEMA,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=warmup_data, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120)
        print("  Warmup done", flush=True)

        all_metrics = []
        early_reject = False

        for bi, budget in enumerate(budgets):
            print(f"\n{'='*80}", flush=True)
            print(f"BUDGET = {budget} (parallel={args.parallel})", flush=True)
            print(f"{'='*80}", flush=True)

            t0 = time.time()
            results = []
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {
                    pool.submit(run_single_request, args.port, state, budget,
                               args.max_tokens, args.model_name): state
                    for state in test_states
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "OK" if result["correct"] else "MISS"
                    print(f"  {result['state_id']:20s}  expected={result['expected_action']:12s}  "
                          f"got={result['actual_action']:12s}  {status}  "
                          f"tokens={result['completion_tokens']}  "
                          f"latency={result['latency_ms']}ms  "
                          f"finish={result['finish_reason']}", flush=True)

            wall_time = time.time() - t0
            results.sort(key=lambda r: r["state_id"])
            metrics = compute_metrics(results, budget, args.max_tokens)
            metrics["wall_time_s"] = round(wall_time, 2)
            all_metrics.append(metrics)

            print(f"\n  Summary (budget={budget}):", flush=True)
            print(f"    Decoder success:      {metrics['decoder_success']:.1%}", flush=True)
            print(f"    Core action accuracy: {metrics['core_action_accuracy']:.1%}", flush=True)
            print(f"    Fail-closed rate:     {metrics['fail_closed_rate']:.1%}", flush=True)
            print(f"    Length failures:      {metrics['length_failure_rate']:.1%}", flush=True)
            print(f"    Mean latency:         {metrics['mean_latency_ms']:.0f}ms", flush=True)
            print(f"    Wall time:            {wall_time:.1f}s", flush=True)
            print(f"    Per-action: {metrics['per_action_accuracy']}", flush=True)

            # Early gate check (only for first budget)
            if bi == 0 and len(budgets) > 1:
                if metrics["core_action_accuracy"] < EARLY_GATE_THRESHOLD:
                    print(f"\n  *** EARLY REJECT: core accuracy {metrics['core_action_accuracy']:.0%} < {EARLY_GATE_THRESHOLD:.0%} ***", flush=True)
                    early_reject = True
                    break

    finally:
        stop_server(proc)

    # Print shootout table
    print_shootout_table(args.model_name, all_metrics)

    # Final verdict
    print(f"\n{'='*80}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'='*80}", flush=True)

    if early_reject:
        verdict = "EARLY_REJECT"
        print(f"  {args.model_name}: EARLY_REJECT (core < {EARLY_GATE_THRESHOLD:.0%})", flush=True)
    else:
        any_pass = any(m["all_gates_pass"] for m in all_metrics)
        if any_pass:
            passing = [m for m in all_metrics if m["all_gates_pass"]]
            best = min(passing, key=lambda m: m["reasoning_budget"])
            verdict = "QUALIFIED"
            print(f"  {args.model_name}: QUALIFIED at budget={best['reasoning_budget']}", flush=True)
            print(f"  Core accuracy: {best['core_action_accuracy']:.0%}", flush=True)
            print(f"  All gates passed", flush=True)
        else:
            verdict = "FAIL"
            best = max(all_metrics, key=lambda m: m["core_action_accuracy"])
            print(f"  {args.model_name}: FAIL", flush=True)
            print(f"  Best core accuracy: {best['core_action_accuracy']:.0%} at budget={best['reasoning_budget']}", flush=True)
            print(f"  Failed gates: {best['failed_gates']}", flush=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "MODEL_POLICY_SHOOTOUT",
        "model": args.model_name,
        "model_path": args.model_path,
        "n_test_states": len(test_states),
        "budgets_tested": [m["reasoning_budget"] for m in all_metrics],
        "max_tokens": args.max_tokens,
        "parallel_slots": args.parallel,
        "early_gate_threshold": EARLY_GATE_THRESHOLD,
        "verdict": verdict,
        "results": all_metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
