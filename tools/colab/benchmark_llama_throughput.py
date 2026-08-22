"""Benchmark llama-server parallel slot throughput.

Tests N={1,2,4,8} parallel slots against the same 8 representative
production requests. Measures aggregate tokens/sec, req/sec, p50/p95
latency, and VRAM usage.

Usage (on Colab, after llama-server is built):
    PYTHONPATH=. python3 tools/colab/benchmark_llama_throughput.py \
        --model-path /content/models/LFM2.5-2.6B-Q5_K_M.gguf \
        --llama-server /content/llama.cpp/build/bin/llama-server \
        --output /content/throughput_benchmark.json
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

# Add repo to path
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output

SYSTEM_PROMPT = (
    "You are a metareasoning controller for a retrieval-verification task.\n"
    "You must choose one bounded action from the frozen seven-action vocabulary:\n"
    "  ANSWER, RETRIEVE, VERIFY, SEARCH_MORE, REASON_MORE, DEFER, STOP\n\n"
    "Respond with exactly one JSON object.\n"
    "No markdown. No explanation. No additional keys.\n"
    'Schema: {"action":"<ACTION>","reason_code":"<CODE>","target_id":null}\n'
    "Allowed ACTION values: ANSWER, DEFER, STOP, VERIFY, RETRIEVE, SEARCH_MORE, REASON_MORE\n\n"
    "ACTION SEMANTICS\n\n"
    "ANSWER: Provide the final answer when the currently verified evidence is sufficient.\n"
    "RETRIEVE: Expose additional evidence using the available retrieval mechanism.\n"
    "VERIFY: Verify a visible evidence item whose status has not yet been established.\n"
    "SEARCH_MORE: Search additional sources when evidence may be insufficient.\n"
    "REASON_MORE: Continue reasoning over currently available evidence.\n"
    "DEFER: Terminate because available evidence is insufficient.\n"
    "STOP: Terminate without answering for a non-epistemic execution reason.\n"
    "Do not use STOP merely because evidence is insufficient; use DEFER for that."
)


def make_benchmark_requests(n: int = 8) -> list[dict]:
    """Build 8 representative production requests covering different actions."""
    states = [
        {"id": "BENCH_ANSWER", "scenario": "H1 confirmed by SUFFICIENT evidence.",
         "evidence": [{"id": "E1", "proposition": "Service up.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "VIABLE"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "ELIMINATED"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False}},
        {"id": "BENCH_DEFER", "scenario": "All hypotheses eliminated.",
         "evidence": [{"id": "E1", "proposition": "Service up.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
                      {"id": "E2", "proposition": "Service down.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]}],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "ELIMINATED"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "ELIMINATED"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False}},
        {"id": "BENCH_VERIFY", "scenario": "Evidence UNVERIFIED, must check.",
         "evidence": [{"id": "E1", "proposition": "Probe says up.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []}],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "VIABLE"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "VIABLE"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True}},
        {"id": "BENCH_RETRIEVE", "scenario": "No evidence yet, retrieval available.",
         "evidence": [],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "VIABLE"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "VIABLE"}],
         "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False}},
        {"id": "BENCH_SEARCH", "scenario": "Initial evidence insufficient, search available.",
         "evidence": [{"id": "E1", "proposition": "Old report.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []}],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "VIABLE"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "VIABLE"}],
         "affordances": {"can_retrieve": False, "can_search": True, "can_verify": True}},
        {"id": "BENCH_REASON", "scenario": "Contradictory evidence, need reasoning.",
         "evidence": [{"id": "E1", "proposition": "Probe A: up.", "verified": True, "state": "MISSING", "supports": ["H1"], "contradicts": []},
                      {"id": "E2", "proposition": "Probe B: down.", "verified": True, "state": "MISSING", "supports": ["H2"], "contradicts": []}],
         "hypotheses": [{"id": "H1", "proposition": "Service operational.", "status": "VIABLE"},
                        {"id": "H2", "proposition": "Service not operational.", "status": "VIABLE"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False}},
        {"id": "BENCH_M3_DEFER", "scenario": "T2 fired, all eliminated, conflict.",
         "evidence": [{"id": "E1", "proposition": "LB returning 503.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
                      {"id": "E2", "proposition": "LB distributing evenly.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]}],
         "hypotheses": [{"id": "H1", "proposition": "LB operational.", "status": "ELIMINATED"},
                        {"id": "H2", "proposition": "LB not operational.", "status": "ELIMINATED"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
         "t2_fired": True, "representation": "M3"},
        {"id": "BENCH_M3_VERIFY", "scenario": "T2 not fired, one SUFFICIENT, one UNVERIFIED.",
         "evidence": [{"id": "E1", "proposition": "Gateway offline.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
                      {"id": "E2", "proposition": "Gateway online.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]}],
         "hypotheses": [{"id": "H1", "proposition": "Gateway operational.", "status": "ELIMINATED"},
                        {"id": "H2", "proposition": "Gateway not operational.", "status": "VIABLE"}],
         "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
         "t2_fired": False, "representation": "M3"},
    ]
    return states[:n]


def build_user_prompt(state: dict) -> str:
    evidence_str = []
    for ev in state.get("evidence", []):
        evidence_str.append({
            "evidence_id": ev["id"], "proposition": ev["proposition"],
            "verification_state": ev["state"], "verified": ev["verified"],
            "supports": ev.get("supports", []), "contradicts": ev.get("contradicts", []),
        })
    return json.dumps({
        "scenario": state["scenario"],
        "representation": state.get("representation", "A1"),
        "t2_fired": state.get("t2_fired", False),
        "action_affordances": state["affordances"],
        "hypotheses": [{"hypothesis_id": h["id"], "proposition": h["proposition"], "status": h["status"]} for h in state["hypotheses"]],
        "evidence_items": evidence_str,
        "prior_actions": [], "prior_outcomes": [],
        "resource_state": {"elapsed_ms": 0, "elapsed_ms_remaining": 30000},
    })


def start_server(llama_server_bin: str, model_path: str, parallel: int, port: int) -> subprocess.Popen:
    cmd = [
        llama_server_bin, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "99", "-fa", "on",
        "--parallel", str(parallel), "--cont-batching",
        "--ctx-size", "4096", "--batch-size", "2048", "--ubatch-size", "512",
        "--temp", "0.0", "--seed", "42", "--threads", "4", "--no-mmap",
        "--reasoning-budget", "0",
    ]
    print(f"  Starting server: parallel={parallel}, port={port}", flush=True)
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


def get_vram_mb() -> float:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception:
        return -1


def run_benchmark_parallel(backend: LocalLlamaBackend, requests: list[dict],
                           max_tokens: int, n_workers: int) -> list[dict]:
    """Run requests concurrently with n_workers threads."""
    def single_request(state: dict) -> dict:
        user_prompt = build_user_prompt(state)
        t0 = time.time()
        try:
            call = backend.generate(
                system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
                temperature=0.0, max_tokens=max_tokens,
            )
            latency = time.time() - t0
            decoded = decode_output(call.raw_output, strict=True)
            action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
            return {
                "state_id": state["id"],
                "action": action,
                "decoder_valid": decoded.valid,
                "latency_s": latency,
                "latency_ms": call.latency_ms,
                "completion_tokens": call.completion_tokens,
                "reasoning_tokens": call.reasoning_tokens,
                "finish_reason": call.finish_reason,
            }
        except Exception as e:
            return {
                "state_id": state["id"], "action": "ERROR", "decoder_valid": False,
                "latency_s": time.time() - t0, "latency_ms": int((time.time() - t0) * 1000),
                "completion_tokens": 0, "reasoning_tokens": 0, "finish_reason": None,
                "error": str(e)[:100],
            }

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(single_request, req): req for req in requests}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def benchmark_slot_config(llama_server_bin: str, model_path: str, parallel: int,
                          port: int, requests: list[dict], max_tokens: int) -> dict:
    """Benchmark a single parallel-slot configuration."""
    print(f"\n{'='*80}", flush=True)
    print(f"BENCHMARK: parallel={parallel}", flush=True)
    print(f"{'='*80}", flush=True)

    proc = start_server(llama_server_bin, model_path, parallel, port)
    try:
        # Warmup
        backend = LocalLlamaBackend(
            model_name="LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=300,
        )
        print("  Warmup request...", flush=True)
        _ = backend.generate(system_prompt=SYSTEM_PROMPT,
                            user_prompt=build_user_prompt(requests[0]),
                            temperature=0.0, max_tokens=max_tokens)

        vram_before = get_vram_mb()

        # Run benchmark: repeat requests to get enough samples
        n_repeats = 4
        all_requests = requests * n_repeats
        print(f"  Running {len(all_requests)} requests with {parallel} workers...", flush=True)

        t0 = time.time()
        results = run_benchmark_parallel(backend, all_requests, max_tokens, parallel)
        wall_time = time.time() - t0

        vram_after = get_vram_mb()

        # Compute metrics
        n = len(results)
        n_valid = sum(1 for r in results if r["decoder_valid"])
        n_errors = sum(1 for r in results if r["action"] == "ERROR")
        total_tokens = sum(r["completion_tokens"] for r in results)
        latencies = sorted([r["latency_s"] for r in results])

        mean_latency = sum(latencies) / n
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        req_per_sec = n / wall_time
        tokens_per_sec = total_tokens / wall_time

        metrics = {
            "parallel": parallel,
            "n_requests": n,
            "n_valid": n_valid,
            "n_errors": n_errors,
            "wall_time_s": round(wall_time, 2),
            "req_per_sec": round(req_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_tokens": total_tokens,
            "mean_latency_s": round(mean_latency, 3),
            "p50_latency_s": round(p50, 3),
            "p95_latency_s": round(p95, 3),
            "vram_mb_before": vram_before,
            "vram_mb_after": vram_after,
            "decoder_failures": n - n_valid,
        }

        print(f"  Results:", flush=True)
        print(f"    req/sec:         {req_per_sec:.2f}", flush=True)
        print(f"    tokens/sec:      {tokens_per_sec:.2f}", flush=True)
        print(f"    mean latency:    {mean_latency:.3f}s", flush=True)
        print(f"    p50 latency:     {p50:.3f}s", flush=True)
        print(f"    p95 latency:     {p95:.3f}s", flush=True)
        print(f"    VRAM:            {vram_after:.0f} MB", flush=True)
        print(f"    decoder failures:{n - n_valid}", flush=True)
        print(f"    wall time:       {wall_time:.2f}s", flush=True)

        return metrics

    finally:
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama-server throughput")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--output", default="/content/throughput_benchmark.json")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--slots", default="1,2,4,8",
                        help="Comma-separated parallel slot counts to test")
    args = parser.parse_args()

    slots = [int(s) for s in args.slots.split(",")]
    requests = make_benchmark_requests(8)
    print(f"Benchmark requests: {len(requests)}", flush=True)
    print(f"Slot configs: {slots}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)

    all_metrics = []
    for parallel in slots:
        metrics = benchmark_slot_config(
            args.llama_server, args.model_path, parallel,
            args.port, requests, args.max_tokens
        )
        all_metrics.append(metrics)

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("THROUGHPUT BENCHMARK SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Slots':>5} | {'req/s':>8} | {'tok/s':>8} | {'mean':>8} | {'p50':>8} | {'p95':>8} | {'VRAM':>8} | {'fails':>5}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['parallel']:5d} | {m['req_per_sec']:8.2f} | {m['tokens_per_sec']:8.2f} | "
              f"{m['mean_latency_s']:8.3f} | {m['p50_latency_s']:8.3f} | {m['p95_latency_s']:8.3f} | "
              f"{m['vram_mb_after']:8.0f} | {m['decoder_failures']:5d}")

    # Select best config
    valid = [m for m in all_metrics if m["decoder_failures"] == 0 and m["n_errors"] == 0]
    if valid:
        best = max(valid, key=lambda m: m["tokens_per_sec"])
        print(f"\n  BEST CONFIG: parallel={best['parallel']} "
              f"({best['tokens_per_sec']:.2f} tok/s, {best['req_per_sec']:.2f} req/s)")
    else:
        print(f"\n  NO VALID CONFIG (all had failures)")

    # Save
    output = {
        "test": "throughput_benchmark",
        "model": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
        "max_tokens": args.max_tokens,
        "n_benchmark_requests": len(requests),
        "n_repeats": 4,
        "results": all_metrics,
        "best_parallel": best["parallel"] if valid else None,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
