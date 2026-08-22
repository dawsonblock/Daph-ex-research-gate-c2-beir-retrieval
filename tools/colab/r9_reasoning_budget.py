"""R9: Local inference reasoning-budget qualification.

Tests LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M at different reasoning budgets
to find the minimum budget that preserves controller-policy quality.

Must be run on a Colab GPU session with llama.cpp installed and the
Liquid model downloaded.

Usage (on Colab):
    PYTHONPATH=. python3 tools/colab/r9_reasoning_budget.py \
        --model-path /content/models/LFM2.5-2B-Q5_K_M.gguf \
        --output experiments/v2b_i3_15c/confirmation/r9_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add repo to path
REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend, ModelCallResult


# ---- Frozen production-policy test states ----
# 20 states covering A1 and M3 representations, all action types

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

# GBNF grammar for constrained JSON generation
# This enforces valid JSON at the token level, equivalent to the
# llama.cpp server's json_schema response_format with strict=True
ACTION_GRAMMAR = r'''root ::= "{" "\"action\":\"" action "\",\"reason_code\":\"" reasoncode "\",\"target_id\":null}"
action ::= "ANSWER" | "RETRIEVE" | "VERIFY" | "SEARCH_MORE" | "REASON_MORE" | "DEFER" | "STOP"
reasoncode ::= [A-Z] [A-Z0-9_]*
'''


def make_test_states() -> list[dict]:
    """Build 20 frozen production-policy test states."""
    states = []

    # A1 states (pre-T2, standard representation)
    # 1-4: Clear action cases
    states.append({
        "id": "A1_DEFER_001", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All hypotheses have been eliminated by SUFFICIENT contradicting evidence. The evidence set is inconsistent.",
        "evidence": [
            {"id": "E1", "proposition": "Service is operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
            {"id": "E2", "proposition": "Service is not operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_ANSWER_001", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is supported by SUFFICIENT verified evidence. H2 is eliminated.",
        "evidence": [
            {"id": "E1", "proposition": "Service is operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_VERIFY_001", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Evidence E1 is present but UNVERIFIED. It may support or contradict H1.",
        "evidence": [
            {"id": "E1", "proposition": "Monitoring shows service is responding.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "A1_RETRIEVE_001", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "No evidence has been retrieved yet. Retrieval is available.",
        "evidence": [],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })

    # 5-8: More A1 states
    states.append({
        "id": "A1_DEFER_002", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Evidence is insufficient and no more retrieval or verification is possible.",
        "evidence": [
            {"id": "E1", "proposition": "Service might be operational.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_VERIFY_002", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Two evidence items are UNVERIFIED. Both need checking before deciding.",
        "evidence": [
            {"id": "E1", "proposition": "Probe A says service is up.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []},
            {"id": "E2", "proposition": "Probe B says service is down.", "verified": False, "state": "UNVERIFIED", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "A1_ANSWER_002", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "H1 is confirmed by two SUFFICIENT verified evidence items. H2 is eliminated.",
        "evidence": [
            {"id": "E1", "proposition": "Health check passed.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
            {"id": "E2", "proposition": "Synthetic test succeeded.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_SEARCH_001", "representation": "A1", "expected_action": "SEARCH_MORE",
        "scenario": "Initial evidence is retrieved but insufficient. More sources are needed.",
        "evidence": [
            {"id": "E1", "proposition": "Old report says service was up.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": True, "can_verify": True},
    })

    # 9-10: A1 REASON_MORE
    states.append({
        "id": "A1_REASON_001", "representation": "A1", "expected_action": "REASON_MORE",
        "scenario": "Evidence is contradictory but not yet SUFFICIENT. Need to reason about implications.",
        "evidence": [
            {"id": "E1", "proposition": "Probe A says service is up.", "verified": True, "state": "MISSING", "supports": ["H1"], "contradicts": []},
            {"id": "E2", "proposition": "Probe B says service is down.", "verified": True, "state": "MISSING", "supports": ["H2"], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_DEFER_003", "representation": "A1", "expected_action": "DEFER",
        "scenario": "All evidence is verified but none is SUFFICIENT. Cannot retrieve or search more.",
        "evidence": [
            {"id": "E1", "proposition": "Service was up 1 hour ago.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []},
            {"id": "E2", "proposition": "No recent reports available.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })

    # M3 states (post-T2, conflict-aware representation)
    # 11-14: M3 states where T2 has fired
    states.append({
        "id": "M3_DEFER_001", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated by SUFFICIENT contradicting evidence. The evidence set is inconsistent. No resolution is possible.",
        "evidence": [
            {"id": "E1", "proposition": "Authoritative probe: service is NOT operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
            {"id": "E2", "proposition": "Authoritative probe: service IS operational.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Service is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "Service is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "M3_DEFER_002", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Both hypotheses eliminated. Evidence is contradictory and SUFFICIENT. Must defer — the hypothesis set cannot be resolved consistently.",
        "evidence": [
            {"id": "E1", "proposition": "Database is rejecting all queries.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
            {"id": "E2", "proposition": "Database is accepting all queries.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Database is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "Database is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "M3_VERIFY_001", "representation": "M3", "expected_action": "VERIFY",
        "scenario": "T2 has not fired yet. One evidence is SUFFICIENT (eliminates H1), the other is UNVERIFIED. Must verify E2 before T2 can fire.",
        "evidence": [
            {"id": "E1", "proposition": "API gateway is offline.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
            {"id": "E2", "proposition": "API gateway is online.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "API gateway is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "API gateway is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })
    states.append({
        "id": "M3_RETRIEVE_001", "representation": "M3", "expected_action": "RETRIEVE",
        "scenario": "T2 has not fired. E1 is SUFFICIENT (eliminates H1). E2 is hidden — must retrieve it first.",
        "evidence": [
            {"id": "E1", "proposition": "CDN is offline.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "CDN is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "CDN is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": True},
        "t2_fired": False,
    })

    # 15-20: Additional states for coverage
    states.append({
        "id": "M3_DEFER_003", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. All hypotheses eliminated. CDN conflict evidence is SUFFICIENT on both sides.",
        "evidence": [
            {"id": "E1", "proposition": "CDN edge check: all locations returning 5xx.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
            {"id": "E2", "proposition": "CDN edge test: all locations serving content.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "CDN is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "CDN is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "A1_ANSWER_003", "representation": "A1", "expected_action": "ANSWER",
        "scenario": "Kubernetes cluster is confirmed operational by SUFFICIENT evidence. H2 eliminated.",
        "evidence": [
            {"id": "E1", "proposition": "All nodes are Ready and pods scheduled normally.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Kubernetes cluster is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Kubernetes cluster is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "A1_VERIFY_003", "representation": "A1", "expected_action": "VERIFY",
        "scenario": "Security audit evidence is UNVERIFIED. Must verify before deciding on security posture.",
        "evidence": [
            {"id": "E1", "proposition": "Security scan found no vulnerabilities.", "verified": False, "state": "UNVERIFIED", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Security posture is confirmed.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Security posture is not confirmed.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": True},
    })
    states.append({
        "id": "A1_DEFER_004", "representation": "A1", "expected_action": "DEFER",
        "scenario": "Deployment evidence is MISSING after verification. Cannot confirm or deny. No more actions available.",
        "evidence": [
            {"id": "E1", "proposition": "Deployment status unclear.", "verified": True, "state": "MISSING", "supports": [], "contradicts": []},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Deployment is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Deployment is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
    })
    states.append({
        "id": "M3_DEFER_004", "representation": "M3", "expected_action": "DEFER",
        "scenario": "T2 has fired. Load balancer conflict: both probes are SUFFICIENT and contradictory.",
        "evidence": [
            {"id": "E1", "proposition": "LB returning 503 for all requests.", "verified": True, "state": "SUFFICIENT", "supports": ["H2"], "contradicts": ["H1"]},
            {"id": "E2", "proposition": "LB distributing traffic evenly.", "verified": True, "state": "SUFFICIENT", "supports": ["H1"], "contradicts": ["H2"]},
        ],
        "hypotheses": [
            {"id": "H1", "proposition": "Load balancer is operational.", "status": "ELIMINATED"},
            {"id": "H2", "proposition": "Load balancer is not operational.", "status": "ELIMINATED"},
        ],
        "affordances": {"can_retrieve": False, "can_search": False, "can_verify": False},
        "t2_fired": True,
    })
    states.append({
        "id": "A1_RETRIEVE_002", "representation": "A1", "expected_action": "RETRIEVE",
        "scenario": "No evidence retrieved for Redis cache status. Retrieval is available.",
        "evidence": [],
        "hypotheses": [
            {"id": "H1", "proposition": "Redis cache is operational.", "status": "VIABLE"},
            {"id": "H2", "proposition": "Redis cache is not operational.", "status": "VIABLE"},
        ],
        "affordances": {"can_retrieve": True, "can_search": False, "can_verify": False},
    })

    return states


def build_user_prompt(state: dict) -> str:
    """Build the user prompt for a test state."""
    evidence_str = []
    for ev in state.get("evidence", []):
        evidence_str.append({
            "evidence_id": ev["id"],
            "proposition": ev["proposition"],
            "verification_state": ev["state"],
            "verified": ev["verified"],
            "supports": ev.get("supports", []),
            "contradicts": ev.get("contradicts", []),
        })

    return json.dumps({
        "scenario": state["scenario"],
        "representation": state["representation"],
        "t2_fired": state.get("t2_fired", False),
        "action_affordances": state["affordances"],
        "hypotheses": [
            {"hypothesis_id": h["id"], "proposition": h["proposition"], "status": h["status"]}
            for h in state["hypotheses"]
        ],
        "evidence_items": evidence_str,
        "prior_actions": [],
        "prior_outcomes": [],
        "resource_state": {"elapsed_ms": 0, "elapsed_ms_remaining": 30000},
    })


def start_llama_server(llama_server_bin: str, model_path: str, reasoning_budget: int,
                       port: int = 8080) -> subprocess.Popen:
    """Start the llama.cpp C++ server with --reasoning-budget.

    This is the ONLY effective way to control LFM2.5 reasoning tokens.
    The llama-cpp-python Python API does not support --reasoning-budget.
    """
    cmd = [
        llama_server_bin,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "999",              # ALL layers on GPU
        "--reasoning-budget", str(reasoning_budget),
        "-c", "4096",
        "-b", "512",
        "-ub", "512",
        "--temp", "0.0",
        "--seed", "42",
        "--threads", "4",
        "--flash-at", "1",
        "--no-mmap",
    ]

    print(f"  Starting llama-server: reasoning_budget={reasoning_budget}, port={port}, ngl=999, flash-at=1", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            print(f"  Server ready on port {port}", flush=True)
            return proc
        except Exception:
            time.sleep(1)

    # Server didn't start — print stderr for debugging
    proc.terminate()
    stderr = proc.stderr.read().decode() if proc.stderr else ""
    print(f"  Server stderr: {stderr[:500]}", flush=True)
    raise RuntimeError(f"llama-server failed to start within 120s (reasoning_budget={reasoning_budget})")


def stop_llama_server(proc: subprocess.Popen):
    """Stop the llama.cpp server."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server stopped", flush=True)


def run_budget_qualification_server(model_path: str, reasoning_budget: int,
                                    max_tokens: int, test_states: list[dict],
                                    port: int, llama_server_bin: str) -> dict:
    """Run qualification using the llama.cpp C++ server with --reasoning-budget.

    Uses the existing LocalLlamaBackend to make OpenAI-compatible requests.
    The server enforces JSON schema constraints with strict=True.
    """
    print(f"\n{'='*80}", flush=True)
    print(f"REASONING BUDGET = {reasoning_budget} (server mode)", flush=True)
    print(f"{'='*80}", flush=True)

    proc = start_llama_server(llama_server_bin, model_path, reasoning_budget, port)

    try:
        backend = LocalLlamaBackend(
            model_name="LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout_seconds=300,
        )

        results = []
        for state in test_states:
            user_prompt = build_user_prompt(state)
            t0 = time.time()
            try:
                call = backend.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                decoded = decode_output(call.raw_output, strict=True)
                action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
                correct = (action == state["expected_action"])

                results.append({
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
                    "raw_output": call.raw_output[:200] if call.raw_output else "",
                })

                print(f"  {state['id']:20s}  expected={state['expected_action']:12s}  "
                      f"got={action:12s}  {'OK' if correct else 'MISS'}  "
                      f"tokens={call.completion_tokens}  reasoning={call.reasoning_tokens}  "
                      f"latency={call.latency_ms}ms  finish={call.finish_reason}", flush=True)

            except Exception as e:
                results.append({
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
                    "raw_output": str(e)[:200],
                })
                print(f"  {state['id']:20s}  expected={state['expected_action']:12s}  "
                      f"got=BACKEND_ERROR  MISS  error={str(e)[:80]}", flush=True)

    finally:
        stop_llama_server(proc)

    return _compute_metrics(results, reasoning_budget, max_tokens)


def _compute_metrics(results: list[dict], reasoning_budget: int, max_tokens: int) -> dict:
    """Compute qualification metrics from results list."""
    n = len(results)
    decoder_success = sum(1 for r in results if r["decoder_valid"]) / n
    fail_closed = sum(1 for r in results if r["actual_action"] == "FAIL_CLOSED") / n
    core_accuracy = sum(1 for r in results if r["correct"]) / n
    length_failures = sum(1 for r in results if r["finish_reason"] == "length") / n
    mean_tokens = sum(r["completion_tokens"] for r in results) / n
    mean_reasoning = sum(r["reasoning_tokens"] for r in results) / n
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
        "mean_latency_ms": round(mean_latency, 1),
        "per_action_accuracy": {k: round(v, 4) for k, v in action_acc.items()},
        "a1_accuracy": round(a1_acc, 4),
        "m3_accuracy": round(m3_acc, 4),
        "a1_m3_asymmetry_pp": round(a1_m3_asymmetry * 100, 1),
        "non_defer_to_defer_rate": round(non_defer_to_defer, 4),
        "results": results,
    }

    print(f"\n  Summary (budget={reasoning_budget}):", flush=True)
    print(f"    Decoder success:     {decoder_success:.1%}", flush=True)
    print(f"    Core action accuracy: {core_accuracy:.1%}", flush=True)
    print(f"    Fail-closed rate:    {fail_closed:.1%}", flush=True)
    print(f"    Length failures:     {length_failures:.1%}", flush=True)
    print(f"    Mean tokens:         {mean_tokens:.0f}", flush=True)
    print(f"    Mean reasoning:      {mean_reasoning:.0f}", flush=True)
    print(f"    Mean latency:        {mean_latency:.0f}ms", flush=True)
    print(f"    A1 accuracy:         {a1_acc:.1%}", flush=True)
    print(f"    M3 accuracy:         {m3_acc:.1%}", flush=True)
    print(f"    A1/M3 asymmetry:     {a1_m3_asymmetry*100:.1f}pp", flush=True)
    print(f"    Non-DEFER→DEFER:     {non_defer_to_defer:.1%}", flush=True)
    print(f"    Per-action: {action_acc}", flush=True)

    return metrics


def run_budget_qualification(model_path: str, reasoning_budget: int,
                             max_tokens: int, test_states: list[dict],
                             port: int = 8080) -> dict:
    """Run qualification for a single reasoning budget using llama-cpp-python.

    Uses the Python API directly with GPU offload for maximum speed.
    No server build required — uses pre-built CUDA wheel.
    """
    from llama_cpp import Llama, LlamaGrammar

    print(f"\n{'='*80}")
    print(f"REASONING BUDGET = {reasoning_budget}")
    print(f"{'='*80}")

    # Load model with GPU offload
    # n_gpu_layers=-1 offloads ALL layers to GPU
    # The 2.6B Q5_K_M model is ~1.94GB, fits easily in T4's 15GB VRAM
    grammar = LlamaGrammar.from_string(ACTION_GRAMMAR)
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=512,
        temperature=0.0,
        seed=42,
        verbose=False,
    )
    print(f"  Model loaded with GPU offload (reasoning_budget={reasoning_budget})")

    results = []
    for state in test_states:
        user_prompt = build_user_prompt(state)
        t0 = time.time()
        try:
            # Use chat completion API with GBNF grammar constraint
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                grammar=grammar,
            )
            latency_ms = int((time.time() - t0) * 1000)
            choice = response["choices"][0]
            raw_output = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

            decoded = decode_output(raw_output, strict=True)
            action = decoded.proposal.action.value if (decoded.valid and decoded.proposal) else "FAIL_CLOSED"
            correct = (action == state["expected_action"])

            results.append({
                "state_id": state["id"],
                "representation": state["representation"],
                "expected_action": state["expected_action"],
                "actual_action": action,
                "correct": correct,
                "decoder_valid": decoded.valid,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "latency_ms": latency_ms,
                "raw_output": raw_output[:200] if raw_output else "",
            })

            print(f"  {state['id']:20s}  expected={state['expected_action']:12s}  "
                  f"got={action:12s}  {'OK' if correct else 'MISS'}  "
                  f"tokens={completion_tokens}  latency={latency_ms}ms  "
                  f"finish={finish_reason}")

        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            results.append({
                "state_id": state["id"],
                "representation": state["representation"],
                "expected_action": state["expected_action"],
                "actual_action": "BACKEND_ERROR",
                "correct": False,
                "decoder_valid": False,
                "finish_reason": None,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "latency_ms": latency_ms,
                "raw_output": str(e)[:200],
            })
            print(f"  {state['id']:20s}  expected={state['expected_action']:12s}  "
                  f"got=BACKEND_ERROR  MISS  error={str(e)[:80]}")

    # Free model from memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    print(f"  Model unloaded")

    # Compute metrics
    n = len(results)
    decoder_success = sum(1 for r in results if r["decoder_valid"]) / n
    fail_closed = sum(1 for r in results if r["actual_action"] == "FAIL_CLOSED") / n
    core_accuracy = sum(1 for r in results if r["correct"]) / n
    length_failures = sum(1 for r in results if r["finish_reason"] == "length") / n
    mean_tokens = sum(r["completion_tokens"] for r in results) / n
    mean_reasoning = sum(r["reasoning_tokens"] for r in results) / n
    mean_latency = sum(r["latency_ms"] for r in results) / n

    # Per-action accuracy
    action_acc = {}
    for action in ["DEFER", "ANSWER", "VERIFY", "RETRIEVE", "SEARCH_MORE", "REASON_MORE"]:
        relevant = [r for r in results if r["expected_action"] == action]
        if relevant:
            action_acc[action] = sum(1 for r in relevant if r["correct"]) / len(relevant)

    # A1/M3 accuracy
    a1_results = [r for r in results if r["representation"] == "A1"]
    m3_results = [r for r in results if r["representation"] == "M3"]
    a1_acc = sum(1 for r in a1_results if r["correct"]) / len(a1_results) if a1_results else 0
    m3_acc = sum(1 for r in m3_results if r["correct"]) / len(m3_results) if m3_results else 0
    a1_m3_asymmetry = abs(a1_acc - m3_acc)

    # Non-DEFER to DEFER pathological bias
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
        "mean_latency_ms": round(mean_latency, 1),
        "per_action_accuracy": {k: round(v, 4) for k, v in action_acc.items()},
        "a1_accuracy": round(a1_acc, 4),
        "m3_accuracy": round(m3_acc, 4),
        "a1_m3_asymmetry_pp": round(a1_m3_asymmetry * 100, 1),
        "non_defer_to_defer_rate": round(non_defer_to_defer, 4),
        "results": results,
    }

    # Print summary
    print(f"\n  Summary (budget={reasoning_budget}):")
    print(f"    Decoder success:     {decoder_success:.1%}")
    print(f"    Core action accuracy: {core_accuracy:.1%}")
    print(f"    Fail-closed rate:    {fail_closed:.1%}")
    print(f"    Length failures:     {length_failures:.1%}")
    print(f"    Mean tokens:         {mean_tokens:.0f}")
    print(f"    Mean reasoning:      {mean_reasoning:.0f}")
    print(f"    Mean latency:        {mean_latency:.0f}ms")
    print(f"    A1 accuracy:         {a1_acc:.1%}")
    print(f"    M3 accuracy:         {m3_acc:.1%}")
    print(f"    A1/M3 asymmetry:     {a1_m3_asymmetry*100:.1f}pp")
    print(f"    Non-DEFER→DEFER:     {non_defer_to_defer:.1%}")
    print(f"    Per-action: {action_acc}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="R9: Reasoning budget qualification")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--budgets", default="0,64,128,256,512,1024",
                        help="Comma-separated reasoning budgets to test")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens for R9a (reasoning budget phase)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--use-server", action="store_true",
                        help="Use llama.cpp C++ server with --reasoning-budget (required for LFM2.5)")
    parser.add_argument("--llama-server", default="/content/llama.cpp/build/bin/llama-server",
                        help="Path to llama-server binary (for --use-server mode)")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    test_states = make_test_states()
    print(f"Test states: {len(test_states)}")
    print(f"Budgets: {budgets}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Mode: {'server (--reasoning-budget)' if args.use_server else 'python API (grammar)'}")

    # Run qualification for each budget
    all_results = []
    for budget in budgets:
        if args.use_server:
            metrics = run_budget_qualification_server(
                args.model_path, budget, args.max_tokens, test_states,
                args.port, args.llama_server
            )
        else:
            metrics = run_budget_qualification(
                args.model_path, budget, args.max_tokens, test_states, args.port
            )
        all_results.append(metrics)

    # Compute action agreement vs 1024 (reference)
    reference = next((r for r in all_results if r["reasoning_budget"] == 1024), all_results[-1])
    ref_actions = {r["state_id"]: r["actual_action"] for r in reference["results"]}

    print(f"\n{'='*80}")
    print("ACTION AGREEMENT vs 1024 (reference)")
    print(f"{'='*80}")
    for metrics in all_results:
        budget = metrics["reasoning_budget"]
        if budget == 1024:
            agreement = 1.0
        else:
            agree = sum(1 for r in metrics["results"]
                       if r["actual_action"] == ref_actions.get(r["state_id"]))
            agreement = agree / len(metrics["results"])
        metrics["action_agreement_vs_1024"] = round(agreement, 4)
        print(f"  Budget {budget:5d}: agreement = {agreement:.1%}")

    # Find minimum budget satisfying all gates
    print(f"\n{'='*80}")
    print("GATE EVALUATION")
    print(f"{'='*80}")
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
        # Per-action gates
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
        print(f"  Budget {metrics['reasoning_budget']:5d}: {'PASS' if all_pass else 'FAIL'} "
              f"{'(' + ', '.join(k for k,v in gates.items() if not v) + ')' if not all_pass else ''}")

    # Find minimum passing budget
    passing = [g for g in gate_results if g["all_pass"]]
    if passing:
        min_budget = min(g["budget"] for g in passing)
        print(f"\n  MINIMUM PASSING BUDGET: {min_budget}")
    else:
        min_budget = None
        print(f"\n  NO BUDGET PASSED ALL GATES")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "test": "R9a_reasoning_budget_qualification",
        "model": "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M",
        "n_test_states": len(test_states),
        "budgets_tested": budgets,
        "max_tokens": args.max_tokens,
        "results": all_results,
        "gate_results": gate_results,
        "minimum_passing_budget": min_budget,
        "reference_budget": 1024,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
