#!/usr/bin/env python3
"""I3.15c: Phase-Eligible Balanced Benchmark — Full Factorial Runner.

4 strata x 2 retrieval difficulty x 3 retrieval systems x 2 arms
  = 200 unique tasks x 3 x 2 = 1200 potential trajectories

Strata:
  T2_CONFLICT_IMMEDIATE: T2 fires at step 0 (both evidence SUFFICIENT)
  T2_CONFLICT_LATE: T2 fires after VERIFY (one evidence starts UNVERIFIED)
  DEFER_CONTROL: evidence insufficient, T2 never fires, expected=DEFER
  ANSWER_CONTROL: evidence sufficient, T2 never fires, expected=ANSWER

Retrieval systems: Q0_BM25, Q3_RERANKED, Q4_ORACLE
Arms: A1_INFERRED, R1_INFERRED

Pre-registered contrasts:
  Delta_T2+ = U(R1) - U(A1) | T2-eligible
  Delta_DEFER- = U(R1) - U(A1) | DEFER_CONTROL
  Delta_ANSWER = U(R1) - U(A1) | ANSWER_CONTROL
  I_phase = Delta_T2+ - Delta_DEFER-

Desired signature:
  Delta_T2+ > 0
  I_phase > 0
  Delta_DEFER- ~ 0
  Delta_ANSWER ~ 0
  false T2 on controls = 0

Cost metrics:
  Delta_Steps_T2+ = Steps(R1) - Steps(A1) | T2-eligible
  P(step-limit | R1, T2+)
  redundant-action counts per trajectory

Frozen identities:
  BENCHMARK_V1: i3_15_t2_eligible corpus + structural validator
  LOCAL_POLICY_V2: LFM2.5-2.6B Q5_K_M, JSON-schema, reasoning-budget 1024
  T2/R1 implementation: from run_i3_12j_factorial.py
  EXPERIMENT_PROTOCOL_V1: pre-registered contrasts and metrics

Usage:
  python scripts/run_i3_15c_factorial.py --backend local
  python scripts/run_i3_15c_factorial.py --backend local --run-experiment
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load i3_12j module
spec_12j = importlib.util.spec_from_file_location(
    "i3_12j", str(REPO_ROOT / "scripts" / "run_i3_12j_factorial.py"))
i3_12j = importlib.util.module_from_spec(spec_12j)
spec_12j.loader.exec_module(i3_12j)
i3_7e = i3_12j.i3_7e

from hrm_adaptive_memory.executive.semantic_relations.i3_15c_task_generator import (
    generate_i3_15c_corpus, validate_t2_eligibility, get_i3_15c_corpus,
    CONFLICT_PASSAGES,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask,
)
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import LocalLlamaBackend, ModelCallResult
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.memory.chunking import Chunk
from hrm_adaptive_memory.retrieval.i3_14_retrieval_ladder import build_retriever

from scripts.run_i3_15_r1_balanced import (
    build_corpus_index, get_required_passage_ids, build_retrieved_evidence_task,
    TOP_K, adapt_local_system_prompt, SHARED_ACTION_SEMANTICS_V1,
    FROZEN_INFERENCE_CONFIG,
)


# ---------------------------------------------------------------------------
# Frozen identities
# ---------------------------------------------------------------------------

BENCHMARK_V1 = {
    "benchmark_id": "i3_15_t2_eligible",
    "version": "BENCHMARK_V1",
    "description": "Phase-Eligible Balanced Benchmark with T2_CONFLICT_IMMEDIATE, T2_CONFLICT_LATE, DEFER_CONTROL, ANSWER_CONTROL",
    "n_strata": 4,
    "n_retrieval_difficulty": 2,
    "n_retrieval_systems": 3,
    "n_arms": 2,
    "n_per_cell": 25,
    "total_unique_tasks": 200,
    "total_potential_trajectories": 1200,
    "seed": 42,
    "strata": [
        "T2_CONFLICT_IMMEDIATE",
        "T2_CONFLICT_LATE",
        "DEFER_CONTROL",
        "ANSWER_CONTROL",
    ],
    "retrieval_systems": ["Q0_BM25", "Q3_RERANKED", "Q4_ORACLE"],
    "arms": ["A1_INFERRED", "R1_INFERRED"],
    "t2_definition": "len(eliminated_hypotheses) == n_hypotheses and n_hypotheses > 0",
    "t2_unchangeable": True,
    "t2_note": "T2 is the mechanism under test. Changing T2 would change the architecture, not fix the benchmark.",
}

T2_R1_IMPLEMENTATION = {
    "implementation_id": "T2_R1_FROM_I3_12J",
    "source_file": "scripts/run_i3_12j_factorial.py",
    "function_r1": "run_r1_trajectory_i3_12",
    "function_a1": "run_trajectory_i3_12",
    "t2_trigger": "len(eliminated) == n_hypotheses and n_hypotheses > 0",
    "r1_routing": "A1 before T2, M3 after T2",
    "snapshot_builder": "make_inferred_snapshot_builder(DeterministicRelationExtractor)",
    "extractor": "DeterministicRelationExtractor",
    "extractor_version": "frozen_v2.6.0",
    "note": "T2/R1 implementation is frozen from I3.12j. No changes to T2 trigger or R1 routing logic.",
}

EXPERIMENT_PROTOCOL_V1 = {
    "protocol_id": "EXPERIMENT_PROTOCOL_V1",
    "pre_registered_contrasts": {
        "Delta_T2+": "U(R1) - U(A1) | T2-eligible (IMMEDIATE + LATE)",
        "Delta_T2_immediate": "U(R1) - U(A1) | T2_CONFLICT_IMMEDIATE",
        "Delta_T2_late": "U(R1) - U(A1) | T2_CONFLICT_LATE",
        "Delta_DEFER-": "U(R1) - U(A1) | DEFER_CONTROL",
        "Delta_ANSWER": "U(R1) - U(A1) | ANSWER_CONTROL",
        "I_phase": "Delta_T2+ - Delta_DEFER-",
    },
    "desired_signature": {
        "Delta_T2+": "> 0",
        "I_phase": "> 0",
        "Delta_DEFER-": "~ 0",
        "Delta_ANSWER": "~ 0",
        "false_T2_on_controls": "= 0",
    },
    "cost_metrics": {
        "Delta_Steps_T2+": "Steps(R1) - Steps(A1) | T2-eligible",
        "P_step_limit_R1_T2+": "P(step-limit | R1, T2-eligible)",
        "redundant_action_counts": "per-trajectory count of repeated consecutive actions",
    },
    "statistical_method": "paired bootstrap CI over tasks",
    "structural_qualification_required": True,
    "structural_qualification_checks": [
        "T2_CONFLICT_IMMEDIATE: T2 fires at initial state = 100%",
        "T2_CONFLICT_LATE: T2 at initial state = 0%, T2 after gold transition = 100%",
        "DEFER_CONTROL: T2 at gold state = 0%",
        "ANSWER_CONTROL: T2 at gold state = 0%",
    ],
    "abort_without_run_experiment": True,
}

OPENROUTER_POLICY_V1 = {
    "policy_id": "OPENROUTER_POLICY_V1",
    "description": "OpenRouter API backend for rapid execution validation. Not the frozen pinned model.",
    "base_url": "https://openrouter.ai/api/v1",
    "default_model": "nvidia/nemotron-3.5-lightning:free",
    "response_format": "json_schema",
    "temperature": 0.0,
    "max_tokens": 2048,
    "reasoning_enabled": False,
    "note": "This backend uses a different model than LOCAL_POLICY_V2. Results are not directly comparable to the pinned-model experiment.",
}


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

class OpenRouterBackend:
    """OpenRouter API backend using OpenAI-compatible endpoint.

    WARNING: This is not the frozen LOCAL_POLICY_V2 backend. It uses a
    hosted model (default nvidia/nemotron-3.5-lightning:free) for speed.
    It is intended for rapid validation only, not for the pinned-model
    causal claim.
    """

    model_name: str = "nvidia/nemotron-3.5-lightning:free"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    reasoning_enabled: bool = False

    def __init__(self, model_name: str | None = None):
        import os
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if model_name:
            self.model_name = model_name

    def generate(self, *, system_prompt: str, user_prompt: str,
                 temperature: float, max_tokens: int) -> ModelCallResult:
        import os, json, time, urllib.error, urllib.request

        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        action_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE",
                        "REASON_MORE", "DEFER", "STOP",
                    ],
                },
                "reason_code": {
                    "type": "string",
                    "pattern": "^[A-Z][A-Z0-9_]*$",
                },
                "target_id": {"type": ["string", "null"]},
            },
            "required": ["action", "reason_code", "target_id"],
            "additionalProperties": False,
        }

        body_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "action_proposal",
                    "strict": True,
                    "schema": action_schema,
                },
            },
        }
        if self.reasoning_enabled:
            body_kwargs["reasoning"] = {"enabled": True}
        payload = json.dumps(body_kwargs).encode()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://devin.ai",
            "X-OpenRouter-Title": "DAPH-I3.15c",
        }

        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = json.loads(resp.read().decode())
                    choice = body.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    raw_output = message.get("content", "") or ""
                    finish_reason = choice.get("finish_reason", "unknown")
                    usage = body.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                    latency_ms = (time.time() - t0) * 1000

                    return ModelCallResult(
                        raw_output=raw_output,
                        model_name=body.get("model", self.model_name),
                        system_fingerprint=body.get("system_fingerprint"),
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        reasoning_tokens=reasoning_tokens,
                        latency_ms=int(latency_ms),
                        provider_raw_output=raw_output,
                    )
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
            except Exception:
                raise

        raise RuntimeError("OpenRouter request failed after retries")

_CORPUS_CACHE: dict[str, Any] = {}


def _get_cached_corpus():
    if "corpus" not in _CORPUS_CACHE:
        corpus_passages = get_i3_15c_corpus()
        chunks = [
            Chunk(
                chunk_id=p.passage_id,
                source_id=p.source,
                source_type="document",
                title=p.domain,
                section="",
                content=p.text,
                token_count=len(p.text.split()),
                metadata={"domain": p.domain, "source": p.source},
            )
            for p in corpus_passages
        ]
        corpus_by_text = {p.text: p.passage_id for p in corpus_passages}
        corpus_by_id = {p.passage_id: p for p in corpus_passages}
        _CORPUS_CACHE["corpus"] = (corpus_passages, corpus_by_text, corpus_by_id, chunks)
        _CORPUS_CACHE["corpus_sha256"] = hashlib.sha256(
            json.dumps([
                {"id": p.passage_id, "text": p.text, "domain": p.domain}
                for p in corpus_passages
            ], sort_keys=True).encode()
        ).hexdigest()
    return _CORPUS_CACHE["corpus"] + (_CORPUS_CACHE["corpus_sha256"],)


_RETRIEVER_CACHE: dict[str, Any] = {}
_RETRIEVER_LOCK = __import__("threading").Lock()


def _get_cached_retriever(retrieval_level: str, chunks):
    if retrieval_level not in _RETRIEVER_CACHE:
        with _RETRIEVER_LOCK:
            if retrieval_level not in _RETRIEVER_CACHE:
                _RETRIEVER_CACHE[retrieval_level] = build_retriever(retrieval_level, chunks)
    return _RETRIEVER_CACHE[retrieval_level]


# ---------------------------------------------------------------------------
# Mechanism receipt
# ---------------------------------------------------------------------------

def build_mechanism_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """Build a mechanism receipt for an R1 trajectory."""
    routing_log = result.get("routing_log", [])
    model_calls = result.get("model_call_log", [])

    representation_by_step = [entry.get("representation", "?") for entry in routing_log]

    # Find pre/post trigger packets
    trigger_step = result.get("r1_trigger_step")
    pre_trigger_hash = None
    post_trigger_hash = None

    if trigger_step is not None and model_calls:
        for call in model_calls:
            if call.get("step") == trigger_step:
                post_trigger_hash = call.get("packet_sha256")
            if call.get("step") == trigger_step - 1:
                pre_trigger_hash = call.get("packet_sha256")

    # If trigger at step 0, pre-trigger is None (no prior step)
    if trigger_step == 0:
        pre_trigger_hash = None

    # Determine t2 eligibility from category
    category = result.get("category", "")
    t2_eligible = category.startswith("t2_conflict")

    return {
        "t2_eligible": t2_eligible,
        "t2_triggered": result.get("r1_triggered", False),
        "trigger_step": trigger_step,
        "representation_by_step": representation_by_step,
        "pre_trigger_packet_hash": pre_trigger_hash,
        "post_trigger_packet_hash": post_trigger_hash,
        "terminal_action": result.get("terminal_action"),
        "n_steps": result.get("steps", 0),
        "hit_step_limit": result.get("terminal_result") == "STEP_LIMIT",
    }


# ---------------------------------------------------------------------------
# Redundant action counting
# ---------------------------------------------------------------------------

def count_redundant_actions(actions: list[str]) -> dict[str, Any]:
    """Count redundant actions in a trajectory.

    Redundant = same action repeated consecutively more than twice.
    """
    if not actions:
        return {"total_redundant": 0, "max_run": 0, "redundant_runs": []}

    runs = []
    current_action = actions[0]
    current_count = 1
    for a in actions[1:]:
        if a == current_action:
            current_count += 1
        else:
            if current_count > 2:
                runs.append({"action": current_action, "count": current_count})
            current_action = a
            current_count = 1
    if current_count > 2:
        runs.append({"action": current_action, "count": current_count})

    total_redundant = sum(r["count"] - 2 for r in runs)
    max_run = max((r["count"] for r in runs), default=0)
    return {
        "total_redundant": total_redundant,
        "max_run": max_run,
        "redundant_runs": runs,
    }


# ---------------------------------------------------------------------------
# Trajectory worker
# ---------------------------------------------------------------------------

def run_single_trajectory(
    task,
    retrieval_level: str,
    arm: str,
    chunks,
    corpus_by_text,
    corpus_by_id,
    max_tokens: int = 2048,
    base_url: str | None = None,
    backend_type: str = "local",
    openrouter_model: str | None = None,
    pre_retrieved_passages: list | None = None,
) -> dict[str, Any]:
    """Run a single trajectory."""
    et = task.evidence_task
    required_ids = get_required_passage_ids(task, corpus_by_text)

    # Retrieve
    if pre_retrieved_passages is not None:
        retrieved_passages = pre_retrieved_passages
    elif retrieval_level == "Q4_ORACLE":
        retrieved_passages = [corpus_by_id[pid] for pid in required_ids if pid in corpus_by_id]
    else:
        retriever = _get_cached_retriever(retrieval_level, chunks)
        query = et.task_summary
        retrieved = retriever.search(query, top_k=TOP_K)
        retrieved_passages = [
            corpus_by_id[c.chunk_id] for c, _ in retrieved
            if c.chunk_id in corpus_by_id
        ]

    new_et = build_retrieved_evidence_task(task, retrieved_passages, corpus_by_text)
    recall = len({p.passage_id for p in retrieved_passages} & required_ids) / max(len(required_ids), 1)

    extractor = DeterministicRelationExtractor()
    snapshot_builder = i3_12j.make_inferred_snapshot_builder(extractor)
    budget = ResourceBudget(
        max_executive_steps=10, max_retrieval_calls=3,
        max_search_calls=2, max_verification_calls=5,
    )
    utility = MetareasoningUtility.from_file(
        REPO_ROOT / "configs" / "v2b_i3_1_utility_v1.json")

    def backend_factory(base_url=None, backend_type="local"):
        if backend_type == "openrouter":
            return OpenRouterBackend(model_name=openrouter_model)
        backend = LocalLlamaBackend()
        if base_url is not None:
            backend.base_url = base_url
        return backend

    if arm == "A1_INFERRED":
        result = i3_12j.run_trajectory_i3_12(
            new_et, budget, utility,
            mode="BASELINE_WITH_AFFORDANCES",
            api_key="", fork_label=f"i3_15c:{et.task_id}:{arm}:{retrieval_level}",
            snapshot_builder=snapshot_builder,
            backend_factory=lambda: backend_factory(base_url, backend_type),
            strict_decode=True,
            max_tokens=max_tokens,
            system_prompt_transform=adapt_local_system_prompt,
        )
    else:
        result = i3_12j.run_r1_trajectory_i3_12(
            new_et, budget, utility,
            api_key="", fork_label=f"i3_15c:{et.task_id}:{arm}:{retrieval_level}",
            snapshot_builder=snapshot_builder,
            backend_factory=lambda: backend_factory(base_url, backend_type),
            strict_decode=True,
            max_tokens=max_tokens,
            system_prompt_transform=adapt_local_system_prompt,
        )

    result["task_id"] = et.task_id
    result["category"] = et.category
    result["expected_terminal"] = et.expected_terminal.value
    result["arm"] = arm
    result["retrieval_level"] = retrieval_level
    result["retrieval_recall"] = round(recall, 3)

    # Add redundant action analysis
    actions = result.get("continuation_actions", [])
    result["redundant_actions"] = count_redundant_actions(actions)

    # Add mechanism receipt for R1
    if arm == "R1_INFERRED":
        result["mechanism_receipt"] = build_mechanism_receipt(result)

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_contrasts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pre-registered contrasts and cost metrics."""
    # Group by (stratum, retrieval_level) and arm
    def stratum_from_category(cat: str) -> str:
        if cat.startswith("t2_conflict_immediate"):
            return "T2_CONFLICT_IMMEDIATE"
        elif cat.startswith("t2_conflict_late"):
            return "T2_CONFLICT_LATE"
        elif cat.startswith("defer_control"):
            return "DEFER_CONTROL"
        elif cat.startswith("answer_control"):
            return "ANSWER_CONTROL"
        return "UNKNOWN"

    def t2_eligible(cat: str) -> bool:
        return cat.startswith("t2_conflict")

    # Pair A1 and R1 by (task_id, retrieval_level)
    pairs = defaultdict(dict)
    for r in results:
        key = (r["task_id"], r["retrieval_level"])
        pairs[key][r["arm"]] = r

    # Compute per-pair deltas
    deltas = []
    for key, arms in pairs.items():
        if "A1_INFERRED" not in arms or "R1_INFERRED" not in arms:
            continue
        a1 = arms["A1_INFERRED"]
        r1 = arms["R1_INFERRED"]
        cat = a1["category"]
        s = stratum_from_category(cat)
        delta_u = r1.get("realized_utility", 0) - a1.get("realized_utility", 0)
        delta_steps = r1.get("steps", 0) - a1.get("steps", 0)
        delta_success = int(r1.get("success", False)) - int(a1.get("success", False))
        deltas.append({
            "task_id": key[0],
            "retrieval_level": key[1],
            "stratum": s,
            "t2_eligible": t2_eligible(cat),
            "delta_utility": delta_u,
            "delta_steps": delta_steps,
            "delta_success": delta_success,
            "a1_utility": a1.get("realized_utility", 0),
            "r1_utility": r1.get("realized_utility", 0),
            "a1_steps": a1.get("steps", 0),
            "r1_steps": r1.get("steps", 0),
            "a1_success": a1.get("success", False),
            "r1_success": r1.get("success", False),
            "r1_t2_triggered": r1.get("r1_triggered", False),
            "r1_hit_step_limit": r1.get("terminal_result") == "STEP_LIMIT",
            "a1_redundant": a1.get("redundant_actions", {}).get("total_redundant", 0),
            "r1_redundant": r1.get("redundant_actions", {}).get("total_redundant", 0),
        })

    # Aggregate by stratum
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def bootstrap_ci(xs, n_boot=10000, confidence=0.95):
        """Paired bootstrap CI."""
        import random as rng
        if not xs:
            return (0.0, 0.0)
        n = len(xs)
        boot_means = []
        for _ in range(n_boot):
            sample = [xs[rng.randint(0, n - 1)] for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        alpha = (1 - confidence) / 2
        lo = boot_means[int(n_boot * alpha)]
        hi = boot_means[int(n_boot * (1 - alpha))]
        return (lo, hi)

    strata = ["T2_CONFLICT_IMMEDIATE", "T2_CONFLICT_LATE",
              "DEFER_CONTROL", "ANSWER_CONTROL"]

    per_stratum = {}
    for s in strata:
        s_deltas = [d for d in deltas if d["stratum"] == s]
        if not s_deltas:
            continue
        du = [d["delta_utility"] for d in s_deltas]
        ds = [d["delta_steps"] for d in s_deltas]
        dsucc = [d["delta_success"] for d in s_deltas]
        ci_u = bootstrap_ci(du)
        ci_s = bootstrap_ci(ds)
        per_stratum[s] = {
            "n": len(s_deltas),
            "mean_delta_utility": mean(du),
            "ci_delta_utility": [ci_u[0], ci_u[1]],
            "mean_delta_steps": mean(ds),
            "ci_delta_steps": [ci_s[0], ci_s[1]],
            "mean_delta_success": mean(dsucc),
            "r1_t2_triggered_count": sum(d["r1_t2_triggered"] for d in s_deltas),
            "r1_step_limit_count": sum(d["r1_hit_step_limit"] for d in s_deltas),
            "r1_step_limit_rate": sum(d["r1_hit_step_limit"] for d in s_deltas) / len(s_deltas),
            "mean_a1_redundant": mean([d["a1_redundant"] for d in s_deltas]),
            "mean_r1_redundant": mean([d["r1_redundant"] for d in s_deltas]),
        }

    # T2-eligible aggregate (IMMEDIATE + LATE)
    t2_pos = [d for d in deltas if d["t2_eligible"]]
    defer_neg = [d for d in deltas if d["stratum"] == "DEFER_CONTROL"]
    answer_neg = [d for d in deltas if d["stratum"] == "ANSWER_CONTROL"]

    contrasts = {
        "Delta_T2+": {
            "n": len(t2_pos),
            "mean": mean([d["delta_utility"] for d in t2_pos]) if t2_pos else 0.0,
            "ci": list(bootstrap_ci([d["delta_utility"] for d in t2_pos])) if t2_pos else [0, 0],
        },
        "Delta_T2_immediate": {
            "n": len([d for d in t2_pos if d["stratum"] == "T2_CONFLICT_IMMEDIATE"]),
            "mean": mean([d["delta_utility"] for d in t2_pos if d["stratum"] == "T2_CONFLICT_IMMEDIATE"]),
            "ci": list(bootstrap_ci([d["delta_utility"] for d in t2_pos if d["stratum"] == "T2_CONFLICT_IMMEDIATE"])),
        },
        "Delta_T2_late": {
            "n": len([d for d in t2_pos if d["stratum"] == "T2_CONFLICT_LATE"]),
            "mean": mean([d["delta_utility"] for d in t2_pos if d["stratum"] == "T2_CONFLICT_LATE"]),
            "ci": list(bootstrap_ci([d["delta_utility"] for d in t2_pos if d["stratum"] == "T2_CONFLICT_LATE"])),
        },
        "Delta_DEFER-": {
            "n": len(defer_neg),
            "mean": mean([d["delta_utility"] for d in defer_neg]) if defer_neg else 0.0,
            "ci": list(bootstrap_ci([d["delta_utility"] for d in defer_neg])) if defer_neg else [0, 0],
        },
        "Delta_ANSWER": {
            "n": len(answer_neg),
            "mean": mean([d["delta_utility"] for d in answer_neg]) if answer_neg else 0.0,
            "ci": list(bootstrap_ci([d["delta_utility"] for d in answer_neg])) if answer_neg else [0, 0],
        },
    }

    # I_phase = Delta_T2+ - Delta_DEFER-
    contrasts["I_phase"] = {
        "mean": contrasts["Delta_T2+"]["mean"] - contrasts["Delta_DEFER-"]["mean"],
        "note": "Delta_T2+ - Delta_DEFER-. Both groups can have expected terminal DEFER, so this removes the trivial explanation that R1 simply helps DEFER tasks.",
    }

    # Cost metrics
    cost = {
        "Delta_Steps_T2+": {
            "mean": mean([d["delta_steps"] for d in t2_pos]) if t2_pos else 0.0,
            "ci": list(bootstrap_ci([d["delta_steps"] for d in t2_pos])) if t2_pos else [0, 0],
        },
        "P_step_limit_R1_T2+": {
            "rate": sum(d["r1_hit_step_limit"] for d in t2_pos) / len(t2_pos) if t2_pos else 0.0,
            "n": len(t2_pos),
        },
        "P_step_limit_A1_T2+": {
            "rate": sum(not d["r1_hit_step_limit"] and d["a1_steps"] >= 10 for d in t2_pos) / len(t2_pos) if t2_pos else 0.0,
            "n": len(t2_pos),
        },
        "mean_redundant_R1_T2+": mean([d["r1_redundant"] for d in t2_pos]) if t2_pos else 0.0,
        "mean_redundant_A1_T2+": mean([d["a1_redundant"] for d in t2_pos]) if t2_pos else 0.0,
    }

    # False T2 on controls
    false_t2 = {
        "DEFER_CONTROL": sum(d["r1_t2_triggered"] for d in defer_neg),
        "ANSWER_CONTROL": sum(d["r1_t2_triggered"] for d in answer_neg),
    }

    return {
        "contrasts": contrasts,
        "cost_metrics": cost,
        "per_stratum": per_stratum,
        "false_t2_on_controls": false_t2,
        "n_pairs": len(deltas),
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="I3.15c factorial runner")
    parser.add_argument("--backend", default="local", choices=["local", "deepseek", "openrouter"])
    parser.add_argument("--openrouter-model", default="openai/gpt-4o-mini-2024-07-18",
                        help="OpenRouter model to use.")
    parser.add_argument("--n-per-cell", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-levels", nargs="+",
                        default=["Q3_RERANKED"])
    parser.add_argument("--arms", nargs="+",
                        default=["A1_INFERRED", "R1_INFERRED"])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--ports", nargs="+",
                        default=["8080"],
                        help="Local llama.cpp server ports to round-robin across.")
    parser.add_argument("--run-experiment", action="store_true",
                        help="Actually run the full experiment. Without this flag, "
                             "only structural qualification and gates are run.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a small smoke test (12 trajectories) instead of the full experiment.")
    args = parser.parse_args()

    print("=" * 80)
    print("I3.15c: Phase-Eligible Balanced Benchmark — Full Factorial Runner")
    print("=" * 80)

    # Print frozen identities
    print(f"\nBenchmark: {BENCHMARK_V1['benchmark_id']} ({BENCHMARK_V1['version']})")
    print(f"  Strata: {BENCHMARK_V1['strata']}")
    print(f"  Retrieval systems (this run): {args.retrieval_levels}")
    print(f"  Arms: {BENCHMARK_V1['arms']}")
    print(f"  Ports: {args.ports}")
    print(f"  Tasks: {BENCHMARK_V1['n_per_cell']} per cell x "
          f"{BENCHMARK_V1['n_strata']} strata x {BENCHMARK_V1['n_retrieval_difficulty']} retrieval difficulty")
    print(f"  = {BENCHMARK_V1['total_unique_tasks']} unique tasks")
    print(f"  Trajectories: {BENCHMARK_V1['total_unique_tasks']} x "
          f"{BENCHMARK_V1['n_retrieval_systems']} retrieval x {BENCHMARK_V1['n_arms']} arms")
    print(f"  = {BENCHMARK_V1['total_potential_trajectories']} potential trajectories")

    print(f"\nInference: {FROZEN_INFERENCE_CONFIG['config_id']}")
    if args.backend == "openrouter":
        print(f"  Backend: OpenRouter (NOT frozen identity)")
        print(f"  Model: {args.openrouter_model}")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Response format: json_schema")
        print(f"  WARNING: Results from this backend are not the pinned-model experiment.")
    else:
        print(f"  Model: {FROZEN_INFERENCE_CONFIG['model_name']}")
        print(f"  Reasoning: {FROZEN_INFERENCE_CONFIG['reasoning']} (budget={FROZEN_INFERENCE_CONFIG['reasoning_budget']})")
        print(f"  Max tokens: {FROZEN_INFERENCE_CONFIG['max_tokens']}")
        print(f"  Response format: {FROZEN_INFERENCE_CONFIG['response_format']}")

    print(f"\nProtocol: {EXPERIMENT_PROTOCOL_V1['protocol_id']}")
    print(f"  Pre-registered contrasts: {list(EXPERIMENT_PROTOCOL_V1['pre_registered_contrasts'].keys())}")
    print(f"  Desired: {EXPERIMENT_PROTOCOL_V1['desired_signature']}")

    # Generate tasks
    tasks = generate_i3_15c_corpus(n_per_cell=args.n_per_cell, seed=args.seed)
    print(f"\nGenerated {len(tasks)} tasks")

    # Structural qualification (zero LLM calls)
    print("\n" + "=" * 80)
    print("STRUCTURAL QUALIFICATION (zero LLM calls)")
    print("=" * 80)
    validation = validate_t2_eligibility(tasks)
    print(f"  T2 positive expected: {validation['t2_positive_expected']}")
    print(f"  T2 positive reachable (gold): {validation['t2_positive_reachable_gold']}")
    print(f"  T2 negative expected: {validation['t2_negative_expected']}")
    print(f"  T2 negative incorrectly reachable (gold): {validation['t2_negative_incorrectly_reachable_gold']}")
    print(f"  Late T2 initial false (correct): {validation['late_t2_initial_false']}")
    print(f"  Late T2 initial incorrectly true: {validation['late_t2_initial_incorrectly_true']}")
    print(f"  Immediate T2 initial true (correct): {validation['immediate_t2_initial_true']}")
    print(f"  Immediate T2 initial incorrectly false: {validation['immediate_t2_initial_incorrectly_false']}")
    print(f"  PASSED: {validation['passed']}")
    print()
    for stratum, info in sorted(validation["per_stratum"].items()):
        print(f"  {stratum}: n={info['n']} "
              f"t2_initial_true={info['t2_initial_true']} "
              f"t2_gold_true={info['t2_gold_true']}")

    if not validation["passed"]:
        print("\nSTRUCTURAL QUALIFICATION FAILED — aborting before any model calls.")
        sys.exit(1)

    # Save structural validation
    output_dir = REPO_ROOT / "experiments" / "v2b_i3_15" / "development" / "i3_15c_factorial"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "structural_validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)

    # Smoke test mode
    if args.smoke:
        print("\n" + "=" * 80)
        print("SMOKE TEST MODE: 12 trajectories")
        print("=" * 80)

        # Select 6 tasks: 2 from each T2-positive subtype, 1 DEFER_CONTROL, 1 ANSWER_CONTROL
        selected = []
        for stratum_prefix in ["t2_conflict_immediate", "t2_conflict_late"]:
            stratum_tasks = [t for t in tasks if t.evidence_task.category.startswith(stratum_prefix)]
            selected.extend(stratum_tasks[:2])
        selected.extend([t for t in tasks if t.evidence_task.category.startswith("defer_control")][:1])
        selected.extend([t for t in tasks if t.evidence_task.category.startswith("answer_control")][:1])

        print(f"Selected {len(selected)} tasks for smoke test")

        corpus_passages, corpus_by_text, corpus_by_id, chunks, corpus_sha = _get_cached_corpus()
        print(f"Corpus: {len(chunks)} passages, SHA256: {corpus_sha[:16]}...")

        # Build work items for parallel execution
        smoke_work = []
        for i, task in enumerate(selected):
            for j, arm in enumerate(args.arms):
                if args.backend == "openrouter":
                    base_url = None
                else:
                    port = args.ports[(i + j) % len(args.ports)]
                    base_url = f"http://127.0.0.1:{port}/v1"
                smoke_work.append((
                    task, "Q3_RERANKED", arm,
                    chunks, corpus_by_text, corpus_by_id,
                    args.max_tokens, base_url, args.backend,
                    args.openrouter_model,
                ))

        n_smoke_workers = 12 if args.backend == "openrouter" else len(args.ports)

        def _smoke_worker(wi):
            (task, retrieval_level, arm,
             chunks, corpus_by_text, corpus_by_id,
             max_tokens, base_url, backend_type, openrouter_model) = wi
            et = task.evidence_task
            try:
                t0 = time.time()
                result = run_single_trajectory(
                    task, retrieval_level, arm,
                    chunks, corpus_by_text, corpus_by_id,
                    max_tokens=max_tokens,
                    base_url=base_url,
                    backend_type=backend_type,
                    openrouter_model=openrouter_model,
                )
                result["wall_time_s"] = round(time.time() - t0, 1)
            except Exception as exc:
                result = {
                    "task_id": et.task_id,
                    "category": et.category,
                    "arm": arm,
                    "retrieval_level": retrieval_level,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "wall_time_s": 0.0,
                }
            return result

        results = []
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=n_smoke_workers) as executor:
            futures = {executor.submit(_smoke_worker, wi): wi for wi in smoke_work}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                actions = result.get("continuation_actions", [])
                t2 = result.get("r1_triggered", False)
                print(f"  {result.get('task_id')} {result.get('arm')}: "
                      f"actions={actions} "
                      f"terminal={result.get('terminal_action')} "
                      f"t2={t2} "
                      f"time={result['wall_time_s']}s")
        print(f"  Total smoke time: {time.time() - t_start:.1f}s")

        # Save smoke results
        with open(output_dir / "smoke_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Compute contrasts
        analysis = compute_contrasts(results)
        print("\nSmoke contrasts:")
        for name, info in analysis["contrasts"].items():
            print(f"  {name}: mean={info.get('mean', 0):.3f} ci={info.get('ci', [0,0])}")
        print(f"  False T2 on controls: {analysis['false_t2_on_controls']}")

        with open(output_dir / "smoke_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        print(f"\nSmoke results saved to {output_dir}")
        return

    # Full experiment
    if not args.run_experiment:
        print("\n" + "=" * 80)
        print("STRUCTURAL QUALIFICATION PASSED")
        print("The full experiment was NOT started.")
        print("Pass --run-experiment to run the full factorial experiment.")
        print("Pass --smoke to run a 12-trajectory smoke test.")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("FULL FACTORIAL EXPERIMENT")
    print("=" * 80)

    corpus_passages, corpus_by_text, corpus_by_id, chunks, corpus_sha = _get_cached_corpus()
    print(f"Corpus: {len(chunks)} passages, SHA256: {corpus_sha[:16]}...")

    n_tasks = len(tasks)
    n_retrieval = len(args.retrieval_levels)
    n_arms = len(args.arms)
    total = n_tasks * n_retrieval * n_arms
    print(f"Trajectories: {n_tasks} tasks x {n_retrieval} retrieval x {n_arms} arms = {total}")

    # Pre-retrieve evidence for all tasks (avoids loading retrieval model in workers)
    print("\nPre-retrieving evidence for all tasks...")
    t_ret_start = time.time()
    pre_retrieved: dict[tuple[str, str], list] = {}
    for task in tasks:
        et = task.evidence_task
        required_ids = get_required_passage_ids(task, corpus_by_text)
        for retrieval_level in args.retrieval_levels:
            if retrieval_level == "Q4_ORACLE":
                retrieved_passages = [corpus_by_id[pid] for pid in required_ids if pid in corpus_by_id]
            else:
                retriever = _get_cached_retriever(retrieval_level, chunks)
                retrieved = retriever.search(et.task_summary, top_k=TOP_K)
                retrieved_passages = [
                    corpus_by_id[c.chunk_id] for c, _ in retrieved
                    if c.chunk_id in corpus_by_id
                ]
            pre_retrieved[(et.task_id, retrieval_level)] = retrieved_passages
    print(f"Pre-retrieval done in {time.time() - t_ret_start:.1f}s")

    # Build work items
    work_items = []
    for i, task in enumerate(tasks):
        for j, retrieval_level in enumerate(args.retrieval_levels):
            for k, arm in enumerate(args.arms):
                if args.backend == "openrouter":
                    base_url = None
                else:
                    port = args.ports[(i + j + k) % len(args.ports)]
                    base_url = f"http://127.0.0.1:{port}/v1"
                retrieved_passages = pre_retrieved[(task.evidence_task.task_id, retrieval_level)]
                work_items.append((
                    task, retrieval_level, arm,
                    chunks, corpus_by_text, corpus_by_id,
                    args.max_tokens, base_url, args.backend,
                    args.openrouter_model, retrieved_passages,
                ))

    n_workers = 8 if args.backend == "openrouter" else len(args.ports)

    def _worker(args_tuple):
        (task, retrieval_level, arm,
         chunks, corpus_by_text, corpus_by_id,
         max_tokens, base_url, backend_type, openrouter_model,
         retrieved_passages) = args_tuple
        et = task.evidence_task
        try:
            t0 = time.time()
            result = run_single_trajectory(
                task, retrieval_level, arm,
                chunks, corpus_by_text, corpus_by_id,
                max_tokens=max_tokens,
                base_url=base_url,
                backend_type=backend_type,
                openrouter_model=openrouter_model,
                pre_retrieved_passages=retrieved_passages,
            )
            result["wall_time_s"] = round(time.time() - t0, 1)
        except Exception as exc:
            result = {
                "task_id": et.task_id,
                "category": et.category,
                "arm": arm,
                "retrieval_level": retrieval_level,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            result["wall_time_s"] = 0.0
        return result

    results = []
    completed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, wi): wi for wi in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 10 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {result.get('task_id')} "
                      f"{result.get('arm')} {result.get('retrieval_level')} "
                      f"({result.get('wall_time_s', 0)}s) "
                      f"ETA: {eta/60:.0f}min")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Compute contrasts
    analysis = compute_contrasts(results)

    # Save analysis
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nPre-registered contrasts:")
    for name, info in analysis["contrasts"].items():
        mean = info.get("mean", 0)
        ci = info.get("ci", [0, 0])
        n = info.get("n", 0)
        print(f"  {name}: mean={mean:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}] n={n}")

    print("\nCost metrics:")
    for name, info in analysis["cost_metrics"].items():
        if isinstance(info, dict) and "mean" in info:
            print(f"  {name}: mean={info['mean']:.3f}")
        elif isinstance(info, dict) and "rate" in info:
            print(f"  {name}: rate={info['rate']:.3f} n={info.get('n', 0)}")
        else:
            print(f"  {name}: {info}")

    print(f"\nFalse T2 on controls: {analysis['false_t2_on_controls']}")

    print("\nPer-stratum:")
    for stratum, info in sorted(analysis["per_stratum"].items()):
        print(f"  {stratum}: n={info['n']} "
              f"ΔU={info['mean_delta_utility']:.4f} "
              f"ΔSteps={info['mean_delta_steps']:.2f} "
              f"R1_T2={info['r1_t2_triggered_count']} "
              f"R1_step_limit={info['r1_step_limit_count']}")

    # Check desired signature
    print("\nDesired signature check:")
    sig = EXPERIMENT_PROTOCOL_V1["desired_signature"]
    d_t2 = analysis["contrasts"]["Delta_T2+"]["mean"]
    i_phase = analysis["contrasts"]["I_phase"]["mean"]
    d_defer = analysis["contrasts"]["Delta_DEFER-"]["mean"]
    d_answer = analysis["contrasts"]["Delta_ANSWER"]["mean"]
    false_t2 = (analysis["false_t2_on_controls"]["DEFER_CONTROL"] +
                analysis["false_t2_on_controls"]["ANSWER_CONTROL"])

    print(f"  Delta_T2+ > 0: {'YES' if d_t2 > 0 else 'NO'} ({d_t2:.4f})")
    print(f"  I_phase > 0: {'YES' if i_phase > 0 else 'NO'} ({i_phase:.4f})")
    print(f"  Delta_DEFER- ~ 0: {'YES' if abs(d_defer) < 0.1 else 'NO'} ({d_defer:.4f})")
    print(f"  Delta_ANSWER ~ 0: {'YES' if abs(d_answer) < 0.1 else 'NO'} ({d_answer:.4f})")
    print(f"  false T2 on controls = 0: {'YES' if false_t2 == 0 else 'NO'} ({false_t2})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
