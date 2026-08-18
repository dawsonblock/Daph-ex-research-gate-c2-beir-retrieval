#!/usr/bin/env python3
"""Build Packet-Level Counterfactual Dataset for V2B-I3.5.2b.

For every decision state s_t encountered along the baseline (AWARE_NO_GOVERNOR)
trajectory in development (300 tasks), this script measures the PACKET TREATMENT
effect, not just the governor ranking effect.

At each baseline state s_t:
  1. Reconstruct the exact controller-visible state from the recorded baseline trajectory.
  2. Build the BASE packet (identical to what was sent in I3.5.1).
  3. Build the GOVERNOR packet (governor.assess() + build_governor_packet).
  4. Call DeepSeek with the GOVERNOR packet → a_gov_packet_model.
  5. Use the recorded baseline proposed_action as a_base_model.
  6. Look up exact oracle Q(s, a) for a_base_model, a_gov_packet_model, and gov_top.
  7. Compute:
       delta_q_governor_top    = Q(s, gov_top)           - Q(s, a_base_model)
       delta_q_packet_treatment = Q(s, a_gov_packet_model) - Q(s, a_base_model)
  8. Decompose:
       A_ranking    = delta_q_governor_top     (governor ranking intelligence)
       A_treatment  = delta_q_packet_treatment (packet treatment effect)
       A_realization = A_treatment - A_ranking (model's ability to convert gov info into behavior)
  9. Continue the trajectory with a_base_model (the recorded baseline action).
     The counterfactual a_gov_packet_model is NEVER executed.

Q-value source tracking:
  Every Q-value lookup records whether the value came from:
    - oracle_q_values  (primary oracle table)
    - proposal_q_values (proposal transition table)
    - fallback_penalty  (fixed penalty for illegal/unexpected actions)

Usage:
    DEEPSEEK_API_KEY=... python scripts/build_i3_5_2_packet_counterfactual_dataset.py \\
        --split structure_dev_v2 \\
        --results experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json \\
        --output-dir experiments/v2b_i3_5_2/development \\
        --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.actions import V2B_ACTIONS
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import DecisionSummary
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor,
    initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.packet_builder import (
    build_base_packet, build_governor_packet,
    packet_json, packet_sha256, assert_no_evaluator_leakage,
)
from hrm_adaptive_memory.executive.i3_5_1.model_prompt import SYSTEM_PROMPT
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor,
    initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_transition_table import (
    OraclePolicyTable,
    build_oracle_policy_table,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.model_backend import DeepSeekBackend, StubBackend
from hrm_adaptive_memory.executive.model_decoder import decode_output
from hrm_adaptive_memory.executive.pinned_model_controller import (
    BACKEND_ERROR_PROPOSAL, FAIL_CLOSED_PROPOSAL,
)
from hrm_adaptive_memory.executive.policy import load_frozen_policy
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.selective_governor.features import (
    extract_features,
)

# Standard action list
VALID_ACTIONS = tuple(
    a for a in DecisionAction
    if a.value in ("ANSWER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE", "DEFER", "STOP")
)

# Fixed fallback penalties (must match build_i3_5_2_shadow_dataset.py)
FALLBACK_PENALTIES: dict[DecisionAction, float] = {
    DecisionAction.ANSWER: -125.11,
    DecisionAction.DEFER: -30.11,
    DecisionAction.STOP: -30.11,
}
DEFAULT_FALLBACK = -125.0


@dataclass(frozen=True)
class QLookupResult:
    """Result of looking up Q(s, a) with source tracking."""
    value: float
    source: str  # "oracle_q_values", "proposal_q_values", "fallback_penalty"


def lookup_q_with_source(
    table: OraclePolicyTable, state_id: str, action: DecisionAction,
) -> QLookupResult:
    """Look up Q(s, a) and record whether it came from oracle, proposal, or fallback."""
    q = table.q_values.get((state_id, action))
    if q is not None:
        return QLookupResult(value=q, source="oracle_q_values")
    pq = table.proposal_q_values.get((state_id, action))
    if pq is not None:
        return QLookupResult(value=pq, source="proposal_q_values")
    # Fallback penalty for illegal/unexpected action
    penalty = FALLBACK_PENALTIES.get(action, DEFAULT_FALLBACK)
    return QLookupResult(value=penalty, source="fallback_penalty")


def classify_delta_q(delta_q: float, threshold: float = 5.0) -> str:
    if delta_q > threshold:
        return "HELP"
    elif delta_q < -threshold:
        return "HARM"
    return "NEUTRAL"


def call_model_with_packet(
    backend: DeepSeekBackend,
    packet: dict[str, Any],
    task_id: str,
    condition_tag: str,
    pair_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    strict_json: bool = True,
) -> tuple[str | None, str | None, str | None, bool]:
    """Call the model with a packet and return (action_str, reason_code, raw_output, success).

    Returns (None, None, raw_or_none, False) on backend/decoder failure.
    """
    assert_no_evaluator_leakage(packet)
    user_prompt = packet_json(packet)

    backend.task_id = task_id
    backend.condition = condition_tag
    backend.pair_id = pair_id

    try:
        call_result = backend.generate(
            system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt,
            temperature=temperature, max_tokens=max_tokens,
        )
    except Exception:
        return None, None, None, False

    raw_output = call_result.raw_output
    outcome = decode_output(raw_output, strict=strict_json)
    if outcome.valid and outcome.proposal:
        return outcome.proposal.action.value, outcome.proposal.reason_code, raw_output, True
    return None, None, raw_output, False


@dataclass
class PacketCounterfactualState:
    """One state-level packet counterfactual record."""
    task_id: str
    topology_id: str
    step_id: int
    state_id: str
    # Actions
    base_model_action: str          # a_base_model (recorded from I3.5.1)
    governor_top_action: str        # a_gov_top (governor.assess() recommendation)
    governor_packet_model_action: str | None  # a_gov_packet_model (counterfactual call)
    governor_packet_reason_code: str | None
    governor_packet_success: bool
    # Agreement flags
    gov_model_agreement: bool | None  # gov_top == a_gov_packet_model
    base_gov_agreement: bool           # a_base_model == gov_top
    base_packet_model_agreement: bool | None  # a_base_model == a_gov_packet_model
    # Q-values
    q_base: float
    q_gov_top: float
    q_gov_packet_model: float | None
    q_base_source: str
    q_gov_top_source: str
    q_gov_packet_model_source: str | None
    # Delta-Q decomposition
    delta_q_governor_top: float       # A_ranking = Q(gov_top) - Q(base)
    delta_q_packet_treatment: float | None  # A_treatment = Q(gov_packet_model) - Q(base)
    delta_q_realization: float | None  # A_realization = A_treatment - A_ranking
    # Labels
    label_governor_top: str
    label_packet_treatment: str | None
    # Governor diagnostics
    governor_reason_code: str | None
    # Features
    features: dict[str, Any]
    # Packet hashes
    base_packet_sha256: str
    governor_packet_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "topology_id": self.topology_id,
            "step_id": self.step_id,
            "state_id": self.state_id,
            "base_model_action": self.base_model_action,
            "governor_top_action": self.governor_top_action,
            "governor_packet_model_action": self.governor_packet_model_action,
            "governor_packet_reason_code": self.governor_packet_reason_code,
            "governor_packet_success": self.governor_packet_success,
            "gov_model_agreement": self.gov_model_agreement,
            "base_gov_agreement": self.base_gov_agreement,
            "base_packet_model_agreement": self.base_packet_model_agreement,
            "q_base": round(self.q_base, 4),
            "q_gov_top": round(self.q_gov_top, 4),
            "q_gov_packet_model": round(self.q_gov_packet_model, 4) if self.q_gov_packet_model is not None else None,
            "q_base_source": self.q_base_source,
            "q_gov_top_source": self.q_gov_top_source,
            "q_gov_packet_model_source": self.q_gov_packet_model_source,
            "delta_q_governor_top": round(self.delta_q_governor_top, 4),
            "delta_q_packet_treatment": round(self.delta_q_packet_treatment, 4) if self.delta_q_packet_treatment is not None else None,
            "delta_q_realization": round(self.delta_q_realization, 4) if self.delta_q_realization is not None else None,
            "label_governor_top": self.label_governor_top,
            "label_packet_treatment": self.label_packet_treatment,
            "governor_reason_code": self.governor_reason_code,
            "features": self.features,
            "base_packet_sha256": self.base_packet_sha256,
            "governor_packet_sha256": self.governor_packet_sha256,
        }


def process_one_task(
    block: dict[str, Any],
    task_map: dict[str, Any],
    policy: Any,
    utility: MetareasoningUtility,
    split_bm: Any,
    api_key: str,
    max_steps: int = 24,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    strict_json: bool = True,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    """Process one task block: replay baseline trajectory, call model with governor packet at each state.

    Returns (state_records, task_summary, model_calls, backend_errors).
    """
    task_id = block["task_id"]
    task = task_map[task_id]
    budget = split_bm.budget_for(task)
    table = build_oracle_policy_table(task=task, policy=policy, utility=utility, budget=budget)

    b_traj = block["trajectories"]["AWARE_NO_GOVERNOR"]
    b_steps = b_traj["steps"]

    # Replay baseline trajectory
    resources = ResourceState(budget)
    i3_runtime = initial_i3_runtime(task, resources)
    adapter = _I3TaskAdapter(task)
    t_runtime = init_task_runtime(adapter, ResourceState(budget))

    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()
    governor = GeneralGovernor()
    cond = get_condition(ConditionID.AWARE_GOVERNOR)

    # Dedicated backend for this task's counterfactual calls
    if dry_run:
        backend = StubBackend()
    else:
        backend = DeepSeekBackend(_api_key=api_key)

    prior_decisions: list[DecisionSummary] = []
    prior_outcomes: list[str] = []

    state_records: list[dict[str, Any]] = []
    model_calls = 0
    backend_errors = 0

    for step_idx, step_data in enumerate(b_steps):
        state_id = table.state_id_for(i3_runtime)

        # Recorded baseline model action (from I3.5.1)
        a_base_str = step_data["proposed_action"]
        a_base = DecisionAction(a_base_str)

        # Build observation (identical to I3.5.1)
        obs = build_observation(
            t_runtime, task, cond,
            tuple(prior_decisions), tuple(prior_outcomes),
        )
        p_actions = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions
        )
        p_outcomes = tuple(prior_outcomes)

        # Governor assessment
        frame = governor.assess(
            observation=obs,
            remaining_steps=max_steps - step_idx,
            prior_actions=p_actions,
            prior_outcomes=p_outcomes,
        )
        gov_top_str = frame.governor_top_action or a_base_str
        gov_top = DecisionAction(gov_top_str)

        # Build packets
        base_packet = build_base_packet(obs)
        gov_packet = build_governor_packet(obs, frame)
        base_pkt_sha = packet_sha256(base_packet)
        gov_pkt_sha = packet_sha256(gov_packet)

        # Extract features
        features = extract_features(
            obs,
            remaining_steps=max_steps - step_idx,
            prior_actions=p_actions,
            prior_outcomes=p_outcomes,
        )

        # Q-value lookups with source tracking
        q_base_result = lookup_q_with_source(table, state_id, a_base)
        q_gov_top_result = lookup_q_with_source(table, state_id, gov_top)

        # Counterfactual: call model with GOVERNOR packet
        # This action is NOT executed — it's shadow inference
        pair_id = f"i352b:{task_id}:step:{step_idx}"
        model_calls += 1
        a_gov_pkt_str, gov_pkt_reason, gov_pkt_raw, gov_pkt_success = call_model_with_packet(
            backend, gov_packet, task_id,
            condition_tag=f"AWARE_GOVERNOR_PACKET_COUNTERFACTUAL",
            pair_id=pair_id,
            temperature=temperature,
            max_tokens=max_tokens,
            strict_json=strict_json,
        )

        if not gov_pkt_success:
            backend_errors += 1

        # Q-value for governor packet model action (if we got one)
        q_gov_pkt_result: QLookupResult | None = None
        a_gov_pkt: DecisionAction | None = None
        if a_gov_pkt_str is not None:
            a_gov_pkt = DecisionAction(a_gov_pkt_str)
            q_gov_pkt_result = lookup_q_with_source(table, state_id, a_gov_pkt)

        # Compute deltas
        delta_q_gov_top = q_gov_top_result.value - q_base_result.value
        delta_q_pkt_treatment: float | None = None
        delta_q_realization: float | None = None
        if q_gov_pkt_result is not None:
            delta_q_pkt_treatment = q_gov_pkt_result.value - q_base_result.value
            delta_q_realization = delta_q_pkt_treatment - delta_q_gov_top

        # Labels
        label_gov_top = classify_delta_q(delta_q_gov_top)
        label_pkt_treatment = (
            classify_delta_q(delta_q_pkt_treatment) if delta_q_pkt_treatment is not None else None
        )

        # Agreement flags
        gov_model_agreement = (gov_top_str == a_gov_pkt_str) if a_gov_pkt_str is not None else None
        base_gov_agreement = (a_base_str == gov_top_str)
        base_pkt_model_agreement = (
            (a_base_str == a_gov_pkt_str) if a_gov_pkt_str is not None else None
        )

        record = PacketCounterfactualState(
            task_id=task_id,
            topology_id=task.semantic_structure_coarse,
            step_id=step_idx,
            state_id=state_id,
            base_model_action=a_base_str,
            governor_top_action=gov_top_str,
            governor_packet_model_action=a_gov_pkt_str,
            governor_packet_reason_code=gov_pkt_reason,
            governor_packet_success=gov_pkt_success,
            gov_model_agreement=gov_model_agreement,
            base_gov_agreement=base_gov_agreement,
            base_packet_model_agreement=base_pkt_model_agreement,
            q_base=q_base_result.value,
            q_gov_top=q_gov_top_result.value,
            q_gov_packet_model=q_gov_pkt_result.value if q_gov_pkt_result else None,
            q_base_source=q_base_result.source,
            q_gov_top_source=q_gov_top_result.source,
            q_gov_packet_model_source=q_gov_pkt_result.source if q_gov_pkt_result else None,
            delta_q_governor_top=delta_q_gov_top,
            delta_q_packet_treatment=delta_q_pkt_treatment,
            delta_q_realization=delta_q_realization,
            label_governor_top=label_gov_top,
            label_packet_treatment=label_pkt_treatment,
            governor_reason_code=frame.governor_reason_code,
            features=features.as_dict(),
            base_packet_sha256=base_pkt_sha,
            governor_packet_sha256=gov_pkt_sha,
        )
        state_records.append(record.as_dict())

        # Step the baseline environment with the RECORDED baseline action
        # The counterfactual governor-packet action is NEVER executed
        exec_res = oracle_executor.execute(i3_runtime, a_base)
        i3_runtime = exec_res.runtime
        t_runtime = task_executor.execute(t_runtime, a_base).runtime

        prior_decisions.append(DecisionSummary(
            f"{task_id}:step:{step_idx}",
            a_base.value,
            step_data["reason_code"],
            exec_res.outcome_code,
        ))
        prior_outcomes.append(exec_res.outcome_code)
        if exec_res.terminal:
            break

    # Task summary
    pkt_treatment_deltas = [
        r["delta_q_packet_treatment"] for r in state_records
        if r["delta_q_packet_treatment"] is not None
    ]
    gov_top_deltas = [r["delta_q_governor_top"] for r in state_records]

    task_summary = {
        "task_id": task_id,
        "topology_id": task.semantic_structure_coarse,
        "steps_count": len(state_records),
        "baseline_success": b_traj["task_success"],
        "baseline_utility": b_traj["realized_utility"],
        "model_calls": model_calls,
        "backend_errors": backend_errors,
        "mean_delta_q_governor_top": round(statistics.mean(gov_top_deltas), 4) if gov_top_deltas else 0.0,
        "mean_delta_q_packet_treatment": round(statistics.mean(pkt_treatment_deltas), 4) if pkt_treatment_deltas else None,
        "has_positive_gov_top": any(d > 5.0 for d in gov_top_deltas),
        "has_positive_pkt_treatment": any(
            d is not None and d > 5.0 for d in [r["delta_q_packet_treatment"] for r in state_records]
        ),
        "has_harmful_gov_top": any(d < -5.0 for d in gov_top_deltas),
        "has_harmful_pkt_treatment": any(
            d is not None and d < -5.0 for d in [r["delta_q_packet_treatment"] for r in state_records]
        ),
    }

    return state_records, task_summary, model_calls, backend_errors


def main():
    parser = argparse.ArgumentParser(
        description="Build I3.5.2b Packet-Level Counterfactual Dataset",
    )
    parser.add_argument("--split", default="structure_dev_v2")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_1/development/e21f63ff4fa9/results.json",
        help="Path to baseline I3.5.1 results.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--output-dir", default="experiments/v2b_i3_5_2/development")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--strict-json", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Use StubBackend instead of DeepSeek (no API calls)")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: DEEPSEEK_API_KEY not set (use --dry-run for testing)", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split(args.split)
    task_map = {t.task_id: t for t in split_bm.tasks}

    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]
    if args.max_tasks is not None:
        blocks = blocks[:args.max_tasks]
    print(f"Loaded {len(blocks)} task blocks from {args.results}")

    policy = load_frozen_policy(args.policy)
    utility = MetareasoningUtility.from_file(args.utility)

    print(f"\nProcessing {len(blocks)} tasks with {args.workers} workers...")
    print(f"Each state: 1 governor-packet model call (counterfactual, not executed)")
    print(f"Estimated model calls: ~{len(blocks) * 3} (mean trajectory length ~3)")

    all_state_records: list[dict[str, Any]] = []
    all_task_summaries: list[dict[str, Any]] = []
    total_model_calls = 0
    total_backend_errors = 0
    completed = 0
    t_start = time.monotonic()

    def run_one(item):
        idx, block = item
        return idx, process_one_task(
            block, task_map, policy, utility, split_bm, api_key,
            max_steps=24,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            strict_json=args.strict_json,
            dry_run=args.dry_run,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (i, b)): i for i, b in enumerate(blocks)}
        for future in as_completed(futures):
            idx, (state_records, task_summary, mc, be) = future.result()
            all_state_records.extend(state_records)
            all_task_summaries.append(task_summary)
            total_model_calls += mc
            total_backend_errors += be
            completed += 1
            if completed % args.progress_every == 0:
                elapsed = time.monotonic() - t_start
                rate = completed / elapsed
                eta = (len(blocks) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(blocks)}] {task_summary['task_id']}: "
                      f"{len(state_records)} states, {mc} calls, "
                      f"mean ΔQ_gov_top={task_summary['mean_delta_q_governor_top']:.1f}, "
                      f"mean ΔQ_pkt={task_summary['mean_delta_q_packet_treatment']}, "
                      f"elapsed={elapsed:.0f}s, eta={eta:.0f}s")

    elapsed = time.monotonic() - t_start
    print(f"\nCompleted {completed} tasks in {elapsed:.0f}s")
    print(f"Total model calls: {total_model_calls}")
    print(f"Total backend errors: {total_backend_errors}")
    print(f"Total state records: {len(all_state_records)}")

    # Sort by task_id then step_id for deterministic output
    all_state_records.sort(key=lambda r: (r["task_id"], r["step_id"]))
    all_task_summaries.sort(key=lambda t: t["task_id"])

    # 1. Save JSONL state ledger
    states_path = out_dir / "packet_counterfactual_states_v1.jsonl"
    with open(states_path, "w") as f:
        for r in all_state_records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nSaved state dataset: {states_path} ({len(all_state_records)} records)")

    # 2. Compute summary statistics
    total_states = len(all_state_records)
    states_with_pkt = [r for r in all_state_records if r["delta_q_packet_treatment"] is not None]
    pkt_count = len(states_with_pkt)

    # Q-value source tracking
    q_source_counter = Counter()
    for r in all_state_records:
        q_source_counter[r["q_base_source"]] += 1
        q_source_counter[r["q_gov_top_source"]] += 1
        if r["q_gov_packet_model_source"] is not None:
            q_source_counter[r["q_gov_packet_model_source"]] += 1

    # Governor-top labels
    gov_top_labels = Counter(r["label_governor_top"] for r in all_state_records)
    pkt_labels = Counter(r["label_packet_treatment"] for r in states_with_pkt)

    # Agreement rates
    gov_model_agree = sum(
        1 for r in all_state_records
        if r["gov_model_agreement"] is True
    )
    base_gov_agree = sum(
        1 for r in all_state_records
        if r["base_gov_agreement"] is True
    )
    base_pkt_agree = sum(
        1 for r in states_with_pkt
        if r["base_packet_model_agreement"] is True
    )

    # Delta-Q distributions
    gov_top_deltas = [r["delta_q_governor_top"] for r in all_state_records]
    pkt_deltas = [r["delta_q_packet_treatment"] for r in states_with_pkt if r["delta_q_packet_treatment"] is not None]
    realization_deltas = [r["delta_q_realization"] for r in states_with_pkt if r["delta_q_realization"] is not None]

    # Substitution matrices
    gov_top_substitutions = Counter()
    gov_top_sub_deltas = defaultdict(list)
    pkt_substitutions = Counter()
    pkt_sub_deltas = defaultdict(list)
    for r in all_state_records:
        gov_top_substitutions[(r["base_model_action"], r["governor_top_action"])] += 1
        gov_top_sub_deltas[(r["base_model_action"], r["governor_top_action"])].append(
            r["delta_q_governor_top"])
        if r["governor_packet_model_action"] is not None:
            pkt_substitutions[(r["base_model_action"], r["governor_packet_model_action"])] += 1
            pkt_sub_deltas[(r["base_model_action"], r["governor_packet_model_action"])].append(
                r["delta_q_packet_treatment"])

    def build_sub_matrix(subs, deltas):
        out = []
        for (b, g), cnt in subs.most_common():
            d = deltas[(b, g)]
            out.append({
                "base_action": b,
                "counterfactual_action": g,
                "count": cnt,
                "percentage": round(cnt / total_states, 4),
                "mean_delta_q": round(statistics.mean(d), 4),
                "min_delta_q": round(min(d), 4),
                "max_delta_q": round(max(d), 4),
                "help_count": sum(1 for x in d if x > 5.0),
                "harm_count": sum(1 for x in d if x < -5.0),
                "neutral_count": sum(1 for x in d if -5.0 <= x <= 5.0),
            })
        return out

    # Build summary
    summary = {
        "schema": "DAPH_V2B_I3_5_2_PACKET_COUNTERFACTUAL_V1",
        "schema_version": 1,
        "split": args.split,
        "total_tasks": len(all_task_summaries),
        "total_decision_states": total_states,
        "states_with_packet_response": pkt_count,
        "states_with_backend_error": total_backend_errors,
        "total_model_calls": total_model_calls,
        "elapsed_seconds": round(elapsed, 2),
        "q_value_source_distribution": dict(q_source_counter.most_common()),
        "governor_top_label_distribution": {
            "HELP": {"count": gov_top_labels.get("HELP", 0),
                     "rate": round(gov_top_labels.get("HELP", 0) / total_states, 4)},
            "NEUTRAL": {"count": gov_top_labels.get("NEUTRAL", 0),
                        "rate": round(gov_top_labels.get("NEUTRAL", 0) / total_states, 4)},
            "HARM": {"count": gov_top_labels.get("HARM", 0),
                     "rate": round(gov_top_labels.get("HARM", 0) / total_states, 4)},
        },
        "packet_treatment_label_distribution": {
            "HELP": {"count": pkt_labels.get("HELP", 0),
                     "rate": round(pkt_labels.get("HELP", 0) / pkt_count, 4) if pkt_count else None},
            "NEUTRAL": {"count": pkt_labels.get("NEUTRAL", 0),
                        "rate": round(pkt_labels.get("NEUTRAL", 0) / pkt_count, 4) if pkt_count else None},
            "HARM": {"count": pkt_labels.get("HARM", 0),
                     "rate": round(pkt_labels.get("HARM", 0) / pkt_count, 4) if pkt_count else None},
        },
        "agreement_rates": {
            "gov_top_vs_gov_packet_model": {
                "agreement_count": gov_model_agree,
                "total": pkt_count,
                "rate": round(gov_model_agree / pkt_count, 4) if pkt_count else None,
            },
            "base_vs_gov_top": {
                "agreement_count": base_gov_agree,
                "total": total_states,
                "rate": round(base_gov_agree / total_states, 4),
            },
            "base_vs_gov_packet_model": {
                "agreement_count": base_pkt_agree,
                "total": pkt_count,
                "rate": round(base_pkt_agree / pkt_count, 4) if pkt_count else None,
            },
        },
        "delta_q_statistics": {
            "governor_top": {
                "mean": round(statistics.mean(gov_top_deltas), 4) if gov_top_deltas else None,
                "median": round(statistics.median(gov_top_deltas), 4) if gov_top_deltas else None,
                "stdev": round(statistics.stdev(gov_top_deltas), 4) if len(gov_top_deltas) > 1 else None,
                "min": round(min(gov_top_deltas), 4) if gov_top_deltas else None,
                "max": round(max(gov_top_deltas), 4) if gov_top_deltas else None,
            },
            "packet_treatment": {
                "mean": round(statistics.mean(pkt_deltas), 4) if pkt_deltas else None,
                "median": round(statistics.median(pkt_deltas), 4) if pkt_deltas else None,
                "stdev": round(statistics.stdev(pkt_deltas), 4) if len(pkt_deltas) > 1 else None,
                "min": round(min(pkt_deltas), 4) if pkt_deltas else None,
                "max": round(max(pkt_deltas), 4) if pkt_deltas else None,
            },
            "realization": {
                "mean": round(statistics.mean(realization_deltas), 4) if realization_deltas else None,
                "median": round(statistics.median(realization_deltas), 4) if realization_deltas else None,
                "stdev": round(statistics.stdev(realization_deltas), 4) if len(realization_deltas) > 1 else None,
                "min": round(min(realization_deltas), 4) if realization_deltas else None,
                "max": round(max(realization_deltas), 4) if realization_deltas else None,
            },
        },
        "governor_top_substitution_matrix": build_sub_matrix(gov_top_substitutions, gov_top_sub_deltas),
        "packet_treatment_substitution_matrix": build_sub_matrix(pkt_substitutions, pkt_sub_deltas),
        "task_summaries": all_task_summaries,
    }

    summary_path = out_dir / "packet_counterfactual_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved summary: {summary_path}")

    # 3. Print summary
    print("\n" + "=" * 78)
    print("V2B-I3.5.2b PACKET-LEVEL COUNTERFACTUAL TREATMENT ANALYSIS")
    print("=" * 78)
    print(f"Total Decision States: {total_states} (across {len(all_task_summaries)} tasks)")
    print(f"States with packet response: {pkt_count}")
    print(f"Backend errors: {total_backend_errors}")
    print(f"Model calls: {total_model_calls}")

    print(f"\n--- Q-Value Source Distribution ---")
    for src, cnt in q_source_counter.most_common():
        print(f"  {src:<25}: {cnt}")

    print(f"\n--- Governor-Top Label Distribution (A_ranking) ---")
    for lbl in ("HELP", "NEUTRAL", "HARM"):
        c = gov_top_labels.get(lbl, 0)
        print(f"  {lbl:<8}: {c:>4} ({c/total_states:>5.1%})")

    print(f"\n--- Packet-Treatment Label Distribution (A_treatment) ---")
    for lbl in ("HELP", "NEUTRAL", "HARM"):
        c = pkt_labels.get(lbl, 0)
        r = c / pkt_count if pkt_count else 0
        print(f"  {lbl:<8}: {c:>4} ({r:>5.1%})")

    print(f"\n--- Agreement Rates ---")
    print(f"  gov_top == gov_packet_model:  {gov_model_agree}/{pkt_count} "
          f"({gov_model_agree/pkt_count:.1%})" if pkt_count else "  N/A")
    print(f"  base == gov_top:              {base_gov_agree}/{total_states} "
          f"({base_gov_agree/total_states:.1%})")
    print(f"  base == gov_packet_model:     {base_pkt_agree}/{pkt_count} "
          f"({base_pkt_agree/pkt_count:.1%})" if pkt_count else "  N/A")

    print(f"\n--- Delta-Q Statistics ---")
    print(f"  A_ranking    (gov_top):    mean={statistics.mean(gov_top_deltas):>8.2f}, "
          f"median={statistics.median(gov_top_deltas):>8.2f}")
    if pkt_deltas:
        print(f"  A_treatment  (pkt_model):  mean={statistics.mean(pkt_deltas):>8.2f}, "
              f"median={statistics.median(pkt_deltas):>8.2f}")
    if realization_deltas:
        print(f"  A_realization (treatment - ranking): mean={statistics.mean(realization_deltas):>8.2f}, "
              f"median={statistics.median(realization_deltas):>8.2f}")

    print(f"\n--- Top Governor-Top Substitutions ---")
    for row in summary["governor_top_substitution_matrix"][:8]:
        print(f"  {row['base_action']:<10} -> {row['counterfactual_action']:<10}: "
              f"N={row['count']:>3} | mean ΔQ={row['mean_delta_q']:>8.2f} | "
              f"HELP={row['help_count']} HARM={row['harm_count']}")

    print(f"\n--- Top Packet-Treatment Substitutions ---")
    for row in summary["packet_treatment_substitution_matrix"][:8]:
        print(f"  {row['base_action']:<10} -> {row['counterfactual_action']:<10}: "
              f"N={row['count']:>3} | mean ΔQ={row['mean_delta_q']:>8.2f} | "
              f"HELP={row['help_count']} HARM={row['harm_count']}")


if __name__ == "__main__":
    main()
