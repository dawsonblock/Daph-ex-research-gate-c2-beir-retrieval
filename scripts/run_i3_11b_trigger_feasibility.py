#!/usr/bin/env python3
"""I3.11b: Offline trigger-feasibility analysis for R1 Epistemic-Conflict Router.

Replays A1 trajectories from the I3.11a stress test and computes M3's
state estimator internally at each step (without exposing M3 context
to the LLM). Measures whether deterministic triggers can identify
M3 rescue opportunities before A1 fails.

Three triggers tested:
  T1: decision_state == INSUFFICIENT
  T2: all hypotheses eliminated by visible verified evidence
  T3: verified mutually incompatible evidence exists AND no visible
      unverified discriminator remains

Metrics:
  Coverage = P(trigger fires before A1 failure | M3 rescue)
  FalseActivation = P(trigger fires | A1 already succeeds)
  TriggerLatency = steps until first trigger activation

Usage:
    PYTHONPATH=. python scripts/run_i3_11b_trigger_feasibility.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
i3_7e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_7e)

from scripts.run_i3_11a_routing_stress import generate_decorrelated_corpus
from hrm_adaptive_memory.executive.evidence_benchmark import (
    initial_evidence_runtime, build_evidence_snapshot, EvidenceExecutor,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import VerificationState


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def replay_a1_with_m3_state(
    task: Any,
    a1_fork: dict,
    budget: ResourceBudget,
) -> list[dict]:
    """Replay A1 trajectory and compute M3 state at each step.

    Returns a list of per-step records with:
      - step index
      - A1 action
      - M3 decision_state (computed internally, not exposed to LLM)
      - M3 state summary fields
      - trigger flags (T1, T2, T3)
    """
    executor = EvidenceExecutor()
    runtime = initial_evidence_runtime(task, ResourceState(budget))

    actions = a1_fork["continuation_actions"]
    outcomes = a1_fork["continuation_outcomes"]

    current_rt = runtime
    prior_actions: list[str] = []
    prior_outcomes: list[str] = []
    step_records: list[dict] = []

    for step_i, (action_name, outcome) in enumerate(zip(actions, outcomes)):
        # Build snapshot BEFORE this action
        snap = build_evidence_snapshot(
            current_rt,
            prior_actions=tuple(prior_actions),
            prior_outcomes=tuple(prior_outcomes),
        )

        # Compute M3 state internally
        packet = i3_7e.build_mdsg_state_with_affordances_packet(snap)
        summary = packet["decision_state_summary"]
        state = summary["decision_state"]

        live_hyps = summary["live_hypotheses"]
        eliminated = summary["eliminated_hypotheses"]
        verified_support = summary["verified_support"]
        verified_contradictions = summary["verified_contradictions"]
        unverified_visible = summary.get("unverified_relevant_evidence", [])

        # T1: decision_state == INSUFFICIENT
        t1 = state == "INSUFFICIENT"

        # T2: all hypotheses eliminated by visible verified evidence
        n_hypotheses = len(task.hypotheses)
        t2 = len(eliminated) == n_hypotheses and n_hypotheses > 0

        # T3: verified mutually incompatible evidence exists AND
        #     no visible unverified discriminator remains
        # "Verified mutually incompatible" = verified evidence supports
        # both sides (both H1 and H2 have verified support, or both
        # have verified contradictions)
        hyp_ids = [h.hypothesis_id for h in task.hypotheses]
        verified_support_by_hyp = {}
        for eid in verified_support:
            for ev in snap.visible_evidence:
                if ev.evidence_id == eid:
                    for h_id in ev.supports:
                        verified_support_by_hyp.setdefault(h_id, set()).add(eid)

        has_bilateral_verified_support = sum(1 for h_id in hyp_ids if h_id in verified_support_by_hyp) >= 2
        has_bilateral_verified_contradiction = (
            len(verified_contradictions) >= 2 and
            any(ev.evidence_id in verified_contradictions for ev in snap.visible_evidence
                if any(h_id in ev.contradicts for h_id in hyp_ids))
        )
        t3 = has_bilateral_verified_support and len(unverified_visible) == 0

        step_records.append({
            "step": step_i,
            "action": action_name,
            "outcome": outcome,
            "decision_state": state,
            "live_hypotheses": live_hyps,
            "eliminated_hypotheses": eliminated,
            "verified_support": verified_support,
            "verified_contradictions": verified_contradictions,
            "unverified_visible": unverified_visible,
            "T1_INSUFFICIENT": t1,
            "T2_all_eliminated": t2,
            "T3_bilateral_verified_no_unverified": t3,
        })

        # Execute the action
        action = DecisionAction(action_name)
        result = executor.execute(current_rt, action)
        current_rt = result.runtime
        prior_actions.append(action_name)
        prior_outcomes.append(outcome)

        if result.terminal:
            break

    return step_records


def main():
    output_dir = ROOT / "experiments/v2b_i3_11/development/i3_11b_trigger_feasibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.11b: Offline Trigger-Feasibility Analysis for R1")
    print("=" * 82)

    # Load corpus and results
    tasks = generate_decorrelated_corpus()
    task_by_id = {t.task_id: t for t in tasks}

    results_path = ROOT / "experiments/v2b_i3_11/development/i3_11a_routing_stress/routing_stress_v1.jsonl"
    results = load_results(results_path)
    print(f"  Loaded {len(results)} task results from I3.11a")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Replay all A1 trajectories
    all_replays: list[dict] = []
    for r in results:
        task = task_by_id.get(r["task_id"])
        if task is None:
            continue
        a1_fork = r["fork_a1"]
        step_records = replay_a1_with_m3_state(task, a1_fork, budget)

        # Find first trigger activation for each trigger
        t1_first = next((s["step"] for s in step_records if s["T1_INSUFFICIENT"]), None)
        t2_first = next((s["step"] for s in step_records if s["T2_all_eliminated"]), None)
        t3_first = next((s["step"] for s in step_records if s["T3_bilateral_verified_no_unverified"]), None)

        replay = {
            "task_id": r["task_id"],
            "category": r["category"],
            "n_hidden": r["n_hidden"],
            "a1_success": r["a1_success"],
            "m3_success": r["m3_success"],
            "m3_rescues_vs_a1": r["m3_rescues_vs_a1"],
            "m3_breaks_vs_a1": r["m3_breaks_vs_a1"],
            "a1_steps": r["a1_steps"],
            "a1_terminal": a1_fork["terminal_action"],
            "step_records": step_records,
            "T1_first_activation": t1_first,
            "T2_first_activation": t2_first,
            "T3_first_activation": t3_first,
        }
        all_replays.append(replay)

    print(f"  Replayed {len(all_replays)} A1 trajectories")

    # === Coverage Analysis ===
    # Coverage = P(trigger fires before A1 failure | M3 rescue)
    # "Before A1 failure" means trigger fires at any step in the A1 trajectory
    m3_rescues = [r for r in all_replays if r["m3_rescues_vs_a1"]]
    a1_succeeds = [r for r in all_replays if r["a1_success"]]
    a1_fails = [r for r in all_replays if not r["a1_success"]]

    print(f"\n{'='*82}")
    print("COVERAGE ANALYSIS")
    print(f"  M3 rescue opportunities (A1 fails, M3 succeeds): {len(m3_rescues)}")
    print(f"  A1 successes: {len(a1_succeeds)}")
    print(f"  A1 failures: {len(a1_fails)}")

    for trigger_name, trigger_key in [
        ("T1 (INSUFFICIENT)", "T1_first_activation"),
        ("T2 (all eliminated)", "T2_first_activation"),
        ("T3 (bilateral verified, no unverified)", "T3_first_activation"),
    ]:
        # Coverage: trigger fires on M3 rescue tasks
        coverage_hits = sum(1 for r in m3_rescues if r[trigger_key] is not None)
        coverage = coverage_hits / max(len(m3_rescues), 1)

        # FalseActivation: trigger fires on A1-success tasks
        false_hits = sum(1 for r in a1_succeeds if r[trigger_key] is not None)
        false_activation = false_hits / max(len(a1_succeeds), 1)

        # TriggerLatency: steps until first activation (on rescue tasks where it fires)
        latencies = [r[trigger_key] for r in m3_rescues if r[trigger_key] is not None]
        median_latency = sorted(latencies)[len(latencies) // 2] if latencies else None
        mean_latency = sum(latencies) / len(latencies) if latencies else None

        print(f"\n  {trigger_name}:")
        print(f"    Coverage = {coverage_hits}/{len(m3_rescues)} = {coverage:.4f}")
        print(f"    FalseActivation = {false_hits}/{len(a1_succeeds)} = {false_activation:.4f}")
        print(f"    TriggerLatency: mean={mean_latency:.2f}, median={median_latency}" if latencies else "    TriggerLatency: N/A")

    # === Per-category breakdown ===
    print(f"\n{'='*82}")
    print("PER-CATEGORY TRIGGER ANALYSIS")
    print(f"  {'Category':<35} {'n':>3} {'A1_ok':>5} {'M3_res':>6} {'T1_cov':>6} {'T1_false':>8} {'T2_cov':>6} {'T3_cov':>6}")

    categories = sorted(set(r["category"] for r in all_replays))
    for cat in categories:
        cr = [r for r in all_replays if r["category"] == cat]
        cn = len(cr)
        a1_ok = sum(1 for r in cr if r["a1_success"])
        m3_res = sum(1 for r in cr if r["m3_rescues_vs_a1"])

        t1_cov = sum(1 for r in cr if r["m3_rescues_vs_a1"] and r["T1_first_activation"] is not None)
        t1_false = sum(1 for r in cr if r["a1_success"] and r["T1_first_activation"] is not None)
        t2_cov = sum(1 for r in cr if r["m3_rescues_vs_a1"] and r["T2_first_activation"] is not None)
        t3_cov = sum(1 for r in cr if r["m3_rescues_vs_a1"] and r["T3_first_activation"] is not None)

        print(f"  {cat:<35} {cn:>3} {a1_ok:>5} {m3_res:>6} {t1_cov:>6} {t1_false:>8} {t2_cov:>6} {t3_cov:>6}")

    # === Detailed trigger T1 analysis ===
    print(f"\n{'='*82}")
    print("T1 (INSUFFICIENT) DETAILED ANALYSIS")

    # Which tasks does T1 fire on?
    t1_fires = [r for r in all_replays if r["T1_first_activation"] is not None]
    t1_no_fire = [r for r in all_replays if r["T1_first_activation"] is None]
    print(f"  T1 fires on {len(t1_fires)}/{len(all_replays)} tasks")
    print(f"  T1 does NOT fire on {len(t1_no_fire)}/{len(all_replays)} tasks")

    # Of tasks where T1 fires, how many are M3 rescues?
    t1_rescues = [r for r in t1_fires if r["m3_rescues_vs_a1"]]
    t1_a1_success = [r for r in t1_fires if r["a1_success"]]
    t1_both_fail = [r for r in t1_fires if not r["a1_success"] and not r["m3_success"]]
    print(f"  Of T1-fires: M3 rescues={len(t1_rescues)}, A1 success={len(t1_a1_success)}, both fail={len(t1_both_fail)}")

    # Of M3 rescues where T1 does NOT fire — why?
    t1_missed_rescues = [r for r in m3_rescues if r["T1_first_activation"] is None]
    print(f"\n  M3 rescues missed by T1: {len(t1_missed_rescues)}")
    for r in t1_missed_rescues[:5]:
        print(f"    {r['task_id']} ({r['category']}): a1_steps={r['a1_steps']}, a1_terminal={r['a1_terminal']}")
        for s in r["step_records"]:
            print(f"      step {s['step']}: action={s['action']}, state={s['decision_state']}, "
                  f"eliminated={s['eliminated_hypotheses']}, verified_support={s['verified_support']}")

    # === False activation analysis ===
    print(f"\n{'='*82}")
    print("FALSE ACTIVATION ANALYSIS (T1 fires on A1-success tasks)")
    for r in t1_a1_success[:5]:
        print(f"  {r['task_id']} ({r['category']}): T1 at step {r['T1_first_activation']}")
        for s in r["step_records"]:
            print(f"    step {s['step']}: action={s['action']}, state={s['decision_state']}, "
                  f"eliminated={s['eliminated_hypotheses']}")

    # === T1 vs T2 vs T3 comparison ===
    print(f"\n{'='*82}")
    print("TRIGGER COMPARISON SUMMARY")
    for trigger_name, trigger_key in [
        ("T1", "T1_first_activation"),
        ("T2", "T2_first_activation"),
        ("T3", "T3_first_activation"),
    ]:
        cov = sum(1 for r in m3_rescues if r[trigger_key] is not None) / max(len(m3_rescues), 1)
        false = sum(1 for r in a1_succeeds if r[trigger_key] is not None) / max(len(a1_succeeds), 1)
        print(f"  {trigger_name}: Coverage={cov:.4f}, FalseActivation={false:.4f}")

    # === Save analysis ===
    analysis = {
        "schema": "DAPH_V2B_I3_11B_TRIGGER_FEASIBILITY_V1",
        "source": "I3.11a routing stress trajectories (A1 replays with M3 state computed internally)",
        "n_tasks": len(all_replays),
        "m3_rescue_opportunities": len(m3_rescues),
        "a1_successes": len(a1_succeeds),
        "a1_failures": len(a1_fails),
        "triggers": {
            "T1_INSUFFICIENT": {
                "description": "decision_state == INSUFFICIENT (M3's frozen state estimator)",
                "coverage": round(sum(1 for r in m3_rescues if r["T1_first_activation"] is not None) / max(len(m3_rescues), 1), 4),
                "false_activation": round(sum(1 for r in a1_succeeds if r["T1_first_activation"] is not None) / max(len(a1_succeeds), 1), 4),
                "trigger_fires_total": sum(1 for r in all_replays if r["T1_first_activation"] is not None),
                "latencies": [r["T1_first_activation"] for r in m3_rescues if r["T1_first_activation"] is not None],
            },
            "T2_all_eliminated": {
                "description": "all hypotheses eliminated by visible verified evidence",
                "coverage": round(sum(1 for r in m3_rescues if r["T2_first_activation"] is not None) / max(len(m3_rescues), 1), 4),
                "false_activation": round(sum(1 for r in a1_succeeds if r["T2_first_activation"] is not None) / max(len(a1_succeeds), 1), 4),
                "trigger_fires_total": sum(1 for r in all_replays if r["T2_first_activation"] is not None),
            },
            "T3_bilateral_verified_no_unverified": {
                "description": "verified mutually incompatible evidence AND no visible unverified discriminator",
                "coverage": round(sum(1 for r in m3_rescues if r["T3_first_activation"] is not None) / max(len(m3_rescues), 1), 4),
                "false_activation": round(sum(1 for r in a1_succeeds if r["T3_first_activation"] is not None) / max(len(a1_succeeds), 1), 4),
                "trigger_fires_total": sum(1 for r in all_replays if r["T3_first_activation"] is not None),
            },
        },
        "r1_design": {
            "name": "R1 — Epistemic-Conflict Router",
            "rule": "A1 until T1 fires (INSUFFICIENT), then M3",
            "key_insight": "M3's state estimator runs internally on controller-visible snapshot without exposing M3 context to LLM. Representation changes only when observed verified evidence justifies it.",
            "does_not_predict": "hidden truth, future verification outcomes, task category",
            "responds_to": "observed verified epistemic structure",
        },
    }

    analysis_path = output_dir / "trigger_feasibility_v1.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\n  Analysis saved: {analysis_path}")

    # Save per-task replays
    replay_path = output_dir / "trigger_feasibility_v1_replays.jsonl"
    with open(replay_path, "w") as f:
        for r in all_replays:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"  Per-task replays saved: {replay_path}")


if __name__ == "__main__":
    main()
