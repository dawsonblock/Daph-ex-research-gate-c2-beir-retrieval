#!/usr/bin/env python3
"""Offline replay of I3.5.3-r1 SELECTIVE_QPIB_BASE_FIRST trajectories.

Reconstructs every controller state from the saved SELECTIVE_QPIB_BASE_FIRST
trajectories, recomputes the governor recommendation a_G, extracts features,
runs the pairwise advantage model, and records every gate evaluation
(including SKIPs).

No DeepSeek API calls are needed — this is pure offline computation using
the saved trajectory steps and the trained pairwise model.

I3.5.3-r2.1 changes:
  - Full floating-point precision internally; rounding only in display fields
  - Criterion-specific artifact naming (tau5_margin5, tau0_margin0, etc.)
  - Full provenance binding (SHA-256 of all inputs)
  - Hard invariant: at tau=0, margin=0, approved == predicted_positive

Usage:
    PYTHONPATH=. python scripts/replay_i3_5_3r1_gate_evaluations.py \\
        --results experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c/results.json \\
        --gate-model experiments/v2b_i3_5_2/development/i353r1/pairwise_advantage_gate_v1.pkl \\
        --delta-threshold 5 --lcb-margin 5
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.cognitive_control.state import DecisionSummary

from hrm_adaptive_memory.executive.governor.assessor import GeneralGovernor
from hrm_adaptive_memory.executive.i3_5_1.conditions import ConditionID, get_condition
from hrm_adaptive_memory.executive.i3_5_1.observation_builder import build_observation
from hrm_adaptive_memory.executive.i3_5_1.trajectory_runner import _I3TaskAdapter
from hrm_adaptive_memory.executive.executor import (
    DeterministicActionExecutor, initial_runtime as init_task_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_benchmark import (
    load_metareasoning_benchmark,
)
from hrm_adaptive_memory.executive.metareasoning_executor import (
    DeterministicMetareasoningExecutor, initial_i3_runtime,
)
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.resources import ResourceState
from hrm_adaptive_memory.executive.selective_governor.features import (
    InterventionFeatures, extract_features,
)
from hrm_adaptive_memory.executive.selective_governor.pairwise_advantage_predictor import (
    PairwiseAdvantagePredictor,
)


def criterion_slug(threshold: float, margin: float) -> str:
    """Generate a filesystem-safe slug from the criterion parameters."""
    s = f"tau{threshold:g}_margin{margin:g}"
    return s.replace(".", "p").replace("-", "m")


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_replay_identity(
    results_sha: str,
    model_sha: str,
    script_sha: str,
    benchmark_sha: str,
    predictor_source_sha: str,
    threshold: float,
    margin: float,
) -> str:
    """Compute a replay identity that binds all inputs + criterion."""
    binding = json.dumps({
        "results_sha": results_sha,
        "model_sha": model_sha,
        "script_sha": script_sha,
        "benchmark_sha": benchmark_sha,
        "predictor_source_sha": predictor_source_sha,
        "threshold": threshold,
        "margin": margin,
    }, sort_keys=True)
    return hashlib.sha256(binding.encode()).hexdigest()


def features_from_step(
    task: Any,
    i3_runtime: Any,
    t_runtime: Any,
    prior_decisions: list[DecisionSummary],
    prior_outcomes: list[str],
    remaining_steps: int,
) -> tuple[InterventionFeatures, Any]:
    """Extract controller-visible features at a trajectory state."""
    cond = get_condition(ConditionID.AWARE_GOVERNOR)
    observation = build_observation(
        t_runtime, task, cond,
        tuple(prior_decisions), tuple(prior_outcomes))
    prior_action_strs = tuple(
        d.selected_action if isinstance(d.selected_action, str)
        else d.selected_action.value for d in prior_decisions)
    features = extract_features(
        observation,
        remaining_steps=remaining_steps,
        prior_actions=prior_action_strs,
        prior_outcomes=tuple(prior_outcomes),
    )
    return features, observation


def replay_trajectory(
    task: Any,
    budget: Any,
    sel_steps: list[dict[str, Any]],
    governor: GeneralGovernor,
    predictor: PairwiseAdvantagePredictor,
    replay_identity_sha: str,
    max_steps: int = 24,
) -> list[dict[str, Any]]:
    """Replay one SELECTIVE_QPIB_BASE_FIRST trajectory offline.

    For each step, reconstruct the state, get a_B from the saved step,
    compute a_G from the governor, extract features, and evaluate the
    pairwise predictor. Record every evaluation with FULL precision.
    """
    oracle_executor = DeterministicMetareasoningExecutor()
    task_executor = DeterministicActionExecutor()

    resources = ResourceState(budget)
    i3_runtime = initial_i3_runtime(task, resources)
    adapter = _I3TaskAdapter(task)
    t_runtime = init_task_runtime(adapter, ResourceState(budget))

    prior_decisions: list[DecisionSummary] = []
    prior_outcomes: list[str] = []
    evaluations: list[dict[str, Any]] = []

    for step_idx, step_data in enumerate(sel_steps):
        a_b_str = step_data["executed_action"]
        remaining = max_steps - step_idx

        # Extract features
        features, observation = features_from_step(
            task, i3_runtime, t_runtime,
            prior_decisions, prior_outcomes, remaining)

        prior_action_strs = tuple(
            d.selected_action if isinstance(d.selected_action, str)
            else d.selected_action.value for d in prior_decisions)

        # Compute governor recommendation
        frame = governor.assess(
            observation=observation,
            remaining_steps=remaining,
            prior_actions=prior_action_strs,
            prior_outcomes=tuple(prior_outcomes),
        )
        a_g_str = frame.governor_top_action or a_b_str

        # Evaluate pairwise predictor if there's a disagreement
        if a_b_str != a_g_str:
            # FULL precision — no rounding before aggregation
            delta_q_pi = float(predictor.predict_advantage(features, a_b_str, a_g_str))
            lcb = float(delta_q_pi - predictor.lcb_margin)
            should_intervene, _, reason = predictor.should_intervene(
                features, a_b_str, a_g_str)

            evaluations.append({
                "task_id": task.task_id,
                "step_id": step_idx,
                "base_action": a_b_str,
                "governor_action": a_g_str,
                # Full precision for scientific calculations
                "predicted_delta_q_pi": delta_q_pi,
                "lcb": lcb,
                # Display-only rounded values
                "predicted_delta_q_pi_display": round(delta_q_pi, 4),
                "lcb_display": round(lcb, 4),
                "threshold": predictor.delta_threshold,
                "margin": predictor.lcb_margin,
                "effective_requirement": predictor.delta_threshold + predictor.lcb_margin,
                "decision": "INTERVENE" if should_intervene else "SKIP",
                "reason": reason,
                "replay_identity_sha256": replay_identity_sha,
            })
        else:
            evaluations.append({
                "task_id": task.task_id,
                "step_id": step_idx,
                "base_action": a_b_str,
                "governor_action": a_g_str,
                "predicted_delta_q_pi": 0.0,
                "lcb": 0.0,
                "predicted_delta_q_pi_display": 0.0,
                "lcb_display": 0.0,
                "threshold": predictor.delta_threshold,
                "margin": predictor.lcb_margin,
                "effective_requirement": predictor.delta_threshold + predictor.lcb_margin,
                "decision": "SKIP_SAME_ACTION",
                "reason": "a_G == a_B",
                "replay_identity_sha256": replay_identity_sha,
            })

        # Step forward using the executed action from the saved trajectory
        from hrm_adaptive_memory.cognitive_control.core import DecisionAction
        a_exec = DecisionAction(a_b_str)
        exec_res = oracle_executor.execute(i3_runtime, a_exec)
        i3_runtime = exec_res.runtime
        t_runtime = task_executor.execute(t_runtime, a_exec).runtime
        prior_decisions.append(DecisionSummary(
            f"{task.task_id}:step:{step_idx}", a_b_str,
            step_data.get("reason_code", ""), exec_res.outcome_code))
        prior_outcomes.append(exec_res.outcome_code)

        if exec_res.terminal:
            break

    return evaluations


def main():
    parser = argparse.ArgumentParser(description="Offline replay of I3.5.3-r1 gate evaluations")
    parser.add_argument(
        "--results",
        default="experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c/results.json",
    )
    parser.add_argument(
        "--gate-model",
        default="experiments/v2b_i3_5_2/development/i353r1/pairwise_advantage_gate_v1.pkl",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="experiments/v2b_i3_5/manifests/v2b_i3_5_benchmark_manifest_v2.json",
    )
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--policy", default="configs/v2b_i3_policy_v1.json")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2b_i3_5_2/development/i353r1_38ecd7e5849c",
    )
    parser.add_argument("--delta-threshold", type=float, default=None)
    parser.add_argument("--lcb-margin", type=float, default=None)
    args = parser.parse_args()

    # Load predictor and apply overrides
    print(f"Loading pairwise advantage gate from {args.gate_model}...")
    predictor = PairwiseAdvantagePredictor.load(args.gate_model)
    if args.delta_threshold is not None:
        predictor.delta_threshold = args.delta_threshold
    if args.lcb_margin is not None:
        predictor.lcb_margin = args.lcb_margin
    print(f"  Threshold: {predictor.delta_threshold}")
    print(f"  LCB margin: {predictor.lcb_margin}")
    print(f"  Effective requirement: predicted ΔQ_π > {predictor.delta_threshold + predictor.lcb_margin}")

    # Compute provenance
    print("\nComputing replay provenance...")
    results_sha = file_sha256(args.results)
    model_sha = file_sha256(args.gate_model)
    script_sha = file_sha256(__file__)
    benchmark_sha = file_sha256(args.benchmark_manifest)
    predictor_source_sha = file_sha256(
        ROOT / "hrm_adaptive_memory/executive/selective_governor/pairwise_advantage_predictor.py")

    # Get source commit
    import subprocess
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        source_commit = "UNKNOWN"

    replay_identity_sha = compute_replay_identity(
        results_sha=results_sha,
        model_sha=model_sha,
        script_sha=script_sha,
        benchmark_sha=benchmark_sha,
        predictor_source_sha=predictor_source_sha,
        threshold=predictor.delta_threshold,
        margin=predictor.lcb_margin,
    )
    print(f"  Replay identity: {replay_identity_sha}")
    print(f"  Results SHA:     {results_sha[:16]}...")
    print(f"  Model SHA:       {model_sha[:16]}...")
    print(f"  Script SHA:      {script_sha[:16]}...")

    # Criterion-specific output directory
    slug = criterion_slug(predictor.delta_threshold, predictor.lcb_margin)
    output_dir = Path(args.output_dir) / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    evals_path = output_dir / f"gate_evaluations_{slug}.jsonl"
    summary_path = output_dir / f"gate_evaluation_summary_{slug}.json"

    print(f"\nOutput slug: {slug}")
    print(f"  Evaluations: {evals_path}")
    print(f"  Summary:     {summary_path}")

    # Load benchmark and results
    print(f"\nLoading benchmark from {args.benchmark_manifest}...")
    benchmark = load_metareasoning_benchmark(args.benchmark_manifest, verify_oracle_cache=False)
    split_bm = benchmark.for_split("structure_dev_v2")
    task_map = {t.task_id: t for t in split_bm.tasks}

    print(f"Loading results from {args.results}...")
    results_data = json.loads(Path(args.results).read_text())
    blocks = results_data["results"]
    print(f"Loaded {len(blocks)} task blocks")

    governor = GeneralGovernor()

    print("\nReplaying SELECTIVE_QPIB_BASE_FIRST trajectories offline...")
    all_evaluations: list[dict[str, Any]] = []

    for i, block in enumerate(blocks):
        task_id = block["task_id"]
        if task_id not in task_map:
            continue
        task = task_map[task_id]
        budget = split_bm.budget_for(task)

        sel_traj = block["trajectories"].get("SELECTIVE_QPIB_BASE_FIRST")
        if sel_traj is None:
            continue
        sel_steps = sel_traj.get("steps", [])
        if not sel_steps:
            continue

        evals = replay_trajectory(
            task=task, budget=budget, sel_steps=sel_steps,
            governor=governor, predictor=predictor,
            replay_identity_sha=replay_identity_sha)
        all_evaluations.extend(evals)

        if (i + 1) % 50 == 0:
            print(f"  Processed [{i+1}/{len(blocks)}] tasks, "
                  f"{len(all_evaluations)} evaluations so far")

    print(f"\nTotal evaluations: {len(all_evaluations)}")

    # Save all evaluations (full precision)
    with open(evals_path, "w") as f:
        for ev in all_evaluations:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
    print(f"Saved: {evals_path}")

    # Analysis — all from full precision
    print("\n" + "=" * 78)
    print(f"GATE EVALUATION DISTRIBUTION (OFFLINE REPLAY — {slug})")
    print("=" * 78)

    n = len(all_evaluations)
    n_same = sum(1 for e in all_evaluations if e["decision"] == "SKIP_SAME_ACTION")
    n_disagree = n - n_same

    print(f"\nTotal evaluations:           {n}")
    print(f"  a_G == a_B (no disagreement): {n_same}")
    print(f"  a_G != a_B (disagreement):    {n_disagree}")

    # For disagreement evaluations only — use full precision values
    disagree_evals = [e for e in all_evaluations if e["decision"] != "SKIP_SAME_ACTION"]

    summary: dict[str, Any] = {
        "schema": "DAPH_V2B_I3_5_3R2_REPLAY_V1",
        "replay_identity_sha256": replay_identity_sha,
        "criterion_slug": slug,
        "n_evaluations": n,
        "n_same_action": n_same,
        "n_disagreements": n_disagree,
        "runtime_params": {
            "delta_threshold": predictor.delta_threshold,
            "lcb_margin": predictor.lcb_margin,
            "effective_requirement": predictor.delta_threshold + predictor.lcb_margin,
        },
        "provenance": {
            "source_results_sha256": results_sha,
            "gate_model_sha256": model_sha,
            "replay_script_sha256": script_sha,
            "benchmark_manifest_sha256": benchmark_sha,
            "predictor_source_sha256": predictor_source_sha,
            "threshold": predictor.delta_threshold,
            "margin": predictor.lcb_margin,
            "source_commit": source_commit,
        },
    }

    if disagree_evals:
        # Full precision predictions
        predictions = [e["predicted_delta_q_pi"] for e in disagree_evals]
        lcbs = [e["lcb"] for e in disagree_evals]

        mean_pred = sum(predictions) / len(predictions)
        max_pred = max(predictions)
        min_pred = min(predictions)
        mean_lcb = sum(lcbs) / len(lcbs)
        max_lcb = max(lcbs)

        pred_positive = sum(1 for p in predictions if p > 0)
        pred_gt_5 = sum(1 for p in predictions if p > 5)
        pred_gt_10 = sum(1 for p in predictions if p > 10)
        lcb_positive = sum(1 for l in lcbs if l > 0)
        lcb_gt_5 = sum(1 for l in lcbs if l > 5)

        n_intervene = sum(1 for e in disagree_evals if e["decision"] == "INTERVENE")
        n_skip = sum(1 for e in disagree_evals if e["decision"] == "SKIP")

        print(f"\n--- Predicted ΔQ_π distribution (N={n_disagree} disagreements) ---")
        print(f"  Mean predicted ΔQ_π:  {mean_pred:+.6f}")
        print(f"  Min predicted ΔQ_π:   {min_pred:+.6f}")
        print(f"  Max predicted ΔQ_π:   {max_pred:+.6f}")
        print(f"  Mean LCB:             {mean_lcb:+.6f}")
        print(f"  Max LCB:              {max_lcb:+.6f}")
        print(f"  Predicted > 0:        {pred_positive}/{n_disagree} ({pred_positive/n_disagree:.1%})")
        print(f"  Predicted > 5:        {pred_gt_5}/{n_disagree} ({pred_gt_5/n_disagree:.1%})")
        print(f"  Predicted > 10:       {pred_gt_10}/{n_disagree} ({pred_gt_10/n_disagree:.1%})")
        print(f"  LCB > 0:              {lcb_positive}/{n_disagree} ({lcb_positive/n_disagree:.1%})")
        print(f"  LCB > 5:              {lcb_gt_5}/{n_disagree} ({lcb_gt_5/n_disagree:.1%})")
        print(f"  Approved (INTERVENE): {n_intervene}/{n_disagree}")
        print(f"  Skipped (SKIP):       {n_skip}/{n_disagree}")

        # Hard invariant: at tau=0, margin=0, approved == pred_positive
        if predictor.delta_threshold == 0.0 and predictor.lcb_margin == 0.0:
            assert n_intervene == pred_positive, (
                f"INVARIANT VIOLATION: at tau=0, margin=0, "
                f"approved ({n_intervene}) != predicted_positive ({pred_positive}). "
                f"This indicates a rounding or logic bug."
            )
            print(f"\n  INVARIANT CHECK: approved == predicted_positive → PASS ({n_intervene} == {pred_positive})")

        # Action pair distribution
        print(f"\n--- Action pair × predicted ΔQ_π ---")
        pair_preds = defaultdict(list)
        for e in disagree_evals:
            pair_preds[(e["base_action"], e["governor_action"])].append(e["predicted_delta_q_pi"])

        for (b, g), preds in sorted(pair_preds.items(), key=lambda x: -len(x[1])):
            pos = sum(1 for p in preds if p > 0)
            print(f"  {b} → {g}: n={len(preds)}, "
                  f"mean={sum(preds)/len(preds):+.6f}, "
                  f"max={max(preds):+.6f}, "
                  f"positive={pos}")

        # Summary with full precision + display values
        summary["prediction_distribution"] = {
            "mean": mean_pred,
            "mean_display": round(mean_pred, 4),
            "min": min_pred,
            "min_display": round(min_pred, 4),
            "max": max_pred,
            "max_display": round(max_pred, 4),
            "mean_lcb": mean_lcb,
            "mean_lcb_display": round(mean_lcb, 4),
            "max_lcb": max_lcb,
            "max_lcb_display": round(max_lcb, 4),
            "pred_positive": pred_positive,
            "pred_gt_5": pred_gt_5,
            "pred_gt_10": pred_gt_10,
            "lcb_positive": lcb_positive,
            "lcb_gt_5": lcb_gt_5,
            "n_intervene": n_intervene,
            "n_skip": n_skip,
        }
        summary["action_pair_distribution"] = {
            f"{b}->{g}": {
                "n": len(preds),
                "mean": sum(preds) / len(preds),
                "mean_display": round(sum(preds) / len(preds), 4),
                "max": max(preds),
                "max_display": round(max(preds), 4),
                "positive": sum(1 for p in preds if p > 0),
            }
            for (b, g), preds in pair_preds.items()
        }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
