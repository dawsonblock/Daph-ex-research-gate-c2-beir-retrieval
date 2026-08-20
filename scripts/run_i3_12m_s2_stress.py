#!/usr/bin/env python3
"""I3.12m: S2 Semantic Stress and Error Propagation experiment.

450 tasks x 6 arms = 2700 trajectories.
Tests how relation extraction errors propagate through the causal chain:
  RelationError -> StateError -> T2Error -> RepresentationError -> TaskOutcome

Primary criterion: LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0
Key metric: SemanticRobustnessGap_R1 = U_R1_GOLD - U_R1_INFERRED (by tier)
Safety metric: FalseT2Rate < 5% for EASY and MEDIUM

Usage:
    DEEPSEEK_API_KEY=... PYTHONPATH=. python scripts/run_i3_12m_s2_stress.py \\
        --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse I3.12j runner infrastructure
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_12j", ROOT / "scripts" / "run_i3_12j_factorial.py")
i3_12j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_12j)

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceExecutor, initial_evidence_runtime, build_evidence_snapshot,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget
from hrm_adaptive_memory.executive.metareasoning_utility import MetareasoningUtility
from hrm_adaptive_memory.executive.semantic_relations.s2_stress_generator import (
    generate_s2_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.integration import (
    build_evidence_snapshot_with_inferred_relations,
    infer_relations_for_runtime,
)
from hrm_adaptive_memory.executive.semantic_relations.serializer import (
    relation_graph_to_supports_contradicts,
)

ARMS = i3_12j.ARMS


def process_one_task_s2(
    semantic_task,
    budget: ResourceBudget,
    utility: MetareasoningUtility,
    api_key: str,
    extractor: DeterministicRelationExtractor,
) -> dict[str, Any]:
    """Run all 6 arms for one S2 task with full causal chain tracking."""
    task = semantic_task.evidence_task
    tier = semantic_task.tier
    semantic_class = semantic_task.semantic_class

    gold_builder = i3_12j.make_gold_snapshot_builder()
    inferred_builder = i3_12j.make_inferred_snapshot_builder(extractor)

    results: dict[str, dict] = {}
    arm_order = i3_12j.counterbalance_6arm(task.task_id)

    for arm_id in arm_order:
        rep, condition = arm_id.rsplit("_", 1)
        sb = gold_builder if condition == "GOLD" else inferred_builder

        if rep == "R1":
            result = i3_12j.run_r1_trajectory_i3_12(
                task=task, budget=budget, utility=utility,
                api_key=api_key, fork_label=arm_id, snapshot_builder=sb,
            )
        else:
            mode = "BASELINE_WITH_AFFORDANCES" if rep == "A1" else "MDSG_STATE_WITH_AFFORDANCES"
            result = i3_12j.run_trajectory_i3_12(
                task=task, budget=budget, utility=utility,
                mode=mode, api_key=api_key, fork_label=arm_id, snapshot_builder=sb,
            )
        results[arm_id] = result

    # Compute relation graph comparison
    runtime = initial_evidence_runtime(task, ResourceState(budget))
    snap_gold = build_evidence_snapshot(runtime)
    snap_inferred, graph_inferred = build_evidence_snapshot_with_inferred_relations(
        runtime, extractor)

    # Check if gold and inferred relation fields match
    relation_graph_match = True
    for ev_g, ev_i in zip(snap_gold.visible_evidence, snap_inferred.visible_evidence):
        if ev_g.supports != ev_i.supports or ev_g.contradicts != ev_i.contradicts:
            relation_graph_match = False
            break

    # Compute MDSG state comparison (at initial snapshot)
    import importlib.util
    _s = importlib.util.spec_from_file_location(
        "i3_7e", ROOT / "scripts" / "run_i3_7e_compact_governor.py")
    i3_7e = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(i3_7e)

    gold_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(snap_gold)
    inf_m3_packet = i3_7e.build_mdsg_state_with_affordances_packet(snap_inferred)
    gold_state = gold_m3_packet.get("decision_state_summary", {}).get("decision_state")
    inf_state = inf_m3_packet.get("decision_state_summary", {}).get("decision_state")
    gold_eliminated = set(gold_m3_packet.get("decision_state_summary", {}).get("eliminated_hypotheses", []))
    inf_eliminated = set(inf_m3_packet.get("decision_state_summary", {}).get("eliminated_hypotheses", []))
    mdsg_state_match = (gold_state == inf_state)
    eliminated_match = (gold_eliminated == inf_eliminated)

    # T2 comparison (at initial snapshot)
    n_hyps = len(task.hypotheses)
    gold_t2 = (len(gold_eliminated) == n_hyps and n_hyps > 0)
    inf_t2 = (len(inf_eliminated) == n_hyps and n_hyps > 0)
    t2_match = (gold_t2 == inf_t2)

    # T2 error types
    t2_false_positive = (not gold_t2) and inf_t2  # inferred says T2 but gold doesn't
    t2_false_negative = gold_t2 and (not inf_t2)  # gold says T2 but inferred doesn't

    # Build summary
    summary = {
        "task_id": task.task_id,
        "tier": tier,
        "semantic_class": semantic_class,
        "category": task.category,
        "expected_terminal": task.expected_terminal.value,
        "correct_hypothesis_id": task.correct_hypothesis_id,
        "n_hypotheses": n_hyps,
        "n_hidden": sum(1 for e in task.evidence_items if not e.retrieved),
        "relation_graph_sha256": graph_inferred.relation_graph_sha256,
        "arm_order": arm_order,
        # Causal chain
        "relation_graph_match": relation_graph_match,
        "gold_mdsg_state": gold_state,
        "inferred_mdsg_state": inf_state,
        "mdsg_state_match": mdsg_state_match,
        "gold_eliminated": sorted(gold_eliminated),
        "inferred_eliminated": sorted(inf_eliminated),
        "eliminated_match": eliminated_match,
        "gold_t2": gold_t2,
        "inferred_t2": inf_t2,
        "t2_match": t2_match,
        "t2_false_positive": t2_false_positive,
        "t2_false_negative": t2_false_negative,
    }

    for arm_id in ARMS:
        r = results[arm_id]
        rep = arm_id.rsplit("_", 1)[0]
        summary[f"u_{arm_id.lower()}"] = r["realized_utility"]
        summary[f"{arm_id.lower()}_success"] = r["success"]
        summary[f"{arm_id.lower()}_steps"] = r["steps"]
        summary[f"{arm_id.lower()}_backend_errors"] = r["backend_errors"]
        summary[f"{arm_id.lower()}_terminal_action"] = r.get("terminal_action")
        if rep == "R1":
            summary[f"{arm_id.lower()}_triggered"] = r.get("r1_triggered", False)
            summary[f"{arm_id.lower()}_trigger_step"] = r.get("r1_trigger_step")

    # Semantic gaps
    for rep in ["a1", "m3", "r1"]:
        summary[f"semantic_gap_{rep}"] = round(
            summary[f"u_{rep}_gold"] - summary[f"u_{rep}_inferred"], 4)

    # R1 vs A1 within each condition
    for cond in ["gold", "inferred"]:
        summary[f"r1_delta_vs_a1_{cond}"] = round(
            summary[f"u_r1_{cond}"] - summary[f"u_a1_{cond}"], 4)
        summary[f"r1_delta_vs_m3_{cond}"] = round(
            summary[f"u_r1_{cond}"] - summary[f"u_m3_{cond}"], 4)

    # T2 trigger comparison
    for cond in ["gold", "inferred"]:
        triggered = summary.get(f"r1_{cond}_triggered", False)
        trigger_step = summary.get(f"r1_{cond}_trigger_step")
        summary[f"r1_{cond}_t2_triggered"] = triggered
        summary[f"r1_{cond}_t2_trigger_step"] = trigger_step
        # Check if T2 trigger matches gold epistemic state
        # (This is a simplification - the actual T2 trigger depends on the trajectory)

    # Attach full forks
    summary["forks"] = {arm: results[arm] for arm in ARMS}

    return summary


def analyze_s2_results(results: list[dict]) -> dict:
    """Compute all S2-specific analysis including per-tier breakdowns."""
    n = len(results)
    report = {
        "n_tasks": n,
        "n_trajectories": n * 6,
        "per_tier": {},
        "per_arm": {},
        "primary_criterion": {},
        "semantic_robustness_gap": {},
        "t2_metrics": {},
        "causal_chain": {},
        "error_propagation": {},
    }

    # Per-tier analysis
    for tier in ["S2-EASY", "S2-MEDIUM", "S2-HARD"]:
        tier_results = [r for r in results if r["tier"] == tier]
        nt = len(tier_results)
        if nt == 0:
            continue

        tier_data = {"n_tasks": nt, "per_arm": {}, "by_semantic_class": {}}

        # Per-arm stats
        for arm in ARMS:
            utils = [r[f"u_{arm.lower()}"] for r in tier_results]
            successes = [r[f"{arm.lower()}_success"] for r in tier_results]
            steps = [r[f"{arm.lower()}_steps"] for r in tier_results]
            errors = [r[f"{arm.lower()}_backend_errors"] for r in tier_results]
            tier_data["per_arm"][arm] = {
                "mean_utility": round(sum(utils) / nt, 4),
                "success_rate": round(sum(successes) / nt, 4),
                "mean_steps": round(sum(steps) / nt, 2),
                "total_backend_errors": sum(errors),
            }

        # Primary criterion for this tier
        r1_inf_minus_a1_inf = [r["u_r1_inferred"] - r["u_a1_inferred"] for r in tier_results]
        lcb = i3_12j.one_sided_lcb(r1_inf_minus_a1_inf)
        ci_lo, ci_hi = i3_12j.paired_bootstrap_ci(r1_inf_minus_a1_inf)
        tier_data["primary_criterion"] = {
            "mean_delta": round(sum(r1_inf_minus_a1_inf) / nt, 4),
            "one_sided_lcb_95": lcb,
            "two_sided_ci_95": [ci_lo, ci_hi],
            "passes": lcb > 0,
        }

        # Semantic robustness gap for this tier
        for rep in ["a1", "m3", "r1"]:
            gaps = [r[f"semantic_gap_{rep}"] for r in tier_results]
            glo, ghi = i3_12j.paired_bootstrap_ci(gaps)
            tier_data.setdefault("semantic_robustness_gap", {})[rep] = {
                "mean_gap": round(sum(gaps) / nt, 4),
                "ci_95": [glo, ghi],
            }

        # T2 metrics for this tier
        # T2 precision: of all tasks where inferred T2 fired, how many also had gold T2?
        inf_t2_fired = [r for r in tier_results if r["inferred_t2"]]
        gold_t2_fired = [r for r in tier_results if r["gold_t2"]]
        tp = sum(1 for r in tier_results if r["inferred_t2"] and r["gold_t2"])
        fp = sum(1 for r in tier_results if r["inferred_t2"] and not r["gold_t2"])
        fn = sum(1 for r in tier_results if not r["inferred_t2"] and r["gold_t2"])
        t2_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        t2_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        false_t2_rate = fp / nt if nt > 0 else 0

        tier_data["t2_metrics"] = {
            "t2_precision": round(t2_precision, 4),
            "t2_recall": round(t2_recall, 4),
            "false_t2_rate": round(false_t2_rate, 4),
            "false_t2_rate_passes_5pct": false_t2_rate < 0.05,
            "gold_t2_count": len(gold_t2_fired),
            "inferred_t2_count": len(inf_t2_fired),
            "tp": tp, "fp": fp, "fn": fn,
        }

        # Causal chain analysis
        relation_errors = sum(1 for r in tier_results if not r["relation_graph_match"])
        state_errors = sum(1 for r in tier_results if r["relation_graph_match"] and not r["mdsg_state_match"])
        t2_errors = sum(1 for r in tier_results if r["mdsg_state_match"] and not r["t2_match"])

        tier_data["causal_chain"] = {
            "relation_graph_match": sum(1 for r in tier_results if r["relation_graph_match"]),
            "relation_graph_mismatch": relation_errors,
            "mdsg_state_match": sum(1 for r in tier_results if r["mdsg_state_match"]),
            "mdsg_state_mismatch": sum(1 for r in tier_results if not r["mdsg_state_match"]),
            "t2_match": sum(1 for r in tier_results if r["t2_match"]),
            "t2_mismatch": sum(1 for r in tier_results if not r["t2_match"]),
            "t2_false_positive": sum(1 for r in tier_results if r["t2_false_positive"]),
            "t2_false_negative": sum(1 for r in tier_results if r["t2_false_negative"]),
        }

        # By semantic class
        for sc in sorted(set(r["semantic_class"] for r in tier_results)):
            sc_results = [r for r in tier_results if r["semantic_class"] == sc]
            nsc = len(sc_results)
            tier_data["by_semantic_class"][sc] = {
                "n_tasks": nsc,
                "relation_errors": sum(1 for r in sc_results if not r["relation_graph_match"]),
                "r1_inferred_success": round(sum(r["r1_inferred_success"] for r in sc_results) / nsc, 4),
                "r1_gold_success": round(sum(r["r1_gold_success"] for r in sc_results) / nsc, 4),
                "semantic_gap_r1": round(sum(r["semantic_gap_r1"] for r in sc_results) / nsc, 4),
                "r1_delta_vs_a1_inferred": round(sum(r["r1_delta_vs_a1_inferred"] for r in sc_results) / nsc, 4),
            }

        report["per_tier"][tier] = tier_data

    # Overall per-arm
    for arm in ARMS:
        utils = [r[f"u_{arm.lower()}"] for r in results]
        successes = [r[f"{arm.lower()}_success"] for r in results]
        report["per_arm"][arm] = {
            "mean_utility": round(sum(utils) / n, 4),
            "success_rate": round(sum(successes) / n, 4),
        }

    # Overall primary criterion
    r1_inf_minus_a1_inf = [r["u_r1_inferred"] - r["u_a1_inferred"] for r in results]
    lcb = i3_12j.one_sided_lcb(r1_inf_minus_a1_inf)
    ci_lo, ci_hi = i3_12j.paired_bootstrap_ci(r1_inf_minus_a1_inf)
    report["primary_criterion"] = {
        "mean_delta": round(sum(r1_inf_minus_a1_inf) / n, 4),
        "one_sided_lcb_95": lcb,
        "two_sided_ci_95": [ci_lo, ci_hi],
        "passes": lcb > 0,
    }

    # Overall semantic robustness gap
    for rep in ["a1", "m3", "r1"]:
        gaps = [r[f"semantic_gap_{rep}"] for r in results]
        glo, ghi = i3_12j.paired_bootstrap_ci(gaps)
        report["semantic_robustness_gap"][rep] = {
            "mean_gap": round(sum(gaps) / n, 4),
            "ci_95": [glo, ghi],
        }

    # Key research table
    report["key_research_table"] = []
    for tier in ["S2-EASY", "S2-MEDIUM", "S2-HARD"]:
        td = report["per_tier"].get(tier, {})
        if not td:
            continue
        # Compute extractor F1 for this tier from relation errors
        relation_match_rate = td["causal_chain"]["relation_graph_match"] / td["n_tasks"]
        # MDSG state accuracy
        state_match_rate = td["causal_chain"]["mdsg_state_match"] / td["n_tasks"]
        # T2 metrics
        t2m = td["t2_metrics"]
        # R1 success
        r1_inf_success = td["per_arm"]["R1_INFERRED"]["success_rate"]
        r1_gold_success = td["per_arm"]["R1_GOLD"]["success_rate"]
        # R1-A1 delta
        r1_a1_delta = td["primary_criterion"]["mean_delta"]

        report["key_research_table"].append({
            "tier": tier,
            "relation_match_rate": round(relation_match_rate, 4),
            "mdsg_state_accuracy": round(state_match_rate, 4),
            "t2_precision": t2m["t2_precision"],
            "t2_recall": t2m["t2_recall"],
            "false_t2_rate": t2m["false_t2_rate"],
            "r1_inferred_success": r1_inf_success,
            "r1_gold_success": r1_gold_success,
            "r1_a1_delta_inferred": r1_a1_delta,
            "semantic_robustness_gap_r1": td.get("semantic_robustness_gap", {}).get("r1", {}).get("mean_gap", 0),
            "primary_passes": td["primary_criterion"]["passes"],
        })

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--utility", default="configs/v2b_i3_1_utility_v1.json")
    parser.add_argument("--output-dir",
        default="experiments/v2b_i3_12/development/i3_12m_s2_stress")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("I3.12m: S2 Semantic Stress and Error Propagation")
    print("  450 tasks x 6 arms = 2700 trajectories")
    print("  Primary: LCB_95(U_R1_INFERRED - U_A1_INFERRED) > 0")
    print("  Key metric: SemanticRobustnessGap by tier")
    print("  Safety: FalseT2Rate < 5% for EASY and MEDIUM")
    print()

    tasks = generate_s2_corpus(seed=42)
    print(f"  Generated {len(tasks)} S2 tasks")

    tier_counts = Counter(t.tier for t in tasks)
    print(f"  Tier distribution: {dict(tier_counts)}")

    extractor = DeterministicRelationExtractor()
    print(f"  Extractor: v{extractor.identity.extractor_version} (FROZEN)")
    print(f"  Extractor SHA256: {extractor.identity.sha256}")

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    # Oracle validation
    executor = EvidenceExecutor()
    all_pass = True
    for st in tasks:
        runtime = initial_evidence_runtime(st.evidence_task, ResourceState(budget))
        current = runtime
        final = None
        for step in st.evidence_task.oracle_resolution_path:
            parts = step.split(":")
            action = DecisionAction(parts[0])
            target = parts[1] if len(parts) > 1 else None
            final = executor.execute(current, action, target_evidence_id=target)
            current = final.runtime
            if final.terminal:
                break
        if not final.task_success:
            all_pass = False
            print(f"  ORACLE FAIL: {st.task_id} ({st.category})")
    print(f"\n  All oracle paths succeed: {all_pass}")
    if not all_pass:
        sys.exit(1)

    utility = MetareasoningUtility.from_file(ROOT / args.utility)

    print(f"\nProcessing {len(tasks)} tasks x 6 arms = {len(tasks) * 6} trajectories")
    print(f"  with {args.workers} workers...")

    all_results = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one_task_s2, st, budget, utility, api_key, extractor): st
            for st in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(tasks) - completed) / rate
                    print(f"  Completed {completed}/{len(tasks)} tasks "
                          f"({rate:.1f}/s, ETA {eta:.0f}s)")
            except Exception as e:
                print(f"  ERROR: {e}")
                completed += 1

    elapsed = time.time() - t0
    print(f"\nCompleted {len(all_results)} tasks in {elapsed:.1f}s")

    all_results.sort(key=lambda r: r["task_id"])

    # Save raw results
    results_path = output_dir / "results_v1.jsonl"
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Raw results: {results_path}")

    # Analyze
    report = analyze_s2_results(all_results)
    report["elapsed_seconds"] = round(elapsed, 1)
    report["extractor_version"] = extractor.identity.extractor_version
    report["extractor_sha256"] = extractor.identity.sha256

    report_path = output_dir / "analysis_v1.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    n = len(all_results)
    print(f"\n{'='*70}")
    print(f"I3.12m Results Summary ({n} tasks, {n*6} trajectories)")
    print(f"{'='*70}")

    print(f"\nKey Research Table:")
    print(f"  {'Tier':<12} {'RelMatch':>10} {'MDSGAcc':>10} {'T2Prec':>10} {'T2Rec':>10} {'FalseT2':>10} {'R1InfSuc':>10} {'R1GoldSuc':>10} {'R1-A1dU':>10} {'GapR1':>10} {'Pass':>6}")
    for row in report["key_research_table"]:
        print(f"  {row['tier']:<12} {row['relation_match_rate']:>10.4f} {row['mdsg_state_accuracy']:>10.4f} "
              f"{row['t2_precision']:>10.4f} {row['t2_recall']:>10.4f} {row['false_t2_rate']:>10.4f} "
              f"{row['r1_inferred_success']:>10.4f} {row['r1_gold_success']:>10.4f} "
              f"{row['r1_a1_delta_inferred']:>10.2f} {row['semantic_robustness_gap_r1']:>10.4f} "
              f"{'YES' if row['primary_passes'] else 'NO':>6}")

    print(f"\nOverall primary criterion:")
    pc = report["primary_criterion"]
    print(f"  LCB_95(U_R1_INF - U_A1_INF) = {pc['one_sided_lcb_95']}")
    print(f"  Mean delta = {pc['mean_delta']}")
    print(f"  PASSES: {pc['passes']}")

    print(f"\nSemantic Robustness Gap (GOLD - INFERRED):")
    for rep in ["a1", "m3", "r1"]:
        sg = report["semantic_robustness_gap"][rep]
        print(f"  {rep.upper()}: mean={sg['mean_gap']:.4f} CI=[{sg['ci_95'][0]}, {sg['ci_95'][1]}]")

    print(f"\nT2 safety metrics:")
    for tier in ["S2-EASY", "S2-MEDIUM", "S2-HARD"]:
        t2m = report["per_tier"][tier]["t2_metrics"]
        print(f"  {tier}: FalseT2Rate={t2m['false_t2_rate']:.4f} "
              f"(< 5%: {'YES' if t2m['false_t2_rate_passes_5pct'] else 'NO'}) "
              f"Prec={t2m['t2_precision']:.4f} Rec={t2m['t2_recall']:.4f}")

    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
