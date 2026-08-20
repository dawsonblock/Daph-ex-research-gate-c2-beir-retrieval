#!/usr/bin/env python3
"""I3.12m: Enhanced S2 analysis with error propagation metrics.

Computes the additional metrics requested for I3.12m:
  - SEMANTIC_CAUSAL classification (INFERRED fails AND GOLD succeeds)
  - P(T2 false positive | false contradiction)
  - P(T2 false negative | missed contradiction)
  - P(task failure | relation error type)
  - Severity-weighted error metrics
  - Catastrophic subgroup check
  - Rescue/Break counts
  - Semantic degradation curve

Usage:
    PYTHONPATH=. python scripts/run_i3_12m_s2_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "i3_12j", ROOT / "scripts" / "run_i3_12j_factorial.py")
i3_12j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i3_12j)

from hrm_adaptive_memory.executive.semantic_relations.s2_stress_generator import (
    generate_s2_corpus, TIER_CONFIG,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_relation_errors_per_task(task_result: dict, extractor: DeterministicRelationExtractor) -> dict:
    """Compute per-task relation error details."""
    # The task result already has relation_graph_match
    # We need to identify specific error types
    # This is computed from the fork data
    forks = task_result.get("forks", {})

    # Get gold and inferred relation graphs from the forks
    # The relation graph is embedded in the snapshot
    # For now, use the match flags already computed
    return {
        "relation_graph_match": task_result.get("relation_graph_match", True),
        "mdsg_state_match": task_result.get("mdsg_state_match", True),
        "t2_match": task_result.get("t2_match", True),
        "t2_false_positive": task_result.get("t2_false_positive", False),
        "t2_false_negative": task_result.get("t2_false_negative", False),
    }


def classify_semantic_causal(task_result: dict) -> str:
    """Classify whether a task outcome difference is semantically causal.

    SEMANTIC_CAUSAL: INFERRED fails AND GOLD succeeds (under matched conditions)
    BACKEND_VARIANCE: Both fail or both succeed (no semantic effect)
    TRAJECTORY_CASCADE: Relation error present but outcome same
    """
    # Check R1 specifically (primary architecture)
    r1_gold_success = task_result.get("r1_gold_success", False)
    r1_inf_success = task_result.get("r1_inferred_success", False)
    relation_match = task_result.get("relation_graph_match", True)

    if not relation_match:
        if not r1_inf_success and r1_gold_success:
            return "SEMANTIC_CAUSAL"
        elif r1_inf_success and not r1_gold_success:
            return "SEMANTIC_BENEFIT"  # relation error accidentally helped
        else:
            return "TRAJECTORY_CASCADE"
    else:
        if r1_gold_success != r1_inf_success:
            return "BACKEND_RESPONSE_VARIANCE"
        else:
            return "NO_DIVERGENCE"


def compute_error_severity_analysis(results: list[dict]) -> dict:
    """Compute P(T2 error | relation error type) and P(task failure | relation error type)."""
    # We need to recompute relation errors per task
    # Since we don't have per-edge errors in the results, we use the
    # relation_graph_match flag and T2 flags

    # Count co-occurrences
    n = len(results)

    # T2 false positive analysis
    t2_fp = sum(1 for r in results if r.get("t2_false_positive", False))
    t2_fn = sum(1 for r in results if r.get("t2_false_negative", False))
    relation_errors = sum(1 for r in results if not r.get("relation_graph_match", True))

    # P(T2 false positive | relation error)
    p_t2_fp_given_rel_error = (t2_fp / relation_errors) if relation_errors > 0 else 0

    # P(T2 false positive | no relation error) - backend variance only
    no_rel_error = n - relation_errors
    t2_fp_no_rel = sum(1 for r in results if r.get("t2_false_positive", False) and r.get("relation_graph_match", True))
    p_t2_fp_given_no_rel_error = (t2_fp_no_rel / no_rel_error) if no_rel_error > 0 else 0

    # Task failure analysis
    r1_inf_failures = [r for r in results if not r.get("r1_inferred_success", False)]
    r1_inf_failures_with_rel_error = [r for r in r1_inf_failures if not r.get("relation_graph_match", True)]
    r1_inf_failures_without_rel_error = [r for r in r1_inf_failures if r.get("relation_graph_match", True)]

    p_task_fail_given_rel_error = (len(r1_inf_failures_with_rel_error) / relation_errors) if relation_errors > 0 else 0
    p_task_fail_given_no_rel_error = (len(r1_inf_failures_without_rel_error) / no_rel_error) if no_rel_error > 0 else 0

    # SEMANTIC_CAUSAL classification
    causal_classes = Counter(classify_semantic_causal(r) for r in results)

    return {
        "n_tasks": n,
        "relation_errors": relation_errors,
        "no_relation_errors": no_rel_error,
        "t2_false_positives": t2_fp,
        "t2_false_negatives": t2_fn,
        "p_t2_fp_given_relation_error": round(p_t2_fp_given_rel_error, 4),
        "p_t2_fp_given_no_relation_error": round(p_t2_fp_given_no_rel_error, 4),
        "p_task_fail_given_relation_error": round(p_task_fail_given_rel_error, 4),
        "p_task_fail_given_no_relation_error": round(p_task_fail_given_no_rel_error, 4),
        "causal_classification": dict(causal_classes),
        "semantic_causal_count": causal_classes.get("SEMANTIC_CAUSAL", 0),
        "semantic_benefit_count": causal_classes.get("SEMANTIC_BENEFIT", 0),
        "trajectory_cascade_count": causal_classes.get("TRAJECTORY_CASCADE", 0),
        "backend_variance_count": causal_classes.get("BACKEND_RESPONSE_VARIANCE", 0),
        "no_divergence_count": causal_classes.get("NO_DIVERGENCE", 0),
    }


def compute_rescue_break(results: list[dict]) -> dict:
    """Compute R1 rescue/break counts vs A1 (inferred condition)."""
    n = len(results)
    rescues = 0  # R1 succeeds, A1 fails
    breaks = 0   # R1 fails, A1 succeeds
    both_succeed = 0
    both_fail = 0

    for r in results:
        r1 = r.get("r1_inferred_success", False)
        a1 = r.get("a1_inferred_success", False)
        if r1 and not a1:
            rescues += 1
        elif not r1 and a1:
            breaks += 1
        elif r1 and a1:
            both_succeed += 1
        else:
            both_fail += 1

    return {
        "n": n,
        "rescues": rescues,
        "breaks": breaks,
        "both_succeed": both_succeed,
        "both_fail": both_fail,
        "rescue_break_ratio": round(rescues / breaks, 4) if breaks > 0 else float("inf") if rescues > 0 else 0,
        "rescues_gt_breaks": rescues > breaks,
    }


def compute_catastrophic_subgroups(results: list[dict]) -> dict:
    """Check for catastrophic subgroups where inferred R1 collapses relative to inferred A1."""
    subgroups = {}

    # By semantic class
    by_class = defaultdict(list)
    for r in results:
        by_class[r["semantic_class"]].append(r)

    for cls, cls_results in by_class.items():
        n = len(cls_results)
        r1_success = sum(r["r1_inferred_success"] for r in cls_results) / n
        a1_success = sum(r["a1_inferred_success"] for r in cls_results) / n
        r1_u = sum(r["u_r1_inferred"] for r in cls_results) / n
        a1_u = sum(r["u_a1_inferred"] for r in cls_results) / n
        delta_u = r1_u - a1_u

        # Catastrophic: R1 success rate < A1 success rate by > 0.15
        # OR R1 utility < A1 utility by > 30
        catastrophic = (r1_success < a1_success - 0.15) or (delta_u < -30)

        subgroups[f"class:{cls}"] = {
            "n": n,
            "r1_success": round(r1_success, 4),
            "a1_success": round(a1_success, 4),
            "r1_a1_delta_u": round(delta_u, 2),
            "catastrophic": catastrophic,
        }

    # By tier
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    for tier, tier_results in by_tier.items():
        n = len(tier_results)
        r1_success = sum(r["r1_inferred_success"] for r in tier_results) / n
        a1_success = sum(r["a1_inferred_success"] for r in tier_results) / n
        r1_u = sum(r["u_r1_inferred"] for r in tier_results) / n
        a1_u = sum(r["u_a1_inferred"] for r in tier_results) / n
        delta_u = r1_u - a1_u

        catastrophic = (r1_success < a1_success - 0.15) or (delta_u < -30)

        subgroups[f"tier:{tier}"] = {
            "n": n,
            "r1_success": round(r1_success, 4),
            "a1_success": round(a1_success, 4),
            "r1_a1_delta_u": round(delta_u, 2),
            "catastrophic": catastrophic,
        }

    # By structural pattern (category)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    for cat, cat_results in by_cat.items():
        n = len(cat_results)
        r1_success = sum(r["r1_inferred_success"] for r in cat_results) / n
        a1_success = sum(r["a1_inferred_success"] for r in cat_results) / n
        r1_u = sum(r["u_r1_inferred"] for r in cat_results) / n
        a1_u = sum(r["u_a1_inferred"] for r in cat_results) / n
        delta_u = r1_u - a1_u

        catastrophic = (r1_success < a1_success - 0.15) or (delta_u < -30)

        subgroups[f"category:{cat}"] = {
            "n": n,
            "r1_success": round(r1_success, 4),
            "a1_success": round(a1_success, 4),
            "r1_a1_delta_u": round(delta_u, 2),
            "catastrophic": catastrophic,
        }

    catastrophic_count = sum(1 for v in subgroups.values() if v["catastrophic"])
    return {
        "subgroups": subgroups,
        "catastrophic_subgroup_count": catastrophic_count,
        "any_catastrophic": catastrophic_count > 0,
    }


def compute_degradation_curve(results: list[dict]) -> list[dict]:
    """Compute the semantic degradation curve: RelationF1 -> MDSGStateAccuracy -> T2Accuracy -> TaskUtility."""
    curve = []

    for tier in ["S2-EASY", "S2-MEDIUM", "S2-HARD"]:
        tier_results = [r for r in results if r["tier"] == tier]
        n = len(tier_results)
        if n == 0:
            continue

        # Relation match rate (proxy for F1 at task level)
        relation_match = sum(1 for r in tier_results if r["relation_graph_match"]) / n

        # MDSG state accuracy
        state_match = sum(1 for r in tier_results if r["mdsg_state_match"]) / n

        # T2 accuracy
        t2_match = sum(1 for r in tier_results if r["t2_match"]) / n

        # T2 precision/recall
        tp = sum(1 for r in tier_results if r["inferred_t2"] and r["gold_t2"])
        fp = sum(1 for r in tier_results if r["inferred_t2"] and not r["gold_t2"])
        fn = sum(1 for r in tier_results if not r["inferred_t2"] and r["gold_t2"])
        t2_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        t2_rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        # Task utility
        r1_inf_u = sum(r["u_r1_inferred"] for r in tier_results) / n
        r1_gold_u = sum(r["u_r1_gold"] for r in tier_results) / n
        a1_inf_u = sum(r["u_a1_inferred"] for r in tier_results) / n
        m3_inf_u = sum(r["u_m3_inferred"] for r in tier_results) / n

        # Success rates
        r1_inf_s = sum(r["r1_inferred_success"] for r in tier_results) / n
        r1_gold_s = sum(r["r1_gold_success"] for r in tier_results) / n
        a1_inf_s = sum(r["a1_inferred_success"] for r in tier_results) / n
        m3_inf_s = sum(r["m3_inferred_success"] for r in tier_results) / n

        # Semantic gaps
        gap_r1 = r1_gold_u - r1_inf_u
        gap_a1 = sum(r["semantic_gap_a1"] for r in tier_results) / n
        gap_m3 = sum(r["semantic_gap_m3"] for r in tier_results) / n

        # R1-A1 delta
        r1_a1_delta = r1_inf_u - a1_inf_u

        curve.append({
            "tier": tier,
            "relation_match_rate": round(relation_match, 4),
            "mdsg_state_accuracy": round(state_match, 4),
            "t2_accuracy": round(t2_match, 4),
            "t2_precision": round(t2_prec, 4),
            "t2_recall": round(t2_rec, 4),
            "false_t2_rate": round(fp / n, 4),
            "r1_inferred_utility": round(r1_inf_u, 2),
            "r1_gold_utility": round(r1_gold_u, 2),
            "a1_inferred_utility": round(a1_inf_u, 2),
            "m3_inferred_utility": round(m3_inf_u, 2),
            "r1_inferred_success": round(r1_inf_s, 4),
            "r1_gold_success": round(r1_gold_s, 4),
            "a1_inferred_success": round(a1_inf_s, 4),
            "m3_inferred_success": round(m3_inf_s, 4),
            "r1_a1_delta_inferred": round(r1_a1_delta, 2),
            "semantic_gap_r1": round(gap_r1, 2),
            "semantic_gap_a1": round(gap_a1, 2),
            "semantic_gap_m3": round(gap_m3, 2),
        })

    return curve


def compute_per_class_detail(results: list[dict]) -> dict:
    """Compute per-semantic-class detailed metrics."""
    by_class = defaultdict(list)
    for r in results:
        by_class[r["semantic_class"]].append(r)

    detail = {}
    for cls in sorted(by_class.keys()):
        cls_results = by_class[cls]
        n = len(cls_results)
        tier = cls_results[0]["tier"]

        relation_match = sum(1 for r in cls_results if r["relation_graph_match"]) / n
        state_match = sum(1 for r in cls_results if r["mdsg_state_match"]) / n
        t2_match = sum(1 for r in cls_results if r["t2_match"]) / n

        tp = sum(1 for r in cls_results if r["inferred_t2"] and r["gold_t2"])
        fp = sum(1 for r in cls_results if r["inferred_t2"] and not r["gold_t2"])
        fn = sum(1 for r in cls_results if not r["inferred_t2"] and r["gold_t2"])

        r1_inf_s = sum(r["r1_inferred_success"] for r in cls_results) / n
        r1_gold_s = sum(r["r1_gold_success"] for r in cls_results) / n
        a1_inf_s = sum(r["a1_inferred_success"] for r in cls_results) / n

        r1_inf_u = sum(r["u_r1_inferred"] for r in cls_results) / n
        r1_gold_u = sum(r["u_r1_gold"] for r in cls_results) / n
        a1_inf_u = sum(r["u_a1_inferred"] for r in cls_results) / n

        gap_r1 = r1_gold_u - r1_inf_u
        r1_a1_delta = r1_inf_u - a1_inf_u

        # Causal classification
        causal = Counter(classify_semantic_causal(r) for r in cls_results)

        detail[cls] = {
            "tier": tier,
            "n": n,
            "relation_match_rate": round(relation_match, 4),
            "mdsg_state_accuracy": round(state_match, 4),
            "t2_accuracy": round(t2_match, 4),
            "t2_precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0,
            "t2_recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0,
            "false_t2_rate": round(fp / n, 4),
            "r1_inferred_success": round(r1_inf_s, 4),
            "r1_gold_success": round(r1_gold_s, 4),
            "a1_inferred_success": round(a1_inf_s, 4),
            "r1_inferred_utility": round(r1_inf_u, 2),
            "r1_gold_utility": round(r1_gold_u, 2),
            "r1_a1_delta_inferred": round(r1_a1_delta, 2),
            "semantic_gap_r1": round(gap_r1, 2),
            "causal_classification": dict(causal),
            "semantic_causal_count": causal.get("SEMANTIC_CAUSAL", 0),
        }

    return detail


def compute_per_edge_error_types(results: list[dict], extractor: DeterministicRelationExtractor) -> dict:
    """Recompute per-edge relation errors and link to T2 outcomes.

    This is the key analysis the user requested:
    - P(T2 false positive | false contradiction)
    - P(T2 false negative | missed contradiction)
    - P(task failure | relation error type)
    - Severity ordering
    """
    tasks = generate_s2_corpus(seed=42)
    task_by_id = {t.task_id: t for t in tasks}

    # For each task, recompute per-edge errors
    edge_errors = []  # list of (task_id, tier, semantic_class, error_type, t2_fp, t2_fn, task_fail)

    for r in results:
        task_id = r["task_id"]
        st = task_by_id.get(task_id)
        if not st:
            continue

        et = st.evidence_task
        tier = r["tier"]
        semantic_class = r["semantic_class"]
        t2_fp = r.get("t2_false_positive", False)
        t2_fn = r.get("t2_false_negative", False)
        task_fail = not r.get("r1_inferred_success", False)

        task_error_types = set()

        for ev in [e for e in et.evidence_items if e.retrieved]:
            for hyp in et.hypotheses:
                result = extractor.extract(
                    evidence_id=ev.evidence_id,
                    evidence_proposition=ev.proposition,
                    hypothesis_id=hyp.hypothesis_id,
                    hypothesis_proposition=hyp.proposition,
                )
                inferred = result.relation.relation.value

                gold_rel = "NEUTRAL"
                for gr in st.gold_relations:
                    if gr.evidence_id == ev.evidence_id and gr.hypothesis_id == hyp.hypothesis_id:
                        gold_rel = gr.relation
                        break

                if gold_rel != inferred:
                    if gold_rel == "SUPPORT" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"
                    elif gold_rel == "SUPPORT" and inferred == "NEUTRAL":
                        error_type = "MISSED_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "CONTRADICT" and inferred == "NEUTRAL":
                        error_type = "MISSED_CONTRADICTION"
                    elif gold_rel == "NEUTRAL" and inferred == "SUPPORT":
                        error_type = "FALSE_SUPPORT"
                    elif gold_rel == "NEUTRAL" and inferred == "CONTRADICT":
                        error_type = "FALSE_CONTRADICTION"
                    else:
                        error_type = "UNKNOWN"

                    task_error_types.add(error_type)
                    edge_errors.append({
                        "task_id": task_id,
                        "tier": tier,
                        "semantic_class": semantic_class,
                        "error_type": error_type,
                        "gold": gold_rel,
                        "inferred": inferred,
                        "t2_false_positive": t2_fp,
                        "t2_false_negative": t2_fn,
                        "task_fail": task_fail,
                    })

    # Aggregate: P(T2 FP | error type), P(task fail | error type)
    by_error_type = defaultdict(list)
    for e in edge_errors:
        by_error_type[e["error_type"]].append(e)

    # Also compute per-task error type presence
    task_error_presence = defaultdict(lambda: {"t2_fp": 0, "t2_fn": 0, "task_fail": 0, "count": 0})
    for r in results:
        task_id = r["task_id"]
        st = task_by_id.get(task_id)
        if not st:
            continue
        # Recompute which error types this task has
        task_types = set()
        for ev in [e for e in st.evidence_task.evidence_items if e.retrieved]:
            for hyp in st.evidence_task.hypotheses:
                result = extractor.extract(
                    evidence_id=ev.evidence_id,
                    evidence_proposition=ev.proposition,
                    hypothesis_id=hyp.hypothesis_id,
                    hypothesis_proposition=hyp.proposition,
                )
                inferred = result.relation.relation.value
                gold_rel = "NEUTRAL"
                for gr in st.gold_relations:
                    if gr.evidence_id == ev.evidence_id and gr.hypothesis_id == hyp.hypothesis_id:
                        gold_rel = gr.relation; break
                if gold_rel != inferred:
                    if gold_rel == "SUPPORT" and inferred == "CONTRADICT":
                        task_types.add("FALSE_CONTRADICTION")
                    elif gold_rel == "SUPPORT" and inferred == "NEUTRAL":
                        task_types.add("MISSED_SUPPORT")
                    elif gold_rel == "CONTRADICT" and inferred == "SUPPORT":
                        task_types.add("FALSE_SUPPORT")
                    elif gold_rel == "CONTRADICT" and inferred == "NEUTRAL":
                        task_types.add("MISSED_CONTRADICTION")
                    elif gold_rel == "NEUTRAL" and inferred == "SUPPORT":
                        task_types.add("FALSE_SUPPORT")
                    elif gold_rel == "NEUTRAL" and inferred == "CONTRADICT":
                        task_types.add("FALSE_CONTRADICTION")

        for et in task_types:
            task_error_presence[et]["count"] += 1
            if r.get("t2_false_positive", False):
                task_error_presence[et]["t2_fp"] += 1
            if r.get("t2_false_negative", False):
                task_error_presence[et]["t2_fn"] += 1
            if not r.get("r1_inferred_success", False):
                task_error_presence[et]["task_fail"] += 1

    severity_analysis = {}
    for error_type in ["FALSE_CONTRADICTION", "MISSED_SUPPORT", "FALSE_SUPPORT", "MISSED_CONTRADICTION"]:
        tep = task_error_presence[error_type]
        n = tep["count"]
        severity_analysis[error_type] = {
            "n_tasks_with_error": n,
            "p_t2_false_positive": round(tep["t2_fp"] / n, 4) if n > 0 else 0,
            "p_t2_false_negative": round(tep["t2_fn"] / n, 4) if n > 0 else 0,
            "p_task_failure": round(tep["task_fail"] / n, 4) if n > 0 else 0,
        }

    return {
        "total_edge_errors": len(edge_errors),
        "by_error_type": {et: len(es) for et, es in by_error_type.items()},
        "severity_analysis": severity_analysis,
        "severity_ordering": sorted(
            severity_analysis.keys(),
            key=lambda x: severity_analysis[x]["p_t2_false_positive"],
            reverse=True,
        ),
    }


def main():
    results_path = ROOT / "experiments/v2b_i3_12/development/i3_12m_s2_stress/results_v1.jsonl"
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results(results_path)
    print(f"Loaded {len(results)} task results")

    extractor = DeterministicRelationExtractor()

    # Compute all enhanced analyses
    report = {
        "analysis_id": "i3_12m_s2_enhanced_analysis_v1",
        "n_tasks": len(results),
    }

    # 1. Error severity analysis
    print("\nComputing error severity analysis...")
    report["error_severity"] = compute_error_severity_analysis(results)
    es = report["error_severity"]
    print(f"  Relation errors: {es['relation_errors']}")
    print(f"  T2 false positives: {es['t2_false_positives']}")
    print(f"  P(T2 FP | relation error) = {es['p_t2_fp_given_relation_error']}")
    print(f"  P(T2 FP | no relation error) = {es['p_t2_fp_given_no_relation_error']}")
    print(f"  P(task fail | relation error) = {es['p_task_fail_given_relation_error']}")
    print(f"  P(task fail | no relation error) = {es['p_task_fail_given_no_relation_error']}")
    print(f"  Causal classification: {es['causal_classification']}")

    # 2. Rescue/Break
    print("\nComputing rescue/break counts...")
    report["rescue_break"] = compute_rescue_break(results)
    rb = report["rescue_break"]
    print(f"  Rescues: {rb['rescues']}, Breaks: {rb['breaks']}")
    print(f"  Ratio: {rb['rescue_break_ratio']}")
    print(f"  Rescues > Breaks: {rb['rescues_gt_breaks']}")

    # 3. Catastrophic subgroups
    print("\nChecking for catastrophic subgroups...")
    report["catastrophic_subgroups"] = compute_catastrophic_subgroups(results)
    cs = report["catastrophic_subgroups"]
    print(f"  Catastrophic subgroup count: {cs['catastrophic_subgroup_count']}")
    if cs["any_catastrophic"]:
        print("  CATASTROPHIC SUBGROUPS:")
        for name, sg in cs["subgroups"].items():
            if sg["catastrophic"]:
                print(f"    {name}: R1={sg['r1_success']:.4f} A1={sg['a1_success']:.4f} delta_u={sg['r1_a1_delta_u']}")

    # 4. Degradation curve
    print("\nComputing semantic degradation curve...")
    report["degradation_curve"] = compute_degradation_curve(results)
    print(f"  {'Tier':<12} {'RelMatch':>10} {'MDSGAcc':>10} {'T2Acc':>10} {'T2Prec':>10} {'T2Rec':>10} {'R1InfU':>10} {'R1GoldU':>10} {'GapR1':>10}")
    for row in report["degradation_curve"]:
        print(f"  {row['tier']:<12} {row['relation_match_rate']:>10.4f} {row['mdsg_state_accuracy']:>10.4f} "
              f"{row['t2_accuracy']:>10.4f} {row['t2_precision']:>10.4f} {row['t2_recall']:>10.4f} "
              f"{row['r1_inferred_utility']:>10.2f} {row['r1_gold_utility']:>10.2f} {row['semantic_gap_r1']:>10.2f}")

    # 5. Per-class detail
    print("\nComputing per-semantic-class detail...")
    report["per_class_detail"] = compute_per_class_detail(results)
    for cls, d in sorted(report["per_class_detail"].items()):
        print(f"  {cls} ({d['tier']}): rel={d['relation_match_rate']:.4f} state={d['mdsg_state_accuracy']:.4f} "
              f"t2={d['t2_accuracy']:.4f} R1inf={d['r1_inferred_success']:.4f} R1gold={d['r1_gold_success']:.4f} "
              f"gap={d['semantic_gap_r1']:.2f} causal={d['semantic_causal_count']}")

    # 6. Semantic gap comparison: is MDSG more or less sensitive?
    print("\nSemantic gap comparison (A1 vs M3 vs R1):")
    for row in report["degradation_curve"]:
        print(f"  {row['tier']}: A1 gap={row['semantic_gap_a1']:.2f}, M3 gap={row['semantic_gap_m3']:.2f}, R1 gap={row['semantic_gap_r1']:.2f}")

    # 7. Per-edge error type severity analysis
    print("\nComputing per-edge error type severity analysis...")
    report["error_type_severity"] = compute_per_edge_error_types(results, extractor)
    ets = report["error_type_severity"]
    print(f"  Total edge errors: {ets['total_edge_errors']}")
    print(f"  By error type: {ets['by_error_type']}")
    print(f"  Severity analysis (P(T2 FP | error type), P(task fail | error type)):")
    for et_name in ["FALSE_CONTRADICTION", "MISSED_SUPPORT", "FALSE_SUPPORT", "MISSED_CONTRADICTION"]:
        sa = ets["severity_analysis"][et_name]
        print(f"    {et_name}: n={sa['n_tasks_with_error']} "
              f"P(T2FP)={sa['p_t2_false_positive']} P(T2FN)={sa['p_t2_false_negative']} "
              f"P(fail)={sa['p_task_failure']}")
    print(f"  Severity ordering by P(T2 FP): {ets['severity_ordering']}")

    # Save
    out_path = ROOT / "experiments/v2b_i3_12/development/i3_12m_s2_stress/enhanced_analysis_v1.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nEnhanced analysis saved: {out_path}")


if __name__ == "__main__":
    main()
