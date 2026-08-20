"""I3.12h: S1 packet-level equivalence analysis.

Verifies that the S1 (raw-semantic) pipeline produces well-formed
packets with complete provenance chains, and that the provenance
chain is traceable from raw evidence text through to the snapshot.

The provenance chain is:
  raw evidence text
       -> hash
  relation extractor
       ->
  relation output
       -> hash
  snapshot
       -> hash
  MDSG state
       ->
  T2
       ->
  representation selected
       ->
  model action

This script does NOT run the model. It verifies the packet structure
and provenance chain up to the snapshot level.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    generate_i3_12_corpus,
)
from hrm_adaptive_memory.executive.semantic_relations.deterministic_rules import (
    DeterministicRelationExtractor,
)
from hrm_adaptive_memory.executive.semantic_relations.integration import (
    build_evidence_snapshot_with_inferred_relations,
    build_evidence_snapshot_oracle,
    infer_relations_for_runtime,
)
from hrm_adaptive_memory.executive.semantic_relations.serializer import (
    relation_graph_to_dict,
    relation_graph_to_supports_contradicts,
)
from hrm_adaptive_memory.executive.semantic_relations.schema import text_sha256
from hrm_adaptive_memory.executive.evidence_benchmark import (
    initial_evidence_runtime,
)
from hrm_adaptive_memory.executive.resources import ResourceState, ResourceBudget


def run_packet_equivalence_analysis(n_per_category: int = 5, seed: int = 42) -> dict:
    """Run the S1 packet-level equivalence analysis.

    Returns a JSON-serializable report.
    """
    tasks = generate_i3_12_corpus(n_per_category=n_per_category, seed=seed)
    ext = DeterministicRelationExtractor()

    budget = ResourceBudget(
        max_executive_steps=24, max_reasoning_tokens=2048,
        max_retrieval_calls=5, max_verification_calls=5,
        max_search_calls=5, max_elapsed_ms=10000,
    )

    report = {
        "analysis_id": "i3_12h_packet_equivalence_v1",
        "extractor_version": ext.identity.extractor_version,
        "extractor_sha256": ext.identity.sha256,
        "n_tasks": len(tasks),
        "n_per_category": n_per_category,
        "seed": seed,
        "tasks": [],
        "summary": {},
    }

    s0_s1_match_count = 0
    s0_s1_mismatch_count = 0
    provenance_complete_count = 0
    provenance_incomplete_count = 0

    for task in tasks:
        runtime = initial_evidence_runtime(task.evidence_task, ResourceState(budget))

        # S0 snapshot (oracle)
        snap_s0 = build_evidence_snapshot_oracle(runtime)

        # S1 snapshot (inferred)
        snap_s1, graph = build_evidence_snapshot_with_inferred_relations(runtime, ext)

        # Compare S0 and S1 visible evidence relations
        relations_match = True
        relation_details = []
        for ev0, ev1 in zip(snap_s0.visible_evidence, snap_s1.visible_evidence):
            s_match = ev0.supports == ev1.supports
            c_match = ev0.contradicts == ev1.contradicts
            if not s_match or not c_match:
                relations_match = False
            relation_details.append({
                "evidence_id": ev0.evidence_id,
                "s0_supports": list(ev0.supports),
                "s1_supports": list(ev1.supports),
                "s0_contradicts": list(ev0.contradicts),
                "s1_contradicts": list(ev1.contradicts),
                "match": s_match and c_match,
            })

        if relations_match:
            s0_s1_match_count += 1
        else:
            s0_s1_mismatch_count += 1

        # Verify provenance chain
        provenance = []
        provenance_complete = True
        for ev in snap_s1.visible_evidence:
            ev_hash = text_sha256(ev.proposition)
            # Find relations for this evidence in the graph
            ev_rels = [r for r in graph.relations if r.evidence_id == ev.evidence_id]
            for rel in ev_rels:
                entry = {
                    "evidence_id": ev.evidence_id,
                    "evidence_sha256": ev_hash,
                    "hypothesis_id": rel.hypothesis_id,
                    "hypothesis_sha256": rel.hypothesis_sha256,
                    "relation": rel.relation.value,
                    "reason_code": rel.reason_code.value,
                    "extractor_sha256": graph.extractor_identity_sha256,
                    "relation_graph_sha256": graph.relation_graph_sha256,
                }
                provenance.append(entry)
                if not ev_hash or not rel.hypothesis_sha256:
                    provenance_complete = False

        if provenance_complete:
            provenance_complete_count += 1
        else:
            provenance_incomplete_count += 1

        # Verify affordances are identical
        affordances_match = (
            snap_s0.can_retrieve == snap_s1.can_retrieve and
            snap_s0.can_search == snap_s1.can_search and
            snap_s0.can_verify == snap_s1.can_verify and
            snap_s0.resource_state == snap_s1.resource_state
        )

        report["tasks"].append({
            "task_id": task.task_id,
            "category": task.category,
            "relations_match": relations_match,
            "affordances_match": affordances_match,
            "provenance_complete": provenance_complete,
            "n_provenance_entries": len(provenance),
            "relation_details": relation_details,
            "provenance": provenance[:4],  # sample for brevity
            "relation_graph_sha256": graph.relation_graph_sha256,
        })

    report["summary"] = {
        "s0_s1_match": s0_s1_match_count,
        "s0_s1_mismatch": s0_s1_mismatch_count,
        "provenance_complete": provenance_complete_count,
        "provenance_incomplete": provenance_incomplete_count,
        "affordances_identical_all": all(t["affordances_match"] for t in report["tasks"]),
        "extractor_identity_sha256": ext.identity.sha256,
    }

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = run_packet_equivalence_analysis(
        n_per_category=args.n_per_category,
        seed=args.seed,
    )

    print(f"I3.12h: S1 Packet-Level Equivalence Analysis")
    print(f"  Tasks: {report['n_tasks']}")
    print(f"  Extractor: v{report['extractor_version']} ({report['extractor_sha256'][:16]})")
    print(f"  S0/S1 match: {report['summary']['s0_s1_match']}/{report['n_tasks']}")
    print(f"  S0/S1 mismatch: {report['summary']['s0_s1_mismatch']}/{report['n_tasks']}")
    print(f"  Provenance complete: {report['summary']['provenance_complete']}/{report['n_tasks']}")
    print(f"  Affordances identical: {report['summary']['affordances_identical_all']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
