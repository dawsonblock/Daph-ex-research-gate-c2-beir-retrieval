#!/usr/bin/env python3
"""Gate C3 I4 opportunity audit — should I4 be implemented?

I4 is defined as "bounded canonicalization through retrieved identity edges."
This audit classifies each of the 8 ambiguous I3 tasks into:

  A = RESOLVABLE_BY_SECOND_IDENTITY_EDGE
      A second explicit runtime-visible identity edge can uniquely resolve
      the candidate canonical.

  B = RESOLVABLE_BY_NON_IDENTITY_CONTEXT
      A non-oracle visible relation/context record uniquely distinguishes
      the correct canonical without using proof metadata.

  C = GENUINELY_CONFLICTING_SURFACE_MAPPING
      The visible evidence genuinely supports multiple mappings; no
      legitimate runtime-visible evidence distinguishes them.

  D = REQUIRED_DISAMBIGUATING_EVIDENCE_ABSENT
      The evaluator can identify the correct canonical, but the evidence
      required to distinguish it is not present in the frozen candidate pool.

Decision threshold (FROZEN before inspection):
  I4OpportunityRate = (n_A + n_B) / ambiguous_task_count
  If I4OpportunityRate < 0.25: I4 = NO_MEASURED_OPPORTUNITY, skip implementation.
  If I4OpportunityRate >= 0.25: authorize narrowly scoped I4 implementation.

Runtime-visible data only. Evaluator truth used only for scoring/classification.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.retrieval.canonicalization import (
    extract_identity_links, _norm)

CORPUS = ROOT / "data/hrm/controlled_gate_c3_v1"
EVIDENCE_DIR = ROOT / "evidence/gate_c3"
OUT = EVIDENCE_DIR / "i4_opportunity"
PARTITION = "c3v1_surface"
THRESHOLD = 0.25  # FROZEN before inspection


class _Rec:
    def __init__(self, eid, content):
        self.evidence_id = eid
        self.content = content


def _load():
    tasks = [json.loads(l) for l in (CORPUS / PARTITION / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / PARTITION / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def classify_ambiguous_task(task, evidence, mappings, true_canonical):
    """Classify one ambiguous task into A/B/C/D.

    Uses runtime-visible evidence only. Evaluator truth (true_canonical) is
    used only to verify which candidate is correct, not to drive runtime logic.
    """
    ev_by_id = {r["evidence_id"]: r for r in evidence}
    tid = task["task_id"]
    meta = task["_oracle_metadata"]
    target_relation = meta["target_relation"]

    # All identity links in the pool
    all_recs = [_Rec(r["evidence_id"], r["content"]) for r in evidence]
    all_identity_links = extract_identity_links(all_recs)

    candidate_canonicals = [_norm(m["canonical"]) for m in mappings]
    identity_record_ids = [m["source_evidence_id"] for m in mappings]

    # --- Check A: second identity edge ---
    # For each candidate canonical, check if there's a second identity edge
    # that uniquely identifies it (e.g., canonical → something_else that only
    # one candidate has).
    for canon in candidate_canonicals:
        second_edges = [
            link for link in all_identity_links
            if _norm(link.surface) == canon and link.record_id not in identity_record_ids
        ]
        # If exactly one candidate has second edges, and the other doesn't,
        # that could disambiguate. But this is about whether a SECOND edge
        # from the canonical leads somewhere unique.
    # Check: does any candidate canonical appear as a surface in another
    # identity record that the other candidate does NOT appear in?
    a_resolvable = False
    for i, canon_i in enumerate(candidate_canonicals):
        has_unique_second_edge = False
        other_canons = [c for j, c in enumerate(candidate_canonicals) if j != i]
        # Look for identity records where canon_i is the surface and maps
        # to something, but none of the other candidates are involved
        for link in all_identity_links:
            if _norm(link.surface) == canon_i and link.record_id not in identity_record_ids:
                # Check that no other candidate canonical appears in this record
                content_norm = _norm(ev_by_id[link.record_id]["content"])
                if not any(oc in content_norm for oc in other_canons):
                    has_unique_second_edge = True
                    break
        if has_unique_second_edge and canon_i == true_canonical:
            a_resolvable = True
            break

    # --- Check B: non-identity context ---
    # Check if the correct canonical appears in non-identity evidence records
    # that also match the question's target relation, while the wrong canonical
    # does NOT appear in such records.
    # Runtime-visible: we check all non-identity evidence records in the pool.
    non_identity_ev = [r for r in evidence if r["metadata"]["record_kind"] != "required_identity"]
    relation_norm = _norm(target_relation)

    canon_in_relation_context = {}
    for canon in candidate_canonicals:
        appears_with_relation = any(
            canon in _norm(r["content"]) and relation_norm in _norm(r["content"])
            for r in non_identity_ev
        )
        canon_in_relation_context[canon] = appears_with_relation

    # Also check: does the canonical appear in ANY non-identity context?
    canon_in_any_context = {}
    for canon in candidate_canonicals:
        appears = any(canon in _norm(r["content"]) for r in non_identity_ev)
        canon_in_any_context[canon] = appears

    # B: exactly one candidate appears in relation context
    true_in_relation = canon_in_relation_context.get(true_canonical, False)
    wrong_canons = [c for c in candidate_canonicals if c != true_canonical]
    wrong_in_relation = any(canon_in_relation_context.get(c, False) for c in wrong_canons)
    b_resolvable_by_relation = true_in_relation and not wrong_in_relation

    # B alternative: exactly one candidate appears in ANY non-identity context
    true_in_any = canon_in_any_context.get(true_canonical, False)
    wrong_in_any = any(canon_in_any_context.get(c, False) for c in wrong_canons)
    b_resolvable_by_any = true_in_any and not wrong_in_any

    b_resolvable = b_resolvable_by_relation or b_resolvable_by_any

    # --- Check C: genuinely conflicting ---
    # Both candidates appear in similar context, no distinguishing signal
    c_genuine_conflict = not a_resolvable and not b_resolvable

    # --- Check D: evidence absent ---
    # The correct canonical cannot be found in any runtime-visible evidence
    # (not even in identity records for this task)
    # This would mean even the task's own identity record is missing
    d_evidence_absent = not true_in_any and not any(
        true_canonical in _norm(r["content"]) for r in evidence
        if r["metadata"]["record_kind"] != "required_identity"
    )

    # Classification
    if a_resolvable:
        classification = "A"
        reason = "A second explicit runtime-visible identity edge can uniquely resolve the candidate canonical."
        runtime_resolvable = True
    elif b_resolvable:
        if b_resolvable_by_relation:
            reason = ("The correct canonical appears in non-identity evidence records that also "
                     "contain the question's target relation, while the wrong canonical does not. "
                     "Relation-context disambiguation is possible.")
        else:
            reason = ("The correct canonical appears in non-identity evidence records while the "
                     "wrong canonical does not appear in any non-identity context. "
                     "Presence-based disambiguation is possible.")
        classification = "B"
        runtime_resolvable = True
    elif c_genuine_conflict:
        classification = "C"
        reason = "The visible evidence genuinely supports multiple mappings; no legitimate runtime-visible evidence distinguishes them."
        runtime_resolvable = False
    elif d_evidence_absent:
        classification = "D"
        reason = "The evidence required to distinguish the correct canonical is not present in the frozen candidate pool."
        runtime_resolvable = False
    else:
        classification = "C"
        reason = "Unclassified; defaulting to genuine conflict."
        runtime_resolvable = False

    # Collect context records that mention the correct canonical
    context_records = [
        r["evidence_id"] for r in non_identity_ev
        if true_canonical in _norm(r["content"])
    ]

    return {
        "task_id": tid,
        "surface": task["_oracle_metadata"]["surfaces"]["subject"],
        "candidate_canonicals": [m["canonical"] for m in mappings],
        "identity_records": identity_record_ids,
        "context_records": context_records,
        "classification": classification,
        "runtime_resolvable": runtime_resolvable,
        "reason": reason,
        "evaluator_correct_canonical": task["_oracle_metadata"]["surfaces"]["canonical"],
        "target_relation": target_relation,
        "canon_in_relation_context": canon_in_relation_context,
        "canon_in_any_context": canon_in_any_context,
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"I4 opportunity audit already exists: {OUT}")

    tasks, evidence = _load()
    task_by_id = {t["task_id"]: t for t in tasks}

    # Load I3 results to get the ambiguous tasks
    receipt = json.loads((EVIDENCE_DIR / "i0_i1_i2_i3_receipt.json").read_text())
    i3_per_task = receipt["I3"]["per_task"]
    ambiguous = [r for r in i3_per_task if r["outcome"] == "AMBIGUOUS"]

    print(f"I4 Opportunity Audit")
    print(f"Frozen threshold: I4OpportunityRate < {THRESHOLD} => skip I4")
    print(f"Ambiguous tasks: {len(ambiguous)}")
    print()

    per_task_results = []
    for r in ambiguous:
        tid = r["task_id"]
        t = task_by_id[tid]
        mappings = r["trace"]["candidate_mappings"]
        true_canonical = r["true_canonical"]
        result = classify_ambiguous_task(t, evidence, mappings, true_canonical)
        per_task_results.append(result)
        print(f"  {tid}: classification={result['classification']} surface={result['surface']!r}")
        print(f"    {result['reason']}")

    # Compute metrics
    n_A = sum(1 for r in per_task_results if r["classification"] == "A")
    n_B = sum(1 for r in per_task_results if r["classification"] == "B")
    n_C = sum(1 for r in per_task_results if r["classification"] == "C")
    n_D = sum(1 for r in per_task_results if r["classification"] == "D")
    n_ambiguous = len(per_task_results)

    i4_opportunity_rate = round((n_A + n_B) / n_ambiguous, 4) if n_ambiguous else 0.0
    second_edge_rate = round(n_A / n_ambiguous, 4) if n_ambiguous else 0.0
    context_disambig_rate = round(n_B / n_ambiguous, 4) if n_ambiguous else 0.0
    true_conflict_rate = round(n_C / n_ambiguous, 4) if n_ambiguous else 0.0
    missing_evidence_rate = round(n_D / n_ambiguous, 4) if n_ambiguous else 0.0

    metrics = {
        "ambiguous_task_count": n_ambiguous,
        "n_A_RESOLVABLE_BY_SECOND_IDENTITY_EDGE": n_A,
        "n_B_RESOLVABLE_BY_NON_IDENTITY_CONTEXT": n_B,
        "n_C_GENUINELY_CONFLICTING_SURFACE_MAPPING": n_C,
        "n_D_REQUIRED_DISAMBIGUATING_EVIDENCE_ABSENT": n_D,
        "I4OpportunityRate": i4_opportunity_rate,
        "SecondIdentityEdgeOpportunityRate": second_edge_rate,
        "ContextDisambiguationOpportunityRate": context_disambig_rate,
        "TrueConflictRate": true_conflict_rate,
        "MissingEvidenceRate": missing_evidence_rate,
        "frozen_threshold": THRESHOLD,
        "decision": "I4_AUTHORIZED" if i4_opportunity_rate >= THRESHOLD else "I4_NO_MEASURED_OPPORTUNITY_SKIP",
    }

    print()
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:50} {v}")

    print()
    if i4_opportunity_rate >= THRESHOLD:
        print(f"  DECISION: I4 AUTHORIZED (I4OpportunityRate={i4_opportunity_rate} >= {THRESHOLD})")
        print(f"  Note: {n_B} of {n_ambiguous} tasks are resolvable by non-identity context (B),")
        print(f"        not by second identity edges (A={n_A}). I4's design as 'bounded")
        print(f"        canonicalization through retrieved identity edges' may need to be")
        print(f"        broadened to include context-based disambiguation, or a separate")
        print(f"        mechanism should handle B-class disambiguation.")
    else:
        print(f"  DECISION: I4 SKIPPED (I4OpportunityRate={i4_opportunity_rate} < {THRESHOLD})")

    # Write artifacts
    OUT.mkdir(parents=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "audit": "gate_c3_i4_opportunity",
        "protocol_version": "v2_pre_measurement_amended",
        "corpus": "controlled_gate_c3_v1",
        "partition": PARTITION,
        "frozen_threshold": THRESHOLD,
        "description": "Classifies each ambiguous I3 task into A/B/C/D to determine whether I4 has measured opportunity.",
        "definitions": {
            "A": "RESOLVABLE_BY_SECOND_IDENTITY_EDGE",
            "B": "RESOLVABLE_BY_NON_IDENTITY_CONTEXT",
            "C": "GENUINELY_CONFLICTING_SURFACE_MAPPING",
            "D": "REQUIRED_DISAMBIGUATING_EVIDENCE_ABSENT",
        },
        "decision_rule": "I4OpportunityRate < 0.25 => skip; >= 0.25 => authorize",
    }, indent=2, sort_keys=True) + "\n")

    (OUT / "per_task.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in per_task_results))

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    def h(b): return hashlib.sha256(b).hexdigest()
    files = ["manifest.json", "per_task.jsonl", "metrics.json"]
    (OUT / "RESULTS.sha256").write_text(
        "".join(f"{h((OUT / f).read_bytes())}  {f}\n" for f in files))

    print(f"\nartifacts: {OUT}")


if __name__ == "__main__":
    main()
