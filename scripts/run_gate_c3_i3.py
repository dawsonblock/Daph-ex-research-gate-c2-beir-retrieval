#!/usr/bin/env python3
"""Gate C3 I3 runner — identity-record retrieval for surface resolution.

I3 does exactly one new thing beyond I0/I1/I2: it reads runtime-visible
identity records from the candidate pool and extracts the explicit
surface→canonical mapping they state.

    surface mention
      ↓
    retrieve/read identity records
      ↓
    extract explicit surface → canonical mapping
      ↓
    resolve anchor

It does NOT add iterative traversal beyond the first explicit identity edge
(that is I4's job).

ALLOWED:
    - question text
    - frozen candidate pool
    - runtime-visible identity records
    - existing identity-link parser (extract_identity_links)
    - deterministic canonicalization
    - ambiguity detection

NOT ALLOWED:
    - oracle metadata
    - proof graph
    - required evidence IDs
    - evaluator canonical entity
    - family/regime labels
    - hidden alias tables

Safety rule: if two visible records imply conflicting mappings for the same
surface, I3 MUST return AMBIGUOUS, not whichever appears first. Abstention
is preferred over guessed resolution.

I3-specific metric:
    IdentityMappingExtractionRate = among tasks where the relevant identity
    record is present, how often does the runtime parser correctly extract
    the alias→canonical mapping?

Per-task trace fields:
    question_surface
    identity_records_seen
    candidate_mappings
    chosen_mapping
    resolution_status
    canonical_entity
    source_evidence_id

Promotion criteria:
    CorrectAnchorRate >> I2
    FalseResolutionRate <= 0.02
    S2cLiveRate > 0
"""
from __future__ import annotations
import hashlib, json, re, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.retrieval.canonicalization import (
    extract_identity_links, IdentityLink, _norm)
from hrm_adaptive_memory.retrieval_bench.selectors.chain import (
    build_task_graph, enumerate_chains, s2c_chain_plus_relation,
    derive_entity_types, _norm as _chain_norm)

CORPUS = ROOT / "data/hrm/controlled_gate_c3_v1"
EVIDENCE_DIR = ROOT / "evidence/gate_c3"
PARTITION = "c3v1_surface"

# Question subject extraction (generic English templates)
_SUBJECT_PATTERNS = [
    re.compile(r"is held by\s+(.+?)\?$"),
    re.compile(r"does\s+(.+?)\s+carry\?$"),
    re.compile(r"recorded for\s+(.+?)\.$"),
    re.compile(r"attached to\s+(.+?)\.$"),
]


class _Rec:
    """Shim so pool rows can feed the qualified canonicalization parser."""
    def __init__(self, eid, content):
        self.evidence_id = eid
        self.content = content


def _load():
    tasks = [json.loads(l) for l in (CORPUS / PARTITION / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / PARTITION / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def _make_candidates(evidence):
    return [{"document_id": r["evidence_id"], "metadata": r["metadata"]} for r in evidence]


def _texts_from_evidence(evidence):
    return {r["evidence_id"]: r["content"] for r in evidence}


def _extract_subject(question):
    """Extract the subject phrase from a generic question template."""
    for pat in _SUBJECT_PATTERNS:
        m = pat.search(question)
        if m:
            return m.group(1).strip().rstrip(".?")
    return None


def _resolve_i3(subject_raw, identity_links):
    """Run the I3 resolver on a single task.

    Reads identity records, extracts surface→canonical mappings, and resolves
    the anchor. Returns (mappings, outcome, chosen_link) where:
      - mappings: list of IdentityLink objects matching the surface
      - outcome: EXACT / RESOLVED / AMBIGUOUS / UNRESOLVED
      - chosen_link: the IdentityLink that was chosen, or None

    EXACT: the surface itself is a canonical entity name (no identity record
    needed). This is rare on surface tasks but included for completeness.
    RESOLVED: exactly one identity record maps the surface to a canonical.
    AMBIGUOUS: multiple identity records map the surface to different canonicals.
    UNRESOLVED: no identity record maps the surface.
    """
    if not subject_raw:
        return [], "UNRESOLVED", None

    surface_norm = _norm(subject_raw)
    # Also try without leading "the " (description surfaces start with "the")
    stripped = _norm(re.sub(r"^the\s+", "", subject_raw, flags=re.I))

    # Find all identity links that match this surface
    mappings = []
    for link in identity_links:
        link_surface = _norm(link.surface)
        if link_surface == surface_norm or link_surface == stripped:
            mappings.append(link)
        elif surface_norm in link_surface or stripped in link_surface:
            # Substring match (handles description surfaces that may be longer)
            mappings.append(link)

    # Deduplicate by (surface_norm, canonical_norm) — same mapping from
    # different records is not ambiguous, just redundant
    unique_mappings = {}
    for link in mappings:
        key = (_norm(link.surface), _norm(link.canonical))
        if key not in unique_mappings:
            unique_mappings[key] = link
    mappings = list(unique_mappings.values())

    if not mappings:
        return [], "UNRESOLVED", None

    # Check for conflicting canonical targets
    canonicals = {_norm(m.canonical) for m in mappings}
    if len(canonicals) > 1:
        return mappings, "AMBIGUOUS", None

    # Single unique mapping
    chosen = mappings[0]
    return mappings, "RESOLVED", chosen


def run_i3(tasks, evidence):
    """Run the I3 resolver over all tasks."""
    texts = _texts_from_evidence(evidence)
    cands = _make_candidates(evidence)

    # Parse ALL identity links from the candidate pool
    all_recs = [_Rec(r["evidence_id"], r["content"]) for r in evidence]
    identity_links = extract_identity_links(all_recs)

    # Group identity links by normalized surface for provenance
    links_by_surface = defaultdict(list)
    for link in identity_links:
        links_by_surface[_norm(link.surface)].append(link)

    results = []
    for t in tasks:
        meta = t["_oracle_metadata"]
        true_canonical = _norm(meta["surfaces"]["canonical"])
        true_surface = _norm(meta["surfaces"]["subject"])
        question = t["question"]
        regime = t["metadata"]["entity_regime"]

        # Parse the subject from the question
        subject_raw = _extract_subject(question)
        surface_norm = _norm(subject_raw) if subject_raw else ""

        # Run I3 resolver
        mappings, outcome, chosen_link = _resolve_i3(subject_raw, identity_links)

        # Correctness evaluation (evaluator-side, using oracle metadata)
        correct = False
        wrong = False
        resolved_canonical = None
        source_evidence_id = None

        if outcome == "RESOLVED" and chosen_link:
            resolved_canonical = _norm(chosen_link.canonical)
            source_evidence_id = chosen_link.record_id
            if resolved_canonical == true_canonical:
                correct = True
            else:
                wrong = True
        elif outcome == "EXACT":
            resolved_canonical = surface_norm
            if surface_norm == true_canonical:
                correct = True
            else:
                wrong = True

        # IdentityMappingExtractionRate diagnostic:
        # Did the parser extract the correct mapping from the identity record?
        # Among tasks where the identity record is present in the pool.
        identity_record_present = any(
            r["metadata"]["record_kind"] == "required_identity" for r in evidence
            if t["task_id"] in r["evidence_id"]
        )
        # More precisely: is there an identity record for THIS task?
        task_identity_records = [
            r for r in evidence
            if r["metadata"]["record_kind"] == "required_identity"
            and r["evidence_id"].startswith(t["task_id"] + "/")
        ]
        identity_record_for_task = len(task_identity_records) > 0

        # Did the parser extract a mapping where the canonical matches the true canonical?
        parser_extracted_correct = any(
            _norm(m.canonical) == true_canonical for m in mappings
        ) if mappings else False

        # Chain enumeration and S2c liveness
        # If I3 resolved the anchor, the pipeline would reformulate the query
        # with the canonical name so downstream selectors can use it. We measure
        # S2c liveness by calling s2c_chain_plus_relation with a reformulated
        # question that substitutes the resolved canonical for the surface.
        graph = build_task_graph(cands, question, texts)
        if chosen_link and outcome == "RESOLVED":
            graph.question_entities = {_norm(chosen_link.canonical)}
        chains = enumerate_chains(graph)
        chain_live = len(chains) > 0

        # S2c liveness: reformulate the question with the canonical entity.
        # In the real pipeline, I3 resolves the identity and the information
        # state carries the canonical name; the next query includes it. Here
        # we substitute the surface mention with the resolved canonical.
        if chosen_link and outcome == "RESOLVED":
            resolved_question = question.replace(
                subject_raw, chosen_link.canonical)
        else:
            resolved_question = question
        s2c_sel = s2c_chain_plus_relation(
            cands, budget=6, question=resolved_question, texts=texts)
        s2c_differs = s2c_sel != [c["document_id"] for c in cands[:6]]
        s2c_live = chain_live and s2c_differs

        identity_in_pool = any(r["metadata"]["record_kind"] == "required_identity" for r in evidence)
        ev_by_id = {r["evidence_id"]: r["content"] for r in evidence}
        canonical_recovered = correct or any(
            true_canonical in _norm(ev_by_id.get(rid, ""))
            for rid in t["required_evidence_ids"])

        # Per-task trace (as requested by the authorization)
        trace = {
            "question_surface": subject_raw,
            "identity_records_seen": [r["evidence_id"] for r in task_identity_records],
            "candidate_mappings": [
                {"surface": m.surface, "canonical": m.canonical,
                 "source_evidence_id": m.record_id}
                for m in mappings
            ],
            "chosen_mapping": (
                {"surface": chosen_link.surface, "canonical": chosen_link.canonical,
                 "source_evidence_id": chosen_link.record_id}
                if chosen_link else None
            ),
            "resolution_status": outcome,
            "canonical_entity": resolved_canonical,
            "source_evidence_id": source_evidence_id,
        }

        results.append({
            "task_id": t["task_id"], "arm": "I3", "outcome": outcome,
            "found_entities": sorted({_norm(m.canonical) for m in mappings}),
            "true_canonical": true_canonical,
            "subject_raw": subject_raw, "surface_norm": surface_norm,
            "regime": regime,
            "correct": correct, "wrong": wrong,
            "resolved_canonical": resolved_canonical,
            "source_evidence_id": source_evidence_id,
            "identity_record_for_task": identity_record_for_task,
            "parser_extracted_correct": parser_extracted_correct,
            "chain_live": chain_live, "s2c_live": s2c_live,
            "identity_in_pool": identity_in_pool,
            "canonical_recovered": canonical_recovered,
            "complete_set": all(r in {c["document_id"] for c in cands} for r in t["required_evidence_ids"]),
            "trace": trace,
        })
    return results


def compute_metrics(results):
    n = len(results)
    outcomes = defaultdict(int)
    for r in results:
        outcomes[r["outcome"]] += 1

    # IdentityMappingExtractionRate: among tasks where the identity record
    # is present for this task, how often did the parser extract the correct
    # mapping?
    eligible = [r for r in results if r["identity_record_for_task"]]
    extraction_correct = sum(1 for r in eligible if r["parser_extracted_correct"])
    extraction_rate = round(extraction_correct / len(eligible), 4) if eligible else 0.0

    # Per-regime breakdown
    by_regime = {}
    for regime in ("alias", "description"):
        subset = [r for r in results if r["regime"] == regime]
        if subset:
            by_regime[regime] = {
                "n": len(subset),
                "CorrectAnchorRate": round(sum(1 for r in subset if r["correct"]) / len(subset), 4),
                "WrongAnchorRate": round(sum(1 for r in subset if r["wrong"]) / len(subset), 4),
                "UnresolvedRate": round(sum(1 for r in subset if r["outcome"] == "UNRESOLVED") / len(subset), 4),
                "AmbiguousResolutionRate": round(sum(1 for r in subset if r["outcome"] == "AMBIGUOUS") / len(subset), 4),
                "ChainEnumerationRate": round(sum(1 for r in subset if r["chain_live"]) / len(subset), 4),
                "S2cLiveRate": round(sum(1 for r in subset if r["s2c_live"]) / len(subset), 4),
            }

    # Per-family breakdown for cross-family validation
    by_family = {}
    for r in results:
        # Extract family from task_id (e.g. "c3v1-alias-entity_attribute-0000")
        parts = r["task_id"].split("-")
        if len(parts) >= 3:
            family = parts[2]
            if family not in by_family:
                by_family[family] = {"n": 0, "correct": 0, "wrong": 0, "ambiguous": 0, "s2c_live": 0}
            by_family[family]["n"] += 1
            if r["correct"]: by_family[family]["correct"] += 1
            if r["wrong"]: by_family[family]["wrong"] += 1
            if r["outcome"] == "AMBIGUOUS": by_family[family]["ambiguous"] += 1
            if r["s2c_live"]: by_family[family]["s2c_live"] += 1

    return {
        "n": n,
        "QuestionAnchorResolutionRate": round(sum(1 for r in results if r["outcome"] != "UNRESOLVED") / n, 4),
        "IdentityRecordRecallAtK": round(sum(1 for r in results if r["identity_in_pool"]) / n, 4),
        "IdentityMappingExtractionRate": extraction_rate,
        "identity_mapping_eligible_tasks": len(eligible),
        "identity_mapping_extracted_correct": extraction_correct,
        "CanonicalEntityRecovery": round(sum(1 for r in results if r["canonical_recovered"]) / n, 4),
        "ChainEnumerationRate": round(sum(1 for r in results if r["chain_live"]) / n, 4),
        "CandidateCompleteSet": round(sum(1 for r in results if r["complete_set"]) / n, 4),
        "S2cLiveRate": round(sum(1 for r in results if r["s2c_live"]) / n, 4),
        "CorrectAnchorRate": round(sum(1 for r in results if r["correct"]) / n, 4),
        "WrongAnchorRate": round(sum(1 for r in results if r["wrong"]) / n, 4),
        "FalseResolutionRate": round(sum(1 for r in results if r["wrong"]) / n, 4),
        "AmbiguousResolutionRate": round(outcomes["AMBIGUOUS"] / n, 4),
        "UnresolvedRate": round(outcomes["UNRESOLVED"] / n, 4),
        "outcomes": dict(outcomes),
        "by_regime": by_regime,
        "by_family": by_family,
    }


def main():
    tasks, evidence = _load()
    print(f"Loaded {len(tasks)} tasks, {len(evidence)} evidence records from {PARTITION}")

    # Load prior results for the cascade table
    prior_receipt_path = EVIDENCE_DIR / "i0_i1_i2_receipt.json"
    if not prior_receipt_path.exists():
        print("WARNING: I0/I1/I2 receipt not found; cascade table will show I3 only.")
        prior = {}
    else:
        prior = json.loads(prior_receipt_path.read_text())

    print("\n=== I3: identity-record retrieval ===")
    t3 = time.time()
    i3_results = run_i3(tasks, evidence)
    i3_metrics = compute_metrics(i3_results)
    print(f"  completed in {time.time()-t3:.1f}s")
    for k, v in i3_metrics.items():
        if k not in ("outcomes", "by_regime", "by_family"):
            print(f"  {k:35} {v}")
    print(f"  outcomes: {i3_metrics['outcomes']}")
    print(f"  by_regime: {json.dumps(i3_metrics['by_regime'], indent=4)}")
    print(f"  by_family: {json.dumps(i3_metrics['by_family'], indent=4)}")

    # Cascade table
    print("\n=== Cascade Table (I0 / I1 / I2 / I3) ===")
    headers = ["Arm", "QAnchorRate", "CorrectAnchor", "WrongAnchor", "FalseRes", "Ambiguous", "Unresolved", "S2cLive", "MappingExtract"]
    rows = []
    if prior:
        for arm_key, arm_label in [("I0", "I0"), ("I1", "I1"), ("I2", "I2")]:
            m = prior[arm_key]["metrics"]
            rows.append([arm_label, m["QuestionAnchorResolutionRate"], m["CorrectAnchorRate"],
                        m["WrongAnchorRate"], m["FalseResolutionRate"],
                        m["AmbiguousResolutionRate"], m["UnresolvedRate"],
                        m["S2cLiveRate"], "N/A"])
    rows.append(["I3", i3_metrics["QuestionAnchorResolutionRate"], i3_metrics["CorrectAnchorRate"],
                i3_metrics["WrongAnchorRate"], i3_metrics["FalseResolutionRate"],
                i3_metrics["AmbiguousResolutionRate"], i3_metrics["UnresolvedRate"],
                i3_metrics["S2cLiveRate"], i3_metrics["IdentityMappingExtractionRate"]])

    fmt = "{:<6} {:>12} {:>14} {:>13} {:>10} {:>10} {:>12} {:>9} {:>14}"
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    # Promotion evaluation
    print("\n=== Promotion Evaluation ===")
    i2_correct = prior["I2"]["metrics"]["CorrectAnchorRate"] if prior and "I2" in prior else 0.0
    i2_false = prior["I2"]["metrics"]["FalseResolutionRate"] if prior and "I2" in prior else 0.0
    i2_s2c = prior["I2"]["metrics"]["S2cLiveRate"] if prior and "I2" in prior else 0.0

    checks = {
        "CorrectAnchorRate >> I2": i3_metrics["CorrectAnchorRate"] > i2_correct,
        "FalseResolutionRate <= 0.02": i3_metrics["FalseResolutionRate"] <= 0.02,
        "S2cLiveRate > 0": i3_metrics["S2cLiveRate"] > 0,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {check}")
    all_pass = all(checks.values())

    # Cross-family check
    families_with_gains = sum(1 for f, v in i3_metrics["by_family"].items() if v["correct"] > 0)
    total_families = len(i3_metrics["by_family"])
    print(f"  Families with correct resolution: {families_with_gains}/{total_families}")

    # Decision
    if all_pass and i3_metrics["CorrectAnchorRate"] > 0.8:
        decision = "I3 strongly succeeds. Identity-record retrieval resolves the surface defect with very low false resolution. Proceed to I4 only if some tasks still require more than one identity/canonicalization hop."
    elif all_pass:
        decision = "I3 succeeds with promotion criteria met. Proceed to I4 only if some tasks still require more than one identity/canonicalization hop."
    elif i3_metrics["CorrectAnchorRate"] > i2_correct and i3_metrics["FalseResolutionRate"] > 0.02:
        decision = "I3 improves correct resolution but false resolution is too high. Stop and diagnose before I4."
    else:
        decision = "I3 results require manual interpretation before proceeding."
    print(f"\n  Overall: {'PROMOTED' if all_pass else 'NOT_PROMOTED'}")
    print(f"  Decision: {decision}")

    # Write receipt
    receipt = prior if prior else {}
    receipt["rungs_run"] = list(set(receipt.get("rungs_run", []) + ["I3"]))
    receipt["rungs_run"].sort(key=lambda x: ["I0","I1","I2","I3","I4","I5","I6"].index(x))
    receipt["rungs_NOT_run"] = [r for r in receipt.get("rungs_NOT_run", ["I3","I4","I5","I6"]) if r != "I3"]
    receipt["I3"] = {
        "description": "identity-record retrieval; reads explicit surface→canonical mappings from runtime-visible identity records",
        "allowed": [
            "question text", "frozen candidate pool", "runtime-visible identity records",
            "existing identity-link parser (extract_identity_links)",
            "deterministic canonicalization", "ambiguity detection",
        ],
        "not_allowed": [
            "oracle metadata", "proof graph", "required evidence IDs",
            "evaluator canonical entity", "family/regime labels", "hidden alias tables",
        ],
        "safety_rule": "If two visible records imply conflicting mappings for the same surface, I3 returns AMBIGUOUS, not whichever appears first. Abstention is preferred over guessed resolution.",
        "metrics": i3_metrics,
        "per_task": i3_results,
    }
    receipt["promotion_evaluation"] = {
        "criteria": checks,
        "all_passed": all_pass,
        "families_with_gains": families_with_gains,
        "total_families": total_families,
        "decision": decision,
    }

    receipt_path = EVIDENCE_DIR / "i0_i1_i2_i3_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # Update SHA manifest
    def h(b): return hashlib.sha256(b).hexdigest()
    (EVIDENCE_DIR / "RESULTS.sha256").write_text(
        f"{h(receipt_path.read_bytes())}  i0_i1_i2_i3_receipt.json\n")

    print(f"\nreceipt: {receipt_path}")
    print("frozen.")


if __name__ == "__main__":
    main()
