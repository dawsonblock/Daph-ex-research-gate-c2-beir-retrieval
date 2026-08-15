#!/usr/bin/env python3
"""Gate C3 I0/I1 runner — surface identity resolution on the sixth-vocabulary corpus.

I0: current surface anchor extraction (the mechanism that is currently inert).
    Uses ``build_task_graph`` from chain.py, which derives entity types from the
    candidate pool and extracts mentions from the question. If the question
    subject is a truncated alias (e.g. "Beachy prism" vs canonical "Fastnet fog
    siren"), the entity type "fog siren" is recognized but the surface "prism"
    is not a qualified entity type, so no anchor is found.

I1: normalization-only negative control.
    Applies case/punctuation/whitespace normalization to both the question and
    the candidate text before running the SAME entity type derivation and mention
    extraction as I0. Normalization CANNOT recover a missing token, and the alias
    surface is a DIFFERENT name from the canonical (different head, truncated
    role), so exact normalized matching cannot bridge them. Only the identity
    record links them, and reading identity records is I3's job, not I1's.

    Therefore I1 ~= I0 is the predicted outcome on truncation failures. An I1
    result indistinguishable from I0 is the predicted negative-control outcome,
    not an engineering setback.

Both rungs use the task's full evidence set as the candidate pool, which
isolates identity resolution from retrieval quality. If the anchor fails even
with all evidence present, retrieval improvements alone cannot fix it.

Metrics measured (per the amended C3 protocol):
    QuestionAnchorResolutionRate
    IdentityRecordRecallAtK
    CanonicalEntityRecovery
    ChainEnumerationRate
    CandidateCompleteSet
    S2cLiveRate
    CorrectAnchorRate
    WrongAnchorRate
    FalseResolutionRate
    AmbiguousResolutionRate
    UnresolvedRate

Resolution contract: EXACT / RESOLVED / AMBIGUOUS / UNRESOLVED.
"""
from __future__ import annotations
import hashlib, json, re, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.retrieval_bench.selectors.chain import (
    build_task_graph, enumerate_chains, s2c_chain_plus_relation,
    derive_entity_types, extract_mentions, _norm)

CORPUS = ROOT / "data/hrm/controlled_gate_c3_v1"
OUT = ROOT / "evidence/gate_c3"
PARTITION = "c3v1_surface"


def _load():
    tasks = [json.loads(l) for l in (CORPUS / PARTITION / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / PARTITION / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def _make_candidates(evidence):
    return [{"document_id": r["evidence_id"], "metadata": r["metadata"]} for r in evidence]


def _texts_from_evidence(evidence):
    return {r["evidence_id"]: r["content"] for r in evidence}


def _normalize_text(text):
    """Case/punctuation/whitespace normalization for I1."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _classify(found_entities, true_canonical, graph):
    """Apply the resolution contract."""
    if not found_entities:
        return "UNRESOLVED", False, False
    if len(found_entities) > 1:
        return "AMBIGUOUS", False, False
    entity = next(iter(found_entities))
    if entity == true_canonical:
        return "EXACT", True, False
    resolved = graph.resolves(entity) if graph else None
    if resolved and resolved == true_canonical:
        return "RESOLVED", True, False
    if resolved:
        return "RESOLVED", False, True  # resolved to wrong canonical
    return "UNRESOLVED", False, False  # found something but it doesn't resolve


def _run_arm(tasks, evidence, arm, normalize=False):
    """Run one resolution arm (I0 or I1) over all tasks."""
    texts = _texts_from_evidence(evidence)
    cands = _make_candidates(evidence)
    results = []
    for t in tasks:
        meta = t["_oracle_metadata"]
        true_canonical = _norm(meta["surfaces"]["canonical"])
        question = t["question"]

        if normalize:
            # I1: normalize both question and candidate text, then run the same
            # entity type derivation + mention extraction as I0.
            norm_texts = {k: _normalize_text(v) for k, v in texts.items()}
            norm_question = _normalize_text(question)
            corpus = [norm_texts.get(c["document_id"], "") for c in cands]
            entity_types = derive_entity_types(corpus + [norm_question])
            found = extract_mentions(norm_question, entity_types)
        else:
            # I0: run the current mechanism as-is.
            graph = build_task_graph(cands, question, texts)
            found = graph.question_entities

        # For classification, we need the graph (for refers_to resolution).
        # I1 uses the original-text graph since normalization doesn't change
        # the identity records' content.
        graph = build_task_graph(cands, question, texts)
        outcome, correct, wrong = _classify(found, true_canonical, graph)

        # Chain enumeration and S2c liveness (same for both arms since the
        # chain mechanism is not changed by normalization).
        chains = enumerate_chains(graph)
        chain_live = len(chains) > 0
        s2c_sel = s2c_chain_plus_relation(cands, budget=6, question=question, texts=texts)
        s2c_differs = s2c_sel != [c["document_id"] for c in cands[:6]]
        s2c_live = chain_live and s2c_differs

        identity_in_pool = any(r["metadata"]["record_kind"] == "required_identity" for r in evidence)
        ev_by_id = {r["evidence_id"]: r["content"] for r in evidence}
        canonical_recovered = correct or any(
            true_canonical in _norm(ev_by_id.get(rid, ""))
            for rid in t["required_evidence_ids"])

        results.append({
            "task_id": t["task_id"], "arm": arm, "outcome": outcome,
            "found_entities": sorted(found), "true_canonical": true_canonical,
            "entity_types_count": len(entity_types) if normalize else len(derive_entity_types([texts.get(c["document_id"], "") for c in cands] + [question])),
            "correct": correct, "wrong": wrong,
            "chain_live": chain_live, "s2c_live": s2c_live,
            "identity_in_pool": identity_in_pool,
            "canonical_recovered": canonical_recovered,
            "complete_set": all(r in {c["document_id"] for c in cands} for r in t["required_evidence_ids"]),
        })
    return results


def compute_metrics(results):
    n = len(results)
    outcomes = defaultdict(int)
    for r in results:
        outcomes[r["outcome"]] += 1
    return {
        "n": n,
        "QuestionAnchorResolutionRate": round(sum(1 for r in results if r["outcome"] != "UNRESOLVED") / n, 4),
        "IdentityRecordRecallAtK": round(sum(1 for r in results if r["identity_in_pool"]) / n, 4),
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
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"C3 evidence dir already exists: {OUT}")
    tasks, evidence = _load()
    print(f"Loaded {len(tasks)} tasks, {len(evidence)} evidence records from {PARTITION}")

    print("\n=== I0: current surface anchor extraction ===")
    t0 = time.time()
    i0_results = _run_arm(tasks, evidence, "I0", normalize=False)
    i0_metrics = compute_metrics(i0_results)
    print(f"  completed in {time.time()-t0:.1f}s")
    for k, v in i0_metrics.items():
        if k != "outcomes": print(f"  {k:35} {v}")
    print(f"  outcomes: {i0_metrics['outcomes']}")

    print("\n=== I1: normalization-only negative control ===")
    t1 = time.time()
    i1_results = _run_arm(tasks, evidence, "I1", normalize=True)
    i1_metrics = compute_metrics(i1_results)
    print(f"  completed in {time.time()-t1:.1f}s")
    for k, v in i1_metrics.items():
        if k != "outcomes": print(f"  {k:35} {v}")
    print(f"  outcomes: {i1_metrics['outcomes']}")

    print("\n=== Summary ===")
    print(f"  I0 QuestionAnchorResolutionRate: {i0_metrics['QuestionAnchorResolutionRate']}")
    print(f"  I1 QuestionAnchorResolutionRate: {i1_metrics['QuestionAnchorResolutionRate']}")
    print(f"  I0 S2cLiveRate:                   {i0_metrics['S2cLiveRate']}")
    print(f"  I1 S2cLiveRate:                   {i1_metrics['S2cLiveRate']}")
    print(f"  I0 CorrectAnchorRate:             {i0_metrics['CorrectAnchorRate']}")
    print(f"  I1 CorrectAnchorRate:             {i1_metrics['CorrectAnchorRate']}")
    print(f"  I0 WrongAnchorRate:               {i0_metrics['WrongAnchorRate']}")
    print(f"  I1 WrongAnchorRate:               {i1_metrics['WrongAnchorRate']}")
    print(f"  I0 UnresolvedRate:                {i0_metrics['UnresolvedRate']}")
    print(f"  I1 UnresolvedRate:                {i1_metrics['UnresolvedRate']}")

    # Write receipt
    OUT.mkdir(parents=True)
    receipt = {
        "gate": "C3_SURFACE_IDENTITY_RESOLUTION",
        "rungs_run": ["I0", "I1"],
        "rungs_NOT_run": ["I2", "I3", "I4", "I5", "I6"],
        "protocol_version": "v2_pre_measurement_amended",
        "corpus": "controlled_gate_c3_v1",
        "partition": PARTITION,
        "vocabulary_domain": "lighthouses",
        "candidate_pool": "full_evidence_set (controlled: isolates resolution from retrieval)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "I0": {
            "description": "current surface anchor extraction (build_task_graph); the mechanism that is currently inert",
            "metrics": i0_metrics,
            "per_task": i0_results,
        },
        "I1": {
            "description": "case/punctuation/whitespace normalization only; NEGATIVE_CONTROL rung",
            "classification": "NEGATIVE_CONTROL",
            "expected_relation_to_I0": "I1 ~= I0 on truncation failures (normalization cannot recover a missing token, and the alias surface is a different name from the canonical)",
            "metrics": i1_metrics,
            "per_task": i1_results,
        },
        "resolution_utility": {
            "formula": "CorrectResolution - lambda * WrongResolution",
            "lambda": 2.0,
            "I0": round(i0_metrics["CorrectAnchorRate"] - 2.0 * i0_metrics["WrongAnchorRate"], 4),
            "I1": round(i1_metrics["CorrectAnchorRate"] - 2.0 * i1_metrics["WrongAnchorRate"], 4),
        },
    }
    receipt_path = OUT / "i0_i1_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    def h(b): return hashlib.sha256(b).hexdigest()
    (OUT / "RESULTS.sha256").write_text(
        f"{h(receipt_path.read_bytes())}  i0_i1_receipt.json\n")

    print(f"\nreceipt: {receipt_path}")
    print("frozen.")


if __name__ == "__main__":
    main()
