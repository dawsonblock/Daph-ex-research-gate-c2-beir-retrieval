#!/usr/bin/env python3
"""Gate C3 I2 runner — prefix/suffix tolerant surface resolution.

I2 is the first rung that attacks the measured truncation defect. It tests
whether a truncated or partial surface mention can resolve to a visible
candidate entity based on generic token/prefix structure — WITHOUT reading
identity-record semantics (that is I3's job).

Permitted operations (all deterministic, runtime-only, no benchmark-specific
vocabulary):
    1. Unicode/case/punctuation normalization
    2. Token-prefix compatibility (surface tokens are a prefix of entity tokens)
    3. One-sided suffix extension (entity = surface + at most 1 extra token)
    4. Head-token preservation (first token must match)
    5. Minimum token-overlap ratio (>= 50% of surface tokens in entity)
    6. Ambiguity detection (abstain if >1 equally plausible candidate remains)

Forbidden:
    - benchmark-specific role lists or lighthouse vocabulary
    - known entity suffixes or alias patterns
    - identity-record parsing (I3's job)
    - oracle metadata at runtime

Additional diagnostic for I2:
    TruncationRecoveryRate = correct resolutions / tasks where the surface
    head token appears in at least one pool entity (i.e., a prefix/suffix
    match was structurally possible). This tells us whether I2 is solving
    the mechanism it was designed for.

Promotion criteria (conservative):
    CorrectAnchorRate > I1
    FalseResolutionRate <= 0.02
    S2cLiveRate > 0
If I2 resolves many tasks but wrong-anchor rate rises materially, do not
promote it just because live rate improves.
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
EVIDENCE_DIR = ROOT / "evidence/gate_c3"
PARTITION = "c3v1_surface"

# --- I2 resolver: generic, deterministic, runtime-only ----------------------
# No benchmark-specific vocabulary, no identity-record parsing, no oracle metadata.

_MIN_OVERLAP_RATIO = 0.5  # at least 50% of surface tokens must appear in entity
_MAX_EXTENSION_TOKENS = 1  # one-sided suffix extension: entity = surface + at most 1 token

# Question templates (generic English, not benchmark-specific):
#   "Which {relation} is held by {subject}?"
#   "What {relation} does {subject} carry?"
#   "State the {relation} recorded for {subject}."
#   "Report the {relation} attached to {subject}."
_SUBJECT_PATTERNS = [
    re.compile(r"(?:is held by|is held by)\s+(.+?)\?$"),
    re.compile(r"(?:does|do)\s+(.+?)\s+(?:carry|carry)\??$"),
    re.compile(r"recorded for\s+(.+?)\.$"),
    re.compile(r"attached to\s+(.+?)\.$"),
]


def _normalize(text):
    """Unicode/case/punctuation normalization."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _extract_subject(question):
    """Extract the subject phrase from a generic question template."""
    for pat in _SUBJECT_PATTERNS:
        m = pat.search(question)
        if m:
            return m.group(1).strip().rstrip(".?")
    return None


def _extract_all_pool_entities(evidence_texts, entity_types):
    """Extract all entity mentions from the pool, keyed by normalized name.

    Returns a set of normalized entity names found across all evidence records.
    """
    all_entities = set()
    for text in evidence_texts:
        mentions = extract_mentions(text, entity_types)
        all_entities.update(mentions)
    return all_entities


def _is_compatible(surface_tokens, entity_tokens):
    """Check if a surface mention is structurally compatible with an entity name.

    Compatibility requires ALL of:
      1. Head-token preservation: surface[0] == entity[0]
      2. Token-prefix: surface tokens are a prefix of entity tokens
         (one-sided: the entity may extend the surface by at most
         _MAX_EXTENSION_TOKENS extra tokens)
      3. Minimum token-overlap ratio: at least _MIN_OVERLAP_RATIO of
         surface tokens appear in the entity

    Returns True if compatible, False otherwise.
    """
    if not surface_tokens or not entity_tokens:
        return False
    # 1. Head-token preservation
    if surface_tokens[0] != entity_tokens[0]:
        return False
    # 2. Token-prefix: surface must be a prefix of entity
    if len(entity_tokens) < len(surface_tokens):
        return False
    if entity_tokens[:len(surface_tokens)] != surface_tokens:
        return False
    # One-sided suffix extension: entity may have at most _MAX_EXTENSION_TOKENS
    # extra tokens beyond the surface
    if len(entity_tokens) - len(surface_tokens) > _MAX_EXTENSION_TOKENS:
        return False
    # 3. Minimum token-overlap ratio
    overlap = len(set(surface_tokens) & set(entity_tokens))
    ratio = overlap / len(surface_tokens)
    if ratio < _MIN_OVERLAP_RATIO:
        return False
    return True


def _resolve_i2(subject_raw, pool_entities_norm):
    """Run the I2 resolver on a single task.

    Returns (found_candidates, outcome) where outcome is one of
    EXACT/RESOLVED/AMBIGUOUS/UNRESOLVED.

    Note: I2 does NOT read identity records. "RESOLVED" here means the resolver
    found a unique compatible entity by prefix/suffix structure. Correctness
    (whether that entity is the TRUE canonical) is evaluated separately.
    """
    if not subject_raw:
        return set(), "UNRESOLVED"

    surface_norm = _normalize(subject_raw)
    surface_tokens = surface_norm.split()
    if not surface_tokens:
        return set(), "UNRESOLVED"

    # Check for exact match first
    if surface_norm in pool_entities_norm:
        return {surface_norm}, "EXACT"

    # Find all compatible entities by prefix/suffix structure
    candidates = set()
    for entity in pool_entities_norm:
        entity_tokens = entity.split()
        if _is_compatible(surface_tokens, entity_tokens):
            candidates.add(entity)

    if not candidates:
        return set(), "UNRESOLVED"
    if len(candidates) == 1:
        return candidates, "RESOLVED"
    return candidates, "AMBIGUOUS"


def _load():
    tasks = [json.loads(l) for l in (CORPUS / PARTITION / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (CORPUS / PARTITION / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def _make_candidates(evidence):
    return [{"document_id": r["evidence_id"], "metadata": r["metadata"]} for r in evidence]


def _texts_from_evidence(evidence):
    return {r["evidence_id"]: r["content"] for r in evidence}


def run_i2(tasks, evidence):
    """Run the I2 resolver over all tasks."""
    texts = _texts_from_evidence(evidence)
    cands = _make_candidates(evidence)
    corpus_texts = [texts.get(c["document_id"], "") for c in cands]

    # Derive entity types from the pool (same as I0/I1)
    entity_types = derive_entity_types(corpus_texts)

    # Extract ALL entity mentions from the pool
    pool_entities_norm = _extract_all_pool_entities(corpus_texts, entity_types)

    results = []
    for t in tasks:
        meta = t["_oracle_metadata"]
        true_canonical = _norm(meta["surfaces"]["canonical"])
        true_surface = _norm(meta["surfaces"]["subject"])
        question = t["question"]
        regime = t["metadata"]["entity_regime"]

        # Parse the subject from the question
        subject_raw = _extract_subject(question)
        surface_norm = _normalize(subject_raw) if subject_raw else ""

        # Run I2 resolver
        found, outcome = _resolve_i2(subject_raw, pool_entities_norm)

        # Correctness evaluation (evaluator-side, using oracle metadata)
        correct = False
        wrong = False
        if outcome == "EXACT" and true_canonical in found:
            correct = True
        elif outcome == "RESOLVED":
            resolved_entity = next(iter(found))
            if resolved_entity == true_canonical:
                correct = True
            else:
                wrong = True
        elif outcome == "AMBIGUOUS":
            # Ambiguity is not wrong, it's abstention
            pass

        # TruncationRecoveryRate diagnostic:
        # Was a prefix/suffix match structurally possible?
        # (surface head token appears in at least one pool entity)
        surface_tokens = surface_norm.split()
        head_in_pool = any(
            surface_tokens and e.split()[0] == surface_tokens[0]
            for e in pool_entities_norm if e.split()
        ) if surface_tokens else False

        # Chain enumeration and S2c liveness
        # I2 changes the anchor; if the anchor resolves, we need to check
        # whether chains can now be enumerated. We inject the resolved entity
        # into the question_entities and rebuild the graph.
        graph = build_task_graph(cands, question, texts)
        if found and outcome in ("EXACT", "RESOLVED"):
            # Inject the found entity as a question entity for chain enumeration
            graph.question_entities = found
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
            "task_id": t["task_id"], "arm": "I2", "outcome": outcome,
            "found_entities": sorted(found), "true_canonical": true_canonical,
            "subject_raw": subject_raw, "surface_norm": surface_norm,
            "regime": regime,
            "correct": correct, "wrong": wrong,
            "head_in_pool": head_in_pool,
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

    # TruncationRecoveryRate: correct / tasks where prefix match was possible
    truncation_eligible = [r for r in results if r["head_in_pool"]]
    truncation_correct = sum(1 for r in truncation_eligible if r["correct"])
    truncation_rate = round(truncation_correct / len(truncation_eligible), 4) if truncation_eligible else 0.0

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
            }

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
        "TruncationRecoveryRate": truncation_rate,
        "truncation_eligible_tasks": len(truncation_eligible),
        "truncation_correct": truncation_correct,
        "outcomes": dict(outcomes),
        "by_regime": by_regime,
    }


def main():
    tasks, evidence = _load()
    print(f"Loaded {len(tasks)} tasks, {len(evidence)} evidence records from {PARTITION}")

    # Load prior I0/I1 results for the cascade table
    prior_receipt_path = EVIDENCE_DIR / "i0_i1_receipt.json"
    if not prior_receipt_path.exists():
        print("WARNING: I0/I1 receipt not found; cascade table will show I2 only.")
        prior = {}
    else:
        prior = json.loads(prior_receipt_path.read_text())

    print("\n=== I2: prefix/suffix tolerant resolution ===")
    t2 = time.time()
    i2_results = run_i2(tasks, evidence)
    i2_metrics = compute_metrics(i2_results)
    print(f"  completed in {time.time()-t2:.1f}s")
    for k, v in i2_metrics.items():
        if k not in ("outcomes", "by_regime"): print(f"  {k:35} {v}")
    print(f"  outcomes: {i2_metrics['outcomes']}")
    print(f"  by_regime: {json.dumps(i2_metrics['by_regime'], indent=4)}")

    # Cascade table
    print("\n=== Cascade Table (I0 / I1 / I2) ===")
    headers = ["Arm", "QAnchorRate", "CorrectAnchor", "WrongAnchor", "FalseRes", "Ambiguous", "Unresolved", "S2cLive", "TruncRecovery"]
    rows = []
    if prior:
        for arm_key, arm_label in [("I0", "I0"), ("I1", "I1")]:
            m = prior[arm_key]["metrics"]
            rows.append([arm_label, m["QuestionAnchorResolutionRate"], m["CorrectAnchorRate"],
                        m["WrongAnchorRate"], m["FalseResolutionRate"],
                        m["AmbiguousResolutionRate"], m["UnresolvedRate"],
                        m["S2cLiveRate"], "N/A"])
    rows.append(["I2", i2_metrics["QuestionAnchorResolutionRate"], i2_metrics["CorrectAnchorRate"],
                i2_metrics["WrongAnchorRate"], i2_metrics["FalseResolutionRate"],
                i2_metrics["AmbiguousResolutionRate"], i2_metrics["UnresolvedRate"],
                i2_metrics["S2cLiveRate"], i2_metrics["TruncationRecoveryRate"]])

    fmt = "{:<6} {:>12} {:>14} {:>13} {:>10} {:>10} {:>12} {:>9} {:>15}"
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))

    # Promotion evaluation
    print("\n=== Promotion Evaluation ===")
    i1_correct = prior["I1"]["metrics"]["CorrectAnchorRate"] if prior else 0.0
    i1_false = prior["I1"]["metrics"]["FalseResolutionRate"] if prior else 0.0
    i1_s2c = prior["I1"]["metrics"]["S2cLiveRate"] if prior else 0.0

    checks = {
        "CorrectAnchorRate > I1": i2_metrics["CorrectAnchorRate"] > i1_correct,
        "FalseResolutionRate <= 0.02": i2_metrics["FalseResolutionRate"] <= 0.02,
        "S2cLiveRate > 0": i2_metrics["S2cLiveRate"] > 0,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {check}")
    all_pass = all(checks.values())
    print(f"\n  Overall: {'PROMOTED' if all_pass else 'NOT_PROMOTED'}")

    # Decision
    if i2_metrics["CorrectAnchorRate"] > i1_correct and i2_metrics["FalseResolutionRate"] <= 0.02:
        decision = "I2 materially recovers truncated mentions with very low false resolution. Keep as surface-normalization layer and proceed to I3 for arbitrary alias cases."
    elif i2_metrics["CorrectAnchorRate"] <= i1_correct and i2_metrics["FalseResolutionRate"] > 0.02:
        decision = "I2 does NOT improve correct resolution and produces too many false resolutions. The surface failures are NOT mostly textual truncation — the alias surface has a different head from the canonical, so prefix/suffix matching finds wrong entities. Move directly to I3 identity-record resolution."
    elif i2_metrics["CorrectAnchorRate"] <= i1_correct and i2_metrics["WrongAnchorRate"] <= 0.02:
        decision = "I2 remains near zero with low false resolution. Surface failures are not mostly textual truncation. Move directly to I3 identity-record resolution."
    elif i2_metrics["S2cLiveRate"] > 0 and i2_metrics["CorrectAnchorRate"] <= i1_correct:
        decision = "I2 improves S2cLiveRate but not canonical accuracy. Stop and diagnose ambiguity before I3."
    else:
        decision = "I2 results require manual interpretation before proceeding."
    print(f"  Decision: {decision}")

    # Write receipt (append to existing I0/I1 receipt)
    receipt = prior if prior else {}
    receipt["rungs_run"] = receipt.get("rungs_run", []) + ["I2"] if "I2" not in receipt.get("rungs_run", []) else receipt["rungs_run"]
    receipt["rungs_NOT_run"] = [r for r in receipt.get("rungs_NOT_run", ["I2","I3","I4","I5","I6"]) if r != "I2"]
    receipt["I2"] = {
        "description": "prefix/suffix tolerant resolution; deterministic, runtime-only, no identity-record parsing",
        "permitted_operations": [
            "unicode/case/punctuation normalization",
            "token-prefix compatibility",
            "one-sided suffix extension (max 1 extra token)",
            "head-token preservation",
            "minimum token-overlap ratio (>= 0.5)",
            "ambiguity detection (abstain if >1 candidate)",
        ],
        "forbidden": [
            "benchmark-specific role lists or lighthouse vocabulary",
            "known entity suffixes or alias patterns",
            "identity-record parsing (I3's job)",
            "oracle metadata at runtime",
        ],
        "parameters": {
            "min_overlap_ratio": _MIN_OVERLAP_RATIO,
            "max_extension_tokens": _MAX_EXTENSION_TOKENS,
        },
        "metrics": i2_metrics,
        "per_task": i2_results,
    }
    receipt["promotion_evaluation"] = {
        "criteria": checks,
        "all_passed": all_pass,
        "decision": decision,
    }

    receipt_path = EVIDENCE_DIR / "i0_i1_i2_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    # Update SHA manifest
    def h(b): return hashlib.sha256(b).hexdigest()
    (EVIDENCE_DIR / "RESULTS.sha256").write_text(
        f"{h(receipt_path.read_bytes())}  i0_i1_i2_receipt.json\n")

    print(f"\nreceipt: {receipt_path}")
    print("frozen.")


if __name__ == "__main__":
    main()
