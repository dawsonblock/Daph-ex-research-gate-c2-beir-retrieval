#!/usr/bin/env python3
"""Build the FRESH confirmation split for c5_integrated_v1. Fail-closed.

Why this exists
---------------
The 500-task qualification split is CONSUMED: it informed the retrieval
diagnosis, the selector diagnosis, the failure decomposition and the design of
S2 itself. It cannot also be the test of the mechanism it shaped. J1
(frozen_rrf + S2) met all 12 frozen criteria on development, and a development
pass is not a generalization claim -- v2.1 also passed development (+0.200)
and then failed qualification (+0.091). Confirmation needs untouched data.

Design constraints, from the protocol
-------------------------------------
  * >= 500 tasks
  * preserve family proportions, identity regimes, bridged/unbridged structure
    and temporal cases
  * reproduce the CORPUS SCALING CHALLENGE -- v2.1 failed because a fixed k=50
    met a 4.15x larger corpus. A confirmation split at development's scale
    would dodge the very condition that broke the mechanism, so this one is
    generated at qualification's scale (50 tasks/family -> ~2216 records).
  * frozen before any result is seen, run exactly once

Additive by construction
------------------------
build_gate_a_v4_dataset.py refuses to run against an existing dataset root,
and re-running it would regenerate development/qualification from scratch --
whose bytes the certified bundles hash. So this script writes ONLY a new
`confirmation/` subdirectory and never touches an existing file. Nothing at
runtime hashes the dataset root recursively (sha256_corpus digests individual
corpus files), so an added directory leaves every existing hash and every
certified bundle intact. That invariant is asserted before writing.

Audits are REUSED, not reimplemented
------------------------------------
The existing audit() is called with the three frozen splits loaded from disk
PLUS confirmation. Confirmation therefore passes every per-split audit
(inferability, retrievability, leakage, oracle isolation) under the same code
that gated the original dataset, and the frozen splits are re-verified as a
side effect. A private copy of those checks would be free to drift.

Two checks are ADDED, because the original audit could not have anticipated a
fourth split:
  * cross-split overlap: confirmation must share no task_id or evidence_id
    with any frozen split
  * confirmation-specific structural diversity: the original audit only
    checked qualification's

Usage:
    python scripts/build_c5_confirmation_split.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# PRE-EXISTING circular import, not introduced here: importing
# hrm_adaptive_memory.experiments.* first triggers experiments/__init__ ->
# context_study -> evaluation/__init__ -> context_gate -> back into the
# partially initialized context_study. Entering from the evaluation side
# resolves it. The test suite never hits this because pytest happens to import
# in an order that already broke the cycle, which is why it has stayed latent.
# Worth fixing in the package itself; deliberately not done here, since
# reshuffling those __init__ files would ripple far beyond this script.
import hrm_adaptive_memory.evaluation  # noqa: E402,F401  (cycle-breaking import)

from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (  # noqa: E402
    build_v4_corpus)
from scripts.build_gate_a_v4_dataset import (  # noqa: E402
    ID_REGIMES, ID_STYLES, SPLITS, audit)

DATASET = ROOT / "data/hrm/controlled_gate_a_v4"
SPLIT_NAME = "confirmation"

#: Distinct from development (9101), qualification (9102) and ood (9103).
#: Asserted distinct at runtime rather than trusted.
CONFIRMATION_SEED = 9104

#: Final size: 50 tasks/family x 10 families = 500, matching qualification.
TASKS_PER_FAMILY = 50

#: Generated per family BEFORE collision filtering. The generator draws entity
#: names from a finite vocabulary, so a 4th split inevitably collides with the
#: 870 already-frozen tasks: across seeds 9104-9123 every seed produced 1-6
#: duplicate question+answer pairs and 23-48 same-question collisions, so no
#: seed choice yields a clean split. Picking the least-colliding seed would be
#: optimizing a number over 20 tries rather than fixing the problem. Instead we
#: over-generate, DROP every colliding task, and trim back to exactly
#: TASKS_PER_FAMILY -- which does yield zero question and zero QA-pair overlap.
OVERGENERATE_PER_FAMILY = 70

#: Minimums applied to confirmation itself, matching what the original audit
#: applies to qualification.
STRUCTURAL_MINIMUMS = (("families", 8), ("iterative_families", 5),
                       ("templates", 40), ("source_clusters", 20),
                       ("opportunity_groups", 4), ("answer_kinds", 4))


def load_frozen_split(name: str) -> tuple[list[dict], list[dict]]:
    base = DATASET / name
    tasks = [json.loads(line) for line
             in (base / "oracle_tasks.jsonl").read_text().splitlines() if line.strip()]
    evidence = [json.loads(line) for line
                in (base / "evidence.jsonl").read_text().splitlines() if line.strip()]
    return tasks, evidence


def check_seed_is_unused() -> list[str]:
    """A reused seed would regenerate an existing split's content verbatim."""
    used = {name: config["seed"] for name, config in SPLITS.items()}
    if CONFIRMATION_SEED in used.values():
        clash = [n for n, s in used.items() if s == CONFIRMATION_SEED]
        return [f"seed {CONFIRMATION_SEED} is already used by {clash}; "
                f"confirmation would duplicate that split's content"]
    return []


def filter_and_trim(tasks: list[dict], evidence: list[dict],
                    frozen: dict[str, tuple[list[dict], list[dict]]],
                    ) -> tuple[list[dict], list[dict], dict]:
    """Drop tasks colliding with any frozen split, then trim to exact size.

    A task is dropped if its question already appears in a frozen split, or if
    any of its REQUIRED evidence records reproduce frozen evidence text.
    Required records are what the mechanism must retrieve and select, so a
    collision there is the one that would make confirmation non-independent;
    an incidental distractor sharing a sentence is measured and reported but
    does not disqualify a task.

    Trimming keeps the FIRST TASKS_PER_FAMILY survivors per family in
    generation order -- deterministic, and independent of any measurement.
    """
    frozen_questions = set().union(
        *[{t["question"] for t in ts} for ts, _ in frozen.values()])
    frozen_contents = set().union(
        *[{e["content"] for e in es} for _, es in frozen.values()])

    evidence_by_task: dict[str, list[dict]] = {}
    for record in evidence:
        evidence_by_task.setdefault(
            record["evidence_id"].split("/", 1)[0], []).append(record)

    dropped = {"question_collision": 0, "required_evidence_collision": 0}
    survivors: list[dict] = []
    for task in tasks:
        if task["question"] in frozen_questions:
            dropped["question_collision"] += 1
            continue
        required = set(task["required_evidence_ids"])
        own = evidence_by_task.get(task["task_id"], [])
        if any(r["content"] in frozen_contents
               for r in own if r["evidence_id"] in required):
            dropped["required_evidence_collision"] += 1
            continue
        survivors.append(task)

    per_family: dict[str, list[dict]] = {}
    for task in survivors:
        per_family.setdefault(task["family"], []).append(task)
    short = {f: len(v) for f, v in per_family.items() if len(v) < TASKS_PER_FAMILY}
    kept: list[dict] = []
    for family in sorted(per_family):
        kept.extend(per_family[family][:TASKS_PER_FAMILY])

    kept_ids = {t["task_id"] for t in kept}
    kept_evidence = [r for r in evidence
                     if r["evidence_id"].split("/", 1)[0] in kept_ids]

    stats = {
        "generated_tasks": len(tasks),
        "dropped": dropped,
        "survivors": len(survivors),
        "kept_tasks": len(kept),
        "kept_evidence": len(kept_evidence),
        "families_short_of_target": short,
    }
    return kept, kept_evidence, stats


def check_cross_split_isolation(confirmation: tuple[list[dict], list[dict]],
                               frozen: dict[str, tuple[list[dict], list[dict]]]
                               ) -> tuple[dict, list[str]]:
    """Confirmation must not reuse CONTENT from a frozen split.

    A first version of this checked task_id / evidence_id overlap and failed
    loudly -- correctly, because it was testing the wrong invariant. Ids here
    are positional labels of the form ``{family}-{ordinal}``, identical across
    every split by construction: development and qualification already share
    all 120 of development's ids. What the seed varies is CONTENT, e.g.
    configuration_chain-0000 asks about "Ibis relay unit" in development and
    "Raven relay unit" in qualification.

    So the isolation that matters is content overlap: questions, answers and
    evidence text. Style/regime overlap is deliberately NOT flagged --
    confirmation shares qualification's styles and regimes on purpose, since
    that is what makes it a like-for-like generalization test rather than a
    distribution shift.
    """
    problems: list[str] = []
    c_tasks, c_evidence = confirmation
    c_questions = {t["question"] for t in c_tasks}
    c_contents = {e["content"] for e in c_evidence}
    # Question+answer pairs: an identical question with a different answer is a
    # contradiction, and an identical pair is reused data.
    c_pairs = {(t["question"], t["answer"]) for t in c_tasks}

    detail: dict[str, dict[str, int]] = {}
    for name, (tasks, evidence) in frozen.items():
        q_overlap = c_questions & {t["question"] for t in tasks}
        pair_overlap = c_pairs & {(t["question"], t["answer"]) for t in tasks}
        content_overlap = c_contents & {e["content"] for e in evidence}
        detail[name] = {
            "shared_questions": len(q_overlap),
            "shared_question_answer_pairs": len(pair_overlap),
            "shared_evidence_content": len(content_overlap),
        }
        if q_overlap:
            problems.append(
                f"content isolation: confirmation shares {len(q_overlap)} "
                f"question(s) with {name} (e.g. {sorted(q_overlap)[:2]})")
        if content_overlap:
            problems.append(
                f"content isolation: confirmation shares "
                f"{len(content_overlap)} evidence record(s) with {name} "
                f"(e.g. {sorted(content_overlap)[:1]})")
    return detail, problems


def check_structural_diversity(tasks: list[dict]) -> tuple[dict, list[str]]:
    structural = {
        "families": len({t["family"] for t in tasks}),
        "iterative_families": len({t["family"] for t in tasks
                                   if t["metadata"]["iterative_family"]}),
        "templates": len({t["template_id"] for t in tasks}),
        "source_clusters": len({t["source_cluster_id"] for t in tasks}),
        "opportunity_groups": len({t["metadata"]["opportunity_group"] for t in tasks}),
        "answer_kinds": len({t["metadata"]["answer_kind"] for t in tasks}),
    }
    problems = [f"confirmation structural: {name}={structural[name]} < {minimum}"
                for name, minimum in STRUCTURAL_MINIMUMS
                if structural[name] < minimum]
    return structural, problems


def check_scale_matches_qualification(c_tasks, c_evidence, frozen) -> tuple[dict, list[str]]:
    """The whole point is to stress fixed-k against a large corpus again."""
    q_tasks, q_evidence = frozen["qualification"]
    d_tasks, d_evidence = frozen["development"]
    scale = {
        "confirmation_tasks": len(c_tasks),
        "confirmation_evidence": len(c_evidence),
        "qualification_evidence": len(q_evidence),
        "development_evidence": len(d_evidence),
        "confirmation_over_development": round(len(c_evidence) / len(d_evidence), 3),
    }
    problems = []
    if len(c_tasks) < 500:
        problems.append(f"scale: {len(c_tasks)} tasks < 500 required")
    # Within 10% of qualification's corpus size: close enough that fixed k=50
    # is under comparable pressure, without demanding an exact match.
    if abs(len(c_evidence) - len(q_evidence)) / len(q_evidence) > 0.10:
        problems.append(
            f"scale: confirmation corpus {len(c_evidence)} records is not "
            f"within 10% of qualification's {len(q_evidence)}, so fixed-k "
            f"pressure would not be comparable")
    return scale, problems


def check_no_existing_file_would_change() -> list[str]:
    """Assert the write is purely additive before writing anything."""
    target = DATASET / SPLIT_NAME
    if target.exists():
        return [f"refusing to overwrite an existing split: {target}"]
    if not DATASET.exists():
        return [f"dataset root missing: {DATASET}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the fresh confirmation split (fail-closed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="audit only; write nothing")
    args = parser.parse_args()

    print("=== building fresh confirmation split for c5_integrated_v1 ===")
    print(f"  seed={CONFIRMATION_SEED}  tasks_per_family={TASKS_PER_FAMILY}")
    print(f"  styles={list(ID_STYLES)}")
    print(f"  regimes={list(ID_REGIMES)}\n")

    problems: list[str] = []
    problems += check_seed_is_unused()
    problems += check_no_existing_file_would_change()
    if problems:
        for line in problems:
            print(f"  FAIL {line}")
        print("\nABORT: preconditions failed. Nothing written.")
        return 1

    print("[1/5] loading the frozen splits (for reuse of the original audit)")
    frozen = {name: load_frozen_split(name) for name in SPLITS}
    for name, (tasks, evidence) in frozen.items():
        print(f"      {name}: {len(tasks)} tasks, {len(evidence)} records")

    print("\n[2/5] generating confirmation in memory only")
    corpus = build_v4_corpus(split=SPLIT_NAME, seed=CONFIRMATION_SEED,
                            tasks_per_family=OVERGENERATE_PER_FAMILY,
                            styles=ID_STYLES, regimes=ID_REGIMES)
    raw_tasks, raw_evidence = list(corpus.tasks), list(corpus.evidence)
    print(f"      over-generated: {len(raw_tasks)} tasks, {len(raw_evidence)} records")
    c_tasks, c_evidence, filter_stats = filter_and_trim(
        raw_tasks, raw_evidence, frozen)
    print(f"      dropped for collisions: {filter_stats['dropped']}")
    print(f"      confirmation: {len(c_tasks)} tasks, {len(c_evidence)} records")
    if filter_stats["families_short_of_target"]:
        problems.append(
            f"filtering: families below {TASKS_PER_FAMILY} after collision "
            f"removal: {filter_stats['families_short_of_target']}")

    print("\n[3/5] auditing (original audit, reused, over all four splits)")
    corpora = {**frozen, SPLIT_NAME: (c_tasks, c_evidence)}
    report = audit(corpora)
    problems += list(report["problems"])

    structural, structural_problems = check_structural_diversity(c_tasks)
    problems += structural_problems
    print(f"      confirmation structural diversity: {structural}")

    isolation_detail, isolation_problems = check_cross_split_isolation(
        (c_tasks, c_evidence), frozen)
    problems += isolation_problems
    print(f"      cross-split CONTENT isolation: "
          f"{'OK' if not isolation_problems else 'FAIL'}  {isolation_detail}")

    scale, scale_problems = check_scale_matches_qualification(
        c_tasks, c_evidence, frozen)
    problems += scale_problems
    print(f"      scale: {scale}")

    for line in problems:
        print(f"      FAIL {line}")
    if problems:
        print("\nABORT: audits failed. Nothing written.")
        return 1
    print("      ALL AUDITS PASS")

    if args.dry_run:
        print("\n[dry run] audits pass; nothing written by request.")
        return 0

    print("\n[4/5] writing (additive: only the new confirmation/ directory)")
    target = DATASET / SPLIT_NAME
    target.mkdir(parents=False)
    (target / "oracle_tasks.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in c_tasks))
    (target / "evidence.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in c_evidence))
    (target / "dataset_manifest.json").write_text(
        json.dumps(dict(corpus.manifest), sort_keys=True, indent=2) + "\n")

    print("[5/5] hashing and freezing")
    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(target.iterdir()) if path.is_file()}
    (target / "CONFIRMATION_AUDIT.json").write_text(json.dumps({
        "generator": "controlled-gate-a-v4",
        "split": SPLIT_NAME,
        "purpose": (
            "Fresh one-shot confirmation split for c5_integrated_v1. The "
            "500-task qualification split is CONSUMED -- it informed the "
            "retrieval diagnosis, selector diagnosis, failure decomposition "
            "and the design of S2 -- so it cannot test the mechanism it "
            "shaped."),
        "seed": CONFIRMATION_SEED,
        "tasks_per_family": TASKS_PER_FAMILY,
        "styles": list(ID_STYLES), "regimes": list(ID_REGIMES),
        "frozen_before_evaluation": True,
        "scale_rationale": (
            "Generated at qualification's corpus scale on purpose. v2.1 failed "
            "because a fixed k=50 met a 4.15x larger corpus; a confirmation "
            "split at development's scale would avoid the exact condition "
            "that broke the mechanism."),
        "scale": scale,
        "collision_filtering": filter_stats,
        "structural_diversity": structural,
        "audit": {k: v for k, v in report.items() if k != "problems"},
        "cross_split_content_isolation": {
            "detail": isolation_detail,
            "what_is_checked": (
                "questions, question+answer pairs and evidence record text "
                "against development, qualification and ood. NOT ids: ids are "
                "positional labels ({family}-{ordinal}) that every split "
                "shares by construction, so id overlap is meaningless here. "
                "NOT styles or regimes either: confirmation shares "
                "qualification's on purpose, which is what makes this a "
                "like-for-like generalization test rather than a "
                "distribution shift."),
        },
        "additive_write": (
            "Only data/hrm/controlled_gate_a_v4/confirmation/ was created. No "
            "existing file was read-modified-written, so every prior corpus "
            "hash and every certified bundle remains valid."),
        "run_exactly_once": (
            "Frozen criteria are already fixed in "
            "configs/gate_c5_integrated_v1.json. This split is run ONCE with "
            "no tuning; a failure is recorded as a negative result, not "
            "iterated against."),
        "digests": digests,
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {target}")
    for name, digest in digests.items():
        print(f"        {digest[:16]}  {name}")
    print("\nCONFIRMATION SPLIT FROZEN. Run it exactly once, with no tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
