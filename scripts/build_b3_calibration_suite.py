#!/usr/bin/env python3
"""Build and freeze the B3 multi-scale calibration suite. Fail-closed.

Dataset generation ONLY. No rho, no candidate-budget policy, no S2, no HRM.
Calibration data creation must not depend on the policy it will later evaluate,
so the policy fields appear here as explicit nulls.

Design: corpus size is the treatment variable
---------------------------------------------
Each scale holds a FIXED task count (15 per family x 10 families = 150) and
reaches its target corpus size by adding independent NON-REQUIRED evidence.
That isolates the thing under study:

    same logical task difficulty  +  increasing retrieval search space

rather than confounding it with larger tasks or deeper proofs. Task INSTANCES
differ per scale (independent seeds) to avoid the statistical dependencies of
repeating the same instances, while the required-set complexity distributions
are audited for tight agreement across scales.

Why the smallest scale is ~700 and not 500
------------------------------------------
The generator couples evidence volume to task count at a measured ~4.41
records/task, so a fixed 150-task structure has a floor near 661 records. The
requested 500-record scale is therefore unreachable without dropping to ~110
tasks, which would both break the fixed-task-count design and give the least
stable retrieval curves at exactly the scale where they matter most. Holding
tasks fixed was judged the more important property, so the ladder starts at 700.
This deviation is recorded rather than silently applied.

Filler evidence
---------------
Drawn from the NON-REQUIRED records (dead ends, near-duplicates, rejected
candidates) of a separate large generator run with its own seed, so distractors
are structurally realistic rather than synthetic noise. The filler pool is
partitioned DISJOINTLY across scales and screened against every consumed split,
so no scale shares filler with another or with prior evaluation data.

Usage:
    python scripts/build_b3_calibration_suite.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Pre-existing experiments <-> evaluation circular import; enter from the
# evaluation side. Documented in build_c5_confirmation_split.py.
import hrm_adaptive_memory.evaluation  # noqa: E402,F401

from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (  # noqa: E402
    build_v4_corpus, verify_inferable)
from scripts.build_gate_a_v4_dataset import (  # noqa: E402
    ID_REGIMES, ID_STYLES, RUNTIME_FORBIDDEN, _norm)
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (  # noqa: E402
    _ANSWER_BEARING_KINDS, _leaks)

OUT = ROOT / "data/hrm/b3_calibration_v1"
DATASET_V4 = ROOT / "data/hrm/controlled_gate_a_v4"

#: Fixed at every scale: corpus size is the treatment, task structure is not.
TASKS_PER_FAMILY = 15

#: Generated per family BEFORE collision filtering. The generator draws entity
#: names from a finite vocabulary, so a new split inevitably reuses questions
#: already present in the consumed splits -- the same problem the confirmation
#: split hit. Over-generate, drop colliding tasks, trim to TASKS_PER_FAMILY.
#: Raised from 40: each successive scale must avoid the consumed splits AND
#: every earlier scale, so the survivor pool shrinks as the suite is built. At
#: 40 the fourth scale ran short on temporal_chain/temporal_update, the families
#: with the tightest vocabulary. 100 leaves margin for all five.
OVERGENERATE_PER_FAMILY = 100

#: (split_id, target evidence records, task seed). Seeds are distinct from the
#: v4 dataset's 9101/9102/9103 and the confirmation split's 9104.
SCALES: tuple[tuple[str, int, int], ...] = (
    ("cal_700", 700, 9201),
    ("cal_1000", 1000, 9202),
    ("cal_1500", 1500, 9203),
    ("cal_2200", 2200, 9204),
    ("cal_3000", 3000, 9205),
)

FILLER_SEED = 9299
#: Sized so the DISJOINT partition across all five scales fits without reuse.
#: Total filler demand is sum(target - base) = 5,095 records; the pool yields
#: roughly 2.36 usable non-required records per generated task, so 150/family
#: (3,575 usable) was insufficient. Filler is never shared between scales --
#: that would manufacture exactly the cross-scale overlap this suite audits
#: against -- so the source is enlarged instead.
#: Raised to 500 after measuring the real screening cost. Answer-leak screening
#: is expensive at the large scales: cal_3000 has 150 answers, many of them short
#: numeric or single-word values, so roughly 60% of remaining filler legitimately
#: contains one and must be rejected. Total filler demand is ~5,065 records, and
#: a 300/family pool (7,179 usable) could not satisfy the last scale after
#: screening. Enlarging the source is the honest fix; loosening the leak screen
#: would put answer text into the distractors.
FILLER_TASKS_PER_FAMILY = 500

#: Frozen prospectively, before any count is observed.
SCALE_TOLERANCE = 0.05

#: Max per-category proportion difference between a scale's structural
#: distributions and the reference scale's. Exact equality is unachievable:
#: collision filtering removes different tasks at each scale, so some residual
#: is unavoidable. This reuses the project's existing 0.05 subgroup-tolerance
#: convention rather than a value chosen to fit the observed data -- after
#: stratified selection the actual residual is about 2pp, comfortably inside it,
#: and every scale's measured residual is recorded in its manifest so a reader
#: can see how tight the match really is rather than trusting the bound.
COMPOSITION_TOLERANCE = 0.05

#: Recorded so later analysis cannot conflate corpus generation with the B3
#: intervention. Deliberately unset here.
BUDGET_METADATA = {
    "retriever_search_depth": None,
    "candidate_pool_budget": None,
    "packet_budget": 6,
    "rho": None, "k_min": None, "k_max": None,
    "note": ("Budget concepts are recorded UNSET. Calibration data creation "
             "must not depend on the policy it will later evaluate."),
}

CONSUMED_SPLITS = ("development", "qualification", "ood", "confirmation")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_consumed() -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Normalized questions, evidence contents and (question, answer) pairs."""
    questions: set[str] = set()
    contents: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for name in CONSUMED_SPLITS:
        base = DATASET_V4 / name
        if not (base / "oracle_tasks.jsonl").is_file():
            continue
        for line in (base / "oracle_tasks.jsonl").read_text().splitlines():
            if line.strip():
                task = json.loads(line)
                questions.add(_norm(task["question"]))
                pairs.add((_norm(task["question"]), _norm(task["answer"])))
        for line in (base / "evidence.jsonl").read_text().splitlines():
            if line.strip():
                contents.add(_norm(json.loads(line)["content"]))
    return questions, contents, pairs


def _composition_key(task: dict) -> tuple:
    """The structural signature that must stay matched across scales."""
    oracle = task.get("_oracle_metadata") or {}
    return (
        len(oracle.get("proof_edges") or []),
        len(task["required_evidence_ids"]),
        task["metadata"]["answer_kind"],
        bool(oracle.get("latent_bridge")),
    )


def composition_by_family(tasks: list[dict]) -> dict[str, dict[tuple, int]]:
    """Per-family multiset of composition keys, used as the match target."""
    out: dict[str, Counter] = {}
    for task in tasks:
        out.setdefault(task["family"], Counter())[_composition_key(task)] += 1
    return {family: dict(counter) for family, counter in out.items()}


def structural_profile(tasks: list[dict]) -> dict[str, Any]:
    """The distributions that must stay matched across scales."""
    def depth(task: dict) -> int:
        return len((task.get("_oracle_metadata") or {}).get("proof_edges") or [])
    return {
        "task_count": len(tasks),
        "family_counts": dict(sorted(Counter(t["family"] for t in tasks).items())),
        "entity_regime_counts": dict(sorted(Counter(
            t["metadata"]["entity_regime"] for t in tasks).items())),
        "bridged_counts": dict(sorted(Counter(
            "bridged" if (t.get("_oracle_metadata") or {}).get("latent_bridge")
            else "unbridged" for t in tasks).items())),
        "answer_kind_counts": dict(sorted(Counter(
            t["metadata"]["answer_kind"] for t in tasks).items())),
        "proof_depth_counts": dict(sorted(Counter(depth(t) for t in tasks).items())),
        "source_cluster_counts": len({t["source_cluster_id"] for t in tasks}),
        "required_set_size_counts": dict(sorted(Counter(
            len(t["required_evidence_ids"]) for t in tasks).items())),
    }


def build_filler_pool(consumed_contents: set[str]) -> list[dict]:
    """Non-required records from an independent run, screened and deterministic.

    Filler evidence_ids are RE-NAMESPACED. The generator names records
    {family}-{ordinal}/{suffix} without reference to the split, and the filler
    run uses a much larger ordinal range than a 15/family scale, so its low
    ordinals collide with every scale's own base ids. That collision showed up
    as 132 duplicate evidence_ids in the first build. Prefixing removes it by
    construction rather than by filtering.
    """
    corpus = build_v4_corpus(split="b3_filler", seed=FILLER_SEED,
                             tasks_per_family=FILLER_TASKS_PER_FAMILY,
                             styles=ID_STYLES, regimes=ID_REGIMES)
    required: set[str] = set()
    for task in corpus.tasks:
        required |= set(task["required_evidence_ids"])
    pool = []
    for record in corpus.evidence:
        if record["evidence_id"] in required:
            continue
        if _norm(record["content"]) in consumed_contents:
            continue
        row = dict(record)
        row["evidence_id"] = f"b3filler/{record['evidence_id']}"
        metadata = dict(row.get("metadata") or {})
        metadata["b3_filler"] = True
        row["metadata"] = metadata
        pool.append(row)
    # Deterministic order: sorted by id, so partitioning replays identically.
    return sorted(pool, key=lambda r: r["evidence_id"])


def take_filler(pool: list[dict], need: int, base_contents: set[str],
                answers: set[str], used_ids: set[str],
                used_contents: set[str]) -> list[dict]:
    """Take `need` filler records that are safe for THIS scale.

    Three screens, each for a hazard the first build actually exhibited:
      * content already in this scale's base evidence -- would be a duplicate;
      * content already used by an earlier scale -- would be cross-scale overlap;
      * content containing one of this scale's ANSWERS -- foreign distractors can
        leak an answer by coincidence, which showed up as 5 distractor leaks.

    Scans the whole pool each time, skipping records ALREADY USED rather than
    advancing a shared cursor. A monotonic cursor was the first implementation
    and it exhausted the pool: rejection is scale-SPECIFIC (a record leaking
    scale A's answer is usually fine for scale B), so a global cursor discarded
    perfectly usable records permanently. Reuse of a record across scales is
    still forbidden -- that is what used_ids enforces -- but reconsideration of
    a merely-skipped record is not.
    """
    taken: list[dict] = []
    for record in pool:
        if len(taken) >= need:
            break
        if record["evidence_id"] in used_ids:
            continue
        normalized = _norm(record["content"])
        if normalized in base_contents or normalized in used_contents:
            continue
        if any(_leaks(record["content"], answer) for answer in answers):
            continue
        taken.append(record)
        used_ids.add(record["evidence_id"])
        used_contents.add(normalized)
    return taken


def audit_scale(split_id: str, tasks: list[dict], evidence: list[dict],
                target: int, consumed: tuple[set, set, set],
                other_scales: dict[str, tuple[list[dict], list[dict]]],
                reference_profile: dict | None) -> tuple[dict, list[str]]:
    """Every check that must pass before a byte is written."""
    problems: list[str] = []
    consumed_q, consumed_c, consumed_p = consumed
    by_id = {r["evidence_id"]: r for r in evidence}
    results: dict[str, Any] = {}

    # --- requested corpus scale --------------------------------------------
    ratio = len(evidence) / target
    results["scale_conformance"] = {
        "target": target, "actual": len(evidence),
        "ratio": round(ratio, 4), "tolerance": SCALE_TOLERANCE}
    if abs(ratio - 1.0) > SCALE_TOLERANCE:
        problems.append(f"{split_id} scale: {len(evidence)} records vs target "
                        f"{target} is outside +/-{SCALE_TOLERANCE:.0%}")

    # --- missing required evidence / inferability --------------------------
    missing = sum(1 for t in tasks for eid in t["required_evidence_ids"]
                  if eid not in by_id)
    if missing:
        problems.append(f"{split_id} missing required evidence records: {missing}")
    not_inferable = {t["task_id"]: verify_inferable(t, by_id) for t in tasks}
    not_inferable = {k: v for k, v in not_inferable.items() if v}
    results["inferability"] = {"not_inferable": len(not_inferable)}
    if not_inferable:
        problems.append(f"{split_id} non-inferable tasks: {len(not_inferable)} "
                        f"(sample {list(not_inferable)[:3]})")

    # --- retrievability ----------------------------------------------------
    subject_unfindable = 0
    for task in tasks:
        text = " || ".join(_norm(by_id[e]["content"])
                           for e in task["required_evidence_ids"] if e in by_id)
        if _norm(task["_oracle_metadata"]["surfaces"]["subject"]) not in text:
            subject_unfindable += 1
    results["retrievability"] = {"subject_not_findable": subject_unfindable}
    if subject_unfindable:
        problems.append(f"{split_id} subject not findable in required evidence: "
                        f"{subject_unfindable}")

    # --- oracle isolation --------------------------------------------------
    leaks_evidence = [r["evidence_id"] for r in evidence
                      if RUNTIME_FORBIDDEN.search(r["content"])]
    leaks_question = [t["task_id"] for t in tasks
                      if RUNTIME_FORBIDDEN.search(t["question"])]
    results["oracle_isolation"] = {
        "latent_ids_in_evidence": len(leaks_evidence),
        "latent_ids_in_questions": len(leaks_question),
        "every_task_has_proof_graph": all("_oracle_metadata" in t for t in tasks)}
    if leaks_evidence or leaks_question:
        problems.append(f"{split_id} leaks latent identifiers")
    if not results["oracle_isolation"]["every_task_has_proof_graph"]:
        problems.append(f"{split_id} missing proof graphs")

    # --- answer leakage ----------------------------------------------------
    by_prefix: dict[str, list[dict]] = {}
    for record in evidence:
        by_prefix.setdefault(record["evidence_id"].split("/", 1)[0], []).append(record)
    question_leaks = sum(1 for t in tasks if _leaks(t["question"], t["answer"]))
    distractor_leaks = sum(
        1 for t in tasks for r in by_prefix.get(t["task_id"], [])
        if r["metadata"]["record_kind"] not in _ANSWER_BEARING_KINDS
        and _leaks(r["content"], t["answer"]))
    results["leakage"] = {"question_leaks": question_leaks,
                          "distractor_leaks": distractor_leaks}
    if question_leaks or distractor_leaks:
        problems.append(f"{split_id} leakage {results['leakage']}")

    # --- duplicates within the scale ---------------------------------------
    dup_tasks = len(tasks) - len({t["task_id"] for t in tasks})
    dup_pairs = len(tasks) - len({(_norm(t["question"]), _norm(t["answer"]))
                                  for t in tasks})
    dup_evidence = len(evidence) - len({r["evidence_id"] for r in evidence})
    results["duplicates"] = {"task_ids": dup_tasks,
                             "question_answer_pairs": dup_pairs,
                             "evidence_ids": dup_evidence}
    for name, count in results["duplicates"].items():
        if count:
            problems.append(f"{split_id} duplicate {name}: {count}")

    # --- overlap with consumed splits (CONTENT, not ids) -------------------
    q_overlap = {_norm(t["question"]) for t in tasks} & consumed_q
    p_overlap = {(_norm(t["question"]), _norm(t["answer"])) for t in tasks} & consumed_p
    c_overlap = {_norm(r["content"]) for r in evidence} & consumed_c
    results["overlap_with_consumed"] = {
        "questions": len(q_overlap), "question_answer_pairs": len(p_overlap),
        "evidence_content": len(c_overlap)}
    for name, count in results["overlap_with_consumed"].items():
        if count:
            problems.append(f"{split_id} shares {count} {name} with a consumed "
                            f"or evaluation split")

    # --- cross-scale overlap ----------------------------------------------
    cross: dict[str, dict[str, int]] = {}
    for other_id, (other_tasks, other_evidence) in other_scales.items():
        cross[other_id] = {
            "questions": len({_norm(t["question"]) for t in tasks}
                             & {_norm(t["question"]) for t in other_tasks}),
            "evidence_content": len({_norm(r["content"]) for r in evidence}
                                    & {_norm(r["content"]) for r in other_evidence}),
        }
        for name, count in cross[other_id].items():
            if count:
                problems.append(f"{split_id} shares {count} {name} with "
                                f"{other_id}")
    results["cross_scale_overlap"] = cross

    # --- balance -----------------------------------------------------------
    profile = structural_profile(tasks)
    results["structural_profile"] = profile
    family_counts = set(profile["family_counts"].values())
    if len(profile["family_counts"]) != 10 or family_counts != {TASKS_PER_FAMILY}:
        problems.append(f"{split_id} family imbalance: {profile['family_counts']}")
    regimes = profile["entity_regime_counts"]
    if set(regimes) != {"canonical", "abbreviation"}:
        problems.append(f"{split_id} entity regimes not both present: {regimes}")
    elif min(regimes.values()) / len(tasks) < 0.35:
        problems.append(f"{split_id} entity-regime imbalance: {regimes}")

    # --- structural comparability across scales ---------------------------
    if reference_profile is not None:
        residuals: dict[str, float] = {}
        total = len(tasks)
        for key in ("family_counts", "proof_depth_counts",
                    "required_set_size_counts", "answer_kind_counts"):
            mine, theirs = profile[key], reference_profile[key]
            categories = set(mine) | set(theirs)
            worst = max(
                (abs(mine.get(c, 0) / total - theirs.get(c, 0) / total)
                 for c in categories), default=0.0)
            residuals[key] = round(worst, 4)
            if worst > COMPOSITION_TOLERANCE:
                problems.append(
                    f"{split_id} {key} deviates {worst:.4f} from the reference "
                    f"scale, above the frozen {COMPOSITION_TOLERANCE} "
                    f"comparability tolerance, so corpus size would not be the "
                    f"only treatment: {mine} vs {theirs}")
        results["composition_residuals"] = residuals
        results["composition_tolerance"] = COMPOSITION_TOLERANCE
    return results, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the B3 calibration suite")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if OUT.exists():
        print(f"ABORT: refusing to overwrite a frozen suite: {OUT}")
        return 1

    print("=== B3 multi-scale calibration suite ===")
    print(f"  fixed tasks/family = {TASKS_PER_FAMILY} at EVERY scale "
          f"(corpus size is the treatment variable)")
    print(f"  scales = {[s[0] for s in SCALES]}")
    print(f"  scale tolerance = +/-{SCALE_TOLERANCE:.0%} (frozen before "
          f"any count observed)\n")

    print("[1/5] loading consumed/evaluation splits for content screening")
    consumed = load_consumed()
    print(f"      {len(consumed[0])} questions, {len(consumed[1])} evidence "
          f"contents, {len(consumed[2])} question-answer pairs")

    print("\n[2/5] building the filler pool (non-required records only)")
    filler = build_filler_pool(consumed[1])
    print(f"      {len(filler)} screened non-required records available")

    print("\n[3/5] generating scales")
    built: dict[str, tuple[list[dict], list[dict]]] = {}
    manifests: dict[str, dict] = {}
    # Accumulate as scales are built so later scales avoid earlier ones too.
    used_filler_ids: set[str] = set()
    reference_composition: dict[str, dict[tuple, int]] | None = None
    seen_questions: set[str] = set(consumed[0])
    seen_contents: set[str] = set(consumed[1])
    used_filler_contents: set[str] = set()
    for split_id, target, seed in SCALES:
        corpus = build_v4_corpus(split=split_id, seed=seed,
                                 tasks_per_family=OVERGENERATE_PER_FAMILY,
                                 styles=ID_STYLES, regimes=ID_REGIMES)
        raw_tasks, raw_evidence = list(corpus.tasks), list(corpus.evidence)
        evidence_by_task: dict[str, list[dict]] = {}
        for record in raw_evidence:
            evidence_by_task.setdefault(
                record["evidence_id"].split("/", 1)[0], []).append(record)
        # Drop tasks colliding on question or required-evidence content with any
        # consumed split or any earlier scale, then trim to the fixed count.
        survivors: dict[str, list[dict]] = {}
        local_questions: set[str] = set()
        local_pairs: set[tuple[str, str]] = set()
        for task in raw_tasks:
            question = _norm(task["question"])
            pair = (question, _norm(task["answer"]))
            # Dedup WITHIN the scale as well as against the ledger. Adding to the
            # ledger only after selecting the whole scale let two tasks inside one
            # scale share a question+answer pair.
            if question in seen_questions or question in local_questions:
                continue
            if pair in local_pairs:
                continue
            # Screen ALL of a task's own evidence, not just its required records.
            # Restricting the screen to required records let a task's dead-end or
            # near-duplicate content collide with another split, which surfaced as
            # single shared evidence records.
            own = evidence_by_task.get(task["task_id"], [])
            if any(_norm(r["content"]) in seen_contents for r in own):
                continue
            if len({_norm(r["content"]) for r in own}) != len(own):
                continue  # internally duplicated content
            local_questions.add(question)
            local_pairs.add(pair)
            survivors.setdefault(task["family"], []).append(task)
        short = {f: len(v) for f, v in survivors.items()
                 if len(v) < TASKS_PER_FAMILY}
        if len(survivors) < 10 or short:
            print(f"      ABORT: {split_id} has families short of "
                  f"{TASKS_PER_FAMILY} after collision filtering: {short}")
            return 1
        # Collision filtering removes different tasks at each scale, which
        # drifts the proof-depth / answer-kind / required-set mix. Corpus size
        # must be the ONLY treatment, so selection is STRATIFIED to match the
        # reference scale's per-family composition instead of taking the first
        # survivors. Fixing the drift is the right move; widening the
        # comparability tolerance until the drift passed would have defeated the
        # design.
        if reference_composition is None:
            tasks = [t for family in sorted(survivors)
                     for t in survivors[family][:TASKS_PER_FAMILY]]
        else:
            tasks = []
            for family in sorted(survivors):
                wanted = Counter(reference_composition.get(family, {}))
                pool_by_key: dict[tuple, list[dict]] = {}
                for task in survivors[family]:
                    pool_by_key.setdefault(_composition_key(task), []).append(task)
                chosen: list[dict] = []
                # First satisfy the reference composition exactly where possible.
                for key, count in sorted(wanted.items()):
                    available = pool_by_key.get(key, [])
                    chosen.extend(available[:count])
                # Then top up deterministically if some key was unavailable.
                if len(chosen) < TASKS_PER_FAMILY:
                    already = {t["task_id"] for t in chosen}
                    for task in survivors[family]:
                        if len(chosen) >= TASKS_PER_FAMILY:
                            break
                        if task["task_id"] not in already:
                            chosen.append(task)
                tasks.extend(chosen[:TASKS_PER_FAMILY])
        kept_ids = {t["task_id"] for t in tasks}
        base_evidence = [r for r in raw_evidence
                         if r["evidence_id"].split("/", 1)[0] in kept_ids]
        for task in tasks:
            seen_questions.add(_norm(task["question"]))
        for record in base_evidence:
            seen_contents.add(_norm(record["content"]))
        need = target - len(base_evidence)
        if reference_composition is None:
            reference_composition = composition_by_family(tasks)
        if need < 0:
            print(f"      ABORT: {split_id} base corpus {len(base_evidence)} "
                  f"already exceeds target {target}")
            return 1
        answers = {t["answer"] for t in tasks}
        # One GLOBAL content ledger covering base AND filler from every scale.
        # Tracking filler separately let a later scale's BASE evidence collide
        # with an earlier scale's FILLER, which surfaced as 1-3 shared records
        # per pair. Screening both against the same ledger removes the class.
        base_contents = {_norm(r["content"]) for r in base_evidence}
        slice_ = take_filler(filler, need, base_contents, answers,
                             used_filler_ids, seen_contents)
        if len(slice_) < need:
            print(f"      ABORT: filler pool exhausted for {split_id} "
                  f"(needed {need}, had {len(slice_)})")
            return 1
        evidence = base_evidence + slice_
        built[split_id] = (tasks, evidence)
        print(f"      {split_id}: {len(tasks)} tasks, {len(base_evidence)} base "
              f"+ {len(slice_)} filler = {len(evidence)} records "
              f"(target {target})")

    print("\n[4/5] auditing (fail-closed; nothing written unless all pass)")
    reference = structural_profile(built[SCALES[0][0]][0])
    all_problems: list[str] = []
    audits: dict[str, dict] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        others = {k: v for k, v in built.items() if k != split_id}
        results, problems = audit_scale(
            split_id, tasks, evidence, target, consumed, others,
            None if split_id == SCALES[0][0] else reference)
        audits[split_id] = results
        all_problems += problems
        print(f"      {split_id}: {'PASS' if not problems else 'FAIL'}")
    for line in all_problems:
        print(f"      FAIL {line}")
    if all_problems:
        print("\nABORT: audits failed. Nothing written.")
        return 1
    print("      ALL AUDITS PASS")

    if args.dry_run:
        print("\n[dry run] audits pass; nothing written by request.")
        return 0

    print("\n[5/5] writing and freezing")
    OUT.mkdir(parents=True)
    digests: dict[str, str] = {}
    for split_id, target, seed in SCALES:
        tasks, evidence = built[split_id]
        directory = OUT / split_id
        directory.mkdir()
        task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in tasks)
        evidence_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in evidence)
        (directory / "oracle_tasks.jsonl").write_text(task_text)
        (directory / "evidence.jsonl").write_text(evidence_text)

        profile = audits[split_id]["structural_profile"]
        manifest = {
            "split_id": split_id,
            "generation_seed": seed,
            "task_count": len(tasks),
            "evidence_count": len(evidence),
            "tasks_per_family": TASKS_PER_FAMILY,
            "family_counts": profile["family_counts"],
            "entity_regime_counts": profile["entity_regime_counts"],
            "bridged_counts": profile["bridged_counts"],
            "answer_kind_counts": profile["answer_kind_counts"],
            "proof_depth_counts": {str(k): v for k, v in profile["proof_depth_counts"].items()},
            "required_set_size_counts": {str(k): v for k, v in profile["required_set_size_counts"].items()},
            "source_cluster_counts": profile["source_cluster_counts"],
            "task_sha256": sha256_text(task_text),
            "evidence_sha256": sha256_text(evidence_text),
            "generator_revision": "controlled-gate-a-v4",
            "generator_config_hash": sha256_text(json.dumps(
                {"tasks_per_family": TASKS_PER_FAMILY, "seed": seed,
                 "styles": list(ID_STYLES), "regimes": list(ID_REGIMES),
                 "filler_seed": FILLER_SEED,
                 "filler_tasks_per_family": FILLER_TASKS_PER_FAMILY},
                sort_keys=True))[:16],
            "target_evidence_count": target,
            "scale_conformance": audits[split_id]["scale_conformance"],
            "budget_metadata": BUDGET_METADATA,
        }
        manifests[split_id] = manifest
        (directory / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (directory / "CALIBRATION_AUDIT.json").write_text(json.dumps({
            "split_id": split_id, "frozen_before_evaluation": True,
            "audit": audits[split_id],
            "policy_independence": (
                "No rho, k_min or k_max was used to build this data. Budget "
                "fields are recorded UNSET. Calibration data creation must not "
                "depend on the policy it will later evaluate."),
        }, indent=2, sort_keys=True) + "\n")

        for path in sorted(directory.iterdir()):
            digests[f"{split_id}/{path.name}"] = hashlib.sha256(
                path.read_bytes()).hexdigest()

    (OUT / "CALIBRATION.sha256").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    (OUT / "SUITE_MANIFEST.json").write_text(json.dumps({
        "suite_id": "b3_calibration_v1",
        "purpose": ("Multi-scale calibration for the B3 candidate-budget "
                    "policy. Corpus size is the treatment variable; task "
                    "structure is held fixed."),
        "fixed_tasks_per_family": TASKS_PER_FAMILY,
        "scales": {s[0]: {"target": s[1], "seed": s[2]} for s in SCALES},
        "filler": {"seed": FILLER_SEED,
                   "tasks_per_family": FILLER_TASKS_PER_FAMILY,
                   "source": "non-required records only, screened against every "
                             "consumed split, partitioned disjointly across scales"},
        "deviation_from_request": (
            "The requested 500-record scale was replaced by 700. The generator "
            "couples evidence volume to task count at ~4.41 records/task, so a "
            "fixed 150-task structure has a floor near 661 records. Reaching "
            "500 would have required ~110 tasks, breaking the fixed-task-count "
            "design and giving the least stable retrieval curves at the "
            "smallest scale. Holding task structure fixed was judged the more "
            "important property."),
        "scale_tolerance": SCALE_TOLERANCE,
        "budget_metadata": BUDGET_METADATA,
        "manifests": manifests,
        "frozen_before_evaluation": True,
        "run_discipline": ("Retrieval evaluation has NOT been run. rho is NOT "
                           "chosen. No S2 or HRM execution has touched this "
                           "data."),
    }, indent=2, sort_keys=True) + "\n")

    print(f"      wrote {OUT}")
    print(f"      {len(digests)} files hashed into CALIBRATION.sha256")
    print("\nCALIBRATION SUITE FROZEN. rho not chosen; no evaluation run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
