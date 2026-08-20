"""I3.15: Task generator for the 2x3 retrieval-fair epistemically-hard benchmark.

Creates tasks across a 2x3 matrix:
  Retrieval: Easy / Hard
  Epistemic: Easy / Medium / Hard

Each task has:
  - Same-domain evidence (retrieval-fair)
  - Varying epistemic demands (direct -> temporal -> multi-hop)
  - Two retrieval query variants (domain-specific vs abstract)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from hrm_adaptive_memory.cognitive_control.core import DecisionAction
from hrm_adaptive_memory.cognitive_control.state import (
    TemporalStatus, VerificationState,
)
from hrm_adaptive_memory.executive.evidence_benchmark import (
    EvidenceItem, EvidenceTask, EvidenceHypothesis,
)
from hrm_adaptive_memory.executive.semantic_relations.raw_semantic_generator import (
    SemanticTask, GoldRelation,
)
from hrm_adaptive_memory.executive.semantic_relations.i3_15_epistemic_corpus import (
    I3_15Passage, EASY_PASSAGES, MEDIUM_PASSAGES, HARD_PASSAGES,
    get_corpus,
)


DOMAIN_SUBJECTS = {
    "api_gateway": "the API gateway",
    "database": "the database",
    "cdn": "the CDN",
    "kubernetes": "the Kubernetes cluster",
    "security": "the security posture",
    "deployment": "the deployment",
    "monitoring": "the monitoring system",
    "cache": "the Redis cache",
    "message_queue": "the message queue",
    "load_balancer": "the load balancer",
}

# Query variants for retrieval difficulty
# Easy: uses domain-specific terms that appear in passages
# Hard: uses the subject but with abstract framing that doesn't match passage vocabulary
QUERY_EASY = "Determine the current status of {subject}."
QUERY_HARD = "Check whether {subject} is up and serving traffic right now."


def _make_hyps(subject: str) -> tuple[EvidenceHypothesis, ...]:
    h1 = EvidenceHypothesis(
        hypothesis_id="H1",
        proposition=f"{subject.capitalize()} is currently operational and confirmed.",
        answer_action=DecisionAction.ANSWER,
        answer_payload=f"confirmed: {subject}",
    )
    h2 = EvidenceHypothesis(
        hypothesis_id="H2",
        proposition=f"{subject.capitalize()} is currently not operational or unconfirmed.",
        answer_action=DecisionAction.DEFER,
        answer_payload=f"insufficient evidence: {subject}",
    )
    return (h1, h2)


def _passage_to_evidence(
    passage: I3_15Passage,
    evidence_id: str,
    retrieved: bool = True,
) -> EvidenceItem:
    """Convert a passage to an EvidenceItem with gold relations."""
    # Map gold_relation to supports/contradicts
    supports = ()
    contradicts = ()
    vstate = VerificationState.MISSING
    vr = "MISSING"

    if passage.gold_relation == "SUPPORT":
        supports = ("H1",)
        contradicts = ("H2",)
        vstate = VerificationState.SUFFICIENT
        vr = "SUFFICIENT"
    elif passage.gold_relation == "CONTRADICT":
        supports = ("H2",)
        contradicts = ("H1",)
        vstate = VerificationState.FALSIFIED
        vr = "FALSIFIED"
    elif passage.gold_relation == "CONDITIONAL":
        # Conditional passages are neutral until the condition is met
        # But in a chain, the final passage resolves the condition
        vstate = VerificationState.MISSING
        vr = "MISSING"
    elif passage.gold_relation == "TEMPORAL":
        vstate = VerificationState.MISSING
        vr = "MISSING"

    return EvidenceItem(
        evidence_id=evidence_id,
        proposition=passage.text,
        source_class="primary" if retrieved else "search",
        supports=supports,
        contradicts=contradicts,
        verification_state=vstate,
        temporal_status=TemporalStatus.CURRENT,
        retrieved=retrieved,
        verify_result=vr,
    )


def _passage_gold_relations(
    passage: I3_15Passage,
    evidence_id: str,
) -> tuple[GoldRelation, ...]:
    """Convert passage gold relation to GoldRelation objects."""
    if passage.gold_relation == "SUPPORT":
        return (
            GoldRelation(evidence_id, "H1", "SUPPORT"),
            GoldRelation(evidence_id, "H2", "CONTRADICT"),
        )
    elif passage.gold_relation == "CONTRADICT":
        return (
            GoldRelation(evidence_id, "H1", "CONTRADICT"),
            GoldRelation(evidence_id, "H2", "SUPPORT"),
        )
    else:
        return (
            GoldRelation(evidence_id, "H1", "NEUTRAL"),
            GoldRelation(evidence_id, "H2", "NEUTRAL"),
        )


# ---------------------------------------------------------------------------
# Task generators for each epistemic difficulty level
# ---------------------------------------------------------------------------

def _gen_easy_task(
    task_id: str,
    domain: str,
    passages: list[I3_15Passage],
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """EpistemicEasy: single passage directly states the answer.

    Balanced 50/50 between SUPPORT (ANSWER expected) and CONTRADICT (DEFER expected).
    Uses task_index parity to guarantee exact balance rather than relying on RNG.
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    # Pick a passage from this domain
    domain_passages = [p for p in passages if p.domain == domain]
    if not domain_passages:
        domain_passages = passages

    # Split into support and contradict, pick based on task_index parity
    support_passages = [p for p in domain_passages if p.gold_relation == "SUPPORT"]
    contradict_passages = [p for p in domain_passages if p.gold_relation == "CONTRADICT"]

    # Fallback to any domain if this domain lacks one type
    if not support_passages:
        support_passages = [p for p in passages if p.gold_relation == "SUPPORT"]
    if not contradict_passages:
        contradict_passages = [p for p in passages if p.gold_relation == "CONTRADICT"]

    # Alternate: even index -> SUPPORT, odd index -> CONTRADICT
    if task_index % 2 == 0 and support_passages:
        p1 = rng.choice(support_passages)
    elif contradict_passages:
        p1 = rng.choice(contradict_passages)
    else:
        p1 = rng.choice(domain_passages)

    is_support = p1.gold_relation == "SUPPORT"
    expected = DecisionAction.ANSWER if is_support else DecisionAction.DEFER
    correct = "H1" if is_support else "H2"

    ev = (_passage_to_evidence(p1, "E1", retrieved=True),)
    gold = _passage_gold_relations(p1, "E1")

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15",
        category=f"epistemic_easy_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", expected.value),
        expected_terminal=expected, correct_hypothesis_id=correct,
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_medium_task(
    task_id: str,
    domain: str,
    passages: list[I3_15Passage],
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """EpistemicMedium: temporal supersession or authority conflict.

    Two passages from the same domain with conflicting status. The more
    recent or more authoritative passage determines the answer.

    Balanced 50/50 between ANSWER-expected (support is more recent) and
    DEFER-expected (contradict is more recent) outcomes.
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    # Find two passages from the same domain with conflicting relations
    domain_passages = [p for p in passages if p.domain == domain]
    support = [p for p in domain_passages if p.gold_relation == "SUPPORT"]
    contradict = [p for p in domain_passages if p.gold_relation == "CONTRADICT"]

    if not support or not contradict:
        # Fallback: use any domain with both
        for d in set(p.domain for p in passages):
            s = [p for p in passages if p.domain == d and p.gold_relation == "SUPPORT"]
            c = [p for p in passages if p.domain == d and p.gold_relation == "CONTRADICT"]
            if s and c:
                support = s
                contradict = c
                subject = DOMAIN_SUBJECTS.get(d, d)
                h = _make_hyps(subject)
                break

    # Alternate: even index -> support is more recent (ANSWER),
    #            odd index  -> contradict is more recent (DEFER)
    if task_index % 2 == 0:
        # Support should be more recent -> pick support with later timestamp
        p_support = rng.choice(support)
        p_contradict = rng.choice(contradict)
        # Ensure support is more recent; if not, swap selection
        if p_support.timestamp < p_contradict.timestamp:
            # Find a support passage that is more recent
            later_support = [p for p in support if p.timestamp >= p_contradict.timestamp]
            if later_support:
                p_support = rng.choice(later_support)
            else:
                # No later support — just use what we have, answer will be DEFER
                pass
        p1 = p_contradict  # earlier
        p2 = p_support     # later
    else:
        # Contradict should be more recent -> pick contradict with later timestamp
        p_support = rng.choice(support)
        p_contradict = rng.choice(contradict)
        # Ensure contradict is more recent
        if p_contradict.timestamp < p_support.timestamp:
            later_contradict = [p for p in contradict if p.timestamp >= p_support.timestamp]
            if later_contradict:
                p_contradict = rng.choice(later_contradict)
            else:
                # No later contradict — just use what we have, answer will be ANSWER
                pass
        p1 = p_support      # earlier
        p2 = p_contradict   # later

    # The answer depends on which is more recent
    if p2.timestamp >= p1.timestamp:
        if p2.gold_relation == "SUPPORT":
            expected = DecisionAction.ANSWER
            correct = "H1"
        else:
            expected = DecisionAction.DEFER
            correct = "H2"
    else:
        if p1.gold_relation == "SUPPORT":
            expected = DecisionAction.ANSWER
            correct = "H1"
        else:
            expected = DecisionAction.DEFER
            correct = "H2"

    ev = (
        _passage_to_evidence(p1, "E1", retrieved=True),
        _passage_to_evidence(p2, "E2", retrieved=True),
    )
    gold = _passage_gold_relations(p1, "E1") + _passage_gold_relations(p2, "E2")

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15",
        category=f"epistemic_medium_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", "VERIFY:E2", expected.value),
        expected_terminal=expected, correct_hypothesis_id=correct,
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


def _gen_hard_task(
    task_id: str,
    domain: str,
    passages: list[I3_15Passage],
    rng: random.Random,
    retrieval_hard: bool,
    task_index: int = 0,
) -> SemanticTask:
    """EpistemicHard: multi-step reasoning chain.

    A chain of 3 passages where the final state depends on all prior steps.
    The answer is determined by the last passage in the chain.

    Balanced 50/50 between chains resolving to SUPPORT (operational) and
    chains resolving to CONTRADICT (non-operational).
    """
    subject = DOMAIN_SUBJECTS.get(domain, domain)
    h = _make_hyps(subject)

    # Find chain endings (passages with depends_on that are the last in their chain)
    # A chain ending is a passage that is not depended on by any other passage
    all_by_id = {p.passage_id: p for p in passages}
    depended_on = set()
    for p in passages:
        depended_on.update(p.depends_on)

    chain_endings = [p for p in passages if p.depends_on and p.passage_id not in depended_on]

    # Find chain endings from this domain
    domain_endings = [p for p in chain_endings if p.domain == domain]

    if not domain_endings:
        # Find any domain with chain endings
        for d in set(p.domain for p in chain_endings):
            domain_endings = [p for p in chain_endings if p.domain == d]
            subject = DOMAIN_SUBJECTS.get(d, d)
            h = _make_hyps(subject)
            break

    if not domain_endings:
        # Fallback: use easy passage
        return _gen_easy_task(task_id, domain, passages, rng, retrieval_hard, task_index)

    # Split endings by gold relation
    support_endings = [p for p in domain_endings if p.gold_relation == "SUPPORT"]
    contradict_endings = [p for p in domain_endings if p.gold_relation == "CONTRADICT"]

    # Alternate: even index -> SUPPORT chain, odd index -> CONTRADICT chain
    if task_index % 2 == 0 and support_endings:
        final = rng.choice(support_endings)
    elif contradict_endings:
        final = rng.choice(contradict_endings)
    elif support_endings:
        final = rng.choice(support_endings)
    else:
        final = rng.choice(domain_endings)

    # Reconstruct the chain by following depends_on
    def _find_chain(p):
        result = []
        for dep_id in p.depends_on:
            if dep_id in all_by_id:
                result.extend(_find_chain(all_by_id[dep_id]))
        result.append(p)
        return result

    chain = _find_chain(final)
    # Deduplicate while preserving order
    seen = set()
    chain_unique = []
    for p in chain:
        if p.passage_id not in seen:
            seen.add(p.passage_id)
            chain_unique.append(p)
    chain = chain_unique

    # Create evidence items for each passage in the chain
    ev = tuple(
        _passage_to_evidence(p, f"E{i+1}", retrieved=True)
        for i, p in enumerate(chain)
    )
    gold = tuple()
    for i, p in enumerate(chain):
        gold = gold + _passage_gold_relations(p, f"E{i+1}")

    # The answer depends on the final passage
    is_support = final.gold_relation == "SUPPORT"
    expected = DecisionAction.ANSWER if is_support else DecisionAction.DEFER
    correct = "H1" if is_support else "H2"

    query_template = QUERY_HARD if retrieval_hard else QUERY_EASY
    query = query_template.format(subject=subject)

    et = EvidenceTask(
        task_id=task_id, split="i3_15",
        category=f"epistemic_hard_{'retr_hard' if retrieval_hard else 'retr_easy'}",
        task_summary=query,
        high_stakes=True, budget_profile="STANDARD",
        hypotheses=h, evidence_items=ev,
        retrieve_exposes=(), search_exposes=(),
        oracle_resolution_path=("VERIFY:E1", expected.value),
        expected_terminal=expected, correct_hypothesis_id=correct,
    )
    return SemanticTask(et, gold, tier="", semantic_class="real_evidence")


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

GENERATORS = {
    "easy": _gen_easy_task,
    "medium": _gen_medium_task,
    "hard": _gen_hard_task,
}


def generate_i3_15_corpus(
    n_per_cell: int = 25,
    seed: int = 42,
) -> list[SemanticTask]:
    """Generate the I3.15 task corpus.

    2x3 matrix: RetrievalEasy/Hard x EpistemicEasy/Medium/Hard
    = 6 cells x n_per_cell tasks

    Within each cell, task_index parity guarantees 50/50 split between
    ANSWER-expected (SUPPORT) and DEFER-expected (CONTRADICT) outcomes.
    """
    rng = random.Random(seed)
    corpus = get_corpus()
    domains = list(DOMAIN_SUBJECTS.keys())

    tasks: list[SemanticTask] = []
    task_idx = 0

    for epistemic in ["easy", "medium", "hard"]:
        for retrieval_hard in [False, True]:
            for i in range(n_per_cell):
                domain = domains[task_idx % len(domains)]
                task_id = f"i3_15_{task_idx:04d}"
                gen = GENERATORS[epistemic]
                task = gen(task_id, domain, corpus, rng, retrieval_hard,
                           task_index=i)
                tasks.append(task)
                task_idx += 1

    return tasks


if __name__ == "__main__":
    tasks = generate_i3_15_corpus(n_per_cell=25, seed=42)
    print(f"Generated {len(tasks)} tasks")
    from collections import Counter
    cats = Counter(t.evidence_task.category for t in tasks)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    # Check first task
    t = tasks[0]
    print(f"\nFirst task: {t.evidence_task.task_id}")
    print(f"  Category: {t.evidence_task.category}")
    print(f"  Query: {t.evidence_task.task_summary}")
    print(f"  Evidence: {len(t.evidence_task.evidence_items)} items")
    for ev in t.evidence_task.evidence_items:
        print(f"    {ev.evidence_id}: {ev.proposition[:80]}...")
