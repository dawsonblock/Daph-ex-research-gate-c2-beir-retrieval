"""EOB-v1 (Executive Opportunity Benchmark) -- D0/D2/D3 regime generator.

Per configs/gate_eob_v1_design.json. D1 (memory-required) is NOT generated
here -- it is sampled from the existing, already-audited b3_calibration_v1
corpus (see scripts/build_eob_v1_suite.py), since that generation problem is
already solved and frozen.

D0 tasks are self-contained: the answer is a pure function of the question
text alone. Every task's recorded answer is independently re-derived by a
REFERENCE SOLVER that parses the rendered question text -- NOT by trusting
the generator's own internal state -- and any disagreement is a hard
rejection at generation time, not a warning. This is the same
"inferability_verified" discipline the b3/C4 corpora already use, applied to
a template family where it is actually possible to write a truly independent
solver (arithmetic/comparison/transform/restatement all parse back cleanly).

D2 adds a redundant CONFIRMING evidence record to a D0 task (paraphrased, not
a verbatim copy of the question). D3 adds a near-miss DISTRACTOR evidence
record (same template, deliberately wrong value for THIS question).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

_ENTITY_NAMES = (
    "Bramwell", "Corvath", "Dellinger", "Everhart", "Farrow", "Grimsby",
    "Halloway", "Ironside", "Juniper", "Kestrel", "Larchmont", "Marrow",
    "Nightshade", "Osprey", "Pelham", "Quillon", "Ravenscroft", "Sable",
    "Thistlewood", "Umbral", "Vellichor", "Wrenfield", "Xylan", "Yarrow",
)

#: Distinct from _ENTITY_NAMES (which comparison-family questions already use
#: as visible answer content) -- these back the LOCATOR every D0 task gets,
#: shaped to match bridge_extraction._V4_ENTITY
#: (r"\b([A-Z][a-z]+(?:\s+[a-z]+){1,3})\b": one capitalized head word plus 1-3
#: lowercase words) so grammar_v4 can actually extract it as an entity from
#: D2/D3 evidence content. Without a shape grammar_v4 recognizes, G2 cannot
#: build a graph edge for the confirming/distractor record regardless of
#: whether C2 retrieval finds it -- this was a real gap caught by dry-run
#: (empty A1 packets on D2/D3 even with the record correctly retrieved).
_LOCATOR_HEADS = (
    "Solberg", "Kessington", "Marrowgate", "Fenwick", "Tarnowski", "Whitlock",
    "Ashgrove", "Barrowfield", "Crestholm", "Dunmore",
)
_LOCATOR_ROLES = (
    "result registry", "output ledger", "value archive", "record index",
)
_RELATION_WORD_BY_FAMILY = {
    "arithmetic": "computed result", "comparison": "ranking outcome",
    "transform": "transformed value", "restatement": "provisioned value",
}


def _make_locator(rng: random.Random) -> str:
    return f"{rng.choice(_LOCATOR_HEADS)} {rng.choice(_LOCATOR_ROLES)}"


@dataclass
class EobTask:
    task_id: str
    regime: str
    family: str
    question: str
    answer: str
    evidence: list[dict] = field(default_factory=list)  # [{evidence_id, content}]
    required_evidence_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ReferenceSolverMismatch(ValueError):
    """A D0/D2/D3 task's recorded answer disagrees with the independent
    reference solver's re-derivation from the question text. Fail closed --
    the task is rejected, never silently kept."""


# --- reference solver: parses rendered question text independently --------

_ARITH_RE = re.compile(
    r"compute (-?\d+) (plus|minus|times) (-?\d+)")
_COMPARISON_RE = re.compile(
    r"(\w+) has (\w+) (-?\d+)\. (\w+) has \2 (-?\d+)\. (\w+) has \2 (-?\d+)\. "
    r"Which entity has the (highest|lowest) \2")
_TRANSFORM_REVERSE_RE = re.compile(r"Reverse the letters in '(\w+)'")
_TRANSFORM_UPPER_RE = re.compile(r"Convert '(\w+)' to uppercase")
_TRANSFORM_CONCAT_RE = re.compile(r"Concatenate '(\w+)' and '(\w+)', in that order")
_RESTATEMENT_RE = re.compile(r"Assume (\w+) is set to (-?\w+)\. What is the value of \1")


def reference_solve(question: str) -> str:
    """Independently re-derive the answer from ONLY the question text.
    Raises ReferenceSolverMismatch (via the caller) if this cannot be done --
    never guesses."""
    m = _ARITH_RE.search(question)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return str({"plus": a + b, "minus": a - b, "times": a * b}[op])

    m = _COMPARISON_RE.search(question)
    if m:
        e1, _attr, v1, e2, v2, e3, v3, direction = (
            m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5)),
            m.group(6), int(m.group(7)), m.group(8))
        pairs = [(e1, v1), (e2, v2), (e3, v3)]
        chosen = max(pairs, key=lambda p: p[1]) if direction == "highest" else min(pairs, key=lambda p: p[1])
        return chosen[0]

    m = _TRANSFORM_REVERSE_RE.search(question)
    if m:
        return m.group(1)[::-1]

    m = _TRANSFORM_UPPER_RE.search(question)
    if m:
        return m.group(1).upper()

    m = _TRANSFORM_CONCAT_RE.search(question)
    if m:
        return m.group(1) + m.group(2)

    m = _RESTATEMENT_RE.search(question)
    if m:
        return m.group(2)

    raise ReferenceSolverMismatch(f"reference_solve: no pattern matched question: {question!r}")


# --- D0 task builders -------------------------------------------------------

def _build_arithmetic(rng: random.Random, task_id: str) -> EobTask:
    a, b = rng.randint(2, 97), rng.randint(2, 97)
    op = rng.choice(["plus", "minus", "times"])
    locator = _make_locator(rng)
    question = f"Reference: {locator}. If you compute {a} {op} {b}, what is the result?"
    answer = {"plus": str(a + b), "minus": str(a - b), "times": str(a * b)}[op]
    return EobTask(task_id, "D0_direct_sufficient", "arithmetic", question, answer,
                   metadata={"a": a, "b": b, "op": op, "locator": locator,
                            "relation_word": _RELATION_WORD_BY_FAMILY["arithmetic"]})


def _build_comparison(rng: random.Random, task_id: str) -> EobTask:
    names = rng.sample(_ENTITY_NAMES, 3)
    attr = rng.choice(["height", "weight", "score"])
    values = rng.sample(range(10, 990), 3)
    direction = rng.choice(["highest", "lowest"])
    locator = _make_locator(rng)
    question = (f"Reference: {locator}. {names[0]} has {attr} {values[0]}. {names[1]} has {attr} {values[1]}. "
               f"{names[2]} has {attr} {values[2]}. Which entity has the {direction} {attr}?")
    pairs = list(zip(names, values))
    answer = (max(pairs, key=lambda p: p[1]) if direction == "highest"
             else min(pairs, key=lambda p: p[1]))[0]
    return EobTask(task_id, "D0_direct_sufficient", "comparison", question, answer,
                   metadata={"names": names, "attr": attr, "values": values, "direction": direction,
                            "locator": locator, "relation_word": _RELATION_WORD_BY_FAMILY["comparison"]})


def _random_token(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))


def _build_transform(rng: random.Random, task_id: str) -> EobTask:
    kind = rng.choice(["reverse", "upper", "concat"])
    locator = _make_locator(rng)
    prefix = f"Reference: {locator}. "
    if kind == "reverse":
        tok = _random_token(rng)
        question = f"{prefix}Reverse the letters in '{tok}'."
        answer = tok[::-1]
    elif kind == "upper":
        tok = _random_token(rng)
        question = f"{prefix}Convert '{tok}' to uppercase."
        answer = tok.upper()
    else:
        t1, t2 = _random_token(rng, 4), _random_token(rng, 4)
        question = f"{prefix}Concatenate '{t1}' and '{t2}', in that order."
        answer = t1 + t2
    return EobTask(task_id, "D0_direct_sufficient", "transform", question, answer,
                   metadata={"kind": kind, "locator": locator,
                            "relation_word": _RELATION_WORD_BY_FAMILY["transform"]})


def _build_restatement(rng: random.Random, task_id: str) -> EobTask:
    var = rng.choice(["X", "Y", "Z", "N", "K"])
    value = str(rng.randint(1, 999))
    locator = _make_locator(rng)
    question = f"Reference: {locator}. Assume {var} is set to {value}. What is the value of {var}?"
    return EobTask(task_id, "D0_direct_sufficient", "restatement", question, value,
                   metadata={"var": var, "value": value, "locator": locator,
                            "relation_word": _RELATION_WORD_BY_FAMILY["restatement"]})


_D0_BUILDERS = {
    "arithmetic": _build_arithmetic,
    "comparison": _build_comparison,
    "transform": _build_transform,
    "restatement": _build_restatement,
}


def build_d0_tasks(seed: int, tasks_per_family: int) -> list[EobTask]:
    """Build D0 tasks, verifying every single one against the independent
    reference solver before returning it. A mismatch aborts the whole build
    (fail-closed) rather than silently dropping the bad task, since a
    solver/template disagreement indicates a real bug that likely affects
    other tasks from the same family too."""
    rng = random.Random(seed)
    tasks: list[EobTask] = []
    ordinal = 0
    for family, builder in _D0_BUILDERS.items():
        for _ in range(tasks_per_family):
            ordinal += 1
            task = builder(rng, f"eob-d0-{family}-{ordinal:04d}")
            solved = reference_solve(task.question)
            if solved != task.answer:
                raise ReferenceSolverMismatch(
                    f"{task.task_id}: generator answer {task.answer!r} != "
                    f"reference_solve {solved!r} for question {task.question!r}")
            tasks.append(task)
    return tasks


# --- D2 (confirming evidence) / D3 (distractor evidence) -------------------

def _confirming_or_distractor_value(task: EobTask, rng: random.Random, *, correct: bool) -> str:
    """The value asserted for this task's locator -- the correct answer if
    correct=True (D2), a verified-different value otherwise (D3)."""
    fam = task.family
    if fam in ("arithmetic", "restatement"):
        if correct:
            return task.answer
        base = int(task.answer)
        offsets = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        return str(base + rng.choice(offsets))
    if fam == "comparison":
        if correct:
            return task.answer
        return rng.choice([n for n in task.metadata["names"] if n != task.answer])
    if fam == "transform":
        if correct:
            return task.answer
        original_last = task.answer[-1]
        replacement = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != original_last])
        return task.answer[:-1] + replacement
    raise ValueError(f"unknown family {fam!r}")


def _b3_style_fact_sentence(locator: str, relation_word: str, value: str) -> str:
    """Deliberately matches the b3 corpus's own formal_registry rendering
    style (hrm_adaptive_memory/experiments/generalization_dataset_v4.py:_render)
    -- 'The {relation} registry records that {subject} is assigned {obj}.' --
    since that is the literal sentence shape grammar_v4/G2 are proven to
    parse into a graph edge for the b3/D1 corpus. locator is entity-shaped
    (matches bridge_extraction._V4_ENTITY) and relation_word is passed
    directly as the graph's `relation` parameter by the runner (NOT extracted
    via extract_target_relation, which is tuned to a different question
    phrasing) -- so RECORD_EXPRESSES_RELATION fires because relation_word is
    guaranteed to be a literal substring of this sentence, by construction."""
    return f"The {relation_word} registry records that {locator} is assigned {value}."


def _paraphrase_confirmation(task: EobTask, rng: random.Random) -> str:
    """A prose restatement of the SAME fact, not a verbatim copy of the
    question -- required by configs/gate_eob_v1_design.json VERIFICATION_
    REQUIRED_BEFORE_FREEZE."""
    value = _confirming_or_distractor_value(task, rng, correct=True)
    return _b3_style_fact_sentence(task.metadata["locator"], task.metadata["relation_word"], value)


def _near_miss_distractor(task: EobTask, rng: random.Random) -> str:
    """Same template, a DIFFERENT value than the correct answer for THIS
    question -- verified not-equal-to-answer below."""
    value = _confirming_or_distractor_value(task, rng, correct=False)
    return _b3_style_fact_sentence(task.metadata["locator"], task.metadata["relation_word"], value)


def _index_record(evidence_id: str, content: str, record_kind: str) -> dict:
    """Full IndexRecord-compatible schema (scripts/run_gate_c4.py:_to_index_records
    requires evidence_id/source_id/content/source_type/metadata) -- EOB-v1's
    evidence records must satisfy the same contract every other corpus in
    this project does, not a stripped-down {evidence_id, content} shape."""
    return {
        "evidence_id": evidence_id, "source_id": evidence_id, "content": content,
        "source_type": "eob_v1_synthetic", "metadata": {"record_kind": record_kind},
    }


def build_d2_tasks(d0_tasks: list[EobTask], seed: int) -> list[EobTask]:
    rng = random.Random(seed)
    out = []
    for i, d0 in enumerate(d0_tasks):
        content = _paraphrase_confirmation(d0, rng)
        ev_id = f"{d0.task_id}/confirm"
        task = EobTask(
            task_id=d0.task_id.replace("d0-", "d2-"), regime="D2_both_sufficient",
            family=d0.family, question=d0.question, answer=d0.answer,
            evidence=[_index_record(ev_id, content, "confirming")],
            required_evidence_ids=[ev_id],
            metadata={**d0.metadata, "d0_source": d0.task_id})
        out.append(task)
    return out


def _distractor_value(family: str, content: str) -> str:
    """Extract the value a _b3_style_fact_sentence asserts ('... is assigned
    {value}.'), so it can be checked against the correct answer precisely --
    substring checks are unreliable here (e.g. '42' is a substring of '142').
    Family-agnostic now: every family shares the same sentence template."""
    if " is assigned " not in content:
        raise ValueError(f"content does not match the expected fact-sentence template: {content!r}")
    return content.rsplit(" is assigned ", 1)[-1].rstrip(".")


def build_d3_tasks(d0_tasks: list[EobTask], seed: int) -> list[EobTask]:
    rng = random.Random(seed)
    out = []
    for d0 in d0_tasks:
        content = _near_miss_distractor(d0, rng)
        distractor_value = _distractor_value(d0.family, content)
        if distractor_value == d0.answer:
            raise ValueError(
                f"{d0.task_id}: near-miss distractor accidentally matches the "
                f"correct answer ({d0.answer!r}) -- not a valid distractor")
        ev_id = f"{d0.task_id}/distractor"
        task = EobTask(
            task_id=d0.task_id.replace("d0-", "d3-"), regime="D3_memory_distractor",
            family=d0.family, question=d0.question, answer=d0.answer,
            evidence=[_index_record(ev_id, content, "distractor")],
            required_evidence_ids=[],  # the distractor is NOT required -- the answer never depended on it
            metadata={**d0.metadata, "d0_source": d0.task_id})
        out.append(task)
    return out


def select_d0_subset(d0_tasks: list[EobTask], seed: int, n: int) -> list[EobTask]:
    """A fixed-seed subset of the base D0 pool, surfaced as the pure
    D0_direct_sufficient regime (no evidence). Per configs/gate_eob_v2_design.json
    BASE_TASK_DECOUPLING: D0/D2/D3 no longer need identical 1:1 sizing --
    ALL base facts back D2/D3 (one confirming/distractor task each), while
    only a SUBSET is surfaced as pure D0. Not leakage: D0/D2/D3 are different
    evidence CONDITIONS on overlapping underlying facts, exactly as EOB-v1's
    D2/D3 already shared their D0 source question/answer by construction --
    this just decouples how many facts exist from how many appear bare."""
    if n > len(d0_tasks):
        raise ValueError(f"cannot select {n} tasks from a pool of {len(d0_tasks)}")
    rng = random.Random(seed)
    return rng.sample(d0_tasks, n)
