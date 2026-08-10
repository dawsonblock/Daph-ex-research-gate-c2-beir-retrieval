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
    question = f"If you compute {a} {op} {b}, what is the result?"
    answer = {"plus": str(a + b), "minus": str(a - b), "times": str(a * b)}[op]
    return EobTask(task_id, "D0_direct_sufficient", "arithmetic", question, answer,
                   metadata={"a": a, "b": b, "op": op})


def _build_comparison(rng: random.Random, task_id: str) -> EobTask:
    names = rng.sample(_ENTITY_NAMES, 3)
    attr = rng.choice(["height", "weight", "score"])
    values = rng.sample(range(10, 990), 3)
    direction = rng.choice(["highest", "lowest"])
    question = (f"{names[0]} has {attr} {values[0]}. {names[1]} has {attr} {values[1]}. "
               f"{names[2]} has {attr} {values[2]}. Which entity has the {direction} {attr}?")
    pairs = list(zip(names, values))
    answer = (max(pairs, key=lambda p: p[1]) if direction == "highest"
             else min(pairs, key=lambda p: p[1]))[0]
    return EobTask(task_id, "D0_direct_sufficient", "comparison", question, answer,
                   metadata={"names": names, "attr": attr, "values": values, "direction": direction})


def _random_token(rng: random.Random, length: int = 5) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))


def _build_transform(rng: random.Random, task_id: str) -> EobTask:
    kind = rng.choice(["reverse", "upper", "concat"])
    if kind == "reverse":
        tok = _random_token(rng)
        question = f"Reverse the letters in '{tok}'."
        answer = tok[::-1]
    elif kind == "upper":
        tok = _random_token(rng)
        question = f"Convert '{tok}' to uppercase."
        answer = tok.upper()
    else:
        t1, t2 = _random_token(rng, 4), _random_token(rng, 4)
        question = f"Concatenate '{t1}' and '{t2}', in that order."
        answer = t1 + t2
    return EobTask(task_id, "D0_direct_sufficient", "transform", question, answer,
                   metadata={"kind": kind})


def _build_restatement(rng: random.Random, task_id: str) -> EobTask:
    var = rng.choice(["X", "Y", "Z", "N", "K"])
    value = str(rng.randint(1, 999))
    question = f"Assume {var} is set to {value}. What is the value of {var}?"
    return EobTask(task_id, "D0_direct_sufficient", "restatement", question, value,
                   metadata={"var": var, "value": value})


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

def _paraphrase_confirmation(task: EobTask, rng: random.Random) -> str:
    """A prose restatement of the SAME fact, not a verbatim copy of the
    question -- required by configs/gate_eob_v1_design.json VERIFICATION_
    REQUIRED_BEFORE_FREEZE."""
    fam = task.family
    if fam == "arithmetic":
        a, b, op = task.metadata["a"], task.metadata["b"], task.metadata["op"]
        word = {"plus": "added to", "minus": "subtracted from", "times": "multiplied by"}[op]
        return f"Records confirm: {b} {word} {a} yields {task.answer}."
    if fam == "comparison":
        names, attr, values = task.metadata["names"], task.metadata["attr"], task.metadata["values"]
        pairs = sorted(zip(names, values), key=lambda p: p[1])
        ranking = ", ".join(f"{n} ({v})" for n, v in pairs)
        return f"Registry note on {attr}, ascending order: {ranking}."
    if fam == "transform":
        kind = task.metadata["kind"]
        if kind == "concat":
            return f"Log entry: the joined token is confirmed as '{task.answer}'."
        return f"Log entry: the transformed token is confirmed as '{task.answer}'."
    if fam == "restatement":
        var, value = task.metadata["var"], task.metadata["value"]
        return f"Configuration audit: variable {var} is currently provisioned as {value}."
    raise ValueError(f"unknown family {fam!r}")


def _near_miss_distractor(task: EobTask, rng: random.Random) -> str:
    """Same template, a DIFFERENT value than the correct answer for THIS
    question -- verified not-equal-to-answer below."""
    fam = task.family
    if fam == "arithmetic":
        a, b, op = task.metadata["a"], task.metadata["b"], task.metadata["op"]
        wrong = str(int(task.answer) + rng.choice([-3, -2, -1, 1, 2, 3]))
        word = {"plus": "added to", "minus": "subtracted from", "times": "multiplied by"}[op]
        return f"Records confirm: {b} {word} {a} yields {wrong}."
    if fam == "comparison":
        names, attr = task.metadata["names"], task.metadata["attr"]
        wrong_entity = rng.choice([n for n in names if n != task.answer])
        return f"Registry note: {wrong_entity} holds the {task.metadata['direction']} recorded {attr}."
    if fam == "transform":
        kind = task.metadata["kind"]
        original_last = task.answer[-1]
        replacement = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != original_last])
        wrong = task.answer[:-1] + replacement
        label = "joined token" if kind == "concat" else "transformed token"
        return f"Log entry: the {label} is confirmed as '{wrong}'."
    if fam == "restatement":
        var = task.metadata["var"]
        wrong = str(int(task.metadata["value"]) + rng.choice([-5, -4, -3, 3, 4, 5]))
        return f"Configuration audit: variable {var} is currently provisioned as {wrong}."
    raise ValueError(f"unknown family {fam!r}")


def build_d2_tasks(d0_tasks: list[EobTask], seed: int) -> list[EobTask]:
    rng = random.Random(seed)
    out = []
    for i, d0 in enumerate(d0_tasks):
        content = _paraphrase_confirmation(d0, rng)
        ev_id = f"{d0.task_id}/confirm"
        task = EobTask(
            task_id=d0.task_id.replace("d0-", "d2-"), regime="D2_both_sufficient",
            family=d0.family, question=d0.question, answer=d0.answer,
            evidence=[{"evidence_id": ev_id, "content": content}],
            required_evidence_ids=[ev_id],
            metadata={**d0.metadata, "d0_source": d0.task_id})
        out.append(task)
    return out


def _distractor_value(family: str, content: str) -> str:
    """Extract the (wrong) value the distractor asserts, per family template,
    so it can be checked against the correct answer precisely -- substring
    checks are unreliable here (e.g. '42' is a substring of '142')."""
    if family in ("arithmetic", "restatement"):
        return content.rstrip(".").rsplit(" ", 1)[-1]
    if family == "comparison":
        return content.split(":")[1].strip().split(" holds")[0].strip()
    if family == "transform":
        return content.split("'")[1]
    raise ValueError(f"unknown family {family!r}")


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
            evidence=[{"evidence_id": ev_id, "content": content}],
            required_evidence_ids=[],  # the distractor is NOT required -- the answer never depended on it
            metadata={**d0.metadata, "d0_source": d0.task_id})
        out.append(task)
    return out
