"""Executive-training split generator: the ANSWER_NOW-viable family.

Per the research-lead directive after the privileged-parsing finding on
EOB-v1/v2's D0/D2/D3: Executive v0's training data must use ONLY b3-native
task phrasing, verified against the REAL, unmodified extract_subject/
extract_target_relation (hrm_adaptive_memory/c4/query_stage.py) -- zero
bypass, zero synthetic identity injection.

The memory-required half of the training split reuses b3_calibration_v1
sampling unchanged (see scripts/build_exec_training_suite.py) -- that
generation problem is already solved. This module builds the OTHER half:
zero-evidence, general-knowledge questions in the exact b3 template ("What
is the {relation} for {subject}?"), which the real parser handles natively
(empirically verified) BECAUSE subject/relation extraction operate on the
question text alone and never touch grammar_v4's entity extractor -- that
extractor is only invoked on EVIDENCE content, and these tasks carry none.

Every fact is a well-established, stable, unambiguous fact (country
capitals, chemical element symbols) -- hand-curated and hand-verified, not
procedurally generated, since these need to be actually TRUE rather than
merely internally consistent. A verification pass re-checks the SAME
subject/relation extraction the real HRM pipeline will use, for every
generated task, before any task is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- hand-verified fact tables ---------------------------------------------
# Deliberately limited to stable, unambiguous, widely-known facts. Excludes
# capitals that have changed recently or are disputed, and excludes obscure
# elements where the "well-known" assumption might not hold for a language
# model's parametric knowledge.

_CAPITALS: tuple[tuple[str, str], ...] = (
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Germany", "Berlin"), ("Canada", "Ottawa"),
    ("Egypt", "Cairo"), ("China", "Beijing"), ("Russia", "Moscow"),
    ("Brazil", "Brasilia"), ("India", "Delhi"), ("Mexico", "Mexico City"),
    ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Norway", "Oslo"),
    ("Sweden", "Stockholm"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Ireland", "Dublin"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Turkey", "Ankara"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"),
    ("Cuba", "Havana"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"),
    ("Morocco", "Rabat"), ("Iceland", "Reykjavik"), ("Hungary", "Budapest"),
    ("Romania", "Bucharest"), ("Ukraine", "Kyiv"), ("Indonesia", "Jakarta"),
)

_ELEMENT_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Carbon", "C"),
    ("Nitrogen", "N"), ("Oxygen", "O"), ("Fluorine", "F"), ("Neon", "Ne"),
    ("Sodium", "Na"), ("Magnesium", "Mg"), ("Aluminum", "Al"), ("Silicon", "Si"),
    ("Phosphorus", "P"), ("Sulfur", "S"), ("Chlorine", "Cl"), ("Potassium", "K"),
    ("Calcium", "Ca"), ("Iron", "Fe"), ("Copper", "Cu"), ("Zinc", "Zn"),
    ("Silver", "Ag"), ("Tin", "Sn"), ("Iodine", "I"), ("Gold", "Au"),
    ("Mercury", "Hg"), ("Lead", "Pb"), ("Nickel", "Ni"), ("Titanium", "Ti"),
    ("Chromium", "Cr"), ("Manganese", "Mn"), ("Cobalt", "Co"), ("Platinum", "Pt"),
    ("Uranium", "U"), ("Barium", "Ba"), ("Krypton", "Kr"), ("Xenon", "Xe"),
)

_RELATION_BY_DOMAIN = {"capital": "capital city", "element": "chemical symbol"}


@dataclass
class ExecTrainingTask:
    task_id: str
    domain: str  # "capital" | "element"
    question: str
    answer: str
    subject: str
    metadata: dict = field(default_factory=dict)


def build_answer_now_tasks() -> list[ExecTrainingTask]:
    """Build the full ANSWER_NOW-viable family from the hand-verified fact
    tables. No sampling/randomness -- the entire curated set is used, since
    it is small and every fact was individually chosen for reliability."""
    tasks: list[ExecTrainingTask] = []
    for i, (subject, answer) in enumerate(_CAPITALS, 1):
        relation = _RELATION_BY_DOMAIN["capital"]
        question = f"What is the {relation} for {subject}?"
        tasks.append(ExecTrainingTask(
            task_id=f"exec-gk-capital-{i:04d}", domain="capital",
            question=question, answer=answer, subject=subject,
            metadata={"relation": relation}))
    for i, (subject, answer) in enumerate(_ELEMENT_SYMBOLS, 1):
        relation = _RELATION_BY_DOMAIN["element"]
        question = f"What is the {relation} for {subject}?"
        tasks.append(ExecTrainingTask(
            task_id=f"exec-gk-element-{i:04d}", domain="element",
            question=question, answer=answer, subject=subject,
            metadata={"relation": relation}))
    return tasks


class ParserVerificationError(ValueError):
    """A generated task's question did not natively round-trip through the
    REAL, unmodified extract_subject/extract_target_relation. Fail closed --
    this is exactly the ambiguity this module exists to eliminate."""


def verify_native_parsing(tasks: list[ExecTrainingTask]) -> None:
    """Re-derive subject and relation from each task's question using the
    ACTUAL certified query_stage functions (no bypass) and assert they match
    the task's own declared subject/relation exactly. Raises on any
    disagreement -- the whole point of this family is that it requires zero
    bypass, so a mismatch here is a hard build failure, not a warning."""
    import hrm_adaptive_memory.evaluation  # noqa: F401  (cycle-breaker)
    from hrm_adaptive_memory.c4.query_stage import extract_subject, extract_target_relation

    for t in tasks:
        got_subject = extract_subject(t.question)
        got_relation = extract_target_relation(t.question)
        if got_subject != t.subject:
            raise ParserVerificationError(
                f"{t.task_id}: extract_subject({t.question!r}) = {got_subject!r}, "
                f"expected {t.subject!r} -- native parsing failed")
        if got_relation != t.metadata["relation"]:
            raise ParserVerificationError(
                f"{t.task_id}: extract_target_relation({t.question!r}) = {got_relation!r}, "
                f"expected {t.metadata['relation']!r} -- native parsing failed")
