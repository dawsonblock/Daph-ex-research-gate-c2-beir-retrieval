"""Versioned operator prompts for R13-A v2.

Every prompt has an ID and a SHA-256 hash. Tournament receipts store
prompt_id and prompt_sha256 so that treatment conditions are auditable.
"""
from __future__ import annotations

import hashlib
from typing import Mapping


BASE_PROMPT_TEMPLATE = (
    "{task_prompt}\n\n"
    "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
)

CRITIQUE_PROMPT_ID = "critique-r13-v2"
CRITIQUE_PROMPT_TEMPLATE = (
    "Problem: {task_prompt}\n\n"
    "Current proposed answer: {current_answer}\n\n"
    "Current reasoning:\n{current_trace}\n\n"
    "Analyze this reasoning and answer carefully. Respond in valid JSON only, with no additional prose:\n"
    "{{\n"
    '  "verdict": "ERROR | NO_ERROR | UNCERTAIN",\n'
    '  "failure_mode": "ARITHMETIC | ALGEBRA | LOGIC | COMBINATORICS | SEQUENCE | INSTRUCTION | EXTRACTION | OTHER",\n'
    '  "first_error": "<specific location and nature of the first error, or empty>",\n'
    '  "explanation": "<why the answer is wrong or why it is correct>",\n'
    '  "repair_instruction": "<what the solver should do differently to avoid the error>",\n'
    '  "confidence": 0.0\n'
    "}}"
)

CRITIQUE_RETRY_PROMPT_ID = "critique-retry-r13-v2"
CRITIQUE_RETRY_PROMPT_TEMPLATE = (
    "Problem: {task_prompt}\n\n"
    "A previous attempt produced the answer '{current_answer}' but contained this error:\n"
    "{error_description}\n\n"
    "Instruction for the repair: {repair_instruction}\n\n"
    "Solve the problem again, following the repair instruction. "
    "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
)

VERIFY_PROMPT_ID = "verify-r13-v2"
VERIFY_PROMPT_TEMPLATE = (
    "Problem: {task_prompt}\n\n"
    "Proposed answer: {current_answer}\n\n"
    "Proposed reasoning:\n{current_trace}\n\n"
    "Verify this answer carefully. Respond in valid JSON only, with no additional prose:\n"
    "{{\n"
    '  "verdict": "CORRECT | INCORRECT | INCONCLUSIVE",\n'
    '  "confidence": 0.0,\n'
    '  "checks": [\n'
    '    {{"check_type": "<check name>", "result": "PASS | FAIL", "evidence": "<explanation>"}}\n'
    '  ],\n'
    '  "failure_mode": "ARITHMETIC | ALGEBRA | LOGIC | COMBINATORICS | SEQUENCE | INSTRUCTION | EXTRACTION | OTHER | NONE",\n'
    '  "correction_needed": true | false,\n'
    '  "correction_suggestion": "<what a correct solution would need, or empty>"\n'
    "}}"
)

VERIFY_CORRECT_PROMPT_ID = "verify-correct-r13-v2"
VERIFY_CORRECT_PROMPT_TEMPLATE = (
    "Problem: {task_prompt}\n\n"
    "The proposed answer '{current_answer}' has been verified as correct. "
    "Restate the answer clearly. "
    "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
)

VERIFY_INCORRECT_PROMPT_ID = "verify-incorrect-r13-v2"
VERIFY_INCORRECT_PROMPT_TEMPLATE = (
    "Problem: {task_prompt}\n\n"
    "The proposed answer '{current_answer}' has failed verification for this reason:\n"
    "{verification_result}\n\n"
    "Solve the problem correctly. "
    "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
)

DIVERSE_PROMPT_SET_ID = "diverse-r13-v2"

DIVERSE_STRATEGIES = {
    "backward": (
        "Solve this problem by working backwards from what the answer must satisfy.\n\n"
        "{task_prompt}\n\n"
        "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
    ),
    "formal_derivation": (
        "Solve this problem using a formal step-by-step derivation. Show each algebraic or logical step explicitly.\n\n"
        "{task_prompt}\n\n"
        "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
    ),
    "alternative_representation": (
        "Solve this problem using a different representation or approach than the obvious one. Try substitution, a concrete example, or an alternative formula.\n\n"
        "{task_prompt}\n\n"
        "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
    ),
    "problem_classification": (
        "First identify what type of problem this is, then apply the standard method for that type.\n\n"
        "{task_prompt}\n\n"
        "Think step by step. At the very end, write 'Answer: <your answer>' on a new line."
    ),
}

PROMPT_REGISTRY: Mapping[str, str] = {
    CRITIQUE_PROMPT_ID: CRITIQUE_PROMPT_TEMPLATE,
    CRITIQUE_RETRY_PROMPT_ID: CRITIQUE_RETRY_PROMPT_TEMPLATE,
    VERIFY_PROMPT_ID: VERIFY_PROMPT_TEMPLATE,
    VERIFY_CORRECT_PROMPT_ID: VERIFY_CORRECT_PROMPT_TEMPLATE,
    VERIFY_INCORRECT_PROMPT_ID: VERIFY_INCORRECT_PROMPT_TEMPLATE,
    **{f"{DIVERSE_PROMPT_SET_ID}:{k}": v for k, v in DIVERSE_STRATEGIES.items()},
}


def prompt_hash(prompt_id: str) -> str:
    text = PROMPT_REGISTRY.get(prompt_id, "")
    return hashlib.sha256(text.encode()).hexdigest()
