"""R13 answer judge: the only module allowed to import check_answer().

This module is the boundary between oracle information and the rest of the system.
No file under daph_x/operators/, daph_x/authority/, or daph_x/value/ should import
check_answer() directly.
"""
from __future__ import annotations

from daph_x.coding.reasoning_tasks import check_answer as _check_answer
from daph_x.operators.types import EvaluationLabels


def is_correct(answer: str, labels: EvaluationLabels) -> bool:
    """Check whether an answer matches the ground-truth label.

    This is the only allowed point of contact with the evaluator.
    All operator and router code must call this function (or the
    equivalent) only inside daph_x/evaluation/ modules.
    """
    return _check_answer(answer, labels.correct_answer, labels.answer_type)
