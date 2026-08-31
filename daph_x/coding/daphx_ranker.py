"""DAPH-X ranker for coding candidates.

Maps coding task state to an epistemic graph and uses Q_MB + Q_res
to rank candidate solutions. The ranking is independent of the model's
own preference — DAPH-X evaluates each candidate based on:

  Q_MB: model-based heuristic (code structure, complexity, edge case coverage)
  Q_res: learned residual correction (if available)

The base action is the model's first candidate (temperature=0, standard prompt).
The DAPH-X action is the candidate with highest Q_X = Q_MB + Q_res.

When they disagree, we fork: execute both and measure ΔU.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import CodingTask
from daph_x.coding.model_interface import ModelCallResult


@dataclass(frozen=True)
class CandidateRanking:
    """DAPH-X's ranking of a candidate solution."""
    candidate_id: str
    q_mb: float
    q_res: float
    q_x: float
    rank: int
    # Pre-decision features
    features: dict[str, float]


@dataclass(frozen=True)
class ForkDecision:
    """A fork decision between base and DAPH-X actions."""
    task_id: str
    base_candidate_id: str
    daphx_candidate_id: str
    base_q_x: float
    daphx_q_x: float
    disagreement: bool
    base_ranking: CandidateRanking
    daphx_ranking: CandidateRanking


def extract_code_features(code: str, task: CodingTask) -> dict[str, float]:
    """Extract pre-execution features from candidate code.

    These features are available BEFORE running the code, so they
    can be used by DAPH-X to rank candidates without executing them.

    Features:
      - Code length (lines, chars)
      - Has type hints
      - Has docstring
      - Number of if/else branches
      - Number of loops
      - Number of try/except blocks
      - Uses recursion
      - Has return statement
      - Complexity estimate (cyclomatic)
      - Edge case indicators (checks for empty, None, etc.)
      - Prompt variant (temperature)
    """
    feats = {}

    if not code or not code.strip():
        feats["code_lines"] = 0
        feats["code_chars"] = 0
        feats["has_type_hints"] = 0.0
        feats["has_docstring"] = 0.0
        feats["n_branches"] = 0
        feats["n_loops"] = 0
        feats["n_try_except"] = 0
        feats["uses_recursion"] = 0.0
        feats["has_return"] = 0.0
        feats["complexity"] = 0
        feats["checks_empty"] = 0.0
        feats["checks_none"] = 0.0
        feats["checks_len"] = 0.0
        feats["has_assert"] = 0.0
        feats["n_function_calls"] = 0
        feats["uses_sorted"] = 0.0
        feats["uses_set"] = 0.0
        feats["uses_dict"] = 0.0
        feats["uses_list_comp"] = 0.0
        feats["parse_error"] = 1.0
        return feats

    feats["parse_error"] = 0.0

    # Basic metrics
    lines = code.strip().split("\n")
    feats["code_lines"] = len(lines)
    feats["code_chars"] = len(code)

    # Parse AST
    try:
        tree = ast.parse(code)
        feats["parse_error"] = 0.0

        # Count constructs
        n_branches = 0
        n_loops = 0
        n_try_except = 0
        n_function_calls = 0
        has_return = False
        uses_recursion = False

        func_name = task.function_name

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                n_branches += 1
            elif isinstance(node, (ast.For, ast.While)):
                n_loops += 1
            elif isinstance(node, ast.Try):
                n_try_except += 1
            elif isinstance(node, ast.Return):
                has_return = True
            elif isinstance(node, ast.Call):
                n_function_calls += 1
                # Check for recursion
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    uses_recursion = True
                elif isinstance(node.func, ast.Attribute):
                    pass

        feats["n_branches"] = n_branches
        feats["n_loops"] = n_loops
        feats["n_try_except"] = n_try_except
        feats["n_function_calls"] = n_function_calls
        feats["uses_recursion"] = 1.0 if uses_recursion else 0.0
        feats["has_return"] = 1.0 if has_return else 0.0

        # Cyclomatic complexity estimate
        feats["complexity"] = 1 + n_branches + n_loops + n_try_except

        # Check for type hints
        has_type_hints = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is not None:
                    has_type_hints = True
                for arg in node.args.args:
                    if arg.annotation is not None:
                        has_type_hints = True
        feats["has_type_hints"] = 1.0 if has_type_hints else 0.0

        # Check for docstring
        has_docstring = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                    has_docstring = True
        feats["has_docstring"] = 1.0 if has_docstring else 0.0

    except SyntaxError:
        feats["parse_error"] = 1.0
        feats["n_branches"] = 0
        feats["n_loops"] = 0
        feats["n_try_except"] = 0
        feats["n_function_calls"] = 0
        feats["uses_recursion"] = 0.0
        feats["has_return"] = 0.0
        feats["complexity"] = 0
        feats["has_type_hints"] = 0.0
        feats["has_docstring"] = 0.0

    # Text-based features (don't require valid AST)
    code_lower = code.lower()
    feats["checks_empty"] = 1.0 if any(
        x in code_lower for x in ["not ", "== ''", "== []", "len(", "if not", "is empty"]
    ) else 0.0
    feats["checks_none"] = 1.0 if "none" in code_lower else 0.0
    feats["checks_len"] = 1.0 if "len(" in code_lower else 0.0
    feats["has_assert"] = 1.0 if "assert " in code_lower else 0.0
    feats["uses_sorted"] = 1.0 if "sorted(" in code_lower else 0.0
    feats["uses_set"] = 1.0 if "set(" in code_lower else 0.0
    feats["uses_dict"] = 1.0 if "dict(" in code_lower or "{}" in code else 0.0
    feats["uses_list_comp"] = 1.0 if "[" in code and "for " in code and " in " in code else 0.0

    return feats


def compute_q_mb(features: dict[str, float], task: CodingTask) -> float:
    """Compute model-based Q value for a candidate.

    This is a heuristic that estimates code quality from static features
    WITHOUT executing the code. It can be wrong — that's the point.

    Q_MB logic:
      - Parse errors are heavily penalized
      - More branches = better edge case handling (up to a point)
      - Type hints and docstrings add small value
      - Edge case checks (empty, None, len) add value
      - Excessive complexity is penalized
      - Recursion on hard problems is rewarded slightly
    """
    if features.get("parse_error", 0.0) == 1.0:
        return -50.0

    q = 50.0  # Base value

    # Edge case handling
    q += features.get("checks_empty", 0.0) * 10.0
    q += features.get("checks_none", 0.0) * 5.0
    q += features.get("checks_len", 0.0) * 5.0

    # Branch coverage
    n_branches = features.get("n_branches", 0)
    q += min(n_branches, 5) * 5.0  # Up to 5 branches rewarded
    if n_branches > 10:
        q -= (n_branches - 10) * 2.0  # Penalize excessive branching

    # Error handling
    q += features.get("n_try_except", 0) * 5.0

    # Code quality
    q += features.get("has_type_hints", 0.0) * 3.0
    q += features.get("has_docstring", 0.0) * 2.0

    # Complexity penalty (too simple or too complex)
    complexity = features.get("complexity", 0)
    if complexity < 2:
        q -= 5.0  # Too simple — might miss edge cases
    elif complexity > 15:
        q -= 10.0  # Too complex — likely bugs

    # Recursion for hard problems
    if task.difficulty == "hard" and features.get("uses_recursion", 0.0):
        q += 5.0

    # Has return
    q += features.get("has_return", 0.0) * 3.0

    # Penalize very short solutions (likely incomplete)
    if features.get("code_lines", 0) < 3:
        q -= 10.0

    return q


def rank_candidates(
    candidates: list[ModelCallResult],
    task: CodingTask,
    q_res_model=None,
    q_res_feature_keys: list[str] | None = None,
) -> list[CandidateRanking]:
    """Rank candidates using Q_MB + Q_res.

    Args:
        candidates: List of model-generated candidates
        task: The coding task
        q_res_model: Optional trained Q_res model (joblib-loaded)
        q_res_feature_keys: Feature keys for the Q_res model

    Returns:
        List of CandidateRanking sorted by Q_X (descending)
    """
    rankings = []

    for candidate in candidates:
        features = extract_code_features(candidate.solution_code, task)
        q_mb = compute_q_mb(features, task)

        # Q_res: learned correction (if model available)
        q_res = 0.0
        if q_res_model is not None and q_res_feature_keys is not None:
            # Build feature vector in the expected order
            x = []
            for k in q_res_feature_keys:
                x.append(float(features.get(k, 0.0)))
            import numpy as np
            q_res = float(q_res_model.predict(np.array([x]))[0])

        q_x = q_mb + q_res

        rankings.append(CandidateRanking(
            candidate_id=candidate.candidate_id,
            q_mb=q_mb,
            q_res=q_res,
            q_x=q_x,
            rank=0,  # Will be set after sorting
            features=features,
        ))

    # Sort by Q_X descending and assign ranks
    rankings.sort(key=lambda r: r.q_x, reverse=True)
    ranked = []
    for i, r in enumerate(rankings):
        ranked.append(CandidateRanking(
            candidate_id=r.candidate_id,
            q_mb=r.q_mb,
            q_res=r.q_res,
            q_x=r.q_x,
            rank=i + 1,
            features=r.features,
        ))

    return ranked


def identify_fork(
    candidates: list[ModelCallResult],
    rankings: list[CandidateRanking],
) -> ForkDecision:
    """Identify the fork between base action and DAPH-X action.

    Base action: the model's first candidate (temperature=0, standard prompt)
    DAPH-X action: the candidate with highest Q_X

    If they're the same, disagreement = False.
    """
    # Base action: first candidate (index 0, temp=0, standard prompt)
    base_candidate = candidates[0]
    base_id = base_candidate.candidate_id

    # DAPH-X action: highest Q_X
    daphx_ranking = rankings[0]
    daphx_id = daphx_ranking.candidate_id

    # Find base ranking
    base_ranking = None
    for r in rankings:
        if r.candidate_id == base_id:
            base_ranking = r
            break
    if base_ranking is None:
        base_ranking = rankings[0]

    disagreement = base_id != daphx_id

    return ForkDecision(
        task_id=base_candidate.task_id,
        base_candidate_id=base_id,
        daphx_candidate_id=daphx_id,
        base_q_x=base_ranking.q_x,
        daphx_q_x=daphx_ranking.q_x,
        disagreement=disagreement,
        base_ranking=base_ranking,
        daphx_ranking=daphx_ranking,
    )
