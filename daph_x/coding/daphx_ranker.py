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

    # ── Deeper correctness signals (v2) ──
    # These features attempt to measure algorithmic correctness rather than
    # just surface-level code quality.

    # Check for explicit return of expected type
    feats["returns_none_explicitly"] = 1.0 if "return None" in code_lower else 0.0
    feats["returns_empty"] = 1.0 if 'return ""' in code or "return []" in code or "return {}" in code else 0.0

    # Check for early return patterns (guard clauses)
    feats["has_early_return"] = 1.0 if code_lower.count("return") > 1 else 0.0

    # Check for explicit base cases (common in DP/recursion)
    feats["has_base_case"] = 1.0 if any(
        x in code_lower for x in ["if n == 0", "if not ", "if len(", "if n == 1", "if i == 0"]
    ) else 0.0

    # Check for DP table initialization
    feats["has_dp_table"] = 1.0 if any(
        x in code_lower for x in ["dp =", "dp[", "memo", "f = [[", "table = ["]
    ) else 0.0
    # Check for proper DP base case initialization (dp[i][0] = 1, dp[0][j] = j, etc.)
    feats["has_dp_init"] = 1.0 if any(
        x in code_lower for x in ["dp[i][0]", "dp[0][j]", "dp[0][0]", "dp[0] = 0", "dp[0] = 1", "for i in range(m + 1)", "for j in range(n + 1)"]
    ) else 0.0

    # Check for stack/deque usage (needed for certain algorithms)
    feats["uses_stack"] = 1.0 if "stack" in code_lower else 0.0
    feats["uses_deque"] = 1.0 if "deque" in code_lower else 0.0
    feats["uses_heap"] = 1.0 if "heap" in code_lower or "heappush" in code_lower or "heappop" in code_lower else 0.0

    # Check for proper iteration (for vs while)
    feats["uses_while"] = 1.0 if "while " in code_lower else 0.0
    feats["uses_for"] = 1.0 if "for " in code_lower else 0.0
    feats["uses_range"] = 1.0 if "range(" in code_lower else 0.0

    # Check for common bug patterns
    feats["has_float_division"] = 1.0 if "/" in code and "//" not in code else 0.0
    feats["uses_int_division"] = 1.0 if "//" in code else 0.0
    feats["has_off_by_one_risk"] = 1.0 if any(
        x in code_lower for x in ["range(1,", "range(0,", "[-1]", "[1:]", "[:1]"]
    ) else 0.0

    # Check for explicit edge case handling for empty/None/zero
    feats["handles_empty_list"] = 1.0 if any(
        x in code_lower for x in ["if not arr", "if not nums", "if not s", "if not str", "if not matrix", "if len(arr) == 0", "if len(nums) == 0"]
    ) else 0.0
    feats["handles_single_element"] = 1.0 if any(
        x in code_lower for x in ["if len(", "== 1:", "n == 1", "len() == 1"]
    ) else 0.0

    # Check for algorithm-appropriate patterns based on task keywords
    # (These are general patterns, not task-specific)
    feats["uses_two_pointers"] = 1.0 if any(
        x in code_lower for x in ["left", "right", "l,", "r,", "i, j", "lo, hi"]
    ) else 0.0
    feats["uses_backtracking"] = 1.0 if "backtrack" in code_lower else 0.0
    feats["uses_bfs"] = 1.0 if "queue" in code_lower or "bfs" in code_lower else 0.0
    feats["uses_dfs"] = 1.0 if "dfs" in code_lower or "visited" in code_lower else 0.0

    # Check for proper comparison operators
    feats["uses_max"] = 1.0 if "max(" in code_lower else 0.0
    feats["uses_min"] = 1.0 if "min(" in code_lower else 0.0
    feats["uses_abs"] = 1.0 if "abs(" in code_lower else 0.0

    # Code structure balance
    n_lines = feats.get("code_lines", 0)
    n_branches = feats.get("n_branches", 0)
    n_loops = feats.get("n_loops", 0)
    # Lines per branch ratio (too many lines per branch = likely complex logic)
    feats["lines_per_branch"] = float(n_lines / max(n_branches, 1))
    # Branch-to-loop ratio
    feats["branch_to_loop_ratio"] = float(n_branches / max(n_loops, 1))

    return feats


def compute_q_mb(features: dict[str, float], task: CodingTask) -> float:
    """Compute model-based Q value for a candidate (v2).

    Improved heuristic that weights CORRECTNESS SIGNALS over surface features.
    The previous version over-weighted edge-case checks, which caused it to
    prefer candidates with more checks but incorrect algorithms.

    v2 logic:
      - Parse errors: heavy penalty (correctness)
      - Has return statement: required (correctness)
      - Algorithm-appropriate patterns: rewarded (correctness)
      - Edge case handling: moderate reward (correctness)
      - Code quality (type hints, docstrings): small reward (quality)
      - Excessive complexity: penalized (bug risk)
      - Too simple for hard tasks: penalized (likely incomplete)
      - Common bug patterns: penalized (correctness)
    """
    if features.get("parse_error", 0.0) == 1.0:
        return -50.0

    q = 50.0  # Base value

    # ── CORRECTNESS SIGNALS (high weight) ──

    # Must have a return statement
    if features.get("has_return", 0.0) == 0.0:
        q -= 20.0  # No return = likely broken

    # Algorithm patterns appropriate for the task
    # These are general patterns, not task-specific hacks
    if features.get("has_dp_table", 0.0):
        q += 12.0  # DP table suggests correct algorithm structure
    if features.get("has_dp_init", 0.0):
        q += 10.0  # Proper DP initialization is a strong correctness signal
    if features.get("uses_two_pointers", 0.0):
        q += 8.0  # Two-pointer is often the right approach
    if features.get("uses_stack", 0.0):
        q += 6.0  # Stack-based solutions are often correct
    if features.get("uses_backtracking", 0.0):
        q += 8.0  # Backtracking for combinatorial problems
    if features.get("uses_bfs", 0.0) or features.get("uses_dfs", 0.0):
        q += 6.0  # Graph traversal patterns
    if features.get("uses_heap", 0.0):
        q += 5.0  # Heap for priority-based algorithms

    # Base case handling (critical for recursion/DP)
    if features.get("has_base_case", 0.0):
        q += 6.0  # Explicit base cases are a correctness signal, but not always correct

    # Edge case handling (moderate weight — can be misleading)
    if features.get("handles_empty_list", 0.0):
        q += 3.0  # Empty input handling (lower weight — can be wrong)
    if features.get("handles_single_element", 0.0):
        q += 2.0  # Single element edge case

    # ── BUG RISK SIGNALS (penalties) ──

    # Float division where integer might be expected
    if features.get("has_float_division", 0.0) and not features.get("uses_int_division", 0.0):
        q -= 3.0  # Potential precision issue

    # Too many lines per branch = complex logic, higher bug risk
    lpb = features.get("lines_per_branch", 0.0)
    if lpb > 10:
        q -= 5.0  # Very long branches are error-prone

    # ── CODE QUALITY (low weight) ──

    # Type hints and docstrings are minor quality signals
    q += features.get("has_type_hints", 0.0) * 2.0
    q += features.get("has_docstring", 0.0) * 1.0

    # ── COMPLEXITY BALANCE ──

    complexity = features.get("complexity", 0)
    n_branches = features.get("n_branches", 0)

    # For hard tasks, some complexity is expected and good
    if task.difficulty == "hard":
        if complexity < 3:
            q -= 8.0  # Too simple for a hard problem — likely incomplete
        elif complexity > 20:
            q -= 6.0  # Excessively complex — bug risk
        elif 5 <= complexity <= 15:
            q += 5.0  # Sweet spot for hard problems
    else:
        if complexity < 2:
            q -= 5.0
        elif complexity > 15:
            q -= 8.0

    # Moderate branch count is good (not too few, not too many)
    if 2 <= n_branches <= 8:
        q += 4.0
    elif n_branches > 12:
        q -= 4.0

    # ── TASK-SPECIFIC ALGORITHM HINTS ──
    # These are general algorithm-category hints, not task-specific hacks.
    # They help Q_MB recognize when a candidate uses the right algorithm type.

    task_desc_lower = task.description.lower()

    # DP problems
    if any(kw in task_desc_lower for kw in ["dynamic", "dp", "minimum", "maximum", "count", "number of ways", "optimal"]):
        if features.get("has_dp_table", 0.0) or features.get("uses_recursion", 0.0):
            q += 5.0  # Right algorithm family

    # Graph problems
    if any(kw in task_desc_lower for kw in ["graph", "island", "course", "schedule", "topological"]):
        if features.get("uses_bfs", 0.0) or features.get("uses_dfs", 0.0):
            q += 5.0

    # String matching problems
    if any(kw in task_desc_lower for kw in ["palindrome", "substring", "pattern", "match", "anagram"]):
        if features.get("uses_two_pointers", 0.0):
            q += 3.0

    # Stack problems
    if any(kw in task_desc_lower for kw in ["stack", "parentheses", "valid", "rectangle", "histogram"]):
        if features.get("uses_stack", 0.0):
            q += 5.0

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
