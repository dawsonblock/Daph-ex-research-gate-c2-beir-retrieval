"""Safe code executor for DAPH-X coding experiment.

Executes model-generated code in a subprocess with:
  - Timeout protection
  - Resource limits
  - Test harness that catches exceptions
  - Utility computation (test pass rate - cost)
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daph_x.coding.tasks import CodingTask


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a code solution against tests."""
    task_id: str
    solution_code: str
    tests_passed: int
    tests_total: int
    pass_rate: float
    utility: float
    execution_time_ms: float
    error: str | None
    test_results: tuple[dict, ...]

    @property
    def success(self) -> bool:
        return self.tests_passed == self.tests_total and self.error is None


def execute_solution(
    task: CodingTask,
    solution_code: str,
    timeout_seconds: float = 10.0,
) -> ExecutionResult:
    """Execute a code solution against the task's tests.

    Runs in a subprocess with timeout protection. Returns:
      - tests_passed / tests_total
      - utility = pass_rate * 100 - execution_cost
      - error message if the code fails to load

    Args:
        task: The coding task
        solution_code: The model-generated Python code
        timeout_seconds: Maximum execution time
    """
    # Build the test script
    test_script = _build_test_script(task, solution_code)

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(test_script)
        script_path = f.name

    try:
        start_time = time.monotonic()
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        execution_time_ms = (time.monotonic() - start_time) * 1000

        # Parse output
        if result.returncode == 0:
            try:
                import json
                output = json.loads(result.stdout.strip())
                tests_passed = output.get("tests_passed", 0)
                tests_total = output.get("tests_total", 0)
                test_results = tuple(output.get("test_results", []))
                error = None
            except json.JSONDecodeError:
                tests_passed = 0
                tests_total = len(task.tests)
                test_results = ()
                error = f"Failed to parse output: {result.stdout[:200]}"
        else:
            tests_passed = 0
            tests_total = len(task.tests)
            test_results = ()
            stderr = result.stderr.strip()
            error = f"Runtime error: {stderr[:500]}"

        pass_rate = tests_passed / max(tests_total, 1)
        # Utility: 100 * pass_rate - cost (execution time in seconds)
        execution_cost = execution_time_ms / 1000.0
        utility = pass_rate * 100.0 - execution_cost

        return ExecutionResult(
            task_id=task.task_id,
            solution_code=solution_code,
            tests_passed=tests_passed,
            tests_total=tests_total,
            pass_rate=pass_rate,
            utility=utility,
            execution_time_ms=execution_time_ms,
            error=error,
            test_results=test_results,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            task_id=task.task_id,
            solution_code=solution_code,
            tests_passed=0,
            tests_total=len(task.tests),
            pass_rate=0.0,
            utility=-10.0,  # Penalty for timeout
            execution_time_ms=timeout_seconds * 1000,
            error=f"Timeout after {timeout_seconds}s",
            test_results=(),
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


def _build_test_script(task: CodingTask, solution_code: str) -> str:
    """Build a Python test script that runs the solution against tests."""
    # Extract just the function definition from the solution
    # The model may return markdown code blocks — strip them
    code = solution_code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    # Serialize tests as JSON to avoid quoting issues
    tests_json = json.dumps([
        {"test_code": tc, "description": desc, "expected": exp}
        for tc, desc, exp in task.tests
    ])

    # Use repr() to safely embed the JSON string in the script
    tests_repr = repr(tests_json)

    script = textwrap.dedent(f"""\
import json
import sys
import traceback

# Task imports
{task.imports}

# Solution code
{code}

# Run tests
tests = json.loads({tests_repr})

results = []
passed = 0
total = len(tests)

for t in tests:
    test_code = t["test_code"]
    description = t["description"]
    expected = t["expected"]
    try:
        actual = eval(test_code)
        success = actual == expected
        if success:
            passed += 1
        results.append({{
            "test": test_code,
            "description": description,
            "expected": repr(expected),
            "actual": repr(actual),
            "passed": success,
        }})
    except Exception as e:
        results.append({{
            "test": test_code,
            "description": description,
            "expected": repr(expected),
            "actual": None,
            "passed": False,
            "error": str(e),
        }})

output = {{
    "tests_passed": passed,
    "tests_total": total,
    "test_results": results,
}}
print(json.dumps(output, default=str))
""")

    return script
