#!/usr/bin/env python3
"""Central build contract for DAPH's staged validation pipeline.

This script intentionally keeps the repo's normalization and execution logic in
one place so that local development, CI, and manual qualification all use the
same stage names and contracts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, List

STAGES: Dict[str, Dict[str, object]] = {
    "install": {
        "label": "Stage 1: install / package setup",
        "description": "Prepare the local environment for development and standard validation.",
        "steps": [
            ["python", "-m", "pip", "install", "-e", ".[dev,pretrained]"],
        ],
    },
    "test": {
        "label": "Stage 2: fast validation",
        "description": "Run the deterministic unit suite and the frozen V2A boundary check.",
        "steps": [
            ["python", "scripts/check_v2a_qualified_boundary.py"],
            ["python", "-m", "pytest", "-q"],
        ],
    },
    "gate": {
        "label": "Stage 3: architecture and parity gates",
        "description": "Run the exact-regression and compute-ordering gates plus the synthetic Phase 0 retention gate.",
        "steps": [
            ["python", "-m", "pytest", "-q", "tests/test_qwen_exfusion_gate0b.py", "tests/test_effort_compute_ordering.py"],
            ["python", "-m", "pytest", "-q", "tests/test_hrm_control_plane.py", "tests/test_hrm_memory.py"],
            ["python", "scripts/run_phase0_retention.py", "--synthetic", "--shallow-continuation", "--output", "runs/phase0-ci"],
        ],
    },
    "package": {
        "label": "Stage 4: packaging / distribution",
        "description": "Build source and wheel distributions for release or distribution review.",
        "steps": [
            ["python", "-m", "build"],
        ],
    },
    "qualification": {
        "label": "Stage 5: research qualification (manual-only)",
        "description": "Run the expensive oracle-regeneration and benchmark qualification suite; this is intentionally not part of the PR path.",
        "steps": [
            ["python", "-m", "pytest", "-q", "tests/qualification"],
        ],
    },
    "ci-fast": {
        "label": "Fast path",
        "description": "Install + run the quick deterministic validation path used for ordinary PRs.",
        "stages": ["install", "test"],
    },
    "ci-full": {
        "label": "Release path",
        "description": "Install, validate, run architecture gates, then build release artifacts.",
        "stages": ["install", "test", "gate", "package"],
    },
}


def format_steps(steps: object) -> str:
    if not isinstance(steps, list):
        return "(no commands defined)"
    if all(isinstance(step, str) for step in steps):
        return f"Runs stages: {', '.join(steps)}"
    rendered: List[str] = []
    for step in steps:
        if isinstance(step, list):
            rendered.append(" ".join(step))
        else:
            rendered.append(str(step))
    return " && ".join(rendered)


def print_contract() -> None:
    print("DAPH build pipeline")
    print("===================")
    print("The build is intentionally staged so the PR path stays fast and the research qualification path remains explicit and manual.")
    print()
    for name, meta in STAGES.items():
        print(f"- {name}: {meta['label']}")
        print(f"  {meta['description']}")
        print(f"  Command: {format_steps(meta.get('steps') or meta.get('stages', []))}")
        print()
    print("Fast path: make install test")
    print("Release path: make install test gate package")
    print("Manual qualification: make qualification")


def run_stage(stage: str) -> int:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    meta = STAGES[stage]
    print(f"Running {stage}: {meta['label']}")
    if "stages" in meta:
        for child_stage in meta["stages"]:
            if run_stage(str(child_stage)) != 0:
                return 1
        return 0
    if "steps" not in meta:
        raise ValueError(f"Stage '{stage}' defines neither a 'steps' list nor child 'stages'.")

    for step in meta["steps"]:
        print(f"$ {' '.join(step)}")
        result = subprocess.run(step, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Organize the DAPH build pipeline into explicit stages for local use and CI.",
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        help="Run a specific pipeline stage.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the pipeline contract without executing a stage.",
    )
    args = parser.parse_args(argv)

    if args.list or args.stage is None:
        print_contract()
        return 0

    return run_stage(args.stage)


if __name__ == "__main__":
    sys.exit(main())
