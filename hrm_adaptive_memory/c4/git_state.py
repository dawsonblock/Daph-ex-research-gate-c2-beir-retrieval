"""Git source-lineage checks, shared by the runner and the certifier.

One implementation, because two copies of "is the tree clean?" would drift and
this is the check that decides whether a result can be certified at all.

The distinction this module exists to make:

    SOURCE dirt      code, config, tests -> FATAL
                     the bundle cannot be tied to a revision
    EVIDENCE dirt    the run's own output -> EXPECTED
                     steps that freeze packets, replay determinism, dry-run and
                     smoke all rewrite tracked files under evidence/

Conflating the two makes ``VALID_RUN: true`` unreachable: a certifying run
necessarily rewrites tracked evidence, so an unscoped ``git status --porcelain``
is guaranteed non-empty by the time certification runs.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Paths whose contents define the source revision. These are hashed for the
# source snapshot and must be clean for certification.
SOURCE_PATHS = ("hrm_adaptive_memory", "daph", "scripts", "configs", "tests",
                "pyproject.toml")

# Paths a run legitimately writes to. Changes here are recorded, not fatal.
OUTPUT_PATHS = ("evidence",)


@dataclass
class GitState:
    """Observed git state of a working tree."""
    is_repo: bool = False
    head: str | None = None
    source_changes: list[str] = field(default_factory=list)
    output_changes: list[str] = field(default_factory=list)
    other_changes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def source_clean(self) -> bool:
        """True if nothing that defines the revision has been modified."""
        return self.is_repo and not self.source_changes and not self.other_changes

    def summary(self) -> dict:
        return {
            "is_repo": self.is_repo,
            "head": self.head,
            "source_clean": self.source_clean,
            "source_changes": self.source_changes[:20],
            "source_change_count": len(self.source_changes),
            "output_change_count": len(self.output_changes),
            "other_changes": self.other_changes[:20],
            "error": self.error,
        }


def _run_git(repo: Path, *args: str) -> tuple[int, str]:
    """Run git and return (returncode, raw stdout).

    stdout is NOT stripped: porcelain status encodes state in columns 1-2, so
    stripping the whole output would delete the leading space of the first
    entry and shift every path by one character.
    """
    try:
        proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, timeout=30)
        return proc.returncode, proc.stdout
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def _classify(entry: str) -> str:
    """Bucket a porcelain entry by the path it touches."""
    # Porcelain v1: 'XY path' or 'XY old -> new' for renames.
    path = entry[3:] if len(entry) > 3 else entry
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    path = path.strip().strip('"')
    top = path.split("/", 1)[0]
    if top in SOURCE_PATHS or path in SOURCE_PATHS:
        return "source"
    if top in OUTPUT_PATHS:
        return "output"
    return "other"


def inspect(repo: Path) -> GitState:
    """Inspect a working tree's revision and cleanliness, scoped by path."""
    state = GitState()

    code, out = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    if code != 0 or out.strip() != "true":
        state.error = f"not a git repository: {repo}"
        return state
    state.is_repo = True

    code, out = _run_git(repo, "rev-parse", "HEAD")
    if code != 0:
        state.error = f"cannot resolve HEAD: {out.strip()}"
        return state
    state.head = out.strip()

    code, out = _run_git(repo, "status", "--porcelain", "--untracked-files=all")
    if code != 0:
        state.error = f"git status failed: {out.strip()}"
        return state

    for line in out.splitlines():
        if not line.strip():
            continue
        bucket = _classify(line)
        if bucket == "source":
            state.source_changes.append(line)
        elif bucket == "output":
            state.output_changes.append(line)
        else:
            state.other_changes.append(line)
    return state


def revision_matches(head: str | None, expected: str) -> bool:
    """True if HEAD matches an expected revision.

    Accepts an abbreviated expected SHA, but requires at least 7 characters so
    a stray short string cannot match many commits.
    """
    if not head or not expected:
        return False
    expected = expected.strip().lower()
    head = head.strip().lower()
    if len(expected) < 7:
        return False
    return head.startswith(expected) if len(expected) < len(head) else head == expected
