"""Core check for 'this run traces back to a certified development config'.

Used from two places that must never drift apart:

  * scripts/colab_c4_requalify.py -- checked EARLY (right after verifying the
    source revision, before installing dependencies or touching the GPU), so
    a mismatch aborts before any of the run's cost is spent, not after it.
  * scripts/certify_c4_run.py's development_lineage gate -- checked again at
    certification time as the authoritative, non-bypassable record. The
    launcher's early check is an optimization (fail fast); the gate is what
    actually decides VALID_RUN.

Both call this same function so the two checks cannot silently diverge.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import git_state
from .snapshot_drift import SnapshotLoadError, load_source_snapshot, verify_no_drift


@dataclass
class LineageCheck:
    """Result of comparing a run against a prior certified development run."""
    ok: bool
    violations: list[str] = field(default_factory=list)
    dev_protocol_sha256: str | None = None
    dev_commit: str | None = None
    this_protocol_sha256: str | None = None
    this_commit: str | None = None
    source_drift: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": self.violations,
            "development_protocol_sha256": self.dev_protocol_sha256,
            "development_commit": self.dev_commit,
            "this_run_protocol_sha256": self.this_protocol_sha256,
            "this_run_commit": self.this_commit,
            "source_drift_from_development": self.source_drift,
        }


def _check_ancestry(dev_commit: str, this_head: str | None,
                    repo: Path) -> list[str]:
    """Diagnose why dev_commit isn't recognized as this run's own HEAD.

    ``git merge-base --is-ancestor`` returns non-zero for two very different
    reasons, and conflating them produces a misleading diagnosis:

      1. dev_commit's object is genuinely absent from local history --
         usually a SHALLOW clone (``git fetch --depth N``), which truncates
         history at the fetch boundary. This is a clone-depth problem, not
         evidence of anything wrong with the actual commit graph. Found the
         hard way: syncing a runner with ``git fetch --depth 1`` after an
         initial full clone silently converts the repo shallow, and a
         5-commit-old dev_commit then reads as "not an ancestor" even though
         it plainly is on the real, remote history.
      2. dev_commit's object IS present locally, but merge-base still says
         it isn't an ancestor -- that is real: a rebase, force-push, or
         cherry-pick actually rewrote history instead of advancing it.

    Only case 2 is genuine divergence. Returns a list of violations (empty
    if the ancestry actually holds once the right check is used).
    """
    if not this_head:
        return ["this run has no resolvable HEAD"]

    has_commit = subprocess.run(
        ["git", "cat-file", "-e", dev_commit],
        cwd=repo, capture_output=True, text=True).returncode == 0

    if not has_commit:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=repo, capture_output=True, text=True).stdout.strip()
        if shallow == "true":
            return [
                f"cannot verify commit ancestry: development's certified "
                f"commit {dev_commit[:12]} is not present in this shallow "
                f"clone's local history. This is a clone-depth problem, not "
                f"necessarily divergence -- fetch full history "
                f"(git fetch --unshallow, or clone/fetch without --depth) "
                f"and retry."]
        return [
            f"development's certified commit {dev_commit[:12]} does not "
            f"exist anywhere in this repository's local history, and the "
            f"repository is not shallow -- this points to real divergence "
            f"(a different repository, or history that was rewritten), not "
            f"an incomplete clone."]

    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", dev_commit, this_head],
        cwd=repo, capture_output=True, text=True)
    if ancestor_check.returncode != 0:
        return [
            f"development's certified commit {dev_commit[:12]} exists in "
            f"this repository's history but is NOT an ancestor of this "
            f"run's commit {this_head[:12]} -- history was rewritten "
            f"(rebase, force-push, or cherry-pick) rather than simply "
            f"advanced."]
    return []


def check_development_lineage(
    *, dev_certification_path: Path, this_protocol_sha256: str | None,
    repo: Path,
) -> LineageCheck:
    """Verify this run's protocol matches, and descends from, a certified
    development run.

    Two hard requirements (both must hold for ``ok`` to be true):

      1. protocol_sha256 equality with what development certified. This is
         the granularity 'no development-side changes' is enforced at: any
         mechanism change requires a new protocol version by this project's
         own convention, and protocol_validation.py cross-checks the live
         code against the declared protocol on every run. Tooling changes
         that don't touch the mechanism do not require a new protocol
         version and are therefore allowed.
      2. development's certified commit must be an ancestor of this run's
         commit -- history was not rewritten or reset backward.

    A full file-level drift report against development's SOURCE_SNAPSHOT.json
    is always attached for visibility, but never itself decides ``ok``.
    """
    violations: list[str] = []

    if not dev_certification_path.is_file():
        return LineageCheck(
            ok=False,
            violations=[
                f"development certification not found: "
                f"{dev_certification_path}. A non-development split cannot "
                f"prove it matches a certified development configuration "
                f"that does not exist."])

    try:
        dev_cert = json.loads(dev_certification_path.read_text())
    except json.JSONDecodeError as exc:
        return LineageCheck(
            ok=False,
            violations=[f"development certification is not valid JSON: {exc}"])

    if not dev_cert.get("VALID_RUN"):
        violations.append(
            "development certification has VALID_RUN=false; a later split "
            "cannot be pinned to a configuration that was never certified")

    dev_protocol_sha = dev_cert.get("protocol_sha256")
    dev_commit = ((dev_cert.get("gates") or {}).get("source_lineage", {})
                  .get("detail", {}).get("git", {}).get("head"))

    if not dev_protocol_sha or not this_protocol_sha256:
        violations.append(
            "cannot compare protocol_sha256: missing from development "
            "certificate or from this run")
    elif this_protocol_sha256 != dev_protocol_sha:
        violations.append(
            f"protocol_sha256 differs from development: this run used "
            f"{str(this_protocol_sha256)[:16]}, development certified "
            f"{str(dev_protocol_sha)[:16]}. A protocol change requires the "
            f"development split to be recertified under the new protocol "
            f"version before a later split can use it.")

    state = git_state.inspect(repo)
    if not state.is_repo:
        violations.append(f"{state.error}: cannot verify commit lineage")
    elif not dev_commit:
        violations.append("development certificate does not record a commit hash")
    elif not git_state.revision_matches(state.head, dev_commit):
        violations.extend(_check_ancestry(dev_commit, state.head, repo))

    drift_summary: dict[str, Any] | None = None
    try:
        dev_snapshot = load_source_snapshot(
            dev_certification_path.parent / "SOURCE_SNAPSHOT.json")
        drift_summary = verify_no_drift(dev_snapshot, repo).summary()
    except SnapshotLoadError as exc:
        drift_summary = {"error": str(exc)}

    return LineageCheck(
        ok=not violations,
        violations=violations,
        dev_protocol_sha256=dev_protocol_sha,
        dev_commit=dev_commit,
        this_protocol_sha256=this_protocol_sha256,
        this_commit=state.head,
        source_drift=drift_summary,
    )
