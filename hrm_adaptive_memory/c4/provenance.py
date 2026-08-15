"""C4 result provenance — immutable hashing and manifest generation.

Every C4 experiment bundle must be self-describing. The manifest captures all
configuration that could affect results, and RESULTS.sha256 provides an
immutable hash of every artifact in the bundle.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_commit(repo: Path) -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    """Compute SHA256 of a JSON-serializable object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()).hexdigest()


def compute_results_hash(bundle_dir: Path) -> str:
    """Compute RESULTS.sha256 for a bundle directory.

    Hashes all .jsonl and .json files (excluding RESULTS.sha256 itself)
    in deterministic filename order.
    """
    files = sorted(
        f for f in bundle_dir.iterdir()
        if f.is_file() and f.name != "RESULTS.sha256"
        and f.suffix in (".jsonl", ".json")
    )
    lines = []
    for f in files:
        digest = _sha256_file(f)
        lines.append(f"{digest}  {f.name}")
    content = "\n".join(lines) + "\n"
    return content


def write_results_hash(bundle_dir: Path) -> str:
    """Write RESULTS.sha256 for a bundle directory. Returns the content."""
    content = compute_results_hash(bundle_dir)
    (bundle_dir / "RESULTS.sha256").write_text(content)
    return content


def verify_results_hash(bundle_dir: Path) -> bool:
    """Verify RESULTS.sha256 for a bundle directory."""
    hash_file = bundle_dir / "RESULTS.sha256"
    if not hash_file.exists():
        return False
    expected = hash_file.read_text()
    actual = compute_results_hash(bundle_dir)
    return expected == actual


# --- Bundle-level root hash (covers everything, including certification/) --

BUNDLE_HASH_FILENAME = "BUNDLE.sha256"

# Never hashed into the bundle root: it is itself the file being written, and
# unlike RESULTS.sha256 (scoped to raw results, verified as a distinct step
# earlier in the pipeline) this one has to be written strictly last.
_BUNDLE_HASH_EXCLUDE = frozenset({BUNDLE_HASH_FILENAME})


def compute_bundle_hash(bundle_dir: Path) -> str:
    """Compute a root hash covering EVERY file under a bundle directory.

    RESULTS.sha256 is deliberately narrow -- direct-child .jsonl/.json files
    only, verified as its own early step. It does not cover certification/
    (CERTIFICATION.json, ENVIRONMENT.lock, PROTOCOL.json, SOURCE_SNAPSHOT.json,
    source_snapshot.zip), so a bundle could pass every RESULTS.sha256 check
    while its lineage artifacts went unverified. This is the file meant to be
    the single root of trust: it walks the whole tree, recursively, and its
    own listing is sorted by relative path for a deterministic result
    independent of filesystem iteration order.
    """
    files = sorted(
        f for f in bundle_dir.rglob("*")
        if f.is_file() and f.name not in _BUNDLE_HASH_EXCLUDE
    )
    lines = []
    for f in files:
        digest = _sha256_file(f)
        rel = f.relative_to(bundle_dir).as_posix()
        lines.append(f"{digest}  {rel}")
    return "\n".join(lines) + "\n"


def write_bundle_hash(bundle_dir: Path) -> str:
    """Write BUNDLE.sha256. Call this LAST -- after every other artifact,
    including CERTIFICATION.json, already exists -- or it won't be covered."""
    content = compute_bundle_hash(bundle_dir)
    (bundle_dir / BUNDLE_HASH_FILENAME).write_text(content)
    return content


def verify_bundle_hash(bundle_dir: Path) -> bool:
    """Verify BUNDLE.sha256 for a bundle directory."""
    hash_file = bundle_dir / BUNDLE_HASH_FILENAME
    if not hash_file.exists():
        return False
    expected = hash_file.read_text()
    actual = compute_bundle_hash(bundle_dir)
    return expected == actual


def build_manifest(
    *,
    repo: Path,
    mode: str,
    split: str,
    arm_ids: list[str],
    task_count: int,
    protocol_sha256: str,
    task_corpus_sha256: str,
    evidence_corpus_sha256: str,
    hrm_model_id: str = "",
    hrm_model_revision: str = "",
    hrm_max_new_tokens: int = 0,
    bge_model_id: str = "",
    bge_revision: str = "",
    pooling: str = "",
    normalization: str = "",
    query_prefix: str = "",
    candidate_budget: int = 0,
    packet_budget: int = 0,
    rrf_k: int = 0,
    query_policy_version: str = "",
    bridge_policy_version: str = "",
    identity_policy_version: str = "",
    selector_version: str = "",
    selector_config_hash: str = "",
    torch_version: str = "",
    transformers_version: str = "",
    device: str = "",
    dtype: str = "",
    batch_size: int = 0,
    generation_condition: str = "",
) -> dict:
    """Build a self-describing manifest for a C4 experiment bundle."""
    # Auto-detect versions if not provided
    if not torch_version:
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            pass
    if not transformers_version:
        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            pass
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    return {
        "mode": mode,
        "split": split,
        "arm_ids": arm_ids,
        "task_count": task_count,
        "git_commit": _git_commit(repo),
        "protocol_sha256": protocol_sha256,
        "task_corpus_sha256": task_corpus_sha256,
        "evidence_corpus_sha256": evidence_corpus_sha256,
        "hrm_model_id": hrm_model_id,
        "hrm_model_revision": hrm_model_revision,
        "hrm_max_new_tokens": hrm_max_new_tokens,
        "bge_model_id": bge_model_id,
        "bge_revision": bge_revision,
        "pooling": pooling,
        "normalization": normalization,
        "query_prefix": query_prefix,
        "candidate_budget": candidate_budget,
        "packet_budget": packet_budget,
        "rrf_k": rrf_k,
        "query_policy_version": query_policy_version,
        "bridge_policy_version": bridge_policy_version,
        "identity_policy_version": identity_policy_version,
        "selector_version": selector_version,
        "selector_config_hash": selector_config_hash,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        "generation_condition": generation_condition,
        "python_version": sys.version.split()[0],
        "created_utc": _utc_now(),
    }


def _utc_now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_corpus(path: Path) -> str:
    """Compute SHA256 of a JSONL corpus file."""
    if not path.exists():
        return ""
    return _sha256_file(path)


# --- Phase 6: Canonical packet hashing ---

def canonical_packet_hash(packet: dict) -> str:
    """Compute SHA-256 of a canonical packet representation.

    The packet is serialized using deterministic JSON:
        json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    This gives clear boundaries for debugging:
        retrieval changed?    → candidate_pool_hash differs
        membership changed?   → membership_hash differs
        ordering changed?     → order_hash differs
        prompt changed?       → prompt_hash differs
    """
    canonical = json.dumps(
        packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_canonical_packet(
    *,
    task_id: str,
    query_hash: str,
    canonical_subject: str,
    candidate_pool_hash: str,
    selector_policy_id: str,
    ordered_selected_ids: list[str],
    ordered_text_sha256: list[str],
) -> dict:
    """Build a canonical packet representation for hashing.

    This is the frozen representation that gets hashed. Any change to
    the packet contents will produce a different hash.
    """
    return {
        "task_id": task_id,
        "query_hash": query_hash,
        "canonical_subject": canonical_subject,
        "candidate_pool_hash": candidate_pool_hash,
        "selector_policy_id": selector_policy_id,
        "ordered_selected_ids": list(ordered_selected_ids),
        "ordered_text_sha256": list(ordered_text_sha256),
    }


def hash_text(text: str) -> str:
    """Compute SHA-256 of a text string."""
    return hashlib.sha256(text.encode()).hexdigest()
