"""The resume key bug this fixes: an earlier C4 patch resumed a run from
``packet_hash`` alone. packet_hash binds the final packet's content, but not
the mechanism that produced it -- a resume could silently reuse a stale
artifact after the prompt template changed, after a retrieval or selector
config changed, after a compressor/graph-parser config changed, after the
model revision changed, or after the source tree itself changed underneath
the run. A stale artifact being indistinguishable from a fresh one is exactly
the kind of provenance gap this whole project's discipline exists to close.

``ExecutionIdentity`` is the generic resume key every experimental arm should
bind to: task + arm + the full mechanism configuration + model + pipeline
version + source lineage, reduced to one canonical SHA-256. Two runs sharing
this hash are provably running the identical mechanism against the identical
input; two runs differing in ANY of these fields must never be treated as
resumable from each other's artifacts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ExecutionIdentity:
    """One canonical identity per (task, arm, mechanism, environment).

    Fields are deliberately explicit rather than a free-form dict: a resume
    key that silently drops a field when a caller forgets to pass it is worse
    than no resume key, since it looks safe while quietly not being safe.
    Config hashes are opaque strings (the caller's own canonical hash of
    whatever configuration object is relevant); this dataclass does not know
    or care what's inside a retrieval config vs a graph-compressor config, it
    only guarantees that if ANY of them differ, the overall identity differs.
    """
    task_id: str
    arm_id: str
    prompt_hash: str
    retrieval_config_hash: str
    selector_config_hash: str
    graph_compressor_config_hash: str
    model_revision: str
    pipeline_version: str
    source_commit: str
    #: Optional extension point for arm-specific config hashes (e.g. G2-v2's
    #: endpoint_recognizer_config_hash, relation_family_hash) that don't apply
    #: to every gate. Sorted before hashing so key order never matters.
    extra_config_hashes: dict[str, str] = field(default_factory=dict)

    def canonical_sha256(self) -> str:
        payload = asdict(self)
        payload["extra_config_hashes"] = dict(sorted(payload["extra_config_hashes"].items()))
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_receipt(self) -> dict[str, object]:
        """Every field plus the canonical hash, for embedding directly in a
        run's receipt/manifest -- not just the hash, so a human (or a future
        certifier) can see WHICH field changed without recomputing anything."""
        return {**asdict(self), "execution_identity_sha256": self.canonical_sha256()}


def resume_is_valid(previous: ExecutionIdentity, current: ExecutionIdentity) -> bool:
    """A resume is valid ONLY if every field matches -- not just packet_hash,
    not just task_id. Equivalent to comparing canonical_sha256(), spelled out
    as its own function so a call site reads as an intentional identity check
    rather than an incidental hash comparison."""
    return previous.canonical_sha256() == current.canonical_sha256()
