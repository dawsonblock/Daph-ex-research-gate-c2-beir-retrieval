"""Signed, externally storable checkpoints for cognitive event histories."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.exceptions import InvalidSignature

from .core import CognitiveControlStore


CHECKPOINT_SCHEMA = "DAPH_COGNITIVE_CHECKPOINT_V1"


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_timestamp(value: str | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("checkpoint created_at must include a timezone")
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class CognitiveCheckpoint:
    schema: str
    event_count: int
    head_event_hash: str
    manifest_sha256: str
    created_at: str
    source_commit: str
    signer_id: str
    public_key: str
    signature: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items()
                if key not in {"public_key", "signature"}}

    def signing_payload(self) -> bytes:
        return _json(self.unsigned())

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "CognitiveCheckpoint":
        return cls(**value)


class Ed25519CheckpointSigner:
    """Explicit local signer; private key material is never written by this module."""

    def __init__(self, private_key: Ed25519PrivateKey):
        self._private_key = private_key

    @classmethod
    def generate(cls) -> "Ed25519CheckpointSigner":
        return cls(Ed25519PrivateKey.generate())

    def public_key_b64(self) -> str:
        raw = self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")

    def sign(self, payload: bytes) -> str:
        return base64.b64encode(self._private_key.sign(payload)).decode("ascii")


def create_checkpoint(store: CognitiveControlStore, *, source_commit: str,
                      signer_id: str, signer: Ed25519CheckpointSigner,
                      created_at: str | None = None) -> CognitiveCheckpoint:
    """Sign the current persisted head without modifying the event history."""
    if not source_commit or not signer_id:
        raise ValueError("source_commit and signer_id are required")
    if not store.manifest_path.is_file():
        raise ValueError("cognitive-control manifest is absent")
    manifest = json.loads(store.manifest_path.read_text())
    if manifest.get("event_count") != len(store.events):
        raise ValueError("cognitive-control manifest does not match loaded history")
    provisional = CognitiveCheckpoint(
        schema=CHECKPOINT_SCHEMA,
        event_count=len(store.events),
        head_event_hash=(store.events[-1]["event_hash"] if store.events else "GENESIS"),
        manifest_sha256=_sha256(store.manifest_path.read_bytes()),
        created_at=_canonical_timestamp(created_at),
        source_commit=source_commit,
        signer_id=signer_id,
        public_key=signer.public_key_b64(),
        signature="",
    )
    return CognitiveCheckpoint(**{**provisional.to_json(),
                                  "signature": signer.sign(provisional.signing_payload())})


def verify_checkpoint(checkpoint: CognitiveCheckpoint) -> bool:
    if checkpoint.schema != CHECKPOINT_SCHEMA or checkpoint.event_count < 0:
        return False
    try:
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(checkpoint.public_key))
        public_key.verify(base64.b64decode(checkpoint.signature), checkpoint.signing_payload())
    except (ValueError, TypeError, InvalidSignature):
        return False
    return True


def write_git_checkpoint(repo_root: str | Path, checkpoint: CognitiveCheckpoint, *,
                         relative_path: str | Path) -> Path:
    """Write a checkpoint suitable for subsequent Git review and commit.

    The caller chooses when to commit/push; the function only writes inside
    the supplied repository root and rejects path traversal.
    """
    root = Path(repo_root).resolve()
    target = (root / relative_path).resolve()
    if root not in target.parents:
        raise ValueError("checkpoint path must remain inside the repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_json(checkpoint.to_json()) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target
