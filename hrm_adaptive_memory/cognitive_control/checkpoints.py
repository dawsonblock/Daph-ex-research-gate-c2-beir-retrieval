"""Signed cognitive-history checkpoints rooted in an external signer registry."""
from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from .core import CognitiveControlStore


CHECKPOINT_SCHEMA = "DAPH_COGNITIVE_CHECKPOINT_V2"
TRUSTED_SIGNERS_SCHEMA = "DAPH_COGNITIVE_TRUSTED_SIGNERS_V1"
_FROZEN_SIGNER_STATUSES = {"FROZEN_FOR_EXPERIMENT", "FROZEN_FOR_QUALIFICATION"}


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


def _git_identity(repository_root: str | Path) -> tuple[str, str]:
    root = Path(repository_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("checkpoint repository must resolve Git HEAD and tree") from error
    if len(commit) != 40 or len(tree) != 40:
        raise ValueError("checkpoint Git identity is malformed")
    return commit, tree


@dataclass(frozen=True)
class TrustedSigner:
    signer_id: str
    key_id: str
    public_key: str

    def __post_init__(self) -> None:
        if not self.signer_id or not self.key_id:
            raise ValueError("trusted signer_id and key_id are required")
        try:
            if len(base64.b64decode(self.public_key, validate=True)) != 32:
                raise ValueError("expected Ed25519 raw public key")
        except (ValueError, TypeError, binascii.Error) as error:
            raise ValueError("trusted signer public_key must be base64 Ed25519 bytes") from error


class TrustedSignerRegistry:
    """External trust root; checkpoint payload keys are never trusted by themselves."""

    def __init__(self, signers: Iterable[TrustedSigner], *, status: str):
        entries = tuple(signers)
        self._signers = {(entry.signer_id, entry.key_id): entry for entry in entries}
        if len(self._signers) != len(entries):
            raise ValueError("trusted signer id/key id pairs must be unique")
        self.status = status

    def require_active(self) -> None:
        if self.status not in _FROZEN_SIGNER_STATUSES:
            raise ValueError("trusted signer registry is not frozen for checkpoint verification")

    def resolve(self, signer_id: str, key_id: str) -> TrustedSigner | None:
        return self._signers.get((signer_id, key_id))


def load_trusted_signers(path: str | Path) -> TrustedSignerRegistry:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != TRUSTED_SIGNERS_SCHEMA:
        raise ValueError("unsupported trusted signer registry schema")
    status = payload.get("status")
    if not isinstance(status, str):
        raise ValueError("trusted signer registry status is required")
    return TrustedSignerRegistry(
        (TrustedSigner(**entry) for entry in payload.get("signers", ())), status=status)


@dataclass(frozen=True)
class CognitiveCheckpoint:
    schema: str
    event_count: int
    head_event_hash: str
    manifest_sha256: str
    created_at: str
    source_commit: str
    source_tree_hash: str
    signer_id: str
    key_id: str
    signature_algorithm: str
    signature: str

    def unsigned(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}

    def signing_payload(self) -> bytes:
        return _json(self.unsigned())

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, object]) -> "CognitiveCheckpoint":
        return cls(**value)


class Ed25519CheckpointSigner:
    """Local signer; the caller must separately register its public key as trusted."""

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


def create_checkpoint(store: CognitiveControlStore, *, repository_root: str | Path,
                      signer_id: str, key_id: str, signer: Ed25519CheckpointSigner,
                      created_at: str | None = None) -> CognitiveCheckpoint:
    """Sign the persisted history head and resolved Git commit/tree identity."""
    if not signer_id or not key_id:
        raise ValueError("signer_id and key_id are required")
    if not store.manifest_path.is_file():
        raise ValueError("cognitive-control manifest is absent")
    manifest = json.loads(store.manifest_path.read_text())
    if manifest.get("event_count") != len(store.events):
        raise ValueError("cognitive-control manifest does not match loaded history")
    source_commit, source_tree_hash = _git_identity(repository_root)
    provisional = CognitiveCheckpoint(
        schema=CHECKPOINT_SCHEMA,
        event_count=len(store.events),
        head_event_hash=(store.events[-1]["event_hash"] if store.events else "GENESIS"),
        manifest_sha256=_sha256(store.manifest_path.read_bytes()),
        created_at=_canonical_timestamp(created_at),
        source_commit=source_commit,
        source_tree_hash=source_tree_hash,
        signer_id=signer_id,
        key_id=key_id,
        signature_algorithm="Ed25519",
        signature="",
    )
    return CognitiveCheckpoint(**{**provisional.to_json(),
                                  "signature": signer.sign(provisional.signing_payload())})


def verify_checkpoint(checkpoint: CognitiveCheckpoint, trusted_signers: TrustedSignerRegistry) -> bool:
    """Verify using a key resolved from an external frozen trust root."""
    if (checkpoint.schema != CHECKPOINT_SCHEMA or checkpoint.event_count < 0
            or checkpoint.signature_algorithm != "Ed25519"):
        return False
    try:
        trusted_signers.require_active()
        signer = trusted_signers.resolve(checkpoint.signer_id, checkpoint.key_id)
        if signer is None:
            return False
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(signer.public_key, validate=True))
        public_key.verify(base64.b64decode(checkpoint.signature, validate=True), checkpoint.signing_payload())
    except (ValueError, TypeError, binascii.Error, InvalidSignature):
        return False
    return True


def write_git_checkpoint(repo_root: str | Path, checkpoint: CognitiveCheckpoint, *,
                         trusted_signers: TrustedSignerRegistry,
                         relative_path: str | Path) -> Path:
    """Write a verified checkpoint suitable for a subsequent Git review/commit."""
    if not verify_checkpoint(checkpoint, trusted_signers):
        raise ValueError("refusing to write a checkpoint without a trusted signature")
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
