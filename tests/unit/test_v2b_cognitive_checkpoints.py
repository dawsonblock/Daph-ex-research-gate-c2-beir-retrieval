"""External checkpoint contracts for the cognitive-control hash chain."""
from __future__ import annotations

from dataclasses import replace
import json

from hrm_adaptive_memory.cognitive_control import (
    CognitiveControlStore, Ed25519CheckpointSigner, create_checkpoint,
    verify_checkpoint, write_git_checkpoint)


def _store(tmp_path):
    store = CognitiveControlStore(tmp_path / "store")
    store.record_provenance(
        entity_id="claim", entity_type="claim", payload={"value": 6},
        activity_id="test", agent_id="daph", agent_type="software",
        source_id="evidence", created_at="2026-08-12T00:00:00Z",
        operation_id="provenance")
    return store


def test_ed25519_checkpoint_binds_event_head_manifest_and_source_commit(tmp_path):
    checkpoint = create_checkpoint(
        _store(tmp_path), source_commit="63faa7aed070e9a6920b0a512a7d6366c5676bed",
        signer_id="test-signer", signer=Ed25519CheckpointSigner.generate(),
        created_at="2026-08-12T01:00:00+00:00")
    assert checkpoint.created_at == "2026-08-12T01:00:00.000000Z"
    assert verify_checkpoint(checkpoint)
    assert not verify_checkpoint(replace(checkpoint, head_event_hash="altered"))


def test_checkpoint_write_is_repo_bounded_and_serializes_verifiable_data(tmp_path):
    checkpoint = create_checkpoint(
        _store(tmp_path), source_commit="63faa7aed070e9a6920b0a512a7d6366c5676bed",
        signer_id="test-signer", signer=Ed25519CheckpointSigner.generate())
    root = tmp_path / "repo"; root.mkdir()
    target = write_git_checkpoint(root, checkpoint, relative_path="evidence/v2b/checkpoint.json")
    assert verify_checkpoint(type(checkpoint).from_json(json.loads(target.read_text())))
