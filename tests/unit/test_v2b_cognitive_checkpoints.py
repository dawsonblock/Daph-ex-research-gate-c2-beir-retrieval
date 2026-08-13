"""Externally trusted checkpoint contracts for the cognitive-control hash chain."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from hrm_adaptive_memory.cognitive_control import (
    CognitiveControlStore, Ed25519CheckpointSigner, TrustedSigner,
    TrustedSignerRegistry, create_checkpoint, verify_checkpoint, write_git_checkpoint)


ROOT = Path(__file__).parents[2]


def _store(tmp_path):
    store = CognitiveControlStore(tmp_path / "store")
    store.record_provenance(
        entity_id="claim", entity_type="claim", payload={"value": 6},
        activity_id="test", agent_id="daph", agent_type="software",
        source_id="evidence", created_at="2026-08-12T00:00:00Z",
        operation_id="provenance")
    return store


def _trust(signer):
    return TrustedSignerRegistry((TrustedSigner("test-signer", "test-key", signer.public_key_b64()),),
                                 status="FROZEN_FOR_EXPERIMENT")


def test_ed25519_checkpoint_binds_event_head_manifest_and_resolved_git_commit_tree(tmp_path):
    signer = Ed25519CheckpointSigner.generate()
    checkpoint = create_checkpoint(
        _store(tmp_path), repository_root=ROOT, signer_id="test-signer", key_id="test-key",
        signer=signer, created_at="2026-08-12T01:00:00+00:00")
    assert checkpoint.created_at == "2026-08-12T01:00:00.000000Z"
    assert len(checkpoint.source_commit) == len(checkpoint.source_tree_hash) == 40
    assert verify_checkpoint(checkpoint, _trust(signer))
    assert not verify_checkpoint(replace(checkpoint, head_event_hash="altered"), _trust(signer))


def test_checkpoint_cannot_choose_its_own_trust_key_or_signer_identity(tmp_path):
    signer, attacker = Ed25519CheckpointSigner.generate(), Ed25519CheckpointSigner.generate()
    checkpoint = create_checkpoint(_store(tmp_path), repository_root=ROOT, signer_id="test-signer",
                                   key_id="test-key", signer=signer)
    forged_unsigned = replace(checkpoint, source_commit="f" * 40, signature="")
    forged = replace(forged_unsigned, signature=attacker.sign(forged_unsigned.signing_payload()))
    assert not verify_checkpoint(forged, _trust(signer))


def test_checkpoint_write_requires_external_trust_and_serializes_verifiable_data(tmp_path):
    signer = Ed25519CheckpointSigner.generate()
    trusted = _trust(signer)
    checkpoint = create_checkpoint(_store(tmp_path), repository_root=ROOT, signer_id="test-signer",
                                   key_id="test-key", signer=signer)
    root = tmp_path / "repo"; root.mkdir()
    target = write_git_checkpoint(root, checkpoint, trusted_signers=trusted,
                                  relative_path="evidence/v2b/checkpoint.json")
    loaded = type(checkpoint).from_json(json.loads(target.read_text()))
    assert verify_checkpoint(loaded, trusted)
