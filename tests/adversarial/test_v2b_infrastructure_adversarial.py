"""Adversarial fail-closed checks for V2B infrastructure."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.cognitive_control.actions import validate_v2b_action
from hrm_adaptive_memory.cognitive_control.qualification import validate_run_configuration
from hrm_adaptive_memory.cognitive_control.metareasoning_qualification import validate_i3_configuration
from hrm_adaptive_memory.external_verification.authority_registry import load_authority_registry


ROOT = Path(__file__).parents[2]


def test_v2b_rejects_specialist_spawning_before_that_action_is_qualified():
    with pytest.raises(ValueError, match="outside the frozen V2B action space"):
        validate_v2b_action("SPAWN_SPECIALIST")


def test_v2b_identity_rejects_placeholder_experiment_configuration():
    configuration = json.loads((ROOT / "experiments/v2b/configs/v2b_infrastructure_only.json").read_text())
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_run_configuration(configuration)


def test_v2b_identity_rejects_i2_development_harness_as_scientific_qualification():
    configuration = json.loads((ROOT / "experiments/v2b/configs/v2b_i2_development.json").read_text())
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_run_configuration(configuration)


def test_v2b_identity_rejects_i3_development_protocol_as_scientific_qualification():
    configuration = json.loads((ROOT / "experiments/v2b/configs/v2b_i3_development.json").read_text())
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_run_configuration(configuration)
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_i3_configuration(configuration)


def test_registry_rejects_tampered_extractor_pin(tmp_path):
    source = json.loads((ROOT / "configs/authority_registry_v2b.json").read_text())
    source["definitions"][0]["extractor_sha256"] = "0" * 64
    path = tmp_path / "registry.json"; path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="pinned hash"):
        load_authority_registry(path, repository_root=ROOT)
