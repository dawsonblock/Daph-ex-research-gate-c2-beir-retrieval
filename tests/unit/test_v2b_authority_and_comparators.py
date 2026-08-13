"""V2B authority contracts and relation-bound comparison semantics."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hrm_adaptive_memory.external_verification.authority_registry import (
    AUTHORITY_NOT_REGISTERED, AuthorityNotRegistered, RegisteredAuthorityAcquirer,
    RegistryStatus, load_authority_registry)
from hrm_adaptive_memory.external_verification.comparators import (
    ComparisonKind, ComparisonOutcome, RelationSchema, ValueType,
    default_comparator_registry)
from hrm_adaptive_memory.external_verification.core import (AcquisitionRequest, EvidenceStore,
                                                            SourceType)
from hrm_adaptive_memory.external_verification.network import PeerBoundResponse
from hrm_adaptive_memory.external_verification.typed_verifier import (
    RelationSchemaRegistry, TypedFieldVerifier)


ROOT = Path(__file__).parents[2]
SOURCE_URI = "https://api.worldbank.org/v2/country/CA?format=json"


def _frozen_registry(tmp_path):
    payload = json.loads((ROOT / "configs/authority_registry_v2b.json").read_text())
    payload["status"] = RegistryStatus.FROZEN_FOR_EXPERIMENT.value
    path = tmp_path / "authority_registry.json"
    path.write_text(json.dumps(payload))
    return load_authority_registry(path, repository_root=ROOT)


class _Transport:
    def __init__(self, final_uri: str = SOURCE_URI):
        self.final_uri = final_uri
        self.checked: list[str] = []

    def fetch(self, uri: str, *, uri_validator=None):
        if uri_validator is not None:
            uri_validator(uri); self.checked.append(uri)
            uri_validator(self.final_uri); self.checked.append(self.final_uri)
        body = b'[{"page":1},[{"name":"Canada","capitalCity":"Ottawa"}]]'
        return PeerBoundResponse(uri, self.final_uri, 200, {"content-type": "application/json"},
                                 body, "8.8.8.8")


def test_authority_registry_binds_authority_relation_endpoint_and_query_contract(tmp_path):
    registry = _frozen_registry(tmp_path)
    definition = registry.resolve(authority_id="world_bank_country_v1", relation="capital_city",
                                  source_uri=SOURCE_URI)
    assert definition.publisher == "World Bank"
    assert registry.identity()["sha256"]
    with pytest.raises(AuthorityNotRegistered, match=AUTHORITY_NOT_REGISTERED):
        registry.resolve(authority_id="world_bank_country_v1", relation="mass", source_uri=SOURCE_URI)
    with pytest.raises(AuthorityNotRegistered, match=AUTHORITY_NOT_REGISTERED):
        registry.resolve(authority_id="world_bank_country_v1", relation="capital_city",
                         source_uri="https://api.worldbank.org/v2/country/CAN?callback=evil")


def test_typed_comparators_have_frozen_compatible_contracts_and_symmetric_tolerance():
    comparators = default_comparator_registry()
    integer = comparators.compare(RelationSchema("atomic_number", ValueType.INTEGER), "6.0", 6)
    assert integer.outcome is ComparisonOutcome.MATCH
    invalid_integer = comparators.compare(RelationSchema("atomic_number", ValueType.INTEGER), "six", 6)
    assert invalid_integer.outcome is ComparisonOutcome.INCONCLUSIVE

    mass = comparators.compare(RelationSchema("mass", ValueType.QUANTITY, canonical_unit="kg"),
                               "1 kg", "1000 g")
    assert mass.outcome is ComparisonOutcome.MATCH
    tolerance = RelationSchema("measurement", ValueType.DECIMAL, ComparisonKind.TOLERANCE,
                               relative_tolerance="0.1")
    assert comparators.compare(tolerance, "100", "110").outcome is ComparisonOutcome.MATCH
    assert comparators.compare(tolerance, "110", "100").outcome is ComparisonOutcome.MATCH
    with pytest.raises(ValueError, match="does not support"):
        RelationSchema("count", ValueType.INTEGER, ComparisonKind.TOLERANCE,
                       absolute_tolerance="1")
    with pytest.raises(ValueError, match="finite nonnegative"):
        RelationSchema("mass", ValueType.QUANTITY, ComparisonKind.TOLERANCE,
                       canonical_unit="kg", absolute_tolerance="-0.1")
    with pytest.raises(ValueError, match="does not support"):
        RelationSchema("tags", ValueType.SET_MEMBERSHIP)


def test_registered_authority_loads_only_verified_module_symbol_and_attests_provenance(tmp_path):
    registry = _frozen_registry(tmp_path)
    transport = _Transport()
    acquirer = RegisteredAuthorityAcquirer(registry, transport, repository_root=ROOT)
    result = acquirer.acquire(
        AcquisitionRequest(SOURCE_URI, source_type=SourceType.UNTRUSTED_CAPTURE_ONLY),
        authority_id="world_bank_country_v1", relation="capital_city")
    assert result.status.value == "SUCCESS"
    assert result.request.source_type is SourceType.AUTHORITATIVE_STRUCTURED_DATA
    assert result.extracted_fields == {"entity": "Canada", "capital_city": "Ottawa"}
    assert result.authority_attestation["extractor_symbol"] == "world_bank_country_v1"
    assert result.authority_attestation["registry_sha256"] == registry.identity()["sha256"]
    assert transport.checked == [SOURCE_URI, SOURCE_URI]


def test_registered_authority_rejects_an_unregistered_redirect_even_if_transport_returns_json(tmp_path):
    registry = _frozen_registry(tmp_path)
    transport = _Transport("https://evil.example/v2/country/CA")
    result = RegisteredAuthorityAcquirer(registry, transport, repository_root=ROOT).acquire(
        AcquisitionRequest(SOURCE_URI), authority_id="world_bank_country_v1", relation="capital_city")
    assert result.status.value == "INVALID_RESPONSE"
    assert AUTHORITY_NOT_REGISTERED in result.detail


def test_typed_verifier_requires_a_valid_registered_authority_attestation(tmp_path):
    registry = _frozen_registry(tmp_path)
    acquirer = RegisteredAuthorityAcquirer(registry, _Transport(), repository_root=ROOT)
    result = acquirer.acquire(AcquisitionRequest(SOURCE_URI), authority_id="world_bank_country_v1",
                              relation="capital_city")
    evidence = EvidenceStore(tmp_path / "evidence").append_acquisition(
        result, claim_record_id="claim-1", acquisition_method=acquirer.ACQUISITION_METHOD,
        acquisition_version=acquirer.ACQUISITION_VERSION)
    claim = SimpleNamespace(record_id="claim-1", canonical_entity="Canada",
                            canonical_relation="capital_city", value="Ottawa")
    verifier = TypedFieldVerifier(
        RelationSchemaRegistry((RelationSchema("capital_city", ValueType.CANONICAL_STRING),)), registry)
    assert verifier.verify(claim, evidence).result.value == "SUPPORTED"
    assert verifier.verify(claim, replace(evidence, authority_attestation={})).reason_code == (
        "AUTHORITY_ATTESTATION_MISSING")


def test_development_registry_cannot_start_truth_bearing_acquisition():
    registry = load_authority_registry(ROOT / "configs/authority_registry_v2b.json", repository_root=ROOT)
    with pytest.raises(ValueError, match="not frozen"):
        RegisteredAuthorityAcquirer(registry, _Transport(), repository_root=ROOT)
