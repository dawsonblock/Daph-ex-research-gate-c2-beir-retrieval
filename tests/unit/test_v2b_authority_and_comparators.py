"""V2B fail-closed authority and relation-typed comparison contracts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hrm_adaptive_memory.external_verification.authority_registry import (
    AUTHORITY_NOT_REGISTERED, AuthorityDefinition, AuthorityNotRegistered,
    AuthorityRegistry, FrozenExtractor, RegisteredAuthorityAcquirer, load_authority_registry)
from hrm_adaptive_memory.external_verification.comparators import (
    ComparisonKind, ComparisonOutcome, RelationSchema, ValueType,
    default_comparator_registry)
from hrm_adaptive_memory.external_verification.core import AcquisitionRequest, SourceType
from hrm_adaptive_memory.external_verification.network import PeerBoundResponse


EXTRACTOR_SHA = hashlib.sha256(b"world-bank-country-v1").hexdigest()
ROOT = Path(__file__).parents[2]
REGISTRY = AuthorityRegistry((AuthorityDefinition(
    authority_id="world_bank_country_v1", publisher="World Bank",
    domains=("api.worldbank.org",), relations=("capital_city", "population"),
    endpoint_patterns=("/v2/country/*",),
    source_type="AUTHORITATIVE_STRUCTURED_DATA", extractor_id="world_bank_country_v1",
    extractor_sha256=EXTRACTOR_SHA, schema_id="world_bank_country_schema_v1"),))


def test_authority_registry_binds_authority_relation_domain_and_endpoint():
    definition = REGISTRY.resolve(
        authority_id="world_bank_country_v1", relation="population",
        source_uri="https://api.worldbank.org/v2/country/CAN?format=json")
    assert definition.publisher == "World Bank"
    assert REGISTRY.identity()["sha256"]
    with pytest.raises(AuthorityNotRegistered, match=AUTHORITY_NOT_REGISTERED):
        REGISTRY.resolve(authority_id="world_bank_country_v1", relation="mass",
                         source_uri="https://api.worldbank.org/v2/country/CAN")
    with pytest.raises(AuthorityNotRegistered, match=AUTHORITY_NOT_REGISTERED):
        REGISTRY.resolve(authority_id="world_bank_country_v1", relation="population",
                         source_uri="https://example.test/v2/country/CAN")


def test_typed_comparators_do_not_fall_back_to_generic_string_equality():
    comparators = default_comparator_registry()
    integer = comparators.compare(
        RelationSchema("atomic_number", ValueType.INTEGER), "6.0", 6)
    assert integer.outcome is ComparisonOutcome.MATCH
    invalid_integer = comparators.compare(
        RelationSchema("atomic_number", ValueType.INTEGER), "six", 6)
    assert invalid_integer.outcome is ComparisonOutcome.INCONCLUSIVE

    mass = comparators.compare(
        RelationSchema("mass", ValueType.QUANTITY, canonical_unit="kg"),
        "1 kg", "1000 g")
    assert mass.outcome is ComparisonOutcome.MATCH
    tolerance = comparators.compare(
        RelationSchema("measurement", ValueType.DECIMAL, ComparisonKind.TOLERANCE,
                       absolute_tolerance="0.01"), "1.00", "1.009")
    assert tolerance.outcome is ComparisonOutcome.MATCH
    date = comparators.compare(
        RelationSchema("launch_date", ValueType.DATE), "2026-01-01", "2026-01-02")
    assert date.outcome is ComparisonOutcome.MISMATCH


def test_registered_authority_path_pins_extractor_instead_of_trusting_caller_labels():
    class Transport:
        def fetch(self, uri):
            return PeerBoundResponse(uri, uri, 200, {"content-type": "application/json"},
                                     b'{"entity":"Canada","population":1}', "8.8.8.8")

    extractor = FrozenExtractor(
        "world_bank_country_v1", EXTRACTOR_SHA,
        lambda raw, content_type: {"entity": "Canada", "population": 1})
    acquirer = RegisteredAuthorityAcquirer(REGISTRY, Transport(), (extractor,))
    result = acquirer.acquire(
        AcquisitionRequest("https://api.worldbank.org/v2/country/CAN",
                           source_type=SourceType.UNTRUSTED_CAPTURE_ONLY),
        authority_id="world_bank_country_v1", relation="population")
    assert result.status.value == "SUCCESS"
    assert result.request.source_type is SourceType.AUTHORITATIVE_STRUCTURED_DATA
    assert result.request.request_metadata["authority_id"] == "world_bank_country_v1"


def test_frozen_authority_registry_verifies_its_extractor_source_hash():
    registry = load_authority_registry(ROOT / "configs/authority_registry_v2b.json", repository_root=ROOT)
    assert registry.resolve(
        authority_id="world_bank_country_v1", relation="capital_city",
        source_uri="https://api.worldbank.org/v2/country/CAN").extractor_id == "world_bank_country_v1"
