"""Frozen authority definitions for controlled, truth-bearing acquisition.

Generic HTTP remains untrusted capture. A relation can become authoritative
only when the registry binds its endpoint, every redirect, the exact extractor
module bytes, the extractor symbol, and the response schema.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit


AUTHORITY_NOT_REGISTERED = "AUTHORITY_NOT_REGISTERED"


class AuthorityNotRegistered(ValueError):
    code = AUTHORITY_NOT_REGISTERED


class RegistryStatus(str, Enum):
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    FROZEN_FOR_EXPERIMENT = "FROZEN_FOR_EXPERIMENT"
    FROZEN_FOR_QUALIFICATION = "FROZEN_FOR_QUALIFICATION"
    RETIRED = "RETIRED"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def canonical_extracted_fields_sha256(fields: Mapping[str, object]) -> str:
    """Return the deterministic digest of an extractor's structured result."""
    return _sha256(_canonical_json(dict(fields)))


@dataclass(frozen=True)
class AuthorityDefinition:
    authority_id: str
    publisher: str
    domains: tuple[str, ...]
    relations: tuple[str, ...]
    endpoint_patterns: tuple[str, ...]
    allowed_query_keys: tuple[str, ...]
    required_query_values: tuple[tuple[str, str], ...]
    source_type: str
    extractor_id: str
    extractor_module: str
    extractor_symbol: str
    extractor_sha256: str
    schema_id: str
    content_types: tuple[str, ...] = ("application/json",)
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.authority_id or not self.publisher or not self.extractor_id or not self.schema_id:
            raise ValueError("authority id, publisher, extractor id, and schema id are required")
        if (not self.domains or not self.relations or not self.endpoint_patterns or not self.content_types
                or any(not domain or domain != domain.lower().rstrip(".") for domain in self.domains)):
            raise ValueError("authority domains, relations, and endpoint patterns are required")
        if (not self.extractor_module or Path(self.extractor_module).is_absolute()
                or ".." in Path(self.extractor_module).parts
                or not self.extractor_symbol.isidentifier()):
            raise ValueError("authority extractor module and symbol must be repository-local identifiers")
        if len(self.extractor_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.extractor_sha256):
            raise ValueError("authority extractor_sha256 must be a lowercase SHA-256 digest")
        if self.source_type not in {
            "AUTHORITATIVE_STRUCTURED_DATA", "OFFICIAL_PRIMARY_SOURCE",
        }:
            raise ValueError("registered authority must use a qualified source type")
        if any(not isinstance(key, str) or not key or not key.replace("_", "").isalnum()
               for key in self.allowed_query_keys):
            raise ValueError("authority allowed query keys must be nonempty names")
        required = dict(self.required_query_values)
        if (len(required) != len(self.required_query_values)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       or key not in self.allowed_query_keys or not value
                       for key, value in self.required_query_values)):
            raise ValueError("authority required query values must use allowed nonempty keys and values")

    def allows(self, *, relation: str, source_uri: str) -> bool:
        parsed = urlsplit(source_uri)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in (None, 443)):
            return False
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname not in self.domains or relation not in self.relations:
            return False
        if not any(fnmatchcase(parsed.path or "/", pattern) for pattern in self.endpoint_patterns):
            return False
        try:
            query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            return False
        if len({key for key, _value in query}) != len(query):
            return False
        query_values = dict(query)
        return (all(key in self.allowed_query_keys for key in query_values)
                and all(query_values.get(key) == value for key, value in self.required_query_values))

    def to_json(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "publisher": self.publisher,
            "domains": list(self.domains),
            "relations": list(self.relations),
            "endpoint_patterns": list(self.endpoint_patterns),
            "allowed_query_keys": list(self.allowed_query_keys),
            "required_query_values": dict(self.required_query_values),
            "source_type": self.source_type,
            "extractor_id": self.extractor_id,
            "extractor_module": self.extractor_module,
            "extractor_symbol": self.extractor_symbol,
            "extractor_sha256": self.extractor_sha256,
            "schema_id": self.schema_id,
            "content_types": list(self.content_types),
            "version": self.version,
        }


class AuthorityRegistry:
    """Read-only registry with explicit lifecycle state and fail-closed lookup."""

    def __init__(self, definitions: Iterable[AuthorityDefinition], *, registry_version: str,
                 status: RegistryStatus):
        entries = tuple(definitions)
        self._by_id = {entry.authority_id: entry for entry in entries}
        if len(self._by_id) != len(entries) or not registry_version:
            raise ValueError("authority ids must be unique and registry_version is required")
        self.registry_version, self.status = registry_version, status

    @property
    def definitions(self) -> tuple[AuthorityDefinition, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def require_truth_bearing(self) -> None:
        if self.status not in {
            RegistryStatus.FROZEN_FOR_EXPERIMENT,
            RegistryStatus.FROZEN_FOR_QUALIFICATION,
        }:
            raise ValueError("authority registry is not frozen for truth-bearing acquisition")

    def resolve(self, *, authority_id: str, relation: str,
                source_uri: str) -> AuthorityDefinition:
        definition = self._by_id.get(authority_id)
        if definition is None or not definition.allows(relation=relation, source_uri=source_uri):
            raise AuthorityNotRegistered(
                f"{AUTHORITY_NOT_REGISTERED}: no registered authority permits "
                f"{authority_id!r}, relation {relation!r}, and URI {source_uri!r}")
        return definition

    def identity(self) -> dict[str, object]:
        payload = {"schema": "DAPH_AUTHORITY_REGISTRY_V2", "registry_version": self.registry_version,
                   "status": self.status.value,
                   "definitions": [entry.to_json() for entry in self.definitions]}
        return {**payload, "sha256": _sha256(_canonical_json(payload))}


def load_authority_registry(path: str | Path, *, repository_root: str | Path | None = None) -> AuthorityRegistry:
    """Load a frozen JSON registry and verify each registered extractor file."""
    registry_path = Path(path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else registry_path.parent.parent
    payload = json.loads(registry_path.read_text())
    if payload.get("schema") != "DAPH_AUTHORITY_REGISTRY_V2":
        raise ValueError("unsupported authority registry schema")
    try:
        status = RegistryStatus(payload["status"])
    except (KeyError, ValueError) as error:
        raise ValueError("authority registry has an unsupported status") from error
    definitions = []
    for raw in payload.get("definitions", ()):
        definition = dict(raw)
        module_path = definition.get("extractor_module")
        if not isinstance(module_path, str):
            raise ValueError("registered authority must identify its extractor module")
        module = (root / module_path).resolve()
        if root not in module.parents or not module.is_file():
            raise ValueError("registered authority extractor module is absent or outside repository")
        if _sha256(module.read_bytes()) != definition.get("extractor_sha256"):
            raise ValueError("registered authority extractor does not match its pinned hash")
        for field_name in ("domains", "relations", "endpoint_patterns", "allowed_query_keys", "content_types"):
            definition[field_name] = tuple(definition.get(field_name, ()))
        required_values = definition.get("required_query_values", {})
        if not isinstance(required_values, dict):
            raise ValueError("authority required_query_values must be a JSON object")
        definition["required_query_values"] = tuple(sorted(required_values.items()))
        definitions.append(AuthorityDefinition(**definition))
    return AuthorityRegistry(definitions, registry_version=str(payload.get("registry_version", "")), status=status)


EMPTY_AUTHORITY_REGISTRY = AuthorityRegistry((), registry_version="0", status=RegistryStatus.RETIRED)


@dataclass(frozen=True)
class VerifiedExtractor:
    extractor_id: str
    module: str
    symbol: str
    sha256: str
    extract: Callable[[bytes, str], Mapping[str, object]]


def load_verified_extractor(repository_root: str | Path,
                           definition: AuthorityDefinition) -> VerifiedExtractor:
    """Load the exact committed extractor module and symbol named by the registry.

    The hash binds executable identity; it does not sandbox code. The pinned
    extractor module remains part of the reviewed, trusted source tree.
    """
    root = Path(repository_root).resolve()
    module_path = (root / definition.extractor_module).resolve()
    if root not in module_path.parents:
        raise ValueError("registered extractor module is outside repository")
    source = module_path.read_bytes()
    if _sha256(source) != definition.extractor_sha256:
        raise ValueError("registered authority extractor does not match its pinned hash")
    module = ModuleType(f"_daph_verified_{definition.extractor_id}_{definition.extractor_sha256[:12]}")
    module.__file__ = str(module_path)
    exec(compile(source, str(module_path), "exec"), module.__dict__)  # noqa: S102 - verified committed bytes
    candidate = getattr(module, definition.extractor_symbol, None)
    if not callable(candidate):
        raise ValueError("registered authority extractor symbol is not callable")
    return VerifiedExtractor(definition.extractor_id, definition.extractor_module,
                             definition.extractor_symbol, definition.extractor_sha256, candidate)


class RegisteredAuthorityAcquirer:
    """Truth-bearing acquisition bound to a frozen registry and verified extractor bytes."""

    ACQUISITION_METHOD = "registered_authority"
    ACQUISITION_VERSION = "3.0.0-v2b"

    def __init__(self, registry: AuthorityRegistry, transport: object, *,
                 repository_root: str | Path):
        registry.require_truth_bearing()
        self.registry = registry
        self.transport = transport
        self.root = Path(repository_root).resolve()

    def _extractor(self, definition: AuthorityDefinition) -> VerifiedExtractor:
        # Reload verified bytes for each acquisition. This deliberately leaves
        # no injectable callable cache between a pinned file/symbol and use.
        extractor = load_verified_extractor(self.root, definition)
        if (extractor.sha256 != definition.extractor_sha256
                or extractor.module != definition.extractor_module
                or extractor.symbol != definition.extractor_symbol):
            raise ValueError("registered authority extractor binding changed")
        return extractor

    def acquire(self, request: object, *, authority_id: str, relation: str):
        # Delayed imports preserve a one-way dependency: frozen V2A core never
        # imports this V2B-only controlled acquisition path.
        from .core import AcquisitionRequest, AcquisitionResult, AcquisitionStatus, SourceType
        from .network import NetworkPolicyError, NetworkTransportError

        if not isinstance(request, AcquisitionRequest):
            raise TypeError("registered acquisition requires AcquisitionRequest")
        try:
            definition = self.registry.resolve(authority_id=authority_id, relation=relation,
                                               source_uri=request.source_uri)
            extractor = self._extractor(definition)
            response = self.transport.fetch(
                request.source_uri,
                uri_validator=lambda uri: self.registry.resolve(
                    authority_id=authority_id, relation=relation, source_uri=uri))
            # Re-check final URI even if a custom transport ignores the callback.
            final_definition = self.registry.resolve(authority_id=authority_id, relation=relation,
                                                     source_uri=response.final_uri)
        except (NetworkPolicyError, NetworkTransportError):
            return AcquisitionResult(AcquisitionStatus.NETWORK_ERROR, request)
        except AuthorityNotRegistered as error:
            return AcquisitionResult(AcquisitionStatus.INVALID_RESPONSE, request, detail=str(error))
        except ValueError:
            return AcquisitionResult(AcquisitionStatus.INVALID_RESPONSE, request,
                                     detail="REGISTERED_EXTRACTOR_MISMATCH")
        if response.status >= 400:
            status = AcquisitionStatus.RATE_LIMITED if response.status == 429 else AcquisitionStatus.NOT_FOUND
            return AcquisitionResult(status, request, detail=f"HTTP {response.status}")
        if response.content_type not in final_definition.content_types:
            return AcquisitionResult(AcquisitionStatus.UNSUPPORTED_CONTENT, request,
                                     raw_content=response.body, content_type=response.content_type)
        try:
            fields = dict(extractor.extract(response.body, response.content_type))
        except (TypeError, ValueError, UnicodeDecodeError):
            return AcquisitionResult(AcquisitionStatus.PARSE_ERROR, request,
                                     raw_content=response.body, content_type=response.content_type)
        registry_identity = self.registry.identity()
        attestation = {
            "schema": "DAPH_AUTHORITY_ATTESTATION_V2",
            "authority_id": final_definition.authority_id,
            "registry_sha256": registry_identity["sha256"],
            "registry_status": self.registry.status.value,
            "relation": relation,
            "extractor_id": extractor.extractor_id,
            "extractor_module": extractor.module,
            "extractor_symbol": extractor.symbol,
            "extractor_sha256": extractor.sha256,
            "schema_id": final_definition.schema_id,
            "source_uri": request.source_uri,
            "final_uri": response.final_uri,
            "peer_ip": response.peer_ip,
            "raw_content_sha256": _sha256(response.body),
            "extracted_fields_sha256": canonical_extracted_fields_sha256(fields),
            "acquisition_method": self.ACQUISITION_METHOD,
            "acquisition_version": self.ACQUISITION_VERSION,
        }
        metadata = {**dict(request.request_metadata), "authority_id": final_definition.authority_id,
                    "authority_registry_sha256": registry_identity["sha256"],
                    "extractor_id": extractor.extractor_id, "extractor_sha256": extractor.sha256,
                    "schema_id": final_definition.schema_id}
        controlled_request = replace(
            request, canonical_source_uri=response.final_uri,
            source_type=SourceType(final_definition.source_type), request_metadata=metadata)
        return AcquisitionResult(
            AcquisitionStatus.SUCCESS, controlled_request, raw_content=response.body,
            content_type=response.content_type, fetched_at="", extracted_fields=fields,
            publisher=final_definition.publisher, publisher_domain=final_definition.domains[0],
            authority_attestation=attestation,
            response_metadata={"http_status": response.status, "final_uri": response.final_uri,
                               "peer_ip": response.peer_ip, "authority_id": final_definition.authority_id})
