"""Frozen authority definitions for controlled, truth-bearing acquisition.

Generic HTTP capture deliberately stays outside this registry.  A relation can
only be acquired as authoritative when an exact authority, domain, endpoint,
extractor fingerprint, and schema are all registered together.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlsplit


AUTHORITY_NOT_REGISTERED = "AUTHORITY_NOT_REGISTERED"


class AuthorityNotRegistered(ValueError):
    code = AUTHORITY_NOT_REGISTERED


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True)
class AuthorityDefinition:
    authority_id: str
    publisher: str
    domains: tuple[str, ...]
    relations: tuple[str, ...]
    endpoint_patterns: tuple[str, ...]
    source_type: str
    extractor_id: str
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
        if len(self.extractor_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.extractor_sha256):
            raise ValueError("authority extractor_sha256 must be a lowercase SHA-256 digest")
        if self.source_type not in {
            "AUTHORITATIVE_STRUCTURED_DATA", "OFFICIAL_PRIMARY_SOURCE",
        }:
            raise ValueError("registered authority must use a qualified source type")

    def allows(self, *, relation: str, source_uri: str) -> bool:
        parsed = urlsplit(source_uri)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in (None, 443)):
            return False
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname not in self.domains or relation not in self.relations:
            return False
        return any(fnmatchcase(parsed.path or "/", pattern) for pattern in self.endpoint_patterns)

    def to_json(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "publisher": self.publisher,
            "domains": list(self.domains),
            "relations": list(self.relations),
            "endpoint_patterns": list(self.endpoint_patterns),
            "source_type": self.source_type,
            "extractor_id": self.extractor_id,
            "extractor_sha256": self.extractor_sha256,
            "schema_id": self.schema_id,
            "content_types": list(self.content_types),
            "version": self.version,
        }


class AuthorityRegistry:
    """Read-only registry with fail-closed authority resolution."""

    def __init__(self, definitions: Iterable[AuthorityDefinition]):
        entries = tuple(definitions)
        self._by_id = {entry.authority_id: entry for entry in entries}
        if len(self._by_id) != len(entries):
            raise ValueError("authority ids must be unique")

    def resolve(self, *, authority_id: str, relation: str,
                source_uri: str) -> AuthorityDefinition:
        definition = self._by_id.get(authority_id)
        if definition is None or not definition.allows(relation=relation, source_uri=source_uri):
            raise AuthorityNotRegistered(
                f"{AUTHORITY_NOT_REGISTERED}: no registered authority permits "
                f"{authority_id!r}, relation {relation!r}, and URI {source_uri!r}")
        return definition

    def identity(self) -> dict[str, object]:
        definitions = [self._by_id[key].to_json() for key in sorted(self._by_id)]
        return {"schema": "DAPH_AUTHORITY_REGISTRY_V1", "definitions": definitions,
                "sha256": _sha256(_canonical_json(definitions))}


def load_authority_registry(path: str | Path, *, repository_root: str | Path | None = None) -> AuthorityRegistry:
    """Load a frozen JSON registry and verify each registered extractor file."""
    registry_path = Path(path).resolve()
    root = Path(repository_root).resolve() if repository_root is not None else registry_path.parent.parent
    payload = json.loads(registry_path.read_text())
    if payload.get("schema") != "DAPH_AUTHORITY_REGISTRY_V1":
        raise ValueError("unsupported authority registry schema")
    definitions = []
    for raw in payload.get("definitions", ()):
        definition = dict(raw)
        module_path = definition.pop("extractor_module", None)
        if not isinstance(module_path, str):
            raise ValueError("registered authority must identify its extractor module")
        module = (root / module_path).resolve()
        if root not in module.parents or not module.is_file():
            raise ValueError("registered authority extractor module is absent or outside repository")
        if _sha256(module.read_bytes()) != definition.get("extractor_sha256"):
            raise ValueError("registered authority extractor does not match its pinned hash")
        for field_name in ("domains", "relations", "endpoint_patterns", "content_types"):
            definition[field_name] = tuple(definition[field_name])
        definitions.append(AuthorityDefinition(**definition))
    return AuthorityRegistry(definitions)


# No endpoint is authoritative merely because a caller says it is. V2B must
# explicitly freeze concrete records before it enables authoritative HTTP.
EMPTY_AUTHORITY_REGISTRY = AuthorityRegistry(())


@dataclass(frozen=True)
class FrozenExtractor:
    extractor_id: str
    sha256: str
    extract: Callable[[bytes, str], Mapping[str, object]]

    @classmethod
    def from_file(cls, extractor_id: str, path: str | Path,
                  extract: Callable[[bytes, str], Mapping[str, object]]) -> "FrozenExtractor":
        raw = Path(path).read_bytes()
        return cls(extractor_id, _sha256(raw), extract)


class RegisteredAuthorityAcquirer:
    """The only path that promotes network bytes to registered authority.

    It is intentionally separate from generic HTTP. The registry dictates the
    relation, endpoint, schema, extractor fingerprint, source type, and
    content type; caller-provided request labels cannot create authority.
    """

    ACQUISITION_METHOD = "registered_authority"
    ACQUISITION_VERSION = "1.0.0-v2b"

    def __init__(self, registry: AuthorityRegistry, transport: object,
                 extractors: Iterable[FrozenExtractor]):
        self.registry = registry
        self.transport = transport
        entries = tuple(extractors)
        self.extractors = {extractor.extractor_id: extractor for extractor in entries}
        if len(self.extractors) != len(entries):
            raise ValueError("extractor ids must be unique")

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
        except AuthorityNotRegistered as error:
            return AcquisitionResult(AcquisitionStatus.INVALID_RESPONSE, request, detail=str(error))
        extractor = self.extractors.get(definition.extractor_id)
        if extractor is None or extractor.sha256 != definition.extractor_sha256:
            return AcquisitionResult(AcquisitionStatus.INVALID_RESPONSE, request,
                                     detail="REGISTERED_EXTRACTOR_MISMATCH")
        try:
            response = self.transport.fetch(request.source_uri)
        except (NetworkPolicyError, NetworkTransportError):
            return AcquisitionResult(AcquisitionStatus.NETWORK_ERROR, request)
        if response.status >= 400:
            status = AcquisitionStatus.RATE_LIMITED if response.status == 429 else AcquisitionStatus.NOT_FOUND
            return AcquisitionResult(status, request, detail=f"HTTP {response.status}")
        if response.content_type not in definition.content_types:
            return AcquisitionResult(AcquisitionStatus.UNSUPPORTED_CONTENT, request,
                                     raw_content=response.body, content_type=response.content_type)
        try:
            fields = dict(extractor.extract(response.body, response.content_type))
        except (TypeError, ValueError, UnicodeDecodeError):
            return AcquisitionResult(AcquisitionStatus.PARSE_ERROR, request,
                                     raw_content=response.body, content_type=response.content_type)
        metadata = {**dict(request.request_metadata), "authority_id": definition.authority_id,
                    "authority_registry_sha256": self.registry.identity()["sha256"],
                    "extractor_id": extractor.extractor_id, "extractor_sha256": extractor.sha256,
                    "schema_id": definition.schema_id}
        controlled_request = replace(
            request, canonical_source_uri=response.final_uri,
            source_type=SourceType(definition.source_type), request_metadata=metadata)
        return AcquisitionResult(
            AcquisitionStatus.SUCCESS, controlled_request, raw_content=response.body,
            content_type=response.content_type, fetched_at="", extracted_fields=fields,
            publisher=definition.publisher, publisher_domain=definition.domains[0],
            response_metadata={"http_status": response.status, "final_uri": response.final_uri,
                               "peer_ip": response.peer_ip, "authority_id": definition.authority_id})
