"""Peer-bound direct HTTPS transport with bounded redirect and body handling."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import http.client
import io
import socket
import ssl
import time
from typing import Callable, Mapping
from urllib.parse import urljoin, urlsplit
import zlib

from .policy import NetworkPolicy, NetworkPolicyError, validate_public_https_uri
from .resolver import PublicResolver, ResolvedEndpoint


class NetworkTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeerBoundResponse:
    source_uri: str
    final_uri: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    peer_ip: str

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";", 1)[0].strip().lower()


class PeerBoundHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a prevalidated IP while retaining hostname TLS/SNI checks."""

    def __init__(self, endpoint: ResolvedEndpoint, peer_ip: str, *, timeout: float):
        super().__init__(endpoint.hostname, endpoint.port, timeout=timeout,
                         context=ssl.create_default_context())
        self._peer_ip = peer_ip
        self._peer_port = endpoint.port

    def connect(self) -> None:
        raw = socket.create_connection((self._peer_ip, self._peer_port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def _read_bounded(response: object, maximum: int) -> bytes:
    length = response.getheader("Content-Length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError as error:
            raise NetworkPolicyError("capture response has an invalid Content-Length") from error
        if declared < 0 or declared > maximum:
            raise NetworkPolicyError("capture response exceeds maximum compressed body size")
    raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise NetworkPolicyError("capture response exceeds maximum compressed body size")
    return raw


def _decompress_bounded(raw: bytes, encoding: str, maximum: int) -> bytes:
    encoding = encoding.strip().lower()
    if not encoding or encoding == "identity":
        return raw
    try:
        if encoding == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
                value = handle.read(maximum + 1)
        elif encoding == "deflate":
            decoder = zlib.decompressobj()
            value = decoder.decompress(raw, maximum + 1)
            if len(value) <= maximum:
                value += decoder.flush(maximum + 1 - len(value))
        else:
            raise NetworkPolicyError("capture response uses unsupported content encoding")
    except (OSError, EOFError, zlib.error) as error:
        raise NetworkPolicyError("capture response has invalid compressed content") from error
    if len(value) > maximum:
        raise NetworkPolicyError("capture response exceeds maximum decompressed body size")
    return value


class PeerBoundHTTPSClient:
    """Direct transport whose actual TLS peer must be a validated DNS answer."""

    def __init__(self, *, policy: NetworkPolicy | None = None,
                 resolver: PublicResolver | None = None,
                 connection_factory: Callable[[ResolvedEndpoint, str, float], object] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.policy = policy or NetworkPolicy()
        self.resolver = resolver or PublicResolver()
        self.connection_factory = connection_factory or (
            lambda endpoint, address, timeout: PeerBoundHTTPSConnection(
                endpoint, address, timeout=timeout))
        self.clock = clock

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise NetworkTransportError("capture request exceeded total deadline")
        return remaining

    @staticmethod
    def _request_target(uri: str) -> str:
        parsed = urlsplit(uri)
        return (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")

    def fetch(self, uri: str, *, uri_validator: Callable[[str], object] | None = None) -> PeerBoundResponse:
        source_uri, current_uri = uri, uri
        deadline = self.clock() + self.policy.total_timeout_seconds
        if uri_validator is not None:
            uri_validator(current_uri)
        for redirect_index in range(self.policy.max_redirects + 1):
            endpoint = self.resolver.resolve(current_uri, self.policy)
            peer_ip = endpoint.addresses[0]
            timeout = min(self.policy.connect_timeout_seconds, self._remaining(deadline))
            connection = self.connection_factory(endpoint, peer_ip, timeout)
            try:
                connection.request("GET", self._request_target(current_uri), headers={
                    "Accept": "application/json,text/csv,text/html,text/plain;q=0.8",
                    "Accept-Encoding": "identity",
                    "User-Agent": "DAPH-V2B-Acquisition/1.0",
                })
                socket_object = connection.sock
                if socket_object is None:
                    raise NetworkTransportError("capture transport did not establish a socket")
                actual_peer = socket_object.getpeername()[0]
                if actual_peer not in endpoint.addresses:
                    raise NetworkPolicyError(
                        "connected peer is not a validated DNS resolution address")
                socket_object.settimeout(min(self.policy.read_timeout_seconds, self._remaining(deadline)))
                response = connection.getresponse()
                headers = {key.lower(): value for key, value in response.getheaders()}
                if response.status in {301, 302, 303, 307, 308}:
                    location = headers.get("location")
                    if not location:
                        raise NetworkTransportError("redirect response lacks Location")
                    if redirect_index == self.policy.max_redirects:
                        raise NetworkPolicyError("capture redirect limit exceeded")
                    current_uri = urljoin(current_uri, location)
                    validate_public_https_uri(current_uri, self.policy)
                    if uri_validator is not None:
                        # Truth-bearing callers bind this to their authority
                        # definition, so every redirect hop stays authorized.
                        uri_validator(current_uri)
                    continue
                raw = _read_bounded(response, self.policy.max_compressed_bytes)
                body = _decompress_bounded(raw, headers.get("content-encoding", ""),
                                           self.policy.max_decompressed_bytes)
                self._remaining(deadline)
                if uri_validator is not None:
                    uri_validator(current_uri)
                return PeerBoundResponse(source_uri, current_uri, response.status, headers, body, actual_peer)
            except (NetworkPolicyError, NetworkTransportError):
                raise
            except (socket.timeout, socket.error, ssl.SSLError, http.client.HTTPException, OSError) as error:
                raise NetworkTransportError("peer-bound HTTPS request failed") from error
            finally:
                connection.close()
        raise NetworkPolicyError("capture redirect limit exceeded")
