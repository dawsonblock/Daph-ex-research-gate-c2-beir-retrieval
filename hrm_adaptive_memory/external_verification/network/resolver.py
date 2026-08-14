"""Single-resolution, public-address-only DNS resolution."""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket

from .policy import NetworkPolicy, NetworkPolicyError, validate_public_https_uri


@dataclass(frozen=True)
class ResolvedEndpoint:
    uri: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


class PublicResolver:
    """Resolve once and reject a URI if *any* DNS answer is non-public."""

    def resolve(self, uri: str, policy: NetworkPolicy) -> ResolvedEndpoint:
        parsed = validate_public_https_uri(uri, policy)
        hostname = parsed.hostname.rstrip(".").lower()
        port = parsed.port or 443
        try:
            results = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as error:
            raise NetworkPolicyError("capture hostname did not resolve") from error
        addresses = tuple(sorted({result[4][0] for result in results}))
        if not addresses:
            raise NetworkPolicyError("capture hostname did not resolve")
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as error:
                raise NetworkPolicyError("resolver returned an invalid IP address") from error
            if not parsed_address.is_global:
                raise NetworkPolicyError(f"non-public capture address is forbidden: {parsed_address}")
        return ResolvedEndpoint(uri=uri, hostname=hostname, port=port, addresses=addresses)
