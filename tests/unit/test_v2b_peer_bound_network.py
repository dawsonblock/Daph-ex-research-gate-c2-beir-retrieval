"""Network controls: direct peer binding, DNS filtering, and body limits."""
from __future__ import annotations

import socket

import pytest

from hrm_adaptive_memory.external_verification.network import (
    NetworkPolicy, NetworkPolicyError, PeerBoundHTTPSClient, PublicResolver,
    ResolvedEndpoint)


class _Socket:
    def __init__(self, peer: str): self.peer = peer
    def getpeername(self): return (self.peer, 443)
    def settimeout(self, timeout): self.timeout = timeout


class _Response:
    status = 200
    def getheaders(self): return [("Content-Type", "application/json"), ("Content-Length", "2")]
    def getheader(self, name): return dict(self.getheaders()).get(name.title())
    def read(self, count): return b"{}"


class _Connection:
    def __init__(self, peer: str): self.sock = _Socket(peer)
    def request(self, *args, **kwargs): pass
    def getresponse(self): return _Response()
    def close(self): self.closed = True


class _Resolver:
    def resolve(self, uri, policy):
        return ResolvedEndpoint(uri, "example.test", 443, ("8.8.8.8",))


def test_peer_bound_transport_rejects_connected_peer_outside_validated_dns_answers():
    client = PeerBoundHTTPSClient(
        resolver=_Resolver(), connection_factory=lambda endpoint, ip, timeout: _Connection("10.0.0.7"))
    with pytest.raises(NetworkPolicyError, match="connected peer"):
        client.fetch("https://example.test/data")


def test_peer_bound_transport_accepts_validated_peer_and_exposes_final_uri():
    client = PeerBoundHTTPSClient(
        resolver=_Resolver(), connection_factory=lambda endpoint, ip, timeout: _Connection("8.8.8.8"))
    response = client.fetch("https://example.test/data")
    assert response.peer_ip == "8.8.8.8"
    assert response.content_type == "application/json"


def test_public_resolver_rejects_private_dns_answers(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
    ])
    with pytest.raises(NetworkPolicyError, match="non-public"):
        PublicResolver().resolve("https://example.test/data", NetworkPolicy())
