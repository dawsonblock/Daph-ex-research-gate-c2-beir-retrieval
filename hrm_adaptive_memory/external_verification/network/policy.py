"""Strict URL and response limits for public HTTPS acquisition."""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit


class NetworkPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class NetworkPolicy:
    allowed_ports: tuple[int, ...] = (443,)
    max_redirects: int = 3
    max_compressed_bytes: int = 8 * 1024 * 1024
    max_decompressed_bytes: int = 8 * 1024 * 1024
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    total_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if (not self.allowed_ports or self.max_redirects < 0
                or self.max_compressed_bytes < 1 or self.max_decompressed_bytes < 1
                or min(self.connect_timeout_seconds, self.read_timeout_seconds,
                       self.total_timeout_seconds) <= 0):
            raise ValueError("network policy limits must be positive")


def validate_public_https_uri(uri: str, policy: NetworkPolicy) -> SplitResult:
    parsed = urlsplit(uri)
    if parsed.scheme != "https" or not parsed.hostname:
        raise NetworkPolicyError("capture URI must use HTTPS with a hostname")
    if parsed.username or parsed.password:
        raise NetworkPolicyError("capture URI userinfo is forbidden")
    try:
        port = parsed.port or 443
    except ValueError as error:
        raise NetworkPolicyError("capture URI has an invalid port") from error
    if port not in policy.allowed_ports:
        raise NetworkPolicyError("capture URI port is not permitted")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise NetworkPolicyError("localhost capture is forbidden")
    if parsed.fragment:
        raise NetworkPolicyError("capture URI fragments are forbidden")
    return parsed
