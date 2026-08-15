"""Governor resources: typed normalization of runtime resource state.

The real ResourceState.as_dict() produces keys like:
  retrieval_calls_remaining, verification_calls_remaining,
  search_calls_remaining, reasoning_tokens_remaining,
  executive_steps_remaining

The governor must never use raw dictionary lookups with short names.
This module provides a single normalization point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


GOVERNOR_RESOURCE_SCHEMA = "DAPH_V2B_I3_5_GOVERNOR_RESOURCES_V1"


@dataclass(frozen=True)
class GovernorResourceState:
    """Typed view of runtime resources for the governor.

    Constructed once from ResourceState.as_dict() and used everywhere.
    """
    retrieval_remaining: int
    verification_remaining: int
    search_remaining: int
    reasoning_tokens_remaining: int
    steps_remaining: int

    @property
    def has_retrieval(self) -> bool:
        return self.retrieval_remaining > 0

    @property
    def has_verification(self) -> bool:
        return self.verification_remaining > 0

    @property
    def has_search(self) -> bool:
        return self.search_remaining > 0

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_tokens_remaining > 0

    @property
    def any_useful_remaining(self) -> bool:
        return self.has_retrieval or self.has_verification or self.has_search or self.has_reasoning

    def remaining_for_channel(self, channel: str) -> int:
        """Get remaining count for a semantic cost channel name."""
        if channel == "retrieval":
            return self.retrieval_remaining
        if channel == "verification":
            return self.verification_remaining
        if channel == "search":
            return self.search_remaining
        if channel == "reasoning":
            return self.reasoning_tokens_remaining
        if channel == "steps":
            return self.steps_remaining
        return 0

    def is_last_resource(self, channel: str) -> bool:
        """Whether consuming this channel would use the last unit."""
        if channel == "steps":
            return False
        return self.remaining_for_channel(channel) <= 1

    def as_dict(self) -> dict[str, int]:
        return {
            "retrieval_remaining": self.retrieval_remaining,
            "verification_remaining": self.verification_remaining,
            "search_remaining": self.search_remaining,
            "reasoning_tokens_remaining": self.reasoning_tokens_remaining,
            "steps_remaining": self.steps_remaining,
        }


def normalize_resources(raw: Mapping[str, int]) -> GovernorResourceState:
    """Construct a GovernorResourceState from a raw resource dict.

    Handles the real runtime keys produced by ResourceState.as_dict().
    Falls back to 0 for any missing key.
    """
    return GovernorResourceState(
        retrieval_remaining=int(raw.get("retrieval_calls_remaining", 0)),
        verification_remaining=int(raw.get("verification_calls_remaining", 0)),
        search_remaining=int(raw.get("search_calls_remaining", 0)),
        reasoning_tokens_remaining=int(raw.get("reasoning_tokens_remaining", 0)),
        steps_remaining=int(raw.get("executive_steps_remaining", 0)),
    )
