"""Extractor identity for I3.12.

The extractor is scientifically part of the treatment. Its identity
must be hashed and recorded so that results are reproducible and
attributable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractorIdentity:
    """Identity record for a semantic relation extractor."""
    extractor_class: str
    extractor_version: str
    relation_schema_version: str
    normalization_rules: tuple[str, ...]
    thresholds: dict[str, float]
    prompt_template: str | None
    model_name: str | None
    model_version: str | None
    hypothesis_serializer: str
    evidence_serializer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "extractor_class": self.extractor_class,
            "extractor_version": self.extractor_version,
            "relation_schema_version": self.relation_schema_version,
            "normalization_rules": list(self.normalization_rules),
            "thresholds": dict(self.thresholds),
            "prompt_template": self.prompt_template,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hypothesis_serializer": self.hypothesis_serializer,
            "evidence_serializer": self.evidence_serializer,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_extractor_identity(
    extractor_class: str,
    extractor_version: str = "1.0.0",
    relation_schema_version: str = "1",
    normalization_rules: tuple[str, ...] = (),
    thresholds: dict[str, float] | None = None,
    prompt_template: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    hypothesis_serializer: str = "default_text",
    evidence_serializer: str = "default_text",
) -> ExtractorIdentity:
    return ExtractorIdentity(
        extractor_class=extractor_class,
        extractor_version=extractor_version,
        relation_schema_version=relation_schema_version,
        normalization_rules=normalization_rules,
        thresholds=thresholds or {},
        prompt_template=prompt_template,
        model_name=model_name,
        model_version=model_version,
        hypothesis_serializer=hypothesis_serializer,
        evidence_serializer=evidence_serializer,
    )


def save_extractor_identity(identity: ExtractorIdentity, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.as_dict(), indent=2, sort_keys=True) + "\n"
    )
