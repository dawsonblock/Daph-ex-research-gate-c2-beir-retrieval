"""CERTIFIED_MEMORY_V1 -- the frozen boundary around the confirmed memory
operation (grammar_v4 -> C2 retrieval -> G2 runtime graph -> frozen path
ranking -> path-coherent packet composer -> packet budget 6 -> HRM-Text-1B),
per configs/certified_memory_v1.json and evidence/gate_hrm/confirmation_2_execute.json
(CONFIRMED_GRAPH_PLUS_PATH_COMPOSITION, delta_HRM_graph_value=+0.0720, LCB2.5=+0.0147>0).

Every field here is recomputed from the ACTUAL current code state rather than
declared as a static string -- the same discipline entity_extractor_config_hash()
already uses, and the same discipline that caught a real (if benign) mismatch
earlier in this project when a standalone hash check omitted a required
process-level pin. Any future executive/controller experiment that invokes
"the memory operation" as a black box MUST call assert_certified_memory_v1_unchanged()
first: this is the mechanical enforcement of "do not let controller development
quietly alter retrieval or graph behavior."
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from hrm_adaptive_memory.c4.bridge_extraction import (
    entity_extractor_config_hash, get_default_boundary_policy)

#: Pinned exactly as recorded in every confirmation-2 receipt's ExecutionIdentity
#: (hrm_adaptive_memory/experiment_integrity/execution_identity.py) -- this is
#: the literal string that was hashed into execution_identity_sha256 for the
#: run that earned CONFIRMED_GRAPH_PLUS_PATH_COMPOSITION, not a re-derived value.
MEMORY_V1_RETRIEVAL_CONFIG_HASH = "C2"
MEMORY_V1_SELECTOR_CONFIG_HASH = "s2_v2+s4_composer_v1"
MEMORY_V1_MODEL_REVISION = "sapientinc/HRM-Text-1B@9f082d68"
MEMORY_V1_MODEL_REVISION_FULL = "sapientinc/HRM-Text-1B @ 9f082d68b8cd0ebc56e33f1c88c45609174c272c"
MEMORY_V1_PIPELINE_VERSION = "hrm_qualification_v1"
MEMORY_V1_PACKET_BUDGET = 6
MEMORY_V1_BOUNDARY_POLICY = "grammar_v4"

#: The commit whose receipts earned the confirmed verdict (verified, not
#: trusted -- see evidence/gate_hrm/confirmation_2_execute.json and the
#: cryptographic source-commit binding check in RESEARCH_STATUS.json's
#: hrm_confirmation_2.source_commit_verification).
MEMORY_V1_SOURCE_COMMIT = "c8672be80f122fba238e04407feede43804e38c0"

#: entity_extractor_config_hash() as recorded by BOTH qualification and
#: confirmation-2 (identical across both runs -- verified live, see
#: RESEARCH_STATUS.json hrm_confirmation_2.RUN_VALID checks).
MEMORY_V1_EXPECTED_EXTRACTOR_HASH = "1c156a7ed8a487bd"

MEMORY_V1_CONFIRMATION_CERTIFICATE = "evidence/gate_hrm/confirmation_2_execute.json"
MEMORY_V1_CONFIRMATION_PROTOCOL = "configs/gate_hrm_confirmation_2_v1.json"


class CertifiedMemoryDriftError(RuntimeError):
    """The running code's memory-stack identity no longer matches
    CERTIFIED_MEMORY_V1. Fail closed -- an executive experiment must never
    silently run against a drifted memory operation and call the result
    comparable to the frozen confirmation."""


@dataclass(frozen=True)
class CertifiedMemoryV1Identity:
    retrieval_config_hash: str
    selector_config_hash: str
    graph_compressor_config_hash: str
    model_revision: str
    pipeline_version: str
    packet_budget: int
    boundary_policy: str

    def canonical_sha256(self) -> str:
        payload = "|".join((
            self.retrieval_config_hash, self.selector_config_hash,
            self.graph_compressor_config_hash, self.model_revision,
            self.pipeline_version, str(self.packet_budget), self.boundary_policy))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def current_certified_memory_v1_identity() -> CertifiedMemoryV1Identity:
    """Recomputed from live code state, not declared. If bridge_extraction's
    grammar tables, stopword lists, or default boundary policy drift, this
    reflects that immediately -- the same mechanism that caught the earlier
    boundary_policy default mismatch."""
    return CertifiedMemoryV1Identity(
        retrieval_config_hash=MEMORY_V1_RETRIEVAL_CONFIG_HASH,
        selector_config_hash=MEMORY_V1_SELECTOR_CONFIG_HASH,
        graph_compressor_config_hash=entity_extractor_config_hash(),
        model_revision=MEMORY_V1_MODEL_REVISION,
        pipeline_version=MEMORY_V1_PIPELINE_VERSION,
        packet_budget=MEMORY_V1_PACKET_BUDGET,
        boundary_policy=get_default_boundary_policy())


#: Computed once, from the identity as it stood at confirmation-2 time
#: (boundary_policy="grammar_v4", extractor hash 1c156a7ed8a487bd). This is
#: MEMORY_V1_CONFIG_HASH -- the single stack-level identity hash referenced by
#: configs/certified_memory_v1.json.
MEMORY_V1_CONFIG_HASH = CertifiedMemoryV1Identity(
    retrieval_config_hash=MEMORY_V1_RETRIEVAL_CONFIG_HASH,
    selector_config_hash=MEMORY_V1_SELECTOR_CONFIG_HASH,
    graph_compressor_config_hash=MEMORY_V1_EXPECTED_EXTRACTOR_HASH,
    model_revision=MEMORY_V1_MODEL_REVISION,
    pipeline_version=MEMORY_V1_PIPELINE_VERSION,
    packet_budget=MEMORY_V1_PACKET_BUDGET,
    boundary_policy=MEMORY_V1_BOUNDARY_POLICY,
).canonical_sha256()


def pin_certified_memory_v1_boundary_policy() -> None:
    """Explicit, deliberate action -- call this ONCE near the top of any
    executive-experiment entrypoint, exactly as run_hrm_qualification.py's
    main() pins boundary_policy before qualification/confirmation ran. This is
    intentionally a SEPARATE function from the assertion below: pinning is an
    action the caller takes on purpose; asserting is a check that must be able
    to observe and reject whatever state the process is ACTUALLY in, including
    a caller who forgot to pin. Folding them together would let the assertion
    silently self-heal a wrong state instead of catching it."""
    from hrm_adaptive_memory.c4.bridge_extraction import set_default_boundary_policy
    set_default_boundary_policy(MEMORY_V1_BOUNDARY_POLICY)


def assert_certified_memory_v1_unchanged() -> CertifiedMemoryV1Identity:
    """Call this before invoking the memory operation as a black box in any
    executive/controller experiment. READ-ONLY: does not modify process state
    (see pin_certified_memory_v1_boundary_policy for that). Raises
    CertifiedMemoryDriftError (fail closed, never a warning) if the live stack
    -- including whatever boundary_policy the process actually happens to be
    in right now -- no longer matches the frozen identity that earned
    CONFIRMED_GRAPH_PLUS_PATH_COMPOSITION."""
    identity = current_certified_memory_v1_identity()
    live_hash = identity.canonical_sha256()
    if live_hash != MEMORY_V1_CONFIG_HASH:
        raise CertifiedMemoryDriftError(
            f"CERTIFIED_MEMORY_V1 drift detected: live identity hash {live_hash} "
            f"!= frozen {MEMORY_V1_CONFIG_HASH}. live={identity!r}. "
            "An executive/controller experiment must not run against a "
            "drifted memory operation -- either the drift is unintentional "
            "(call pin_certified_memory_v1_boundary_policy() before this "
            "assertion, or fix the code) or intentional (freeze a new "
            "CERTIFIED_MEMORY_V2 and re-confirm before using it in any "
            "executive experiment).")
    return identity
