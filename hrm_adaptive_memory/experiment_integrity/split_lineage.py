"""Mandatory split lineage, generalized beyond C4 certification.

This project now has several classes of split with very different permitted
uses: development, calibration, qualification_1 (consumed), confirmation_1
(consumed), and a future confirmation_2 that must stay untouched until it is
run exactly once. A dataset manifest today (e.g.
data/hrm/b3_calibration_v1/cal_700/dataset_manifest.json) carries a
``split_id`` but not an explicit, machine-checkable ``split_role``,
``consumed_status``, or ``permitted_uses`` -- so nothing currently stops a
future script from reading a consumed split and treating it as fresh. This
module makes that check mechanical instead of relying on a human remembering
which splits are burned.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SplitRole(str, Enum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    QUALIFICATION_CONSUMED = "qualification_consumed"
    CONFIRMATION_CONSUMED = "confirmation_consumed"
    #: A split built specifically to train/threshold-fit a controller (e.g.
    #: exec_training_v1 for ANSWER_PROBE_GATE_V1's logistic escalation
    #: policy), now used for that purpose. Distinct from
    #: QUALIFICATION_CONSUMED/CONFIRMATION_CONSUMED, which describe gate
    #: RUNS against a frozen mechanism -- this describes a split consumed to
    #: FIT a policy's parameters/threshold, not to evaluate a frozen one.
    TRAINING_CONSUMED = "training_consumed"
    FUTURE_CONFIRMATION = "future_confirmation"


class PermittedUse(str, Enum):
    MECHANISM_SELECTION = "mechanism_selection"
    THRESHOLD_SELECTION = "threshold_selection"
    CERTIFICATION = "certification"
    CONFIRMATION = "confirmation"
    DIAGNOSIS_ONLY = "diagnosis_only"
    HYPOTHESIS_GENERATION_ONLY = "hypothesis_generation_only"


#: A consumed split may ONLY be used for retrospective diagnosis -- never to
#: select a mechanism, choose a threshold, or certify/confirm anything. This
#: is the exact rule this project has enforced by hand (and stated explicitly
#: in commit messages) for qualification_1 and confirmation_1; encoding it
#: here makes a violation a mechanical check failure instead of a discipline
#: lapse.
CONSUMED_SPLIT_ALLOWED_USES = frozenset(
    {PermittedUse.DIAGNOSIS_ONLY, PermittedUse.HYPOTHESIS_GENERATION_ONLY})

_ROLE_DEFAULT_PERMITTED_USES: dict[SplitRole, frozenset[PermittedUse]] = {
    SplitRole.DEVELOPMENT: frozenset(
        {PermittedUse.MECHANISM_SELECTION, PermittedUse.THRESHOLD_SELECTION,
         PermittedUse.DIAGNOSIS_ONLY, PermittedUse.HYPOTHESIS_GENERATION_ONLY}),
    SplitRole.CALIBRATION: frozenset(
        {PermittedUse.MECHANISM_SELECTION, PermittedUse.THRESHOLD_SELECTION,
         PermittedUse.DIAGNOSIS_ONLY, PermittedUse.HYPOTHESIS_GENERATION_ONLY}),
    SplitRole.QUALIFICATION_CONSUMED: CONSUMED_SPLIT_ALLOWED_USES,
    SplitRole.CONFIRMATION_CONSUMED: CONSUMED_SPLIT_ALLOWED_USES,
    SplitRole.TRAINING_CONSUMED: CONSUMED_SPLIT_ALLOWED_USES,
    SplitRole.FUTURE_CONFIRMATION: frozenset({PermittedUse.CONFIRMATION}),
}


class SplitLineageError(ValueError):
    """A manifest's split lineage is missing, ambiguous, or a caller tried to
    use a split for something it does not permit. Fail closed, never warn."""


@dataclass(frozen=True)
class SplitManifest:
    split_id: str
    split_role: SplitRole
    consumed_status: bool
    permitted_uses: frozenset[PermittedUse]
    dataset_hash: str


def parse_split_manifest(raw: dict) -> SplitManifest:
    """Build a SplitManifest from a raw dict, rejecting anything missing or
    ambiguous rather than defaulting silently. This is the function a
    certifier calls before trusting ANY split-derived result."""
    missing = [f for f in ("split_id", "split_role", "consumed_status",
                           "permitted_uses", "dataset_hash") if f not in raw]
    if missing:
        raise SplitLineageError(
            f"split manifest missing required field(s): {missing}. "
            "A manifest without explicit split lineage cannot be certified.")

    split_id = raw["split_id"]
    if not isinstance(split_id, str) or not split_id:
        raise SplitLineageError(f"split_id must be a non-empty string, got {split_id!r}")

    try:
        role = SplitRole(raw["split_role"])
    except ValueError:
        raise SplitLineageError(
            f"split_role {raw['split_role']!r} is not one of {[r.value for r in SplitRole]}"
        ) from None

    consumed = raw["consumed_status"]
    if not isinstance(consumed, bool):
        raise SplitLineageError(f"consumed_status must be a bool, got {consumed!r}")
    role_implies_consumed = role in (SplitRole.QUALIFICATION_CONSUMED,
                                     SplitRole.CONFIRMATION_CONSUMED,
                                     SplitRole.TRAINING_CONSUMED)
    if role_implies_consumed and not consumed:
        raise SplitLineageError(
            f"split_role={role.value} implies consumed_status=True, got False -- "
            "ambiguous lineage, refusing to guess which is correct")
    if consumed and not role_implies_consumed:
        raise SplitLineageError(
            f"consumed_status=True but split_role={role.value} does not imply "
            "consumption -- ambiguous lineage")

    raw_uses = raw["permitted_uses"]
    if not isinstance(raw_uses, (list, tuple)) or not raw_uses:
        raise SplitLineageError("permitted_uses must be a non-empty list")
    try:
        uses = frozenset(PermittedUse(u) for u in raw_uses)
    except ValueError:
        raise SplitLineageError(
            f"permitted_uses contains an unrecognized value: {raw_uses!r}") from None
    allowed = _ROLE_DEFAULT_PERMITTED_USES[role]
    if not uses <= allowed:
        raise SplitLineageError(
            f"permitted_uses {sorted(u.value for u in uses)} exceeds what "
            f"split_role={role.value} allows ({sorted(u.value for u in allowed)})")

    dataset_hash = raw["dataset_hash"]
    if not isinstance(dataset_hash, str) or not dataset_hash:
        raise SplitLineageError(f"dataset_hash must be a non-empty string, got {dataset_hash!r}")

    return SplitManifest(split_id=split_id, split_role=role, consumed_status=consumed,
                         permitted_uses=uses, dataset_hash=dataset_hash)


def require_permitted_use(manifest: SplitManifest, use: PermittedUse) -> None:
    """Raise if ``use`` is not permitted for this split. Call this at the top
    of any script that is about to select a mechanism, choose a threshold,
    certify, or confirm against a specific split -- the exact prevention this
    module exists for: a future confirmation/diagnostic split accidentally
    being used to pick a mechanism or threshold."""
    if use not in manifest.permitted_uses:
        raise SplitLineageError(
            f"split {manifest.split_id!r} (role={manifest.split_role.value}) does not "
            f"permit use={use.value}. Permitted: {sorted(u.value for u in manifest.permitted_uses)}")
