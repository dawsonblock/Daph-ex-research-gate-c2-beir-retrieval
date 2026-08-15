"""Scientific scoring module for V2B-I3.4.1.

Restores the frozen I3.2.2 IG/DG/TR decomposition that was conflated in
the original I3.4 scaffold.

For latent task s, observation condition M, information state B_M(s), and
controller π:

    V_L^*(s)   = latent optimal value
    V_O^M(B)   = optimal value available from observable information
    V_π^M(s)   = realized controller value

Then:

    IG_M(s) = V_L^*(s) - V_O^M(B_M(s))    (information gap)
    DG_M(s) = V_O^M(B_M(s)) - V_π^M(s)    (decision gap)
    TR_M(s) = V_L^*(s) - V_π^M(s)          (total regret)

Algebraically:  TR_M(s) = IG_M(s) + DG_M(s)

Individual task contributions to IG or DG can be negative because the
observable oracle is an expectation over an information class.  Do NOT
clamp individual contributions.  Non-negativity guarantees apply at the
information-class expectation level:

    IG(B) ≥ 0  and  DG(B) ≥ 0

for the information-class expectation, and consequently at the correctly
weighted condition aggregate.

Schema identity: ``DAPH_V2B_I3_4_SCIENTIFIC_SCORING_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCORING_SCHEMA = "DAPH_V2B_I3_4_SCIENTIFIC_SCORING_V1"
SCORING_VERSION = 1

# Pre-frozen numerical tolerance for the TR = IG + DG identity check.
# This accommodates floating-point accumulation error in the oracle tables.
IDENTITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class I34ScientificTaskContribution:
    """Per-task scientific contribution under one condition.

    Individual contributions can be negative.  Non-negativity is guaranteed
    only at the information-class expectation level.
    """

    task_id: str
    condition: str
    latent_optimal_value: float          # V_L^*(s)
    observable_optimal_value: float      # V_O^M(B_M(s))
    controller_value: float              # V_π^M(s)
    information_gap_contribution: float  # IG_M(s) = V_L^* - V_O^M
    decision_gap_contribution: float     # DG_M(s) = V_O^M - V_π^M
    total_regret_contribution: float     # TR_M(s) = V_L^* - V_π^M
    information_class_hash: str
    observable_oracle_set_sha256: str
    latent_oracle_table_sha256: str

    def verify_identity(self, tolerance: float = IDENTITY_TOLERANCE) -> bool:
        """Check that TR = IG + DG within tolerance."""
        return abs(self.total_regret_contribution
                   - self.information_gap_contribution
                   - self.decision_gap_contribution) < tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "latent_optimal_value": self.latent_optimal_value,
            "observable_optimal_value": self.observable_optimal_value,
            "controller_value": self.controller_value,
            "information_gap_contribution": self.information_gap_contribution,
            "decision_gap_contribution": self.decision_gap_contribution,
            "total_regret_contribution": self.total_regret_contribution,
            "information_class_hash": self.information_class_hash,
            "observable_oracle_set_sha256": self.observable_oracle_set_sha256,
            "latent_oracle_table_sha256": self.latent_oracle_table_sha256,
        }


@dataclass(frozen=True)
class I34ScientificAggregate:
    """Condition-level aggregate of scientific contributions.

    Non-negativity of IG and DG holds at this level when the prior is
    TASK_UNIFORM and the observable oracle is the correct expectation
    over the information class.
    """

    condition: str
    task_count: int
    expected_latent_value: float       # E[V_L^*]
    expected_observable_value: float   # E[V_O^M]
    expected_controller_value: float   # E[V_π^M]
    information_gap: float             # IG = E[IG_M(s)]
    decision_gap: float                # DG = E[DG_M(s)]
    total_regret: float                # TR = E[TR_M(s)]
    tr_minus_ig_minus_dg: float        # |TR - IG - DG|, must be < tolerance
    prior: str                         # e.g. "TASK_UNIFORM"

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "task_count": self.task_count,
            "expected_latent_value": self.expected_latent_value,
            "expected_observable_value": self.expected_observable_value,
            "expected_controller_value": self.expected_controller_value,
            "information_gap": self.information_gap,
            "decision_gap": self.decision_gap,
            "total_regret": self.total_regret,
            "tr_minus_ig_minus_dg": self.tr_minus_ig_minus_dg,
            "prior": self.prior,
        }

    def verify_identity(self, tolerance: float = IDENTITY_TOLERANCE) -> bool:
        """Check that TR = IG + DG within tolerance at the aggregate level."""
        return self.tr_minus_ig_minus_dg < tolerance

    def verify_nonnegativity(self, tolerance: float = IDENTITY_TOLERANCE) -> bool:
        """Check that IG ≥ 0 and DG ≥ 0 at the aggregate level.

        This holds when the prior is TASK_UNIFORM and the observable oracle
        is the correct expectation over the information class.
        """
        return (self.information_gap >= -tolerance
                and self.decision_gap >= -tolerance)


@dataclass(frozen=True)
class I34PairedDelta:
    """Paired difference between blind and aware conditions for one task.

    The primary scientific observation is d_i = DG_blind,i - DG_aware,i.
    """

    task_id: str
    delta_ig: float   # ΔIG = IG_blind - IG_aware
    delta_dg: float   # ΔDG = DG_blind - DG_aware
    delta_tr: float   # ΔTR = TR_blind - TR_aware
    delta_cost: float  # cost_blind - cost_aware (optional, 0 if unused)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "delta_ig": self.delta_ig,
            "delta_dg": self.delta_dg,
            "delta_tr": self.delta_tr,
            "delta_cost": self.delta_cost,
        }


def compute_task_contribution(
    *,
    task_id: str,
    condition: str,
    latent_optimal_value: float,
    observable_optimal_value: float,
    controller_value: float,
    information_class_hash: str,
    observable_oracle_set_sha256: str,
    latent_oracle_table_sha256: str,
) -> I34ScientificTaskContribution:
    """Compute the IG/DG/TR decomposition for one task under one condition.

    Individual contributions are NOT clamped.  IG or DG can be negative
    for a single task because V_O^M(B) is an expectation over the
    information class, not a per-task optimum.
    """
    ig = latent_optimal_value - observable_optimal_value
    dg = observable_optimal_value - controller_value
    tr = latent_optimal_value - controller_value
    return I34ScientificTaskContribution(
        task_id=task_id,
        condition=condition,
        latent_optimal_value=latent_optimal_value,
        observable_optimal_value=observable_optimal_value,
        controller_value=controller_value,
        information_gap_contribution=ig,
        decision_gap_contribution=dg,
        total_regret_contribution=tr,
        information_class_hash=information_class_hash,
        observable_oracle_set_sha256=observable_oracle_set_sha256,
        latent_oracle_table_sha256=latent_oracle_table_sha256,
    )


def compute_aggregate(
    *,
    condition: str,
    contributions: Iterable[I34ScientificTaskContribution],
    prior: str = "TASK_UNIFORM",
) -> I34ScientificAggregate:
    """Aggregate per-task contributions into a condition-level summary.

    Uses equal weights (TASK_UNIFORM prior).  The aggregate IG and DG
    should be non-negative when the observable oracle is the correct
    expectation over the information class.
    """
    contributions = tuple(contributions)
    n = len(contributions)
    if n == 0:
        return I34ScientificAggregate(
            condition=condition, task_count=0,
            expected_latent_value=0.0, expected_observable_value=0.0,
            expected_controller_value=0.0,
            information_gap=0.0, decision_gap=0.0, total_regret=0.0,
            tr_minus_ig_minus_dg=0.0, prior=prior)

    e_latent = sum(c.latent_optimal_value for c in contributions) / n
    e_observable = sum(c.observable_optimal_value for c in contributions) / n
    e_controller = sum(c.controller_value for c in contributions) / n
    ig = sum(c.information_gap_contribution for c in contributions) / n
    dg = sum(c.decision_gap_contribution for c in contributions) / n
    tr = sum(c.total_regret_contribution for c in contributions) / n
    residual = abs(tr - ig - dg)
    return I34ScientificAggregate(
        condition=condition, task_count=n,
        expected_latent_value=e_latent,
        expected_observable_value=e_observable,
        expected_controller_value=e_controller,
        information_gap=ig, decision_gap=dg, total_regret=tr,
        tr_minus_ig_minus_dg=residual, prior=prior)


def compute_paired_deltas(
    *,
    blind_contributions: Mapping[str, I34ScientificTaskContribution],
    aware_contributions: Mapping[str, I34ScientificTaskContribution],
    blind_costs: Mapping[str, float] | None = None,
    aware_costs: Mapping[str, float] | None = None,
) -> list[I34PairedDelta]:
    """Compute paired per-task deltas between blind and aware conditions.

    Only tasks present in both conditions are paired.  The primary
    scientific observation is d_i = DG_blind,i - DG_aware,i.

    Returns a list of I34PairedDelta, one per paired task.
    """
    blind_costs = blind_costs or {}
    aware_costs = aware_costs or {}
    paired_task_ids = sorted(set(blind_contributions) & set(aware_contributions))
    deltas: list[I34PairedDelta] = []
    for task_id in paired_task_ids:
        b = blind_contributions[task_id]
        a = aware_contributions[task_id]
        deltas.append(I34PairedDelta(
            task_id=task_id,
            delta_ig=b.information_gap_contribution - a.information_gap_contribution,
            delta_dg=b.decision_gap_contribution - a.decision_gap_contribution,
            delta_tr=b.total_regret_contribution - a.total_regret_contribution,
            delta_cost=blind_costs.get(task_id, 0.0) - aware_costs.get(task_id, 0.0),
        ))
    return deltas


def mean_delta_dg(deltas: Iterable[I34PairedDelta]) -> float:
    """Compute ΔDG = (1/N) Σ d_i where d_i = DG_blind,i - DG_aware,i."""
    deltas = tuple(deltas)
    if not deltas:
        return 0.0
    return sum(d.delta_dg for d in deltas) / len(deltas)


def mean_delta_ig(deltas: Iterable[I34PairedDelta]) -> float:
    """Compute ΔIG = (1/N) Σ (IG_blind,i - IG_aware,i)."""
    deltas = tuple(deltas)
    if not deltas:
        return 0.0
    return sum(d.delta_ig for d in deltas) / len(deltas)


def mean_delta_tr(deltas: Iterable[I34PairedDelta]) -> float:
    """Compute ΔTR = (1/N) Σ (TR_blind,i - TR_aware,i)."""
    deltas = tuple(deltas)
    if not deltas:
        return 0.0
    return sum(d.delta_tr for d in deltas) / len(deltas)


def verify_all_identities(
    contributions: Iterable[I34ScientificTaskContribution],
    tolerance: float = IDENTITY_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Verify TR = IG + DG for every task contribution.

    Returns (all_pass, list_of_failing_task_ids).
    """
    failures: list[str] = []
    for c in contributions:
        if not c.verify_identity(tolerance):
            failures.append(c.task_id)
    return (len(failures) == 0, failures)


def scoring_module_sha256() -> str:
    """Canonical SHA-256 of this module's source code."""
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
