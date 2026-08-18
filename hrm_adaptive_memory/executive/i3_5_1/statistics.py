"""Statistical analysis for I3.5.1 factorial experiment.

Primary estimand: governor effect on DG
  ΔDG_gov|aware = DG_{AWARE,OFF} - DG_{AWARE,ON}

Secondary estimands:
  ΔDG_gov|blind = DG_{BLIND,OFF} - DG_{BLIND,ON}
  Δ_interaction = ΔDG_gov|blind - ΔDG_gov|aware

Bootstrap methods:
  - Paired task bootstrap (resample tasks)
  - Topology-cluster bootstrap (resample topology clusters)

Schema identity: DAPH_V2B_I3_5_1_STATISTICS_V1 (frozen).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..i3_4_statistical_analysis import (
    BootstrapResult, paired_bootstrap as _paired_bootstrap,
    topology_cluster_bootstrap as _topology_cluster_bootstrap,
)

STATISTICS_SCHEMA = "DAPH_V2B_I3_5_1_STATISTICS_V1"
STATISTICS_VERSION = 1
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95


@dataclass(frozen=True)
class FactorialStats:
    """Statistical results for the factorial experiment."""
    n_tasks: int
    # Primary: governor effect on DG (aware)
    mean_delta_dg_gov_aware: float
    ci_gov_aware: tuple[float, float]
    # Secondary: governor effect on DG (blind)
    mean_delta_dg_gov_blind: float
    ci_gov_blind: tuple[float, float]
    # State effect (no governor)
    mean_delta_dg_state_no_gov: float
    ci_state_no_gov: tuple[float, float]
    # State effect (with governor)
    mean_delta_dg_state_gov: float
    ci_state_gov: tuple[float, float]
    # Interaction
    mean_delta_interaction: float
    ci_interaction: tuple[float, float]
    # Topology-cluster bootstrap for primary
    topo_mean_gov_aware: float
    topo_ci_gov_aware: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STATISTICS_SCHEMA,
            "schema_version": STATISTICS_VERSION,
            "n_tasks": self.n_tasks,
            "mean_delta_dg_gov_aware": self.mean_delta_dg_gov_aware,
            "ci_gov_aware": list(self.ci_gov_aware),
            "mean_delta_dg_gov_blind": self.mean_delta_dg_gov_blind,
            "ci_gov_blind": list(self.ci_gov_blind),
            "mean_delta_dg_state_no_gov": self.mean_delta_dg_state_no_gov,
            "ci_state_no_gov": list(self.ci_state_no_gov),
            "mean_delta_dg_state_gov": self.mean_delta_dg_state_gov,
            "ci_state_gov": list(self.ci_state_gov),
            "mean_delta_interaction": self.mean_delta_interaction,
            "ci_interaction": list(self.ci_interaction),
            "topo_mean_gov_aware": self.topo_mean_gov_aware,
            "topo_ci_gov_aware": list(self.topo_ci_gov_aware),
        }


def compute_factorial_stats(
    contributions: list,
    *,
    topology_map: dict[str, str] | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> FactorialStats:
    """Compute all factorial statistics from contributions."""
    n = len(contributions)

    # Primary: governor effect on DG (aware)
    deltas_gov_aware = [c.delta_dg_gov_aware for c in contributions]
    mean_gov_aware = sum(deltas_gov_aware) / n
    ci_gov_aware = _bootstrap_ci(deltas_gov_aware, iterations, seed)

    # Secondary: governor effect on DG (blind)
    deltas_gov_blind = [c.delta_dg_gov_blind for c in contributions]
    mean_gov_blind = sum(deltas_gov_blind) / n
    ci_gov_blind = _bootstrap_ci(deltas_gov_blind, iterations, seed + 1)

    # State effect (no governor)
    deltas_state_no_gov = [c.delta_dg_state_no_gov for c in contributions]
    mean_state_no_gov = sum(deltas_state_no_gov) / n
    ci_state_no_gov = _bootstrap_ci(deltas_state_no_gov, iterations, seed + 2)

    # State effect (with governor)
    deltas_state_gov = [c.delta_dg_state_gov for c in contributions]
    mean_state_gov = sum(deltas_state_gov) / n
    ci_state_gov = _bootstrap_ci(deltas_state_gov, iterations, seed + 3)

    # Interaction
    deltas_interaction = [c.delta_interaction for c in contributions]
    mean_interaction = sum(deltas_interaction) / n
    ci_interaction = _bootstrap_ci(deltas_interaction, iterations, seed + 4)

    # Topology-cluster bootstrap for primary (if topology map provided)
    if topology_map:
        topo_result = _topology_cluster_bootstrap(
            deltas_gov_aware,
            [topology_map.get(c.task_id, "UNKNOWN") for c in contributions],
            iterations=iterations,
            seed=seed + 5,
        )
        topo_mean = topo_result.mean
        topo_ci = (topo_result.ci_lower, topo_result.ci_upper)
    else:
        topo_mean = mean_gov_aware
        topo_ci = ci_gov_aware

    return FactorialStats(
        n_tasks=n,
        mean_delta_dg_gov_aware=mean_gov_aware,
        ci_gov_aware=ci_gov_aware,
        mean_delta_dg_gov_blind=mean_gov_blind,
        ci_gov_blind=ci_gov_blind,
        mean_delta_dg_state_no_gov=mean_state_no_gov,
        ci_state_no_gov=ci_state_no_gov,
        mean_delta_dg_state_gov=mean_state_gov,
        ci_state_gov=ci_state_gov,
        mean_delta_interaction=mean_interaction,
        ci_interaction=ci_interaction,
        topo_mean_gov_aware=topo_mean,
        topo_ci_gov_aware=topo_ci,
    )


def _bootstrap_ci(
    deltas: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Paired bootstrap CI."""
    n = len(deltas)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - CI_LEVEL) / 2 * iterations)
    hi_idx = int((1 + CI_LEVEL) / 2 * iterations)
    return (means[lo_idx], means[hi_idx])


def save_stats(
    stats: FactorialStats,
    path: str | Path,
    *,
    experiment_identity_sha256: str,
    source_results_sha256: str,
    source_scores_sha256: str,
    statistics_implementation_sha256: str,
) -> str:
    """Save stats with provenance. Return file SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.as_dict()
    payload["experiment_identity_sha256"] = experiment_identity_sha256
    payload["source_results_sha256"] = source_results_sha256
    payload["source_scores_sha256"] = source_scores_sha256
    payload["statistics_implementation_sha256"] = statistics_implementation_sha256
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
