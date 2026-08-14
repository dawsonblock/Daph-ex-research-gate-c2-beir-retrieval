"""Statistical analysis for V2B-I3.4.1 scientific evaluation.

Implements:
- Paired bootstrap (task-level resampling) for primary inference.
- Topology-cluster bootstrap (cluster-level resampling) for structural
  held-out inference.

Both methods preserve blind/aware pairing.  The primary success criterion
is LCB_95(ΔDG) > 0 from the paired bootstrap.

Schema identity: ``DAPH_V2B_I3_4_STATISTICAL_ANALYSIS_V1`` (frozen).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .i3_4_scientific_scoring import I34PairedDelta, mean_delta_dg

STATS_SCHEMA = "DAPH_V2B_I3_4_STATISTICAL_ANALYSIS_V1"
STATS_VERSION = 1

DEFAULT_ITERATIONS = 10_000
DEFAULT_CI_LEVEL = 0.95
DEFAULT_SEED = 42  # Frozen before evaluation


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap procedure."""

    method: str
    point_estimate: float
    ci_level: float
    lower_bound: float
    upper_bound: float
    iterations: int
    seed: int
    resampling_unit: str
    n_units: int
    bias: float = 0.0
    std_error: float = 0.0

    @property
    def ci_significant_positive(self) -> bool:
        """True if the lower bound is strictly positive."""
        return self.lower_bound > 0.0

    @property
    def ci_significant_negative(self) -> bool:
        """True if the upper bound is strictly negative."""
        return self.upper_bound < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "point_estimate": self.point_estimate,
            "ci_level": self.ci_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "iterations": self.iterations,
            "seed": self.seed,
            "resampling_unit": self.resampling_unit,
            "n_units": self.n_units,
            "bias": self.bias,
            "std_error": self.std_error,
            "ci_significant_positive": self.ci_significant_positive,
        }


def paired_bootstrap(
    deltas: Sequence[I34PairedDelta],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Paired bootstrap on per-task ΔDG values.

    Resampling unit: task.  Each resample draws N tasks with replacement
    and computes the mean ΔDG.  The CI is the percentile interval.

    The primary success criterion is lower_bound > 0.
    """
    n = len(deltas)
    if n == 0:
        return BootstrapResult(
            method="paired_bootstrap", point_estimate=0.0,
            ci_level=ci_level, lower_bound=0.0, upper_bound=0.0,
            iterations=iterations, seed=seed,
            resampling_unit="task", n_units=0)

    rng = random.Random(seed)
    point_estimate = mean_delta_dg(deltas)
    bootstrap_means: list[float] = []
    for _ in range(iterations):
        sampled = [deltas[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(mean_delta_dg(sampled))

    bootstrap_means.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(alpha / 2.0 * iterations)
    upper_idx = int((1.0 - alpha / 2.0) * iterations)
    lower_idx = min(lower_idx, iterations - 1)
    upper_idx = min(upper_idx, iterations - 1)
    lower = bootstrap_means[lower_idx]
    upper = bootstrap_means[upper_idx]
    bias = sum(bootstrap_means) / iterations - point_estimate
    std_error = (sum((m - point_estimate) ** 2 for m in bootstrap_means)
                 / iterations) ** 0.5
    return BootstrapResult(
        method="paired_bootstrap", point_estimate=point_estimate,
        ci_level=ci_level, lower_bound=lower, upper_bound=upper,
        iterations=iterations, seed=seed,
        resampling_unit="task", n_units=n,
        bias=bias, std_error=std_error)


def topology_cluster_bootstrap(
    deltas: Sequence[I34PairedDelta],
    *,
    cluster_map: Mapping[str, str],
    iterations: int = DEFAULT_ITERATIONS,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_SEED,
    estimand: str = "topology_uniform",
) -> BootstrapResult:
    """Topology-cluster bootstrap for structural held-out inference.

    Resampling unit: topology cluster (identified by transition_topology_sha256).
    Clusters are resampled with replacement.

    Two estimands are supported:

    - ``topology_uniform`` (default, primary for structural-generalization
      claims): compute the mean ΔDG within each topology cluster first, then
      bootstrap/average the cluster means equally.  This gives equal weight
      per topology, which is the correct estimand for "generalization across
      unseen topologies."

    - ``task_population`` (secondary): concatenate all tasks from selected
      clusters and compute the task-level mean.  This gives equal weight per
      task and is the correct estimand for "expected performance over the
      task population."

    The primary inferential CI for structural-generalization claims uses
    ``topology_uniform``.  Report ``task_population`` as a secondary metric.
    """
    n = len(deltas)
    if n == 0:
        return BootstrapResult(
            method="topology_cluster_bootstrap", point_estimate=0.0,
            ci_level=ci_level, lower_bound=0.0, upper_bound=0.0,
            iterations=iterations, seed=seed,
            resampling_unit="topology_cluster", n_units=0)

    # Group deltas by cluster.
    clusters: dict[str, list[I34PairedDelta]] = {}
    for d in deltas:
        cluster_id = cluster_map.get(d.task_id, "__UNKNOWN__")
        clusters.setdefault(cluster_id, []).append(d)

    cluster_ids = sorted(clusters.keys())
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return BootstrapResult(
            method="topology_cluster_bootstrap", point_estimate=0.0,
            ci_level=ci_level, lower_bound=0.0, upper_bound=0.0,
            iterations=iterations, seed=seed,
            resampling_unit="topology_cluster", n_units=0)

    # Precompute per-cluster mean ΔDG for topology_uniform estimand.
    cluster_means: dict[str, float] = {}
    for cid in cluster_ids:
        cluster_means[cid] = mean_delta_dg(clusters[cid])

    rng = random.Random(seed)

    if estimand == "topology_uniform":
        # Primary: equal weight per topology.
        # Point estimate = mean of cluster means.
        point_estimate = sum(cluster_means[cid] for cid in cluster_ids) / n_clusters

        bootstrap_means: list[float] = []
        for _ in range(iterations):
            sampled_cluster_means = [
                cluster_means[cluster_ids[rng.randrange(n_clusters)]]
                for _ in range(n_clusters)
            ]
            bootstrap_means.append(sum(sampled_cluster_means) / n_clusters)
    else:
        # Secondary: task-population (task-weighted).
        point_estimate = mean_delta_dg(deltas)

        bootstrap_means = []
        for _ in range(iterations):
            all_sampled: list[I34PairedDelta] = []
            for _ in range(n_clusters):
                cid = cluster_ids[rng.randrange(n_clusters)]
                all_sampled.extend(clusters[cid])
            bootstrap_means.append(mean_delta_dg(all_sampled))

    bootstrap_means.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(alpha / 2.0 * iterations)
    upper_idx = int((1.0 - alpha / 2.0) * iterations)
    lower_idx = min(lower_idx, iterations - 1)
    upper_idx = min(upper_idx, iterations - 1)
    lower = bootstrap_means[lower_idx]
    upper = bootstrap_means[upper_idx]
    bias = sum(bootstrap_means) / iterations - point_estimate
    std_error = (sum((m - point_estimate) ** 2 for m in bootstrap_means)
                 / iterations) ** 0.5
    return BootstrapResult(
        method="topology_cluster_bootstrap", point_estimate=point_estimate,
        ci_level=ci_level, lower_bound=lower, upper_bound=upper,
        iterations=iterations, seed=seed,
        resampling_unit="topology_cluster", n_units=n_clusters,
        bias=bias, std_error=std_error)


def bonferroni_correct(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Bonferroni correction to a family of p-values.

    Returns a list of booleans indicating whether each test is significant
    after correction.
    """
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p <= threshold for p in p_values]


def stats_module_sha256() -> str:
    """Canonical SHA-256 of this module's source code."""
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
