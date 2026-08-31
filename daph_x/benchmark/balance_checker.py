"""Balance checker for DAPH-X M4 benchmark.

Computes harm rate conditioned on every coarse feature.
Flags any single feature with AUROC > 0.80 when used alone
to classify harm. This prevents template-classification shortcuts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def compute_feature_auroc(values: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC for a single feature against binary labels.

    Uses the rank-based formula. Returns 0.5 if all labels are the same.
    """
    if len(np.unique(labels)) < 2:
        return 0.5

    # Rank-based AUROC
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    # Handle ties by assigning average rank
    unique_vals = np.unique(values)
    for v in unique_vals:
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    sum_ranks_pos = ranks[labels == 1].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


def check_balance(
    features: list[dict],
    labels: list[int],
    threshold: float = 0.80,
) -> dict:
    """Check if any single feature almost deterministically identifies harm.

    For each feature, compute AUROC when used alone to classify harm.
    Flag any feature with AUROC > threshold.

    Args:
        features: List of feature dicts (one per intervention)
        labels: List of harm labels (1 = harmful, 0 = safe)
        threshold: AUROC threshold for flagging

    Returns:
        Dict with per-feature AUROC and pass/fail status
    """
    labels_arr = np.array(labels, dtype=int)
    feature_names = set()
    for f in features:
        feature_names.update(f.keys())

    feature_aurocs = {}
    flagged_features = []

    for fname in sorted(feature_names):
        values = []
        for f in features:
            v = f.get(fname, 0)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            elif isinstance(v, str):
                # Skip string features (would need encoding)
                values = None
                break
            values.append(float(v))

        if values is None or len(values) != len(labels_arr):
            continue

        values_arr = np.array(values, dtype=float)
        auroc = compute_feature_auroc(values_arr, labels_arr)
        feature_aurocs[fname] = round(auroc, 4)

        if auroc > threshold:
            flagged_features.append({
                "feature": fname,
                "auroc": round(auroc, 4),
                "harm_rate_when_high": float(labels_arr[values_arr > np.median(values_arr)].mean())
                    if len(values_arr) > 0 else 0.0,
                "harm_rate_when_low": float(labels_arr[values_arr <= np.median(values_arr)].mean())
                    if len(values_arr) > 0 else 0.0,
            })

    # Compute harm rate conditioned on each discrete feature value
    conditional_rates = {}
    for fname in sorted(feature_names):
        rates = {}
        for f, l in zip(features, labels):
            v = f.get(fname)
            if v is None:
                continue
            v_str = str(v)
            if v_str not in rates:
                rates[v_str] = {"harm": 0, "total": 0}
            rates[v_str]["total"] += 1
            if l == 1:
                rates[v_str]["harm"] += 1
        if rates:
            conditional_rates[fname] = {
                v: {
                    "harm_rate": r["harm"] / max(1, r["total"]),
                    "n": r["total"],
                }
                for v, r in sorted(rates.items())
            }

    return {
        "feature_aurocs": feature_aurocs,
        "flagged_features": flagged_features,
        "conditional_harm_rates": conditional_rates,
        "threshold": threshold,
        "passed": len(flagged_features) == 0,
    }
