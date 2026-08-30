"""Q_V3R3: Uncertainty-aware Q model with OOD gating.

Improvements over Q_V3R2-A:
1. Bootstrap ensemble of GBTs for epistemic uncertainty
2. LCB-based authority: force only when LCB(a) = Q_hat(a) - lambda*sigma(a)
   remains dominant
3. OOD support-density gating: refuse to force when state is far from
   training support
4. New D1 DEFER-ready training stratum: terminal DEFER states where
   continuation is legal but causally dominated

The Q_V3R3 model is a SEPARATE candidate. Q_V3R2-A is untouched as the
historical control.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class QEnsemble:
    """Bootstrap ensemble of GBT regressors for uncertainty-aware Q estimation.

    Attributes:
        models: List of fitted GBT models
        feature_keys: Ordered feature names
        n_estimators: Number of bootstrap models
        lambda_lcb: LCB penalty coefficient
        ood_threshold: Maximum distance for in-support classification
        support_centroids: Training cluster centroids for OOD detection
        support_labels: Cluster labels for each training sample
    """
    models: list[GradientBoostingRegressor]
    feature_keys: list[str]
    n_estimators: int
    lambda_lcb: float
    ood_threshold: float
    support_centroids: np.ndarray | None
    support_labels: np.ndarray | None

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Mean prediction across ensemble."""
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Standard deviation across ensemble (epistemic uncertainty)."""
        preds = np.array([m.predict(X) for m in self.models])
        return preds.std(axis=0)

    def predict_lcb(self, X: np.ndarray, lambda_lcb: float | None = None) -> np.ndarray:
        """Lower confidence bound: Q_hat - lambda * sigma."""
        lam = lambda_lcb if lambda_lcb is not None else self.lambda_lcb
        return self.predict_mean(X) - lam * self.predict_std(X)

    def support_density(self, X: np.ndarray) -> np.ndarray:
        """Compute support-density score for each sample.

        Returns the negative mean distance to the k nearest training
        centroids. Higher (closer to 0) = more in-support.
        """
        if self.support_centroids is None:
            return np.zeros(len(X))

        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X, self.support_centroids)
        # Mean distance to 5 nearest centroids
        k = min(5, len(self.support_centroids))
        nearest = np.sort(dists, axis=1)[:, :k]
        return -nearest.mean(axis=1)

    def is_in_support(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Boolean mask: True if sample is in training support."""
        thresh = threshold if threshold is not None else self.ood_threshold
        return self.support_density(X) >= -thresh

    def predict_q_values(
        self,
        feature_vectors: dict[str, np.ndarray],
        legal_actions: list[str],
        use_lcb: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Predict Q values for each action at each state.

        Args:
            feature_vectors: action -> feature matrix (n_states, n_features)
            legal_actions: Actions to evaluate
            use_lcb: If True, use LCB; if False, use mean

        Returns:
            action -> {"mean": float, "std": float, "lcb": float}
        """
        results = {}
        for action in legal_actions:
            if action not in feature_vectors:
                continue
            X = feature_vectors[action]
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            mean = self.predict_mean(X)[0]
            std = self.predict_std(X)[0]
            lcb = mean - self.lambda_lcb * std

            results[action] = {
                "mean": float(mean),
                "std": float(std),
                "lcb": float(lcb),
            }

        return results

    def save(self, path: Path):
        """Save ensemble to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "QEnsemble":
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def train_q_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_keys: list[str],
    n_estimators: int = 20,
    gbt_params: dict | None = None,
    lambda_lcb: float = 1.0,
    ood_threshold: float = 5.0,
    n_support_clusters: int = 50,
    random_state: int = 42,
) -> QEnsemble:
    """Train a bootstrap ensemble of GBT regressors.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        feature_keys: Ordered feature names
        n_estimators: Number of bootstrap models
        gbt_params: GBT hyperparameters
        lambda_lcb: LCB penalty coefficient
        ood_threshold: OOD distance threshold
        n_support_clusters: Number of clusters for support estimation
        random_state: Random seed

    Returns:
        Fitted QEnsemble
    """
    if gbt_params is None:
        gbt_params = dict(n_estimators=200, max_depth=4)

    rng = np.random.RandomState(random_state)
    n_samples = len(X_train)

    models = []
    for i in range(n_estimators):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        params = {**gbt_params, "random_state": random_state + i}
        model = GradientBoostingRegressor(**params)
        model.fit(X_boot, y_boot)
        models.append(model)

    # Compute support centroids using KMeans
    from sklearn.cluster import KMeans
    n_clusters = min(n_support_clusters, len(X_train))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    support_centroids = kmeans.cluster_centers_
    support_labels = kmeans.labels_

    return QEnsemble(
        models=models,
        feature_keys=feature_keys,
        n_estimators=n_estimators,
        lambda_lcb=lambda_lcb,
        ood_threshold=ood_threshold,
        support_centroids=support_centroids,
        support_labels=support_labels,
    )


def uncertainty_gated_authority(
    q_values: dict[str, dict[str, float]],
    legal_actions: list[str],
    in_support: bool,
    threshold: float = 5.0,
    epsilon: float = 3.0,
    cert_answer: bool = False,
    cert_defer: bool = False,
) -> tuple[str | None, str]:
    """Uncertainty-gated authority decision.

    Force only when:
    1. Certificate passes (cert_answer or cert_defer)
    2. LCB gap >= threshold
    3. State is in training support

    Args:
        q_values: action -> {"mean", "std", "lcb"}
        legal_actions: Legal actions
        in_support: Whether state is in training support
        threshold: Q gap threshold (frozen at 5.0)
        epsilon: Near-optimal epsilon (frozen at 3.0)
        cert_answer: ANSWER certificate passed
        cert_defer: DEFER certificate passed

    Returns:
        (forced_action, reason_code)
    """
    # Gate 1: OOD support
    if not in_support:
        return None, "OOD_GATED"

    # Filter to legal actions with Q values
    legal_q = {a: q_values[a] for a in legal_actions if a in q_values}
    if not legal_q:
        return None, "NO_Q_VALUES"

    # Use LCB for decision
    lcb_values = {a: q_values[a]["lcb"] for a in legal_q}
    sorted_actions = sorted(lcb_values.items(), key=lambda x: -x[1])

    best_action = sorted_actions[0][0]
    best_lcb = sorted_actions[0][1]
    second_lcb = sorted_actions[1][1] if len(sorted_actions) > 1 else best_lcb

    # Gate 2: LCB gap
    lcb_gap = best_lcb - second_lcb
    if lcb_gap < threshold:
        return None, "LCB_GAP_INSUFFICIENT"

    # Gate 3: sole near-optimal (using LCB)
    near_optimal = [a for a, lcb in lcb_values.items() if best_lcb - lcb <= epsilon]
    if len(near_optimal) > 1:
        return None, "NOT_SOLE_NEAR_OPTIMAL"

    # Gate 4: certificate
    if best_action == "ANSWER" and cert_answer:
        return "ANSWER", "CERT_LCB_OOD_PASS"
    if best_action == "DEFER" and cert_defer:
        return "DEFER", "CERT_LCB_OOD_PASS"

    return None, "CERT_MISMATCH"
