"""PAV bootstrap ensemble (PAV_V2).

Only build if PAV_V1 (single GBT) is insufficient.
5-model bootstrap ensemble for uncertainty estimation.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from daph.pav.model import extract_pav_features, LearnedPAV


class EnsemblePAV:
    """PAV_V2: 5-model bootstrap ensemble.

    Provides mean and std for uncertainty estimation.
    Only train if PAV_V1 helps but needs uncertainty.
    """

    def __init__(self, models: list[GradientBoostingRegressor], feature_keys: list[str]):
        self.models = models
        self.feature_keys = feature_keys
        self.n_models = len(models)

    @classmethod
    def train(
        cls,
        records: list[dict],
        feature_keys: list[str] | None = None,
        n_models: int = 5,
        seed: int = 42,
    ) -> "EnsemblePAV":
        if feature_keys is None:
            if not records:
                raise ValueError("No records to train on")
            sample_feats = extract_pav_features(
                records[0]["features_before"], records[0]["action"],
            )
            feature_keys = sorted(sample_feats.keys())

        rng = np.random.RandomState(seed)
        n = len(records)
        models = []

        for i in range(n_models):
            indices = rng.choice(n, size=n, replace=True)
            X = []
            y = []
            for idx in indices:
                r = records[idx]
                feats = extract_pav_features(r["features_before"], r["action"])
                X.append([feats.get(k, 0) for k in feature_keys])
                target = r.get("progress_components", {}).get("progress", 0.0)
                y.append(target)

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed + i,
            )
            model.fit(np.array(X), np.array(y))
            models.append(model)

        return cls(models=models, feature_keys=feature_keys)

    def predict_with_uncertainty(
        self, state_features: dict, action: str,
    ) -> tuple[float, float]:
        """Return (mean, std) across ensemble."""
        feats = extract_pav_features(state_features, action)
        X = np.array([[feats.get(k, 0) for k in self.feature_keys]])
        preds = [float(m.predict(X)[0]) for m in self.models]
        return float(np.mean(preds)), float(np.std(preds))
