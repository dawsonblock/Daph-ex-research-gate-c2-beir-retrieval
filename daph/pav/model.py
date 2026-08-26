"""PAV learned model (PAV_V1: GradientBoostingRegressor).

Estimates step progress from state/action features.
Only train this if StructuralPAV (PAV_B0) is insufficient.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from daph.pav.types import PAVPrediction, PAVScoreResult
from daph.intervention.checkpoint import StateCheckpoint


def extract_pav_features(state_features: dict, action: str) -> dict:
    """Extract features for PAV prediction.

    These are the same features used by Q_CAUSAL_V1 plus
    progress-relevant features.
    """
    feats = {
        "n_live": state_features.get("n_live", 0),
        "n_eliminated": state_features.get("n_eliminated", 0),
        "n_untested": state_features.get("n_untested", 0),
        "n_total_hypotheses": state_features.get("n_total_hypotheses", 0),
        "n_visible_evidence": state_features.get("n_visible_evidence", 0),
        "n_verified": state_features.get("n_verified", 0),
        "n_supporting": state_features.get("n_supporting", 0),
        "n_contradicting": state_features.get("n_contradicting", 0),
        "n_stale": state_features.get("n_stale", 0),
        "retrieval_remaining": state_features.get("retrieval_remaining", 0),
        "search_remaining": state_features.get("search_remaining", 0),
        "verify_remaining": state_features.get("verify_remaining", 0),
        "steps_remaining": state_features.get("steps_remaining", 0),
        "can_retrieve": int(state_features.get("can_retrieve", False)),
        "can_search": int(state_features.get("can_search", False)),
        "can_verify": int(state_features.get("can_verify", False)),
        "searched": int(state_features.get("searched", False)),
        "reasoning_complete": int(state_features.get("reasoning_complete", False)),
        "same_action_run_length": state_features.get("same_action_run_length", 0),
        "retrieval_count": state_features.get("retrieval_count", 0),
        "search_count": state_features.get("search_count", 0),
        "verify_count": state_features.get("verify_count", 0),
    }
    for a in ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]:
        feats[f"a_{a}"] = int(action == a)
    # Interaction features
    feats["n_live_x_retrieve"] = feats["n_live"] * feats["a_RETRIEVE"]
    feats["n_live_x_verify"] = feats["n_live"] * feats["a_VERIFY"]
    feats["n_untested_x_verify"] = feats["n_untested"] * feats["a_VERIFY"]
    feats["steps_low_x_retrieve"] = int(feats["steps_remaining"] <= 3) * feats["a_RETRIEVE"]
    feats["repeated_action"] = int(feats["same_action_run_length"] >= 2)
    return feats


class LearnedPAV:
    """PAV_V1: GradientBoostingRegressor for step progress prediction.

    Trained on transition records where the target is the observed
    progress score (from StructuralPAV) or terminal delta-success.
    """

    def __init__(self, model: GradientBoostingRegressor, feature_keys: list[str]):
        self.model = model
        self.feature_keys = feature_keys

    @classmethod
    def train(
        cls,
        records: list[dict],
        feature_keys: list[str] | None = None,
    ) -> "LearnedPAV":
        """Train a PAV_V1 model from transition records."""
        if feature_keys is None:
            # Extract from first record
            if not records:
                raise ValueError("No records to train on")
            sample_feats = extract_pav_features(
                records[0]["features_before"], records[0]["action"],
            )
            feature_keys = sorted(sample_feats.keys())

        X = []
        y = []
        for r in records:
            feats = extract_pav_features(r["features_before"], r["action"])
            X.append([feats.get(k, 0) for k in feature_keys])
            # Target: progress score (or terminal delta if available)
            target = r.get("progress_components", {}).get("progress", 0.0)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)

        return cls(model=model, feature_keys=feature_keys)

    def save(self, path: str | Path, feature_schema: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.model, f)
        with open(p.parent / "pav_feature_schema.json", "w") as f:
            json.dump(feature_schema, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedPAV":
        p = Path(path)
        with open(p, "rb") as f:
            model = pickle.load(f)
        with open(p.parent / "pav_feature_schema.json") as f:
            schema = json.load(f)
        return cls(model=model, feature_keys=schema["feature_keys"])

    def predict(self, state_features: dict, action: str) -> float:
        feats = extract_pav_features(state_features, action)
        X = np.array([[feats.get(k, 0) for k in self.feature_keys]])
        return float(self.model.predict(X)[0])
