#!/usr/bin/env python3
"""Train the I3.5 model ladder on causal action data.

Model ladder:
  B0:      Global action prior (mean Q per action across all states)
  B1:      Phase-conditioned prior (mean Q per action × phase)
  LINEAR:  Linear regression Q(s,a) from state features
  GBT:     Gradient-boosted tree Q(s,a) from state features
  Q_OBS:   GBT trained on observational data (policy behavior)
  Q_CAUSAL: GBT trained on causal data (forced actions)

The key promotion comparison is:
  Q_phi > B0  (paired bootstrap CI excluding zero)

Output:
  experiments/i3_5/models/model_ladder_v1.json
  experiments/i3_5/models/model_ladder_v1_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

ACTION_NAMES = ["ANSWER", "DEFER", "RETRIEVE", "VERIFY", "SEARCH_MORE", "REASON_MORE"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}

FEATURE_KEYS = [
    "n_live", "n_eliminated", "n_untested", "n_total_hypotheses",
    "n_visible_evidence", "n_hidden_evidence",
    "n_verified", "n_supporting", "n_contradicting", "n_stale",
    "retrieval_remaining", "search_remaining", "verify_remaining", "steps_remaining",
    "can_retrieve", "can_search", "can_verify",
    "searched", "reasoning_complete",
    "same_action_run_length", "retrieval_count", "search_count", "verify_count",
]


def extract_features(record: dict) -> np.ndarray:
    """Extract numeric feature vector from a causal record."""
    sf = record["state_features"]
    feats = []
    for key in FEATURE_KEYS:
        val = sf.get(key, 0)
        if val is None:
            val = 0
        if isinstance(val, bool):
            val = int(val)
        feats.append(float(val))
    # Add one-hot action encoding
    action = record["forced_action"]
    action_idx = ACTION_TO_IDX.get(action, -1)
    for i in range(len(ACTION_NAMES)):
        feats.append(1.0 if i == action_idx else 0.0)
    return np.array(feats)


def extract_action_features(record: dict) -> np.ndarray:
    """Extract only the state features (without action one-hot)."""
    sf = record["state_features"]
    feats = []
    for key in FEATURE_KEYS:
        val = sf.get(key, 0)
        if val is None:
            val = 0
        if isinstance(val, bool):
            val = int(val)
        feats.append(float(val))
    return np.array(feats)


def get_action_onehot(action: str) -> np.ndarray:
    """One-hot encode an action."""
    vec = np.zeros(len(ACTION_NAMES))
    idx = ACTION_TO_IDX.get(action, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class B0GlobalPrior:
    """B0: Global action prior. Q(a) = mean U over all (s,a) pairs."""

    def __init__(self):
        self.q_values: dict[str, float] = {}

    def fit(self, records: list[dict]) -> None:
        by_action: dict[str, list[float]] = defaultdict(list)
        for r in records:
            by_action[r["forced_action"]].append(r["terminal_utility"])
        self.q_values = {
            a: sum(us) / len(us) if us else 0.0
            for a, us in by_action.items()
        }

    def predict(self, state_features: dict, action: str) -> float:
        return self.q_values.get(action, 0.0)

    def predict_all(self, state_features: dict) -> dict[str, float]:
        return {a: self.q_values.get(a, 0.0) for a in ACTION_NAMES}

    def as_dict(self) -> dict:
        return {"model": "B0", "q_values": self.q_values}


class B1PhaseConditioned:
    """B1: Phase-conditioned prior. Q(phase, a) = mean U for (phase, a)."""

    def __init__(self):
        self.q_table: dict[str, dict[str, float]] = {}

    def _get_phase(self, state_features: dict) -> str:
        """Derive a coarse phase from state features."""
        n_live = state_features.get("n_live", 0)
        n_eliminated = state_features.get("n_eliminated", 0)
        n_untested = state_features.get("n_untested", 0)
        if n_eliminated > 0 and n_live == 0:
            return "ALL_ELIMINATED"
        elif n_live == 1 and n_eliminated == 1:
            return "ONE_LIVE"
        elif n_live == 2:
            return "TWO_LIVE"
        elif n_untested > 0:
            return "UNTESTED"
        else:
            return "OTHER"

    def fit(self, records: list[dict]) -> None:
        by_phase_action: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in records:
            phase = self._get_phase(r["state_features"])
            by_phase_action[phase][r["forced_action"]].append(r["terminal_utility"])

        self.q_table = {}
        for phase, action_utils in by_phase_action.items():
            self.q_table[phase] = {
                a: sum(us) / len(us) if us else 0.0
                for a, us in action_utils.items()
            }

    def predict(self, state_features: dict, action: str) -> float:
        phase = self._get_phase(state_features)
        return self.q_table.get(phase, {}).get(action, 0.0)

    def predict_all(self, state_features: dict) -> dict[str, float]:
        phase = self._get_phase(state_features)
        phase_q = self.q_table.get(phase, {})
        return {a: phase_q.get(a, 0.0) for a in ACTION_NAMES}

    def as_dict(self) -> dict:
        return {"model": "B1", "q_table": self.q_table}


class StateActionModel:
    """Wrapper for sklearn models that predict Q(s,a) from state+action features."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, state_features: dict, action: str) -> float:
        """Predict Q(s,a) for a single state-action pair."""
        sf_vec = np.array([
            float(state_features.get(k, 0) if state_features.get(k) is not None else 0)
            if not isinstance(state_features.get(k, 0), bool)
            else float(int(state_features.get(k, False)))
            for k in FEATURE_KEYS
        ])
        action_vec = get_action_onehot(action)
        x = np.concatenate([sf_vec, action_vec]).reshape(1, -1)
        return float(self.model.predict(x)[0])

    def predict_all(self, state_features: dict) -> dict[str, float]:
        """Predict Q(s,a) for all actions."""
        sf_vec = np.array([
            float(state_features.get(k, 0) if state_features.get(k) is not None else 0)
            if not isinstance(state_features.get(k, 0), bool)
            else float(int(state_features.get(k, False)))
            for k in FEATURE_KEYS
        ])
        results = {}
        for action in ACTION_NAMES:
            action_vec = get_action_onehot(action)
            x = np.concatenate([sf_vec, action_vec]).reshape(1, -1)
            results[action] = float(self.model.predict(x)[0])
        return results

    def as_dict(self) -> dict:
        return {"model": self.name}


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def load_causal_data(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def top_action_accuracy(predictions: list[dict[str, float]], correct_actions: list[str]) -> float:
    """Fraction of cases where argmax Q(s,a) = correct action."""
    correct = 0
    for preds, correct_a in zip(predictions, correct_actions):
        predicted_a = max(preds, key=preds.get) if preds else "UNKNOWN"
        if predicted_a == correct_a:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def action_regret(predictions: list[dict[str, float]], correct_actions: list[str],
                  q_star: list[dict[str, float]]) -> float:
    """Mean regret: Q*(s, correct) - Q*(s, predicted)."""
    total_regret = 0.0
    for preds, correct_a, qstar in zip(predictions, correct_actions, q_star):
        predicted_a = max(preds, key=preds.get) if preds else "UNKNOWN"
        if predicted_a != correct_a:
            total_regret += qstar.get(correct_a, 0.0) - qstar.get(predicted_a, 0.0)
    return total_regret / len(predictions) if predictions else 0.0


def paired_bootstrap_ci(differences: list[float], n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval."""
    if not differences:
        return 0.0, 0.0, 0.0
    diffs = np.array(differences)
    n = len(diffs)
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_means.append(np.mean(diffs[idx]))
    mean_diff = float(np.mean(diffs))
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_diff, lower, upper


def evaluate_model(model, records: list[dict], model_name: str) -> dict:
    """Evaluate a model on the causal records."""
    predictions = []
    correct_actions = []
    q_star = []
    utilities = []

    for r in records:
        preds = model.predict_all(r["state_features"])
        predictions.append(preds)
        correct_actions.append(r["correct_first_action"])
        q_star.append({r["forced_action"]: r["terminal_utility"]})
        utilities.append(r["terminal_utility"])

    # Top-action accuracy
    top_acc = top_action_accuracy(predictions, correct_actions)

    # Mean predicted Q for the selected action
    mean_selected_q = np.mean([
        max(preds.values()) if preds else 0.0
        for preds in predictions
    ])

    # Mean actual utility
    mean_utility = np.mean(utilities)

    return {
        "model": model_name,
        "top_action_accuracy": round(top_acc, 4),
        "mean_selected_q": round(float(mean_selected_q), 4),
        "mean_actual_utility": round(float(mean_utility), 4),
        "n_records": len(records),
    }


def cross_val_predict_q(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Cross-validated predictions to avoid overfitting."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions


def main():
    output_dir = REPO_ROOT / "experiments/i3_5/models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load causal data
    causal_path = REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1.jsonl"
    records = load_causal_data(causal_path)
    print(f"Loaded {len(records)} causal records")

    # Prepare feature matrix
    X = np.array([extract_features(r) for r in records])
    y = np.array([r["terminal_utility"] for r in records])
    print(f"Feature matrix: {X.shape}")

    # ---- B0: Global prior ----
    print("\n--- B0: Global Action Prior ---")
    b0 = B0GlobalPrior()
    b0.fit(records)
    print(f"  Q values: {b0.q_values}")
    b0_eval = evaluate_model(b0, records, "B0")
    print(f"  Top-action accuracy: {b0_eval['top_action_accuracy']}")
    print(f"  Mean selected Q: {b0_eval['mean_selected_q']}")

    # ---- B1: Phase-conditioned ----
    print("\n--- B1: Phase-Conditioned Prior ---")
    b1 = B1PhaseConditioned()
    b1.fit(records)
    print(f"  Phases: {list(b1.q_table.keys())}")
    for phase, q in b1.q_table.items():
        print(f"    {phase}: {q}")
    b1_eval = evaluate_model(b1, records, "B1")
    print(f"  Top-action accuracy: {b1_eval['top_action_accuracy']}")

    # ---- LINEAR: Ridge regression ----
    print("\n--- LINEAR: Ridge Regression ---")
    linear_model = StateActionModel(Ridge(alpha=1.0), "LINEAR")
    linear_cv_preds = cross_val_predict_q(Ridge(alpha=1.0), X, y, n_splits=5)
    linear_mse = mean_squared_error(y, linear_cv_preds)
    print(f"  CV MSE: {linear_mse:.4f}")

    # Fit on full data for evaluation
    linear_model.fit(X, y)
    linear_eval = evaluate_model(linear_model, records, "LINEAR")
    print(f"  Top-action accuracy: {linear_eval['top_action_accuracy']}")

    # ---- GBT (Q_CAUSAL): Gradient-boosted tree ----
    print("\n--- GBT (Q_CAUSAL): Gradient-Boosted Tree ---")
    gbt_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    }
    gbt_model = StateActionModel(GradientBoostingRegressor(**gbt_params), "Q_CAUSAL")
    gbt_cv_preds = cross_val_predict_q(
        GradientBoostingRegressor(**gbt_params), X, y, n_splits=5
    )
    gbt_mse = mean_squared_error(y, gbt_cv_preds)
    print(f"  CV MSE: {gbt_mse:.4f}")

    gbt_model.fit(X, y)
    gbt_eval = evaluate_model(gbt_model, records, "Q_CAUSAL")
    print(f"  Top-action accuracy: {gbt_eval['top_action_accuracy']}")

    # ---- Q_OBS: Observational model (train on "natural" policy data) ----
    # For observational data, we simulate a policy that always picks the
    # "obvious" action: ANSWER if n_supporting > 0, DEFER otherwise.
    # This creates a biased dataset where only certain (s,a) pairs appear.
    print("\n--- Q_OBS: Observational GBT (biased policy) ---")
    obs_records = []
    for r in records:
        sf = r["state_features"]
        # Simulate observational policy: pick "obvious" action
        if sf.get("n_supporting", 0) > 0 and sf.get("n_contradicting", 0) == 0:
            obs_action = "ANSWER"
        elif sf.get("n_eliminated", 0) > 0 and sf.get("n_live", 0) == 0:
            obs_action = "DEFER"
        elif sf.get("can_retrieve", False):
            obs_action = "RETRIEVE"
        elif sf.get("can_verify", False):
            obs_action = "VERIFY"
        elif sf.get("can_search", False):
            obs_action = "SEARCH_MORE"
        else:
            obs_action = "DEFER"

        # Only keep records where forced_action matches the observational policy
        if r["forced_action"] == obs_action:
            obs_records.append(r)

    print(f"  Observational records: {len(obs_records)} (of {len(records)} causal)")
    X_obs = np.array([extract_features(r) for r in obs_records])
    y_obs = np.array([r["terminal_utility"] for r in obs_records])

    qobs_model = StateActionModel(
        GradientBoostingRegressor(**gbt_params), "Q_OBS"
    )
    if len(obs_records) > 10:
        qobs_cv_preds = cross_val_predict_q(
            GradientBoostingRegressor(**gbt_params), X_obs, y_obs, n_splits=5
        )
        qobs_mse = mean_squared_error(y_obs, qobs_cv_preds)
        print(f"  CV MSE: {qobs_mse:.4f}")
        qobs_model.fit(X_obs, y_obs)
        qobs_eval = evaluate_model(qobs_model, records, "Q_OBS")
    else:
        qobs_eval = {"model": "Q_OBS", "top_action_accuracy": 0.0, "mean_selected_q": 0.0,
                     "mean_actual_utility": 0.0, "n_records": len(obs_records), "note": "insufficient data"}
    print(f"  Top-action accuracy: {qobs_eval['top_action_accuracy']}")

    # ---- Promotion comparison: paired bootstrap ----
    print("\n=== Promotion Comparisons ===")

    # For each model, compute per-record predicted Q for the selected action
    # and compare against B0
    model_preds = {
        "B0": b0, "B1": b1, "LINEAR": linear_model,
        "Q_CAUSAL": gbt_model, "Q_OBS": qobs_model,
    }

    # Compute per-task best Q for each model
    # Group records by task_id
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)

    comparisons = []
    for model_name, model in model_preds.items():
        if model_name == "B0":
            continue

        # Per-task: predicted utility of model's selected action vs B0's selected action
        diffs = []
        for task_id, task_records in by_task.items():
            # Get state features (same for all records of this task at step 0)
            sf = task_records[0]["state_features"]

            # Model's prediction
            model_preds_task = model.predict_all(sf)
            model_action = max(model_preds_task, key=model_preds_task.get)

            # B0's prediction
            b0_preds_task = b0.predict_all(sf)
            b0_action = max(b0_preds_task, key=b0_preds_task.get)

            # Actual causal utility for each
            model_util = 0.0
            b0_util = 0.0
            for r in task_records:
                if r["forced_action"] == model_action:
                    model_util = r["terminal_utility"]
                if r["forced_action"] == b0_action:
                    b0_util = r["terminal_utility"]

            diffs.append(model_util - b0_util)

        mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(diffs)
        sig = ci_lo > 0 or ci_hi < 0
        comparisons.append({
            "comparison": f"{model_name} - B0",
            "mean_diff": round(mean_diff, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "significant": sig,
            "n_tasks": len(by_task),
        })
        print(f"  {model_name} - B0: mean={mean_diff:+.4f}, CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], significant={sig}")

    # ---- Save model ladder ----
    model_ladder = {
        "B0": b0.as_dict(),
        "B1": b1.as_dict(),
        "LINEAR": linear_eval,
        "Q_CAUSAL": gbt_eval,
        "Q_OBS": qobs_eval,
        "evaluations": {
            "B0": b0_eval,
            "B1": b1_eval,
            "LINEAR": linear_eval,
            "Q_CAUSAL": gbt_eval,
            "Q_OBS": qobs_eval,
        },
        "comparisons": comparisons,
        "feature_keys": FEATURE_KEYS + [f"action_{a}" for a in ACTION_NAMES],
        "action_names": ACTION_NAMES,
        "n_records": len(records),
        "n_observational_records": len(obs_records),
        "cv_mse": {
            "LINEAR": round(float(linear_mse), 6),
            "Q_CAUSAL": round(float(gbt_mse), 6),
            "Q_OBS": round(float(qobs_mse), 6) if len(obs_records) > 10 else None,
        },
    }

    # Compute SHA
    ladder_content = json.dumps(model_ladder, sort_keys=True)
    ladder_sha = hashlib.sha256(ladder_content.encode()).hexdigest()

    manifest = {
        "model_ladder_sha256": ladder_sha,
        "n_records": len(records),
        "n_observational_records": len(obs_records),
        "feature_dim": X.shape[1],
        "gbt_params": gbt_params,
        "causal_data_sha256": json.loads(
            open(REPO_ROOT / "experiments/i3_5/causal/causal_actions_v1_manifest.json").read()
        ).get("causal_data_sha256"),
    }

    # Save
    ladder_path = output_dir / "model_ladder_v1.json"
    with open(ladder_path, "w") as f:
        json.dump(model_ladder, f, indent=2, sort_keys=True)
    print(f"\nWritten model ladder to {ladder_path}")

    manifest_path = output_dir / "model_ladder_v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Written manifest to {manifest_path}")

    # Summary
    print(f"\n=== Model Ladder Summary ===")
    print(f"  {'Model':15s} {'Top-Acc':>8s} {'Mean Q':>8s} {'Mean U':>8s}")
    for name, ev in model_ladder["evaluations"].items():
        print(f"  {name:15s} {ev['top_action_accuracy']:8.4f} {ev['mean_selected_q']:8.4f} {ev['mean_actual_utility']:8.4f}")

    print(f"\n=== Promotion Gates ===")
    for c in comparisons:
        gate = "PASS" if c["significant"] and c["mean_diff"] > 0 else "FAIL"
        print(f"  {c['comparison']:20s}: {c['mean_diff']:+.4f} CI=[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] -> {gate}")


if __name__ == "__main__":
    main()
