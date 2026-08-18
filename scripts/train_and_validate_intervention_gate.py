#!/usr/bin/env python3
"""5-Fold Task-Grouped Cross-Validation for Selective Governor Gate.

Trains and evaluates:
  1. Continuous ΔQ regression model: f_1(s) = E[ΔQ | s]
  2. Calibrated Harm probability classifier: f_2(s) = P(ΔQ < -5.0 | s)
  3. Rule-based gate with FOLD-ISOLATED rule discovery.

Key scientific fixes (I3.5.2a revision):
  - Rule discovery is now fold-isolated: rules are discovered from training
    data only within each fold, not from the full dataset.
  - Probability calibration via isotonic regression is applied to the harm
    classifier within each fold.
  - Brier score is reported alongside the base-rate Brier for context.
  - Expected Calibration Error (ECE) is reported.
  - The report explicitly states whether rules were fold-isolated or global.

Usage:
    python scripts/train_and_validate_intervention_gate.py \\
        --dataset experiments/v2b_i3_5_2/development/intervention_states_v1.jsonl \\
        --output experiments/v2b_i3_5_2/development/cross_validation_report_v1.json \\
        --rule-discovery-mode fold_isolated
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrm_adaptive_memory.executive.selective_governor.features import (
    FEATURE_NAMES,
    InterventionFeatures,
)
from hrm_adaptive_memory.executive.selective_governor.intervention_gate import (
    InterventionDecision,
    SelectiveGovernorGate,
)
from hrm_adaptive_memory.executive.selective_governor.model import (
    BaseInterventionPredictor,
    CalibratedLinearPredictor,
    InterventionPrediction,
    RuleBasedInterventionPredictor,
)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_spearman_corr(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation without external dependencies."""
    n = len(x)
    if n < 2:
        return 0.0

    def get_ranks(vals: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = get_ranks(x)
    ry = get_ranks(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def compute_roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """Compute ROC-AUC without external dependencies using rank sums."""
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    paired = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    for rank, (score, label) in enumerate(paired, start=1):
        if label == 1:
            rank_sum_pos += rank

    auc = (rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def compute_brier_score(y_true: list[int], y_prob: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(y_prob, y_true)) / len(y_true) if y_true else 0.0


def compute_ece(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with n_bins equal-width bins."""
    if not y_true:
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [lo <= p < hi for p in y_prob]
        count = sum(mask)
        if count == 0:
            continue
        avg_prob = sum(p for p, m in zip(y_prob, mask) if m) / count
        avg_true = sum(y for y, m in zip(y_true, mask) if m) / count
        ece += (count / n) * abs(avg_prob - avg_true)
    return ece


def compute_reliability_bins(
    y_true: list[int], y_prob: list[float], n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute reliability diagram bins for calibration visualization."""
    if not y_true:
        return []
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [lo <= p < hi for p in y_prob]
        count = sum(mask)
        if count == 0:
            bins.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "count": 0})
            continue
        avg_prob = sum(p for p, m in zip(y_prob, mask) if m) / count
        avg_true = sum(y for y, m in zip(y_true, mask) if m) / count
        bins.append({
            "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
            "count": count,
            "avg_predicted_prob": round(avg_prob, 4),
            "avg_actual_rate": round(avg_true, 4),
        })
    return bins


class IsotonicCalibrator:
    """Simple isotonic regression for probability calibration.

    Fits a non-decreasing mapping from raw probabilities to calibrated
    probabilities using the pool-adjacent-violators algorithm (PAVA).
    """

    def __init__(self):
        self._x_bins: list[float] = []
        self._y_bins: list[float] = []

    def fit(self, raw_probs: list[float], y_true: list[int]) -> None:
        """Fit isotonic regression: calibrated = iso(raw_prob)."""
        paired = sorted(zip(raw_probs, y_true))
        xs = [p[0] for p in paired]
        ys = [float(p[1]) for p in paired]

        # Pool-adjacent-violators algorithm
        blocks = [(xs[i], ys[i], 1) for i in range(len(xs))]  # (sum_x, sum_y, count)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            i = 0
            while i < len(blocks):
                if i + 1 < len(blocks):
                    avg_i = blocks[i][1] / blocks[i][2]
                    avg_j = blocks[i + 1][1] / blocks[i + 1][2]
                    if avg_i > avg_j:
                        # Merge
                        merged = (
                            blocks[i][0] + blocks[i + 1][0],
                            blocks[i][1] + blocks[i + 1][1],
                            blocks[i][2] + blocks[i + 1][2],
                        )
                        new_blocks.append(merged)
                        i += 2
                        changed = True
                        continue
                new_blocks.append(blocks[i])
                i += 1
            blocks = new_blocks

        self._x_bins = [b[0] / b[2] for b in blocks]
        self._y_bins = [b[1] / b[2] for b in blocks]

    def calibrate(self, raw_prob: float) -> float:
        """Map a raw probability to a calibrated probability."""
        if not self._x_bins:
            return raw_prob
        # Find the bin
        for i in range(len(self._x_bins) - 1):
            if raw_prob <= self._x_bins[i + 1]:
                return self._y_bins[i]
        return self._y_bins[-1]


class RidgeLinearRegression:
    """Standardized ridge regression in pure Python for continuous delta-Q prediction."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, X: list[list[float]], y: list[float]):
        n_samples = len(X)
        n_features = len(X[0])

        self.means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        self.stds = [
            math.sqrt(sum((X[i][j] - self.means[j]) ** 2 for i in range(n_samples)) / n_samples) + 1e-6
            for j in range(n_features)
        ]

        X_norm = [[(X[i][j] - self.means[j]) / self.stds[j] for j in range(n_features)] for i in range(n_samples)]
        self.weights = [0.0] * n_features
        self.bias = statistics.mean(y)

        lr = 0.01
        for _ in range(300):
            grad_b = 0.0
            grad_w = [0.0] * n_features
            for xi, yi in zip(X_norm, y):
                pred = self.bias + sum(w * x for w, x in zip(self.weights, xi))
                err = pred - yi
                grad_b += err
                for j in range(n_features):
                    grad_w[j] += err * xi[j]

            self.bias -= lr * (grad_b / n_samples)
            for j in range(n_features):
                self.weights[j] -= lr * (grad_w[j] / n_samples + self.alpha * self.weights[j])

    def predict(self, xi: list[float]) -> float:
        if not self.weights or not self.means:
            return 0.0
        xi_norm = [(xi[j] - self.means[j]) / self.stds[j] for j in range(len(xi))]
        return self.bias + sum(w * x for w, x in zip(self.weights, xi_norm))


class LogisticRegressionClassifier:
    """Standardized logistic regression in pure Python for P(HARM) probability."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: list[float] = []
        self.bias: float = 0.0
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]):
        n_samples = len(X)
        n_features = len(X[0])

        self.means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        self.stds = [
            math.sqrt(sum((X[i][j] - self.means[j]) ** 2 for i in range(n_samples)) / n_samples) + 1e-6
            for j in range(n_features)
        ]

        X_norm = [[(X[i][j] - self.means[j]) / self.stds[j] for j in range(n_features)] for i in range(n_samples)]
        self.weights = [0.0] * n_features
        pos_rate = max(0.01, min(0.99, sum(y) / n_samples))
        self.bias = math.log(pos_rate / (1.0 - pos_rate))

        lr = 0.01
        for _ in range(300):
            grad_b = 0.0
            grad_w = [0.0] * n_features
            for xi, yi in zip(X_norm, y):
                logit = max(min(self.bias + sum(w * x for w, x in zip(self.weights, xi)), 20.0), -20.0)
                prob = 1.0 / (1.0 + math.exp(-logit))
                err = prob - yi
                grad_b += err
                for j in range(n_features):
                    grad_w[j] += err * xi[j]

            self.bias -= lr * (grad_b / n_samples)
            for j in range(n_features):
                self.weights[j] -= lr * (grad_w[j] / n_samples + self.alpha * self.weights[j])

    def predict_prob(self, xi: list[float]) -> float:
        if not self.weights or not self.means:
            return 0.5
        xi_norm = [(xi[j] - self.means[j]) / self.stds[j] for j in range(len(xi))]
        logit = max(min(self.bias + sum(w * x for w, x in zip(self.weights, xi_norm)), 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-logit))


# ---------------------------------------------------------------------------
# Fold-Isolated Rule Discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveredRule:
    """A rule discovered from training data."""
    description: str
    prior_action_count_min: int
    last_action: str | None
    verification_states: frozenset[str]
    expected_delta_utility: float
    harm_probability: float
    help_probability: float
    confidence: float
    reason: str
    # Training statistics
    train_count: int
    train_help_rate: float
    train_harm_rate: float
    train_mean_delta_q: float


def discover_rules_from_training(
    train_states: list[dict[str, Any]],
    *,
    min_count: int = 3,
    min_help_rate: float = 0.50,
    max_harm_rate: float = 0.15,
    delta_q_threshold: float = 5.0,
) -> list[DiscoveredRule]:
    """Discover positive intervention rules from TRAINING data only.

    Groups training states by (prior_action_count bracket, last_action,
    verification_state) and identifies regions with high help rate and
    low harm rate.

    This is the fold-isolated rule discovery that prevents information
    leakage from the full dataset into cross-validation.
    """
    # Group by feature combinations
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for s in train_states:
        f = s["features"]
        # Bracket prior_action_count: 0, 1, 2, 3+
        pac = f["prior_action_count"]
        pac_bracket = pac if pac < 3 else 3
        key = (pac_bracket, f["last_action"], f["verification_state"])
        groups[key].append(s)

    discovered: list[DiscoveredRule] = []
    for (pac_bracket, last_act, verif_state), members in groups.items():
        n = len(members)
        if n < min_count:
            continue
        deltas = [m["delta_q"] for m in members]
        help_n = sum(1 for d in deltas if d > delta_q_threshold)
        harm_n = sum(1 for d in deltas if d < -delta_q_threshold)
        help_rate = help_n / n
        harm_rate = harm_n / n
        mean_dq = statistics.mean(deltas)

        if help_rate >= min_help_rate and harm_rate <= max_harm_rate and mean_dq > delta_q_threshold:
            # This is a positive intervention region
            pac_min = pac_bracket if pac_bracket < 3 else 3
            verif_set = frozenset([verif_state]) if verif_state else frozenset()
            desc = f"pac>={pac_min}, last={last_act}, verif={verif_state}"
            discovered.append(DiscoveredRule(
                description=desc,
                prior_action_count_min=pac_min,
                last_action=last_act,
                verification_states=verif_set,
                expected_delta_utility=round(mean_dq, 2),
                harm_probability=round(harm_rate, 4),
                help_probability=round(help_rate, 4),
                confidence=round(min(0.95, 0.50 + help_rate * 0.50), 4),
                reason=f"FOLD_DISCOVERED:HELP_REGION:{desc}",
                train_count=n,
                train_help_rate=round(help_rate, 4),
                train_harm_rate=round(harm_rate, 4),
                train_mean_delta_q=round(mean_dq, 2),
            ))

    return discovered


class FoldIsolatedPredictor(BaseInterventionPredictor):
    """Predictor that uses fold-isolated discovered rules plus learned fallback.

    Rules are discovered from training data only. The learned linear/logistic
    models provide fallback predictions for states not covered by any rule.
    """

    def __init__(
        self,
        rules: list[DiscoveredRule],
        regressor: RidgeLinearRegression | None = None,
        classifier: LogisticRegressionClassifier | None = None,
        calibrator: IsotonicCalibrator | None = None,
    ):
        self.rules = rules
        self.regressor = regressor
        self.classifier = classifier
        self.calibrator = calibrator

    def predict(self, features: InterventionFeatures) -> InterventionPrediction:
        try:
            # Rule 0: Step 0 is always dangerous -> SKIP
            if features.prior_action_count == 0:
                return InterventionPrediction(
                    expected_delta_utility=-25.0,
                    harm_probability=0.90,
                    help_probability=0.02,
                    confidence=0.95,
                    reason="STEP0_HAZARD_SKIP",
                )

            # Check fold-discovered positive rules
            for rule in self.rules:
                if features.prior_action_count < rule.prior_action_count_min:
                    continue
                if rule.last_action is not None and features.last_action != rule.last_action:
                    continue
                if rule.verification_states and features.verification_state not in rule.verification_states:
                    continue
                # Rule matches
                harm_p = rule.harm_probability
                if self.calibrator:
                    harm_p = self.calibrator.calibrate(harm_p)
                return InterventionPrediction(
                    expected_delta_utility=rule.expected_delta_utility,
                    harm_probability=harm_p,
                    help_probability=rule.help_probability,
                    confidence=rule.confidence,
                    reason=rule.reason,
                )

            # Fallback to learned model
            if self.regressor and self.classifier:
                vec = features.to_numeric_vector()
                pred_dq = self.regressor.predict(vec)
                harm_p = self.classifier.predict_prob(vec)
                if self.calibrator:
                    harm_p = self.calibrator.calibrate(harm_p)
                return InterventionPrediction(
                    expected_delta_utility=pred_dq,
                    harm_probability=harm_p,
                    help_probability=max(0.0, 1.0 - harm_p) if pred_dq > 0 else 0.0,
                    confidence=abs(harm_p - 0.5) * 2.0,
                    reason="LEARNED_MODEL_FALLBACK",
                )

            return InterventionPrediction(
                expected_delta_utility=-5.0,
                harm_probability=0.80,
                help_probability=0.0,
                confidence=0.50,
                reason="CONSERVATIVE_DEFAULT_SKIP",
            )
        except Exception as e:
            return InterventionPrediction(
                expected_delta_utility=-999.0,
                harm_probability=1.0,
                help_probability=0.0,
                confidence=0.0,
                reason=f"EXCEPTION_FALLBACK:{e}",
            )


def main():
    parser = argparse.ArgumentParser(description="Task-Grouped Cross-Validation for Selective Gate")
    parser.add_argument(
        "--dataset",
        default="experiments/v2b_i3_5_2/development/intervention_states_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/v2b_i3_5_2/development/cross_validation_report_v1.json",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rule-discovery-mode",
        choices=["fold_isolated", "global"],
        default="fold_isolated",
        help="fold_isolated: discover rules within each fold's training data only. "
             "global: use pre-defined rules (NOT fold-isolated, for comparison only).",
    )
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    print(f"Loaded {len(records)} decision state records from {args.dataset}")

    # Group records by task_id to prevent any state leakage across folds
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        task_groups[r["task_id"]].append(r)

    unique_tasks = sorted(task_groups.keys())
    print(f"Total unique tasks: {len(unique_tasks)}")

    # Deterministic task-grouped split
    rng = random.Random(args.seed)
    shuffled_tasks = list(unique_tasks)
    rng.shuffle(shuffled_tasks)

    folds: list[list[str]] = [[] for _ in range(args.k_folds)]
    for idx, t_id in enumerate(shuffled_tasks):
        folds[idx % args.k_folds].append(t_id)

    print(f"Constructed {args.k_folds} task-grouped folds:")
    for f_idx, fold_tasks in enumerate(folds):
        n_states = sum(len(task_groups[t]) for t in fold_tasks)
        print(f"  Fold {f_idx + 1}: {len(fold_tasks)} tasks, {n_states} states")

    print(f"\nRule discovery mode: {args.rule_discovery_mode}")
    if args.rule_discovery_mode == "global":
        print("  WARNING: Global rules are NOT fold-isolated.")
        print("  CV measures OOF evaluation of a development-derived fixed rule policy.")
        print("  This is NOT unbiased generalization performance.")
    else:
        print("  Rules are discovered from training data only within each fold.")

    # Out-of-fold prediction accumulators
    oof_actual_dq: list[float] = []
    oof_pred_dq: list[float] = []
    oof_actual_harm: list[int] = []
    oof_pred_harm_prob: list[float] = []
    oof_pred_harm_prob_calibrated: list[float] = []
    oof_approved_decisions: list[dict[str, Any]] = []

    fold_metrics = []
    all_discovered_rules: list[dict[str, Any]] = []

    for f_idx in range(args.k_folds):
        test_tasks = set(folds[f_idx])
        train_tasks = set(unique_tasks) - test_tasks

        train_states = [r for t in train_tasks for r in task_groups[t]]
        test_states = [r for t in test_tasks for r in task_groups[t]]

        # Train models on train_states
        X_train = [InterventionFeatures(**s["features"]).to_numeric_vector() for s in train_states]
        y_train_dq = [s["delta_q"] for s in train_states]
        y_train_harm = [1 if s["delta_q"] < -5.0 else 0 for s in train_states]

        regressor = RidgeLinearRegression(alpha=1.0)
        regressor.fit(X_train, y_train_dq)

        classifier = LogisticRegressionClassifier(alpha=1.0)
        classifier.fit(X_train, y_train_harm)

        # Calibrate classifier probabilities using isotonic regression on training data
        train_raw_probs = [classifier.predict_prob(x) for x in X_train]
        calibrator = IsotonicCalibrator()
        calibrator.fit(train_raw_probs, y_train_harm)

        # Rule discovery
        if args.rule_discovery_mode == "fold_isolated":
            rules = discover_rules_from_training(train_states)
            print(f"\n  Fold {f_idx + 1}: Discovered {len(rules)} positive rules from {len(train_states)} training states")
            for r in rules:
                print(f"    {r.reason}: N={r.train_count}, help={r.train_help_rate:.1%}, "
                      f"harm={r.train_harm_rate:.1%}, mean ΔQ={r.train_mean_delta_q:.1f}")
                all_discovered_rules.append({
                    "fold": f_idx + 1,
                    "description": r.description,
                    "reason": r.reason,
                    "train_count": r.train_count,
                    "train_help_rate": r.train_help_rate,
                    "train_harm_rate": r.train_harm_rate,
                    "train_mean_delta_q": r.train_mean_delta_q,
                    "expected_delta_utility": r.expected_delta_utility,
                    "harm_probability": r.harm_probability,
                })
        else:
            # Global rules (NOT fold-isolated) - for comparison only
            rules = []

        # Build predictor
        if args.rule_discovery_mode == "fold_isolated":
            predictor = FoldIsolatedPredictor(
                rules=rules,
                regressor=regressor,
                classifier=classifier,
                calibrator=calibrator,
            )
        else:
            # Use the global RuleBasedInterventionPredictor (pre-defined rules from full dataset)
            # This is NOT fold-isolated — for comparison only
            predictor = RuleBasedInterventionPredictor()

        gate = SelectiveGovernorGate(
            predictor=predictor,
            delta_u_threshold=5.0,
            max_harm_probability=0.15,
            min_confidence=0.60,
        )

        # Evaluate on test_states
        fold_interventions = 0
        fold_intervened_deltas = []
        fold_intervened_harms = 0
        fold_intervened_helps = 0

        for s in test_states:
            feats = InterventionFeatures(**s["features"])
            actual_dq = s["delta_q"]
            is_harm = 1 if actual_dq < -5.0 else 0

            pred = predictor.predict(feats)
            should_intervene = (
                pred.expected_delta_utility > gate.delta_u_threshold
                and pred.harm_probability < gate.max_harm_probability
                and pred.confidence >= gate.min_confidence
            )

            # Also get raw and calibrated harm probabilities for metrics
            raw_harm_p = classifier.predict_prob(feats.to_numeric_vector())
            cal_harm_p = calibrator.calibrate(raw_harm_p)

            oof_actual_dq.append(actual_dq)
            oof_pred_dq.append(pred.expected_delta_utility)
            oof_actual_harm.append(is_harm)
            oof_pred_harm_prob.append(pred.harm_probability)
            oof_pred_harm_prob_calibrated.append(cal_harm_p)

            if should_intervene:
                fold_interventions += 1
                fold_intervened_deltas.append(actual_dq)
                if actual_dq < -5.0:
                    fold_intervened_harms += 1
                elif actual_dq > 5.0:
                    fold_intervened_helps += 1

                oof_approved_decisions.append({
                    "task_id": s["task_id"],
                    "step_id": s["step_id"],
                    "actual_delta_q": actual_dq,
                    "pred_delta_q": pred.expected_delta_utility,
                    "pred_harm_prob": pred.harm_probability,
                    "reason": pred.reason,
                })

        int_rate = fold_interventions / len(test_states) if test_states else 0.0
        precision_help = fold_intervened_helps / fold_interventions if fold_interventions > 0 else 0.0
        harm_rate = fold_intervened_harms / fold_interventions if fold_interventions > 0 else 0.0
        mean_gain = statistics.mean(fold_intervened_deltas) if fold_intervened_deltas else 0.0

        fold_metrics.append({
            "fold": f_idx + 1,
            "test_states": len(test_states),
            "train_states": len(train_states),
            "discovered_rules_count": len(rules) if args.rule_discovery_mode == "fold_isolated" else None,
            "interventions": fold_interventions,
            "intervention_rate": round(int_rate, 4),
            "precision_help": round(precision_help, 4),
            "harm_rate": round(harm_rate, 4),
            "mean_delta_q_intervened": round(mean_gain, 4),
        })

    # Overall Out-of-Fold Evaluation
    oof_mae = statistics.mean(abs(p - a) for p, a in zip(oof_pred_dq, oof_actual_dq))
    oof_spearman = compute_spearman_corr(oof_pred_dq, oof_actual_dq)
    oof_roc_auc = compute_roc_auc(oof_actual_harm, oof_pred_harm_prob)
    oof_brier = compute_brier_score(oof_actual_harm, oof_pred_harm_prob)
    oof_brier_calibrated = compute_brier_score(oof_actual_harm, oof_pred_harm_prob_calibrated)
    oof_ece = compute_ece(oof_actual_harm, oof_pred_harm_prob)
    oof_ece_calibrated = compute_ece(oof_actual_harm, oof_pred_harm_prob_calibrated)

    # Base-rate Brier: what you'd get by always predicting the mean harm rate
    harm_prevalence = sum(oof_actual_harm) / len(oof_actual_harm) if oof_actual_harm else 0.0
    base_rate_brier = harm_prevalence * (1 - harm_prevalence)

    # Reliability diagram
    reliability = compute_reliability_bins(oof_actual_harm, oof_pred_harm_prob)
    reliability_calibrated = compute_reliability_bins(oof_actual_harm, oof_pred_harm_prob_calibrated)

    total_approved = len(oof_approved_decisions)
    total_approved_helps = sum(1 for d in oof_approved_decisions if d["actual_delta_q"] > 5.0)
    total_approved_harms = sum(1 for d in oof_approved_decisions if d["actual_delta_q"] < -5.0)
    overall_precision = total_approved_helps / total_approved if total_approved > 0 else 0.0
    overall_harm_rate = total_approved_harms / total_approved if total_approved > 0 else 0.0
    approved_deltas = [d["actual_delta_q"] for d in oof_approved_decisions]
    mean_approved_gain = statistics.mean(approved_deltas) if approved_deltas else 0.0
    worst_decile_gain = sorted(approved_deltas)[int(0.10 * len(approved_deltas))] if approved_deltas else 0.0

    cv_report = {
        "schema": "DAPH_V2B_I3_5_2_CROSS_VALIDATION_REPORT_V1",
        "schema_version": 1,
        "k_folds": args.k_folds,
        "total_tasks": len(unique_tasks),
        "total_states": len(records),
        "rule_discovery_mode": args.rule_discovery_mode,
        "rule_discovery_independence": (
            "FOLD_ISOLATED" if args.rule_discovery_mode == "fold_isolated" else "NOT_ISOLATED"
        ),
        "harm_prevalence": round(harm_prevalence, 4),
        "out_of_fold_metrics": {
            "mae_delta_q": round(oof_mae, 4),
            "spearman_correlation_delta_q": round(oof_spearman, 4),
            "roc_auc_harm": round(oof_roc_auc, 4),
            "brier_score_harm": round(oof_brier, 4),
            "brier_score_harm_calibrated": round(oof_brier_calibrated, 4),
            "base_rate_brier_score": round(base_rate_brier, 4),
            "brier_vs_base_rate": round(oof_brier - base_rate_brier, 4),
            "brier_calibrated_vs_base_rate": round(oof_brier_calibrated - base_rate_brier, 4),
            "ece_harm": round(oof_ece, 4),
            "ece_harm_calibrated": round(oof_ece_calibrated, 4),
            "total_interventions_approved": total_approved,
            "intervention_rate": round(total_approved / len(records), 4),
            "precision_help": round(overall_precision, 4),
            "harm_rate_on_interventions": round(overall_harm_rate, 4),
            "mean_delta_q_on_interventions": round(mean_approved_gain, 4),
            "worst_decile_delta_q": round(worst_decile_gain, 4),
        },
        "calibration_diagnostics": {
            "reliability_bins_raw": reliability,
            "reliability_bins_calibrated": reliability_calibrated,
            "harm_prevalence": round(harm_prevalence, 4),
            "base_rate_brier": round(base_rate_brier, 4),
            "interpretation": (
                "Brier score is compared to the base-rate Brier (always predicting "
                "the mean harm prevalence). A Brier score above the base-rate indicates "
                "poor calibration. ROC-AUC measures discrimination (ranking), not "
                "calibration. ECE measures average absolute difference between predicted "
                "and actual probabilities in bins."
            ),
        },
        "fold_breakdown": fold_metrics,
        "discovered_rules_by_fold": all_discovered_rules if args.rule_discovery_mode == "fold_isolated" else None,
        "sample_approved_interventions": oof_approved_decisions[:10],
        "scientific_caveats": [
            "These CV metrics measure out-of-fold performance of a predictor that "
            "combines fold-isolated discovered rules with learned linear/logistic fallback.",
            "The Brier score should be compared to the base-rate Brier. A value above "
            "the base rate indicates poor probability calibration despite good discrimination.",
            "These are state-level counterfactual results (governor ranking), NOT packet-level "
            "treatment results. The packet treatment experiment (I3.5.2b) is required to "
            "measure whether the model can actually exploit governor information.",
            "End-to-end selective trajectory improvement has NOT been demonstrated.",
        ],
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(cv_report, indent=2) + "\n")
    print(f"\nSaved Cross-Validation Report to: {out_file}")

    print("\n" + "=" * 78)
    print(f"5-FOLD TASK-GROUPED CROSS-VALIDATION RESULTS (RULE DISCOVERY: {args.rule_discovery_mode.upper()})")
    print("=" * 78)
    print(f"Total States: {len(records)} across {len(unique_tasks)} tasks")
    print(f"Harm prevalence: {harm_prevalence:.1%} ({sum(oof_actual_harm)}/{len(oof_actual_harm)})")
    print(f"\n  --- Regression Metrics ---")
    print(f"  MAE (ΔQ):                     {oof_mae:.4f}")
    print(f"  Spearman rank correlation:     {oof_spearman:.4f}")
    print(f"\n  --- Harm Probability Metrics ---")
    print(f"  ROCAUC P(HARM):                {oof_roc_auc:.4f}  (discrimination / ranking)")
    print(f"  Brier score P(HARM):           {oof_brier:.4f}  (raw)")
    print(f"  Brier score P(HARM):           {oof_brier_calibrated:.4f}  (calibrated)")
    print(f"  Base-rate Brier:               {base_rate_brier:.4f}  (always predict prevalence)")
    print(f"  Brier vs base-rate:            {oof_brier - base_rate_brier:+.4f}  (raw)")
    print(f"  Brier vs base-rate:            {oof_brier_calibrated - base_rate_brier:+.4f}  (calibrated)")
    print(f"  ECE (raw):                     {oof_ece:.4f}")
    print(f"  ECE (calibrated):              {oof_ece_calibrated:.4f}")
    print(f"\n  --- Gate Performance ---")
    print(f"  Intervention rate:             {total_approved}/{len(records)} ({total_approved/len(records):.1%})")
    print(f"  Precision of INTERVENE (HELP): {overall_precision:.1%}")
    print(f"  Harm rate on INTERVENE:        {overall_harm_rate:.1%}")
    print(f"  E[ΔQ | INTERVENE]:             +{mean_approved_gain:.2f} Q-points")
    print(f"  Worst-decile ΔQ:               {worst_decile_gain:.2f} Q-points")

    if args.rule_discovery_mode == "fold_isolated":
        print(f"\n  --- Fold-Isolated Rule Discovery ---")
        print(f"  Total rules discovered across folds: {len(all_discovered_rules)}")
        for r in all_discovered_rules:
            print(f"    Fold {r['fold']}: {r['reason']} "
                  f"(N={r['train_count']}, help={r['train_help_rate']:.1%}, "
                  f"harm={r['train_harm_rate']:.1%}, ΔQ={r['train_mean_delta_q']:.1f})")


if __name__ == "__main__":
    main()
