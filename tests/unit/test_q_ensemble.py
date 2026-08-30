"""Tests for Q_V3R3 ensemble and uncertainty-gated authority."""
import numpy as np
import pytest
from pathlib import Path

from daph.models.q_ensemble import (
    QEnsemble,
    train_q_ensemble,
    uncertainty_gated_authority,
)


@pytest.fixture
def small_ensemble():
    """Train a small ensemble for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2] + rng.randn(100) * 0.1
    return train_q_ensemble(
        X_train=X, y_train=y,
        feature_keys=[f"f{i}" for i in range(5)],
        n_estimators=5,
        gbt_params=dict(n_estimators=50, max_depth=3),
        lambda_lcb=1.0,
        ood_threshold=3.0,
        n_support_clusters=10,
        random_state=42,
    )


class TestQEnsemble:
    """Test QEnsemble properties."""

    def test_ensemble_has_n_models(self, small_ensemble):
        assert len(small_ensemble.models) == 5

    def test_predict_mean_shape(self, small_ensemble):
        X = np.random.randn(10, 5)
        mean = small_ensemble.predict_mean(X)
        assert mean.shape == (10,)

    def test_predict_std_nonnegative(self, small_ensemble):
        X = np.random.randn(10, 5)
        std = small_ensemble.predict_std(X)
        assert (std >= 0).all()

    def test_lcb_below_mean(self, small_ensemble):
        X = np.random.randn(10, 5)
        mean = small_ensemble.predict_mean(X)
        lcb = small_ensemble.predict_lcb(X)
        assert (lcb <= mean + 1e-6).all()

    def test_support_density_in_support(self, small_ensemble):
        """In-support samples should have higher density than OOD."""
        # In-support: similar to training data
        X_in = np.random.randn(5, 5) * 0.5
        # OOD: far from training data
        X_out = np.random.randn(5, 5) * 10 + 100

        density_in = small_ensemble.support_density(X_in)
        density_out = small_ensemble.support_density(X_out)

        assert (density_in > density_out).all()

    def test_is_in_support_boolean(self, small_ensemble):
        X = np.random.randn(5, 5)
        mask = small_ensemble.is_in_support(X)
        assert mask.dtype == bool
        assert mask.shape == (5,)

    def test_save_and_load(self, small_ensemble, tmp_path):
        """Ensemble can be saved and loaded."""
        path = tmp_path / "test_ensemble.pkl"
        small_ensemble.save(path)
        loaded = QEnsemble.load(path)

        X = np.random.randn(5, 5)
        original = small_ensemble.predict_mean(X)
        recovered = loaded.predict_mean(X)
        np.testing.assert_array_almost_equal(original, recovered)


class TestUncertaintyGatedAuthority:
    """Test uncertainty-gated authority decisions."""

    def test_ood_gates_force(self):
        """OOD state should not force."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 1, "lcb": 99},
            "DEFER": {"mean": 50, "std": 1, "lcb": 49},
            "REASON_MORE": {"mean": 60, "std": 1, "lcb": 59},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=False,
            cert_answer=True,
        )
        assert action is None
        assert reason == "OOD_GATED"

    def test_lcb_gap_insufficient(self):
        """Small LCB gap should not force."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 10, "lcb": 90},
            "DEFER": {"mean": 95, "std": 5, "lcb": 90},
            "REASON_MORE": {"mean": 50, "std": 1, "lcb": 49},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=True,
            cert_answer=True,
        )
        assert action is None
        assert reason == "LCB_GAP_INSUFFICIENT"

    def test_high_uncertainty_gates_force(self):
        """High uncertainty should reduce LCB and prevent forcing."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 50, "lcb": 50},
            "DEFER": {"mean": 60, "std": 1, "lcb": 59},
            "REASON_MORE": {"mean": 55, "std": 1, "lcb": 54},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=True,
            cert_answer=True,
        )
        # LCB gap: 50 - 59 = -9 < 5.0 → insufficient
        assert action is None

    def test_cert_lcb_ood_pass(self):
        """All gates pass → force."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 2, "lcb": 98},
            "DEFER": {"mean": 50, "std": 2, "lcb": 48},
            "REASON_MORE": {"mean": 60, "std": 2, "lcb": 58},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=True,
            cert_answer=True,
        )
        assert action == "ANSWER"
        assert reason == "CERT_LCB_OOD_PASS"

    def test_cert_mismatch(self):
        """Best action doesn't match certificate → no force."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 2, "lcb": 98},
            "DEFER": {"mean": 50, "std": 2, "lcb": 48},
            "REASON_MORE": {"mean": 60, "std": 2, "lcb": 58},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=True,
            cert_answer=False,
            cert_defer=True,
        )
        # ANSWER has highest LCB but cert_defer is True, cert_answer is False
        assert action is None
        assert reason == "CERT_MISMATCH"

    def test_not_sole_near_optimal(self):
        """Two actions within epsilon → no force."""
        q_values = {
            "ANSWER": {"mean": 100, "std": 2, "lcb": 98},
            "DEFER": {"mean": 99, "std": 2, "lcb": 97},
            "REASON_MORE": {"mean": 50, "std": 2, "lcb": 48},
        }
        action, reason = uncertainty_gated_authority(
            q_values=q_values,
            legal_actions=["ANSWER", "DEFER", "REASON_MORE"],
            in_support=True,
            cert_answer=True,
        )
        # LCB gap: 98 - 97 = 1 < 5.0 → insufficient (also not sole near-optimal)
        assert action is None
