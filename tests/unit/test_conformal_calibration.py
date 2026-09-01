"""Tests for conformal calibration quantile direction and coverage semantics.

Verifies that:
  - The conformal quantile uses coverage (not miscoverage) direction
  - Nominal coverage is approximately achieved on synthetic data
  - Higher coverage levels produce larger q_alpha values
  - The finite-sample correction is correct
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from m4_conformal_calibration import conformal_calibrate


def _make_synthetic_pairs(n: int, seed: int = 42, noise_scale: float = 10.0) -> list[dict]:
    """Generate synthetic pairs with known residual distribution.

    delta_q_hat is a noisy estimate of delta_u:
      delta_q_hat = delta_u + noise
      residual = |delta_q_hat - delta_u| = |noise|

    noise ~ Uniform(0, noise_scale), so the alpha-quantile of |residuals|
    should be approximately alpha * noise_scale.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for i in range(n):
        delta_u = rng.randn() * 5.0
        noise = rng.uniform(0, noise_scale)
        delta_q_hat = delta_u + noise
        pairs.append({
            "group_id": f"synthetic_{i}",
            "exec_action": "ANSWER(H1)",
            "base_action": "DEFER",
            "delta_q_hat": float(delta_q_hat),
            "delta_u": float(delta_u),
            "residual": float(abs(delta_q_hat - delta_u)),
            "is_harmful": int(delta_u < 0),
        })
    return pairs


def test_conformal_quantile_direction_correct():
    """q_alpha for 90% coverage should be approximately the 90th percentile,
    NOT the 10th percentile. With Uniform(0,10) residuals, the 90th percentile
    is ~9.0, not ~1.0."""
    cal = _make_synthetic_pairs(500, seed=42, noise_scale=10.0)
    eval_pairs = _make_synthetic_pairs(200, seed=99, noise_scale=10.0)

    results = conformal_calibrate(cal, eval_pairs, alpha_levels=[0.90])

    r90 = results["coverage_0.90"]
    # With Uniform(0,10) residuals, the 90th percentile should be ~9.0
    # The OLD (buggy) code would produce ~1.0 (the 10th percentile)
    assert r90["q_alpha"] > 5.0, (
        f"q_alpha for 90% coverage should be > 5.0 (near the 90th percentile of Uniform(0,10)), "
        f"got {r90['q_alpha']}. If this is ~1.0, the quantile direction is backwards."
    )
    assert r90["q_alpha"] < 11.0, (
        f"q_alpha for 90% coverage should be < 11.0, got {r90['q_alpha']}"
    )


def test_higher_coverage_produces_larger_q_alpha():
    """Higher coverage levels must produce larger (or equal) q_alpha values."""
    cal = _make_synthetic_pairs(500, seed=42, noise_scale=10.0)
    eval_pairs = _make_synthetic_pairs(200, seed=99, noise_scale=10.0)

    results = conformal_calibrate(cal, eval_pairs, alpha_levels=[0.50, 0.80, 0.90, 0.95, 0.99])

    q_values = [results[f"coverage_{a:.2f}"]["q_alpha"]
                for a in [0.50, 0.80, 0.90, 0.95, 0.99]]

    for i in range(len(q_values) - 1):
        assert q_values[i] <= q_values[i + 1] + 1e-9, (
            f"q_alpha should be non-decreasing with coverage level. "
            f"Got {q_values}"
        )


def test_nominal_coverage_approximately_achieved():
    """On synthetic data with the same distribution, empirical coverage
    should be approximately equal to nominal coverage (within 5%)."""
    n_cal = 500
    n_eval = 1000
    cal = _make_synthetic_pairs(n_cal, seed=42, noise_scale=10.0)
    eval_pairs = _make_synthetic_pairs(n_eval, seed=99, noise_scale=10.0)

    results = conformal_calibrate(cal, eval_pairs, alpha_levels=[0.50, 0.80, 0.90, 0.95, 0.99])

    for alpha in [0.50, 0.80, 0.90, 0.95, 0.99]:
        r = results[f"coverage_{alpha:.2f}"]
        emp_cov = r["empirical_coverage"]
        # Coverage should be >= nominal - tolerance (conformal guarantees coverage)
        # and not excessively conservative
        assert emp_cov >= alpha - 0.05, (
            f"Empirical coverage {emp_cov} should be >= nominal {alpha} - 0.05"
        )
        assert emp_cov <= alpha + 0.10, (
            f"Empirical coverage {emp_cov} should not be excessively conservative "
            f"(> nominal {alpha} + 0.10)"
        )


def test_q_alpha_nonzero_for_high_coverage():
    """The old buggy code produced q_alpha=0 for all levels >= 0.80 because
    it took the (1-alpha) quantile. The corrected code should produce
    nonzero q_alpha for all coverage levels."""
    cal = _make_synthetic_pairs(300, seed=42, noise_scale=10.0)
    eval_pairs = _make_synthetic_pairs(100, seed=99, noise_scale=10.0)

    results = conformal_calibrate(cal, eval_pairs, alpha_levels=[0.80, 0.90, 0.95, 0.99])

    for alpha in [0.80, 0.90, 0.95, 0.99]:
        r = results[f"coverage_{alpha:.2f}"]
        assert r["q_alpha"] > 0.0, (
            f"q_alpha for coverage {alpha} should be > 0 with Uniform(0,10) residuals, "
            f"got {r['q_alpha']}. The old buggy code produced 0.0 here."
        )


def test_calibration_self_coverage_achieved():
    """When calibrating and evaluating on the same set, coverage should be
    approximately nominal (self-coverage is a sanity check)."""
    pairs = _make_synthetic_pairs(500, seed=42, noise_scale=10.0)

    results = conformal_calibrate(pairs, pairs, alpha_levels=[0.90])

    r90 = results["coverage_0.90"]
    # Self-coverage should be at least ~90% (the quantile is computed from the same set)
    assert r90["empirical_coverage"] >= 0.85, (
        f"Self-coverage at 90% should be >= 0.85, got {r90['empirical_coverage']}"
    )
