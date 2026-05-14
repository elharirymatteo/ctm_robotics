"""Tests for the linear-decodability R²_CV protocol."""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)


def test_perfect_decoding():
    """If hidden_var is a linear function of z, R² should be ≈ 1.0."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 1000
    z = np.random.randn(T, 16)
    w_true = np.random.randn(16)
    vy = z @ w_true + 0.01 * np.random.randn(T)

    out = linear_decodability(z, {"vy": vy})
    assert out["vy"]["r2_cv"] > 0.95, f"Expected R² > 0.95, got {out['vy']['r2_cv']}"


def test_no_signal():
    """If hidden_var is independent of z, R² should be ≈ 0 (or slightly negative under CV)."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 1000
    z = np.random.randn(T, 16)
    vy = np.random.randn(T)

    out = linear_decodability(z, {"vy": vy})
    assert out["vy"]["r2_cv"] < 0.1, f"Expected R² < 0.1, got {out['vy']['r2_cv']}"


def test_delta_r2_isolates_z_information():
    """Visible vars predict vy on their own; ΔR² measures z's marginal contribution."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 2000
    visible = np.random.randn(T, 4)
    z_independent = np.random.randn(T, 16)
    vy = visible.sum(axis=1) + z_independent[:, 0] * 2.0 + 0.1 * np.random.randn(T)

    out = linear_decodability(z_independent, {"vy": vy}, visible_vars=visible)
    assert out["vy"]["r2_cv_visible"] > 0.1
    assert out["vy"]["r2_cv"] > out["vy"]["r2_cv_visible"]
    assert out["vy"]["delta_r2"] > 0.05


def test_multiple_targets():
    """linear_decodability handles a dict of multiple target variables."""
    from ctm_robotics.analysis.decodability import linear_decodability

    T = 500
    z = np.random.randn(T, 16)
    w1 = np.random.randn(16)
    w2 = np.random.randn(16)
    targets = {
        "vx": z @ w1 + 0.01 * np.random.randn(T),
        "vy": z @ w2 + 0.01 * np.random.randn(T),
    }

    out = linear_decodability(z, targets)
    assert set(out.keys()) == {"vx", "vy"}
    assert out["vx"]["r2_cv"] > 0.95
    assert out["vy"]["r2_cv"] > 0.95


def test_short_input_raises():
    """Refuse to fit on too few samples (less than 2× n_features)."""
    from ctm_robotics.analysis.decodability import linear_decodability

    z = np.random.randn(10, 16)
    vy = np.random.randn(10)

    with pytest.raises(ValueError, match="too few samples"):
        linear_decodability(z, {"vy": vy})
