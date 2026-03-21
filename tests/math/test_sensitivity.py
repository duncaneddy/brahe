"""
Tests for Sensitivity computation module.

Tests numerical and analytical sensitivity providers (∂f/∂p).
"""

import numpy as np
import brahe as bh


# Simple linear system with parameters:
# dx/dt = state[1] + params[0], dy/dt = -state[0] + params[1]
# Analytical sensitivity: [[1, 0], [0, 1]] (identity)
def linear_dynamics_with_params(t, state, params):
    return np.array([state[1] + params[0], -state[0] + params[1]])


def analytical_sensitivity_fn(t, state, params):
    return np.array([[1.0, 0.0], [0.0, 1.0]])


# =============================================================================
# NumericalSensitivity constructor tests
# =============================================================================


def test_numerical_sensitivity_default():
    provider = bh.NumericalSensitivity(linear_dynamics_with_params)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.with_fixed_offset(1e-6).compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_numerical_sensitivity_central():
    provider = bh.NumericalSensitivity.central(
        linear_dynamics_with_params
    ).with_fixed_offset(1e-6)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_numerical_sensitivity_forward():
    provider = bh.NumericalSensitivity.forward(
        linear_dynamics_with_params
    ).with_fixed_offset(1e-6)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_numerical_sensitivity_backward():
    provider = bh.NumericalSensitivity.backward(
        linear_dynamics_with_params
    ).with_fixed_offset(1e-6)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# Builder method tests
# =============================================================================


def test_numerical_sensitivity_with_fixed_offset():
    provider = bh.NumericalSensitivity.central(
        linear_dynamics_with_params
    ).with_fixed_offset(1e-7)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_numerical_sensitivity_with_percentage():
    provider = bh.NumericalSensitivity.central(
        linear_dynamics_with_params
    ).with_percentage(1e-6)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_numerical_sensitivity_with_adaptive():
    provider = bh.NumericalSensitivity.central(
        linear_dynamics_with_params
    ).with_adaptive(1.0, 1e-10)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_numerical_sensitivity_with_method():
    provider = (
        bh.NumericalSensitivity(linear_dynamics_with_params)
        .with_method(bh.DifferenceMethod.FORWARD)
        .with_fixed_offset(1e-6)
    )
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-5)


# =============================================================================
# AnalyticSensitivity tests
# =============================================================================


def test_analytic_sensitivity():
    provider = bh.AnalyticSensitivity(analytical_sensitivity_fn)
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    result = provider.compute(0.0, state, params)

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_analytic_sensitivity_with_list_inputs():
    provider = bh.AnalyticSensitivity(analytical_sensitivity_fn)
    result = provider.compute(0.0, [1.0, 0.5], [0.1, 0.2])

    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# =============================================================================
# Method chaining order tests
# =============================================================================


def test_sensitivity_chaining_order():
    """Verify builder chaining works in different orders."""
    state = np.array([1.0, 0.5])
    params = np.array([0.1, 0.2])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Order 1: method -> offset
    p1 = (
        bh.NumericalSensitivity(linear_dynamics_with_params)
        .with_method(bh.DifferenceMethod.CENTRAL)
        .with_fixed_offset(1e-6)
    )
    r1 = p1.compute(0.0, state, params)
    np.testing.assert_allclose(r1, expected, atol=1e-6)

    # Order 2: offset -> method
    p2 = (
        bh.NumericalSensitivity(linear_dynamics_with_params)
        .with_fixed_offset(1e-6)
        .with_method(bh.DifferenceMethod.CENTRAL)
    )
    r2 = p2.compute(0.0, state, params)
    np.testing.assert_allclose(r2, expected, atol=1e-6)


# =============================================================================
# Nonlinear system tests
# =============================================================================


def test_nonlinear_sensitivity():
    """Test sensitivity with a nonlinear parameter-dependent system."""

    # dx/dt = params[0] * state[0]^2 + state[1]
    # dy/dt = -state[0] + params[1] * state[1]
    # ∂f/∂p = [[state[0]^2, 0], [0, state[1]]]
    def nonlinear_dynamics(t, state, params):
        return np.array(
            [params[0] * state[0] ** 2 + state[1], -state[0] + params[1] * state[1]]
        )

    def nonlinear_analytical(t, state, params):
        return np.array([[state[0] ** 2, 0.0], [0.0, state[1]]])

    state = np.array([2.0, 3.0])
    params = np.array([1.0, 0.5])

    # Test analytical
    analytical = bh.AnalyticSensitivity(nonlinear_analytical)
    jac_a = analytical.compute(0.0, state, params)
    expected = np.array([[4.0, 0.0], [0.0, 3.0]])
    np.testing.assert_allclose(jac_a, expected, atol=1e-10)

    # Test numerical
    numerical = bh.NumericalSensitivity.central(nonlinear_dynamics).with_fixed_offset(
        1e-6
    )
    jac_n = numerical.compute(0.0, state, params)
    np.testing.assert_allclose(jac_n, expected, atol=1e-5)


# =============================================================================
# Higher-dimensional system tests
# =============================================================================


def test_higher_dimensional_sensitivity():
    """Test sensitivity with 4D state and 3 parameters."""

    # State: [x1, x2, x3, x4], Params: [p1, p2, p3]
    # f1 = x2 + p1, f2 = -x1 + p2, f3 = x4 + p3, f4 = -x3
    # ∂f/∂p = [[1,0,0], [0,1,0], [0,0,1], [0,0,0]]
    def dynamics_4d(t, state, params):
        return np.array(
            [
                state[1] + params[0],
                -state[0] + params[1],
                state[3] + params[2],
                -state[2],
            ]
        )

    state = np.array([1.0, 2.0, 3.0, 4.0])
    params = np.array([0.1, 0.2, 0.3])

    provider = bh.NumericalSensitivity.central(dynamics_4d).with_fixed_offset(1e-6)
    result = provider.compute(0.0, state, params)

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_sensitivity_with_lambda():
    """Test that lambda functions work as dynamics."""
    provider = bh.NumericalSensitivity(
        lambda t, s, p: np.array([s[1] + p[0], -s[0] + p[1]])
    ).with_fixed_offset(1e-6)

    result = provider.compute(0.0, np.array([1.0, 0.5]), np.array([0.1, 0.2]))
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-6)
