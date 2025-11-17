"""
Tests for Jacobian computation module.

Tests numerical and analytical Jacobian providers for dynamic-sized systems.
"""

import numpy as np
import brahe as bh


def test_difference_method_enum():
    """Test DifferenceMethod enum values."""
    assert bh.DifferenceMethod.FORWARD != bh.DifferenceMethod.CENTRAL
    assert bh.DifferenceMethod.CENTRAL != bh.DifferenceMethod.BACKWARD
    assert bh.DifferenceMethod.FORWARD == bh.DifferenceMethod.FORWARD

    # Test string representations
    assert "Forward" in str(bh.DifferenceMethod.FORWARD)
    assert "Central" in str(bh.DifferenceMethod.CENTRAL)
    assert "Backward" in str(bh.DifferenceMethod.BACKWARD)


def test_perturbation_strategy_enum():
    """Test PerturbationStrategy creation."""
    adaptive = bh.PerturbationStrategy.adaptive(1.0, 1.0)
    assert "Adaptive" in str(adaptive)

    fixed = bh.PerturbationStrategy.fixed(1e-6)
    assert "Fixed" in str(fixed)

    percentage = bh.PerturbationStrategy.percentage(1e-6)
    assert "Percentage" in str(percentage)


# Simple linear system: dx/dt = Ax where A = [[0, 1], [-1, 0]]
# Analytical Jacobian is constant: A
def linear_dynamics(t, state):
    """Harmonic oscillator dynamics."""
    return np.array([state[1], -state[0]])


def analytical_jacobian(t, state):
    """Analytical Jacobian for harmonic oscillator."""
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


def test_danalytic_jacobian():
    """Test DAnalyticJacobian provider."""
    provider = bh.AnalyticJacobian(analytical_jacobian)
    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(jacobian, expected, atol=1e-10)


def test_danalytic_jacobian_with_list():
    """Test DAnalyticJacobian with Python list input."""
    provider = bh.AnalyticJacobian(analytical_jacobian)
    state = [1.0, 0.5]  # Python list instead of numpy array
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(jacobian, expected, atol=1e-10)


def test_dnumerical_jacobian_central():
    """Test DNumericalJacobian with central differences."""
    provider = bh.NumericalJacobian.central(linear_dynamics).with_fixed_offset(1e-6)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(jacobian, expected, atol=1e-8)


def test_dnumerical_jacobian_forward():
    """Test DNumericalJacobian with forward differences."""
    provider = bh.NumericalJacobian.forward(linear_dynamics).with_fixed_offset(1e-6)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    # Forward differences less accurate than central
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


def test_dnumerical_jacobian_backward():
    """Test DNumericalJacobian with backward differences."""
    provider = bh.NumericalJacobian.backward(linear_dynamics).with_fixed_offset(1e-6)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    # Backward differences less accurate than central
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


def test_dnumerical_jacobian_default():
    """Test DNumericalJacobian with default settings (central, adaptive)."""
    provider = bh.NumericalJacobian(linear_dynamics)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(jacobian, expected, atol=1e-6)


def test_dnumerical_jacobian_with_method():
    """Test DNumericalJacobian with method chaining."""
    provider = (
        bh.NumericalJacobian(linear_dynamics)
        .with_method(bh.DifferenceMethod.FORWARD)
        .with_fixed_offset(1e-6)
    )

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


def test_perturbation_strategies():
    """Test different perturbation strategies."""
    # Test adaptive perturbation
    provider = bh.NumericalJacobian.central(linear_dynamics).with_adaptive(1.0, 1.0)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(jacobian, expected, atol=1e-6)

    # Test percentage perturbation
    provider = bh.NumericalJacobian.central(linear_dynamics).with_percentage(1e-6)

    jacobian = provider.compute(0.0, state)
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


def test_nonlinear_system():
    """Test with a nonlinear system."""

    # Nonlinear system: dx/dt = x^2 + y, dy/dt = -x + y^2
    def nonlinear_dynamics(t, state):
        x, y = state[0], state[1]
        return np.array([x * x + y, -x + y * y])

    # Analytical Jacobian: [[2x, 1], [-1, 2y]]
    def nonlinear_jacobian(t, state):
        x, y = state[0], state[1]
        return np.array([[2 * x, 1.0], [-1.0, 2 * y]])

    state = np.array([2.0, 3.0])

    # Test analytical
    analytical_provider = bh.AnalyticJacobian(nonlinear_jacobian)
    jac_analytical = analytical_provider.compute(0.0, state)

    expected = np.array([[4.0, 1.0], [-1.0, 6.0]])
    np.testing.assert_allclose(jac_analytical, expected, atol=1e-10)

    # Test numerical
    numerical_provider = bh.NumericalJacobian.central(
        nonlinear_dynamics
    ).with_fixed_offset(1e-6)
    jac_numerical = numerical_provider.compute(0.0, state)

    # Numerical should match analytical closely
    np.testing.assert_allclose(jac_numerical, expected, atol=1e-6)


def test_higher_dimensional_system():
    """Test with a 4-dimensional system."""

    # Coupled oscillators
    def coupled_oscillators(t, state):
        # x1' = x2, x2' = -x1 + 0.1*x3
        # x3' = x4, x4' = -x3 + 0.1*x1
        x1, x2, x3, x4 = state
        return np.array([x2, -x1 + 0.1 * x3, x4, -x3 + 0.1 * x1])

    # Analytical Jacobian
    def coupled_jacobian(t, state):
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, -1.0, 0.0],
            ]
        )

    state = np.array([1.0, 0.0, 0.5, 0.0])

    # Test analytical
    analytical_provider = bh.AnalyticJacobian(coupled_jacobian)
    jac_analytical = analytical_provider.compute(0.0, state)

    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, -1.0, 0.0],
        ]
    )

    assert jac_analytical.shape == (4, 4)
    np.testing.assert_allclose(jac_analytical, expected, atol=1e-10)

    # Test numerical
    numerical_provider = bh.NumericalJacobian.central(
        coupled_oscillators
    ).with_fixed_offset(1e-6)
    jac_numerical = numerical_provider.compute(0.0, state)

    assert jac_numerical.shape == (4, 4)
    np.testing.assert_allclose(jac_numerical, expected, atol=1e-6)


def test_dnumerical_jacobian_with_lambda():
    """Test that lambda functions work as dynamics."""
    provider = bh.NumericalJacobian(
        lambda t, s: np.array([s[1], -s[0]])
    ).with_fixed_offset(1e-6)

    state = np.array([1.0, 0.5])
    jacobian = provider.compute(0.0, state)

    expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(jacobian, expected, atol=1e-8)


def test_jacobian_time_dependence():
    """Test Jacobian with time-dependent system."""

    # Time-dependent system: dx/dt = t*x + y, dy/dt = -x + t*y
    def time_dependent_dynamics(t, state):
        x, y = state[0], state[1]
        return np.array([t * x + y, -x + t * y])

    # Jacobian: [[t, 1], [-1, t]]
    def time_dependent_jacobian(t, state):
        return np.array([[t, 1.0], [-1.0, t]])

    state = np.array([1.0, 2.0])
    t = 0.5

    # Test analytical
    analytical_provider = bh.AnalyticJacobian(time_dependent_jacobian)
    jac_analytical = analytical_provider.compute(t, state)

    expected = np.array([[0.5, 1.0], [-1.0, 0.5]])
    np.testing.assert_allclose(jac_analytical, expected, atol=1e-10)

    # Test numerical
    numerical_provider = bh.NumericalJacobian.central(
        time_dependent_dynamics
    ).with_fixed_offset(1e-6)
    jac_numerical = numerical_provider.compute(t, state)

    np.testing.assert_allclose(jac_numerical, expected, atol=1e-6)
