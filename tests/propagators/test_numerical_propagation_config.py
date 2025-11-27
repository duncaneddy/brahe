"""
Tests for NumericalPropagationConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/numerical_propagation_config.rs
"""

from brahe import (
    IntegrationMethod,
    NumericalPropagationConfig,
)


# =============================================================================
# IntegrationMethod Tests
# =============================================================================


def test_integrationmethod_classattrs():
    """Test IntegrationMethod class attributes"""
    assert IntegrationMethod.RK4 is not None
    assert IntegrationMethod.RKF45 is not None
    assert IntegrationMethod.DP54 is not None
    assert IntegrationMethod.RKN1210 is not None


def test_integrationmethod_is_adaptive():
    """Test IntegrationMethod.is_adaptive()"""
    assert not IntegrationMethod.RK4.is_adaptive()
    assert IntegrationMethod.RKF45.is_adaptive()
    assert IntegrationMethod.DP54.is_adaptive()
    assert IntegrationMethod.RKN1210.is_adaptive()


# =============================================================================
# NumericalPropagationConfig Tests
# =============================================================================


def test_numericalpropagationconfig_default():
    """Test NumericalPropagationConfig.default()"""
    config = NumericalPropagationConfig.default()
    assert config is not None
    # Default is DP54
    assert config.method is not None


def test_numericalpropagationconfig_with_method():
    """Test NumericalPropagationConfig.with_method()"""
    config = NumericalPropagationConfig.with_method(IntegrationMethod.RKF45)
    assert config is not None


def test_numericalpropagationconfig_high_precision():
    """Test NumericalPropagationConfig.high_precision()"""
    config = NumericalPropagationConfig.high_precision()
    assert config is not None


def test_numericalpropagationconfig_tolerances():
    """Test tolerance getters on NumericalPropagationConfig"""
    config = NumericalPropagationConfig.default()
    assert config.abs_tol > 0
    assert config.rel_tol > 0
