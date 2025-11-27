"""
Tests for NumericalPropagationConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/numerical_propagation_config.rs
"""

from brahe import (
    IntegrationMethod,
    IntegratorConfig,
    NumericalPropagationConfig,
    VariationalConfig,
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


def test_numericalpropagationconfig_new():
    """Test NumericalPropagationConfig.new() constructor with all components"""
    config = NumericalPropagationConfig.new(
        IntegrationMethod.RKN1210,
        IntegratorConfig.adaptive(1e-12, 1e-10),
        VariationalConfig(),
    )

    # Check method via repr since __eq__ may not be implemented
    assert "RKN1210" in repr(config.method)
    assert config.abs_tol == 1e-12
    assert config.rel_tol == 1e-10


def test_numericalpropagationconfig_new_with_variational():
    """Test NumericalPropagationConfig.new() with non-default variational config"""
    variational = VariationalConfig(enable_stm=True, store_stm_history=True)
    config = NumericalPropagationConfig.new(
        IntegrationMethod.DP54,
        IntegratorConfig.adaptive(1e-8, 1e-6),
        variational,
    )

    assert config.abs_tol == 1e-8
    assert config.rel_tol == 1e-6
    assert config.variational.enable_stm is True
    assert config.variational.store_stm_history is True
    assert config.variational.enable_sensitivity is False
