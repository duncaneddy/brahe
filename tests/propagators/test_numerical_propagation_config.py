"""
Tests for NumericalPropagationConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/numerical_propagation_config.rs
"""

from brahe import (
    IntegrationMethod,
    IntegratorConfig,
    InterpolationMethod,
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
    """Test NumericalPropagationConfig() constructor with all components"""
    config = NumericalPropagationConfig(
        IntegrationMethod.RKN1210,
        IntegratorConfig.adaptive(1e-12, 1e-10),
        VariationalConfig(),
    )

    # Check method via repr since __eq__ may not be implemented
    assert "RKN1210" in repr(config.method)
    assert config.abs_tol == 1e-12
    assert config.rel_tol == 1e-10


def test_numericalpropagationconfig_new_with_variational():
    """Test NumericalPropagationConfig() with non-default variational config"""
    variational = VariationalConfig(enable_stm=True, store_stm_history=True)
    config = NumericalPropagationConfig(
        IntegrationMethod.DP54,
        IntegratorConfig.adaptive(1e-8, 1e-6),
        variational,
    )

    assert config.abs_tol == 1e-8
    assert config.rel_tol == 1e-6
    assert config.variational.enable_stm is True
    assert config.variational.store_stm_history is True
    assert config.variational.enable_sensitivity is False


# =============================================================================
# Acceleration Storage and Interpolation Method Tests
# =============================================================================


def test_numericalpropagationconfig_default_accelerations():
    """Test NumericalPropagationConfig default acceleration settings"""
    config = NumericalPropagationConfig.default()
    # store_accelerations defaults to True
    assert config.store_accelerations is True
    # interpolation_method defaults to HermiteCubic
    assert config.interpolation_method == InterpolationMethod.HERMITE_CUBIC


def test_numericalpropagationconfig_with_store_accelerations():
    """Test NumericalPropagationConfig.with_store_accelerations()"""
    config = NumericalPropagationConfig.default().with_store_accelerations(False)
    assert config.store_accelerations is False

    config2 = NumericalPropagationConfig.default().with_store_accelerations(True)
    assert config2.store_accelerations is True


def test_numericalpropagationconfig_with_interpolation_method():
    """Test NumericalPropagationConfig.with_interpolation_method()"""
    # Lagrange is a method that takes degree
    lagrange_8 = InterpolationMethod.lagrange(8)
    config = NumericalPropagationConfig.default().with_interpolation_method(lagrange_8)
    assert config.interpolation_method.degree == 8

    config2 = NumericalPropagationConfig.default().with_interpolation_method(
        InterpolationMethod.HERMITE_CUBIC
    )
    assert config2.interpolation_method == InterpolationMethod.HERMITE_CUBIC

    config3 = NumericalPropagationConfig.default().with_interpolation_method(
        InterpolationMethod.HERMITE_QUINTIC
    )
    assert config3.interpolation_method == InterpolationMethod.HERMITE_QUINTIC


def test_numericalpropagationconfig_high_precision_new_fields():
    """Test high_precision() sets correct acceleration and interpolation defaults"""
    config = NumericalPropagationConfig.high_precision()
    # high_precision should also have store_accelerations=True and interpolation=HermiteCubic
    assert config.store_accelerations is True
    assert config.interpolation_method == InterpolationMethod.HERMITE_CUBIC


def test_numericalpropagationconfig_builder_chaining():
    """Test builder pattern chaining with new methods"""
    config = (
        NumericalPropagationConfig.default()
        .with_store_accelerations(True)
        .with_interpolation_method(InterpolationMethod.HERMITE_CUBIC)
    )

    assert config.store_accelerations is True
    assert config.interpolation_method == InterpolationMethod.HERMITE_CUBIC


def test_numericalpropagationconfig_store_accelerations_setter():
    """Test store_accelerations property setter"""
    config = NumericalPropagationConfig.default()
    config.store_accelerations = False
    assert config.store_accelerations is False
    config.store_accelerations = True
    assert config.store_accelerations is True


def test_numericalpropagationconfig_interpolation_method_setter():
    """Test interpolation_method property setter"""
    config = NumericalPropagationConfig.default()
    lagrange_8 = InterpolationMethod.lagrange(8)
    config.interpolation_method = lagrange_8
    assert config.interpolation_method.degree == 8
    config.interpolation_method = InterpolationMethod.LINEAR
    assert config.interpolation_method == InterpolationMethod.LINEAR
