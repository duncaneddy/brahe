"""
Tests for ForceModelConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/force_model_config.rs
"""

import brahe
from brahe import (
    AtmosphericModel,
    EclipseModel,
    ForceModelConfig,
)


# =============================================================================
# AtmosphericModel Tests
# =============================================================================


def test_atmosphericmodel_classattrs():
    """Test AtmosphericModel class attributes"""
    assert AtmosphericModel.HARRIS_PRIESTER is not None
    assert AtmosphericModel.NRLMSISE00 is not None


def test_atmosphericmodel_exponential():
    """Test AtmosphericModel.exponential() class method"""
    model = AtmosphericModel.exponential(8500.0, 1.225e-12, 0.0)
    assert model is not None


# =============================================================================
# EclipseModel Tests
# =============================================================================


def test_eclipsemodel_classattrs():
    """Test EclipseModel class attributes"""
    assert EclipseModel.CONICAL is not None
    assert EclipseModel.CYLINDRICAL is not None


# =============================================================================
# ForceModelConfig Tests
# =============================================================================


def test_forcemodelconfig_default():
    """Test ForceModelConfig.default()"""
    config = ForceModelConfig.default()
    assert config is not None
    assert config.requires_params()


def test_forcemodelconfig_high_fidelity():
    """Test ForceModelConfig.high_fidelity()"""
    config = ForceModelConfig.high_fidelity()
    assert config is not None
    assert config.requires_params()


def test_forcemodelconfig_earth_gravity():
    """Test ForceModelConfig.earth_gravity()"""
    config = ForceModelConfig.earth_gravity()
    assert config is not None
    assert not config.requires_params()  # No drag/SRP, no params needed


def test_forcemodelconfig_two_body():
    """Test ForceModelConfig.two_body()"""
    config = ForceModelConfig.two_body()
    assert config is not None
    assert not config.requires_params()


def test_forcemodelconfig_conservative_forces():
    """Test ForceModelConfig.conservative_forces()"""
    config = ForceModelConfig.conservative_forces()
    assert config is not None


def test_forcemodelconfig_leo_default():
    """Test ForceModelConfig.leo_default()"""
    config = ForceModelConfig.leo_default()
    assert config is not None
    assert config.requires_params()


def test_forcemodelconfig_geo_default():
    """Test ForceModelConfig.geo_default()"""
    config = ForceModelConfig.geo_default()
    assert config is not None


# =============================================================================
# Tides Configuration Tests
# =============================================================================


def test_tides_config_roundtrip():
    solid = brahe.SolidTideConfig(frequency_dependent=True)
    assert solid.frequency_dependent is True
    tides = brahe.TidesConfiguration(permanent=brahe.PermanentTideConfig.AUTO, solid=solid)
    cfg = brahe.ForceModelConfig.two_body()
    cfg.tides = tides
    assert cfg.tides is not None
    assert cfg.tides.solid.frequency_dependent is True


def test_forcemodelconfig_tides_kwarg():
    """Test that ForceModelConfig constructor accepts a tides kwarg and round-trips it."""
    solid = brahe.SolidTideConfig(frequency_dependent=True)
    tides = brahe.TidesConfiguration(permanent=brahe.PermanentTideConfig.AUTO, solid=solid)
    cfg = brahe.ForceModelConfig(tides=tides)
    assert cfg.tides is not None
    assert cfg.tides.solid is not None
    assert cfg.tides.solid.frequency_dependent is True
