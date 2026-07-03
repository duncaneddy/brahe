"""
Tests for ForceModelConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/force_model_config.rs
and src/propagators/central_body.rs
"""

import pytest

from brahe import (
    AtmosphericModel,
    CentralBody,
    DragConfiguration,
    EclipseModel,
    EphemerisSource,
    ForceModelConfig,
    FrameTransformationModel,
    GravityConfiguration,
    OccultingBody,
    ParameterSource,
    ThirdBody,
    ThirdBodyConfiguration,
    ZonalHarmonicsDegree,
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
# CentralBody Tests
# =============================================================================


def test_centralbody_properties():
    """Mirrors central_body::tests::test_central_body_properties"""
    assert CentralBody.Earth.naif_id() == 399
    assert CentralBody.Moon.gm() == pytest.approx(4902800066000.0)
    assert CentralBody.Mars.naif_id() == 4
    assert CentralBody.EMB.gm() == 0.0
    assert CentralBody.SSB.is_barycenter()
    assert CentralBody.Moon.fixed_frame() is not None
    assert CentralBody.EMB.fixed_frame() is None


def test_centralbody_from_naif_id():
    """Mirrors central_body::tests::test_from_naif_id"""
    assert CentralBody.from_naif_id(301) == CentralBody.Moon

    enceladus = CentralBody.from_naif_id(602)
    assert enceladus.naif_id() == 602
    assert 7.0e9 < enceladus.gm() < 7.4e9
    assert enceladus.fixed_frame() is not None

    with pytest.raises(ValueError):
        CentralBody.from_naif_id(-42)


def test_centralbody_default_is_earth():
    """ForceModelConfig() central_body defaults to CentralBody.Earth"""
    assert ForceModelConfig().central_body == CentralBody.Earth


def test_centralbody_custom():
    """Test CentralBody.Custom(...) construction"""
    custom = CentralBody.Custom(name="TestBody", naif_id=12345, gm=1.0e10)
    assert custom.naif_id() == 12345
    assert custom.gm() == 1.0e10
    assert custom.radius() is None
    assert custom.omega_vector() is None
    assert custom.fixed_frame() is None
    assert not custom.is_barycenter()


def test_centralbody_equality():
    """CentralBody equality is by-value"""
    assert CentralBody.Earth == CentralBody.Earth
    assert CentralBody.Earth != CentralBody.Moon


# =============================================================================
# OccultingBody Tests
# =============================================================================


def test_occultingbody_radius_and_naif_ids():
    """Mirrors force_model_config::tests::test_occulting_body_radius_and_naif_ids"""
    assert OccultingBody.Earth.naif_id() == 399
    assert OccultingBody.Earth.naif_position_id() == 399
    assert OccultingBody.Moon.naif_id() == 301
    # Mars: physical body (499) vs. SPK position id (4) differ.
    assert OccultingBody.Mars.naif_id() == 499
    assert OccultingBody.Mars.naif_position_id() == 4

    custom = OccultingBody.Custom(name="Europa", naif_id=502, radius=1560.8e3)
    assert custom.radius() == 1560.8e3
    assert custom.naif_id() == 502
    assert custom.naif_position_id() == 502


# =============================================================================
# ThirdBody Tests
# =============================================================================


def test_thirdbody_naif_ids_and_gm():
    """Mirrors force_model_config::ThirdBody naif_id/gm doctests"""
    assert ThirdBody.SUN.naif_id() == 10
    assert ThirdBody.PHOBOS.naif_id() == 401
    assert ThirdBody.DEIMOS.naif_id() == 402
    assert ThirdBody.EARTH.naif_id() == 399

    custom = ThirdBody.Custom(name="Ceres", naif_id=2000001, gm=6.26325e10)
    assert custom.naif_id() == 2000001
    assert custom.gm() == 6.26325e10


def test_thirdbodyconfiguration_with_mars_system_bodies():
    """ThirdBodyConfiguration accepts the new Earth/Phobos/Deimos variants"""
    config = ThirdBodyConfiguration(
        ephemeris_source=EphemerisSource.DE440s,
        bodies=[ThirdBody.PHOBOS, ThirdBody.DEIMOS],
    )
    bodies = config.bodies
    assert bodies[0].naif_id() == 401
    assert bodies[1].naif_id() == 402


# =============================================================================
# ForceModelConfig central_body / for_body / defaults / validate
# =============================================================================


def test_forcemodelconfig_lunar_mars_cislunar_defaults_valid():
    """Mirrors test_lunar_mars_cislunar_defaults_valid"""
    for config in [
        ForceModelConfig.lunar_default(),
        ForceModelConfig.mars_default(),
        ForceModelConfig.cislunar_default(),
    ]:
        config.validate()

    assert ForceModelConfig.lunar_default().central_body == CentralBody.Moon
    assert ForceModelConfig.lunar_default().drag is None
    assert ForceModelConfig.mars_default().central_body == CentralBody.Mars
    assert ForceModelConfig.cislunar_default().central_body == CentralBody.EMB


def test_forcemodelconfig_for_body_constructs_expected_fields():
    """Mirrors test_for_body_constructs_expected_fields"""
    config = ForceModelConfig.for_body(
        CentralBody.Mars,
        GravityConfiguration.point_mass(),
        relativity=True,
        mass=ParameterSource.value(500.0),
    )
    assert config.central_body == CentralBody.Mars
    assert config.drag is None
    assert config.srp is None
    assert config.third_body is None
    assert config.relativity
    # FrameTransformationModel has no __eq__; compare via repr (matches the
    # existing test convention for equality-less config enums in this module).
    assert repr(config.frame_transform) == repr(
        FrameTransformationModel.FULL_EARTH_ROTATION
    )


def test_forcemodelconfig_validate_rejects_harris_priester_non_earth():
    """Mirrors test_validate_rejects_harris_priester_non_earth"""
    config = ForceModelConfig(
        drag=DragConfiguration(
            model=AtmosphericModel.HARRIS_PRIESTER,
            area=ParameterSource.parameter_index(1),
            cd=ParameterSource.parameter_index(2),
        ),
    )
    config.central_body = CentralBody.Moon

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "HarrisPriester" in message
    assert "Moon" in message


def test_forcemodelconfig_validate_rejects_nrlmsise00_non_earth():
    """Mirrors test_validate_rejects_nrlmsise00_non_earth"""
    config = ForceModelConfig(
        drag=DragConfiguration(
            model=AtmosphericModel.NRLMSISE00,
            area=ParameterSource.parameter_index(1),
            cd=ParameterSource.parameter_index(2),
        ),
    )
    config.central_body = CentralBody.Mars

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "NRLMSISE00" in message
    assert "Mars" in message


def test_forcemodelconfig_validate_rejects_earth_zonal_non_earth():
    """Mirrors test_validate_rejects_earth_zonal_non_earth"""
    config = ForceModelConfig(
        gravity=GravityConfiguration.earth_zonal(ZonalHarmonicsDegree.J2)
    )
    config.central_body = CentralBody.Moon

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "EarthZonal" in message
    assert "Moon" in message


def test_forcemodelconfig_validate_rejects_earth_rotation_only_non_earth():
    """Mirrors test_validate_rejects_earth_rotation_only_non_earth"""
    config = ForceModelConfig(
        frame_transform=FrameTransformationModel.EARTH_ROTATION_ONLY
    )
    config.central_body = CentralBody.Mars

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "EarthRotationOnly" in message
    assert "Mars" in message


def test_forcemodelconfig_validate_rejects_low_precision_ephemeris_non_earth():
    """Mirrors test_validate_rejects_low_precision_ephemeris_non_earth"""
    config = ForceModelConfig(
        third_body=ThirdBodyConfiguration(
            ephemeris_source=EphemerisSource.LowPrecision,
            bodies=[ThirdBody.SUN],
        ),
    )
    config.central_body = CentralBody.Moon

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "LowPrecision" in message
    assert "Moon" in message


def test_forcemodelconfig_validate_allows_low_precision_earth_sun_moon():
    """Mirrors test_validate_allows_low_precision_earth_sun_moon"""
    config = ForceModelConfig(
        third_body=ThirdBodyConfiguration(
            ephemeris_source=EphemerisSource.LowPrecision,
            bodies=[ThirdBody.SUN, ThirdBody.MOON],
        ),
    )
    config.central_body = CentralBody.Earth

    config.validate()


def test_forcemodelconfig_validate_rejects_low_precision_earth_planet():
    """Mirrors test_validate_rejects_low_precision_earth_planet"""
    config = ForceModelConfig(
        third_body=ThirdBodyConfiguration(
            ephemeris_source=EphemerisSource.LowPrecision,
            bodies=[ThirdBody.MARS],
        ),
    )
    config.central_body = CentralBody.Earth

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "LowPrecision" in message
    assert "Mars" in message


def test_forcemodelconfig_validate_rejects_third_body_same_naif_id_as_central_body():
    """Mirrors test_validate_rejects_third_body_same_naif_id_as_central_body"""
    config = ForceModelConfig(
        third_body=ThirdBodyConfiguration(
            ephemeris_source=EphemerisSource.DE440s,
            bodies=[ThirdBody.EARTH],
        ),
    )
    config.central_body = CentralBody.Earth

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "NAIF" in message


def test_forcemodelconfig_validate_rejects_spherical_harmonic_barycenter():
    """Mirrors test_validate_rejects_spherical_harmonic_barycenter"""
    config = ForceModelConfig(
        gravity=GravityConfiguration.spherical_harmonic(degree=20, order=20)
    )
    config.central_body = CentralBody.EMB

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "SphericalHarmonic" in message
    assert "Earth-Moon Barycenter" in message


def test_forcemodelconfig_validate_rejects_drag_barycenter():
    """Mirrors test_validate_rejects_drag_barycenter"""
    config = ForceModelConfig(
        drag=DragConfiguration(
            model=AtmosphericModel.exponential(10e3, 1.0, 0.0),
            area=ParameterSource.parameter_index(1),
            cd=ParameterSource.parameter_index(2),
        ),
    )
    config.central_body = CentralBody.SSB

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value).lower()
    assert "drag" in message
    assert "solar system barycenter" in message


def test_forcemodelconfig_validate_rejects_drag_without_radius_and_omega():
    """Mirrors test_validate_rejects_drag_without_radius_and_omega"""
    custom = CentralBody.Custom(name="TestBody", naif_id=12345, gm=1.0e10)
    config = ForceModelConfig(
        drag=DragConfiguration(
            model=AtmosphericModel.exponential(10e3, 1.0, 0.0),
            area=ParameterSource.parameter_index(1),
            cd=ParameterSource.parameter_index(2),
        ),
    )
    config.central_body = custom

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "drag" in message.lower()
    assert "TestBody" in message


def test_forcemodelconfig_validate_rejects_custom_spherical_harmonic_without_fixed_frame():
    """Mirrors test_validate_rejects_custom_spherical_harmonic_without_fixed_frame"""
    custom = CentralBody.Custom(name="TestBody", naif_id=12345, gm=1.0e10, radius=1.0e6)
    config = ForceModelConfig(
        gravity=GravityConfiguration.spherical_harmonic(degree=20, order=20)
    )
    config.central_body = custom

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "SphericalHarmonic" in message
    assert "TestBody" in message
