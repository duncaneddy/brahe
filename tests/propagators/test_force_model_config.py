"""
Tests for ForceModelConfig and related types Python bindings

These tests mirror the Rust tests from src/propagators/force_model_config.rs
and src/propagators/central_body.rs
"""

import warnings

import pytest

import brahe
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
    assert config.tides is not None
    assert config.tides.solid is not None
    assert config.tides.solid.frequency_dependent


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
    assert CentralBody.Mars.naif_id() == 499
    assert CentralBody.EMB.gm() == 0.0
    assert CentralBody.SSB.is_barycenter()
    assert CentralBody.Moon.fixed_frame() is not None
    assert CentralBody.EMB.fixed_frame() is None


def test_centralbody_all_builtin_accessors():
    """Mirrors central_body::tests::test_central_body_all_builtin_accessors:
    exercise every accessor arm for each built-in variant."""
    import numpy as np

    # gm
    assert CentralBody.Moon.gm() == pytest.approx(brahe.GM_MOON)
    assert CentralBody.Mars.gm() == pytest.approx(brahe.GM_MARS)
    assert CentralBody.Earth.gm() == pytest.approx(brahe.GM_EARTH)
    assert CentralBody.EMB.gm() == 0.0
    assert CentralBody.SSB.gm() == 0.0
    # radius
    assert CentralBody.Earth.radius() == pytest.approx(brahe.R_EARTH)
    assert CentralBody.Moon.radius() == pytest.approx(brahe.R_MOON)
    assert CentralBody.Mars.radius() == pytest.approx(brahe.R_MARS)
    assert CentralBody.EMB.radius() is None
    assert CentralBody.SSB.radius() is None
    # naif_id
    assert CentralBody.Earth.naif_id() == 399
    assert CentralBody.Moon.naif_id() == 301
    assert CentralBody.Mars.naif_id() == 499
    assert CentralBody.EMB.naif_id() == 3
    assert CentralBody.SSB.naif_id() == 0
    # omega_vector
    np.testing.assert_allclose(
        CentralBody.Earth.omega_vector(), [0.0, 0.0, brahe.OMEGA_EARTH]
    )
    np.testing.assert_allclose(
        CentralBody.Moon.omega_vector(), [0.0, 0.0, brahe.OMEGA_MOON]
    )
    np.testing.assert_allclose(
        CentralBody.Mars.omega_vector(), [0.0, 0.0, brahe.OMEGA_MARS]
    )
    assert CentralBody.EMB.omega_vector() is None
    assert CentralBody.SSB.omega_vector() is None
    # inertial_frame
    assert CentralBody.Earth.inertial_frame() == brahe.ReferenceFrame.GCRF
    assert CentralBody.Moon.inertial_frame() == brahe.ReferenceFrame.LCI
    assert CentralBody.Mars.inertial_frame() == brahe.ReferenceFrame.MCI
    assert CentralBody.EMB.inertial_frame() == brahe.ReferenceFrame.EMBI
    assert CentralBody.SSB.inertial_frame() == brahe.ReferenceFrame.SSBI
    # fixed_frame
    assert CentralBody.Earth.fixed_frame() == brahe.ReferenceFrame.ITRF
    assert CentralBody.Moon.fixed_frame() == brahe.ReferenceFrame.LFPA
    assert CentralBody.Mars.fixed_frame() == brahe.ReferenceFrame.MCMF
    assert CentralBody.EMB.fixed_frame() is None
    assert CentralBody.SSB.fixed_frame() is None
    # is_barycenter
    assert not CentralBody.Earth.is_barycenter()
    assert CentralBody.EMB.is_barycenter()
    assert CentralBody.SSB.is_barycenter()


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
    assert OccultingBody.Mars.naif_id() == 499
    assert OccultingBody.Mars.naif_position_id() == 499

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
    """ThirdBodyConfiguration accepts the Earth/Phobos/Deimos variants"""
    entries = [
        ThirdBodyConfiguration(ThirdBody.PHOBOS),
        ThirdBodyConfiguration(
            ThirdBody.DEIMOS, ephemeris_source=EphemerisSource.DE440s
        ),
    ]
    assert entries[0].body.naif_id() == 401
    assert entries[1].body.naif_id() == 402


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
    assert config.third_bodies is None
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
        third_bodies=ThirdBodyConfiguration(
            ThirdBody.SUN, ephemeris_source=EphemerisSource.LowPrecision
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
        third_bodies=[
            ThirdBodyConfiguration(
                ThirdBody.SUN, ephemeris_source=EphemerisSource.LowPrecision
            ),
            ThirdBodyConfiguration(
                ThirdBody.MOON, ephemeris_source=EphemerisSource.LowPrecision
            ),
        ],
    )
    config.central_body = CentralBody.Earth

    config.validate()


def test_forcemodelconfig_validate_rejects_low_precision_earth_planet():
    """Mirrors test_validate_rejects_low_precision_earth_planet"""
    config = ForceModelConfig(
        third_bodies=ThirdBodyConfiguration(
            ThirdBody.MARS, ephemeris_source=EphemerisSource.LowPrecision
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
        third_bodies=[ThirdBody.EARTH],
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


def test_forcemodelconfig_validate_rejects_tides_non_earth():
    """Mirrors test_validate_rejects_tides_non_earth"""
    config = ForceModelConfig(
        tides=brahe.TidesConfiguration(permanent=brahe.PermanentTideConfig.AUTO)
    )
    config.central_body = CentralBody.Moon

    with pytest.raises(RuntimeError) as exc_info:
        config.validate()
    message = str(exc_info.value)
    assert "TidesConfiguration" in message
    assert "Moon" in message


# =============================================================================
# Tides Configuration Tests
# =============================================================================


def test_tides_config_roundtrip():
    solid = brahe.SolidTideConfig(frequency_dependent=True)
    assert solid.frequency_dependent is True
    tides = brahe.TidesConfiguration(
        permanent=brahe.PermanentTideConfig.AUTO, solid=solid
    )
    cfg = brahe.ForceModelConfig.two_body()
    cfg.tides = tides
    assert cfg.tides is not None
    assert cfg.tides.solid.frequency_dependent is True


def test_forcemodelconfig_tides_kwarg():
    """Test that ForceModelConfig constructor accepts a tides kwarg and round-trips it."""
    solid = brahe.SolidTideConfig(frequency_dependent=True)
    tides = brahe.TidesConfiguration(
        permanent=brahe.PermanentTideConfig.AUTO, solid=solid
    )
    cfg = brahe.ForceModelConfig(tides=tides)
    assert cfg.tides is not None
    assert cfg.tides.solid is not None
    assert cfg.tides.solid.frequency_dependent is True


def test_tides_config_zero_tide_with_solid_warns():
    """ConvertTo(non-tide-free) + solid tides double-counts the permanent tide and warns."""
    solid = brahe.SolidTideConfig(frequency_dependent=False)
    for system in (
        brahe.GravityModelTideSystem.ZeroTide,
        brahe.GravityModelTideSystem.MeanTide,
    ):
        with pytest.warns(UserWarning, match="double-counts the permanent tide"):
            tides = brahe.TidesConfiguration(
                permanent=brahe.PermanentTideConfig.convert_to(system), solid=solid
            )
        # The configuration is still honored.
        assert tides.solid is not None


def test_tides_config_consistent_combinations_do_not_warn():
    """AUTO/OFF/ConvertTo(TideFree) with solid tides, and ConvertTo without solid, are fine."""
    solid = brahe.SolidTideConfig(frequency_dependent=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        brahe.TidesConfiguration(permanent=brahe.PermanentTideConfig.AUTO, solid=solid)
        brahe.TidesConfiguration(permanent=brahe.PermanentTideConfig.OFF, solid=solid)
        brahe.TidesConfiguration(
            permanent=brahe.PermanentTideConfig.convert_to(
                brahe.GravityModelTideSystem.TideFree
            ),
            solid=solid,
        )
        brahe.TidesConfiguration(
            permanent=brahe.PermanentTideConfig.convert_to(
                brahe.GravityModelTideSystem.ZeroTide
            ),
            solid=None,
        )


def test_third_body_barycenter_planet_split():
    assert brahe.ThirdBody.MARS.naif_id() == 499
    assert brahe.ThirdBody.MARS.gm() == brahe.GM_MARS
    assert brahe.ThirdBody.MARS_BARYCENTER.naif_id() == 4
    assert brahe.ThirdBody.MARS_BARYCENTER.gm() == brahe.GM_MARS_SYSTEM
    assert brahe.ThirdBody.JUPITER.naif_id() == 599
    assert brahe.ThirdBody.JUPITER.gm() == brahe.GM_JUPITER
    assert brahe.ThirdBody.JUPITER_BARYCENTER.naif_id() == 5
    assert brahe.ThirdBody.JUPITER_BARYCENTER.gm() == brahe.GM_JUPITER_SYSTEM
    assert brahe.ThirdBody.SATURN.naif_id() == 699
    assert brahe.ThirdBody.SATURN.gm() == brahe.GM_SATURN
    assert brahe.ThirdBody.SATURN_BARYCENTER.naif_id() == 6
    assert brahe.ThirdBody.SATURN_BARYCENTER.gm() == brahe.GM_SATURN_SYSTEM
    assert brahe.ThirdBody.URANUS.naif_id() == 799
    assert brahe.ThirdBody.URANUS.gm() == brahe.GM_URANUS
    assert brahe.ThirdBody.URANUS_BARYCENTER.naif_id() == 7
    assert brahe.ThirdBody.URANUS_BARYCENTER.gm() == brahe.GM_URANUS_SYSTEM
    assert brahe.ThirdBody.NEPTUNE.naif_id() == 899
    assert brahe.ThirdBody.NEPTUNE.gm() == brahe.GM_NEPTUNE
    assert brahe.ThirdBody.NEPTUNE_BARYCENTER.naif_id() == 8
    assert brahe.ThirdBody.NEPTUNE_BARYCENTER.gm() == brahe.GM_NEPTUNE_SYSTEM


def test_third_body_configuration_defaults():
    cfg = brahe.ThirdBodyConfiguration(brahe.ThirdBody.SUN)
    assert cfg.body == brahe.ThirdBody.SUN
    assert cfg.ephemeris_source == brahe.EphemerisSource.DE440s


def test_force_model_third_bodies_coercion():
    # Single bare body
    fc = brahe.ForceModelConfig(third_bodies=brahe.ThirdBody.SUN)
    assert len(fc.third_bodies) == 1
    assert fc.third_bodies[0].body == brahe.ThirdBody.SUN

    # Mixed list of bodies and configurations
    fc = brahe.ForceModelConfig(
        third_bodies=[
            brahe.ThirdBody.SUN,
            brahe.ThirdBodyConfiguration(brahe.ThirdBody.MOON),
        ]
    )
    assert len(fc.third_bodies) == 2
    assert fc.third_bodies[1].body == brahe.ThirdBody.MOON

    # Default is None
    assert brahe.ForceModelConfig().third_bodies is None

    # Setter accepts the same coercions
    fc.third_bodies = brahe.ThirdBody.MOON
    assert len(fc.third_bodies) == 1
    fc.third_bodies = None
    assert fc.third_bodies is None


def test_third_body_body_fixed_frame():
    assert brahe.ThirdBody.EARTH.body_fixed_frame() == brahe.ReferenceFrame.ITRF
    assert brahe.ThirdBody.MOON.body_fixed_frame() == brahe.ReferenceFrame.LFPA
    assert brahe.ThirdBody.MARS.body_fixed_frame() == brahe.ReferenceFrame.MCMF
    assert brahe.ThirdBody.MARS_BARYCENTER.body_fixed_frame() is None


def test_validate_third_body_gravity_rules():
    """Mirrors the Rust test of the same name."""
    # EarthZonal on a non-Earth third body is rejected
    config = ForceModelConfig.cislunar_default()
    config.third_bodies = [
        ThirdBodyConfiguration(
            ThirdBody.MOON,
            gravity=GravityConfiguration.earth_zonal(ZonalHarmonicsDegree.J2),
        )
    ]
    with pytest.raises(RuntimeError, match="EarthZonal"):
        config.validate()

    # EarthZonal on ThirdBody.EARTH is accepted
    config.third_bodies = [
        ThirdBodyConfiguration(
            ThirdBody.EARTH,
            gravity=GravityConfiguration.earth_zonal(ZonalHarmonicsDegree.J2),
        )
    ]
    config.validate()

    # SphericalHarmonic on a barycenter variant is rejected (no fixed frame)
    config.third_bodies = [
        ThirdBodyConfiguration(
            ThirdBody.JUPITER_BARYCENTER,
            gravity=GravityConfiguration.spherical_harmonic(degree=8, order=8),
        )
    ]
    with pytest.raises(RuntimeError, match="body-fixed frame"):
        config.validate()

    # SphericalHarmonic on Earth as a third body is accepted
    config.third_bodies = [
        ThirdBodyConfiguration(
            ThirdBody.EARTH,
            gravity=GravityConfiguration.spherical_harmonic(degree=8, order=8),
        )
    ]
    config.validate()

    # SphericalHarmonic on a Custom third body is rejected
    config.third_bodies = [
        ThirdBodyConfiguration(
            ThirdBody.Custom(name="Ceres", naif_id=2000001, gm=6.26325e10),
            gravity=GravityConfiguration.spherical_harmonic(degree=4, order=4),
        )
    ]
    with pytest.raises(RuntimeError, match="body-fixed frame"):
        config.validate()


def test_validate_attributed_drag_body():
    """Mirrors the Rust test of the same name."""
    # EMB central + NRLMSISE-00 drag attributed to Earth: accepted
    config = ForceModelConfig.cislunar_default()
    config.mass = ParameterSource.value(1000.0)
    config.drag = DragConfiguration(
        model=AtmosphericModel.NRLMSISE00,
        area=ParameterSource.value(10.0),
        cd=ParameterSource.value(2.2),
        body=CentralBody.Earth,
    )
    config.validate()

    # EMB central + drag with no attributed body: rejected (barycenter)
    config.drag = DragConfiguration(
        model=AtmosphericModel.NRLMSISE00,
        area=ParameterSource.value(10.0),
        cd=ParameterSource.value(2.2),
    )
    assert config.drag.body is None
    with pytest.raises(RuntimeError):
        config.validate()

    # Harris-Priester attributed to the Moon: rejected (Earth-only model)
    config.drag = DragConfiguration(
        model=AtmosphericModel.HARRIS_PRIESTER,
        area=ParameterSource.value(10.0),
        cd=ParameterSource.value(2.2),
        body=CentralBody.Moon,
    )
    with pytest.raises(RuntimeError, match="HarrisPriester"):
        config.validate()

    # Drag attributed to a barycenter: rejected (no radius/spin)
    config.drag = DragConfiguration(
        model=AtmosphericModel.exponential(scale_height=8500.0, rho0=1.225, h0=0.0),
        area=ParameterSource.value(10.0),
        cd=ParameterSource.value(2.2),
        body=CentralBody.EMB,
    )
    with pytest.raises(RuntimeError):
        config.validate()

    # Earth central with NRLMSISE-00, no attribution: still accepted
    ForceModelConfig.leo_default().validate()
