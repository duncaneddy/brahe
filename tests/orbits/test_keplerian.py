import pytest
import math
import numpy as np
import brahe
from brahe import AngleFormat


def test_orbital_period():
    T = brahe.orbital_period(brahe.R_EARTH + 500e3)
    assert T == pytest.approx(5676.977164028288, abs=1e-12)


def test_orbital_period_general():
    a = brahe.R_EARTH + 500e3
    gm = brahe.GM_EARTH
    T = brahe.orbital_period_general(a, gm)

    assert T == pytest.approx(5676.977164028288, abs=1e-12)


def test_orbital_period_general_moon():
    a = brahe.R_MOON + 500e3
    gm = brahe.GM_MOON
    T = brahe.orbital_period_general(a, gm)

    assert T == pytest.approx(9500.531451174307, abs=1e-12)


def test_orbital_period_from_state_circular():
    """Test orbital_period_from_state with a circular orbit"""
    # Create a circular orbit at 500 km altitude
    r = brahe.R_EARTH + 500e3
    v = np.sqrt(brahe.GM_EARTH / r)

    # Create ECI state vector (circular equatorial orbit)
    state_eci = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    # Compute period from state
    period = brahe.orbital_period_from_state(state_eci, brahe.GM_EARTH)

    # Should match the period from semi-major axis
    expected_period = brahe.orbital_period_general(r, brahe.GM_EARTH)
    assert period == pytest.approx(expected_period, abs=1e-8)
    assert period == pytest.approx(5676.977164028288, abs=1e-8)


def test_orbital_period_from_state_elliptical():
    """Test orbital_period_from_state with an elliptical orbit"""
    # Create an elliptical orbit with known semi-major axis
    a = brahe.R_EARTH + 500e3
    e = 0.1

    # Compute position and velocity at perigee
    r_perigee = a * (1.0 - e)
    v_perigee = np.sqrt(brahe.GM_EARTH * (2.0 / r_perigee - 1.0 / a))

    # Create ECI state vector at perigee
    state_eci = np.array([r_perigee, 0.0, 0.0, 0.0, v_perigee, 0.0])

    # Compute period from state
    period = brahe.orbital_period_from_state(state_eci, brahe.GM_EARTH)

    # Should match the period from semi-major axis
    expected_period = brahe.orbital_period_general(a, brahe.GM_EARTH)
    assert period == pytest.approx(expected_period, abs=1e-8)


def test_orbital_period_from_state_different_gm():
    """Test orbital_period_from_state with lunar gravitational parameter"""
    # Test with lunar orbit
    r = brahe.R_MOON + 100e3
    v = np.sqrt(brahe.GM_MOON / r)

    state_eci = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    period = brahe.orbital_period_from_state(state_eci, brahe.GM_MOON)
    expected_period = brahe.orbital_period_general(r, brahe.GM_MOON)

    assert period == pytest.approx(expected_period, abs=1e-8)


def test_orbital_period_from_state_invalid_length():
    """Test that orbital_period_from_state raises error for invalid state length"""
    # Test with wrong length array
    state_invalid = np.array([1.0, 2.0, 3.0])  # Only 3 elements instead of 6

    with pytest.raises(ValueError, match="state_eci must be a 6-element array"):
        brahe.orbital_period_from_state(state_invalid, brahe.GM_EARTH)


def test_mean_motion():
    n = brahe.mean_motion(brahe.R_EARTH + 500e3, angle_format=AngleFormat.RADIANS)
    assert n == pytest.approx(0.0011067836148773837, abs=1e-12)

    n = brahe.mean_motion(brahe.R_EARTH + 500e3, angle_format=AngleFormat.DEGREES)
    assert n == pytest.approx(0.0634140299667068, abs=1e-12)

    n = brahe.mean_motion(brahe.R_EARTH + 500e3, angle_format=AngleFormat.DEGREES)
    assert n == pytest.approx(0.0634140299667068, abs=1e-12)


def test_mean_motion_general():
    n = brahe.mean_motion_general(
        brahe.R_EARTH + 500e3, brahe.GM_MOON, angle_format=AngleFormat.RADIANS
    )
    assert n != pytest.approx(0.0011067836148773837, abs=1e-12)

    n = brahe.mean_motion_general(
        brahe.R_EARTH + 500e3, brahe.GM_MOON, angle_format=AngleFormat.DEGREES
    )
    assert n != pytest.approx(0.0634140299667068, abs=1e-12)


def test_semimajor_axis():
    a = brahe.semimajor_axis(0.0011067836148773837, AngleFormat.RADIANS)
    assert a == pytest.approx(brahe.R_EARTH + 500e3, abs=1e-6)

    a = brahe.semimajor_axis(0.0634140299667068, AngleFormat.DEGREES)
    assert a == pytest.approx(brahe.R_EARTH + 500e3, abs=1e-6)


def test_semimajor_axis_general():
    a = brahe.semimajor_axis_general(
        0.0011067836148773837, brahe.GM_MOON, angle_format=AngleFormat.RADIANS
    )
    assert a != pytest.approx(brahe.R_EARTH + 500e3, abs=1e-6)

    a = brahe.semimajor_axis_general(
        0.0634140299667068, brahe.GM_MOON, angle_format=AngleFormat.DEGREES
    )
    assert a != pytest.approx(brahe.R_EARTH + 500e3, abs=1e-6)


def test_semimajor_axis_from_orbital_period():
    period = brahe.orbital_period_general(brahe.R_EARTH + 500e3, brahe.GM_EARTH)
    a = brahe.semimajor_axis_from_orbital_period(period)
    assert a == pytest.approx(brahe.R_EARTH + 500e3, abs=1e-6)


def test_semimajor_axis_from_orbital_period_general():
    period = brahe.orbital_period_general(brahe.R_SUN + 1000e3, brahe.GM_SUN)
    a = brahe.semimajor_axis_from_orbital_period_general(period, brahe.GM_SUN)
    assert a == pytest.approx(brahe.R_SUN + 1000e3, abs=1e-6)


def test_perigee_velocity():
    vp = brahe.perigee_velocity(brahe.R_EARTH + 500e3, 0.001)
    assert vp == pytest.approx(7620.224976404526, abs=1e-12)


def test_perigee_velocity_general():
    vp = brahe.periapsis_velocity(brahe.R_MOON + 500e3, 0.001, brahe.GM_MOON)
    assert vp == pytest.approx(1481.5842246768275, abs=1e-12)


def test_apogee_velocity():
    vp = brahe.apogee_velocity(brahe.R_EARTH + 500e3, 0.001)
    assert vp == pytest.approx(7604.999751676446, abs=1e-12)


def test_apogee_velocity_general():
    vp = brahe.apoapsis_velocity(brahe.R_MOON + 500e3, 0.001, brahe.GM_MOON)
    assert vp == pytest.approx(1478.624016435715, abs=1e-12)


def test_sun_synchronous_inclination():
    vp = brahe.sun_synchronous_inclination(
        brahe.R_EARTH + 500e3, 0.001, AngleFormat.DEGREES
    )
    assert vp == pytest.approx(97.40172901366881, abs=1e-12)


def test_anomaly_eccentric_to_mean():
    # 0
    M = brahe.anomaly_eccentric_to_mean(0.0, 0.0, angle_format=AngleFormat.RADIANS)
    assert M == 0

    M = brahe.anomaly_eccentric_to_mean(0.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert M == 0

    # 180
    M = brahe.anomaly_eccentric_to_mean(
        math.pi / 2, 0.1, angle_format=AngleFormat.RADIANS
    )
    assert M == pytest.approx(1.4707963267948965, abs=1e-12)

    M = brahe.anomaly_eccentric_to_mean(90.0, 0.1, angle_format=AngleFormat.DEGREES)
    assert M == pytest.approx(84.27042204869177, abs=1e-3)

    # 180
    M = brahe.anomaly_eccentric_to_mean(math.pi, 0.0, angle_format=AngleFormat.RADIANS)
    assert M == pytest.approx(math.pi, abs=1e-12)

    M = brahe.anomaly_eccentric_to_mean(180.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert M == 180.0


def test_anomaly_mean_to_eccentric():
    # 0
    E = brahe.anomaly_mean_to_eccentric(0.0, 0.0, angle_format=AngleFormat.RADIANS)
    assert E == 0

    E = brahe.anomaly_mean_to_eccentric(0.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert E == 0

    # 180
    E = brahe.anomaly_mean_to_eccentric(
        1.4707963267948965, 0.1, angle_format=AngleFormat.RADIANS
    )
    assert E == pytest.approx(math.pi / 2, abs=1e-12)

    E = brahe.anomaly_mean_to_eccentric(
        84.27042204869177, 0.1, angle_format=AngleFormat.DEGREES
    )
    assert E == pytest.approx(90.0, abs=1e-12)

    # 180
    E = brahe.anomaly_mean_to_eccentric(math.pi, 0.0, angle_format=AngleFormat.RADIANS)
    assert E == pytest.approx(math.pi, abs=1e-12)

    E = brahe.anomaly_mean_to_eccentric(180.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert E == 180.0

    # Large Eccentricities
    E = brahe.anomaly_mean_to_eccentric(180.0, 0.9, angle_format=AngleFormat.DEGREES)
    assert E == 180.0


def test_anomaly_true_to_eccentric():
    # 0 degrees
    anm_ecc = brahe.anomaly_true_to_eccentric(0.0, 0.0, AngleFormat.RADIANS)
    assert anm_ecc == 0.0

    anm_ecc = brahe.anomaly_true_to_eccentric(0.0, 0.0, AngleFormat.DEGREES)
    assert anm_ecc == 0.0

    # 180 degrees
    anm_ecc = brahe.anomaly_true_to_eccentric(math.pi, 0.0, AngleFormat.RADIANS)
    assert anm_ecc == math.pi

    anm_ecc = brahe.anomaly_true_to_eccentric(180.0, 0.0, AngleFormat.DEGREES)
    assert anm_ecc == 180.0

    # 90 degrees
    anm_ecc = brahe.anomaly_true_to_eccentric(math.pi / 2.0, 0.0, AngleFormat.RADIANS)
    assert anm_ecc == pytest.approx(math.pi / 2.0, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(90.0, 0.0, AngleFormat.DEGREES)
    assert anm_ecc == pytest.approx(90.0, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(math.pi / 2.0, 0.1, AngleFormat.RADIANS)
    assert anm_ecc == pytest.approx(1.4706289056333368, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(90.0, 0.1, AngleFormat.DEGREES)
    assert anm_ecc == pytest.approx(84.26082952273322, abs=1e-12)


def test_anomaly_eccentric_to_true():
    # 0 degrees
    anm_true = brahe.anomaly_eccentric_to_true(0.0, 0.0, AngleFormat.RADIANS)
    assert anm_true == 0.0

    anm_true = brahe.anomaly_eccentric_to_true(0.0, 0.0, AngleFormat.DEGREES)
    assert anm_true == 0.0

    # 180 degrees
    anm_true = brahe.anomaly_eccentric_to_true(math.pi, 0.0, AngleFormat.RADIANS)
    assert anm_true == math.pi

    anm_true = brahe.anomaly_eccentric_to_true(180.0, 0.0, AngleFormat.DEGREES)
    assert anm_true == 180.0

    # 90 degrees
    anm_true = brahe.anomaly_eccentric_to_true(math.pi / 2.0, 0.0, AngleFormat.RADIANS)
    assert anm_true == pytest.approx(math.pi / 2.0, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(90.0, 0.0, AngleFormat.DEGREES)
    assert anm_true == pytest.approx(90.0, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(math.pi / 2.0, 0.1, AngleFormat.RADIANS)
    assert anm_true == pytest.approx(1.6709637479564563, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(90.0, 0.1, AngleFormat.DEGREES)
    assert anm_true == pytest.approx(95.73917047726677, abs=1e-12)


def test_anomaly_true_to_mean():
    # 0 degrees
    m = brahe.anomaly_true_to_mean(0.0, 0.0, AngleFormat.RADIANS)
    assert m == 0.0

    m = brahe.anomaly_true_to_mean(0.0, 0.0, AngleFormat.DEGREES)
    assert m == 0.0

    # 180 degrees
    m = brahe.anomaly_true_to_mean(math.pi, 0.0, AngleFormat.RADIANS)
    assert m == math.pi

    m = brahe.anomaly_true_to_mean(180.0, 0.0, AngleFormat.DEGREES)
    assert m == 180.0

    # 90 degrees
    m = brahe.anomaly_true_to_mean(math.pi / 2.0, 0.1, AngleFormat.RADIANS)
    assert m == pytest.approx(1.3711301619226748, abs=1e-12)

    m = brahe.anomaly_true_to_mean(90.0, 0.1, AngleFormat.DEGREES)
    assert m == pytest.approx(78.55997144125844, abs=1e-12)


def test_anomaly_mean_to_true():
    # 0 degrees
    nu = brahe.anomaly_mean_to_true(0.0, 0.0, AngleFormat.RADIANS)
    assert nu == 0.0

    nu = brahe.anomaly_mean_to_true(0.0, 0.0, AngleFormat.DEGREES)
    assert nu == 0.0

    # 180 degrees
    nu = brahe.anomaly_mean_to_true(math.pi, 0.0, AngleFormat.RADIANS)
    assert nu == math.pi

    nu = brahe.anomaly_mean_to_true(180.0, 0.0, AngleFormat.DEGREES)
    assert nu == 180.0

    # 90 degrees
    nu = brahe.anomaly_mean_to_true(math.pi / 2.0, 0.1, AngleFormat.RADIANS)
    assert nu == pytest.approx(1.7694813731148669, abs=1e-12)

    nu = brahe.anomaly_mean_to_true(90.0, 0.1, AngleFormat.DEGREES)
    assert nu == pytest.approx(101.38381460649556, abs=1e-12)


def test_periapsis_altitude():
    """Test periapsis_altitude with Earth orbit"""
    # Test with Earth
    a = brahe.R_EARTH + 500e3  # 500 km mean altitude orbit
    e = 0.01  # slight eccentricity
    alt = brahe.periapsis_altitude(a, e, brahe.R_EARTH)

    # Periapsis distance should be a(1-e), altitude is that minus R_EARTH
    expected = a * (1.0 - e) - brahe.R_EARTH
    assert alt == pytest.approx(expected, abs=1.0)

    # Verify it's less than mean altitude
    assert alt < 500e3


def test_perigee_altitude():
    """Test perigee_altitude (Earth-specific function)"""
    # Test Earth-specific function
    a = brahe.R_EARTH + 420e3  # ISS-like orbit (420 km mean altitude)
    e = 0.0005  # very small eccentricity
    alt = brahe.perigee_altitude(a, e)

    # Should match general function with R_EARTH
    expected = brahe.periapsis_altitude(a, e, brahe.R_EARTH)
    assert alt == pytest.approx(expected, abs=1e-6)

    # For very small eccentricity, should be close to mean altitude
    assert alt > 416e3 and alt < 420e3


def test_apoapsis_altitude():
    """Test apoapsis_altitude with Moon orbit"""
    # Test with Moon
    a = brahe.R_MOON + 100e3  # 100 km altitude orbit
    e = 0.05  # moderate eccentricity
    alt = brahe.apoapsis_altitude(a, e, brahe.R_MOON)

    # Apoapsis distance should be a(1+e), altitude is that minus R_MOON
    expected = a * (1.0 + e) - brahe.R_MOON
    assert alt == pytest.approx(expected, abs=1.0)

    # Should be higher than mean altitude
    assert alt > 100e3


def test_apogee_altitude():
    """Test apogee_altitude (Earth-specific function) with highly eccentric orbit"""
    # Test Earth-specific function with highly eccentric orbit (Molniya-type)
    a = 26554e3  # ~26554 km semi-major axis
    e = 0.7  # highly eccentric
    alt = brahe.apogee_altitude(a, e)

    # Should match general function with R_EARTH
    expected = brahe.apoapsis_altitude(a, e, brahe.R_EARTH)
    assert alt == pytest.approx(expected, abs=1e-6)

    # For highly eccentric orbit, apogee should be much higher than mean
    assert alt > 30000e3  # > 30000 km altitude


def test_altitude_symmetry():
    """Test that periapsis and apoapsis are symmetric around semi-major axis"""
    a = brahe.R_EARTH + 1000e3
    e = 0.1

    peri_alt = brahe.perigee_altitude(a, e)
    apo_alt = brahe.apogee_altitude(a, e)

    # Mean altitude should be approximately average of peri and apo
    mean_alt = (peri_alt + apo_alt) / 2.0
    expected_mean = a - brahe.R_EARTH
    assert mean_alt == pytest.approx(expected_mean, abs=1.0)


def test_circular_orbit_altitudes():
    """Test that for circular orbit (e=0), perigee and apogee are equal"""
    a = brahe.R_EARTH + 600e3
    e = 0.0

    peri_alt = brahe.perigee_altitude(a, e)
    apo_alt = brahe.apogee_altitude(a, e)

    assert peri_alt == pytest.approx(apo_alt, abs=1e-6)
    assert peri_alt == pytest.approx(600e3, abs=1.0)
