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
    vp = brahe.periapsis_velocity(brahe.R_MOON + 500e3, 0.001, gm=brahe.GM_MOON)
    assert vp == pytest.approx(1481.5842246768275, abs=1e-12)


def test_apogee_velocity():
    vp = brahe.apogee_velocity(brahe.R_EARTH + 500e3, 0.001)
    assert vp == pytest.approx(7604.999751676446, abs=1e-12)


def test_apogee_velocity_general():
    vp = brahe.apoapsis_velocity(brahe.R_MOON + 500e3, 0.001, gm=brahe.GM_MOON)
    assert vp == pytest.approx(1478.624016435715, abs=1e-12)


def test_sun_synchronous_inclination():
    vp = brahe.sun_synchronous_inclination(
        brahe.R_EARTH + 500e3, 0.001, angle_format=AngleFormat.DEGREES
    )
    assert vp == pytest.approx(97.40172901366881, abs=1e-12)


def test_geo_sma():
    a_geo = brahe.geo_sma()
    assert a_geo == pytest.approx(42164172.0, abs=1.0)


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
    anm_ecc = brahe.anomaly_true_to_eccentric(
        0.0, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_ecc == 0.0

    anm_ecc = brahe.anomaly_true_to_eccentric(
        0.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_ecc == 0.0

    # 180 degrees
    anm_ecc = brahe.anomaly_true_to_eccentric(
        math.pi, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_ecc == math.pi

    anm_ecc = brahe.anomaly_true_to_eccentric(
        180.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_ecc == 180.0

    # 90 degrees
    anm_ecc = brahe.anomaly_true_to_eccentric(
        math.pi / 2.0, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_ecc == pytest.approx(math.pi / 2.0, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(
        90.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_ecc == pytest.approx(90.0, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(
        math.pi / 2.0, 0.1, angle_format=AngleFormat.RADIANS
    )
    assert anm_ecc == pytest.approx(1.4706289056333368, abs=1e-12)

    anm_ecc = brahe.anomaly_true_to_eccentric(
        90.0, 0.1, angle_format=AngleFormat.DEGREES
    )
    assert anm_ecc == pytest.approx(84.26082952273322, abs=1e-12)


def test_anomaly_eccentric_to_true():
    # 0 degrees
    anm_true = brahe.anomaly_eccentric_to_true(
        0.0, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_true == 0.0

    anm_true = brahe.anomaly_eccentric_to_true(
        0.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_true == 0.0

    # 180 degrees
    anm_true = brahe.anomaly_eccentric_to_true(
        math.pi, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_true == math.pi

    anm_true = brahe.anomaly_eccentric_to_true(
        180.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_true == 180.0

    # 90 degrees
    anm_true = brahe.anomaly_eccentric_to_true(
        math.pi / 2.0, 0.0, angle_format=AngleFormat.RADIANS
    )
    assert anm_true == pytest.approx(math.pi / 2.0, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(
        90.0, 0.0, angle_format=AngleFormat.DEGREES
    )
    assert anm_true == pytest.approx(90.0, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(
        math.pi / 2.0, 0.1, angle_format=AngleFormat.RADIANS
    )
    assert anm_true == pytest.approx(1.6709637479564563, abs=1e-12)

    anm_true = brahe.anomaly_eccentric_to_true(
        90.0, 0.1, angle_format=AngleFormat.DEGREES
    )
    assert anm_true == pytest.approx(95.73917047726677, abs=1e-12)


def test_anomaly_true_to_mean():
    # 0 degrees
    m = brahe.anomaly_true_to_mean(0.0, 0.0, angle_format=AngleFormat.RADIANS)
    assert m == 0.0

    m = brahe.anomaly_true_to_mean(0.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert m == 0.0

    # 180 degrees
    m = brahe.anomaly_true_to_mean(math.pi, 0.0, angle_format=AngleFormat.RADIANS)
    assert m == math.pi

    m = brahe.anomaly_true_to_mean(180.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert m == 180.0

    # 90 degrees
    m = brahe.anomaly_true_to_mean(math.pi / 2.0, 0.1, angle_format=AngleFormat.RADIANS)
    assert m == pytest.approx(1.3711301619226748, abs=1e-12)

    m = brahe.anomaly_true_to_mean(90.0, 0.1, angle_format=AngleFormat.DEGREES)
    assert m == pytest.approx(78.55997144125844, abs=1e-12)


def test_anomaly_mean_to_true():
    # 0 degrees
    nu = brahe.anomaly_mean_to_true(0.0, 0.0, angle_format=AngleFormat.RADIANS)
    assert nu == 0.0

    nu = brahe.anomaly_mean_to_true(0.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert nu == 0.0

    # 180 degrees
    nu = brahe.anomaly_mean_to_true(math.pi, 0.0, angle_format=AngleFormat.RADIANS)
    assert nu == math.pi

    nu = brahe.anomaly_mean_to_true(180.0, 0.0, angle_format=AngleFormat.DEGREES)
    assert nu == 180.0

    # 90 degrees
    nu = brahe.anomaly_mean_to_true(
        math.pi / 2.0, 0.1, angle_format=AngleFormat.RADIANS
    )
    assert nu == pytest.approx(1.7694813731148669, abs=1e-12)

    nu = brahe.anomaly_mean_to_true(90.0, 0.1, angle_format=AngleFormat.DEGREES)
    assert nu == pytest.approx(101.38381460649556, abs=1e-12)


def test_periapsis_altitude():
    """Test periapsis_altitude with Earth orbit"""
    # Test with Earth
    a = brahe.R_EARTH + 500e3  # 500 km mean altitude orbit
    e = 0.01  # slight eccentricity
    alt = brahe.periapsis_altitude(a, e, r_body=brahe.R_EARTH)

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
    expected = brahe.periapsis_altitude(a, e, r_body=brahe.R_EARTH)
    assert alt == pytest.approx(expected, abs=1e-6)

    # For very small eccentricity, should be close to mean altitude
    assert alt > 416e3 and alt < 420e3


def test_apoapsis_altitude():
    """Test apoapsis_altitude with Moon orbit"""
    # Test with Moon
    a = brahe.R_MOON + 100e3  # 100 km altitude orbit
    e = 0.05  # moderate eccentricity
    alt = brahe.apoapsis_altitude(a, e, r_body=brahe.R_MOON)

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
    expected = brahe.apoapsis_altitude(a, e, r_body=brahe.R_EARTH)
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


# ============================================================================
# Vector Input Tests - Functions accepting Keplerian element vectors
# ============================================================================


def test_orbital_period_vector():
    """Test orbital_period with Keplerian elements vector"""
    a = brahe.R_EARTH + 500e3
    oe = [a, 0.01, math.radians(45), 0, 0, 0]

    # Test with vector
    T_vec = brahe.orbital_period(oe)
    # Test with scalar (for comparison)
    T_scalar = brahe.orbital_period(a)

    assert T_vec == pytest.approx(T_scalar, abs=1e-12)
    assert T_vec == pytest.approx(5676.977164028288, abs=1e-12)


def test_orbital_period_general_vector():
    """Test orbital_period_general with Keplerian elements vector"""
    a = brahe.R_EARTH + 500e3
    oe = [a, 0.01, math.radians(45), 0, 0, 0]

    T_vec = brahe.orbital_period_general(oe, brahe.GM_EARTH)
    T_scalar = brahe.orbital_period_general(a, brahe.GM_EARTH)

    assert T_vec == pytest.approx(T_scalar, abs=1e-12)


def test_mean_motion_vector():
    """Test mean_motion with Keplerian elements vector"""
    a = brahe.R_EARTH + 500e3
    oe = [a, 0.01, math.radians(45), 0, 0, 0]

    n_vec = brahe.mean_motion(oe, angle_format=AngleFormat.RADIANS)
    n_scalar = brahe.mean_motion(a, angle_format=AngleFormat.RADIANS)

    assert n_vec == pytest.approx(n_scalar, abs=1e-12)
    assert n_vec == pytest.approx(0.0011067836148773837, abs=1e-12)


def test_mean_motion_general_vector():
    """Test mean_motion_general with Keplerian elements vector"""
    a = brahe.R_EARTH + 500e3
    oe = [a, 0.01, 45, 0, 0, 0]

    n_vec = brahe.mean_motion_general(
        oe, brahe.GM_EARTH, angle_format=AngleFormat.DEGREES
    )
    n_scalar = brahe.mean_motion_general(
        a, brahe.GM_EARTH, angle_format=AngleFormat.DEGREES
    )

    assert n_vec == pytest.approx(n_scalar, abs=1e-12)


def test_perigee_velocity_vector():
    """Test perigee_velocity with Keplerian elements vector"""
    a = 26554000.0
    e = 0.72
    oe = [a, e, math.radians(63.4), 0, 0, 0]

    v_vec = brahe.perigee_velocity(oe)
    v_scalar = brahe.perigee_velocity(a, e)

    assert v_vec == pytest.approx(v_scalar, abs=1e-6)


def test_periapsis_velocity_vector():
    """Test periapsis_velocity with Keplerian elements vector"""
    a = 5e11
    e = 0.95
    oe = [a, e, math.radians(10), 0, 0, 0]

    v_vec = brahe.periapsis_velocity(oe, gm=brahe.GM_SUN)
    v_scalar = brahe.periapsis_velocity(a, e, gm=brahe.GM_SUN)

    assert v_vec == pytest.approx(v_scalar, abs=1e-6)


def test_apogee_velocity_vector():
    """Test apogee_velocity with Keplerian elements vector"""
    a = 24400000.0
    e = 0.73
    oe = [a, e, math.radians(7), 0, 0, 0]

    v_vec = brahe.apogee_velocity(oe)
    v_scalar = brahe.apogee_velocity(a, e)

    assert v_vec == pytest.approx(v_scalar, abs=1e-6)


def test_apoapsis_velocity_vector():
    """Test apoapsis_velocity with Keplerian elements vector"""
    a = 10000000.0
    e = 0.3
    oe = [a, e, math.radians(30), 0, 0, 0]

    v_vec = brahe.apoapsis_velocity(oe, gm=brahe.GM_MARS)
    v_scalar = brahe.apoapsis_velocity(a, e, gm=brahe.GM_MARS)

    assert v_vec == pytest.approx(v_scalar, abs=1e-6)


def test_periapsis_distance_vector():
    """Test periapsis_distance with Keplerian elements vector"""
    a = 8000000.0
    e = 0.2
    oe = [a, e, math.radians(45), 0, 0, 0]

    r_vec = brahe.periapsis_distance(oe)
    r_scalar = brahe.periapsis_distance(a, e)

    assert r_vec == pytest.approx(r_scalar, abs=1e-6)


def test_apoapsis_distance_vector():
    """Test apoapsis_distance with Keplerian elements vector"""
    a = 8000000.0
    e = 0.2
    oe = [a, e, math.radians(45), 0, 0, 0]

    r_vec = brahe.apoapsis_distance(oe)
    r_scalar = brahe.apoapsis_distance(a, e)

    assert r_vec == pytest.approx(r_scalar, abs=1e-6)


def test_periapsis_altitude_vector():
    """Test periapsis_altitude with Keplerian elements vector"""
    a = brahe.R_EARTH + 500e3
    e = 0.01
    oe = [a, e, math.radians(45), 0, 0, 0]

    alt_vec = brahe.periapsis_altitude(oe, r_body=brahe.R_EARTH)
    alt_scalar = brahe.periapsis_altitude(a, e, r_body=brahe.R_EARTH)

    assert alt_vec == pytest.approx(alt_scalar, abs=1e-6)


def test_perigee_altitude_vector():
    """Test perigee_altitude with Keplerian elements vector"""
    a = brahe.R_EARTH + 420e3
    e = 0.0005
    oe = [a, e, math.radians(51.6), 0, 0, 0]

    alt_vec = brahe.perigee_altitude(oe)
    alt_scalar = brahe.perigee_altitude(a, e)

    assert alt_vec == pytest.approx(alt_scalar, abs=1e-6)


def test_apoapsis_altitude_vector():
    """Test apoapsis_altitude with Keplerian elements vector"""
    a = brahe.R_MOON + 100e3
    e = 0.05
    oe = [a, e, math.radians(30), 0, 0, 0]

    alt_vec = brahe.apoapsis_altitude(oe, r_body=brahe.R_MOON)
    alt_scalar = brahe.apoapsis_altitude(a, e, r_body=brahe.R_MOON)

    assert alt_vec == pytest.approx(alt_scalar, abs=1e-6)


def test_apogee_altitude_vector():
    """Test apogee_altitude with Keplerian elements vector"""
    a = 26554000.0
    e = 0.7
    oe = [a, e, math.radians(63.4), 0, 0, 0]

    alt_vec = brahe.apogee_altitude(oe)
    alt_scalar = brahe.apogee_altitude(a, e)

    assert alt_vec == pytest.approx(alt_scalar, abs=1e-6)


def test_sun_synchronous_inclination_vector():
    """Test sun_synchronous_inclination with Keplerian elements vector"""
    a = brahe.R_EARTH + 600e3
    e = 0.001
    oe = [a, e, 97.8, 0, 0, 0]

    inc_vec = brahe.sun_synchronous_inclination(oe, angle_format=AngleFormat.DEGREES)
    inc_scalar = brahe.sun_synchronous_inclination(
        a, e, angle_format=AngleFormat.DEGREES
    )

    assert inc_vec == pytest.approx(inc_scalar, abs=1e-6)


def test_anomaly_eccentric_to_mean_vector():
    """Test anomaly_eccentric_to_mean with Keplerian elements vector"""
    E = math.pi / 4
    e = 0.1
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, E]

    M_vec = brahe.anomaly_eccentric_to_mean(oe, angle_format=AngleFormat.RADIANS)
    M_scalar = brahe.anomaly_eccentric_to_mean(E, e, angle_format=AngleFormat.RADIANS)

    assert M_vec == pytest.approx(M_scalar, abs=1e-12)


def test_anomaly_mean_to_eccentric_vector():
    """Test anomaly_mean_to_eccentric with Keplerian elements vector"""
    M = 1.5
    e = 0.3
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, M]

    E_vec = brahe.anomaly_mean_to_eccentric(oe, angle_format=AngleFormat.RADIANS)
    E_scalar = brahe.anomaly_mean_to_eccentric(M, e, angle_format=AngleFormat.RADIANS)

    assert E_vec == pytest.approx(E_scalar, abs=1e-12)


def test_anomaly_true_to_eccentric_vector():
    """Test anomaly_true_to_eccentric with Keplerian elements vector"""
    nu = math.pi / 3
    e = 0.2
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, nu]

    E_vec = brahe.anomaly_true_to_eccentric(oe, angle_format=AngleFormat.RADIANS)
    E_scalar = brahe.anomaly_true_to_eccentric(nu, e, angle_format=AngleFormat.RADIANS)

    assert E_vec == pytest.approx(E_scalar, abs=1e-12)


def test_anomaly_eccentric_to_true_vector():
    """Test anomaly_eccentric_to_true with Keplerian elements vector"""
    E = math.pi / 4
    e = 0.4
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, E]

    nu_vec = brahe.anomaly_eccentric_to_true(oe, angle_format=AngleFormat.RADIANS)
    nu_scalar = brahe.anomaly_eccentric_to_true(E, e, angle_format=AngleFormat.RADIANS)

    assert nu_vec == pytest.approx(nu_scalar, abs=1e-12)


def test_anomaly_true_to_mean_vector():
    """Test anomaly_true_to_mean with Keplerian elements vector"""
    nu = math.pi / 2
    e = 0.15
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, nu]

    M_vec = brahe.anomaly_true_to_mean(oe, angle_format=AngleFormat.RADIANS)
    M_scalar = brahe.anomaly_true_to_mean(nu, e, angle_format=AngleFormat.RADIANS)

    assert M_vec == pytest.approx(M_scalar, abs=1e-12)


def test_anomaly_mean_to_true_vector():
    """Test anomaly_mean_to_true with Keplerian elements vector"""
    M = 2.0
    e = 0.25
    oe = [brahe.R_EARTH + 500e3, e, math.radians(45), 0, 0, M]

    nu_vec = brahe.anomaly_mean_to_true(oe, angle_format=AngleFormat.RADIANS)
    nu_scalar = brahe.anomaly_mean_to_true(M, e, angle_format=AngleFormat.RADIANS)

    assert nu_vec == pytest.approx(nu_scalar, abs=1e-12)


def test_vector_with_numpy_array():
    """Test that numpy arrays work as Keplerian element vectors"""
    a = brahe.R_EARTH + 500e3
    e = 0.01
    oe_np = np.array([a, e, math.radians(45), 0, 0, 0])

    T_np = brahe.orbital_period(oe_np)
    T_scalar = brahe.orbital_period(a)

    assert T_np == pytest.approx(T_scalar, abs=1e-12)


def test_vector_with_list():
    """Test that Python lists work as Keplerian element vectors"""
    a = brahe.R_EARTH + 500e3
    e = 0.01
    oe_list = [a, e, math.radians(45), 0, 0, 0]

    T_list = brahe.orbital_period(oe_list)
    T_scalar = brahe.orbital_period(a)

    assert T_list == pytest.approx(T_scalar, abs=1e-12)


def test_vector_wrong_length():
    """Test that wrong-length vectors raise appropriate errors"""
    oe_short = [brahe.R_EARTH + 500e3, 0.01]

    with pytest.raises(ValueError, match="Expected array or list of length 6"):
        brahe.orbital_period(oe_short)


def test_vector_missing_e_parameter():
    """Test that scalar input without e parameter raises error"""
    a = brahe.R_EARTH + 500e3

    with pytest.raises(ValueError, match="Parameter 'e' is required"):
        brahe.perigee_velocity(a)  # Missing required 'e' parameter
