"""
Tests for SEZ (South, East, Zenith) topocentric frame transformations.

These tests mirror the Rust tests in src/relative_motion/ecef_sez.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe


def _site_state():
    """A fixed site at 30 deg E, 45 deg N, 500 m altitude."""
    r = brahe.position_geodetic_to_ecef(
        np.array([30.0, 45.0, 500.0]), brahe.AngleFormat.DEGREES
    )
    return np.array([r[0], r[1], r[2], 0.0, 0.0, 0.0])


def _moving_site(dt):
    """A site moving over the ellipsoid at aircraft-like speed, advanced by
    `dt` seconds along a straight ECEF line."""
    r0 = brahe.position_geodetic_to_ecef(
        np.array([30.0, 45.0, 10e3]), brahe.AngleFormat.DEGREES
    )
    v = np.array([-120.0, 180.0, 90.0])
    r = r0 + v * dt
    return np.array([r[0], r[1], r[2], v[0], v[1], v[2]])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _omega_fd(x_minus, x0, x_plus, dt):
    r_dot = (
        brahe.rotation_ecef_to_sez(x_plus) - brahe.rotation_ecef_to_sez(x_minus)
    ) / (2.0 * dt)
    m = -(r_dot @ brahe.rotation_ecef_to_sez(x0).T)
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def test_rotation_ecef_to_sez_matches_ellipsoid_rotation():
    """Rust: test_rotation_ecef_to_sez_matches_ellipsoid_rotation"""
    x = _site_state()
    lla = brahe.position_ecef_to_geodetic(x[:3], brahe.AngleFormat.RADIANS)
    expected = brahe.rotation_ellipsoid_to_sez(lla, brahe.AngleFormat.RADIANS)
    np.testing.assert_array_equal(brahe.rotation_ecef_to_sez(x), expected)
    np.testing.assert_array_equal(brahe.rotation_sez_to_ecef(x), expected.T)


def test_rotation_sez_axes_match_definition():
    """Rust: test_rotation_sez_axes_match_definition"""
    x = _site_state()
    m = brahe.rotation_sez_to_ecef(x)
    s, e, z = m[:, 0], m[:, 1], m[:, 2]
    lon, lat = np.radians(30.0), np.radians(45.0)
    np.testing.assert_allclose(
        z,
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        atol=1e-12,
    )
    np.testing.assert_allclose(e, [-np.sin(lon), np.cos(lon), 0.0], atol=1e-12)
    np.testing.assert_allclose(s, np.cross(e, z), atol=1e-12)
    # S points south: negative z component in the northern hemisphere
    assert s[2] < 0.0
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_state_ecef_to_sez_position_matches_relative_position():
    """Rust: test_state_ecef_to_sez_position_matches_relative_position"""
    x_site = _site_state()
    r_target = x_site[:3] + np.array([200e3, 300e3, 400e3])
    x_target = np.array([r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0])
    expected = brahe.relative_position_ecef_to_sez(
        x_site[:3], r_target, brahe.EllipsoidalConversionType.GEODETIC
    )
    rel = brahe.state_ecef_to_sez(x_site, x_target)
    np.testing.assert_allclose(rel[:3], expected, atol=1e-9)
    # A static target seen from a static site has no relative velocity
    assert np.linalg.norm(rel[3:6]) == approx(0.0, abs=1e-15)


def test_omega_sez_static_site_is_zero():
    """Rust: test_omega_sez_static_site_is_zero"""
    np.testing.assert_array_equal(brahe.omega_sez(_site_state()), np.zeros(3))


def test_omega_sez_moving_site_matches_finite_difference():
    """Rust: test_omega_sez_moving_site_matches_finite_difference"""
    dt = 0.05
    x0 = _moving_site(0.0)
    omega_fd = _omega_fd(_moving_site(-dt), x0, _moving_site(dt), dt)
    omega = brahe.omega_sez(x0)
    np.testing.assert_allclose(omega_fd, omega, atol=1e-10)
    assert np.linalg.norm(omega) > 1e-6


def test_omega_sez_components_follow_transport_rate():
    """Rust: test_omega_sez_components_follow_transport_rate"""
    # Pure eastward motion at the equator: only the longitude rate,
    # about the polar axis, which is -S there
    r0 = brahe.position_geodetic_to_ecef(
        np.array([0.0, 0.0, 0.0]), brahe.AngleFormat.DEGREES
    )
    v = np.array([0.0, 100.0, 0.0])
    x = np.array([r0[0], r0[1], r0[2], v[0], v[1], v[2]])
    omega = brahe.omega_sez(x)
    lon_dot = 100.0 / brahe.WGS84_A
    np.testing.assert_allclose(omega, [-lon_dot, 0.0, 0.0], atol=1e-15)

    # Pure northward motion at the equator: only the latitude rate about -E
    v = np.array([0.0, 0.0, 100.0])
    x = np.array([r0[0], r0[1], r0[2], v[0], v[1], v[2]])
    meridian_radius = brahe.WGS84_A * (1.0 - brahe.WGS84_F * (2.0 - brahe.WGS84_F))
    np.testing.assert_allclose(
        brahe.omega_sez(x), [0.0, -100.0 / meridian_radius, 0.0], atol=1e-15
    )


def test_state_sez_to_ecef_round_trip():
    """Rust: test_state_sez_to_ecef_round_trip"""
    for x_site in (_site_state(), _moving_site(0.0)):
        x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
        x_target = brahe.state_sez_to_ecef(x_site, x_rel)
        np.testing.assert_allclose(
            brahe.state_ecef_to_sez(x_site, x_target), x_rel, atol=1e-8
        )


def test_batch_sez_match_scalar():
    """Rust: test_batch_sez_match_scalar"""
    sites = np.array([_moving_site(10.0 * i) for i in range(3)])
    targets = np.array(
        [s + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for s in sites]
    )

    rot = brahe.rotation_ecef_to_sez(sites)
    rot_inv = brahe.rotation_sez_to_ecef(sites)
    omegas = brahe.omega_sez(sites)
    rel = brahe.state_ecef_to_sez(sites, targets)
    rel_broadcast = brahe.state_ecef_to_sez(sites[:1], targets)
    back = brahe.state_sez_to_ecef(sites, rel)

    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    assert rel.shape == (3, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_ecef_to_sez(sites[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_sez_to_ecef(sites[i]))
        np.testing.assert_array_equal(omegas[i], brahe.omega_sez(sites[i]))
        np.testing.assert_array_equal(
            rel[i], brahe.state_ecef_to_sez(sites[i], targets[i])
        )
        np.testing.assert_array_equal(
            rel_broadcast[i], brahe.state_ecef_to_sez(sites[0], targets[i])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_sez_to_ecef(sites[i], rel[i])
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_ecef_to_sez(sites.T, axis=0), rot)
    np.testing.assert_array_equal(brahe.omega_sez(sites.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_ecef_to_sez(sites.T, targets.T, axis=0), rel.T
    )


def test_batch_sez_length_mismatch_raises():
    """Rust: mirrors the broadcast-rule error checks in test_batch_sez_match_scalar"""
    sites = np.array([_moving_site(10.0 * i) for i in range(3)])
    targets = np.array(
        [s + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for s in sites]
    )
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_ecef_to_sez(sites[:2], targets)
