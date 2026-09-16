"""
Tests for ENZ (East, North, Zenith) topocentric frame transformations.

These tests mirror the Rust tests in src/relative_motion/ecef_enz.rs
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
        brahe.rotation_ecef_to_enz(x_plus) - brahe.rotation_ecef_to_enz(x_minus)
    ) / (2.0 * dt)
    m = -(r_dot @ brahe.rotation_ecef_to_enz(x0).T)
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def test_rotation_ecef_to_enz_matches_ellipsoid_rotation():
    """Rust: test_rotation_ecef_to_enz_matches_ellipsoid_rotation"""
    x = _site_state()
    lla = brahe.position_ecef_to_geodetic(x[:3], brahe.AngleFormat.RADIANS)
    expected = brahe.rotation_ellipsoid_to_enz(lla, brahe.AngleFormat.RADIANS)
    np.testing.assert_allclose(brahe.rotation_ecef_to_enz(x), expected, atol=1e-15)
    np.testing.assert_allclose(brahe.rotation_enz_to_ecef(x), expected.T, atol=1e-15)


def test_rotation_enz_is_permuted_sez():
    """Rust: test_rotation_enz_is_permuted_sez"""
    x = _site_state()
    m_sez = brahe.rotation_sez_to_ecef(x)
    m_enz = brahe.rotation_enz_to_ecef(x)
    s_sez, e_sez, z_sez = m_sez[:, 0], m_sez[:, 1], m_sez[:, 2]
    e_enz, n_enz, z_enz = m_enz[:, 0], m_enz[:, 1], m_enz[:, 2]
    np.testing.assert_array_equal(e_enz, e_sez)
    np.testing.assert_array_equal(n_enz, -s_sez)
    np.testing.assert_array_equal(z_enz, z_sez)


def test_omega_enz_is_sez_rate_in_enz_axes():
    """Rust: test_omega_enz_is_sez_rate_in_enz_axes"""
    x = _moving_site(0.0)
    omega_sez = brahe.omega_sez(x)
    omega_enz = brahe.omega_enz(x)
    assert omega_enz[0] == omega_sez[1]
    assert omega_enz[1] == -omega_sez[0]
    assert omega_enz[2] == omega_sez[2]


def test_omega_enz_moving_site_matches_finite_difference():
    """Rust: test_omega_enz_moving_site_matches_finite_difference"""
    dt = 0.05
    x0 = _moving_site(0.0)
    omega_fd = _omega_fd(_moving_site(-dt), x0, _moving_site(dt), dt)
    omega = brahe.omega_enz(x0)
    np.testing.assert_allclose(omega_fd, omega, atol=1e-10)
    assert np.linalg.norm(omega) > 1e-6


def test_state_ecef_to_enz_position_matches_relative_position():
    """Rust: test_state_ecef_to_enz_position_matches_relative_position"""
    x_site = _site_state()
    r_target = x_site[:3] + np.array([200e3, 300e3, 400e3])
    x_target = np.array([r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0])
    expected = brahe.relative_position_ecef_to_enz(
        x_site[:3], r_target, brahe.EllipsoidalConversionType.GEODETIC
    )
    rel = brahe.state_ecef_to_enz(x_site, x_target)
    np.testing.assert_allclose(rel[:3], expected, atol=1e-9)
    # A static target seen from a static site has no relative velocity
    assert np.linalg.norm(rel[3:6]) == approx(0.0, abs=1e-15)


def test_jacobian_enz_ecef_forms():
    """Rust: test_jacobian_enz_ecef_forms"""
    x = _moving_site(0.0)
    j_i = brahe.jacobian_enz_to_ecef(x, brahe.OrbitRelativeFrameVariant.INERTIAL)
    r = brahe.rotation_enz_to_ecef(x)
    zeros = np.zeros((3, 3))
    block = np.block([[r, zeros], [zeros, r]])
    np.testing.assert_allclose(j_i, block, atol=1e-15)

    j_r = brahe.jacobian_enz_to_ecef(x, brahe.OrbitRelativeFrameVariant.ROTATING)
    coupling = j_r[3:, :3]
    expected = r @ _skew(brahe.omega_enz(x))
    np.testing.assert_allclose(coupling, expected, atol=1e-18)

    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        forward = brahe.jacobian_enz_to_ecef(x, variant)
        inverse = brahe.jacobian_ecef_to_enz(x, variant)
        np.testing.assert_allclose(inverse @ forward, np.eye(6), atol=1e-12)


def test_covariance_enz_ecef_round_trip():
    """Rust: test_covariance_enz_ecef_round_trip"""
    x = _moving_site(0.0)
    p = np.zeros((6, 6))
    for i in range(3):
        p[i, i] = 100.0
        p[3 + i, 3 + i] = 0.01
    p[0, 1] = 25.0
    p[1, 0] = 25.0
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        p_ecef = brahe.covariance_enz_to_ecef(x, p, variant)
        p_back = brahe.covariance_ecef_to_enz(x, p_ecef, variant)
        assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)


def test_state_enz_to_ecef_round_trip():
    """Rust: test_state_enz_to_ecef_round_trip"""
    for x_site in (_site_state(), _moving_site(0.0)):
        x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
        x_target = brahe.state_enz_to_ecef(x_site, x_rel)
        np.testing.assert_allclose(
            brahe.state_ecef_to_enz(x_site, x_target), x_rel, atol=1e-8
        )


def test_batch_enz_match_scalar():
    """Rust: test_batch_enz_match_scalar"""
    sites = np.array([_moving_site(10.0 * i) for i in range(3)])
    targets = np.array(
        [s + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for s in sites]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])

    rot = brahe.rotation_ecef_to_enz(sites)
    rot_inv = brahe.rotation_enz_to_ecef(sites)
    omegas = brahe.omega_enz(sites)
    rel = brahe.state_ecef_to_enz(sites, targets)
    rel_broadcast = brahe.state_ecef_to_enz(sites[:1], targets)
    back = brahe.state_enz_to_ecef(sites, rel)

    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    assert rel.shape == (3, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_ecef_to_enz(sites[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_enz_to_ecef(sites[i]))
        np.testing.assert_array_equal(omegas[i], brahe.omega_enz(sites[i]))
        np.testing.assert_array_equal(
            rel[i], brahe.state_ecef_to_enz(sites[i], targets[i])
        )
        np.testing.assert_array_equal(
            rel_broadcast[i], brahe.state_ecef_to_enz(sites[0], targets[i])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_enz_to_ecef(sites[i], rel[i])
        )

    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        jac = brahe.jacobian_enz_to_ecef(sites, variant)
        jac_inv = brahe.jacobian_ecef_to_enz(sites, variant)
        cov = brahe.covariance_enz_to_ecef(sites, covs, variant)
        cov_broadcast = brahe.covariance_enz_to_ecef(sites, covs[:1], variant)
        cov_inv = brahe.covariance_ecef_to_enz(sites, covs, variant)

        assert jac.shape == (3, 6, 6)
        assert cov.shape == (3, 6, 6)

        for i in range(3):
            np.testing.assert_array_equal(
                jac[i], brahe.jacobian_enz_to_ecef(sites[i], variant)
            )
            np.testing.assert_array_equal(
                jac_inv[i], brahe.jacobian_ecef_to_enz(sites[i], variant)
            )
            np.testing.assert_array_equal(
                cov[i], brahe.covariance_enz_to_ecef(sites[i], covs[i], variant)
            )
            np.testing.assert_array_equal(
                cov_broadcast[i],
                brahe.covariance_enz_to_ecef(sites[i], covs[0], variant),
            )
            np.testing.assert_array_equal(
                cov_inv[i],
                brahe.covariance_ecef_to_enz(sites[i], covs[i], variant),
            )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_ecef_to_enz(sites.T, axis=0), rot)
    np.testing.assert_array_equal(brahe.omega_enz(sites.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_ecef_to_enz(sites.T, targets.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_enz_to_ecef(
            sites.T, covs, brahe.OrbitRelativeFrameVariant.ROTATING, axis=0
        ),
        cov,
    )

    x_nd = np.stack([sites, sites])
    cov_nd = brahe.covariance_enz_to_ecef(
        x_nd, covs[0], brahe.OrbitRelativeFrameVariant.ROTATING
    )
    assert cov_nd.shape == (2, 3, 6, 6)
    np.testing.assert_array_equal(
        cov_nd[1],
        brahe.covariance_enz_to_ecef(
            sites, covs[0], brahe.OrbitRelativeFrameVariant.ROTATING
        ),
    )

    # Batch length mismatches raise
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_ecef_to_enz(sites[:2], targets)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_enz_to_ecef(sites[:2], covs, variant)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_enz_to_ecef(sites[0], np.eye(7), variant)
