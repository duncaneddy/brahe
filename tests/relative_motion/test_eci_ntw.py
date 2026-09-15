"""
Tests for NTW (Normal, Tangential, Cross-track) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_ntw.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe


def _eccentric_state(dt=0.0):
    sma = brahe.R_EARTH + 700e3
    n = brahe.mean_motion(sma, brahe.AngleFormat.DEGREES)
    oe = np.array([sma, 0.1, 97.8, 15.0, 30.0, 45.0 + n * dt])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _circular_state():
    oe = np.array([brahe.R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _mars_state(dt=0.0):
    sma = brahe.R_MARS + 400e3
    n = brahe.mean_motion_general(sma, brahe.GM_MARS, brahe.AngleFormat.DEGREES)
    oe = np.array([sma, 0.05, 92.6, 45.0, 270.0, 10.0 + n * dt])
    return brahe.state_koe_to_inertial_for_body(
        oe, brahe.CentralBody.Mars, brahe.AngleFormat.DEGREES
    )


def _omega_fd(rotation, x_minus, x0, x_plus, dt):
    r_dot = (rotation(x_plus) - rotation(x_minus)) / (2.0 * dt)
    m = -(r_dot @ rotation(x0).T)
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def test_rotation_ntw_to_eci_axes_match_definition(eop):
    """Rust: test_rotation_ntw_to_eci_axes_match_definition"""
    x = _eccentric_state(0.0)
    r, v = x[:3], x[3:]
    v_hat = v / np.linalg.norm(v)
    h_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))

    m = brahe.rotation_ntw_to_eci(x)
    x_axis, y_axis, z_axis = m[:, 0], m[:, 1], m[:, 2]

    np.testing.assert_allclose(y_axis, v_hat, atol=1e-15)
    np.testing.assert_allclose(z_axis, h_hat, atol=1e-15)
    np.testing.assert_allclose(x_axis, np.cross(v_hat, h_hat), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)


def test_rotation_ntw_equals_rtn_on_circular_orbit(eop):
    """Rust: test_rotation_ntw_equals_rtn_on_circular_orbit"""
    x = _circular_state()
    np.testing.assert_allclose(
        brahe.rotation_ntw_to_eci(x), brahe.rotation_rtn_to_eci(x), atol=1e-12
    )


def test_rotation_ntw_differs_from_rtn_by_flight_path_angle(eop):
    """Rust: test_rotation_ntw_differs_from_rtn_by_flight_path_angle"""
    x = _eccentric_state(0.0)
    r, v = x[:3], x[3:]
    sin_gamma = np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v))
    cos_gamma = np.sqrt(1.0 - sin_gamma * sin_gamma)

    ntw = brahe.rotation_ntw_to_eci(x)
    rtn = brahe.rotation_rtn_to_eci(x)
    y_ntw = ntw[:, 1]
    r_axis, t_axis = rtn[:, 0], rtn[:, 1]
    assert np.dot(y_ntw, t_axis) == approx(cos_gamma, abs=1e-14)
    assert np.dot(y_ntw, r_axis) == approx(sin_gamma, abs=1e-14)
    assert abs(sin_gamma) > 1e-3


def test_rotation_eci_to_ntw_is_transpose(eop):
    """Rust: test_rotation_eci_to_ntw_is_transpose"""
    x = _eccentric_state(0.0)
    np.testing.assert_array_equal(
        brahe.rotation_eci_to_ntw(x), brahe.rotation_ntw_to_eci(x).T
    )


def test_omega_ntw_circular_orbit_matches_mean_motion(eop):
    """Rust: test_omega_ntw_circular_orbit_matches_mean_motion"""
    x = _circular_state()
    omega = brahe.omega_ntw(x)
    assert omega[0] == approx(0.0, abs=1e-18)
    assert omega[1] == approx(0.0, abs=1e-18)
    assert omega[2] == approx(
        brahe.mean_motion(brahe.R_EARTH + 700e3, brahe.AngleFormat.RADIANS), abs=1e-12
    )
    np.testing.assert_allclose(omega, brahe.omega_rtn(x), atol=1e-12)


def test_omega_ntw_matches_finite_difference(eop):
    """Rust: test_omega_ntw_matches_finite_difference"""
    dt = 0.05
    x0 = _eccentric_state(0.0)
    omega_fd = _omega_fd(
        brahe.rotation_eci_to_ntw,
        _eccentric_state(-dt),
        x0,
        _eccentric_state(dt),
        dt,
    )
    np.testing.assert_allclose(omega_fd, brahe.omega_ntw(x0), atol=1e-9)
    assert np.linalg.norm(brahe.omega_ntw(x0) - brahe.omega_rtn(x0)) > 1e-6


def test_omega_ntw_for_body_matches_finite_difference_about_mars(eop):
    """Rust: test_omega_ntw_for_body_matches_finite_difference_about_mars"""
    dt = 0.05
    x0 = _mars_state(0.0)
    omega_fd = _omega_fd(
        brahe.rotation_eci_to_ntw,
        _mars_state(-dt),
        x0,
        _mars_state(dt),
        dt,
    )
    omega_body = brahe.omega_ntw_for_body(x0, brahe.GM_MARS)
    np.testing.assert_allclose(omega_fd, omega_body, atol=1e-9)
    assert np.linalg.norm(omega_body - brahe.omega_ntw(x0)) > 1e-6


def test_earth_functions_equal_for_body_with_gm_earth(eop):
    """Rust: test_earth_functions_equal_for_body_with_gm_earth"""
    x_chief = _eccentric_state(0.0)
    x_deputy = _eccentric_state(0.0) + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])
    p = np.eye(6) * 4.0
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        np.testing.assert_array_equal(
            brahe.omega_ntw(x_chief), brahe.omega_ntw_for_body(x_chief, brahe.GM_EARTH)
        )
        np.testing.assert_array_equal(
            brahe.jacobian_ntw_to_eci(x_chief, variant),
            brahe.jacobian_ntw_to_inertial_for_body(x_chief, brahe.GM_EARTH, variant),
        )
        np.testing.assert_array_equal(
            brahe.jacobian_eci_to_ntw(x_chief, variant),
            brahe.jacobian_inertial_to_ntw_for_body(x_chief, brahe.GM_EARTH, variant),
        )
        np.testing.assert_array_equal(
            brahe.covariance_ntw_to_eci(x_chief, p, variant),
            brahe.covariance_ntw_to_inertial_for_body(
                x_chief, p, brahe.GM_EARTH, variant
            ),
        )
        np.testing.assert_array_equal(
            brahe.covariance_eci_to_ntw(x_chief, p, variant),
            brahe.covariance_inertial_to_ntw_for_body(
                x_chief, p, brahe.GM_EARTH, variant
            ),
        )
    np.testing.assert_array_equal(
        brahe.state_eci_to_ntw(x_chief, x_deputy),
        brahe.state_inertial_to_ntw_for_body(x_chief, x_deputy, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.state_ntw_to_eci(x_chief, x_rel),
        brahe.state_ntw_to_inertial_for_body(x_chief, x_rel, brahe.GM_EARTH),
    )


def test_jacobian_ntw_to_eci_inertial_is_block_diagonal(eop):
    """Rust: test_jacobian_ntw_to_eci_inertial_is_block_diagonal"""
    x = _eccentric_state(0.0)
    j = brahe.jacobian_ntw_to_eci(x, brahe.OrbitRelativeFrameVariant.INERTIAL)
    r = brahe.rotation_ntw_to_eci(x)
    np.testing.assert_allclose(j[:3, :3], r, atol=1e-15)
    np.testing.assert_allclose(j[3:, 3:], r, atol=1e-15)
    assert np.linalg.norm(j[3:, :3]) == approx(0.0, abs=1e-15)


def test_jacobian_ntw_to_eci_rotating_coupling(eop):
    """Rust: test_jacobian_ntw_to_eci_rotating_coupling"""
    x = _eccentric_state(0.0)
    j = brahe.jacobian_ntw_to_eci(x, brahe.OrbitRelativeFrameVariant.ROTATING)
    expected = brahe.rotation_ntw_to_eci(x) @ _skew(brahe.omega_ntw(x))
    coupling = j[3:, :3]
    np.testing.assert_allclose(coupling, expected, atol=1e-18)
    assert np.linalg.norm(coupling) > 0.0


def test_jacobian_ntw_eci_inverse_identity(eop):
    """Rust: test_jacobian_ntw_eci_inverse_identity"""
    x = _eccentric_state(0.0)
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        forward = brahe.jacobian_ntw_to_eci(x, variant)
        inverse = brahe.jacobian_eci_to_ntw(x, variant)
        np.testing.assert_allclose(inverse @ forward, np.eye(6), atol=1e-12)


def test_covariance_ntw_eci_round_trip(eop):
    """Rust: test_covariance_ntw_eci_round_trip"""
    x = _eccentric_state(0.0)
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
        p_eci = brahe.covariance_ntw_to_eci(x, p, variant)
        p_back = brahe.covariance_eci_to_ntw(x, p_eci, variant)
        assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)
        np.testing.assert_allclose(p_eci, p_eci.T, atol=1e-18)


def test_state_eci_to_ntw_equals_rtn_on_circular_orbit(eop):
    """Rust: test_state_eci_to_ntw_equals_rtn_on_circular_orbit"""
    x_chief = _circular_state()
    x_deputy = brahe.state_koe_to_eci(
        np.array([brahe.R_EARTH + 701e3, 0.0005, 97.85, 15.05, 30.05, 45.05]),
        brahe.AngleFormat.DEGREES,
    )
    np.testing.assert_allclose(
        brahe.state_eci_to_ntw(x_chief, x_deputy),
        brahe.state_eci_to_rtn(x_chief, x_deputy),
        atol=1e-6,
    )


def test_state_ntw_to_eci_round_trip(eop):
    """Rust: test_state_ntw_to_eci_round_trip"""
    x_chief = _eccentric_state(0.0)
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_ntw_to_eci(x_chief, x_rel)
    np.testing.assert_allclose(
        brahe.state_eci_to_ntw(x_chief, x_deputy), x_rel, atol=1e-8
    )

    x_mars = _mars_state(0.0)
    x_deputy_mars = brahe.state_ntw_to_inertial_for_body(x_mars, x_rel, brahe.GM_MARS)
    np.testing.assert_allclose(
        brahe.state_inertial_to_ntw_for_body(x_mars, x_deputy_mars, brahe.GM_MARS),
        x_rel,
        atol=1e-8,
    )


def test_batch_ntw_match_scalar(eop):
    """Rust: test_batch_ntw_match_scalar"""
    chiefs = np.array([_eccentric_state(10.0 * i) for i in range(3)])
    deputies = np.array(
        [c + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for c in chiefs]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    gm = brahe.GM_MARS

    rot = brahe.rotation_ntw_to_eci(chiefs)
    rot_inv = brahe.rotation_eci_to_ntw(chiefs)
    omegas = brahe.omega_ntw(chiefs)
    omegas_body = brahe.omega_ntw_for_body(chiefs, gm)
    jac = brahe.jacobian_ntw_to_eci(chiefs, variant)
    jac_body = brahe.jacobian_ntw_to_inertial_for_body(chiefs, gm, variant)
    jac_inv = brahe.jacobian_eci_to_ntw(chiefs, variant)
    jac_inv_body = brahe.jacobian_inertial_to_ntw_for_body(chiefs, gm, variant)
    cov = brahe.covariance_ntw_to_eci(chiefs, covs, variant)
    cov_body = brahe.covariance_ntw_to_inertial_for_body(chiefs, covs[0], gm, variant)
    cov_inv = brahe.covariance_eci_to_ntw(chiefs[0], covs, variant)
    cov_inv_body = brahe.covariance_inertial_to_ntw_for_body(chiefs, covs, gm, variant)
    rel = brahe.state_eci_to_ntw(chiefs, deputies)
    rel_body = brahe.state_inertial_to_ntw_for_body(chiefs[0], deputies, gm)
    back = brahe.state_ntw_to_eci(chiefs, rel)
    back_body = brahe.state_ntw_to_inertial_for_body(chiefs, rel, gm)

    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    assert jac.shape == (3, 6, 6)
    assert cov.shape == (3, 6, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_ntw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_eci_to_ntw(chiefs[i]))
        np.testing.assert_array_equal(omegas[i], brahe.omega_ntw(chiefs[i]))
        np.testing.assert_array_equal(
            omegas_body[i], brahe.omega_ntw_for_body(chiefs[i], gm)
        )
        np.testing.assert_array_equal(
            jac[i], brahe.jacobian_ntw_to_eci(chiefs[i], variant)
        )
        np.testing.assert_array_equal(
            jac_body[i], brahe.jacobian_ntw_to_inertial_for_body(chiefs[i], gm, variant)
        )
        np.testing.assert_array_equal(
            jac_inv[i], brahe.jacobian_eci_to_ntw(chiefs[i], variant)
        )
        np.testing.assert_array_equal(
            jac_inv_body[i],
            brahe.jacobian_inertial_to_ntw_for_body(chiefs[i], gm, variant),
        )
        np.testing.assert_array_equal(
            cov[i], brahe.covariance_ntw_to_eci(chiefs[i], covs[i], variant)
        )
        np.testing.assert_array_equal(
            cov_body[i],
            brahe.covariance_ntw_to_inertial_for_body(chiefs[i], covs[0], gm, variant),
        )
        np.testing.assert_array_equal(
            cov_inv[i], brahe.covariance_eci_to_ntw(chiefs[0], covs[i], variant)
        )
        np.testing.assert_array_equal(
            cov_inv_body[i],
            brahe.covariance_inertial_to_ntw_for_body(chiefs[i], covs[i], gm, variant),
        )
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_ntw(chiefs[i], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_body[i],
            brahe.state_inertial_to_ntw_for_body(chiefs[0], deputies[i], gm),
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_ntw_to_eci(chiefs[i], rel[i])
        )
        np.testing.assert_array_equal(
            back_body[i], brahe.state_ntw_to_inertial_for_body(chiefs[i], rel[i], gm)
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_ntw_to_eci(chiefs.T, axis=0), rot)
    np.testing.assert_array_equal(brahe.omega_ntw(chiefs.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_eci_to_ntw(chiefs.T, deputies.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_ntw_to_eci(chiefs.T, covs, variant, axis=0), cov
    )


def test_batch_ntw_length_mismatch_raises(eop):
    """Rust: mirrors the broadcast-rule error checks in test_batch_ntw_match_scalar"""
    chiefs = np.array([_eccentric_state()] * 2)
    deputies = np.array([_eccentric_state()] * 3)
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_ntw(chiefs, deputies)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_ntw_to_eci(chiefs, np.array([np.eye(6)] * 3), variant)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_ntw_to_eci(chiefs[0], np.eye(7), variant)
