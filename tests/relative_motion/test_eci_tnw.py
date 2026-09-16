"""
Tests for TNW (Tangential, Normal, Cross-track) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_tnw.rs
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


def test_rotation_tnw_to_eci_axes_match_definition(eop):
    """Rust: test_rotation_tnw_to_eci_axes_match_definition"""
    x = _eccentric_state(0.0)
    r, v = x[:3], x[3:]
    v_hat = v / np.linalg.norm(v)
    h_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))

    m = brahe.rotation_tnw_to_eci(x)
    x_axis, y_axis, z_axis = m[:, 0], m[:, 1], m[:, 2]

    np.testing.assert_allclose(x_axis, v_hat, atol=1e-15)
    np.testing.assert_allclose(z_axis, h_hat, atol=1e-15)
    np.testing.assert_allclose(y_axis, np.cross(h_hat, v_hat), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)


def test_rotation_tnw_is_reordered_ntw(eop):
    """Rust: test_rotation_tnw_is_reordered_ntw"""
    x = _eccentric_state(0.0)
    ntw = brahe.rotation_ntw_to_eci(x)
    tnw = brahe.rotation_tnw_to_eci(x)
    # X_TNW = Y_NTW, Y_TNW = -X_NTW, Z_TNW = Z_NTW
    np.testing.assert_array_equal(tnw[:, 0], ntw[:, 1])
    np.testing.assert_array_equal(tnw[:, 1], -ntw[:, 0])
    np.testing.assert_array_equal(tnw[:, 2], ntw[:, 2])


def test_rotation_tnw_on_circular_orbit_is_permuted_rtn(eop):
    """Rust: test_rotation_tnw_on_circular_orbit_is_permuted_rtn"""
    x = _circular_state()
    rtn = brahe.rotation_rtn_to_eci(x)
    tnw = brahe.rotation_tnw_to_eci(x)
    # X = T, Y = -R (nadir), Z = N
    np.testing.assert_allclose(tnw[:, 0], rtn[:, 1], atol=1e-12)
    np.testing.assert_allclose(tnw[:, 1], -rtn[:, 0], atol=1e-12)
    np.testing.assert_allclose(tnw[:, 2], rtn[:, 2], atol=1e-12)


def test_rotation_eci_to_tnw_is_transpose(eop):
    """Rust: test_rotation_eci_to_tnw_is_transpose"""
    x = _eccentric_state(0.0)
    np.testing.assert_array_equal(
        brahe.rotation_eci_to_tnw(x), brahe.rotation_tnw_to_eci(x).T
    )


def test_omega_tnw_equals_ntw_rate_about_z(eop):
    """Rust: test_omega_tnw_equals_ntw_rate_about_z"""
    x = _eccentric_state(0.0)
    np.testing.assert_array_equal(brahe.omega_tnw(x), brahe.omega_ntw(x))
    np.testing.assert_array_equal(
        brahe.omega_tnw_for_body(x, brahe.GM_MARS),
        brahe.omega_ntw_for_body(x, brahe.GM_MARS),
    )
    assert brahe.omega_tnw(x)[0] == approx(0.0, abs=1e-18)
    assert brahe.omega_tnw(x)[1] == approx(0.0, abs=1e-18)


def test_omega_tnw_matches_finite_difference(eop):
    """Rust: test_omega_tnw_matches_finite_difference"""
    dt = 0.05
    x0 = _eccentric_state(0.0)
    omega_fd = _omega_fd(
        brahe.rotation_eci_to_tnw,
        _eccentric_state(-dt),
        x0,
        _eccentric_state(dt),
        dt,
    )
    np.testing.assert_allclose(omega_fd, brahe.omega_tnw(x0), atol=1e-9)
    assert np.linalg.norm(brahe.omega_tnw(x0) - brahe.omega_rtn(x0)) > 1e-6


def test_omega_tnw_for_body_matches_finite_difference_about_mars(eop):
    """Rust: test_omega_tnw_for_body_matches_finite_difference_about_mars"""
    dt = 0.05
    x0 = _mars_state(0.0)
    omega_fd = _omega_fd(
        brahe.rotation_eci_to_tnw,
        _mars_state(-dt),
        x0,
        _mars_state(dt),
        dt,
    )
    omega_body = brahe.omega_tnw_for_body(x0, brahe.GM_MARS)
    np.testing.assert_allclose(omega_fd, omega_body, atol=1e-9)
    assert np.linalg.norm(omega_body - brahe.omega_tnw(x0)) > 1e-6


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
            brahe.omega_tnw(x_chief), brahe.omega_tnw_for_body(x_chief, brahe.GM_EARTH)
        )
        np.testing.assert_array_equal(
            brahe.jacobian_tnw_to_eci(x_chief, variant),
            brahe.jacobian_tnw_to_inertial_for_body(x_chief, brahe.GM_EARTH, variant),
        )
        np.testing.assert_array_equal(
            brahe.jacobian_eci_to_tnw(x_chief, variant),
            brahe.jacobian_inertial_to_tnw_for_body(x_chief, brahe.GM_EARTH, variant),
        )
        np.testing.assert_array_equal(
            brahe.covariance_tnw_to_eci(x_chief, p, variant),
            brahe.covariance_tnw_to_inertial_for_body(
                x_chief, p, brahe.GM_EARTH, variant
            ),
        )
        np.testing.assert_array_equal(
            brahe.covariance_eci_to_tnw(x_chief, p, variant),
            brahe.covariance_inertial_to_tnw_for_body(
                x_chief, p, brahe.GM_EARTH, variant
            ),
        )
    np.testing.assert_array_equal(
        brahe.state_eci_to_tnw(x_chief, x_deputy),
        brahe.state_inertial_to_tnw_for_body(x_chief, x_deputy, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.state_tnw_to_eci(x_chief, x_rel),
        brahe.state_tnw_to_inertial_for_body(x_chief, x_rel, brahe.GM_EARTH),
    )


def test_jacobian_tnw_to_eci_inertial_is_block_diagonal(eop):
    """Rust: test_jacobian_tnw_to_eci_inertial_is_block_diagonal"""
    x = _eccentric_state(0.0)
    j = brahe.jacobian_tnw_to_eci(x, brahe.OrbitRelativeFrameVariant.INERTIAL)
    r = brahe.rotation_tnw_to_eci(x)
    zeros = np.zeros((3, 3))
    block = np.block([[r, zeros], [zeros, r]])
    np.testing.assert_allclose(j, block, atol=1e-15)


def test_jacobian_tnw_to_eci_rotating_coupling(eop):
    """Rust: test_jacobian_tnw_to_eci_rotating_coupling"""
    x = _eccentric_state(0.0)
    j = brahe.jacobian_tnw_to_eci(x, brahe.OrbitRelativeFrameVariant.ROTATING)
    expected = brahe.rotation_tnw_to_eci(x) @ _skew(brahe.omega_tnw(x))
    coupling = j[3:, :3]
    np.testing.assert_allclose(coupling, expected, atol=1e-18)
    assert np.linalg.norm(coupling) > 0.0


def test_jacobian_tnw_eci_inverse_identity(eop):
    """Rust: test_jacobian_tnw_eci_inverse_identity"""
    x = _eccentric_state(0.0)
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        forward = brahe.jacobian_tnw_to_eci(x, variant)
        inverse = brahe.jacobian_eci_to_tnw(x, variant)
        np.testing.assert_allclose(inverse @ forward, np.eye(6), atol=1e-12)


def test_covariance_tnw_eci_round_trip(eop):
    """Rust: test_covariance_tnw_eci_round_trip"""
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
        p_eci = brahe.covariance_tnw_to_eci(x, p, variant)
        p_back = brahe.covariance_eci_to_tnw(x, p_eci, variant)
        assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)
        np.testing.assert_allclose(p_eci, p_eci.T, atol=1e-18)


def test_state_eci_to_tnw_is_permuted_rtn_on_circular_orbit(eop):
    """Rust: test_state_eci_to_tnw_is_permuted_rtn_on_circular_orbit"""
    x_chief = _circular_state()
    x_deputy = brahe.state_koe_to_eci(
        np.array([brahe.R_EARTH + 701e3, 0.0005, 97.85, 15.05, 30.05, 45.05]),
        brahe.AngleFormat.DEGREES,
    )
    rtn = brahe.state_eci_to_rtn(x_chief, x_deputy)
    tnw = brahe.state_eci_to_tnw(x_chief, x_deputy)
    expected = np.array([rtn[1], -rtn[0], rtn[2], rtn[4], -rtn[3], rtn[5]])
    np.testing.assert_allclose(tnw, expected, atol=1e-6)


def test_state_tnw_to_eci_round_trip(eop):
    """Rust: test_state_tnw_to_eci_round_trip"""
    x_chief = _eccentric_state(0.0)
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_tnw_to_eci(x_chief, x_rel)
    np.testing.assert_allclose(
        brahe.state_eci_to_tnw(x_chief, x_deputy), x_rel, atol=1e-8
    )

    x_mars = _mars_state(0.0)
    x_deputy_mars = brahe.state_tnw_to_inertial_for_body(x_mars, x_rel, brahe.GM_MARS)
    np.testing.assert_allclose(
        brahe.state_inertial_to_tnw_for_body(x_mars, x_deputy_mars, brahe.GM_MARS),
        x_rel,
        atol=1e-8,
    )


def test_batch_tnw_match_scalar(eop):
    """Rust: test_batch_tnw_match_scalar"""
    chiefs = np.array([_eccentric_state(10.0 * i) for i in range(3)])
    deputies = np.array(
        [c + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for c in chiefs]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    gm = brahe.GM_MARS

    rot = brahe.rotation_tnw_to_eci(chiefs)
    rot_inv = brahe.rotation_eci_to_tnw(chiefs)
    omegas = brahe.omega_tnw(chiefs)
    omegas_body = brahe.omega_tnw_for_body(chiefs, gm)
    jac = brahe.jacobian_tnw_to_eci(chiefs, variant)
    jac_body = brahe.jacobian_tnw_to_inertial_for_body(chiefs, gm, variant)
    jac_inv = brahe.jacobian_eci_to_tnw(chiefs, variant)
    jac_inv_body = brahe.jacobian_inertial_to_tnw_for_body(chiefs, gm, variant)
    cov = brahe.covariance_tnw_to_eci(chiefs, covs, variant)
    cov_body = brahe.covariance_tnw_to_inertial_for_body(chiefs, covs[0], gm, variant)
    cov_inv = brahe.covariance_eci_to_tnw(chiefs[0], covs, variant)
    cov_inv_body = brahe.covariance_inertial_to_tnw_for_body(chiefs, covs, gm, variant)
    rel = brahe.state_eci_to_tnw(chiefs, deputies)
    rel_body = brahe.state_inertial_to_tnw_for_body(chiefs[0], deputies, gm)
    back = brahe.state_tnw_to_eci(chiefs, rel)
    back_body = brahe.state_tnw_to_inertial_for_body(chiefs, rel, gm)

    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    assert jac.shape == (3, 6, 6)
    assert cov.shape == (3, 6, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_tnw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_eci_to_tnw(chiefs[i]))
        np.testing.assert_array_equal(omegas[i], brahe.omega_tnw(chiefs[i]))
        np.testing.assert_array_equal(
            omegas_body[i], brahe.omega_tnw_for_body(chiefs[i], gm)
        )
        np.testing.assert_array_equal(
            jac[i], brahe.jacobian_tnw_to_eci(chiefs[i], variant)
        )
        np.testing.assert_array_equal(
            jac_body[i], brahe.jacobian_tnw_to_inertial_for_body(chiefs[i], gm, variant)
        )
        np.testing.assert_array_equal(
            jac_inv[i], brahe.jacobian_eci_to_tnw(chiefs[i], variant)
        )
        np.testing.assert_array_equal(
            jac_inv_body[i],
            brahe.jacobian_inertial_to_tnw_for_body(chiefs[i], gm, variant),
        )
        np.testing.assert_array_equal(
            cov[i], brahe.covariance_tnw_to_eci(chiefs[i], covs[i], variant)
        )
        np.testing.assert_array_equal(
            cov_body[i],
            brahe.covariance_tnw_to_inertial_for_body(chiefs[i], covs[0], gm, variant),
        )
        np.testing.assert_array_equal(
            cov_inv[i], brahe.covariance_eci_to_tnw(chiefs[0], covs[i], variant)
        )
        np.testing.assert_array_equal(
            cov_inv_body[i],
            brahe.covariance_inertial_to_tnw_for_body(chiefs[i], covs[i], gm, variant),
        )
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_tnw(chiefs[i], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_body[i],
            brahe.state_inertial_to_tnw_for_body(chiefs[0], deputies[i], gm),
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_tnw_to_eci(chiefs[i], rel[i])
        )
        np.testing.assert_array_equal(
            back_body[i], brahe.state_tnw_to_inertial_for_body(chiefs[i], rel[i], gm)
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_tnw_to_eci(chiefs.T, axis=0), rot)
    np.testing.assert_array_equal(brahe.omega_tnw(chiefs.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_eci_to_tnw(chiefs.T, deputies.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_tnw_to_eci(chiefs.T, covs, variant, axis=0), cov
    )


def test_batch_tnw_length_mismatch_raises(eop):
    """Rust: mirrors the broadcast-rule error checks in test_batch_tnw_match_scalar"""
    chiefs = np.array([_eccentric_state()] * 2)
    deputies = np.array([_eccentric_state()] * 3)
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_tnw(chiefs, deputies)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_tnw_to_eci(chiefs, np.array([np.eye(6)] * 3), variant)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_tnw_to_eci(chiefs[0], np.eye(7), variant)


def test_batch_tnw_covariance_preserves_state_batch_shape(eop):
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    states = np.array([_eccentric_state(dt=60.0 * i) for i in range(4)]).reshape(
        2, 2, 6
    )
    p = np.eye(6)
    p_out = brahe.covariance_tnw_to_eci(states, p, variant)
    assert p_out.shape == (2, 2, 6, 6)
    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(
                p_out[i, j], brahe.covariance_tnw_to_eci(states[i, j], p, variant)
            )

    x = _eccentric_state()
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    p_single_state = brahe.covariance_eci_to_tnw(x, covs, variant)
    assert p_single_state.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_single_state[k], brahe.covariance_eci_to_tnw(x, covs[k], variant)
        )

    batch = np.array([_eccentric_state(dt=60.0 * i) for i in range(3)])
    p_singleton_batch = brahe.covariance_tnw_to_eci(batch[:1], covs, variant)
    assert p_singleton_batch.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_singleton_batch[k],
            brahe.covariance_tnw_to_eci(batch[0], covs[k], variant),
        )


def test_batch_tnw_empty_covariance_batch(eop):
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    x = _eccentric_state()
    empty = brahe.covariance_tnw_to_eci(x, np.zeros((0, 6, 6)), variant)
    assert empty.shape == (0, 6, 6)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_tnw_to_eci(x, np.zeros((0, 5, 5)), variant)
