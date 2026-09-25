"""
Tests for NSW (Nadir, Sun, Normal) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_nsw.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe

SMA = brahe.R_EARTH + 700e3


def _sc_state(dt=0.0):
    n = brahe.mean_motion(SMA, brahe.AngleFormat.DEGREES)
    oe = np.array([SMA, 0.05, 97.8, 15.0, 30.0, 45.0 + n * dt])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _sun_state(dt=0.0):
    r = np.array([0.9 * brahe.AU, 0.4 * brahe.AU, 0.17 * brahe.AU])
    v = np.array([-12.0e3, 26.0e3, 11.0e3])
    r_t = r + v * dt
    return np.array([r_t[0], r_t[1], r_t[2], v[0], v[1], v[2]])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _omega_fd(x_minus, s_minus, x0, s0, x_plus, s_plus, dt):
    r_dot = (
        brahe.rotation_eci_to_nsw(x_plus, s_plus)
        - brahe.rotation_eci_to_nsw(x_minus, s_minus)
    ) / (2.0 * dt)
    m = -(r_dot @ brahe.rotation_eci_to_nsw(x0, s0).T)
    return 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])


def test_rotation_nsw_to_eci_axes_match_definition(eop):
    """Rust: test_rotation_nsw_to_eci_axes_match_definition"""
    x = _sc_state(0.0)
    s = _sun_state(0.0)
    r = x[:3]
    sun_dir = (s[:3] - r) / np.linalg.norm(s[:3] - r)

    m = brahe.rotation_nsw_to_eci(x, s)
    x_axis, y_axis, z_axis = m[:, 0], m[:, 1], m[:, 2]

    np.testing.assert_allclose(x_axis, -r / np.linalg.norm(r), atol=1e-15)
    assert x_axis.dot(y_axis) == approx(0.0, abs=1e-15)
    assert y_axis.dot(sun_dir) > 0.0
    # Y is the Sun direction with its nadir component removed
    projected = sun_dir - sun_dir.dot(x_axis) * x_axis
    np.testing.assert_allclose(
        y_axis, projected / np.linalg.norm(projected), atol=1e-15
    )
    np.testing.assert_allclose(z_axis, np.cross(x_axis, y_axis), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_nsw_is_invariant_to_sun_offset_along_nadir(eop):
    """Rust: test_rotation_nsw_is_invariant_to_sun_offset_along_nadir"""
    # X lies along the position vector, so passing the Sun state relative to
    # the spacecraft instead of relative to the center changes neither the
    # axes nor the rate.
    x = _sc_state(0.0)
    s_center = _sun_state(0.0)
    s_shifted = s_center - x

    np.testing.assert_allclose(
        brahe.rotation_nsw_to_eci(x, s_center),
        brahe.rotation_nsw_to_eci(x, s_shifted),
        atol=1e-12,
    )
    omega_center = brahe.omega_nsw(x, s_center)
    omega_shifted = brahe.omega_nsw(x, s_shifted)
    assert np.linalg.norm(omega_center - omega_shifted) / np.linalg.norm(
        omega_center
    ) == approx(0.0, abs=1e-12)


def test_rotation_nsw_sun_along_nadir_falls_back_to_along_track(eop):
    """Rust: test_rotation_nsw_sun_along_nadir_falls_back_to_along_track"""
    x = _sc_state(0.0)
    r = x[:3]
    r_sun = r - r / np.linalg.norm(r) * brahe.AU
    s = np.array([r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0])
    m = brahe.rotation_nsw_to_eci(x, s)
    y = m[:, 1]
    t_axis = brahe.rotation_rtn_to_eci(x)[:, 1]
    np.testing.assert_allclose(y, t_axis, atol=1e-12)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_eci_to_nsw_is_transpose(eop):
    """Rust: test_rotation_eci_to_nsw_is_transpose"""
    x = _sc_state(0.0)
    s = _sun_state(0.0)
    np.testing.assert_array_equal(
        brahe.rotation_eci_to_nsw(x, s), brahe.rotation_nsw_to_eci(x, s).T
    )


def test_omega_nsw_matches_finite_difference_with_moving_sun(eop):
    """Rust: test_omega_nsw_matches_finite_difference_with_moving_sun"""
    dt = 0.05
    x0 = _sc_state(0.0)
    s0 = _sun_state(0.0)
    omega_fd = _omega_fd(
        _sc_state(-dt), _sun_state(-dt), x0, s0, _sc_state(dt), _sun_state(dt), dt
    )
    omega = brahe.omega_nsw(x0, s0)
    np.testing.assert_allclose(omega_fd, omega, atol=1e-9)
    # All three components are generally nonzero
    assert abs(omega[0]) > 1e-8 and abs(omega[1]) > 1e-8 and abs(omega[2]) > 1e-8


def test_omega_nsw_fixed_sun_matches_finite_difference(eop):
    """Rust: test_omega_nsw_fixed_sun_matches_finite_difference"""
    dt = 0.05
    x0 = _sc_state(0.0)
    s = _sun_state(0.0)
    s_fixed = np.array([s[0], s[1], s[2], 0.0, 0.0, 0.0])
    omega_fd = _omega_fd(
        _sc_state(-dt), s_fixed, x0, s_fixed, _sc_state(dt), s_fixed, dt
    )
    np.testing.assert_allclose(omega_fd, brahe.omega_nsw(x0, s_fixed), atol=1e-9)
    assert np.linalg.norm(brahe.omega_nsw(x0, s_fixed) - brahe.omega_nsw(x0, s)) > 1e-9


def test_omega_nsw_degenerate_branch_matches_finite_difference(eop):
    """Rust: test_omega_nsw_degenerate_branch_matches_finite_difference"""
    # Keep the Sun exactly along nadir at every epoch so the fallback branch
    # is exercised; its rate is then the along-track frame's rate.
    dt = 0.05

    def sun_on_nadir(x):
        r = x[:3]
        r_sun = r - r / np.linalg.norm(r) * brahe.AU
        return np.array([r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0])

    x0 = _sc_state(0.0)
    x_minus = _sc_state(-dt)
    x_plus = _sc_state(dt)
    omega_fd = _omega_fd(
        x_minus,
        sun_on_nadir(x_minus),
        x0,
        sun_on_nadir(x0),
        x_plus,
        sun_on_nadir(x_plus),
        dt,
    )
    np.testing.assert_allclose(
        omega_fd, brahe.omega_nsw(x0, sun_on_nadir(x0)), atol=1e-9
    )


def test_jacobian_nsw_to_eci_inertial_is_block_diagonal(eop):
    """Rust: test_jacobian_nsw_to_eci_inertial_is_block_diagonal"""
    x, s = _sc_state(0.0), _sun_state(0.0)
    j = brahe.jacobian_nsw_to_eci(x, s, brahe.OrbitRelativeFrameVariant.INERTIAL)
    r = brahe.rotation_nsw_to_eci(x, s)
    zeros = np.zeros((3, 3))
    block = np.block([[r, zeros], [zeros, r]])
    np.testing.assert_allclose(j, block, atol=1e-15)


def test_jacobian_nsw_to_eci_rotating_coupling(eop):
    """Rust: test_jacobian_nsw_to_eci_rotating_coupling"""
    x, s = _sc_state(0.0), _sun_state(0.0)
    j = brahe.jacobian_nsw_to_eci(x, s, brahe.OrbitRelativeFrameVariant.ROTATING)
    expected = brahe.rotation_nsw_to_eci(x, s) @ _skew(brahe.omega_nsw(x, s))
    coupling = j[3:, :3]
    np.testing.assert_allclose(coupling, expected, atol=1e-18)


def test_jacobian_nsw_eci_inverse_identity(eop):
    """Rust: test_jacobian_nsw_eci_inverse_identity"""
    x, s = _sc_state(0.0), _sun_state(0.0)
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        forward = brahe.jacobian_nsw_to_eci(x, s, variant)
        inverse = brahe.jacobian_eci_to_nsw(x, s, variant)
        np.testing.assert_allclose(inverse @ forward, np.eye(6), atol=1e-12)


def test_covariance_nsw_eci_round_trip(eop):
    """Rust: test_covariance_nsw_eci_round_trip"""
    x, s = _sc_state(0.0), _sun_state(0.0)
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
        p_eci = brahe.covariance_nsw_to_eci(x, s, p, variant)
        p_back = brahe.covariance_eci_to_nsw(x, s, p_eci, variant)
        assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)


def test_state_nsw_to_eci_round_trip(eop):
    """Rust: test_state_nsw_to_eci_round_trip"""
    x_chief, s = _sc_state(0.0), _sun_state(0.0)
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_nsw_to_eci(x_chief, x_rel, s)
    np.testing.assert_allclose(
        brahe.state_eci_to_nsw(x_chief, x_deputy, s), x_rel, atol=1e-8
    )


def test_state_eci_to_nsw_applies_transport_term(eop):
    """Rust: test_state_eci_to_nsw_applies_transport_term"""
    x_chief, s = _sc_state(0.0), _sun_state(0.0)
    x_deputy = x_chief + np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = brahe.rotation_eci_to_nsw(x_chief, s)
    rho = r @ np.array([1000.0, 0.0, 0.0])
    expected_v = -np.cross(brahe.omega_nsw(x_chief, s), rho)
    rel = brahe.state_eci_to_nsw(x_chief, x_deputy, s)
    np.testing.assert_allclose(rel[3:], expected_v, atol=1e-12)


def test_batch_nsw_match_scalar(eop):
    """Rust: test_batch_nsw_match_scalar"""
    chiefs = np.array([_sc_state(10.0 * i) for i in range(3)])
    suns = np.array([_sun_state(10.0 * i) for i in range(3)])
    deputies = np.array(
        [c + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for c in chiefs]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])

    rot = brahe.rotation_nsw_to_eci(chiefs, suns)
    rot_inv = brahe.rotation_eci_to_nsw(chiefs, suns)
    omegas = brahe.omega_nsw(chiefs, suns)
    rot_bsun = brahe.rotation_nsw_to_eci(chiefs, suns[:1])
    omegas_bsun = brahe.omega_nsw(chiefs, suns[:1])
    rel = brahe.state_eci_to_nsw(chiefs, deputies, suns)
    back = brahe.state_nsw_to_eci(chiefs, rel, suns)

    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    assert rel.shape == (3, 6)

    for i in range(3):
        np.testing.assert_array_equal(
            rot[i], brahe.rotation_nsw_to_eci(chiefs[i], suns[i])
        )
        np.testing.assert_array_equal(
            rot_inv[i], brahe.rotation_eci_to_nsw(chiefs[i], suns[i])
        )
        np.testing.assert_array_equal(omegas[i], brahe.omega_nsw(chiefs[i], suns[i]))
        np.testing.assert_array_equal(
            rot_bsun[i], brahe.rotation_nsw_to_eci(chiefs[i], suns[0])
        )
        np.testing.assert_array_equal(
            omegas_bsun[i], brahe.omega_nsw(chiefs[i], suns[0])
        )
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_nsw(chiefs[i], deputies[i], suns[i])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_nsw_to_eci(chiefs[i], rel[i], suns[i])
        )

    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        jac = brahe.jacobian_nsw_to_eci(chiefs, suns, variant)
        jac_inv = brahe.jacobian_eci_to_nsw(chiefs, suns, variant)
        cov = brahe.covariance_nsw_to_eci(chiefs, suns, covs, variant)
        cov_inv = brahe.covariance_eci_to_nsw(chiefs, suns, covs, variant)
        cov_bcov = brahe.covariance_nsw_to_eci(chiefs, suns, covs[:1], variant)

        assert jac.shape == (3, 6, 6)
        assert cov.shape == (3, 6, 6)

        for i in range(3):
            np.testing.assert_array_equal(
                jac[i], brahe.jacobian_nsw_to_eci(chiefs[i], suns[i], variant)
            )
            np.testing.assert_array_equal(
                jac_inv[i], brahe.jacobian_eci_to_nsw(chiefs[i], suns[i], variant)
            )
            np.testing.assert_array_equal(
                cov[i],
                brahe.covariance_nsw_to_eci(chiefs[i], suns[i], covs[i], variant),
            )
            np.testing.assert_array_equal(
                cov_inv[i],
                brahe.covariance_eci_to_nsw(chiefs[i], suns[i], covs[i], variant),
            )
            np.testing.assert_array_equal(
                cov_bcov[i],
                brahe.covariance_nsw_to_eci(chiefs[i], suns[i], covs[0], variant),
            )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(
        brahe.rotation_nsw_to_eci(chiefs.T, suns.T, axis=0), rot
    )
    np.testing.assert_array_equal(brahe.omega_nsw(chiefs.T, suns.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_eci_to_nsw(chiefs.T, deputies.T, suns.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_nsw_to_eci(
            chiefs.T, suns.T, covs, brahe.OrbitRelativeFrameVariant.ROTATING, axis=0
        ),
        cov,
    )

    x_nd = np.stack([chiefs, chiefs])
    s_nd = np.stack([suns, suns])
    cov_nd = brahe.covariance_nsw_to_eci(
        x_nd, s_nd, covs[0], brahe.OrbitRelativeFrameVariant.ROTATING
    )
    assert cov_nd.shape == (2, 3, 6, 6)
    np.testing.assert_array_equal(
        cov_nd[1],
        brahe.covariance_nsw_to_eci(
            chiefs, suns, covs[0], brahe.OrbitRelativeFrameVariant.ROTATING
        ),
    )


def test_batch_nsw_length_mismatch_raises(eop):
    """Rust: mirrors the broadcast-rule error checks in test_batch_nsw_match_scalar"""
    chiefs = np.array([_sc_state(10.0 * i) for i in range(3)])
    suns = np.array([_sun_state(10.0 * i) for i in range(3)])
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.rotation_nsw_to_eci(chiefs[:2], suns)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_nsw_to_eci(chiefs[:2], suns, covs, variant)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_nsw_to_eci(chiefs[0], suns[0], np.eye(7), variant)


def test_batch_nsw_covariance_preserves_state_batch_shape(eop):
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    s = _sun_state()
    states = np.array([_sc_state(10.0 * i) for i in range(4)]).reshape(2, 2, 6)
    p = np.eye(6)
    p_out = brahe.covariance_nsw_to_eci(states, s, p, variant)
    assert p_out.shape == (2, 2, 6, 6)
    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(
                p_out[i, j], brahe.covariance_nsw_to_eci(states[i, j], s, p, variant)
            )

    x = _sc_state()
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    p_single_state = brahe.covariance_eci_to_nsw(x, s, covs, variant)
    assert p_single_state.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_single_state[k], brahe.covariance_eci_to_nsw(x, s, covs[k], variant)
        )

    batch = np.array([_sc_state(10.0 * i) for i in range(3)])
    p_singleton_batch = brahe.covariance_nsw_to_eci(batch[:1], s, covs, variant)
    assert p_singleton_batch.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_singleton_batch[k],
            brahe.covariance_nsw_to_eci(batch[0], s, covs[k], variant),
        )


def test_batch_nsw_empty_covariance_batch(eop):
    variant = brahe.OrbitRelativeFrameVariant.ROTATING
    s = _sun_state()
    x = _sc_state()
    empty = brahe.covariance_nsw_to_eci(x, s, np.zeros((0, 6, 6)), variant)
    assert empty.shape == (0, 6, 6)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_nsw_to_eci(x, s, np.zeros((0, 5, 5)), variant)
