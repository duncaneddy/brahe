"""
Tests for EQW (equinoctial) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_eqw.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe

SMA = brahe.R_EARTH + 700e3


def _state(e, i, raan, argp, m):
    oe = np.array([SMA, e, i, raan, argp, m])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def test_rotation_eqw_e_axis_is_ascending_node(eop):
    """Rust: test_rotation_eqw_e_axis_is_ascending_node"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    h_hat = np.cross(x[:3], x[3:])
    h_hat = h_hat / np.linalg.norm(h_hat)

    m = brahe.rotation_eqw_to_eci(x)
    e, q, w = m[:, 0], m[:, 1], m[:, 2]
    node = np.array([np.cos(np.radians(15.0)), np.sin(np.radians(15.0)), 0.0])
    np.testing.assert_allclose(e, node, atol=1e-12)
    assert e[2] == approx(0.0, abs=1e-15)
    np.testing.assert_allclose(w, h_hat, atol=1e-15)
    np.testing.assert_allclose(q, np.cross(w, e), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_eqw_retrograde_orbit_node_follows_momentum(eop):
    """Rust: test_rotation_eqw_retrograde_orbit_node_follows_momentum"""
    # For i > 90 deg the node computed from z_hat x h_hat is still the ascending node
    x = _state(0.05, 120.0, 200.0, 10.0, 80.0)
    e = brahe.rotation_eqw_to_eci(x)[:, 0]
    node = np.array([np.cos(np.radians(200.0)), np.sin(np.radians(200.0)), 0.0])
    np.testing.assert_allclose(e, node, atol=1e-12)


def test_rotation_eqw_equatorial_orbit_uses_x_axis(eop):
    """Rust: test_rotation_eqw_equatorial_orbit_uses_x_axis"""
    x = _state(0.1, 0.0, 15.0, 30.0, 45.0)
    m = brahe.rotation_eqw_to_eci(x)
    e = m[:, 0]
    np.testing.assert_allclose(e, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_eqw_equatorial_retrograde_is_right_handed(eop):
    """Rust: test_rotation_eqw_equatorial_retrograde_is_right_handed"""
    x = _state(0.1, 180.0, 15.0, 30.0, 45.0)
    m = brahe.rotation_eqw_to_eci(x)
    np.testing.assert_allclose(m[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(m[:, 1], np.array([0.0, -1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(m[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-12)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_eqw_near_equatorial_fallback_is_orthonormal(eop):
    """Rust: test_rotation_eqw_near_equatorial_fallback_is_orthonormal"""
    x = _state(0.1, 3e-8, 15.0, 30.0, 45.0)
    m = brahe.rotation_eqw_to_eci(x)
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-15)
    e = m[:, 0]
    np.testing.assert_allclose(e, np.array([1.0, 0.0, 0.0]), atol=1e-9)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_eci_to_eqw_is_transpose(eop):
    """Rust: test_rotation_eci_to_eqw_is_transpose"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    np.testing.assert_array_equal(
        brahe.rotation_eci_to_eqw(x), brahe.rotation_eqw_to_eci(x).T
    )


def test_jacobian_eqw_is_block_diagonal_and_invertible(eop):
    """Rust: test_jacobian_eqw_is_block_diagonal_and_invertible"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    j = brahe.jacobian_eqw_to_eci(x)
    r = brahe.rotation_eqw_to_eci(x)
    zeros = np.zeros((3, 3))
    block = np.block([[r, zeros], [zeros, r]])
    np.testing.assert_allclose(j, block, atol=1e-15)
    np.testing.assert_allclose(brahe.jacobian_eci_to_eqw(x) @ j, np.eye(6), atol=1e-12)


def test_covariance_eqw_eci_round_trip(eop):
    """Rust: test_covariance_eqw_eci_round_trip"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    p = np.zeros((6, 6))
    for i in range(3):
        p[i, i] = 100.0
        p[3 + i, 3 + i] = 0.01
    p[0, 1] = 25.0
    p[1, 0] = 25.0
    p_eci = brahe.covariance_eqw_to_eci(x, p)
    p_back = brahe.covariance_eci_to_eqw(x, p_eci)
    assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)


def test_state_eci_to_eqw_is_pure_rotation_of_relative_state(eop):
    """Rust: test_state_eci_to_eqw_is_pure_rotation_of_relative_state"""
    x_chief = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_deputy = _state(0.1015, 97.85, 15.05, 30.05, 45.05)
    r = brahe.rotation_eci_to_eqw(x_chief)
    diff = x_deputy - x_chief
    expected_p = r @ diff[:3]
    expected_v = r @ diff[3:]
    rel = brahe.state_eci_to_eqw(x_chief, x_deputy)
    np.testing.assert_allclose(rel[:3], expected_p, atol=1e-9)
    np.testing.assert_allclose(rel[3:], expected_v, atol=1e-12)


def test_state_eqw_position_of_chief_is_in_plane(eop):
    """Rust: test_state_eqw_position_of_chief_is_in_plane"""
    # The chief's own position in EQW is r [cos u, sin u, 0] with u the argument of latitude
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    r_eqw = brahe.rotation_eci_to_eqw(x) @ x[:3]
    assert r_eqw[2] == approx(0.0, abs=1e-6)
    assert np.linalg.norm(r_eqw) == approx(np.linalg.norm(x[:3]), abs=1e-9)


def test_state_eqw_to_eci_round_trip(eop):
    """Rust: test_state_eqw_to_eci_round_trip"""
    x_chief = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_eqw_to_eci(x_chief, x_rel)
    np.testing.assert_allclose(
        brahe.state_eci_to_eqw(x_chief, x_deputy), x_rel, atol=1e-8
    )


def test_batch_eqw_match_scalar(eop):
    """Rust: test_batch_eqw_match_scalar"""
    chiefs = np.array([_state(0.1, 97.8, 15.0 + i, 30.0, 45.0) for i in range(3)])
    deputies = np.array(
        [c + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for c in chiefs]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])

    rot = brahe.rotation_eqw_to_eci(chiefs)
    rot_inv = brahe.rotation_eci_to_eqw(chiefs)
    jac = brahe.jacobian_eqw_to_eci(chiefs)
    jac_inv = brahe.jacobian_eci_to_eqw(chiefs)
    # covariance_eqw_to_eci broadcasts a single covariance across all chiefs
    cov = brahe.covariance_eqw_to_eci(chiefs, covs[0])
    cov_inv = brahe.covariance_eci_to_eqw(chiefs, covs)
    # state_eci_to_eqw broadcasts a single chief across all deputies
    rel = brahe.state_eci_to_eqw(chiefs[0], deputies)
    back = brahe.state_eqw_to_eci(chiefs, rel)

    assert rot.shape == (3, 3, 3)
    assert jac.shape == (3, 6, 6)
    assert cov.shape == (3, 6, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_eqw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_eci_to_eqw(chiefs[i]))
        np.testing.assert_array_equal(jac[i], brahe.jacobian_eqw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(jac_inv[i], brahe.jacobian_eci_to_eqw(chiefs[i]))
        np.testing.assert_array_equal(
            cov[i], brahe.covariance_eqw_to_eci(chiefs[i], covs[0])
        )
        np.testing.assert_array_equal(
            cov_inv[i], brahe.covariance_eci_to_eqw(chiefs[i], covs[i])
        )
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_eqw(chiefs[0], deputies[i])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_eqw_to_eci(chiefs[i], rel[i])
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_eqw_to_eci(chiefs.T, axis=0), rot)
    np.testing.assert_array_equal(
        brahe.state_eci_to_eqw(chiefs[0].T, deputies.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_eci_to_eqw(chiefs.T, covs, axis=0), cov_inv
    )

    # Length mismatch raises
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_eqw(chiefs[:2], deputies)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_eqw_to_eci(chiefs[:2], covs)


def test_batch_eqw_covariance_preserves_state_batch_shape(eop):
    states = np.array(
        [_state(0.1, 97.8, 15.0, 30.0, 45.0 + i) for i in range(4)]
    ).reshape(2, 2, 6)
    p = np.eye(6)
    p_out = brahe.covariance_eqw_to_eci(states, p)
    assert p_out.shape == (2, 2, 6, 6)
    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(
                p_out[i, j], brahe.covariance_eqw_to_eci(states[i, j], p)
            )

    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    p_single_state = brahe.covariance_eci_to_eqw(x, covs)
    assert p_single_state.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_single_state[k], brahe.covariance_eci_to_eqw(x, covs[k])
        )

    batch = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0 + i) for i in range(3)])
    p_singleton_batch = brahe.covariance_eqw_to_eci(batch[:1], covs)
    assert p_singleton_batch.shape == (3, 6, 6)
    for k in range(3):
        np.testing.assert_array_equal(
            p_singleton_batch[k], brahe.covariance_eqw_to_eci(batch[0], covs[k])
        )


def test_batch_eqw_empty_covariance_batch(eop):
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    empty = brahe.covariance_eqw_to_eci(x, np.zeros((0, 6, 6)))
    assert empty.shape == (0, 6, 6)
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_eqw_to_eci(x, np.zeros((0, 5, 5)))


def test_batch_eqw_length_mismatch_raises(eop):
    batch_two = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0)] * 2)
    batch_three = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0)] * 3)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_eqw(batch_two, batch_three)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_eqw_to_eci(batch_two, np.array([np.eye(6)] * 3))
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_eqw_to_eci(batch_two[0], np.eye(7))
