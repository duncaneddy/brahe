"""
Tests for PQW (perifocal) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_pqw.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe

SMA = brahe.R_EARTH + 700e3


def _state(e, i, raan, argp, m):
    oe = np.array([SMA, e, i, raan, argp, m])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _unit(v):
    return v / np.linalg.norm(v)


def test_rotation_pqw_p_axis_points_to_periapsis(eop):
    """Rust: test_rotation_pqw_p_axis_points_to_periapsis"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_peri = _state(0.1, 97.8, 15.0, 30.0, 0.0)
    p_expected = _unit(x_peri[:3])
    h_hat = _unit(np.cross(x[:3], x[3:]))

    m = brahe.rotation_pqw_to_eci(x)
    p, q, w = m[:, 0], m[:, 1], m[:, 2]
    np.testing.assert_allclose(p, p_expected, atol=1e-9)
    np.testing.assert_allclose(w, h_hat, atol=1e-15)
    np.testing.assert_allclose(q, np.cross(w, p), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_pqw_matches_classical_element_rotation(eop):
    """Rust: test_rotation_pqw_matches_classical_element_rotation"""
    i, raan, argp = 97.8, 15.0, 30.0
    x = _state(0.1, i, raan, argp, 45.0)
    ci, si = np.cos(np.radians(i)), np.sin(np.radians(i))
    cr, sr = np.cos(np.radians(raan)), np.sin(np.radians(raan))
    cw, sw = np.cos(np.radians(argp)), np.sin(np.radians(argp))
    p_expected = np.array(
        [
            cr * cw - sr * sw * ci,
            sr * cw + cr * sw * ci,
            sw * si,
        ]
    )
    p = brahe.rotation_pqw_to_eci(x)[:, 0]
    np.testing.assert_allclose(p, p_expected, atol=1e-9)


def test_rotation_pqw_circular_orbit_uses_node_line(eop):
    """Rust: test_rotation_pqw_circular_orbit_uses_node_line"""
    x = _state(0.0, 97.8, 15.0, 30.0, 45.0)
    p = brahe.rotation_pqw_to_eci(x)[:, 0]
    node = np.array([np.cos(np.radians(15.0)), np.sin(np.radians(15.0)), 0.0])
    np.testing.assert_allclose(p, node, atol=1e-9)


def test_rotation_pqw_circular_equatorial_orbit_uses_x_axis(eop):
    """Rust: test_rotation_pqw_circular_equatorial_orbit_uses_x_axis"""
    x = _state(0.0, 0.0, 15.0, 30.0, 45.0)
    m = brahe.rotation_pqw_to_eci(x)
    np.testing.assert_allclose(m[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_pqw_circular_equatorial_retrograde_is_right_handed(eop):
    """Rust: test_rotation_pqw_circular_equatorial_retrograde_is_right_handed"""
    x = _state(0.0, 180.0, 15.0, 30.0, 45.0)
    m = brahe.rotation_pqw_to_eci(x)
    np.testing.assert_allclose(m[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(m[:, 1], np.array([0.0, -1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(m[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-12)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)


def test_rotation_pqw_for_body_about_mars(eop):
    """Rust: test_rotation_pqw_for_body_about_mars"""
    oe = np.array([brahe.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
    x = brahe.state_koe_to_inertial_for_body(
        oe, brahe.CentralBody.Mars, brahe.AngleFormat.DEGREES
    )
    oe_peri = oe.copy()
    oe_peri[5] = 0.0
    x_peri = brahe.state_koe_to_inertial_for_body(
        oe_peri, brahe.CentralBody.Mars, brahe.AngleFormat.DEGREES
    )
    p = brahe.rotation_pqw_to_inertial_for_body(x, brahe.GM_MARS)[:, 0]
    np.testing.assert_allclose(p, _unit(x_peri[:3]), atol=1e-9)
    # With Earth's GM the eccentricity vector is wrong, so P is not the periapsis
    p_wrong = brahe.rotation_pqw_to_eci(x)[:, 0]
    assert np.linalg.norm(p_wrong - p) > 1e-3


def test_rotation_eci_to_pqw_is_transpose(eop):
    """Rust: test_rotation_eci_to_pqw_is_transpose"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    np.testing.assert_array_equal(
        brahe.rotation_eci_to_pqw(x), brahe.rotation_pqw_to_eci(x).T
    )
    np.testing.assert_array_equal(
        brahe.rotation_inertial_to_pqw_for_body(x, brahe.GM_MARS),
        brahe.rotation_pqw_to_inertial_for_body(x, brahe.GM_MARS).T,
    )


def test_earth_functions_equal_for_body_with_gm_earth(eop):
    """Rust: test_earth_functions_equal_for_body_with_gm_earth"""
    x_chief = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_deputy = x_chief + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])
    p = np.eye(6) * 4.0
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    np.testing.assert_array_equal(
        brahe.rotation_pqw_to_eci(x_chief),
        brahe.rotation_pqw_to_inertial_for_body(x_chief, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.jacobian_pqw_to_eci(x_chief),
        brahe.jacobian_pqw_to_inertial_for_body(x_chief, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.jacobian_eci_to_pqw(x_chief),
        brahe.jacobian_inertial_to_pqw_for_body(x_chief, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.covariance_pqw_to_eci(x_chief, p),
        brahe.covariance_pqw_to_inertial_for_body(x_chief, p, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.covariance_eci_to_pqw(x_chief, p),
        brahe.covariance_inertial_to_pqw_for_body(x_chief, p, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.state_eci_to_pqw(x_chief, x_deputy),
        brahe.state_inertial_to_pqw_for_body(x_chief, x_deputy, brahe.GM_EARTH),
    )
    np.testing.assert_array_equal(
        brahe.state_pqw_to_eci(x_chief, x_rel),
        brahe.state_pqw_to_inertial_for_body(x_chief, x_rel, brahe.GM_EARTH),
    )


def test_jacobian_pqw_is_block_diagonal_and_invertible(eop):
    """Rust: test_jacobian_pqw_is_block_diagonal_and_invertible"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    j = brahe.jacobian_pqw_to_eci(x)
    r = brahe.rotation_pqw_to_eci(x)
    zeros = np.zeros((3, 3))
    block = np.block([[r, zeros], [zeros, r]])
    np.testing.assert_allclose(j, block, atol=1e-15)
    np.testing.assert_allclose(brahe.jacobian_eci_to_pqw(x) @ j, np.eye(6), atol=1e-12)


def test_covariance_pqw_eci_round_trip(eop):
    """Rust: test_covariance_pqw_eci_round_trip"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    p = np.zeros((6, 6))
    for i in range(3):
        p[i, i] = 100.0
        p[3 + i, 3 + i] = 0.01
    p[0, 1] = 25.0
    p[1, 0] = 25.0
    p_eci = brahe.covariance_pqw_to_eci(x, p)
    p_back = brahe.covariance_eci_to_pqw(x, p_eci)
    assert np.linalg.norm(p_back - p) / np.linalg.norm(p) == approx(0.0, abs=1e-12)
    # Isotropic covariance is invariant under the pure rotation
    iso = np.eye(6) * 100.0
    assert np.linalg.norm(brahe.covariance_eci_to_pqw(x, iso) - iso) == approx(
        0.0, abs=1e-12
    )


def test_state_eci_to_pqw_is_pure_rotation_of_relative_state(eop):
    """Rust: test_state_eci_to_pqw_is_pure_rotation_of_relative_state"""
    x_chief = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_deputy = _state(0.1015, 97.85, 15.05, 30.05, 45.05)
    r = brahe.rotation_eci_to_pqw(x_chief)
    diff = x_deputy - x_chief
    expected_p = r @ diff[:3]
    expected_v = r @ diff[3:]
    rel = brahe.state_eci_to_pqw(x_chief, x_deputy)
    np.testing.assert_allclose(rel[:3], expected_p, atol=1e-9)
    np.testing.assert_allclose(rel[3:], expected_v, atol=1e-12)


def test_state_pqw_position_of_chief_is_in_plane(eop):
    """Rust: test_state_pqw_position_of_chief_is_in_plane"""
    x = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    r_pqw = brahe.rotation_eci_to_pqw(x) @ x[:3]
    assert r_pqw[2] == approx(0.0, abs=1e-6)
    assert np.linalg.norm(r_pqw) == approx(np.linalg.norm(x[:3]), abs=1e-9)
    assert r_pqw[1] > 0.0


def test_state_pqw_to_eci_round_trip(eop):
    """Rust: test_state_pqw_to_eci_round_trip"""
    x_chief = _state(0.1, 97.8, 15.0, 30.0, 45.0)
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_pqw_to_eci(x_chief, x_rel)
    np.testing.assert_allclose(
        brahe.state_eci_to_pqw(x_chief, x_deputy), x_rel, atol=1e-8
    )


def test_batch_pqw_match_scalar(eop):
    """Rust: test_batch_pqw_match_scalar"""
    chiefs = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0 + i) for i in range(3)])
    deputies = np.array(
        [c + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3]) for c in chiefs]
    )
    covs = np.array([np.eye(6) * (i + 1.0) for i in range(3)])
    gm = brahe.GM_MARS

    rot = brahe.rotation_pqw_to_eci(chiefs)
    rot_body = brahe.rotation_pqw_to_inertial_for_body(chiefs, gm)
    rot_inv = brahe.rotation_eci_to_pqw(chiefs)
    rot_inv_body = brahe.rotation_inertial_to_pqw_for_body(chiefs, gm)
    jac = brahe.jacobian_pqw_to_eci(chiefs)
    jac_body = brahe.jacobian_pqw_to_inertial_for_body(chiefs, gm)
    jac_inv = brahe.jacobian_eci_to_pqw(chiefs)
    jac_inv_body = brahe.jacobian_inertial_to_pqw_for_body(chiefs, gm)
    # covariance_pqw_to_eci broadcasts a single covariance across all chiefs
    cov = brahe.covariance_pqw_to_eci(chiefs, covs[0])
    cov_body = brahe.covariance_pqw_to_inertial_for_body(chiefs, covs, gm)
    cov_inv = brahe.covariance_eci_to_pqw(chiefs, covs)
    cov_inv_body = brahe.covariance_inertial_to_pqw_for_body(chiefs, covs, gm)
    # state_eci_to_pqw broadcasts a single chief across all deputies
    rel = brahe.state_eci_to_pqw(chiefs[0], deputies)
    rel_body = brahe.state_inertial_to_pqw_for_body(chiefs, deputies, gm)
    back = brahe.state_pqw_to_eci(chiefs, rel)
    back_body = brahe.state_pqw_to_inertial_for_body(chiefs, rel, gm)

    assert rot.shape == (3, 3, 3)
    assert jac.shape == (3, 6, 6)
    assert cov.shape == (3, 6, 6)

    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_pqw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(
            rot_body[i], brahe.rotation_pqw_to_inertial_for_body(chiefs[i], gm)
        )
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_eci_to_pqw(chiefs[i]))
        np.testing.assert_array_equal(
            rot_inv_body[i],
            brahe.rotation_inertial_to_pqw_for_body(chiefs[i], gm),
        )
        np.testing.assert_array_equal(jac[i], brahe.jacobian_pqw_to_eci(chiefs[i]))
        np.testing.assert_array_equal(
            jac_body[i], brahe.jacobian_pqw_to_inertial_for_body(chiefs[i], gm)
        )
        np.testing.assert_array_equal(jac_inv[i], brahe.jacobian_eci_to_pqw(chiefs[i]))
        np.testing.assert_array_equal(
            jac_inv_body[i],
            brahe.jacobian_inertial_to_pqw_for_body(chiefs[i], gm),
        )
        np.testing.assert_array_equal(
            cov[i], brahe.covariance_pqw_to_eci(chiefs[i], covs[0])
        )
        np.testing.assert_array_equal(
            cov_body[i],
            brahe.covariance_pqw_to_inertial_for_body(chiefs[i], covs[i], gm),
        )
        np.testing.assert_array_equal(
            cov_inv[i], brahe.covariance_eci_to_pqw(chiefs[i], covs[i])
        )
        np.testing.assert_array_equal(
            cov_inv_body[i],
            brahe.covariance_inertial_to_pqw_for_body(chiefs[i], covs[i], gm),
        )
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_pqw(chiefs[0], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_body[i],
            brahe.state_inertial_to_pqw_for_body(chiefs[i], deputies[i], gm),
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_pqw_to_eci(chiefs[i], rel[i])
        )
        np.testing.assert_array_equal(
            back_body[i], brahe.state_pqw_to_inertial_for_body(chiefs[i], rel[i], gm)
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_pqw_to_eci(chiefs.T, axis=0), rot)
    np.testing.assert_array_equal(
        brahe.state_eci_to_pqw(chiefs[0].T, deputies.T, axis=0), rel.T
    )
    np.testing.assert_array_equal(
        brahe.covariance_eci_to_pqw(chiefs.T, covs, axis=0), cov_inv
    )


def test_batch_pqw_length_mismatch_raises(eop):
    """Rust: mirrors the broadcast-rule error checks in test_batch_pqw_match_scalar"""
    chiefs = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0)] * 2)
    deputies = np.array([_state(0.1, 97.8, 15.0, 30.0, 45.0)] * 3)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_pqw(chiefs, deputies)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.covariance_pqw_to_eci(chiefs, np.array([np.eye(6)] * 3))
    with pytest.raises(ValueError, match="6x6"):
        brahe.covariance_pqw_to_eci(chiefs[0], np.eye(7))
