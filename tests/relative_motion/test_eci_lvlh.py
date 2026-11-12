"""
Tests for LVLH (Local-Vertical Local-Horizontal) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_lvlh.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe


def _inclined_test_state():
    oe = np.array([brahe.R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _advanced_state(dt):
    n = brahe.mean_motion(brahe.R_EARTH + 700e3, brahe.AngleFormat.DEGREES)
    oe = np.array([brahe.R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0 + n * dt])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def test_rotation_lvlh_to_eci_axes_match_definition(eop):
    x = _inclined_test_state()
    r, v = x[:3], x[3:]
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)

    m = brahe.rotation_lvlh_to_eci(x)
    np.testing.assert_allclose(m[:, 2], -r_hat, atol=1e-15)
    np.testing.assert_allclose(m[:, 1], -h_hat, atol=1e-15)
    np.testing.assert_allclose(m[:, 0], np.cross(h_hat, r_hat), atol=1e-15)
    assert np.linalg.det(m) == approx(1.0, abs=1e-14)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)


def test_rotation_lvlh_is_signed_permutation_of_rtn(eop):
    x = _inclined_test_state()
    rtn = brahe.rotation_rtn_to_eci(x)
    lvlh = brahe.rotation_lvlh_to_eci(x)
    np.testing.assert_array_equal(lvlh[:, 0], rtn[:, 1])
    np.testing.assert_array_equal(lvlh[:, 1], -rtn[:, 2])
    np.testing.assert_array_equal(lvlh[:, 2], -rtn[:, 0])


def test_rotation_eci_to_lvlh_is_transpose(eop):
    x = _inclined_test_state()
    forward = brahe.rotation_lvlh_to_eci(x)
    inverse = brahe.rotation_eci_to_lvlh(x)
    np.testing.assert_array_equal(inverse, forward.T)
    np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-14)


def test_omega_lvlh_matches_rtn_rate_about_minus_y(eop):
    x = _inclined_test_state()
    f_dot = brahe.omega_rtn(x)[2]
    np.testing.assert_allclose(brahe.omega_lvlh(x), [0.0, -f_dot, 0.0], atol=1e-18)


def test_omega_lvlh_matches_finite_difference(eop):
    dt = 0.05
    x0 = _inclined_test_state()
    r0 = brahe.rotation_eci_to_lvlh(x0)
    r_dot = (
        brahe.rotation_eci_to_lvlh(_advanced_state(dt))
        - brahe.rotation_eci_to_lvlh(_advanced_state(-dt))
    ) / (2.0 * dt)
    m = -(r_dot @ r0.T)
    omega_fd = 0.5 * np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    np.testing.assert_allclose(omega_fd, brahe.omega_lvlh(x0), atol=1e-9)


def test_state_eci_to_lvlh_matches_permuted_rtn(eop):
    x_chief = _inclined_test_state()
    x_deputy = brahe.state_koe_to_eci(
        np.array([brahe.R_EARTH + 701e3, 0.1015, 97.85, 15.05, 30.05, 45.05]),
        brahe.AngleFormat.DEGREES,
    )
    rtn = brahe.state_eci_to_rtn(x_chief, x_deputy)
    lvlh = brahe.state_eci_to_lvlh(x_chief, x_deputy)
    expected = np.array([rtn[1], -rtn[2], -rtn[0], rtn[4], -rtn[5], -rtn[3]])
    np.testing.assert_allclose(lvlh, expected, atol=1e-9)


def test_state_lvlh_to_eci_round_trip(eop):
    x_chief = _inclined_test_state()
    x_rel = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])
    x_deputy = brahe.state_lvlh_to_eci(x_chief, x_rel)
    np.testing.assert_allclose(
        brahe.state_eci_to_lvlh(x_chief, x_deputy), x_rel, atol=1e-8
    )


def test_batch_lvlh_match_scalar(eop):
    chiefs = np.array(
        [
            brahe.state_koe_to_eci(
                np.array(
                    [brahe.R_EARTH + 700e3 + 1e3 * i, 0.01, 97.8, 15.0, 30.0, 45.0 + i]
                ),
                brahe.AngleFormat.DEGREES,
            )
            for i in range(3)
        ]
    )
    deputies = np.array(
        [
            brahe.state_koe_to_eci(
                np.array(
                    [
                        brahe.R_EARTH + 701e3 + 1e3 * i,
                        0.0115,
                        97.85,
                        15.05,
                        30.05,
                        45.05 + i,
                    ]
                ),
                brahe.AngleFormat.DEGREES,
            )
            for i in range(3)
        ]
    )

    rot = brahe.rotation_lvlh_to_eci(chiefs)
    rot_inv = brahe.rotation_eci_to_lvlh(chiefs)
    omegas = brahe.omega_lvlh(chiefs)
    rel = brahe.state_eci_to_lvlh(chiefs, deputies)
    rel_one_chief = brahe.state_eci_to_lvlh(chiefs[0], deputies)
    back = brahe.state_lvlh_to_eci(chiefs, rel)
    assert rot.shape == (3, 3, 3)
    assert omegas.shape == (3, 3)
    for i in range(3):
        np.testing.assert_array_equal(rot[i], brahe.rotation_lvlh_to_eci(chiefs[i]))
        np.testing.assert_array_equal(rot_inv[i], brahe.rotation_eci_to_lvlh(chiefs[i]))
        np.testing.assert_array_equal(omegas[i], brahe.omega_lvlh(chiefs[i]))
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_lvlh(chiefs[i], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_one_chief[i], brahe.state_eci_to_lvlh(chiefs[0], deputies[i])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_lvlh_to_eci(chiefs[i], rel[i])
        )

    # Column layout: components along axis 0
    np.testing.assert_array_equal(brahe.rotation_lvlh_to_eci(chiefs.T, axis=0), rot)
    np.testing.assert_array_equal(brahe.omega_lvlh(chiefs.T, axis=0), omegas.T)
    np.testing.assert_array_equal(
        brahe.state_eci_to_lvlh(chiefs.T, deputies.T, axis=0), rel.T
    )


def test_batch_lvlh_length_mismatch_raises(eop):
    chiefs = np.array([_inclined_test_state()] * 2)
    deputies = np.array([_inclined_test_state()] * 3)
    with pytest.raises(ValueError, match="Batch lengths"):
        brahe.state_eci_to_lvlh(chiefs, deputies)
