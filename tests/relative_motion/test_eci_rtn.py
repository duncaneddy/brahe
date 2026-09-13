"""
Tests for RTN (Radial-Tangential-Normal) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_rtn.rs
"""

import numpy as np
import pytest
from pytest import approx

import brahe


def get_test_state():
    """Helper function to generate a test orbital state."""
    sma = brahe.R_EARTH + 700e3  # Semi-major axis in meters
    return np.array([sma, 0.0, 0.0, 0.0, brahe.perigee_velocity(sma, 0.0), 0.0])


def test_rotation_rtn_to_eci(eop):
    """
    Test that the RTN-to-ECI rotation matrix correctly transforms vectors.

    Mirrors test_rotation_rtn_to_eci in Rust.
    """
    x_eci = get_test_state()
    p_eci = x_eci[:3]

    # Get RTN rotation matrix
    r_rtn = brahe.rotation_rtn_to_eci(x_eci)

    # Confirm that multiplying by the radial unit vector yields the position vector
    r_eci = r_rtn @ np.array([1.0, 0.0, 0.0]) * np.linalg.norm(p_eci)

    # The transformed vector should match the original position vector
    assert np.linalg.norm(r_eci - p_eci) == approx(0.0, abs=1e-6)


def test_rotation_eci_to_rtn_inverse(eop):
    """
    Test that ECI-to-RTN and RTN-to-ECI rotation matrices are inverses.

    Mirrors test_rotation_eci_to_rtn_inverse in Rust.
    """
    x_eci = get_test_state()

    # Get both rotation matrices
    r_rtn_to_eci = brahe.rotation_rtn_to_eci(x_eci)
    r_eci_to_rtn = brahe.rotation_eci_to_rtn(x_eci)

    # Confirm that the product of the two rotation matrices is the identity matrix
    identity = r_rtn_to_eci @ r_eci_to_rtn
    expected_identity = np.eye(3)

    assert np.linalg.norm(identity - expected_identity) == approx(0.0, abs=1e-10)


def test_rotation_rtn_to_eci_properties(eop):
    """
    Test additional properties of the RTN frame transformation.
    """
    x_eci = get_test_state()
    r_rtn = brahe.rotation_rtn_to_eci(x_eci)

    # The rotation matrix should be orthogonal (det = ±1, R^T R = I)
    det = np.linalg.det(r_rtn)
    assert det == approx(1.0, abs=1e-10) or det == approx(-1.0, abs=1e-10)

    # For a proper rotation matrix, det should be +1
    assert det == approx(1.0, abs=1e-10)

    # R^T R should equal identity
    rtranspose_r = r_rtn.T @ r_rtn
    assert np.linalg.norm(rtranspose_r - np.eye(3)) == approx(0.0, abs=1e-10)


def test_rotation_eci_to_rtn_is_transpose(eop):
    """
    Test that ECI-to-RTN is the transpose of RTN-to-ECI.
    """
    x_eci = get_test_state()

    r_rtn_to_eci = brahe.rotation_rtn_to_eci(x_eci)
    r_eci_to_rtn = brahe.rotation_eci_to_rtn(x_eci)

    # ECI-to-RTN should be the transpose of RTN-to-ECI
    assert np.linalg.norm(r_eci_to_rtn - r_rtn_to_eci.T) == approx(0.0, abs=1e-15)


def test_state_eci_to_rtn(eop):
    """
    Test transformation of absolute chief/deputy ECI states to relative RTN state.

    Verifies that:
    - The radial component is positive when deputy is farther from Earth
    - The transformation produces expected relative positions
    """
    x_chief = get_test_state()
    # Deputy offset by 100m, 200m, 300m in ECI, with small velocity differences
    x_deputy = x_chief + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])

    x_rel_rtn = brahe.state_eci_to_rtn(x_chief, x_deputy)

    # Verify we get a 6D state vector
    assert x_rel_rtn.shape == (6,)

    # For this simple offset, the radial component should be positive (deputy is farther from Earth)
    # since the position difference adds to the position magnitude
    assert x_rel_rtn[0] > 0  # Positive radial component

    # Total relative position magnitude should be approximately the offset magnitude
    relative_pos_mag = np.linalg.norm(x_rel_rtn[:3])
    offset_mag = np.linalg.norm(np.array([100.0, 200.0, 300.0]))
    assert relative_pos_mag == approx(offset_mag, abs=1.0)  # Within 1m


def test_state_rtn_to_eci_and_back(eop):
    """
    Test round-trip transformation: ECI -> RTN -> ECI.

    Mirrors test_state_eci_to_rtn_and_back in Rust.
    """
    x_chief = get_test_state()
    x_deputy = x_chief + np.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])

    # Transform to RTN frame
    x_rel_rtn = brahe.state_eci_to_rtn(x_chief, x_deputy)

    # Transform back to ECI
    x_deputy_reconstructed = brahe.state_rtn_to_eci(x_chief, x_rel_rtn)

    # Should recover original deputy state
    assert np.linalg.norm(x_deputy - x_deputy_reconstructed) == approx(0.0, abs=1e-6)


def test_state_eci_to_rtn_and_back_non_aligned_orbit(eop):
    """
    Test ECI -> RTN -> ECI round-trip with a non-axis-aligned inclined orbit.

    This catches frame-mixing bugs in the velocity transformation that are
    invisible when the orbit is axis-aligned (RTN == ECI axes).

    Mirrors test_state_eci_to_rtn_and_back_non_aligned in Rust.
    """
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    oe_deputy = np.array([brahe.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.DEGREES)
    x_deputy = brahe.state_koe_to_eci(oe_deputy, brahe.AngleFormat.DEGREES)

    # Round-trip: ECI -> RTN -> ECI
    x_rel_rtn = brahe.state_eci_to_rtn(x_chief, x_deputy)
    x_deputy_reconstructed = brahe.state_rtn_to_eci(x_chief, x_rel_rtn)

    pos_err = np.linalg.norm(x_deputy[:3] - x_deputy_reconstructed[:3])
    vel_err = np.linalg.norm(x_deputy[3:] - x_deputy_reconstructed[3:])

    assert pos_err == approx(0.0, abs=1e-8)
    assert vel_err == approx(0.0, abs=1e-8)


def test_state_rtn_to_eci_and_back_non_aligned_orbit(eop):
    """
    Test RTN -> ECI -> RTN round-trip with a non-axis-aligned inclined orbit.

    Mirrors test_state_rtn_to_eci_and_back_non_aligned in Rust.
    """
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.DEGREES)

    # Known RTN offset
    x_rel_rtn = np.array([1000.0, 500.0, -300.0, 0.1, -0.05, 0.02])

    # Round-trip: RTN -> ECI -> RTN
    x_deputy = brahe.state_rtn_to_eci(x_chief, x_rel_rtn)
    x_rel_rtn_recovered = brahe.state_eci_to_rtn(x_chief, x_deputy)

    pos_err = np.linalg.norm(x_rel_rtn[:3] - x_rel_rtn_recovered[:3])
    vel_err = np.linalg.norm(x_rel_rtn[3:] - x_rel_rtn_recovered[3:])

    assert pos_err == approx(0.0, abs=1e-8)
    assert vel_err == approx(0.0, abs=1e-8)


def test_batch_rtn(eop):
    chiefs = np.array(
        [
            brahe.state_koe_to_eci(
                np.array(
                    [brahe.R_EARTH + 700e3 + 1e3 * i, 0.001, 97.8, 15.0, 30.0, 45.0 + i]
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
                        0.0015,
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
    R = brahe.rotation_rtn_to_eci(chiefs)
    Rt = brahe.rotation_eci_to_rtn(chiefs)
    assert R.shape == (3, 3, 3)
    rel = brahe.state_eci_to_rtn(chiefs, deputies)
    rel_one = brahe.state_eci_to_rtn(chiefs[0], deputies)
    rel_one_dep = brahe.state_eci_to_rtn(chiefs, deputies[0])
    back = brahe.state_rtn_to_eci(chiefs, rel)
    back_one = brahe.state_rtn_to_eci(chiefs[0], rel_one)
    assert (
        rel.shape == (3, 6) and rel_one.shape == (3, 6) and rel_one_dep.shape == (3, 6)
    )
    for i in range(3):
        np.testing.assert_array_equal(R[i], brahe.rotation_rtn_to_eci(chiefs[i]))
        np.testing.assert_array_equal(Rt[i], brahe.rotation_eci_to_rtn(chiefs[i]))
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_rtn(chiefs[i], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_one[i], brahe.state_eci_to_rtn(chiefs[0], deputies[i])
        )
        np.testing.assert_array_equal(
            rel_one_dep[i], brahe.state_eci_to_rtn(chiefs[i], deputies[0])
        )
        np.testing.assert_array_equal(
            back[i], brahe.state_rtn_to_eci(chiefs[i], rel[i])
        )
        np.testing.assert_array_equal(
            back_one[i], brahe.state_rtn_to_eci(chiefs[0], rel_one[i])
        )
    np.testing.assert_allclose(back, deputies, atol=1e-6)
    np.testing.assert_array_equal(
        brahe.state_eci_to_rtn(chiefs.T, deputies.T, axis=0), rel.T
    )
    with pytest.raises(ValueError):
        brahe.state_eci_to_rtn(chiefs[:2], deputies)


def test_batch_rtn_length_one_broadcast(eop):
    chiefs = np.array(
        [
            brahe.state_koe_to_eci(
                np.array(
                    [brahe.R_EARTH + 700e3 + 1e3 * i, 0.001, 97.8, 15.0, 30.0, 45.0 + i]
                ),
                brahe.AngleFormat.DEGREES,
            )
            for i in range(3)
        ]
    )
    deputies = chiefs + np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
    rel = brahe.state_eci_to_rtn(chiefs[:1], deputies)
    assert rel.shape == (3, 6)
    for i in range(3):
        np.testing.assert_array_equal(
            rel[i], brahe.state_eci_to_rtn(chiefs[0], deputies[i])
        )
    rel_b = brahe.state_eci_to_rtn(chiefs, deputies[:1])
    for i in range(3):
        np.testing.assert_array_equal(
            rel_b[i], brahe.state_eci_to_rtn(chiefs[i], deputies[0])
        )
    np.testing.assert_array_equal(
        brahe.state_eci_to_rtn(chiefs[:1].T, deputies.T, axis=0), rel.T
    )


# ============================================================================
# RTN Covariance
# ============================================================================


def _inclined_test_state():
    oe = np.array([brahe.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
    return brahe.state_koe_to_eci(oe, brahe.AngleFormat.DEGREES)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def test_jacobian_rtn_to_eci_inertial_is_block_diagonal(eop):
    """Rust: test_jacobian_rtn_to_eci_inertial_is_block_diagonal"""
    x = _inclined_test_state()
    j = brahe.jacobian_rtn_to_eci(x, brahe.OrbitRelativeFrameVariant.INERTIAL)
    r = brahe.rotation_rtn_to_eci(x)

    np.testing.assert_allclose(j[0:3, 0:3], r, atol=1e-15, rtol=0)
    np.testing.assert_allclose(j[3:6, 3:6], r, atol=1e-15, rtol=0)
    np.testing.assert_allclose(j[3:6, 0:3], np.zeros((3, 3)), atol=1e-15, rtol=0)
    np.testing.assert_allclose(j[0:3, 3:6], np.zeros((3, 3)), atol=1e-15, rtol=0)


def test_jacobian_rtn_to_eci_rotating_coupling(eop):
    """Rust: test_jacobian_rtn_to_eci_rotating_coupling"""
    x = _inclined_test_state()
    j = brahe.jacobian_rtn_to_eci(x, brahe.OrbitRelativeFrameVariant.ROTATING)
    r = brahe.rotation_rtn_to_eci(x)
    expected = r @ _skew(brahe.omega_rtn(x))

    np.testing.assert_allclose(j[3:6, 0:3], expected, atol=1e-18, rtol=0)
    assert np.linalg.norm(j[3:6, 0:3]) > 0.0
    np.testing.assert_allclose(j[0:3, 3:6], np.zeros((3, 3)), atol=1e-15, rtol=0)


def test_jacobian_eci_to_rtn_rotating_coupling(eop):
    """Rust: test_jacobian_eci_to_rtn_rotating_coupling"""
    x = _inclined_test_state()
    j = brahe.jacobian_eci_to_rtn(x, brahe.OrbitRelativeFrameVariant.ROTATING)
    r = brahe.rotation_eci_to_rtn(x)
    expected = -_skew(brahe.omega_rtn(x)) @ r

    np.testing.assert_allclose(j[3:6, 0:3], expected, atol=1e-18, rtol=0)


def test_jacobian_rtn_eci_inverse_identity(eop):
    """Rust: test_jacobian_rtn_eci_inverse_identity"""
    x = _inclined_test_state()
    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        forward = brahe.jacobian_rtn_to_eci(x, variant)
        inverse = brahe.jacobian_eci_to_rtn(x, variant)
        np.testing.assert_allclose(inverse @ forward, np.eye(6), atol=1e-12, rtol=0)
        np.testing.assert_allclose(forward @ inverse, np.eye(6), atol=1e-12, rtol=0)


def test_covariance_rtn_eci_round_trip(eop):
    """Rust: test_covariance_rtn_eci_round_trip"""
    x = _inclined_test_state()
    p = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
    p[0, 1] = p[1, 0] = 25.0
    p[2, 5] = p[5, 2] = 0.4

    for variant in (
        brahe.OrbitRelativeFrameVariant.INERTIAL,
        brahe.OrbitRelativeFrameVariant.ROTATING,
    ):
        p_eci = brahe.covariance_rtn_to_eci(x, p, variant)
        p_back = brahe.covariance_eci_to_rtn(x, p_eci, variant)
        assert np.linalg.norm(p_back - p) / np.linalg.norm(p) < 1e-12
        np.testing.assert_allclose(p_eci, p_eci.T, atol=1e-18, rtol=0)


def test_covariance_eci_to_rtn_inertial_is_pure_rotation(eop):
    """Rust: test_covariance_eci_to_rtn_inertial_is_pure_rotation"""
    x = _inclined_test_state()
    p = np.eye(6) * 100.0

    p_rtn = brahe.covariance_eci_to_rtn(x, p, brahe.OrbitRelativeFrameVariant.INERTIAL)
    np.testing.assert_allclose(p_rtn, p, atol=1e-12, rtol=0)

    p_rot = brahe.covariance_eci_to_rtn(x, p, brahe.OrbitRelativeFrameVariant.ROTATING)
    assert np.linalg.norm(p_rot - p) > 1e-6
