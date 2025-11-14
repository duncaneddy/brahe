"""
Tests for RTN (Radial-Tangential-Normal) frame transformations.

These tests mirror the Rust tests in src/relative_motion/eci_rtn.rs
"""

import brahe
import numpy as np
from pytest import approx


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
