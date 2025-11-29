"""
Tests for ECI to Relative Orbital Elements (ROE) transformations.

These tests mirror the Rust tests in src/relative_motion/eci_roe.rs
"""

import brahe
import numpy as np
from pytest import approx


def test_state_eci_to_roe_degrees(eop):
    """
    Test conversion from chief/deputy ECI states to ROE using degrees.

    Mirrors test_state_eci_to_roe_degrees in Rust.
    """
    # Define chief and deputy orbital elements
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    oe_deputy = np.array([brahe.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

    # Convert to ECI states
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.DEGREES)
    x_deputy = brahe.state_koe_to_eci(oe_deputy, brahe.AngleFormat.DEGREES)

    # Compute ROE from ECI states
    roe_from_eci = brahe.state_eci_to_roe(x_chief, x_deputy, brahe.AngleFormat.DEGREES)

    # Compute expected ROE directly from orbital elements
    roe_from_oe = brahe.state_oe_to_roe(oe_chief, oe_deputy, brahe.AngleFormat.DEGREES)

    # Results should match
    assert roe_from_eci[0] == approx(roe_from_oe[0], abs=1e-10)
    assert roe_from_eci[1] == approx(roe_from_oe[1], abs=1e-10)
    assert roe_from_eci[2] == approx(roe_from_oe[2], abs=1e-10)
    assert roe_from_eci[3] == approx(roe_from_oe[3], abs=1e-10)
    assert roe_from_eci[4] == approx(roe_from_oe[4], abs=1e-10)
    assert roe_from_eci[5] == approx(roe_from_oe[5], abs=1e-10)


def test_state_roe_to_eci_degrees(eop):
    """
    Test conversion from chief ECI state and ROE to deputy ECI state using degrees.

    Mirrors test_state_roe_to_eci_degrees in Rust.
    """
    # Define chief orbital elements
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.DEGREES)

    # Define ROE
    roe = np.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])

    # Compute deputy ECI from chief ECI and ROE
    x_deputy = brahe.state_roe_to_eci(x_chief, roe, brahe.AngleFormat.DEGREES)

    # Verify that the deputy state is valid (position magnitude should be reasonable)
    pos_mag = np.linalg.norm(x_deputy[:3])
    assert pos_mag > brahe.R_EARTH  # Deputy should be above Earth's surface
    assert pos_mag < brahe.R_EARTH + 2000e3  # Deputy should be in reasonable orbit

    # Velocity magnitude should be reasonable for this orbit
    vel_mag = np.linalg.norm(x_deputy[3:])
    assert vel_mag > 6000.0  # Reasonable orbital velocity
    assert vel_mag < 9000.0


def test_state_eci_roe_roundtrip_degrees(eop):
    """
    Test roundtrip conversion: ECI -> ROE -> ECI using degrees.

    Mirrors test_state_eci_roe_roundtrip_degrees in Rust.
    """
    # Define chief and deputy orbital elements
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    oe_deputy = np.array([brahe.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

    # Convert to ECI states
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.DEGREES)
    x_deputy_orig = brahe.state_koe_to_eci(oe_deputy, brahe.AngleFormat.DEGREES)

    # ECI -> ROE -> ECI roundtrip
    roe = brahe.state_eci_to_roe(x_chief, x_deputy_orig, brahe.AngleFormat.DEGREES)
    x_deputy_recovered = brahe.state_roe_to_eci(x_chief, roe, brahe.AngleFormat.DEGREES)

    # Should recover original deputy state
    assert x_deputy_recovered[0] == approx(x_deputy_orig[0], abs=1e-3)
    assert x_deputy_recovered[1] == approx(x_deputy_orig[1], abs=1e-3)
    assert x_deputy_recovered[2] == approx(x_deputy_orig[2], abs=1e-3)
    assert x_deputy_recovered[3] == approx(x_deputy_orig[3], abs=1e-6)
    assert x_deputy_recovered[4] == approx(x_deputy_orig[4], abs=1e-6)
    assert x_deputy_recovered[5] == approx(x_deputy_orig[5], abs=1e-6)


def test_state_eci_to_roe_radians(eop):
    """
    Test conversion from chief/deputy ECI states to ROE using radians.

    Mirrors test_state_eci_to_roe_radians in Rust.
    """
    # Define chief and deputy orbital elements in radians
    oe_chief = np.array(
        [
            brahe.R_EARTH + 700e3,
            0.001,
            97.8 * brahe.DEG2RAD,
            15.0 * brahe.DEG2RAD,
            30.0 * brahe.DEG2RAD,
            45.0 * brahe.DEG2RAD,
        ]
    )
    oe_deputy = np.array(
        [
            brahe.R_EARTH + 701e3,
            0.0015,
            97.85 * brahe.DEG2RAD,
            15.05 * brahe.DEG2RAD,
            30.05 * brahe.DEG2RAD,
            45.05 * brahe.DEG2RAD,
        ]
    )

    # Convert to ECI states
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.RADIANS)
    x_deputy = brahe.state_koe_to_eci(oe_deputy, brahe.AngleFormat.RADIANS)

    # Compute ROE from ECI states
    roe_from_eci = brahe.state_eci_to_roe(x_chief, x_deputy, brahe.AngleFormat.RADIANS)

    # Compute expected ROE directly from orbital elements
    roe_from_oe = brahe.state_oe_to_roe(oe_chief, oe_deputy, brahe.AngleFormat.RADIANS)

    # Results should match
    assert roe_from_eci[0] == approx(roe_from_oe[0], abs=1e-10)
    assert roe_from_eci[1] == approx(roe_from_oe[1], abs=1e-10)
    assert roe_from_eci[2] == approx(roe_from_oe[2], abs=1e-10)
    assert roe_from_eci[3] == approx(roe_from_oe[3], abs=1e-10)
    assert roe_from_eci[4] == approx(roe_from_oe[4], abs=1e-10)
    assert roe_from_eci[5] == approx(roe_from_oe[5], abs=1e-10)


def test_state_eci_roe_roundtrip_radians(eop):
    """
    Test roundtrip conversion: ECI -> ROE -> ECI using radians.

    Mirrors test_state_eci_roe_roundtrip_radians in Rust.
    """
    # Define chief and deputy orbital elements in radians
    oe_chief = np.array(
        [
            brahe.R_EARTH + 700e3,
            0.001,
            97.8 * brahe.DEG2RAD,
            15.0 * brahe.DEG2RAD,
            30.0 * brahe.DEG2RAD,
            45.0 * brahe.DEG2RAD,
        ]
    )
    oe_deputy = np.array(
        [
            brahe.R_EARTH + 701e3,
            0.0015,
            97.85 * brahe.DEG2RAD,
            15.05 * brahe.DEG2RAD,
            30.05 * brahe.DEG2RAD,
            45.05 * brahe.DEG2RAD,
        ]
    )

    # Convert to ECI states
    x_chief = brahe.state_koe_to_eci(oe_chief, brahe.AngleFormat.RADIANS)
    x_deputy_orig = brahe.state_koe_to_eci(oe_deputy, brahe.AngleFormat.RADIANS)

    # ECI -> ROE -> ECI roundtrip
    roe = brahe.state_eci_to_roe(x_chief, x_deputy_orig, brahe.AngleFormat.RADIANS)
    x_deputy_recovered = brahe.state_roe_to_eci(x_chief, roe, brahe.AngleFormat.RADIANS)

    # Should recover original deputy state
    assert x_deputy_recovered[0] == approx(x_deputy_orig[0], abs=1e-3)
    assert x_deputy_recovered[1] == approx(x_deputy_orig[1], abs=1e-3)
    assert x_deputy_recovered[2] == approx(x_deputy_orig[2], abs=1e-3)
    assert x_deputy_recovered[3] == approx(x_deputy_orig[3], abs=1e-6)
    assert x_deputy_recovered[4] == approx(x_deputy_orig[4], abs=1e-6)
    assert x_deputy_recovered[5] == approx(x_deputy_orig[5], abs=1e-6)
