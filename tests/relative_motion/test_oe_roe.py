"""
Tests for Relative Orbital Elements (ROE) transformations.

These tests mirror the Rust tests in src/relative_motion/oe_roe.rs
"""

import brahe
import numpy as np
from pytest import approx


def test_state_oe_to_roe_degrees(eop):
    """
    Test conversion from chief/deputy OE to ROE using degrees.

    Mirrors test_state_oe_to_roe in Rust.
    """
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    oe_deputy = np.array([brahe.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

    roe = brahe.state_oe_to_roe(oe_chief, oe_deputy, brahe.AngleFormat.DEGREES)

    assert roe[0] == approx(1.412801276516814e-4, abs=1e-12)
    assert roe[1] == approx(9.321422137829084e-2, abs=1e-12)
    assert roe[2] == approx(4.323577088687794e-4, abs=1e-12)
    assert roe[3] == approx(2.511333388799496e-4, abs=1e-12)
    assert roe[4] == approx(5.0e-2, abs=1e-12)
    assert roe[5] == approx(4.953739202357540e-2, abs=1e-12)


def test_state_roe_to_oe_degrees(eop):
    """
    Test roundtrip conversion: OE -> ROE -> OE using degrees.

    Mirrors test_state_roe_to_oe in Rust.
    """
    # Test roundtrip: OE -> ROE -> OE
    oe_chief = np.array([brahe.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
    oe_deputy_orig = np.array(
        [brahe.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]
    )

    # Convert to ROE
    roe = brahe.state_oe_to_roe(oe_chief, oe_deputy_orig, brahe.AngleFormat.DEGREES)

    # Convert back to OE
    oe_deputy = brahe.state_roe_to_oe(oe_chief, roe, brahe.AngleFormat.DEGREES)

    # Should match the original deputy OE
    assert oe_deputy[0] == approx(brahe.R_EARTH + 701e3, abs=1e-6)
    assert oe_deputy[1] == approx(0.0015, abs=1e-9)
    assert oe_deputy[2] == approx(97.85, abs=1e-6)
    assert oe_deputy[3] == approx(15.05, abs=1e-6)
    assert oe_deputy[4] == approx(30.05, abs=1e-6)
    assert oe_deputy[5] == approx(45.05, abs=1e-6)


def test_state_oe_to_roe_radians(eop):
    """
    Test conversion from chief/deputy OE to ROE using radians.

    Mirrors test_state_oe_to_roe_radians in Rust.
    """
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

    roe = brahe.state_oe_to_roe(oe_chief, oe_deputy, brahe.AngleFormat.RADIANS)

    # Expected values are the same as degrees test but angles in radians
    assert roe[0] == approx(1.412801276516814e-4, abs=1e-12)
    assert roe[1] == approx(9.321422137829084e-2 * brahe.DEG2RAD, abs=1e-12)
    assert roe[2] == approx(4.323577088687794e-4, abs=1e-12)
    assert roe[3] == approx(2.511333388799496e-4, abs=1e-12)
    assert roe[4] == approx(5.0e-2 * brahe.DEG2RAD, abs=1e-12)
    assert roe[5] == approx(4.953739202357540e-2 * brahe.DEG2RAD, abs=1e-12)


def test_state_roe_to_oe_radians(eop):
    """
    Test roundtrip conversion: OE -> ROE -> OE using radians.

    Mirrors test_state_roe_to_oe_radians in Rust.
    """
    # Test roundtrip: OE -> ROE -> OE (using radians)
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
    oe_deputy_orig = np.array(
        [
            brahe.R_EARTH + 701e3,
            0.0015,
            97.85 * brahe.DEG2RAD,
            15.05 * brahe.DEG2RAD,
            30.05 * brahe.DEG2RAD,
            45.05 * brahe.DEG2RAD,
        ]
    )

    # Convert to ROE
    roe = brahe.state_oe_to_roe(oe_chief, oe_deputy_orig, brahe.AngleFormat.RADIANS)

    # Convert back to OE
    oe_deputy = brahe.state_roe_to_oe(oe_chief, roe, brahe.AngleFormat.RADIANS)

    # Should match the original deputy OE
    assert oe_deputy[0] == approx(brahe.R_EARTH + 701e3, abs=1e-6)
    assert oe_deputy[1] == approx(0.0015, abs=1e-9)
    assert oe_deputy[2] == approx(97.85 * brahe.DEG2RAD, abs=1e-6)
    assert oe_deputy[3] == approx(15.05 * brahe.DEG2RAD, abs=1e-6)
    assert oe_deputy[4] == approx(30.05 * brahe.DEG2RAD, abs=1e-6)
    assert oe_deputy[5] == approx(45.05 * brahe.DEG2RAD, abs=1e-6)
