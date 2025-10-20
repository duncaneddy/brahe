# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Example demonstrating conversion between true and eccentric anomaly.

This example shows how to convert between true anomaly (nu) and eccentric anomaly (E)
for orbits with different eccentricities. The relationship depends on the orbital geometry.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Test case 1: Low eccentricity (near-circular orbit)
    nu_start = 45.0  # True anomaly in degrees
    e_low = 0.01  # Low eccentricity

    # Convert true to eccentric anomaly
    E = bh.anomaly_true_to_eccentric(nu_start, e_low, bh.AngleFormat.DEGREES)

    # Convert back to true anomaly
    nu_end = bh.anomaly_eccentric_to_true(E, e_low, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert nu_end == pytest.approx(nu_start, abs=1e-8)

    # Test case 2: Moderate eccentricity
    nu_start = 90.0  # True anomaly in degrees
    e_mod = 0.4  # Moderate eccentricity

    # Convert true to eccentric anomaly
    E = bh.anomaly_true_to_eccentric(nu_start, e_mod, bh.AngleFormat.DEGREES)

    # Convert back to true anomaly
    nu_end = bh.anomaly_eccentric_to_true(E, e_mod, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert nu_end == pytest.approx(nu_start, abs=1e-8)

    # Test case 3: High eccentricity (elliptical orbit)
    nu_start = 135.0  # True anomaly in degrees
    e_high = 0.8  # High eccentricity

    # Convert true to eccentric anomaly
    E = bh.anomaly_true_to_eccentric(nu_start, e_high, bh.AngleFormat.DEGREES)

    # Convert back to true anomaly
    nu_end = bh.anomaly_eccentric_to_true(E, e_high, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert nu_end == pytest.approx(nu_start, abs=1e-8)

    print("✓ True-eccentric anomaly conversions validated successfully!")
