# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Example demonstrating conversion between eccentric and mean anomaly.

This example shows how to convert between eccentric anomaly (E) and mean anomaly (M)
for orbits with different eccentricities. The conversion involves solving Kepler's equation.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Test case 1: Low eccentricity (near-circular orbit)
    E_start = 45.0  # Eccentric anomaly in degrees
    e_low = 0.01  # Low eccentricity

    # Convert eccentric to mean anomaly
    M = bh.anomaly_eccentric_to_mean(E_start, e_low, bh.AngleFormat.DEGREES)

    # Convert back to eccentric anomaly
    E_end = bh.anomaly_mean_to_eccentric(M, e_low, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert E_end == pytest.approx(E_start, abs=1e-8)

    # Test case 2: Moderate eccentricity
    E_start = 60.0  # Eccentric anomaly in degrees
    e_mod = 0.3  # Moderate eccentricity

    # Convert eccentric to mean anomaly
    M = bh.anomaly_eccentric_to_mean(E_start, e_mod, bh.AngleFormat.DEGREES)

    # Convert back to eccentric anomaly
    E_end = bh.anomaly_mean_to_eccentric(M, e_mod, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert E_end == pytest.approx(E_start, abs=1e-8)

    # Test case 3: High eccentricity (elliptical orbit)
    E_start = 120.0  # Eccentric anomaly in degrees
    e_high = 0.7  # High eccentricity

    # Convert eccentric to mean anomaly
    M = bh.anomaly_eccentric_to_mean(E_start, e_high, bh.AngleFormat.DEGREES)

    # Convert back to eccentric anomaly
    E_end = bh.anomaly_mean_to_eccentric(M, e_high, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert E_end == pytest.approx(E_start, abs=1e-8)

    print("✓ Eccentric-mean anomaly conversions validated successfully!")
