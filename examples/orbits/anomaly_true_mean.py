# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Demonstrates conversion between true anomaly and mean anomaly.

True anomaly describes the actual angular position of a body in its orbit,
while mean anomaly represents a fictitious uniform motion. This example
shows how to convert between them for different eccentricities.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Define orbital parameters
    nu_start = 45.0  # True anomaly in degrees
    e = 0.3  # Eccentricity

    # Convert true anomaly to mean anomaly
    M = bh.anomaly_true_to_mean(nu_start, e, bh.AngleFormat.DEGREES)

    # Convert back to true anomaly
    nu_end = bh.anomaly_mean_to_true(M, e, bh.AngleFormat.DEGREES)

    # Verify round-trip conversion
    assert nu_start == pytest.approx(nu_end, abs=1e-12)

    # Show progression for different eccentricities
    print("True → Mean Anomaly Conversion:")
    for ecc in [0.0, 0.1, 0.3, 0.5, 0.7]:
        mean = bh.anomaly_true_to_mean(nu_start, ecc, bh.AngleFormat.DEGREES)
        print(f"  e={ecc:.1f}: ν={nu_start:6.2f}° → M={mean:6.2f}°")

    print("\n✓ Anomaly conversions validated successfully!")
