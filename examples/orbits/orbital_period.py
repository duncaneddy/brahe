# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Demonstrates calculation of orbital period using Kepler's Third Law.

The orbital period depends only on the semi-major axis and central body
gravitational parameter, independent of eccentricity.
"""

import brahe as bh
import pytest

if __name__ == "__main__":
    # Define orbital parameters
    a = bh.R_EARTH + 500e3  # Semi-major axis: 500 km altitude (meters)

    # Calculate orbital period
    T = bh.orbital_period(a)  # Returns period in seconds

    # Expected period for ~500 km LEO is approximately 94.6 minutes
    expected_minutes = 94.6
    assert T / 60 == pytest.approx(expected_minutes, rel=0.01)

    print("Orbital Period Calculation:")
    print("  Altitude: 500 km")
    print(f"  Semi-major axis: {a / 1000:.1f} km")
    print(f"  Period: {T:.2f} seconds")
    print(f"  Period: {T / 60:.2f} minutes")
    print(f"  Period: {T / 3600:.4f} hours")

    # Show periods for different altitudes
    print("\nPeriods for various altitudes:")
    for alt_km in [200, 400, 600, 800, 1000]:
        a_temp = bh.R_EARTH + alt_km * 1000
        T_temp = bh.orbital_period(a_temp)
        print(f"  {alt_km:4d} km: {T_temp / 60:6.2f} minutes")

    print("\n✓ Orbital period calculations validated successfully!")
