# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Example demonstrating mean motion calculations.

This example shows how to calculate mean motion from semi-major axis and vice versa
for different orbit types. Mean motion represents the average angular velocity of
an orbiting object.
"""

import brahe as bh
import pytest
import math

if __name__ == "__main__":
    # Example 1: Low Earth Orbit (LEO)
    # LEO satellites typically complete ~15 orbits per day
    a_leo = bh.R_EARTH + 500e3  # 500 km altitude

    # Calculate mean motion in radians/second
    n_leo = bh.mean_motion(a_leo, bh.AngleFormat.RADIANS)

    # Convert to revolutions per day
    revs_per_day_leo = n_leo * 86400.0 / (2.0 * math.pi)

    # Verify typical LEO has ~15-16 revolutions per day
    assert 14.0 < revs_per_day_leo < 16.0

    # Calculate semi-major axis from mean motion (round-trip)
    a_leo_check = bh.semimajor_axis(n_leo, bh.AngleFormat.RADIANS)
    assert a_leo_check == pytest.approx(a_leo, rel=1e-10)

    # Example 2: Geostationary Orbit (GEO)
    # GEO satellites complete exactly 1 orbit per sidereal day
    a_geo = bh.R_EARTH + 35786e3  # 35786 km altitude

    # Calculate mean motion in radians/second
    n_geo = bh.mean_motion(a_geo, bh.AngleFormat.RADIANS)

    # Calculate orbital period (should be ~1 sidereal day = 86164 seconds)
    T_geo = 2.0 * math.pi / n_geo

    # Verify period is approximately 1 sidereal day
    sidereal_day = 86164.0905  # seconds
    assert T_geo == pytest.approx(sidereal_day, rel=1e-4)

    # Calculate semi-major axis from mean motion (round-trip)
    a_geo_check = bh.semimajor_axis(n_geo, bh.AngleFormat.RADIANS)
    assert a_geo_check == pytest.approx(a_geo, rel=1e-10)

    # Example 3: Medium Earth Orbit (MEO) - GPS altitude
    # GPS satellites are at ~20,200 km altitude
    a_meo = bh.R_EARTH + 20200e3  # 20200 km altitude

    # Calculate mean motion in degrees/second
    n_meo_deg = bh.mean_motion(a_meo, bh.AngleFormat.DEGREES)

    # Convert to radians/second for period calculation
    n_meo_rad = math.radians(n_meo_deg)
    T_meo = 2.0 * math.pi / n_meo_rad

    # GPS orbital period is approximately 12 hours (43200 seconds)
    assert T_meo == pytest.approx(43200.0, rel=5e-2)

    # Calculate semi-major axis from mean motion (round-trip)
    a_meo_check = bh.semimajor_axis(n_meo_deg, bh.AngleFormat.DEGREES)
    assert a_meo_check == pytest.approx(a_meo, rel=1e-10)

    # Example 4: Verify relationship with orbital_period function
    T_leo_direct = bh.orbital_period(a_leo)
    T_leo_from_n = 2.0 * math.pi / n_leo
    assert T_leo_direct == pytest.approx(T_leo_from_n, rel=1e-10)

    print("✓ Mean motion calculations validated successfully!")
