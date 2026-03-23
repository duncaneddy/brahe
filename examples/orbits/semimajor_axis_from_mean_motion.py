# /// script
# dependencies = ["brahe"]
# ///
"""
Compute semi-major axis from mean motion.

This example demonstrates computing the semi-major axis from mean motion,
useful when working with TLE data or orbit design specifications that
provide mean motion instead of semi-major axis.
"""

import brahe as bh

bh.initialize_eop()

# Example 1: ISS-like orbit with ~15.5 revolutions per day
n_iss = 15.5 * 360.0 / 86400.0  # Convert revs/day to deg/s
a_iss = bh.semimajor_axis(n_iss, bh.AngleFormat.DEGREES)
altitude_iss = a_iss - bh.R_EARTH

print("ISS-like Orbit (15.5 revs/day):")
print(f"  Mean motion: {n_iss:.6f} deg/s")
print(f"  Semi-major axis: {a_iss:.3f} m")
print(f"  Altitude: {altitude_iss / 1e3:.3f} km")

# Example 2: Geosynchronous orbit (1 revolution per day)
n_geo = 1.0 * 360.0 / 86400.0  # 1 rev/day in deg/s
a_geo = bh.semimajor_axis(n_geo, bh.AngleFormat.DEGREES)
altitude_geo = a_geo - bh.R_EARTH

print("\nGeosynchronous Orbit (1 rev/day):")
print(f"  Mean motion: {n_geo:.6f} deg/s")
print(f"  Semi-major axis: {a_geo:.3f} m")
print(f"  Altitude: {altitude_geo / 1e3:.3f} km")

# Example 3: Using radians
n_leo_rad = 0.001  # rad/s
a_leo = bh.semimajor_axis(n_leo_rad, bh.AngleFormat.RADIANS)

print("\nLEO from radians/s:")
print(f"  Mean motion: {n_leo_rad:.6f} rad/s")
print(f"  Semi-major axis: {a_leo:.3f} m")
print(f"  Altitude: {(a_leo - bh.R_EARTH) / 1e3:.3f} km")

# Verify round-trip conversion
n_verify = bh.mean_motion(a_iss, bh.AngleFormat.DEGREES)
print("\nRound-trip verification:")
print(f"  Original mean motion: {n_iss:.6f} deg/s")
print(f"  Computed mean motion: {n_verify:.6f} deg/s")
print(f"  Difference: {abs(n_iss - n_verify):.2e} deg/s")
