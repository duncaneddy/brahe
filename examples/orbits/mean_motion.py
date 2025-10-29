# /// script
# dependencies = ["brahe"]
# ///
"""
Compute mean motion from semi-major axis.

This example demonstrates computing the mean motion (average angular rate)
of a satellite from its semi-major axis.
"""

import brahe as bh

bh.initialize_eop()

# Define orbit parameters
a_leo = bh.R_EARTH + 500.0e3  # LEO satellite at 500 km altitude
a_geo = bh.R_EARTH + 35786e3  # GEO satellite

# Compute mean motion in radians/s (Earth-specific)
n_leo_rad = bh.mean_motion(a_leo, bh.AngleFormat.RADIANS)
n_geo_rad = bh.mean_motion(a_geo, bh.AngleFormat.RADIANS)

print("Mean Motion in radians/second:")
print(f"  LEO (500 km): {n_leo_rad:.6f} rad/s")
print(f"  GEO:          {n_geo_rad:.6f} rad/s")

# Compute mean motion in degrees/s
n_leo_deg = bh.mean_motion(a_leo, bh.AngleFormat.DEGREES)
n_geo_deg = bh.mean_motion(a_geo, bh.AngleFormat.DEGREES)

print("\nMean Motion in degrees/second:")
print(f"  LEO (500 km): {n_leo_deg:.6f} deg/s")
print(f"  GEO:          {n_geo_deg:.6f} deg/s")

# Convert to degrees/day (common unit for TLEs)
print("\nMean Motion in degrees/day:")
print(f"  LEO (500 km): {n_leo_deg * 86400:.3f} deg/day")
print(f"  GEO:          {n_geo_deg * 86400:.3f} deg/day")

# Verify using general function
n_leo_general = bh.mean_motion_general(a_leo, bh.GM_EARTH, bh.AngleFormat.RADIANS)
print(f"\nVerification (general function): {n_leo_general:.6f} rad/s")
print(f"Difference: {abs(n_leo_rad - n_leo_general):.2e} rad/s")

# Expected output:
# Mean Motion in radians/second:
#   LEO (500 km): 0.001107 rad/s
#   GEO:          0.000073 rad/s

# Mean Motion in degrees/second:
#   LEO (500 km): 0.063414 deg/s
#   GEO:          0.004178 deg/s

# Mean Motion in degrees/day:
#   LEO (500 km): 5478.972 deg/day
#   GEO:          360.986 deg/day

# Verification (general function): 0.001107 rad/s
# Difference: 0.00e+00 rad/s
