# /// script
# dependencies = ["brahe"]
# ///
"""
Compute apoapsis properties for an orbit.

This example demonstrates computing apoapsis velocity and distance
for a given orbit, including Earth-specific apogee functions.
"""

import brahe as bh

bh.initialize_eop()

# Define orbit parameters
a = bh.R_EARTH + 500.0e3  # Semi-major axis (m)
e = 0.01  # Eccentricity

# Compute apoapsis velocity (generic)
apoapsis_velocity = bh.apoapsis_velocity(a, e, bh.GM_EARTH)
print(f"Apoapsis velocity: {apoapsis_velocity:.3f} m/s")

# Compute as an apogee velocity (Earth-specific)
apogee_velocity = bh.apogee_velocity(a, e)
print(f"Apogee velocity:   {apogee_velocity:.3f} m/s")

# Compute apoapsis distance
apoapsis_distance = bh.apoapsis_distance(a, e)
print(f"Apoapsis distance: {apoapsis_distance:.3f} m")

# Expected output:
# Apoapsis velocity: 7536.859 m/s
# Apogee velocity:   7536.859 m/s
# Apoapsis distance: 6946917.663 m
