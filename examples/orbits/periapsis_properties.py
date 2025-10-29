# /// script
# dependencies = ["brahe"]
# ///
"""
Compute periapsis properties for an orbit.

This example demonstrates computing periapsis velocity and distance
for a given orbit, including Earth-specific perigee functions.
"""

import brahe as bh

bh.initialize_eop()

# Define orbit parameters
a = bh.R_EARTH + 500.0e3  # Semi-major axis (m)
e = 0.01  # Eccentricity

# Compute periapsis velocity (generic)
periapsis_velocity = bh.periapsis_velocity(a, e, bh.GM_EARTH)
print(f"Periapsis velocity: {periapsis_velocity:.3f} m/s")

# Compute as a perigee velocity (Earth-specific)
perigee_velocity = bh.perigee_velocity(a, e)
print(f"Perigee velocity:   {perigee_velocity:.3f} m/s")

# Compute periapsis distance
periapsis_distance = bh.periapsis_distance(a, e)
print(f"Periapsis distance: {periapsis_distance:.3f} m")

# Expected output:
# Periapsis velocity: 7689.119 m/s
# Perigee velocity:   7689.119 m/s
# Periapsis distance: 6809354.937 m
