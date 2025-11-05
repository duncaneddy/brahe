# /// script
# dependencies = ["brahe"]
# ///
"""
Compute periapsis properties for an orbit.

This example demonstrates computing periapsis velocity, distance, and altitude
for a given orbit, including Earth-specific perigee functions.
"""

import brahe as bh

bh.initialize_eop()

# Define orbit parameters
a = bh.R_EARTH + 500.0e3  # Semi-major axis (m)
e = 0.01  # Eccentricity

# Compute periapsis velocity (generic)
periapsis_velocity = bh.periapsis_velocity(a, e, gm=bh.GM_EARTH)
print(f"Periapsis velocity: {periapsis_velocity:.3f} m/s")

# Compute as a perigee velocity (Earth-specific)
perigee_velocity = bh.perigee_velocity(a, e)
print(f"Perigee velocity:   {perigee_velocity:.3f} m/s")

# Compute periapsis distance
periapsis_distance = bh.periapsis_distance(a, e)
print(f"Periapsis distance: {periapsis_distance / 1e3:.3f} km")

# Compute periapsis altitude (generic)
periapsis_altitude = bh.periapsis_altitude(a, e, r_body=bh.R_EARTH)
print(f"Periapsis altitude: {periapsis_altitude / 1e3:.3f} km")

# Compute as a perigee altitude (Earth-specific)
perigee_altitude = bh.perigee_altitude(a, e)
print(f"Perigee altitude:   {perigee_altitude / 1e3:.3f} km")

# Expected output:
# Periapsis velocity: 7689.119 m/s
# Perigee velocity:   7689.119 m/s
# Periapsis distance: 6809.355 km
# Periapsis altitude: 431.219 km
# Perigee altitude:   431.219 km
