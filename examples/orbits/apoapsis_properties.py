# /// script
# dependencies = ["brahe"]
# ///
"""
Compute apoapsis properties for an orbit.

This example demonstrates computing apoapsis velocity, distance, and altitude
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
print(f"Apoapsis distance: {apoapsis_distance / 1e3:.3f} km")

# Compute apoapsis altitude (generic)
apoapsis_altitude = bh.apoapsis_altitude(a, e, bh.R_EARTH)
print(f"Apoapsis altitude: {apoapsis_altitude / 1e3:.3f} km")

# Compute as an apogee altitude (Earth-specific)
apogee_altitude = bh.apogee_altitude(a, e)
print(f"Apogee altitude:   {apogee_altitude / 1e3:.3f} km")

# Expected output:
# Apoapsis velocity: 7536.859 m/s
# Apogee velocity:   7536.859 m/s
# Apoapsis distance: 6946.918 km
# Apoapsis altitude: 568.781 km
# Apogee altitude:   568.781 km
