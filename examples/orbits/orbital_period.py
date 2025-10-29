# /// script
# dependencies = ["brahe"]
# ///
"""
Compute orbital period for Earth-orbiting satellites.

This example demonstrates computing the orbital period from semi-major axis
for both Earth-specific and general gravitational parameter cases.
"""

import brahe as bh

bh.initialize_eop()

# Define orbit parameters
a = bh.R_EARTH + 500.0e3  # Semi-major axis (m) - LEO orbit at 500 km altitude

# Compute orbital period for Earth orbit (uses GM_EARTH internally)
period_earth = bh.orbital_period(a)
print(f"Orbital period (Earth): {period_earth:.3f} s")
print(f"Orbital period (Earth): {period_earth / 60:.3f} min")

# Compute orbital period for general body (explicit GM)
period_general = bh.orbital_period_general(a, bh.GM_EARTH)
print(f"Orbital period (general): {period_general:.3f} s")

# Verify they match
print(f"Difference: {abs(period_earth - period_general):.2e} s")

# Example with approximate GEO altitude
a_geo = bh.R_EARTH + 35786e3
period_geo = bh.orbital_period(a_geo)
print(f"\nGEO orbital period: {period_geo / 3600:.3f} hours")

# Expected output:
# Orbital period (Earth): 5676.977 s
# Orbital period (Earth): 94.616 min
# Orbital period (general): 5676.977 s
# Difference: 0.00e0 s

# GEO orbital period: 23.934 hours
