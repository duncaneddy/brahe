# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute orbital period from a state vector.

This example demonstrates computing the orbital period directly from a
Cartesian state vector in ECI coordinates, which is useful when you have
satellite state data but don't know the orbital elements.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define orbital elements for a LEO satellite
a = bh.R_EARTH + 500.0e3  # Semi-major axis (m)
e = 0.01  # Eccentricity
i = 97.8  # Inclination (degrees)
raan = 15.0  # Right ascension of ascending node (degrees)
argp = 30.0  # Argument of periapsis (degrees)
nu = 45.0  # True anomaly (degrees)

# Convert to Cartesian state
oe = np.array([a, e, i, raan, argp, nu])
state_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

print("ECI State (position in km, velocity in km/s):")
print(
    f"  r = [{state_eci[0] / 1e3:.3f}, {state_eci[1] / 1e3:.3f}, {state_eci[2] / 1e3:.3f}] km"
)
print(
    f"  v = [{state_eci[3] / 1e3:.3f}, {state_eci[4] / 1e3:.3f}, {state_eci[5] / 1e3:.3f}] km/s"
)

# Compute orbital period from state vector
period = bh.orbital_period_from_state(state_eci, bh.GM_EARTH)
print(f"\nOrbital period from state: {period:.3f} s")
print(f"Orbital period from state: {period / 60:.3f} min")

# Verify against period computed from semi-major axis
period_from_sma = bh.orbital_period(a)
print(f"\nOrbital period from SMA: {period_from_sma:.3f} s")
print(f"Difference: {abs(period - period_from_sma):.2e} s")

# Expected output:
# ECI State (position in km, velocity in km/s):
#   r = [1848.964, -434.937, 6560.411] km
#   v = [-7.098, -2.173, 1.913] km/s

# Orbital period from state: 5676.977 s
# Orbital period from state: 94.616 min

# Orbital period from SMA: 5676.977 s
# Difference: 3.64e-12 s
