# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between Keplerian orbital elements and Cartesian state vectors
"""

import brahe as bh
import numpy as np
from math import pi

bh.initialize_eop()

# Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
# LEO satellite: 500 km altitude, 97.8° inclination (~sun-synchronous)
oe_deg = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        pi / 4,  # Inclination (rad)
        pi / 8,  # Right ascension of ascending node (rad)
        pi / 2,  # Argument of periapsis (rad)
        3 * pi / 4,  # Mean anomaly (rad)
    ]
)

# Convert orbital elements to Cartesian state using degrees
state = bh.state_osculating_to_cartesian(oe_deg, bh.AngleFormat.RADIANS)
print("Cartesian state [x, y, z, vx, vy, vz] (m, m/s):")
print(f"Position: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]")
print(f"Velocity: [{state[3]:.6f}, {state[4]:.6f}, {state[5]:.6f}]")
# Cartesian state  (m, m/s):
# Position: [-3117582.037, -5092452.343, -3511765.495]
# Velocity: [6408.435846, -1407.501408, -3752.763969]
