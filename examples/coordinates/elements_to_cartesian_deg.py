# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between Keplerian orbital elements and Cartesian state vectors
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
# LEO satellite: 500 km altitude, 97.8° inclination (~sun-synchronous)
oe_deg = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert orbital elements to Cartesian state using degrees
state = bh.state_osculating_to_cartesian(oe_deg, bh.AngleFormat.DEGREES)
print("Cartesian state [x, y, z, vx, vy, vz] (m, m/s):")
print(f"Position: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]")
print(f"Velocity: [{state[3]:.6f}, {state[4]:.6f}, {state[5]:.6f}]")
# Cartesian state  (m, m/s):
# Position: [1848964.106, -434937.468, 6560410.530]
# Velocity: [-7098.379734, -2173.344867, 1913.333385]
