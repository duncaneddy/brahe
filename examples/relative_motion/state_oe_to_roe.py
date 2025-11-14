# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Convert chief and deputy satellite orbital elements to Relative Orbital Elements (ROE)
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define chief satellite orbital elements
# LEO orbit: 700 km altitude, nearly circular, sun-synchronous inclination
oe_chief = np.array(
    [
        bh.R_EARTH + 700e3,  # Semi-major axis (m)
        0.001,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of perigee (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Define deputy satellite with small orbital element differences
# This creates a quasi-periodic relative orbit
oe_deputy = np.array(
    [
        bh.R_EARTH + 701e3,  # 1 km higher semi-major axis
        0.0015,  # Slightly higher eccentricity
        97.85,  # 0.05° higher inclination
        15.05,  # Small RAAN difference
        30.05,  # Small argument of perigee difference
        45.05,  # Small mean anomaly difference
    ]
)

# Convert to Relative Orbital Elements (ROE)
roe = bh.state_oe_to_roe(oe_chief, oe_deputy, bh.AngleFormat.DEGREES)

print("Relative Orbital Elements (ROE):")
print(f"da (relative SMA):        {roe[0]:.6e}")
print(f"dλ (relative mean long):  {roe[1]:.6f}°")
print(f"dex (rel ecc x-comp):     {roe[2]:.6e}")
print(f"dey (rel ecc y-comp):     {roe[3]:.6e}")
print(f"dix (rel inc x-comp):     {roe[4]:.6f}°")
print(f"diy (rel inc y-comp):     {roe[5]:.6f}°")
# Relative Orbital Elements (ROE):
# da (relative SMA):        1.412801e-4
# dλ (relative mean long):  0.093214°
# dex (rel ecc x-comp):     4.323577e-4
# dey (rel ecc y-comp):     2.511333e-4
# dix (rel inc x-comp):     0.050000°
# diy (rel inc y-comp):     0.049537°
