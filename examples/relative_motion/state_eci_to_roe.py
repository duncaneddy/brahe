# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Convert chief and deputy satellite ECI states to Relative Orbital Elements (ROE)
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
        97.85,  # 0.05 deg higher inclination
        15.05,  # Small RAAN difference
        30.05,  # Small argument of perigee difference
        45.05,  # Small mean anomaly difference
    ]
)

# Convert orbital elements to ECI state vectors
x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)

print("Chief ECI State:")
print(f"  Position: [{x_chief[0]:.3f}, {x_chief[1]:.3f}, {x_chief[2]:.3f}] m")
print(f"  Velocity: [{x_chief[3]:.3f}, {x_chief[4]:.3f}, {x_chief[5]:.3f}] m/s")

print("\nDeputy ECI State:")
print(f"  Position: [{x_deputy[0]:.3f}, {x_deputy[1]:.3f}, {x_deputy[2]:.3f}] m")
print(f"  Velocity: [{x_deputy[3]:.3f}, {x_deputy[4]:.3f}, {x_deputy[5]:.3f}] m/s")

# Convert ECI states directly to Relative Orbital Elements (ROE)
roe = bh.state_eci_to_roe(x_chief, x_deputy, bh.AngleFormat.DEGREES)

print("\nRelative Orbital Elements (ROE):")
print(f"  da (relative SMA):        {roe[0]:.6e}")
print(f"  d_lambda (relative mean long):  {roe[1]:.6f} deg")
print(f"  dex (rel ecc x-comp):     {roe[2]:.6e}")
print(f"  dey (rel ecc y-comp):     {roe[3]:.6e}")
print(f"  dix (rel inc x-comp):     {roe[4]:.6f} deg")
print(f"  diy (rel inc y-comp):     {roe[5]:.6f} deg")
# Chief ECI State:
#   Position: [4652982.458, 1200261.918, 5093905.755] m
#   Velocity: [-5189.098, 3310.839, 4550.927] m/s
#
# Deputy ECI State:
#   Position: [4654145.691, 1200531.587, 5095024.654] m
#   Velocity: [-5189.999, 3311.448, 4550.982] m/s
#
# Relative Orbital Elements (ROE):
#   da (relative SMA):        1.412801e-04
#   d_lambda (relative mean long):  0.093214 deg
#   dex (rel ecc x-comp):     4.323577e-04
#   dey (rel ecc y-comp):     2.511333e-04
#   dix (rel inc x-comp):     0.050000 deg
#   diy (rel inc y-comp):     0.049537 deg
