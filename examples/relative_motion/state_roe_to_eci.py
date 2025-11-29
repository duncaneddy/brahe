# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Convert chief satellite ECI state and ROE to deputy satellite ECI state
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

# Convert chief orbital elements to ECI state
x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)

print("Chief ECI State:")
print(f"  Position: [{x_chief[0]:.3f}, {x_chief[1]:.3f}, {x_chief[2]:.3f}] m")
print(f"  Velocity: [{x_chief[3]:.3f}, {x_chief[4]:.3f}, {x_chief[5]:.3f}] m/s")

# Define Relative Orbital Elements (ROE)
# This defines a small relative orbit around the chief
roe = np.array(
    [
        1.413e-4,  # da: relative semi-major axis (dimensionless)
        0.093,  # d_lambda: relative mean longitude (deg)
        4.324e-4,  # dex: relative eccentricity x-component
        2.511e-4,  # dey: relative eccentricity y-component
        0.05,  # dix: relative inclination x-component (deg)
        0.05,  # diy: relative inclination y-component (deg)
    ]
)

print("\nRelative Orbital Elements (ROE):")
print(f"  da (relative SMA):        {roe[0]:.6e}")
print(f"  d_lambda (relative mean long):  {roe[1]:.6f} deg")
print(f"  dex (rel ecc x-comp):     {roe[2]:.6e}")
print(f"  dey (rel ecc y-comp):     {roe[3]:.6e}")
print(f"  dix (rel inc x-comp):     {roe[4]:.6f} deg")
print(f"  diy (rel inc y-comp):     {roe[5]:.6f} deg")

# Convert chief ECI state and ROE to deputy ECI state
x_deputy = bh.state_roe_to_eci(x_chief, roe, bh.AngleFormat.DEGREES)

print("\nDeputy ECI State (computed from ROE):")
print(f"  Position: [{x_deputy[0]:.3f}, {x_deputy[1]:.3f}, {x_deputy[2]:.3f}] m")
print(f"  Velocity: [{x_deputy[3]:.3f}, {x_deputy[4]:.3f}, {x_deputy[5]:.3f}] m/s")

# Compute relative distance
rel_pos = x_deputy[:3] - x_chief[:3]
rel_dist = np.linalg.norm(rel_pos)
print(f"\nRelative distance: {rel_dist:.1f} m")
# Chief ECI State:
#   Position: [4652982.458, 1200261.918, 5093905.755] m
#   Velocity: [-5189.098, 3310.839, 4550.927] m/s
#
# Relative Orbital Elements (ROE):
#   da (relative SMA):        1.413000e-04
#   d_lambda (relative mean long):  0.093000 deg
#   dex (rel ecc x-comp):     4.324000e-04
#   dey (rel ecc y-comp):     2.511000e-04
#   dix (rel inc x-comp):     0.050000 deg
#   diy (rel inc y-comp):     0.050000 deg
#
# Deputy ECI State (computed from ROE):
#   Position: [4654145.325, 1200531.447, 5095024.258] m
#   Velocity: [-5189.999, 3311.448, 4550.982] m/s
#
# Relative distance: 1617.7 m
