# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform chief and deputy satellite states from ECI to relative RTN coordinates
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
oe_deputy = np.array(
    [
        bh.R_EARTH + 701e3,  # 1 km higher semi-major axis
        0.0015,  # Slightly higher eccentricity
        97.85,  # 0.05° higher inclination
        15.05,  # Small RAAN difference
        30.05,  # Small argument of perigee difference
        45.00,  # Same mean anomaly
    ]
)

# Convert to Cartesian ECI states
x_chief = bh.state_osculating_to_cartesian(oe_chief, bh.AngleFormat.DEGREES)
x_deputy = bh.state_osculating_to_cartesian(oe_deputy, bh.AngleFormat.DEGREES)

# Transform to relative RTN state
x_rel_rtn = bh.state_eci_to_rtn(x_chief, x_deputy)

print("Relative state in RTN frame:")
print(f"Radial (R):      {x_rel_rtn[0]:.3f} m")
print(f"Along-track (T): {x_rel_rtn[1]:.3f} m")
print(f"Cross-track (N): {x_rel_rtn[2]:.3f} m")
print(f"Velocity R:      {x_rel_rtn[3]:.6f} m/s")
print(f"Velocity T:      {x_rel_rtn[4]:.6f} m/s")
print(f"Velocity N:      {x_rel_rtn[5]:.6f} m/s\n")
# Radial (R):      -1508.659 m
# Along-track (T): 11576.951 m
# Cross-track (N): 4401.874 m
# Velocity R:      -17.504100 m/s
# Velocity T:      12.730654 m/s
# Velocity N:      7.959939 m/s

# Calculate total relative distance
relative_distance = np.linalg.norm(x_rel_rtn[:3])
print(f"Total relative distance: {relative_distance:.3f} m")
# Total relative distance: 12477.113 m
