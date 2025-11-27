# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between geocentric spherical and ECEF Cartesian coordinates
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define a satellite state (convert orbital elements to ECEF state)
epc = bh.Epoch(2024, 1, 1, 0, 0, 0.0, time_system=bh.UTC)
state_oe = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.0,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)
state_ecef = bh.state_eci_to_ecef(
    epc, bh.state_koe_to_eci(state_oe, bh.AngleFormat.DEGREES)
)
print("ECEF Cartesian state [x, y, z, vx, vy, vz] (m, m/s):")
print(f"Position: [{state_ecef[0]:.3f}, {state_ecef[1]:.3f}, {state_ecef[2]:.3f}]")
print(f"Velocity: [{state_ecef[3]:.6f}, {state_ecef[4]:.6f}, {state_ecef[5]:.6f}]\n")
# Position: [-735665.465, -1838913.314, 6586801.432]
# Velocity: [-1060.370171, 7357.551468, 1935.662061]

# Convert ECEF Cartesian to geocentric position
ecef_pos = state_ecef[0:3]
geocentric = bh.position_ecef_to_geocentric(ecef_pos, bh.AngleFormat.DEGREES)
print("Geocentric coordinates (spherical Earth model):")
print(f"Longitude: {geocentric[0]:.4f}° = {np.radians(geocentric[0]):.6f} rad")
print(f"Latitude:  {geocentric[1]:.4f}° = {np.radians(geocentric[1]):.6f} rad")
print(f"Altitude:  {geocentric[2]:.1f} m")
# Longitude: -111.8041° = -1.951350 rad
# Latitude:  73.2643° = 1.278704 rad
# Altitude:  499999.3 m
