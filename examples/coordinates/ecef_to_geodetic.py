# /// script
# dependencies = ["brahe"]
# ///
"""
Convert between ECEF Cartesian and geodetic (WGS84 ellipsoid) coordinates
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

# Convert ECEF Cartesian to geodetic position
ecef_pos = state_ecef[0:3]
geodetic = bh.position_ecef_to_geodetic(ecef_pos, bh.AngleFormat.DEGREES)
print("Geodetic coordinates (WGS84 ellipsoid model):")
print(f"Longitude: {geodetic[0]:.4f}° = {np.radians(geodetic[0]):.6f} rad")
print(f"Latitude:  {geodetic[1]:.4f}° = {np.radians(geodetic[1]):.6f} rad")
print(f"Altitude:  {geodetic[2]:.1f} m")
# Longitude: -111.8041° = -1.951350 rad
# Latitude:  73.3622° = 1.280412 rad
# Altitude:  519618.1 m
