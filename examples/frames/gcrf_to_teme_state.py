# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform a state vector from GCRF to the true equator and mean equinox of date (TEME)
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

# Define orbital elements in degrees
# LEO satellite: 500 km altitude, sun-synchronous orbit
oe = np.array(
    [
        bh.R_EARTH + 500e3,  # Semi-major axis (m)
        0.01,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of periapsis (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

epc = bh.Epoch(2024, 1, 1, 12, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

# Convert to GCRF Cartesian state
state_gcrf = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

print("\nGCRF state vector:")
print(f"  Position: [{state_gcrf[0]:.3f}, {state_gcrf[1]:.3f}, {state_gcrf[2]:.3f}] m")
print(
    f"  Velocity: [{state_gcrf[3]:.6f}, {state_gcrf[4]:.6f}, {state_gcrf[5]:.6f}] m/s\n"
)

# Transform to TEME at the given epoch
state_teme = bh.state_gcrf_to_teme(epc, state_gcrf)

print("TEME state vector:")
print(f"  Position: [{state_teme[0]:.3f}, {state_teme[1]:.3f}, {state_teme[2]:.3f}] m")
print(
    f"  Velocity: [{state_teme[3]:.6f}, {state_teme[4]:.6f}, {state_teme[5]:.6f}] m/s\n"
)

pos_diff = np.linalg.norm(state_gcrf[0:3] - state_teme[0:3])
print(f"Position difference norm: {pos_diff:.3f} m")
