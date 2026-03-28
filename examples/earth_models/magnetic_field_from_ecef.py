# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute magnetic field along a satellite orbit starting from orbital elements
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define a LEO orbit and compute the ECEF state
epc = bh.Epoch(2025, 3, 15, 12, 0, 0.0, time_system=bh.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 51.6, 45.0, 30.0, 60.0])  # ISS-like orbit
state_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
state_ecef = bh.state_eci_to_ecef(epc, state_eci)

# Convert ECEF position to geodetic coordinates
x_ecef = state_ecef[0:3]
x_geod = bh.position_ecef_to_geodetic(x_ecef, bh.AngleFormat.DEGREES)

print(f"Epoch: {epc}")
print(
    f"Geodetic position: lon={x_geod[0]:.2f} deg, lat={x_geod[1]:.2f} deg, alt={x_geod[2] / 1e3:.1f} km"
)

# Compute the magnetic field at the satellite location using IGRF
b_enz = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
b_total = np.linalg.norm(b_enz)

print("\nIGRF field at satellite:")
print(f"  B_east:   {b_enz[0]:10.1f} nT")
print(f"  B_north:  {b_enz[1]:10.1f} nT")
print(f"  B_zenith: {b_enz[2]:10.1f} nT")
print(f"  |B|:      {b_total:10.1f} nT")

# Get the field in ECEF frame (useful for torque calculations in the body frame)
b_ecef = bh.igrf_ecef(epc, x_geod, bh.AngleFormat.DEGREES)
print("\nIGRF field in ECEF frame:")
print(f"  B_x: {b_ecef[0]:10.1f} nT")
print(f"  B_y: {b_ecef[1]:10.1f} nT")
print(f"  B_z: {b_ecef[2]:10.1f} nT")
