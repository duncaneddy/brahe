# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute IGRF-14 magnetic field at a geodetic location
"""

import brahe as bh
import numpy as np

# Compute the IGRF magnetic field at 60 degrees latitude, 400 km altitude
epc = bh.Epoch(2025, 1, 1, 0, 0, 0.0, time_system=bh.UTC)
x_geod = np.array([0.0, 60.0, 400e3])  # lon=0 deg, lat=60 deg, alt=400 km

b_enz = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

print("IGRF-14 magnetic field at (lon=0, lat=60, alt=400 km)")
print(f"  B_east:   {b_enz[0]:10.1f} nT")
print(f"  B_north:  {b_enz[1]:10.1f} nT")
print(f"  B_zenith: {b_enz[2]:10.1f} nT")

# Compute derived quantities
b_h = np.sqrt(b_enz[0] ** 2 + b_enz[1] ** 2)  # Horizontal intensity
b_total = np.linalg.norm(b_enz)  # Total intensity
inclination = np.degrees(np.arctan2(-b_enz[2], b_h))  # Positive downward
declination = np.degrees(np.arctan2(b_enz[0], b_enz[1]))

print(f"\n  Horizontal intensity: {b_h:10.1f} nT")
print(f"  Total intensity:     {b_total:10.1f} nT")
print(f"  Inclination:         {inclination:10.2f} deg")
print(f"  Declination:         {declination:10.2f} deg")
