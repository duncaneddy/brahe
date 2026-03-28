# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute WMMHR-2025 magnetic field and compare full vs truncated resolution
"""

import brahe as bh
import numpy as np

epc = bh.Epoch(2025, 1, 1, 0, 0, 0.0, time_system=bh.UTC)
x_geod = np.array([120.0, 0.0, 0.0])  # lon=120 deg, lat=0, alt=0 m (equator)

# Full resolution (degree 133) -- includes crustal field detail
b_full = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)

print("WMMHR-2025 at (lon=120, lat=0, alt=0) -- Full resolution (nmax=133)")
print(f"  B_east:   {b_full[0]:10.1f} nT")
print(f"  B_north:  {b_full[1]:10.1f} nT")
print(f"  B_zenith: {b_full[2]:10.1f} nT")

b_h = np.sqrt(b_full[0] ** 2 + b_full[1] ** 2)
b_total = np.linalg.norm(b_full)
inclination = np.degrees(np.arctan2(-b_full[2], b_h))
declination = np.degrees(np.arctan2(b_full[0], b_full[1]))

print(f"\n  Total intensity: {b_total:10.1f} nT")
print(f"  Inclination:     {inclination:10.2f} deg")
print(f"  Declination:     {declination:10.2f} deg")

# Truncated resolution (degree 13) -- core field only, like standard WMM
b_low = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES, nmax=13)
diff = np.linalg.norm(b_full - b_low)

print("\nTruncated resolution (nmax=13):")
print(f"  B_east:   {b_low[0]:10.1f} nT")
print(f"  B_north:  {b_low[1]:10.1f} nT")
print(f"  B_zenith: {b_low[2]:10.1f} nT")
print(f"\n  Difference from full resolution: {diff:.1f} nT")
