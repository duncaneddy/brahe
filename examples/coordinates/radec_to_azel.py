# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute a star's azimuth/elevation as seen from a ground site at a given
epoch, and convert back from azimuth/elevation to right ascension/declination.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
site = np.array([-122.17, 37.43, 100.0])  # Stanford, deg/deg/m
x_radec = np.array([101.28, -16.72, 1.0])  # Sirius, deg/deg/(unit range)

x_azel = bh.position_radec_to_azel(x_radec, site, epc, bh.AngleFormat.DEGREES)
print(f"Azimuth: {x_azel[0]:.4f} deg, Elevation: {x_azel[1]:.4f} deg")

x_radec_back = bh.position_azel_to_radec(x_azel, site, epc, bh.AngleFormat.DEGREES)
print(f"RA: {x_radec_back[0]:.6f} deg, Dec: {x_radec_back[1]:.6f} deg")

assert abs(x_radec_back[0] - x_radec[0]) < 1e-6
assert abs(x_radec_back[1] - x_radec[1]) < 1e-6
