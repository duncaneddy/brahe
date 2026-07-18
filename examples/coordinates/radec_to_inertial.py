# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Convert a star's right ascension/declination to an inertial unit vector and
back, and propagate its position forward in time using proper motion.
"""

import brahe as bh
import numpy as np
import pytest

# Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
ra = 269.45402305  # deg
dec = 4.66828815  # deg

# Convert RA/Dec to an inertial unit vector (range = 1.0)
x_radec = np.array([ra, dec, 1.0])
x_inertial = bh.position_radec_to_inertial(x_radec, bh.AngleFormat.DEGREES)
print(f"Unit vector: [{x_inertial[0]:.6f}, {x_inertial[1]:.6f}, {x_inertial[2]:.6f}]")

# Convert back to RA/Dec
x_radec_back = bh.position_inertial_to_radec(x_inertial, bh.AngleFormat.DEGREES)
print(f"RA: {x_radec_back[0]:.8f} deg, Dec: {x_radec_back[1]:.8f} deg")

assert x_radec_back[0] == pytest.approx(ra, abs=1e-9)
assert x_radec_back[1] == pytest.approx(dec, abs=1e-9)

# Propagate the star's position forward 10 years using its proper motion,
# parallax, and radial velocity (ESA SP-1200 Vol. 1, §1.5.5).
epoch_from = bh.Epoch.from_mjd(48348.5625, bh.TimeSystem.TT)
epoch_to = bh.Epoch.from_mjd(48348.5625 + 10.0 * 365.25, bh.TimeSystem.TT)

ra_new, dec_new = bh.apply_proper_motion(
    ra,
    dec,
    -797.84,  # pm_ra* (mu_alpha* = mu_alpha * cos(dec)), mas/yr
    10326.93,  # pm_dec, mas/yr
    549.30,  # parallax, mas
    -106.8,  # radial_velocity, km/s
    epoch_from,
    epoch_to,
    bh.AngleFormat.DEGREES,
)
print(f"After 10 yr: RA: {ra_new:.6f} deg, Dec: {dec_new:.6f} deg")

assert ra_new != pytest.approx(ra, abs=1e-6)
