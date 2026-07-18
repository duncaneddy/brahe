# /// script
# dependencies = ["brahe", "numpy", "pytest"]
# ///
"""
Convert a Cartesian inertial state to right ascension/declination/range with
rates and back, then apply the same site-relative subtract-then-convert
pattern to a topocentric line-of-sight state.
"""

import brahe as bh
import numpy as np
import pytest

# --8<-- [start:state_transforms]
# RA/Dec/range and their rates: [ra, dec, range, ra_dot, dec_dot, range_dot]
x_radec = np.array([45.0, 30.0, 7000e3, 0.01, -0.005, 50.0])
x_inertial = bh.state_radec_to_inertial(x_radec, bh.AngleFormat.DEGREES)
print(
    f"Inertial state: pos=[{x_inertial[0]:.3f}, {x_inertial[1]:.3f}, {x_inertial[2]:.3f}] m"
)
print(
    f"                vel=[{x_inertial[3]:.6f}, {x_inertial[4]:.6f}, {x_inertial[5]:.6f}] m/s"
)

x_radec_back = bh.state_inertial_to_radec(x_inertial, bh.AngleFormat.DEGREES)
print(
    f"RA/Dec round-trip: ra={x_radec_back[0]:.6f} deg, dec={x_radec_back[1]:.6f} deg, "
    f"range={x_radec_back[2]:.3f} m"
)
print(
    f"                   ra_dot={x_radec_back[3]:.6f} deg/s, "
    f"dec_dot={x_radec_back[4]:.6f} deg/s, range_dot={x_radec_back[5]:.3f} m/s"
)

assert x_radec_back[0] == pytest.approx(x_radec[0], abs=1e-9)
assert x_radec_back[3] == pytest.approx(x_radec[3], abs=1e-9)
# --8<-- [end:state_transforms]

# --8<-- [start:topocentric]
# A satellite and an observing site, both as Cartesian inertial states (m, m/s)
x_sat = np.array([8000e3, 1000e3, 500e3, -1000.0, 7000.0, 2000.0])
x_site = np.array([6378e3, 0.0, 0.0, 0.0, 0.0, 0.0])

x_topocentric = x_sat - x_site
x_radec_topo = bh.state_inertial_to_radec(x_topocentric, bh.AngleFormat.DEGREES)
print(
    f"\nTopocentric line of sight: ra={x_radec_topo[0]:.6f} deg, "
    f"dec={x_radec_topo[1]:.6f} deg, range={x_radec_topo[2]:.3f} m"
)
# --8<-- [end:topocentric]
