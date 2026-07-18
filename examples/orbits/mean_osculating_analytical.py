# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Convert a single Keplerian state between mean and osculating elements using
the analytical Brouwer-Lyddane method.

Demonstrates mean -> osculating -> mean: the osculating state differs from
the mean state by short-period J2 oscillations, and round-tripping recovers
the original mean state only approximately (first-order truncation of the
underlying series).
"""

import numpy as np
import brahe as bh

bh.initialize_eop()


def print_koe(label, koe):
    print(
        f"{label}: a={koe[0]:.3f} m, e={koe[1]:.6f}, i={koe[2]:.3f} deg, "
        f"raan={koe[3]:.3f} deg, argp={koe[4]:.3f} deg, M={koe[5]:.3f} deg"
    )


# Mean Keplerian elements for a LEO satellite (angles in degrees)
mean = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0])
method = bh.MeanElementMethod.BROUWER_LYDDANE

# Mean -> osculating
osc = bh.state_koe_mean_to_osc(mean, method, bh.AngleFormat.DEGREES)

# Osculating -> mean (round trip back toward the original mean state)
mean_recovered = bh.state_koe_osc_to_mean(osc, method, bh.AngleFormat.DEGREES)

print_koe("Mean elements       ", mean)
print_koe("Osculating elements ", osc)
print_koe("Recovered mean      ", mean_recovered)

print(f"\nSemi-major axis, osculating - mean:       {osc[0] - mean[0]:+.3f} m")
print(f"Semi-major axis, round-trip residual:     {mean_recovered[0] - mean[0]:+.6f} m")
print(
    f"Argument of perigee, round-trip residual: {mean_recovered[4] - mean[4]:+.6e} deg"
)
