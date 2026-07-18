# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Recover an osculating Keplerian state from a mean state using the numerical
method's iterative mean -> osculating inverse.

Unlike osculating -> mean (a direct average), mean -> osculating with the
numerical method has no closed form: it differentially corrects a trial
osculating state, numerically propagating it across the averaging window
and comparing the forward-averaged mean against the target, until the
residual converges.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Mean Keplerian elements for a LEO satellite (angles in degrees)
mean = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 0.0])
period = bh.orbital_period(mean[0])

# Dynamics used to test each trial osculating state during differential
# correction: gravity-only (no drag/SRP parameters required).
inverse = bh.MeanElementInverseConfig(
    bh.ForceModelConfig.earth_gravity(),
    bh.NumericalPropagationConfig.default(),
    1.0,  # tolerance
    50,  # max_iterations
)
config = bh.MeanElementNumericalMethodConfig(
    period, bh.WindowAlignment.CENTERED, bh.WindowEdgeHandling.PRESERVE_WINDOW, inverse
)
method = bh.MeanElementMethod.numerical(config)

epoch = bh.Epoch.from_gps_seconds(0.0)
out_epochs, out_states = bh.batch_state_koe_mean_to_osc(
    [epoch], mean.reshape(1, 6), method, bh.AngleFormat.DEGREES
)
osc = out_states[0]

print(
    f"Mean state:            a={mean[0]:.3f} m, e={mean[1]:.6f}, i={mean[2]:.3f} deg, "
    f"raan={mean[3]:.3f} deg, argp={mean[4]:.3f} deg, M={mean[5]:.3f} deg"
)
print(
    f"Recovered osculating:  a={osc[0]:.3f} m, e={osc[1]:.6f}, i={osc[2]:.3f} deg, "
    f"raan={osc[3]:.3f} deg, argp={osc[4]:.3f} deg, M={osc[5]:.3f} deg"
)
print(f"\nSemi-major axis, osculating - mean: {osc[0] - mean[0]:+.3f} m")
