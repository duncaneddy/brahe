# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Recover mean Keplerian elements from an osculating trajectory using
windowed numerical averaging.

Synthesizes one period of osculating states by evaluating the analytical
Brouwer-Lyddane mean -> osculating mapping at a sweep of mean anomalies
(holding the other mean elements fixed; no numerical propagation is used).
The synthesized trajectory is then averaged back down to mean elements with
a centered window spanning the full period.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Mean Keplerian elements for a LEO satellite (angles in degrees); only the
# mean anomaly varies across the synthesized trajectory.
a, e, i, raan, argp = bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0
period = bh.orbital_period(a)

# Sample one full period at 121 points so the exact midpoint (M = 180 deg)
# falls on the epoch grid.
n_samples = 121
mean_anomalies = np.linspace(0.0, 360.0, n_samples)
epoch0 = bh.Epoch.from_gps_seconds(0.0)
epochs = [epoch0 + k / (n_samples - 1) * period for k in range(n_samples)]

method_analytical = bh.MeanElementMethod.BROUWER_LYDDANE
osc_states = np.zeros((n_samples, 6))
for k, m in enumerate(mean_anomalies):
    mean_state = np.array([a, e, i, raan, argp, m])
    osc_states[k, :] = bh.state_koe_mean_to_osc(
        mean_state, method_analytical, bh.AngleFormat.DEGREES
    )

# Average the synthesized osculating trajectory over a centered window
# spanning one full period. With TRUNCATE edge handling, only output epochs
# whose window is fully covered by the input data survive; since the window
# equals the full data span, that is exactly the midpoint epoch.
config = bh.MeanElementNumericalMethodConfig(
    period, bh.WindowAlignment.CENTERED, bh.WindowEdgeHandling.TRUNCATE
)
method_numerical = bh.MeanElementMethod.numerical(config)

out_epochs, out_states = bh.batch_state_koe_osc_to_mean(
    epochs, osc_states, method_numerical, bh.AngleFormat.DEGREES
)

print(f"Input osculating samples:                      {n_samples}")
print(f"Output mean samples after windowed averaging:  {len(out_epochs)}")

# Only the exact window-center epoch survives TRUNCATE edge handling here,
# since the averaging window spans the full data range.
recovered = out_states[0]
original = np.array([a, e, i, raan, argp, mean_anomalies[n_samples // 2]])

print(
    f"\nRecovered mean state: a={recovered[0]:.3f} m, e={recovered[1]:.6f}, "
    f"i={recovered[2]:.3f} deg, raan={recovered[3]:.3f} deg, "
    f"argp={recovered[4]:.3f} deg, M={recovered[5]:.3f} deg"
)
print(
    f"Residual vs. synthesized mean state (a in m, angles in deg): {recovered - original}"
)
