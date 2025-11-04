"""
Keplerian Trajectory Plot Example - Matplotlib Backend

This script demonstrates how to plot Keplerian orbital elements (a, e, i, Ω, ω, ν) vs time
using the matplotlib backend.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# ISS TLE for November 3, 2025
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"

# Create SGP4 propagator
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)

# Define time range for one orbital period (~92 minutes for ISS)
epoch = prop.epoch
duration = 92.0 * 60.0  # seconds

# Generate trajectory by propagating
prop.propagate_to(epoch + duration)
traj = prop.trajectory

# Create Keplerian trajectory plot
fig = bh.plot_keplerian_trajectory(
    [{"trajectory": traj, "color": "green", "label": "ISS"}],
    sma_units="km",
    angle_units="deg",
    backend="matplotlib",
)

# Save figure
fig.savefig(
    "docs/figures/plot_keplerian_trajectory_matplotlib.png",
    dpi=150,
    bbox_inches="tight",
)
print(
    "Keplerian trajectory plot (matplotlib) saved to: docs/figures/plot_keplerian_trajectory_matplotlib.png"
)
