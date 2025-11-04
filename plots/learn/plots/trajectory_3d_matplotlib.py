"""
3D Trajectory Plotting Example - Matplotlib Backend

This script demonstrates how to create a 3D trajectory plot in the ECI frame
using the matplotlib backend. Shows the ISS orbit around Earth.
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

# Create 3D trajectory plot
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "ISS"}],
    units="km",
    show_earth=True,
    backend="matplotlib",
)

# Save figure
fig.savefig(
    "docs/figures/plot_trajectory_3d_matplotlib.png", dpi=150, bbox_inches="tight"
)
print(
    "3D trajectory plot (matplotlib) saved to: docs/figures/plot_trajectory_3d_matplotlib.png"
)
