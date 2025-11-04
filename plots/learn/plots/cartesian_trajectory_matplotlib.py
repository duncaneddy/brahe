"""
Cartesian Trajectory Plot Example - Matplotlib Backend

This script demonstrates how to plot Cartesian state elements (x, y, z, vx, vy, vz) vs time
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
duration = bh.orbital_period(prop.semi_major_axis)
print(f"Propagating from {epoch} for {duration} seconds.")

# Generate trajectory by propagating
prop.propagate_to(epoch + duration)
traj = prop.trajectory

# Create Cartesian trajectory plot in light mode
fig = bh.plot_cartesian_trajectory(
    [{"trajectory": traj, "color": "blue", "label": "ISS"}],
    position_units="km",
    velocity_units="km/s",
    backend="matplotlib",
    matplotlib_config={"dark_mode": False},
)

# Save light mode figure
fig.savefig(
    "docs/figures/plot_cartesian_trajectory_matplotlib_light.svg",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Cartesian trajectory plot (matplotlib, light mode) saved to: docs/figures/plot_cartesian_trajectory_matplotlib_light.svg"
)

# Create Cartesian trajectory plot in dark mode
fig = bh.plot_cartesian_trajectory(
    [{"trajectory": traj, "color": "blue", "label": "ISS"}],
    position_units="km",
    velocity_units="km/s",
    backend="matplotlib",
    matplotlib_config={"dark_mode": True},
)

# Set background color to match Plotly dark theme
fig.patch.set_facecolor("#1c1e24")
for ax in fig.get_axes():
    ax.set_facecolor("#1c1e24")

# Save dark mode figure
fig.savefig(
    "docs/figures/plot_cartesian_trajectory_matplotlib_dark.svg",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Cartesian trajectory plot (matplotlib, dark mode) saved to: docs/figures/plot_cartesian_trajectory_matplotlib_dark.svg"
)
