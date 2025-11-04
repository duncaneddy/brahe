"""
Comparing Propagators (Cartesian) Example - Matplotlib Backend

This script demonstrates how to compare different propagators (Keplerian vs SGP4)
by plotting their Cartesian state trajectories side-by-side using the matplotlib backend.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# ISS TLE for November 3, 2025
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"

# Create SGP4 propagator
prop_sgp = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)
epoch = prop_sgp.epoch

# Get initial Cartesian state from SGP4 propagator for Keplerian propagator
initial_state = prop_sgp.state_eci(epoch)

# Create Keplerian propagator with same initial state
prop_kep = bh.KeplerianPropagator.from_eci(epoch, initial_state, 60.0)

# Propagate both for 4 orbital periods to see differences
duration = 4 * bh.orbital_period(prop_sgp.semi_major_axis)
print(
    f"Propagating from {epoch} for {duration:.0f} seconds ({duration / 3600:.1f} hours)."
)

# Propagate both propagators
prop_kep.propagate_to(epoch + duration)
prop_sgp.propagate_to(epoch + duration)

# Get trajectories
traj_kep = prop_kep.trajectory
traj_sgp = prop_sgp.trajectory

print(f"Keplerian trajectory: {len(traj_kep)} states")
print(f"SGP4 trajectory: {len(traj_sgp)} states")

# Create comparison plot in light mode
fig = bh.plot_cartesian_trajectory(
    [
        {"trajectory": traj_kep, "color": "blue", "label": "Keplerian"},
        {"trajectory": traj_sgp, "color": "red", "label": "SGP4"},
    ],
    position_units="km",
    velocity_units="km/s",
    backend="matplotlib",
    matplotlib_config={"dark_mode": False},
)

# Save light mode figure
fig.savefig(
    "docs/figures/comparing_propagators_cartesian_matplotlib_light.svg",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Comparing propagators (Cartesian) plot (matplotlib, light mode) saved to: docs/figures/comparing_propagators_cartesian_matplotlib_light.svg"
)

# Create comparison plot in dark mode
fig = bh.plot_cartesian_trajectory(
    [
        {"trajectory": traj_kep, "color": "blue", "label": "Keplerian"},
        {"trajectory": traj_sgp, "color": "red", "label": "SGP4"},
    ],
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
    "docs/figures/comparing_propagators_cartesian_matplotlib_dark.svg",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Comparing propagators (Cartesian) plot (matplotlib, dark mode) saved to: docs/figures/comparing_propagators_cartesian_matplotlib_dark.svg"
)
