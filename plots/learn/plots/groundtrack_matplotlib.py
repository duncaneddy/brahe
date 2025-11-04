"""
Ground Track Plotting Example - Matplotlib Backend

This script demonstrates how to create a ground track plot using the matplotlib backend.
It shows the ISS ground track with a ground station communication cone.
"""

import brahe as bh
import numpy as np
import matplotlib.pyplot as plt

# Initialize EOP data
bh.initialize_eop()

# ISS TLE for November 3, 2025
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"

# Create SGP4 propagator
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)

# Define ground station (Cape Canaveral)
lat = np.radians(28.3922)  # Latitude in radians
lon = np.radians(-80.6077)  # Longitude in radians
alt = 0.0  # Altitude in meters
station = bh.PointLocation(lat, lon, alt).with_name("Cape Canaveral")

# Define time range for one orbital period (~92 minutes for ISS)
epoch = prop.epoch
duration = 92.0 * 60.0  # seconds

# Generate trajectory by propagating
prop.propagate_to(epoch + duration)
traj = prop.trajectory

# Create ground track plot in light mode
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": traj, "color": "red"}],
    ground_stations=[{"stations": [station], "color": "blue", "alpha": 0.3}],
    gs_cone_altitude=420e3,  # ISS altitude
    gs_min_elevation=10.0,
    backend="matplotlib",
)

# Save light mode figure
fig.savefig(
    "docs/figures/plot_groundtrack_matplotlib_light.svg", dpi=300, bbox_inches="tight"
)
print(
    "Ground track plot (matplotlib, light mode) saved to: docs/figures/plot_groundtrack_matplotlib_light.svg"
)
plt.close(fig)

# Create ground track plot in dark mode
with plt.style.context("dark_background"):
    fig = bh.plot_groundtrack(
        trajectories=[{"trajectory": traj, "color": "red"}],
        ground_stations=[{"stations": [station], "color": "blue", "alpha": 0.3}],
        gs_cone_altitude=420e3,  # ISS altitude
        gs_min_elevation=10.0,
        backend="matplotlib",
    )

    # Set background color to match Plotly dark theme
    fig.patch.set_facecolor("#1c1e24")
    for ax in fig.get_axes():
        ax.set_facecolor("#1c1e24")

    # Save dark mode figure
    fig.savefig(
        "docs/figures/plot_groundtrack_matplotlib_dark.svg",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Ground track plot (matplotlib, dark mode) saved to: docs/figures/plot_groundtrack_matplotlib_dark.svg"
    )
    plt.close(fig)
