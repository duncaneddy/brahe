"""
Access Elevation Plot Example - Matplotlib Backend

This script demonstrates how to create an elevation vs time plot using the matplotlib backend.
Shows satellite elevation angle during a ground station pass.
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

# Define time range (one day to capture multiple passes)
epoch = prop.epoch
duration = 24.0 * 3600.0  # 24 hours in seconds

# Compute access windows
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
accesses = bh.location_accesses([station], [prop], epoch, epoch + duration, constraint)

# Create elevation plots (light and dark mode)
if len(accesses) > 0:
    # Light mode
    fig = bh.plot_access_elevation(
        [{"access_window": accesses[0]}],  # Use first access window
        prop,  # Propagator for interpolation
        backend="matplotlib",
    )

    # Save light mode figure
    fig.savefig(
        "docs/figures/plot_access_elevation_matplotlib_light.svg",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Access elevation plot (matplotlib, light mode) saved to: docs/figures/plot_access_elevation_matplotlib_light.svg"
    )
    plt.close(fig)

    # Dark mode
    with plt.style.context("dark_background"):
        fig = bh.plot_access_elevation(
            [{"access_window": accesses[0]}],  # Use first access window
            prop,  # Propagator for interpolation
            backend="matplotlib",
        )

        # Set background color to match Plotly dark theme
        fig.patch.set_facecolor("#1c1e24")
        for ax in fig.get_axes():
            ax.set_facecolor("#1c1e24")

        # Save dark mode figure
        fig.savefig(
            "docs/figures/plot_access_elevation_matplotlib_dark.svg",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "Access elevation plot (matplotlib, dark mode) saved to: docs/figures/plot_access_elevation_matplotlib_dark.svg"
        )
        plt.close(fig)
else:
    print("No access windows found in the specified time range")
