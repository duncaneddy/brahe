"""
3D Trajectory Plotting Example - Matplotlib Backend

This script demonstrates how to create a 3D trajectory plot in the ECI frame
using the matplotlib backend. Shows the ISS orbit around Earth with different
texture options for Earth visualization.
"""

import numpy as np
import brahe as bh
import matplotlib.pyplot as plt

# Initialize EOP data
bh.initialize_eop()

# Define epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# ISS-like orbit (LEO, 51.6° inclination, ~400 km altitude)
oe_iss = np.array(
    [
        bh.R_EARTH + 420e3,  # Semi-major axis (m)
        0.0005,  # Eccentricity
        np.radians(51.6),  # Inclination
        np.radians(45.0),  # RAAN
        np.radians(30.0),  # Argument of perigee
        np.radians(0.0),  # Mean anomaly
    ]
)
state_iss = bh.state_osculating_to_cartesian(oe_iss, bh.AngleFormat.RADIANS)
prop_iss = bh.KeplerianPropagator.from_eci(epoch, state_iss, 60.0)

# Polar orbit (Sun-synchronous-like, ~550 km altitude)
oe_polar = np.array(
    [
        bh.R_EARTH + 550e3,  # Semi-major axis (m)
        0.001,  # Eccentricity
        np.radians(97.8),  # Inclination (near-polar)
        np.radians(180.0),  # RAAN
        np.radians(60.0),  # Argument of perigee
        np.radians(0.0),  # Mean anomaly
    ]
)
state_polar = bh.state_osculating_to_cartesian(oe_polar, bh.AngleFormat.RADIANS)
prop_polar = bh.KeplerianPropagator.from_eci(epoch, state_polar, 60.0)

# Define time range - one orbital period of the lower orbit
# Generate trajectories
prop_iss.propagate_to(epoch + bh.orbital_period(oe_iss[0]))
traj_iss = prop_iss.trajectory

prop_polar.propagate_to(epoch + bh.orbital_period(oe_polar[0]))
traj_polar = prop_polar.trajectory

# Create 3D trajectory plot in light mode with matplotlib
fig = bh.plot_trajectory_3d(
    [
        {"trajectory": traj_iss, "color": "red", "label": "LEO 51.6° (~420 km)"},
        {"trajectory": traj_polar, "color": "cyan", "label": "Polar 97.8° (~550 km)"},
    ],
    units="km",
    show_earth=True,
    backend="matplotlib",
)

# Save light mode figure
fig.savefig(
    "docs/figures/plot_trajectory_3d_matplotlib_light.svg", dpi=300, bbox_inches="tight"
)
print(
    "3D trajectory plot (matplotlib) saved to: docs/figures/plot_trajectory_3d_matplotlib_light.svg"
)
plt.close(fig)

# Create 3D trajectory plot in dark mode
with plt.style.context("dark_background"):
    fig = bh.plot_trajectory_3d(
        [
            {"trajectory": traj_iss, "color": "red", "label": "LEO 51.6° (~420 km)"},
            {
                "trajectory": traj_polar,
                "color": "cyan",
                "label": "Polar 97.8° (~550 km)",
            },
        ],
        units="km",
        show_earth=True,
        backend="matplotlib",
    )

    # Set background color to match Plotly dark theme
    fig.patch.set_facecolor("#1c1e24")
    for ax in fig.get_axes():
        ax.set_facecolor("#1c1e24")

    # Save dark mode figure
    fig.savefig(
        "docs/figures/plot_trajectory_3d_matplotlib_dark.svg",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "3D trajectory plot (matplotlib, dark mode, Blue Marble) saved to: docs/figures/plot_trajectory_3d_matplotlib_dark.svg"
    )
    plt.close(fig)

print("\nAll matplotlib figures generated successfully!")
