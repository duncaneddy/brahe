"""
Quick Start Example - Matplotlib Backend

This example demonstrates the basics of brahe plotting by creating a simple LEO orbit
and visualizing it in 3D with a static matplotlib plot.
"""

import brahe as bh
import numpy as np
import matplotlib.pyplot as plt

# Initialize EOP data
bh.initialize_eop()

# Create a simple LEO orbit
epoch = bh.Epoch.from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)

# Create propagator and generate trajectory
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
prop.propagate_to(epoch + bh.orbital_period(oe[0]))
traj = prop.trajectory

# Create a 3D plot in light mode
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO"}],
    show_earth=True,
    backend="matplotlib",
)

# Save light mode figure
fig.savefig(
    "docs/figures/plot_quickstart_example_matplotlib_light.svg",
    dpi=300,
    bbox_inches="tight",
)
print(
    "Quick start example plot (matplotlib, light mode) saved to: docs/figures/plot_quickstart_example_matplotlib_light.svg"
)
plt.close(fig)

# Create a 3D plot in dark mode
with plt.style.context("dark_background"):
    fig = bh.plot_trajectory_3d(
        [{"trajectory": traj, "color": "red", "label": "LEO"}],
        show_earth=True,
        backend="matplotlib",
    )

    # Set background color to match Plotly dark theme
    fig.patch.set_facecolor("#1c1e24")
    for ax in fig.get_axes():
        ax.set_facecolor("#1c1e24")

    # Save dark mode figure
    fig.savefig(
        "docs/figures/plot_quickstart_example_matplotlib_dark.svg",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "Quick start example plot (matplotlib, dark mode) saved to: docs/figures/plot_quickstart_example_matplotlib_dark.svg"
    )
    plt.close(fig)
