"""
Quick Start Example - Matplotlib Backend

This example demonstrates the basics of brahe plotting by creating a simple LEO orbit
and visualizing it in 3D with a static matplotlib plot.
"""

import brahe as bh
import numpy as np

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

# Create a 3D plot
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO"}],
    show_earth=True,
    backend="matplotlib",
)

# Save figure
fig.savefig(
    "docs/figures/plot_quickstart_example_matplotlib.png", dpi=150, bbox_inches="tight"
)
print(
    "Quick start example plot (matplotlib) saved to: docs/figures/plot_quickstart_example_matplotlib.png"
)
