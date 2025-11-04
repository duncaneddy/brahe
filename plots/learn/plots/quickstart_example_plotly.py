"""
Quick Start Example - Plotly Backend

This example demonstrates the basics of brahe plotting by creating a simple LEO orbit
and visualizing it in 3D with an interactive plotly plot.
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

# Configuration
SCRIPT_NAME = "plot_" + pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Create a simple LEO orbit
epoch = bh.Epoch.from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 0.0, 0.0, 0.0])

# Create propagator and generate trajectory
prop = bh.KeplerianPropagator.from_keplerian(epoch, oe, bh.AngleFormat.DEGREES, 5.0)
prop.propagate_to(epoch + bh.orbital_period(oe[0]))
traj = prop.trajectory

# Create an interactive 3D plot
fig = bh.plot_trajectory_3d(
    [{"trajectory": traj, "color": "red", "label": "LEO"}],
    show_earth=True,
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
