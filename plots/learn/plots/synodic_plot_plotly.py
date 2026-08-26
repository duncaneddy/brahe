#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Synodic Frame Trajectory Plotting Example - Plotly Backend

This script demonstrates how to create an interactive 3D trajectory plot in
the Earth-Moon Rotating (EMR) frame using the plotly backend. Shows a LEO
orbit alongside the Earth and Moon, both fixed on the synodic x-axis.
"""

import os
import pathlib
import sys

import numpy as np

import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data and the DE440s ephemeris the EMR transform needs
bh.initialize_eop()
bh.load_common_spice_kernels()

# LEO orbit (51.6 deg inclination, ~500 km altitude)
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
prop.propagate_to(epoch + bh.orbital_period(oe[0]))

# Create the Earth-Moon Rotating 3D trajectory plot
fig = bh.plot_earth_moon_rotating_3d(
    [{"trajectory": prop.trajectory, "color": "red", "label": "LEO"}],
    backend="plotly",
)

# Save as interactive HTML (small file size, no large textures)
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nSynodic plot generated successfully!")
