#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly"]
# FLAGS = ["CI-ONLY"]
# TIMEOUT = 300
# ///

"""
Downloading TLE Data and Visualizing GPS Satellites

This example demonstrates how to:
1. Download TLE data from CelesTrak for the GPS Satellites
2. Create SGP4 propagators for all satellites
3. Propagate satellites to current epoch
4. Visualize the constellation in 3D space

The example shows the complete workflow from data download to visualization.
"""

import time
import brahe as bh

bh.initialize_eop()

# Download TLE data for all GPS satellites from CelesTrak
# The get_tles_as_propagators function:
#   - Downloads latest TLE data (cached for 6 hours)
#   - Parses each TLE into an SGP4 propagator
#   - Sets default propagation step size (60 seconds)
print("Downloading GPS TLEs from CelesTrak...")
start_time = time.time()
propagators = bh.datasets.celestrak.get_tles_as_propagators("gps-ops", 60.0)
elapsed = time.time() - start_time
print(
    f"Initialized propagators for {len(propagators)} GPS satellites in {elapsed:.2f} seconds."
)

ts = time.time()
# Propagate each satellite one orbit
for prop in propagators:
    orbital_period = bh.orbital_period(prop.semi_major_axis)
    prop.propagate_to(prop.epoch + orbital_period)
te = time.time() - ts
print(f"Propagated all satellites to one orbit in {te:.2f} seconds.")

# Create interactive 3D plot with Earth texture
print("\nCreating 3D visualization of satellites...")
ts = time.time()
fig = bh.plot_trajectory_3d(
    [
        {
            "trajectory": prop.trajectory,
            "mode": "markers",
            "size": 2,
            "label": prop.get_name(),
        }
        for prop in propagators
    ],
    units="km",
    show_earth=True,
    earth_texture="natural_earth_50m",
    backend="plotly",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=2.0,
)
te = time.time() - ts
print(f"Created base 3D plot in {te:.2f} seconds.")

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# ruff: noqa: E402
import os
import pathlib
import sys

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Save the figure as themed HTML
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
