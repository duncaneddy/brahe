"""
Ground Track with NASA NEN Ground Stations Example

This script demonstrates plotting ground tracks with the NASA Near Earth Network (NEN)
ground stations, showing communication coverage at 550km altitude with 10° minimum elevation.
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
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Load NASA NEN ground stations
nen_stations = bh.datasets.groundstations.load("nasa nen")
print(f"Loaded {len(nen_stations)} NASA NEN stations")

# Create a LEO satellite at 550km altitude
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 550e3, 0.001, np.radians(51.6), 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("LEO Sat")

# Propagate for 2 orbits
duration = 2 * bh.orbital_period(oe[0])
prop.propagate_to(epoch + duration)

# Create ground track plot with NASA NEN stations
# The communication cones show regions where the satellite is visible above 10° elevation
fig = bh.plot_groundtrack(
    trajectories=[{"trajectory": prop.trajectory, "color": "red", "line_width": 2}],
    ground_stations=[{"stations": nen_stations, "color": "blue", "alpha": 0.3}],
    gs_cone_altitude=550e3,  # Satellite altitude in meters
    gs_min_elevation=10.0,  # Minimum elevation angle in degrees
    basemap="natural_earth",
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
