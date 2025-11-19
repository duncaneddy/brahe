#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly"]
# FLAGS = ["IGNORE"]
# ///

"""
Downloading TLE Data and Visualizing Starlink Constellation

This example demonstrates how to:
1. Download TLE data from CelesTrak for the Starlink constellation
2. Create SGP4 propagators for all satellites
3. Propagate satellites to current epoch
4. Visualize the constellation in 3D space

The example shows the complete workflow from data download to visualization.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import time
import brahe as bh

bh.initialize_eop()
# --8<-- [end:preamble]

# Download TLE data for all Starlink satellites from CelesTrak
# The get_tles_as_propagators function:
#   - Downloads latest TLE data (cached for 6 hours)
#   - Parses each TLE into an SGP4 propagator
#   - Sets default propagation step size (60 seconds)
print("Downloading Starlink TLEs from CelesTrak...")
start_time = time.time()
# --8<-- [start:download_starlink]
propagators = bh.datasets.celestrak.get_tles_as_propagators("starlink", 60.0)

# Filter out any re-enerting spacecraft with < 350 km semi-major axis
# This can sometimes cause numerical issues with the propagator for very low orbit
# when eccentricity becomes negative.
propagators = [
    prop for prop in propagators if prop.semi_major_axis >= (bh.R_EARTH + 350.0e3)
]

# --8<-- [end:download_starlink]
elapsed = time.time() - start_time
print(
    f"Initialized propagators for {len(propagators)} Starlink satellites in {elapsed:.2f} seconds."
)

# --8<-- [start:inspect_data]
# Inspect the first satellite
first_sat = propagators[0]
print(f"\nFirst satellite: {first_sat.get_name()}")
print(f"Epoch: {first_sat.epoch}")
print(f"Semi-major axis: {first_sat.semi_major_axis / 1000:.1f} km")
print(f"Inclination: {first_sat.inclination:.1f} degrees")
print(f"Eccentricity: {first_sat.eccentricity:.6f}")
# --8<-- [end:inspect_data]


# Create interactive 3D plot with Earth texture
print("\nCreating 3D visualization of satellites...")
ts = time.time()
# --8<-- [start:orbit_visualization]
fig = bh.plot_trajectory_3d(
    [],  # Empty trajectory list; we'll add markers for each satellite
    units="km",
    show_earth=True,
    earth_texture="natural_earth_50m",
    backend="plotly",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=3.0,
    sphere_resolution_lon=600,  # Reduce sphere texture resolution for performance
    sphere_resolution_lat=300,
)
# --8<-- [end:orbit_visualization]
te = time.time() - ts
print(f"Created base 3D plot in {te:.2f} seconds.")

ts = time.time()
# --8<-- [start:add_satellite_markers]
# Get the current time for display
epc = bh.Epoch.now()

# For each satellite, add a marker at the current position
for prop in propagators:
    state = prop.state_eci(epc)
    fig.add_scatter3d(
        x=[state[0] / 1000],
        y=[state[1] / 1000],
        z=[state[2] / 1000],
        mode="markers",
        marker=dict(size=2, color="white"),
        name=prop.get_name(),
        showlegend=False,
    )
# --8<-- [end:add_satellite_markers]
te = time.time() - ts
print(f"Added satellite markers in {te:.2f} seconds.")
# --8<-- [end:all]

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
