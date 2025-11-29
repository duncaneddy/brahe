#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly"]
# FLAGS = ["CI-ONLY"]
# TIMEOUT = 300
# ///

"""
Visualizing a Walker Star Constellation

This example demonstrates how to:
1. Generate an Iridium-like 66:6:2 Walker Star constellation
2. Create Keplerian propagators for all satellites
3. Propagate each satellite for one orbital period
4. Visualize the constellation in 3D space

Walker Star spreads orbital planes over 180 degrees of RAAN (vs 360 for Delta),
providing enhanced polar coverage patterns.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import brahe as bh

bh.initialize_eop()
# --8<-- [end:preamble]

# Create epoch for constellation
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# --8<-- [start:create_constellation]
# Create an Iridium-like 66:6:2 Walker Star constellation
walker = bh.WalkerConstellationGenerator(
    t=66,
    p=6,
    f=2,
    semi_major_axis=bh.R_EARTH + 780e3,  # Iridium altitude
    eccentricity=0.0,
    inclination=86.4,  # Near-polar inclination
    argument_of_perigee=0.0,
    reference_raan=0.0,
    reference_mean_anomaly=0.0,
    epoch=epoch,
    angle_format=bh.AngleFormat.DEGREES,
    pattern=bh.WalkerPattern.STAR,  # 180 deg RAAN spread
).with_base_name("IRIDIUM")
# --8<-- [end:create_constellation]

print(f"Created {walker.total_satellites} satellite Walker Star constellation")
print(f"Orbital planes: {walker.num_planes}")
print(f"RAAN spacing: 180/{walker.num_planes} = {180 / walker.num_planes:.0f} degrees")

# --8<-- [start:propagate]
# Generate Keplerian propagators and propagate for one orbit
propagators = walker.as_keplerian_propagators(60.0)

# Propagate each satellite for one complete orbit
for prop in propagators:
    # Get semi-major axis from Keplerian elements [a, e, i, raan, argp, M]
    koe = prop.state_koe_osc(prop.initial_epoch, bh.AngleFormat.RADIANS)
    orbital_period = bh.orbital_period(koe[0])
    prop.propagate_to(prop.initial_epoch + orbital_period)
# --8<-- [end:propagate]

print(f"\nPropagated all {len(propagators)} satellites for one orbital period")

# --8<-- [start:visualization]
# Create interactive 3D plot with Earth texture
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
# --8<-- [end:visualization]
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
