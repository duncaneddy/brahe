"""
Ground Track Multiple Spacecraft Example

This script demonstrates how to plot ground tracks for multiple satellites simultaneously
with different colors and line styles.
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

# Define epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create three different LEO satellites with different orbits

# Satellite 1: Sun-synchronous orbit (polar, high inclination)
oe1 = np.array([bh.R_EARTH + 700e3, 0.001, 98.0, 0.0, 0.0, 0.0])
state1 = bh.state_osculating_to_cartesian(oe1, bh.AngleFormat.DEGREES)
prop1 = bh.KeplerianPropagator.from_eci(epoch, state1, 60.0).with_name("Sun-Sync")

# Satellite 2: Medium inclination orbit
oe2 = np.array(
    [
        bh.R_EARTH + 600e3,
        0.001,
        55.0,
        45.0,
        0.0,
        90.0,
    ]
)
state2 = bh.state_osculating_to_cartesian(oe2, bh.AngleFormat.DEGREES)
prop2 = bh.KeplerianPropagator.from_eci(epoch, state2, 60.0).with_name("Mid-Inc")

# Satellite 3: Equatorial orbit
oe3 = np.array(
    [
        bh.R_EARTH + 800e3,
        0.001,
        5.0,
        90.0,
        0.0,
        180.0,
    ]
)
state3 = bh.state_osculating_to_cartesian(oe3, bh.AngleFormat.DEGREES)
prop3 = bh.KeplerianPropagator.from_eci(epoch, state3, 60.0).with_name("Equatorial")

# Propagate all satellites for 2 orbits
duration = 2 * bh.orbital_period(oe1[0])

prop1.propagate_to(epoch + duration)
prop2.propagate_to(epoch + duration)
prop3.propagate_to(epoch + duration)

# Create ground track plot with all three satellites
fig = bh.plot_groundtrack(
    trajectories=[
        {"trajectory": prop1.trajectory, "color": "red", "line_width": 2},
        {"trajectory": prop2.trajectory, "color": "blue", "line_width": 2},
        {"trajectory": prop3.trajectory, "color": "green", "line_width": 2},
    ],
    basemap="natural_earth",
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
