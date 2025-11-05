"""
Ground Track Orbit Configuration Example

This script demonstrates how to configure the number of orbits displayed in a ground track plot
using the track_length and track_units parameters.
"""

import os
import pathlib
import sys
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# ISS TLE for November 3, 2025
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"

# Create SGP4 propagator
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)
epoch = prop.epoch

# Propagate for 5 complete orbits (~460 minutes for ISS)
duration = 5 * 92.0 * 60.0  # 5 orbits × 92 minutes/orbit × 60 seconds/minute
prop.propagate_to(epoch + duration)
traj = prop.trajectory

# Create ground track plots showing different numbers of orbits
# This demonstrates using track_length to control how much of the trajectory is displayed

fig = bh.plot_groundtrack(
    trajectories=[
        # Show only the last 1 orbit (most recent)
        {
            "trajectory": traj,
            "color": "blue",
            "line_width": 2,
            "track_length": 1,
            "track_units": "orbits",
        },
        # Show the last 3 orbits
        {
            "trajectory": traj,
            "color": "red",
            "line_width": 2,
            "track_length": 3,
            "track_units": "orbits",
        },
        # Show all 5 orbits
        {
            "trajectory": traj,
            "color": "green",
            "line_width": 1,
            "track_length": 5,
            "track_units": "orbits",
        },
    ],
    basemap="natural_earth",
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
