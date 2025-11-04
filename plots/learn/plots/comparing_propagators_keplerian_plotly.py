"""
Comparing Propagators (Keplerian) Example - Plotly Backend

This script demonstrates how to compare different propagators (Keplerian vs SGP4)
by plotting their Keplerian element trajectories side-by-side using the plotly backend for interactive visualization.
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
prop_sgp = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)
epoch = prop_sgp.epoch

# Create Keplerian propagator with same initial Cartesian state as SGP4
# This ensures both propagators store states in the same representation (Cartesian)
prop_kep = bh.KeplerianPropagator.from_eci(epoch, prop_sgp.state_eci(epoch), 60.0)

# Propagate both for 4 orbital periods to see differences
duration = 4 * bh.orbital_period(prop_sgp.semi_major_axis)

# Propagate both propagators
prop_kep.propagate_to(epoch + duration)
prop_sgp.propagate_to(epoch + duration)

# Get trajectories
traj_kep = prop_kep.trajectory
traj_sgp = prop_sgp.trajectory

# Create comparison plot using Keplerian elements with fixed angle limits
fig = bh.plot_keplerian_trajectory(
    [
        {"trajectory": prop_kep.trajectory, "color": "blue", "label": "Keplerian"},
        {"trajectory": prop_sgp.trajectory, "color": "red", "label": "SGP4"},
    ],
    sma_units="km",
    angle_units="deg",
    backend="plotly",
    plotly_config={"set_angle_ylim": True},
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
