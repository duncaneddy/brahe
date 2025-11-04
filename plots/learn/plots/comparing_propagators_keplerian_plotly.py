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
sgp_elements = prop_sgp.get_elements(angle_format=bh.AngleFormat.DEGREES)
print(f"SGP4 Propagator initialized at epoch {epoch}.")
print(
    f"SGP4 Elements: a={sgp_elements[0] / 1000:.1f} km, "
    f"e={sgp_elements[1]:.6f}, "
    f"i={sgp_elements[2]:.3f} deg, "
    f"RAAN={sgp_elements[3]:.3f} deg, "
    f"arg_periapsis={sgp_elements[4]:.3f} deg, "
    f"mean_anomaly={sgp_elements[5]:.3f} deg"
)

# Compute initial state from SGP4 Propagator as osculating state
sgp_oe_state = prop_sgp.state_as_osculating_elements(
    epoch, angle_format=bh.AngleFormat.DEGREES
)
print(
    f"SGP4 Osculating Elements at epoch: a={sgp_oe_state[0] / 1000:.1f} km, "
    f"e={sgp_oe_state[1]:.6f}, "
    f"i={sgp_oe_state[2]:.3f} deg, "
    f"RAAN={sgp_oe_state[3]:.3f} deg, "
    f"arg_periapsis={sgp_oe_state[4]:.3f} deg, "
    f"mean_anomaly={sgp_oe_state[5]:.3f} deg"
)

sgp_cart_state = prop_sgp.state_eci(epoch)
print(
    f"SGP4 ECI State at epoch: x={sgp_cart_state[0] / 1000:.1f} km, "
    f"y={sgp_cart_state[1] / 1000:.1f} km, "
    f"z={sgp_cart_state[2] / 1000:.1f} km, "
    f"vx={sgp_cart_state[3]:.3f} m/s, "
    f"vy={sgp_cart_state[4]:.3f} m/s, "
    f"vz={sgp_cart_state[5]:.3f} m/s"
)

# Create Keplerian propagator with same initial Cartesian state as SGP4
# This ensures both propagators store states in the same representation (Cartesian)
prop_kep = bh.KeplerianPropagator.from_eci(epoch, sgp_cart_state, 60.0)

# Propagate both for 4 orbital periods to see differences
duration = 4 * bh.orbital_period(prop_sgp.semi_major_axis)
print(
    f"Propagating from {epoch} for {duration:.0f} seconds ({duration / 3600:.1f} hours)."
)

# Propagate both propagators
prop_kep.propagate_to(epoch + duration)
prop_sgp.propagate_to(epoch + duration)

# Get trajectories
traj_kep = prop_kep.trajectory
traj_sgp = prop_sgp.trajectory

# print first 3 states for verification
for i in range(3):
    t_kep, state_kep = traj_kep.get(i)
    t_sgp, state_sgp = traj_sgp.get(i)
    print(
        f"Keplerian Traj t={t_kep} [{state_kep[0] / 1e3:.1f}, {state_kep[1] / 1e3:.1f}, {state_kep[2] / 1e3:.1f}] km, {state_kep[3]:.1f}, {state_kep[4]:.1f}, {state_kep[5]:.1f} m/s"
    )
    print(
        f"SGP4 Traj      t={t_sgp} [{state_sgp[0] / 1e3:.1f}, {state_sgp[1] / 1e3:.1f}, {state_sgp[2] / 1e3:.1f}] km, {state_sgp[3]:.1f}, {state_sgp[4]:.1f}, {state_sgp[5]:.1f} m/s"
    )

# Create comparison plot using Keplerian elements
fig = bh.plot_keplerian_trajectory(
    [
        {"trajectory": traj_kep, "color": "blue", "label": "Keplerian"},
        {"trajectory": traj_sgp, "color": "red", "label": "SGP4"},
    ],
    sma_units="km",
    angle_units="deg",
    backend="plotly",
)

# Save themed HTML files
light_path, dark_path = save_themed_html(fig, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
