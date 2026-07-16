#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Mars Reconnaissance Orbiter (MRO) Science Orbit

This example demonstrates how to:
1. Set up a sun-synchronous, near-polar low Mars orbit (LMO) in Mars-centered
   elements
2. Propagate it with the full Mars force model (50x50 GMM-2B gravity,
   atmospheric drag, SRP, Sun third body)
3. Track how the orbital elements evolve over a multi-day propagation
4. Visualize the trajectory in 3D around a textured Mars

The Mars Reconnaissance Orbiter has flown a ~255 x 320 km sun-synchronous
orbit since 2006, imaging the surface at consistent local solar time on every
pass. Its 92.6 degree inclination is chosen so that the nodal precession
driven by Mars's oblateness (J2) matches the planet's mean motion around the
Sun, keeping the orbit plane's orientation relative to the Sun fixed.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import os
import pathlib
import sys

import numpy as np
import plotly.graph_objects as go

import brahe as bh

bh.initialize_eop()
bh.load_common_spice_kernels()
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:orbit_setup]
# MRO-like sun-synchronous science orbit: ~255 x 320 km, i = 92.6 deg,
# node placed for a mid-afternoon local solar time.
r_p = bh.R_MARS + 255e3
r_a = bh.R_MARS + 320e3
a = (r_p + r_a) / 2
e = (r_a - r_p) / (r_a + r_p)
oe = np.array([a, e, 92.6, 45.0, 270.0, 0.0])  # [m, -, deg, deg, deg, deg]
state0 = bh.state_koe_to_eci_for_body(oe, bh.GM_MARS, bh.AngleFormat.DEGREES)

epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
params = np.array([2180.0, 20.0, 2.2, 20.0, 1.3])  # mass, drag_area, Cd, srp_area, Cr

print(
    f"Periapsis radius: {r_p / 1e3:.1f} km (altitude: {(r_p - bh.R_MARS) / 1e3:.0f} km)"
)
print(
    f"Apoapsis radius:  {r_a / 1e3:.1f} km (altitude: {(r_a - bh.R_MARS) / 1e3:.0f} km)"
)
print(f"Semi-major axis: {a / 1e3:.1f} km, eccentricity: {e:.4f}")
# --8<-- [end:orbit_setup]

# --8<-- [start:propagation]
# mars_default(): 50x50 GMM-2B gravity, exponential atmospheric drag, SRP
# occulted by Mars, and Sun third-body perturbations (downloads the gravity
# model on first run).
force_config = bh.ForceModelConfig.mars_default()
prop = bh.NumericalOrbitPropagator(
    epoch, state0, bh.NumericalPropagationConfig.default(), force_config, params
)
duration = 2 * 86400.0
print(f"\nPropagating {duration / 86400.0:.0f} days...")
prop.propagate_to(epoch + duration)
print("  Complete!")
# --8<-- [end:propagation]

# --8<-- [start:element_history]
# state_bci returns the propagator's native state in the central body's
# body-centered inertial frame (MCI for a Mars-centered propagator).
# state_eci would instead always return an Earth-centered state, adding
# Mars's Earth-relative position, which is not what we want here.
dt = 120.0
epochs = [epoch + t for t in np.arange(0.0, duration, dt)]
sma_km, ecc, inc_deg = [], [], []
for epc in epochs:
    x = prop.state_bci(epc)
    koe = bh.state_eci_to_koe_for_body(x, bh.GM_MARS, bh.AngleFormat.DEGREES)
    sma_km.append(koe[0] / 1e3)
    ecc.append(koe[1])
    inc_deg.append(koe[2])
sma_km = np.array(sma_km)
ecc = np.array(ecc)
inc_deg = np.array(inc_deg)
times_hours = np.arange(0.0, duration, dt) / 3600.0
alt_p_km = sma_km * (1 - ecc) - bh.R_MARS / 1e3
# --8<-- [end:element_history]

# --8<-- [start:element_plot]
fig_elements = go.Figure()

fig_elements.add_trace(
    go.Scatter(
        x=times_hours.tolist(),
        y=sma_km.tolist(),
        mode="lines",
        line=dict(color="orange", width=2),
        name="Semi-major axis (km)",
        yaxis="y1",
    )
)
fig_elements.add_trace(
    go.Scatter(
        x=times_hours.tolist(),
        y=ecc.tolist(),
        mode="lines",
        line=dict(color="teal", width=2),
        name="Eccentricity",
        yaxis="y2",
    )
)
fig_elements.add_trace(
    go.Scatter(
        x=times_hours.tolist(),
        y=inc_deg.tolist(),
        mode="lines",
        line=dict(color="purple", width=2),
        name="Inclination (deg)",
        yaxis="y3",
    )
)

fig_elements.update_layout(
    title="MRO Osculating Element Evolution",
    xaxis=dict(title="Time (hours)", domain=[0.0, 0.85]),
    yaxis=dict(title="Semi-major axis (km)", tickfont=dict(color="orange")),
    yaxis2=dict(
        title="Eccentricity",
        tickfont=dict(color="teal"),
        overlaying="y",
        side="right",
    ),
    yaxis3=dict(
        title="Inclination (deg)",
        tickfont=dict(color="purple"),
        overlaying="y",
        side="right",
        position=1.0,
        anchor="free",
    ),
    height=500,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(orientation="h", y=-0.2),
)
# --8<-- [end:element_plot]

# --8<-- [start:plot_3d]
fig_3d = bh.plot_trajectory_3d(
    [{"trajectory": prop.trajectory, "color": "orange", "label": "MRO"}],
    central_body="mars",
    backend="plotly",
)
# --8<-- [end:plot_3d]

# Validation
print(f"\nMin periapsis altitude: {alt_p_km.min():.2f} km")
print(f"Mean inclination: {inc_deg.mean():.4f} deg (target: 92.6 deg)")

assert alt_p_km.min() > 0, "Orbit descended below the Mars surface"
# The osculating inclination oscillates by about 1.1 deg peak-to-peak over
# the 2-day propagation (50x50 gravity, drag, and SRP all perturb the
# instantaneous element beyond the mean value used to design the orbit), so
# the mean sits about 0.4 deg below the design inclination of 92.6 deg.
assert abs(inc_deg.mean() - 92.6) < 0.5, (
    f"Mean inclination drifted too far from 92.6 deg: {inc_deg.mean():.4f} deg"
)

print("\nExample validated successfully!")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save themed figures
light_path, dark_path = save_themed_html(
    fig_elements, OUTDIR / f"{SCRIPT_NAME}_elements"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(fig_3d, OUTDIR / f"{SCRIPT_NAME}_3d")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nMRO Mars Orbit Example Complete!")
