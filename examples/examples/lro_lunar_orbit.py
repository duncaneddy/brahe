#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Lunar Reconnaissance Orbiter (LRO) Science Orbit

This example demonstrates how to:
1. Set up a frozen, near-polar low lunar orbit (LLO) in a lunar-equatorial
   basis and rotate it into the propagator's frame
2. Propagate it with the full lunar force model (50x50 GRGM660PRIM gravity,
   SRP, Earth/Sun third bodies) and compare against a point-mass Moon
3. Quantify how lunar gravity anomalies ("mascons") perturb a low lunar orbit
4. Visualize the trajectory in 3D around a textured Moon

This example uses an LRO-like ~30 x 180 km class polar orbit. The Lunar
Reconnaissance Orbiter has flown a polar, near-frozen low lunar science
orbit since 2009, mapping the Moon at high resolution. This example's orbit
is a "frozen orbit": the argument of perilune is chosen so that the
long-period perturbation from the Moon's lumpy gravity field averages out,
keeping the eccentricity and perilune altitude nearly constant over many
orbits rather than drifting or decaying. With the argument of perilune at
270 deg, perilune sits over the Moon's south pole, so this frozen geometry
is defined relative to the lunar spin pole.

The propagator integrates in LCI, whose axes are ICRF-aligned: the LCI z-axis
is the ICRF pole, which sits ~22 deg from the Moon's spin pole. An inclination
passed straight to state_koe_to_eci would therefore be measured against the
wrong pole. Instead, state_koe_to_inertial_for_body references the elements to
the Moon's mean equator at J2000 (the plane normal to the lunar IAU pole) and
returns the state directly in LCI, so the 85.2 deg polar inclination and the
south-pole perilune are referenced to the Moon's equator as intended.
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
# LRO-like frozen science orbit: ~30 x 180 km polar orbit. Perilune is
# kept over the southern hemisphere (argument of perilune 270 deg).
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
params = np.array([1000.0, 0.0, 0.0, 10.0, 1.3])  # mass, -, -, srp_area, Cr

r_p = bh.R_MOON + 30e3
r_a = bh.R_MOON + 180e3
a = (r_p + r_a) / 2
e = (r_a - r_p) / (r_a + r_p)
oe = np.array([a, e, 85.2, 0.0, 270.0, 0.0])  # [m, -, deg, deg, deg, deg]

# state_koe_to_inertial_for_body references the elements to the Moon's mean
# equator at J2000 (the plane normal to the lunar IAU pole) and returns the
# state directly in the LCI frame the propagator integrates in, so the 85.2 deg
# inclination and south-pole perilune are measured against the Moon's equator
# rather than the ICRF pole.
state0 = bh.state_koe_to_inertial_for_body(
    oe, bh.CentralBody.Moon, bh.AngleFormat.DEGREES
)

# Lunar mean pole (ICRF) at J2000, used below to confirm the orbit geometry:
# the third row of the ICRF -> lunar body-fixed (IAU) rotation is the spin pole.
j2000 = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.TDB)
moon_pole = np.asarray(bh.rotation_icrf_to_body_fixed_iau(301, j2000))[2, :]

print(
    f"Perilune radius: {r_p / 1e3:.1f} km (altitude: {(r_p - bh.R_MOON) / 1e3:.0f} km)"
)
print(
    f"Apolune radius:  {r_a / 1e3:.1f} km (altitude: {(r_a - bh.R_MOON) / 1e3:.0f} km)"
)
print(f"Semi-major axis: {a / 1e3:.1f} km, eccentricity: {e:.4f}")
# --8<-- [end:orbit_setup]

# --8<-- [start:propagation]
# Full lunar force model: 50x50 GRGM660PRIM gravity, SRP, Earth+Sun third
# bodies (downloads the gravity model and kernels on first run).
prop_full = bh.NumericalOrbitPropagator(
    epoch,
    state0,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.lunar_default(),
    params,
)
# Point-mass comparison: same orbit, Moon treated as a point mass.
prop_pm = bh.NumericalOrbitPropagator(
    epoch,
    state0,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.for_body(
        bh.CentralBody.Moon, bh.GravityConfiguration.point_mass()
    ),
    params,
)
duration = 7 * 86400.0
print(f"\nPropagating {duration / 86400.0:.0f} days (full model vs. point mass)...")
prop_full.propagate_to(epoch + duration)
prop_pm.propagate_to(epoch + duration)
print("  Complete!")
# --8<-- [end:propagation]

# --8<-- [start:perilune_history]
# state_bci returns the state in the propagator's native Moon-centered
# inertial (LCI) frame. state_eci would instead add the Moon's Earth-relative
# position, which is not what we want when measuring altitude above the
# lunar surface.
dt = 120.0
epochs = [epoch + t for t in np.arange(0.0, duration, dt)]
r_full = np.array([np.linalg.norm(prop_full.state_bci(e)[:3]) for e in epochs])
r_pm = np.array([np.linalg.norm(prop_pm.state_bci(e)[:3]) for e in epochs])
alt_full_km = (r_full - bh.R_MOON) / 1e3
alt_pm_km = (r_pm - bh.R_MOON) / 1e3
times_days = np.arange(0.0, duration, dt) / 86400.0
# --8<-- [end:perilune_history]

# --8<-- [start:altitude_plot]
# Full minus point-mass altitude: overlaying the two raw traces over 7 days
# (~90 orbits) is an unreadable smear, so plot their difference instead.
diff_km = alt_full_km - alt_pm_km

fig_altitude = go.Figure()

fig_altitude.add_trace(
    go.Scatter(
        x=times_days.tolist(),
        y=diff_km.tolist(),
        mode="lines",
        line=dict(color="red", width=2),
        name="Full − point mass",
    )
)

fig_altitude.update_layout(
    title="Modeled Altitude Difference: 50x50 Gravity Field minus Point Mass",
    xaxis_title="Time (days)",
    yaxis_title="Altitude difference (km)",
    height=500,
    margin=dict(l=60, r=40, t=60, b=60),
)
# --8<-- [end:altitude_plot]

# --8<-- [start:plot_3d]
fig_3d = bh.plot_trajectory_3d(
    [{"trajectory": prop_full.trajectory, "color": "red", "label": "LRO"}],
    time_range=(epoch + (duration - 12 * 3600.0), epoch + duration),
    central_body="moon",
    backend="plotly",
)
# --8<-- [end:plot_3d]

# Validation
divergence_km = np.abs(alt_full_km - alt_pm_km).max()

# Inclination of the initial state relative to the Moon's spin pole confirms
# the orbit plane was built about the lunar equator, not the ICRF pole.
h0 = np.cross(state0[:3], state0[3:])
h0 /= np.linalg.norm(h0)
inc0_deg = np.degrees(np.arccos(np.clip(np.dot(h0, moon_pole), -1.0, 1.0)))
print(f"\nInitial inclination (rel. Moon pole): {inc0_deg:.4f} deg (target: 85.2 deg)")
print(f"Min perilune altitude (full model): {alt_full_km.min():.2f} km")
print(f"Max altitude divergence (full vs. point mass): {divergence_km:.3f} km")

assert abs(inc0_deg - 85.2) < 0.1, (
    f"Initial inclination not at design value rel. Moon pole: {inc0_deg:.4f} deg"
)
assert alt_full_km.min() > 0, "Orbit impacted the lunar surface"
assert divergence_km > 12.0, (
    f"Full and point-mass solutions did not diverge as expected: {divergence_km:.3f} km"
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
    fig_altitude, OUTDIR / f"{SCRIPT_NAME}_altitude"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(fig_3d, OUTDIR / f"{SCRIPT_NAME}_3d")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nLRO Lunar Orbit Example Complete!")
