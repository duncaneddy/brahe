#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
LEO to GEO Hohmann Transfer

This example demonstrates how to:
1. Calculate Hohmann transfer parameters (delta-v, transfer time)
2. Use NumericalOrbitPropagator with event callbacks for impulsive maneuvers
3. Execute a two-burn orbit transfer from LEO (400 km) to GEO (35,786 km)
4. Visualize orbit geometry, altitude profile, and velocity changes

The example shows the classic Hohmann transfer - the most fuel-efficient two-impulse
transfer between coplanar circular orbits.
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
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:orbital_parameters]
# LEO starting orbit: 400 km circular altitude
r_leo = bh.R_EARTH + 400e3  # meters

# GEO target orbit: 35,786 km altitude (geostationary)
r_geo = bh.R_EARTH + 35786e3  # meters

print(
    f"LEO radius: {r_leo / 1e3:.1f} km (altitude: {(r_leo - bh.R_EARTH) / 1e3:.0f} km)"
)
print(
    f"GEO radius: {r_geo / 1e3:.1f} km (altitude: {(r_geo - bh.R_EARTH) / 1e3:.0f} km)"
)
# --8<-- [end:orbital_parameters]

# --8<-- [start:hohmann_calculations]
# Circular orbit velocities (vis-viva equation for circular orbit: v = sqrt(mu/r))
v_leo = np.sqrt(bh.GM_EARTH / r_leo)
v_geo = np.sqrt(bh.GM_EARTH / r_geo)

# Transfer orbit parameters
# Semi-major axis is the average of the two radii
a_transfer = (r_leo + r_geo) / 2

# Eccentricity of transfer ellipse
e_transfer = (r_geo - r_leo) / (r_geo + r_leo)

# Velocities on transfer orbit using vis-viva equation: v^2 = mu(2/r - 1/a)
v_perigee_transfer = np.sqrt(bh.GM_EARTH * (2 / r_leo - 1 / a_transfer))
v_apogee_transfer = np.sqrt(bh.GM_EARTH * (2 / r_geo - 1 / a_transfer))

# Delta-v magnitudes
dv1 = v_perigee_transfer - v_leo  # First burn: prograde at perigee (LEO)
dv2 = v_geo - v_apogee_transfer  # Second burn: prograde at apogee (GEO)

# Transfer time: half the period of the transfer ellipse
transfer_time = np.pi * np.sqrt(a_transfer**3 / bh.GM_EARTH)

print("\nHohmann Transfer Parameters:")
print(f"  Transfer semi-major axis: {a_transfer / 1e3:.1f} km")
print(f"  Transfer eccentricity:    {e_transfer:.4f}")
print(f"  LEO circular velocity:    {v_leo / 1e3:.3f} km/s")
print(f"  GEO circular velocity:    {v_geo / 1e3:.3f} km/s")
print(f"  First burn (perigee):     {dv1 / 1e3:.3f} km/s")
print(f"  Second burn (apogee):     {dv2 / 1e3:.3f} km/s")
print(f"  Total delta-v:            {(dv1 + dv2) / 1e3:.3f} km/s")
print(f"  Transfer time:            {transfer_time / 3600:.2f} hours")
# --8<-- [end:hohmann_calculations]

# --8<-- [start:initial_state]
# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial state: circular LEO orbit (at perigee of transfer)
# Keplerian elements: [a, e, i, raan, argp, M]
oe_initial = np.array([r_leo, 0.0001, 0.0, 0.0, 0.0, 0.0])
state_initial = bh.state_koe_to_eci(oe_initial, bh.AngleFormat.DEGREES)

print("\nInitial State (ECI):")
print(
    f"  Position: [{state_initial[0] / 1e3:.1f}, {state_initial[1] / 1e3:.1f}, {state_initial[2] / 1e3:.1f}] km"
)
print(
    f"  Velocity: [{state_initial[3] / 1e3:.3f}, {state_initial[4] / 1e3:.3f}, {state_initial[5] / 1e3:.3f}] km/s"
)
# --8<-- [end:initial_state]

# --8<-- [start:event_callbacks]
# Define event callbacks for impulsive maneuvers
# Each callback receives the event epoch and current state,
# and returns (new_state, EventAction)


def first_burn_callback(event_epoch, event_state):
    """Apply first delta-v at departure (prograde burn at perigee)."""
    new_state = event_state.copy()
    # Add delta-v in velocity direction (prograde)
    v = event_state[3:6]
    v_hat = v / np.linalg.norm(v)
    new_state[3:6] += dv1 * v_hat
    print(f"First burn applied: dv = {dv1 / 1e3:.3f} km/s (prograde)")
    return (new_state, bh.EventAction.CONTINUE)


def second_burn_callback(event_epoch, event_state):
    """Apply second delta-v at arrival (prograde burn at apogee)."""
    new_state = event_state.copy()
    v = event_state[3:6]
    v_hat = v / np.linalg.norm(v)
    new_state[3:6] += dv2 * v_hat
    print(f"Second burn applied: dv = {dv2 / 1e3:.3f} km/s (prograde)")
    return (new_state, bh.EventAction.CONTINUE)


# --8<-- [end:event_callbacks]

# --8<-- [start:single_propagator]
# Create a single propagator with event-based maneuvers
# This is cleaner than multi-stage propagation for simple burns

# Timing
burn1_time_s = 1.0  # First burn shortly after start
burn2_time_s = burn1_time_s + transfer_time  # Second burn at apogee
geo_period = bh.orbital_period(r_geo)
total_time = burn2_time_s + geo_period  # Continue for one GEO orbit

print("\nPropagation Timeline:")
print(f"  Burn 1 at t = {burn1_time_s:.1f} s")
print(f"  Burn 2 at t = {burn2_time_s / 3600:.2f} hours")
print(f"  Total simulation: {total_time / 3600:.2f} hours")

# Create the propagator with two-body dynamics
prop = bh.NumericalOrbitPropagator(
    epoch,
    state_initial,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# Add time-based events for the burns
event1 = bh.TimeEvent(epoch + burn1_time_s, "First Burn").with_callback(
    first_burn_callback
)
event2 = bh.TimeEvent(epoch + burn2_time_s, "Second Burn").with_callback(
    second_burn_callback
)

prop.add_event_detector(event1)
prop.add_event_detector(event2)

# Propagate through both burns plus one GEO orbit
print("\nPropagating...")
prop.propagate_to(epoch + total_time)
print("  Complete!")

# Verify final orbit
final_koe = prop.state_koe(prop.current_epoch, bh.AngleFormat.DEGREES)
final_altitude = final_koe[0] - bh.R_EARTH
print("\nFinal GEO Orbit:")
print(f"  Semi-major axis: {final_koe[0] / 1e3:.1f} km")
print(f"  Altitude:        {final_altitude / 1e3:.1f} km (target: 35786 km)")
print(f"  Eccentricity:    {final_koe[1]:.6f}")
# --8<-- [end:single_propagator]

# --8<-- [start:sample_trajectory]
# Sample trajectory data for plotting
# The single propagator stores the complete trajectory with maneuvers
times_hours = []
altitudes_km = []
velocities_km_s = []

dt = 60.0  # 1-minute sampling
t = 0.0

while t <= total_time:
    current_epoch = epoch + t
    state = prop.state_eci(current_epoch)

    r_mag = np.linalg.norm(state[:3])
    v_mag = np.linalg.norm(state[3:6])

    times_hours.append(t / 3600.0)
    altitudes_km.append((r_mag - bh.R_EARTH) / 1e3)
    velocities_km_s.append(v_mag / 1e3)

    t += dt

print(f"\nSampled {len(times_hours)} trajectory points")
# --8<-- [end:sample_trajectory]

# --8<-- [start:orbit_geometry_plot]
# Create 2D top-down view of orbit geometry
theta = np.linspace(0, 2 * np.pi, 200)

# Earth circle
earth_x = bh.R_EARTH * np.cos(theta) / 1e3
earth_y = bh.R_EARTH * np.sin(theta) / 1e3

# LEO orbit circle
leo_x = r_leo * np.cos(theta) / 1e3
leo_y = r_leo * np.sin(theta) / 1e3

# GEO orbit circle
geo_x = r_geo * np.cos(theta) / 1e3
geo_y = r_geo * np.sin(theta) / 1e3

# Transfer ellipse (only upper half: theta from 0 to pi)
theta_transfer = np.linspace(0, np.pi, 100)
p_transfer = a_transfer * (1 - e_transfer**2)
r_transfer = p_transfer / (1 + e_transfer * np.cos(theta_transfer))
transfer_x = r_transfer * np.cos(theta_transfer) / 1e3
transfer_y = r_transfer * np.sin(theta_transfer) / 1e3

fig_geometry = go.Figure()

# Earth (filled)
fig_geometry.add_trace(
    go.Scatter(
        x=earth_x.tolist(),
        y=earth_y.tolist(),
        fill="toself",
        fillcolor="lightblue",
        line=dict(color="steelblue", width=1),
        name="Earth",
        hoverinfo="name",
    )
)

# LEO orbit
fig_geometry.add_trace(
    go.Scatter(
        x=leo_x.tolist(),
        y=leo_y.tolist(),
        mode="lines",
        line=dict(color="blue", width=2, dash="dash"),
        name=f"LEO ({(r_leo - bh.R_EARTH) / 1e3:.0f} km)",
    )
)

# GEO orbit
fig_geometry.add_trace(
    go.Scatter(
        x=geo_x.tolist(),
        y=geo_y.tolist(),
        mode="lines",
        line=dict(color="green", width=2, dash="dash"),
        name=f"GEO ({(r_geo - bh.R_EARTH) / 1e3:.0f} km)",
    )
)

# Transfer ellipse
fig_geometry.add_trace(
    go.Scatter(
        x=transfer_x.tolist(),
        y=transfer_y.tolist(),
        mode="lines",
        line=dict(color="red", width=3),
        name="Transfer Orbit",
    )
)

# Burn 1 marker (at LEO, right side)
fig_geometry.add_trace(
    go.Scatter(
        x=[r_leo / 1e3],
        y=[0],
        mode="markers+text",
        marker=dict(size=15, color="red", symbol="star"),
        text=[f"Burn 1<br>{dv1 / 1e3:.2f} km/s"],
        textposition="bottom right",
        textfont=dict(size=10),
        name="Burn 1",
        showlegend=False,
    )
)

# Burn 2 marker (at GEO, left side - apogee)
fig_geometry.add_trace(
    go.Scatter(
        x=[-r_geo / 1e3],
        y=[0],
        mode="markers+text",
        marker=dict(size=15, color="red", symbol="star"),
        text=[f"Burn 2<br>{dv2 / 1e3:.2f} km/s"],
        textposition="top left",
        textfont=dict(size=10),
        name="Burn 2",
        showlegend=False,
    )
)

fig_geometry.update_layout(
    title="LEO to GEO Hohmann Transfer Geometry",
    xaxis_title="X (km)",
    yaxis_title="Y (km)",
    yaxis_scaleanchor="x",
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
)
# --8<-- [end:orbit_geometry_plot]

# --8<-- [start:altitude_profile_plot]
# Create altitude vs time plot
fig_altitude = go.Figure()

fig_altitude.add_trace(
    go.Scatter(
        x=times_hours,
        y=altitudes_km,
        mode="lines",
        line=dict(color="blue", width=2),
        name="Altitude",
    )
)

# Reference lines for initial and target altitudes
fig_altitude.add_hline(
    y=400,
    line_dash="dash",
    line_color="gray",
    annotation_text="LEO: 400 km",
    annotation_position="top right",
)

fig_altitude.add_hline(
    y=35786,
    line_dash="dash",
    line_color="gray",
    annotation_text="GEO: 35,786 km",
    annotation_position="top right",
)

# Burn markers
burn1_time_hr = burn1_time_s / 3600.0
burn2_time_hr = burn2_time_s / 3600.0

fig_altitude.add_vline(
    x=burn1_time_hr,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Burn 1: {dv1 / 1e3:.2f} km/s",
    annotation_position="top left",
)

fig_altitude.add_vline(
    x=burn2_time_hr,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Burn 2: {dv2 / 1e3:.2f} km/s",
    annotation_position="top left",
)

fig_altitude.update_layout(
    title="Altitude During Hohmann Transfer",
    xaxis_title="Time (hours)",
    yaxis_title="Altitude (km)",
    showlegend=False,
    height=500,
    margin=dict(l=60, r=40, t=60, b=60),
)
# --8<-- [end:altitude_profile_plot]

# --8<-- [start:velocity_profile_plot]
# Create velocity vs time plot
fig_velocity = go.Figure()

fig_velocity.add_trace(
    go.Scatter(
        x=times_hours,
        y=velocities_km_s,
        mode="lines",
        line=dict(color="blue", width=2),
        name="Velocity",
    )
)

# Reference lines for circular orbit velocities
fig_velocity.add_hline(
    y=v_leo / 1e3,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"LEO: {v_leo / 1e3:.2f} km/s",
    annotation_position="top right",
)

fig_velocity.add_hline(
    y=v_geo / 1e3,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"GEO: {v_geo / 1e3:.2f} km/s",
    annotation_position="bottom right",
)

# Burn markers
fig_velocity.add_vline(
    x=burn1_time_hr,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Burn 1: +{dv1 / 1e3:.2f} km/s",
    annotation_position="top left",
)

fig_velocity.add_vline(
    x=burn2_time_hr,
    line_dash="dot",
    line_color="red",
    annotation_text=f"Burn 2: +{dv2 / 1e3:.2f} km/s",
    annotation_position="top left",
)

fig_velocity.update_layout(
    title="Velocity During Hohmann Transfer",
    xaxis_title="Time (hours)",
    yaxis_title="Velocity (km/s)",
    showlegend=False,
    height=500,
    margin=dict(l=60, r=40, t=60, b=60),
)
# --8<-- [end:velocity_profile_plot]

# Validation
altitude_gain = final_altitude - (r_leo - bh.R_EARTH)
assert altitude_gain > 30000e3, f"Altitude gain too small: {altitude_gain / 1e3:.0f} km"
assert final_koe[1] < 0.01, f"Final orbit not circular enough: e = {final_koe[1]:.6f}"

print("\nExample validated successfully!")
print(f"  Altitude gain: {altitude_gain / 1e3:.0f} km")
print(f"  Final eccentricity: {final_koe[1]:.6f}")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save themed figures
light_path, dark_path = save_themed_html(
    fig_geometry, OUTDIR / f"{SCRIPT_NAME}_geometry"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(
    fig_altitude, OUTDIR / f"{SCRIPT_NAME}_altitude"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(
    fig_velocity, OUTDIR / f"{SCRIPT_NAME}_velocity"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nGEO Hohmann Transfer Example Complete!")
