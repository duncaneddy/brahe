#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Earth-Moon Free-Return Trajectory

This example demonstrates how to:
1. Design a translunar injection (TLI) geometry from a parking orbit
2. Target a circumlunar free-return trajectory by searching on the TLI delta-v
3. Fly the mission with an event-triggered impulsive burn in the Earth-Moon
   barycentric (EMBI) frame with an Earth spherical-harmonic force model
4. Visualize the resulting figure-8 in the Earth-Moon Rotating (EMR) frame

A free-return trajectory swings around the far side of the Moon and comes back
to Earth on its own, using lunar gravity to bend the path home without a
dedicated return burn. That passive safety is why the early Apollo lunar
missions flew it: Apollo 13's abort after its oxygen tank ruptured relied on
exactly this property. Artemis I did not fly a strict free return; Artemis II,
the first crewed Artemis flight, does.
See https://en.wikipedia.org/wiki/Free-return_trajectory for background.
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

# --8<-- [start:geometry]
# Free-return geometry: depart a 400 km (ISS-like) parking orbit from a point
# near the Moon's antipode at the expected arrival time, in the Moon's
# instantaneous orbital plane, with a prograde TLI burn. Rotating the departure
# point ahead of the pure antipode by AIM_OFFSET_DEG sets up a flyby that swings
# around the far side of the Moon and bends the trajectory back onto an
# Earth-return leg. A real mission aims a two-dimensional B-plane target (miss
# distance and approach angle); AIM_OFFSET_DEG is a simplified stand-in for that
# second dimension.
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
TRANSFER_TIME = 3.1 * 86400.0
AIM_OFFSET_DEG = 10.0
DEPART_ALT = 400e3


def _rodrigues(vec, axis, angle):
    """Rotate ``vec`` about unit ``axis`` by ``angle`` (Rodrigues' formula)."""
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(angle))
    )


x_moon = bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, epoch + TRANSFER_TIME)
r_moon, v_moon = x_moon[:3], x_moon[3:]
h_hat = np.cross(r_moon, v_moon)
h_hat /= np.linalg.norm(h_hat)

r0 = bh.R_EARTH + DEPART_ALT
u_antipode = -r_moon / np.linalg.norm(r_moon)  # opposite the arrival point
u_hat = _rodrigues(u_antipode, h_hat, np.radians(AIM_OFFSET_DEG))
t_hat = np.cross(h_hat, u_hat)  # prograde, same sense as the Moon
v_circ = np.sqrt(bh.GM_EARTH / r0)

# Back the parking-orbit state up by a short coast so the TLI burn fires exactly
# at the designed departure point and epoch. The mission is integrated about the
# Earth-Moon barycenter (EMBI frame), so the departure state is translated from
# ECI into EMBI with ``state_eci_to_emb``.
T_PARK = 1800.0  # 30 min of parking-orbit coast before TLI
n_park = np.sqrt(bh.GM_EARTH / r0**3)
start_epoch = epoch - T_PARK
u0 = _rodrigues(u_hat, h_hat, -n_park * T_PARK)
t0 = np.cross(h_hat, u0)
state_park = bh.state_eci_to_emb(start_epoch, np.concatenate([r0 * u0, v_circ * t0]))
# --8<-- [end:geometry]

# --8<-- [start:force_model]
# Integrate about the Earth-Moon barycenter (EMBI): the barycenter has no mass
# of its own, so the central gravity term is zero. Earth carries a 5x5
# spherical-harmonic field as an attributed third body (evaluated at the
# object's Earth-relative position), and the Moon and Sun are point-mass
# perturbers from DE440s. The lunar term is what bends the path home; the Sun is
# a smaller but non-negligible perturbation over the multi-day flight.
force_config = bh.ForceModelConfig.for_body(
    bh.CentralBody.EMB,
    bh.GravityConfiguration.zero(),
    third_body=[
        bh.ThirdBodyConfiguration(
            bh.ThirdBody.EARTH,
            gravity=bh.GravityConfiguration.spherical_harmonic(degree=5, order=5),
        ),
        bh.ThirdBody.MOON,
        bh.ThirdBody.SUN,
    ],
)
force_config.validate()
# --8<-- [end:force_model]


# --8<-- [start:targeting]
# Geodetic altitude above Earth from an EMBI-centered state. The integration
# state is barycentric, so it is translated to ECI before the altitude is
# computed - a plain AltitudeEvent would measure altitude above the barycenter,
# not the Earth. This scalar drives the terminal re-entry event.
def geodetic_altitude(event_epoch, event_state):
    x_eci = bh.state_emb_to_eci(event_epoch, event_state)
    x_ecef = bh.position_eci_to_ecef(event_epoch, x_eci[:3])
    return bh.position_ecef_to_geodetic(x_ecef, bh.AngleFormat.DEGREES)[2]


# Fly a candidate mission: coast the parking orbit up to the design epoch, apply
# the TLI impulsively through a TimeEvent callback, then integrate. The same
# builder is used to score candidates during targeting and to fly the final
# mission, so the trajectory the targeter searches is exactly the one flown.
def fly(dv, duration, terminal=False):
    """Propagate a candidate mission with a ``dv`` TLI at ``epoch``."""

    def tli_callback(event_epoch, event_state):
        # Burn prograde relative to Earth (the parking-orbit velocity), not
        # relative to the EMBI integration frame: translate to ECI, add the
        # delta-v along the Earth-relative velocity, translate back.
        x_eci = bh.state_emb_to_eci(event_epoch, event_state)
        x_eci[3:6] += dv * x_eci[3:6] / np.linalg.norm(x_eci[3:6])
        return (bh.state_eci_to_emb(event_epoch, x_eci), bh.EventAction.CONTINUE)

    prop = bh.NumericalOrbitPropagator(
        start_epoch,
        state_park,
        bh.NumericalPropagationConfig.default(),
        force_config,
        None,
    )
    prop.add_event_detector(bh.TimeEvent(epoch, "TLI").with_callback(tli_callback))
    if terminal:
        prop.add_event_detector(
            bh.ValueEvent(
                "Re-entry interface",
                geodetic_altitude,
                120e3,
                bh.EventDirection.DECREASING,
            ).set_terminal()
        )
    prop.propagate_to(epoch + duration)
    return prop


# The miss distance at the Moon is a V-shaped function of the TLI delta-v: too
# little energy and the transfer apogee never reaches lunar distance, too much
# and the spacecraft races past ahead of the Moon. The free-return branch is the
# ascending side of the V, where the perilune radius grows with delta-v. A
# coarse scan locates that branch; a bisection then refines the delta-v to a
# target perilune. This scalar search stands in for the Lambert solvers and
# differential-correction targeters a real mission uses.
def min_moon_distance(dv):
    """Propagate a candidate TLI and return the closest lunar approach [m]."""
    prop = fly(dv, 6.0 * 86400.0)
    return min(
        np.linalg.norm(
            prop.state_eci(epoch + t)[:3]
            - bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, epoch + t)[:3]
        )
        for t in np.arange(60.0, 6.0 * 86400.0, 600.0)
    )


TARGET_PERILUNE = bh.R_MOON + 2000e3

# Coarse scan over the near-escape delta-v range to reveal the V and locate its
# minimum (the closest reachable approach for this geometry).
dv_grid = np.arange(3.06e3, 3.13e3, 5.0)
perilunes = np.array([min_moon_distance(dv) for dv in dv_grid])
i_min = int(np.argmin(perilunes))

# Bracket the target on the ascending (free-return) branch, then bisect. The
# V-minimum sits below the target; walk up the ascending side until the next
# grid point crosses the target and refine within that interval.
if TARGET_PERILUNE <= perilunes[i_min]:
    raise ValueError("Target perilune is below the closest reachable approach")
i = i_min
while i + 1 < len(dv_grid) and perilunes[i + 1] < TARGET_PERILUNE:
    i += 1
if i + 1 >= len(dv_grid):
    raise ValueError("Target perilune not reached within the scanned delta-v range")
dv_lo, dv_hi = dv_grid[i], dv_grid[i + 1]
for _ in range(40):
    dv_mid = 0.5 * (dv_lo + dv_hi)
    if min_moon_distance(dv_mid) < TARGET_PERILUNE:
        dv_lo = dv_mid
    else:
        dv_hi = dv_mid
dv_tli = 0.5 * (dv_lo + dv_hi)
print(f"TLI delta-v: {dv_tli / 1e3:.4f} km/s")
# --8<-- [end:targeting]

# --8<-- [start:final_run]
# Fly the tuned design to completion: the terminal 120 km altitude event stops
# the propagation at the atmospheric entry interface on the return leg. The
# flight time to that point falls out of the propagation rather than being
# prescribed.
MISSION_TIME = 12.0 * 86400.0
prop = fly(dv_tli, MISSION_TIME, terminal=True)

# The terminal re-entry event ends the flight before MISSION_TIME; measure the
# flown time from TLI and sample only the arc the propagator actually flew.
flight_time = prop.current_epoch() - epoch
print(f"Flight time to re-entry: {flight_time / 86400.0:.2f} days")
# --8<-- [end:final_run]

# --8<-- [start:distance_history]
# Distance from Earth and from the Moon over the whole flight, sampled from the
# trajectory the propagator recorded. A 1 s offset keeps every sample off the
# TLI event epoch, where the state is discontinuous (pre- vs. post-burn); the
# final epoch is appended so the re-entry point itself is captured.
dt = 600.0
sample_times = np.append(np.arange(1.0, flight_time, dt), flight_time)
sample_epochs = [epoch + t for t in sample_times]
times_days = sample_times / 86400.0
earth_dists = np.array([np.linalg.norm(prop.state_eci(e)[:3]) for e in sample_epochs])
moon_dists = np.array(
    [
        np.linalg.norm(
            prop.state_eci(e)[:3] - bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, e)[:3]
        )
        for e in sample_epochs
    ]
)
# --8<-- [end:distance_history]

# --8<-- [start:distance_plot]
fig_distance = go.Figure()

fig_distance.add_trace(
    go.Scatter(
        x=times_days.tolist(),
        y=(earth_dists / 1e3).tolist(),
        mode="lines",
        line=dict(color="steelblue", width=2),
        name="Distance from Earth",
    )
)
fig_distance.add_trace(
    go.Scatter(
        x=times_days.tolist(),
        y=(moon_dists / 1e3).tolist(),
        mode="lines",
        line=dict(color="gray", width=2, dash="dash"),
        name="Distance from Moon",
    )
)

fig_distance.update_layout(
    title="Free-Return Trajectory: Distance from Earth and Moon",
    xaxis_title="Time (days)",
    yaxis_title="Distance (km)",
    height=500,
    margin=dict(l=60, r=40, t=60, b=60),
)
# --8<-- [end:distance_plot]

# --8<-- [start:plot_emr]
# In the Earth-Moon Rotating (EMR) frame the Moon is held fixed on the axis, so
# the free-return path traces the characteristic figure-8 that is invisible in
# an inertial frame. Place the fixed body spheres at the perilune epoch so the
# lunar swing-by aligns with the Moon, and sample the trajectory in the EMR
# frame for both a 3D and a top-down (X-Y) view.
i_perilune = int(np.argmin(moon_dists))
reference_epoch = sample_epochs[i_perilune]

emr_states = np.array(
    [prop.state_in_frame(bh.ReferenceFrame.EMR, e) for e in sample_epochs]
)
emr_xyz_km = emr_states[:, :3] / 1e3
emr_vel = emr_states[:, 3:6]

# Direction-of-travel arrows at a handful of points evenly spaced along the arc.
arrow_idx = np.linspace(len(sample_epochs) * 0.05, len(sample_epochs) * 0.96, 8)
arrow_idx = arrow_idx.astype(int)


def _emr_body_xy_km(naif_id):
    """Body position in the EMR frame at the perilune epoch [km]."""
    return (
        bh.position_frame_to_frame(
            bh.ReferenceFrame.BodyCenteredICRF(naif_id),
            bh.ReferenceFrame.EMR,
            reference_epoch,
            np.zeros(3),
        )
        / 1e3
    )


earth_xy = _emr_body_xy_km(bh.NAIFId.EARTH)
moon_xy = _emr_body_xy_km(bh.NAIFId.MOON)

# 3D view: textured Earth and Moon with the trajectory, plus cone arrows along
# the path. The cones use absolute sizing so they stay a fixed size in the scene.
fig_emr = bh.plot_earth_moon_rotating_3d(
    [{"trajectory": prop.trajectory, "color": "#fc3d21", "label": "Free return"}],
    backend="plotly",
    reference_epoch=reference_epoch,
    view_elevation=50.0,
    view_azimuth=-120.0,
    view_distance=2.4,
)
arrow_dir = emr_vel[arrow_idx] / np.linalg.norm(
    emr_vel[arrow_idx], axis=1, keepdims=True
)
fig_emr.add_trace(
    go.Cone(
        x=emr_xyz_km[arrow_idx, 0],
        y=emr_xyz_km[arrow_idx, 1],
        z=emr_xyz_km[arrow_idx, 2],
        u=arrow_dir[:, 0],
        v=arrow_dir[:, 1],
        w=arrow_dir[:, 2],
        sizemode="absolute",
        sizeref=0.45,
        anchor="tail",
        showscale=False,
        colorscale=[[0, "#fc3d21"], [1, "#fc3d21"]],
        name="Direction of travel",
    )
)

# 2D top-down (X-Y) view: the figure-8 with Earth and Moon drawn to scale and
# arrow markers rotated to the local direction of travel. The self-crossing near
# Earth is the signature of the circumlunar free return. Plotly's arrow marker
# points along +Y at angle 0 and rotates clockwise, so the heading is measured
# clockwise from +Y.
heading_deg = np.degrees(np.arctan2(emr_vel[arrow_idx, 0], emr_vel[arrow_idx, 1]))
fig_emr_2d = go.Figure()
fig_emr_2d.add_trace(
    go.Scatter(
        x=emr_xyz_km[:, 0].tolist(),
        y=emr_xyz_km[:, 1].tolist(),
        mode="lines",
        line=dict(color="#fc3d21", width=2),
        name="Free return",
    )
)
fig_emr_2d.add_trace(
    go.Scatter(
        x=emr_xyz_km[arrow_idx, 0].tolist(),
        y=emr_xyz_km[arrow_idx, 1].tolist(),
        mode="markers",
        marker=dict(
            symbol="arrow", size=13, angle=heading_deg.tolist(), color="#fc3d21"
        ),
        showlegend=False,
    )
)
theta = np.linspace(0.0, 2.0 * np.pi, 120)
for center, radius, color, name in [
    (earth_xy, bh.R_EARTH / 1e3, "#3f7bc2", "Earth"),
    (moon_xy, bh.R_MOON / 1e3, "#9aa0a6", "Moon"),
]:
    fig_emr_2d.add_trace(
        go.Scatter(
            x=(center[0] + radius * np.cos(theta)).tolist(),
            y=(center[1] + radius * np.sin(theta)).tolist(),
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color=color, width=1),
            name=name,
        )
    )
fig_emr_2d.update_layout(
    title="Free Return in the Earth-Moon Rotating Frame (top-down)",
    xaxis_title="X (km)",
    yaxis_title="Y (km)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
)
# --8<-- [end:plot_emr]

# --8<-- [start:validation]
# Confirm a genuine circumlunar free return. The lunar pass clears the surface
# but stays within 20,000 km of the Moon's center, and the return leg descends
# below 1,000 km altitude - the propagation is terminated at the 120 km entry
# interface, so the minimum Earth distance is that entry crossing. The flyby
# must also be a retrograde selenocentric pass: the spacecraft's angular
# momentum about the Moon points opposite the Moon's orbital angular momentum
# about Earth, which is the signature of a far-side circumlunar swing-by (the
# figure-8), as opposed to a near-side pass in front of the Moon.
perilune = moon_dists.min()
return_radius = earth_dists[i_perilune:].min()
return_altitude_km = (return_radius - bh.R_EARTH) / 1e3

x_sc = prop.state_eci(reference_epoch)
x_moon_peri = bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, reference_epoch)
h_selenocentric = np.cross(x_sc[:3] - x_moon_peri[:3], x_sc[3:] - x_moon_peri[3:])
h_moon_orbit = np.cross(x_moon_peri[:3], x_moon_peri[3:])
retrograde_pass = np.dot(h_selenocentric, h_moon_orbit) < 0.0

print(
    f"\nPerilune radius: {perilune / 1e3:.0f} km "
    f"(altitude: {(perilune - bh.R_MOON) / 1e3:.0f} km)"
)
print(f"Return entry-interface altitude: {return_altitude_km:.0f} km")
print(f"Retrograde far-side pass: {retrograde_pass}")

assert bh.R_MOON < perilune < bh.R_MOON + 20000e3
assert return_radius < bh.R_EARTH + 1000e3, (
    f"No free return: return altitude {return_altitude_km:.0f} km"
)
assert retrograde_pass, (
    "Flyby is a near-side (prograde) pass, not a far-side free return"
)

print("\nExample validated successfully!")
# --8<-- [end:validation]
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save themed figures
light_path, dark_path = save_themed_html(
    fig_distance, OUTDIR / f"{SCRIPT_NAME}_distance"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(fig_emr, OUTDIR / f"{SCRIPT_NAME}_emr")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

light_path, dark_path = save_themed_html(fig_emr_2d, OUTDIR / f"{SCRIPT_NAME}_emr_2d")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nEarth-Moon Free-Return Example Complete!")
