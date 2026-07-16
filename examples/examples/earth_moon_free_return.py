#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Earth-Moon Free-Return Trajectory

This example demonstrates how to:
1. Design a translunar injection (TLI) geometry from a parking orbit
2. Target a free-return trajectory by searching on the TLI delta-v
3. Fly the mission as designed with an event-triggered impulsive burn
4. Visualize the resulting figure-8 in the Earth-Moon Rotating (EMR) frame

A free-return trajectory swings around the Moon and comes back to Earth on
its own, using lunar gravity to bend the path home without a dedicated
return burn - the safety margin that let Apollo 8, 10, 11, and 13 abort to
a survivable re-entry if their service propulsion system had failed. Apollo
13 relied on exactly this property after its oxygen tank ruptured. Artemis I
flew the same class of trajectory (an extended "hybrid" free return)
uncrewed in 2022.
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
# Free-return geometry: depart a 185 km parking orbit from a point near the
# Moon's antipode at the expected arrival time, in the Moon's instantaneous
# orbital plane, with a prograde TLI burn. A real mission aims a
# two-dimensional B-plane target (miss distance and approach angle);
# AIM_OFFSET_DEG is a simplified stand-in for that second dimension. Rotating
# the departure point ahead of the pure antipode sets up a lunar flyby that
# bends the trajectory back onto an Earth-return leg instead of flinging it
# onto an escape or a distant loop.
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
TRANSFER_TIME = 3.1 * 86400.0
AIM_OFFSET_DEG = 16.0


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

r0 = bh.R_EARTH + 185e3
u_antipode = -r_moon / np.linalg.norm(r_moon)  # opposite the arrival point
u_hat = _rodrigues(u_antipode, h_hat, np.radians(AIM_OFFSET_DEG))
t_hat = np.cross(h_hat, u_hat)  # prograde, same sense as the Moon
v_circ = np.sqrt(bh.GM_EARTH / r0)
# --8<-- [end:geometry]

# --8<-- [start:force_model]
# Point-mass Earth with Moon and Sun third bodies from DE440s: the dynamics
# that shape a free-return trajectory, nothing more. The lunar third-body
# term is what bends the path home; the Sun is a smaller but non-negligible
# perturbation over an eight-day flight.
force_config = bh.ForceModelConfig.for_body(
    bh.CentralBody.Earth,
    bh.GravityConfiguration.point_mass(),
    third_body=[
        bh.ThirdBodyConfiguration(bh.ThirdBody.MOON, bh.EphemerisSource.DE440s),
        bh.ThirdBodyConfiguration(bh.ThirdBody.SUN, bh.EphemerisSource.DE440s),
    ],
)
# --8<-- [end:force_model]


# --8<-- [start:targeting]
# The miss distance at the Moon is a V-shaped function of the TLI delta-v:
# too little energy and the transfer apogee never reaches lunar distance,
# too much and the spacecraft races past ahead of the Moon. The free-return
# branch is the ascending side of the V, where the perilune radius grows
# with delta-v. A coarse scan locates that branch; a bisection then refines
# the delta-v to a target perilune. This scalar search stands in for the
# Lambert solvers and differential-correction targeters a real mission uses;
# the geometry above already fixes everything but the transfer's energy.
def min_moon_distance(dv):
    """Propagate a candidate TLI and return the closest lunar approach [m]."""
    state = np.concatenate([r0 * u_hat, (v_circ + dv) * t_hat])
    prop = bh.NumericalOrbitPropagator(
        epoch, state, bh.NumericalPropagationConfig.default(), force_config, None
    )
    prop.propagate_to(epoch + 6.0 * 86400.0)
    dists = [
        np.linalg.norm(
            prop.state_eci(epoch + t)[:3]
            - bh.spk_state(bh.NAIFId.MOON, bh.NAIFId.EARTH, epoch + t)[:3]
        )
        for t in np.arange(0.0, 6.0 * 86400.0, 600.0)
    ]
    return min(dists)


TARGET_PERILUNE = bh.R_MOON + 12000e3

# Coarse scan over the near-escape delta-v range to reveal the V and locate
# its minimum (the closest reachable approach for this geometry).
dv_grid = np.arange(3.10e3, 3.19e3, 5.0)
perilunes = np.array([min_moon_distance(dv) for dv in dv_grid])
i_min = int(np.argmin(perilunes))

# Bracket the target on the ascending (free-return) branch, then bisect.
# The V-minimum sits below the target; walk up the ascending side until the
# next grid point crosses the target and refine within that interval.
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
# Final run told as flown: start one short parking-orbit arc before the design
# epoch, coast up to it, apply the tuned TLI impulsively through an event
# callback, then let lunar gravity carry the spacecraft around the Moon and
# back. The burn fires at ``epoch`` - the same instant the targeter injected -
# so the flown trajectory matches the design. A terminal altitude event stops
# the propagation at the 120 km atmospheric entry interface on the return leg.
T_PARK = 1800.0  # 30 min of parking-orbit coast before TLI
n_park = np.sqrt(bh.GM_EARTH / r0**3)

# Back the parking-orbit state up by the coast arc so the burn happens exactly
# at the designed departure point and epoch.
start_epoch = epoch - T_PARK
u0 = _rodrigues(u_hat, h_hat, -n_park * T_PARK)
t0 = np.cross(h_hat, u0)
state_park = np.concatenate([r0 * u0, v_circ * t0])


def tli_callback(event_epoch, event_state):
    new_state = event_state.copy()
    v_dir = event_state[3:6] / np.linalg.norm(event_state[3:6])
    new_state[3:6] += dv_tli * v_dir
    print(f"TLI applied: {dv_tli / 1e3:.4f} km/s prograde")
    return (new_state, bh.EventAction.CONTINUE)


prop = bh.NumericalOrbitPropagator(
    start_epoch, state_park, bh.NumericalPropagationConfig.default(), force_config, None
)
prop.add_event_detector(bh.TimeEvent(epoch, "TLI").with_callback(tli_callback))
prop.add_event_detector(
    bh.AltitudeEvent(
        120e3, "Re-entry interface", bh.EventDirection.DECREASING
    ).set_terminal()
)
MISSION_TIME = 10.0 * 86400.0
prop.propagate_to(epoch + MISSION_TIME)

# The terminal re-entry event ends the flight before MISSION_TIME; measure the
# flown time from TLI and sample only the arc the propagator actually flew.
flight_time = prop.current_epoch() - epoch
print(f"Flight time to re-entry: {flight_time / 86400.0:.2f} days")
# --8<-- [end:final_run]

# --8<-- [start:distance_history]
# Distance from Earth and from the Moon over the whole flight, sampled from
# the trajectory the propagator recorded. A 1 s offset keeps every sample off
# the TLI event epoch, where the state is discontinuous (pre- vs. post-burn);
# the final epoch is appended so the re-entry point itself is captured.
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
# In the Earth-Moon Rotating frame the Moon is held fixed on the axis, so the
# free-return path traces the characteristic figure-8 that is invisible in an
# inertial frame.
fig_emr = bh.plot_earth_moon_rotating_3d(
    [{"trajectory": prop.trajectory, "color": "red", "label": "Free return"}],
    backend="plotly",
)
# --8<-- [end:plot_emr]

# --8<-- [start:validation]
# The lunar pass clears the surface but stays within 20,000 km of the Moon's
# center, and the return leg descends below 1,000 km altitude - the propagation
# is terminated at the 120 km entry interface, so the minimum Earth distance is
# that entry crossing (the true perigee lies below it, inside the atmosphere).
perilune = moon_dists.min()
i_perilune = int(np.argmin(moon_dists))
return_radius = earth_dists[i_perilune:].min()
return_altitude_km = (return_radius - bh.R_EARTH) / 1e3

print(
    f"\nPerilune radius: {perilune / 1e3:.0f} km "
    f"(altitude: {(perilune - bh.R_MOON) / 1e3:.0f} km)"
)
print(f"Return entry-interface altitude: {return_altitude_km:.0f} km")

assert bh.R_MOON < perilune < bh.R_MOON + 20000e3
assert return_radius < bh.R_EARTH + 1000e3, (
    f"No free return: return altitude {return_altitude_km:.0f} km"
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

print("\nEarth-Moon Free-Return Example Complete!")
