#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Mars Reconnaissance Orbiter (MRO) Science Orbit

This example demonstrates how to:
1. Set up a sun-synchronous, near-polar low Mars orbit (LMO) in a
   Mars-equatorial basis and rotate it into the propagator's frame
2. Propagate it with the full Mars force model (50x50 GMM-2B gravity,
   atmospheric drag, SRP, Sun third body)
3. Track how the orbital elements evolve over a multi-day propagation
4. Visualize the trajectory in 3D around a textured Mars

The Mars Reconnaissance Orbiter has flown a ~255 x 320 km sun-synchronous
orbit since 2006, imaging the surface at consistent local solar time on every
pass. Its 92.6 degree inclination is chosen so that the nodal precession
driven by Mars's oblateness (J2) matches the planet's mean motion around the
Sun, keeping the orbit plane's orientation relative to the Sun fixed.

The propagator integrates in MCI, whose axes are ICRF-aligned: the MCI z-axis
is the ICRF pole, which sits ~37 deg from Mars's spin pole. An inclination
passed straight to state_koe_to_eci would therefore be measured against the
wrong pole. Instead, state_koe_to_inertial_for_body references the elements to
Mars's mean equator at J2000 (the plane normal to the Mars IAU pole) and
returns the state directly in MCI, so the 92.6 deg inclination is referenced to
Mars's equator as intended. state_inertial_to_koe_for_body inverts it, so the
osculating inclination it reports is already measured against the Mars equator.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import os
import pathlib
import sys

import numpy as np

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
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
params = np.array([2180.0, 20.0, 2.2, 20.0, 1.3])  # mass, drag_area, Cd, srp_area, Cr

# Mars mean pole (ICRF) at J2000: the reference plane state_koe_to_inertial_for_body
# uses is the Mars equator at J2000, whose pole is the third row of the ICRF ->
# Mars body-fixed (IAU) rotation. x_eq is the ascending node of that equator on
# the ICRF equator (ICRF pole x spin pole) and y_eq completes the triad; they
# span the equatorial plane the RAAN below is measured in.
j2000 = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.TDB)
mars_pole = np.asarray(bh.rotation_icrf_to_body_fixed_iau(499, j2000))[2, :]
x_eq = np.cross([0.0, 0.0, 1.0], mars_pole)
x_eq /= np.linalg.norm(x_eq)
y_eq = np.cross(mars_pole, x_eq)

# Sun-synchronous node placement. At the ascending node the local solar time
# is 12:00 + (RAAN - sun_ra) / (15 deg/hr), where sun_ra is the Sun's right
# ascension in the Mars-equatorial basis, so a 15:00 (mid-afternoon) node
# needs RAAN = sun_ra + 45 deg. The Sun direction is taken from the Mars
# barycenter (NAIF 4); the barycenter-to-center offset is negligible here.
sun_mci = bh.spk_state(10, 4, epoch)[:3]
sun_ra = np.degrees(np.arctan2(sun_mci @ y_eq, sun_mci @ x_eq)) % 360.0
raan = (sun_ra + 45.0) % 360.0

r_p = bh.R_MARS + 255e3
r_a = bh.R_MARS + 320e3
a = (r_p + r_a) / 2
e = (r_a - r_p) / (r_a + r_p)
oe = np.array([a, e, 92.6, raan, 270.0, 0.0])  # [m, -, deg, deg, deg, deg]

# state_koe_to_inertial_for_body references the elements to Mars's mean equator
# at J2000 and returns the state directly in the MCI frame the propagator
# integrates in, so the 92.6 deg inclination is measured against Mars's equator.
state0 = bh.state_koe_to_inertial_for_body(
    oe, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
)

print(
    f"Periapsis radius: {r_p / 1e3:.1f} km (altitude: {(r_p - bh.R_MARS) / 1e3:.0f} km)"
)
print(
    f"Apoapsis radius:  {r_a / 1e3:.1f} km (altitude: {(r_a - bh.R_MARS) / 1e3:.0f} km)"
)
print(f"Semi-major axis: {a / 1e3:.1f} km, eccentricity: {e:.4f}")
print(f"Mars spin pole (ICRF): {np.array2string(mars_pole, precision=4)}")
print(f"RAAN for 15:00 LTAN: {raan:.2f} deg (Sun RA {sun_ra:.2f} deg)")
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
sma_km, ecc, inc_deg, raan_deg, argp_deg, anom_deg = [], [], [], [], [], []
for epc in epochs:
    x = prop.state_bci(epc)
    # state_inertial_to_koe_for_body references the elements to Mars's mean
    # equator at J2000, so koe[2] is already the Mars-pole-relative inclination
    # (no manual re-measurement against the ICRF pole needed).
    koe = bh.state_inertial_to_koe_for_body(
        x, bh.CentralBody.Mars, bh.AngleFormat.DEGREES
    )
    sma_km.append(koe[0] / 1e3)
    ecc.append(koe[1])
    inc_deg.append(koe[2])
    raan_deg.append(koe[3])
    argp_deg.append(koe[4])
    anom_deg.append(koe[5])
sma_km = np.array(sma_km)
ecc = np.array(ecc)
inc_deg = np.array(inc_deg)
raan_deg = np.array(raan_deg)
argp_deg = np.array(argp_deg)
anom_deg = np.array(anom_deg)
times_sec = np.arange(0.0, duration, dt)
alt_p_km = sma_km * (1 - ecc) - bh.R_MARS / 1e3
# --8<-- [end:element_history]

# --8<-- [start:element_plot]
# plot_keplerian_trajectory's raw-array input assumes SI/radians (sma in
# meters, angles in radians) regardless of the display units requested via
# sma_units/angle_units. The time column must be elapsed seconds so the
# plotter's auto hours/minutes axis labeling applies correctly. The
# inclination here is already Mars-pole-relative, since
# state_inertial_to_koe_for_body references elements to the Mars equator.
koe_history = np.column_stack(
    (
        times_sec,
        sma_km * 1e3,
        ecc,
        np.radians(inc_deg),
        np.radians(raan_deg),
        np.radians(argp_deg),
        np.radians(anom_deg),
    )
)
fig_elements = bh.plot_keplerian_trajectory(
    [{"trajectory": koe_history, "label": "MRO"}],
    angle_units="deg",
    sma_units="km",
    backend="plotly",
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
print(f"Mean inclination (rel. Mars pole): {inc_deg.mean():.4f} deg (target: 92.6 deg)")

assert alt_p_km.min() > 0, "Orbit descended below the Mars surface"
assert abs(inc_deg[0] - 92.6) < 0.01, (
    f"Initial inclination not at design value rel. Mars pole: {inc_deg[0]:.4f} deg"
)
# Measured against Mars's spin pole, the osculating inclination stays within
# ~0.1 deg of the 92.6 deg design value: J2 drives the nodal precession that
# makes the orbit sun-synchronous but no secular change in inclination, so
# the remaining variation is a bounded short-period oscillation.
assert abs(inc_deg.mean() - 92.6) < 0.1, (
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
