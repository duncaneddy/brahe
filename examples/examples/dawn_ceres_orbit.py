#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# FLAGS = ["NETWORK"]
# ///
"""
Dawn at Ceres: a user-defined rotating central body

This example demonstrates how to:
1. Define an entirely user-supplied central body (GM, radius, spin pole,
   prime meridian) for a body brahe has no built-in support for
2. Register a custom body-fixed frame from an IAU-style pole/prime-meridian
   rotation model
3. Propagate the Dawn spacecraft's LAMO (Low Altitude Mapping Orbit) around
   Ceres using that body and frame
4. Report the body-fixed state through the custom frame and visualize the
   trajectory in 3D around a textured Ceres

Unlike Earth, the Moon, and Mars, Ceres has no built-in constants in brahe:
no built-in `CentralBody` variant, no named inertial/fixed frame pair, and no
spin model. Everything about Ceres in this example - its gravitational
parameter, radius, spin pole, prime meridian, and body-fixed frame - is
supplied by the user via `CentralBody.Custom` and `register_custom_frame`.
The same recipe applies to any body brahe doesn't have built-in constants
for: another dwarf planet, an asteroid, or a comet nucleus.

NASA's Dawn spacecraft orbited Ceres from 2015 to 2018, spending much of its
final year in LAMO: a ~375 km, near-circular polar orbit used for its highest
resolution gravity and neutron/gamma-ray mapping. This example is inspired by
that LAMO phase rather than reproducing its exact mission parameters.

The propagator integrates in a Ceres-centered inertial frame whose axes are
ICRF-aligned: its z-axis is the ICRF pole, which sits ~23 deg from Ceres's
spin pole. An inclination passed straight to state_koe_to_eci would therefore
be measured against the wrong pole. Instead, state_koe_to_inertial_for_body
references the elements to Ceres's mean equator at J2000 -- and because Ceres
is a Custom body, it reads the pole from the user-registered body-fixed frame
-- so the 90 deg polar inclination is referenced to Ceres's equator as intended.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import os
import pathlib
import sys

import numpy as np

import brahe as bh

# No `bh.initialize_eop()` or `bh.load_common_spice_kernels()` call is needed
# here: this example never converts to/from Earth-relative frames or queries
# an ephemeris, so it has no dependency on Earth orientation data or SPICE
# kernels. Ceres (NAIF ID 2000001) has no SPK coverage in the DE kernels
# brahe loads, which is also why `state_bci` (not `state_eci`) is used below.
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:body_definition]
# Ceres physical model (IAU WGCCRE 2015 pole/rotation; GM/radius consistent
# with Dawn-era determinations):
CERES_NAIF_ID = 2000001
GM_CERES = 6.26325e10  # m^3/s^2
R_CERES = 469.7e3  # m
CERES_POLE_RA = 291.418  # deg, ICRF right ascension of the spin pole
CERES_POLE_DEC = 66.764  # deg, ICRF declination of the spin pole
CERES_W0 = 170.650  # deg, prime meridian angle at J2000 TT
CERES_W_RATE = 952.1532635  # deg/day
# --8<-- [end:body_definition]

# --8<-- [start:body_fixed_frame]
# A body-fixed frame from the IAU-style pole/prime-meridian model, registered
# as a custom frame: rotation(epoch) -> ICRF-to-body DCM.
_J2000_TT_MJD = 51544.5


def ceres_rotation(epc):
    d = epc.mjd_as_time_system(bh.TimeSystem.TT) - _J2000_TT_MJD
    alpha = np.radians(CERES_POLE_RA)
    delta = np.radians(CERES_POLE_DEC)
    w = np.radians((CERES_W0 + CERES_W_RATE * d) % 360.0)
    return (
        bh.Rz(w, bh.AngleFormat.RADIANS)
        @ bh.Rx(np.pi / 2 - delta, bh.AngleFormat.RADIANS)
        @ bh.Rz(np.pi / 2 + alpha, bh.AngleFormat.RADIANS)
    )


def ceres_omega(epc=None):
    return np.array([0.0, 0.0, np.radians(CERES_W_RATE) / 86400.0])


# The `1` is an arbitrarily chosen registry key for this custom frame; it
# just needs to be unique among registered custom frames.
bh.register_custom_frame(1, ceres_rotation, ceres_omega)
ceres_fixed = bh.ReferenceFrame.BodyFixedCustom(CERES_NAIF_ID, 1)
# --8<-- [end:body_fixed_frame]

# --8<-- [start:force_model]
ceres = bh.CentralBody.Custom(
    "Ceres",
    CERES_NAIF_ID,
    GM_CERES,
    radius=R_CERES,
    omega=ceres_omega(),
    fixed_frame=ceres_fixed,
)

# Gravity-only, point-mass force model: no third bodies or SRP (the loaded
# DE kernels carry no Ceres ephemeris). See the accompanying docs page for
# the Horizons SPK path to enabling perturbations and why ICGEM's Ceres
# product is not a drop-in gravity field.
force_config = bh.ForceModelConfig.for_body(ceres, bh.GravityConfiguration.point_mass())
# --8<-- [end:force_model]

# --8<-- [start:orbit_setup]
# Dawn LAMO (Low Altitude Mapping Orbit): ~375 km circular polar orbit.
epoch = bh.Epoch.from_datetime(2016, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

oe = np.array(
    [R_CERES + 375e3, 0.001, 90.0, 0.0, 0.0, 0.0]
)  # [a [m], e [-], i [deg], RAAN [deg], argp [deg], M [deg]]

state0 = bh.state_koe_to_inertial_for_body(oe, ceres, bh.AngleFormat.DEGREES)

# Ceres spin pole (ICRF) from the IAU pole constants, used below to confirm the
# orbit geometry (declination / right ascension -> unit pole vector). This is
# the same pole the registered frame carries, evaluated directly here.
_pole_ra = np.radians(CERES_POLE_RA)
_pole_dec = np.radians(CERES_POLE_DEC)
ceres_pole = np.array(
    [
        np.cos(_pole_dec) * np.cos(_pole_ra),
        np.cos(_pole_dec) * np.sin(_pole_ra),
        np.sin(_pole_dec),
    ]
)

print(f"LAMO altitude: {(oe[0] - R_CERES) / 1e3:.1f} km, inclination: {oe[2]:.1f} deg")
print(f"Ceres spin pole (ICRF): {np.array2string(ceres_pole, precision=4)}")
# --8<-- [end:orbit_setup]

# --8<-- [start:propagation]
prop = bh.NumericalOrbitPropagator(
    epoch, state0, bh.NumericalPropagationConfig.default(), force_config, None
)
duration = 2 * 86400.0
print(f"\nPropagating {duration / 86400.0:.0f} days...")
prop.propagate_to(epoch + duration)
print("  Complete!")

# state_bci returns the propagator's native state in the central body's
# body-centered inertial frame (here, Ceres-centered). state_eci would
# instead try to re-center the state onto Earth via SPK ephemeris data, which
# raises for Ceres since no SPK kernel covers NAIF ID 2000001.
dt = 120.0
epochs = [epoch + t for t in np.arange(0.0, duration, dt)]
radii_km = np.array([np.linalg.norm(prop.state_bci(epc)[:3]) for epc in epochs]) / 1e3
print(f"\nMin radius: {radii_km.min():.2f} km, max radius: {radii_km.max():.2f} km")
print(f"(Ceres radius: {R_CERES / 1e3:.1f} km)")

# Body-fixed position through the registered frame:
x_fixed = prop.state_in_frame(ceres_fixed, epoch + duration)
print(
    f"\nBody-fixed state at t+{duration / 86400.0:.0f} d: {np.array2string(x_fixed, precision=1)}"
)
# --8<-- [end:propagation]

# --8<-- [start:plot_3d]
fig_3d = bh.plot_trajectory_3d(
    [{"trajectory": prop.trajectory, "color": "cyan", "label": "Dawn (LAMO)"}],
    central_body="ceres",
    backend="plotly",
)
# --8<-- [end:plot_3d]

# Validation
# Inclination relative to the Ceres spin pole confirms the orbit plane was
# built about the Ceres equator, not the ICRF pole. Point-mass gravity has no
# J2, so this inclination is conserved across the propagation.
incs_deg = []
for epc in epochs:
    x = prop.state_bci(epc)
    h = np.cross(x[:3], x[3:])
    h /= np.linalg.norm(h)
    incs_deg.append(np.degrees(np.arccos(np.clip(np.dot(h, ceres_pole), -1.0, 1.0))))
incs_deg = np.array(incs_deg)
print(
    f"\nInclination rel. Ceres pole: {incs_deg.min():.4f} to {incs_deg.max():.4f} deg"
)

assert np.all(np.abs(incs_deg - 90.0) < 0.1), (
    f"Inclination departed from the 90 deg polar design rel. Ceres pole: "
    f"{incs_deg.min():.4f} to {incs_deg.max():.4f} deg"
)
assert radii_km.min() * 1e3 > R_CERES, "Orbit descended below the Ceres surface"
assert radii_km.max() * 1e3 < R_CERES + 1000e3, (
    "Orbit exceeded the expected LAMO altitude bound"
)
assert np.all(np.isfinite(x_fixed)), "Body-fixed state contains non-finite values"

print("\nExample validated successfully!")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save themed figures
light_path, dark_path = save_themed_html(fig_3d, OUTDIR / f"{SCRIPT_NAME}_3d")
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nDawn at Ceres Example Complete!")
