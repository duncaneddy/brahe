# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["IGNORE"]
# ///
"""
Propagating a spacecraft orbit about the Moon with ForceModelConfig.lunar_default().

lunar_default() configures a CentralBody.Moon force model: 50x50 GRGM660PRIM
lunar gravity, solar radiation pressure (occulted by the Moon and Earth), and
Earth/Sun third-body perturbations from the DE440s ephemeris. The propagator
integrates in the Moon-Centered Inertial (LCI) frame; state_in_frame converts
the result into the Moon-fixed LFPA frame for reporting a body-fixed ground
track position.

First run downloads the GRGM660PRIM gravity model and the moon_pa_de440
binary PCK, caching them under $BRAHE_CACHE (~/.cache/brahe by default).
"""

import numpy as np
import brahe as bh

# Initialize EOP data and the DE440s planetary ephemeris used for third-body
# perturbations and LCI <-> LFPA frame conversions.
bh.initialize_eop()
bh.initialize_ephemeris()

# Initial epoch
epoch = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Circular low lunar orbit (LLO) at 100 km altitude, expressed directly in
# the Moon-Centered Inertial (LCI) frame the propagator integrates in.
a = bh.R_MOON + 100e3
v = bh.periapsis_velocity(a, 0.0, gm=bh.GM_MOON)
state = np.array([a, 0.0, 0.0, 0.0, v, 0.0])

# Spacecraft parameters, indexed per lunar_default()'s ParameterSource
# assignments: [mass, _, _, srp_area, Cr]. lunar_default() has no drag
# model, so indices 1 and 2 (drag area, Cd) are unused placeholders.
params = np.array([500.0, 0.0, 0.0, 2.0, 1.3])

force_config = bh.ForceModelConfig.lunar_default()
force_config.validate()

prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    force_config,
    params,
)

# Propagate for 6 hours
final_epoch = epoch + 6.0 * 3600.0
prop.propagate_to(final_epoch)

# state_in_frame routes the propagator's native LCI state through the
# reference frame router into any other supported frame. LFPA is the Moon-
# fixed, DE440 principal-axis frame.
x0_lfpa = prop.state_in_frame(epoch, bh.ReferenceFrame.LFPA)
xf_lfpa = prop.state_in_frame(final_epoch, bh.ReferenceFrame.LFPA)

print(f"Initial epoch: {epoch}")
print(f"Final epoch:   {final_epoch}")
print("\nInitial state (LFPA, Moon-fixed):")
print(
    f"  Position (km): [{x0_lfpa[0] / 1e3:.3f}, {x0_lfpa[1] / 1e3:.3f}, {x0_lfpa[2] / 1e3:.3f}]"
)
print(f"  Velocity (m/s): [{x0_lfpa[3]:.3f}, {x0_lfpa[4]:.3f}, {x0_lfpa[5]:.3f}]")
print("\nFinal state (LFPA, Moon-fixed):")
print(
    f"  Position (km): [{xf_lfpa[0] / 1e3:.3f}, {xf_lfpa[1] / 1e3:.3f}, {xf_lfpa[2] / 1e3:.3f}]"
)
print(f"  Velocity (m/s): [{xf_lfpa[3]:.3f}, {xf_lfpa[4]:.3f}, {xf_lfpa[5]:.3f}]")

# Validate propagation completed and the orbit remains bound to the Moon
assert prop.current_epoch() == final_epoch
r_final = np.linalg.norm(xf_lfpa[:3])
assert bh.R_MOON < r_final < bh.R_MOON + 500e3
print("\nExample validated successfully!")
